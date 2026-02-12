import pandas as pd
import app.ai_wrapper as ai # Assuming ai_wrapper.py contains the necessary AI model interaction functions
from typing import Dict, Any, Optional, Union, List
import json
from app.simple_logger import get_logger 

logger = get_logger(__name__)
# Helper function to assess how deep the categories dictionary is
def dict_depth(d: dict) -> int:
    if not isinstance(d, dict) or not d:  
        return 0
    return 1 + max((dict_depth(v) for v in d.values()), default=0)


def add_uncategorised_to_dict(
    d: Dict[str, Any],
    levels_in_hierarchy: int = 2
) -> Dict[str, Any]:
    """
    Adds default "Uncategorised" options to a category dictionary.

    Args:
        d (Dict[str, Any]): The category dictionary to modify.
        levels_in_hierarchy (int): Depth of the hierarchy (1 or 2).

    Returns:
        Dict[str, Any]: The modified dictionary.
    """
    if levels_in_hierarchy == 1:
        if 'Uncategorised: Not enough information' not in d:
            d['Uncategorised: Not enough information'] = \
                "There isn't enough information to categorise this content"
        if 'Uncategorised: No appropriate category' not in d:
            d['Uncategorised: No appropriate category'] = \
                "There isn't an appropriate category for this content"
    elif levels_in_hierarchy == 2:
        if 'Uncategorised' not in d:
            d['Uncategorised'] = {
                "Not enough information": "There isn't enough information to categorise this content",
                "No category": "There isn't an appropriate category for this content",
            }
    else:
        raise ValueError(f"levels_in_hierarchy must be 1 or 2, got {levels_in_hierarchy}")
    return d


class Categorise:
    def __init__(self,
                 default_gen_model: str = "openai/o3-mini-high",
                 default_run_model: str = "openai/gpt-4o-mini",
                 default_levels_in_hierarchy: int = 2,
                 verbose_mode: bool = False):
        self.default_gen_model = default_gen_model
        self.default_run_model = default_run_model
        self.default_levels_in_hierarchy = default_levels_in_hierarchy
        self.verbose_mode = verbose_mode
        self.categories_dict = None
        
    def generate_categories(self,
                            df: pd.DataFrame,
                            column_to_categorise: str,
                            context_prompt: str,
                            sample_size: int = 1000,
                            max_categories: int = 20,
                            model: Optional[str] = None,
                            levels_in_hierarchy: Optional[int] = None) -> Dict[str, Any]:
        model = model or self.default_gen_model
        current_levels = levels_in_hierarchy or self.default_levels_in_hierarchy

        df_sample = df.sample(n=min(sample_size, len(df)))

        if current_levels == 1:
            json_format_instruction = """
Output a JSON object with a maximum of {max_categories_val} categories with the following format:
{{
  "category name 1": "description for category 1",
  "category name 2": "description for category 2"
}}
"""
            hierarchy_instruction = "one level of categories"
            max_cat_type_for_prompt = "categories"
        else:
            json_format_instruction = """
Output a JSON object with a maximum of {max_categories_val} subcategories in total with the following format:
{{
  "main category 1": {{
    "subcategory 1.1": "description for subcategory 1.1",
    "subcategory 1.2": "description for subcategory 1.2"
  }}
}}
"""
            hierarchy_instruction = "two levels: category and subcategory"
            max_cat_type_for_prompt = "subcategories"

        prompt = f"""
You are a qualitative analyst trained to deal calmly with any topic.
From the texts below, generate categories. The context within which the texts are to be categorised is:
<context>{context_prompt}</context> 

{json_format_instruction.format(max_categories_val=max_categories)}

texts={json.dumps(list(df_sample[column_to_categorise].astype(str).values))}
"""
        num_tokens_for_response = ai.get_context_window(model) - int(ai.get_num_tokens(prompt) * 1.5) - 1000
        logger.info("Calculated tokens for generation call", function="generate_categories", tokens=num_tokens_for_response)

        if self.verbose_mode:
            logger.info("Prompt for generation", function="generate_categories", prompt=prompt)

        result_str = ai.get_llm_text_response(prompt, model=model, max_tokens=num_tokens_for_response)

        if self.verbose_mode:
            logger.info("Initial generation result", function="generate_categories", result=result_str)

        prompt2 = f"""
Please refine your previous output, regenerating a new set of categories within the 
context of <context>{context_prompt}</context>. Try to make sure that the categories are well-defined 
so as to avoid categories covering the same or similar concepts. The titles of the categories should be 
concise and descriptive.
Ensure max {max_categories} {max_cat_type_for_prompt}, keeping format {hierarchy_instruction}.
"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": result_str},
            {"role": "user", "content": prompt2}
        ]

        num_tokens_for_refinement = num_tokens_for_response - 1000
        logger.info("Calculated tokens for refinement call", function="generate_categories", tokens=num_tokens_for_refinement)

        result2_str = ai.get_llm_text_response(messages, model=model, max_tokens=num_tokens_for_refinement)

        if self.verbose_mode:
            logger.info("Refined generation result", function="generate_categories", result=result2_str)

        try:
            extracted_json_list = ai.extract_json_objects(result2_str)
            if not extracted_json_list:
                logger.warning("No JSON extracted from refinement", function="generate_categories", response=result2_str)
                return {}
            result_dict = extracted_json_list[0]
            if not isinstance(result_dict, dict):
                logger.error("Extracted JSON not a dictionary", function="generate_categories", type=str(type(result_dict)), response=result2_str)
                return {}
        except Exception as e:
            logger.error("Exception during JSON extraction", function="generate_categories", error=str(e), response=result2_str)
            result_dict = {}

        return result_dict

    def run_categorisation(self,
                        df: pd.DataFrame,
                        categories_dict: Dict[str, Any],
                        column_to_categorise: str,
                        context_prompt: str,
                        add_uncategorised_opt: bool = True,
                        response_column: str = 'category',
                        model: Optional[str] = None,
                        levels_in_hierarchy: Optional[int] = None) -> pd.DataFrame:
        model_to_use = model or self.default_run_model

        if not categories_dict:
            logger.warning("Empty categories_dict provided", function="run_categorisation")
            df[response_column] = "Error: No categories provided"
            return df

        levels = levels_in_hierarchy or dict_depth(categories_dict)
        if levels < 1 or levels > 2:
            raise ValueError(f"categories_dict depth must be 1 or 2; got {levels}")

        is_one_level = (levels == 1)

        local_categories_dict = categories_dict.copy()
        if add_uncategorised_opt:
            local_categories_dict = add_uncategorised_to_dict(local_categories_dict, levels_in_hierarchy=levels)

        try:
            categories_list = ai.get_categories_as_list(local_categories_dict, include_descriptions=False, one_level_hierarchy=is_one_level)
        except Exception as e:
            logger.error("Error processing categories_dict", function="run_categorisation", error=str(e), categories=str(local_categories_dict), levels=levels)
            df[response_column] = "Error: Could not process category list"
            return df

        if not categories_list:
            logger.warning("Generated category list is empty", function="run_categorisation")
            df[response_column] = "Uncategorised (empty list)"
            return df

        def cat_prompt_func(row_series: pd.Series) -> str:
            item_text = row_series[column_to_categorise] if not pd.isna(row_series[column_to_categorise]) else ""
            base_prompt = f"""
You are a trained qualitative analyst.
Considering this item: <item>{str(item_text)}</item>
From the following list of categories, please choose the most appropriate one according to the context of {context_prompt}.

categories= {categories_list}
"""
            if is_one_level:
                base_prompt += "Only respond with the chosen category name!"
            else:
                base_prompt += "Only respond with 'category.subcategory' format!"
            return base_prompt

        def is_valid_func(row_series: pd.Series) -> bool:
            if response_column not in row_series.index:
                logger.warning("Missing response column", function="run_categorisation", row=row_series.name)
                return False
            if pd.isna(row_series[response_column]):
                logger.warning("NaN in response column", function="run_categorisation", row=row_series.name)
                return False
            response_val_str = str(row_series[response_column])
            if response_val_str not in categories_list:
                logger.warning("Invalid category assignment", function="run_categorisation", row=row_series.name, value=response_val_str)
                return False
            return True

        results_df = ai.iterate_df_rows(df, get_prompt=cat_prompt_func, is_valid=is_valid_func, response_column=response_column, model=model_to_use)
        return results_df

    def catify(self,
                   df: pd.DataFrame,
                   column_to_categorise: str,
                   gen_context: str,
                   run_context: str,
                   sample_size: int = 100,
                   max_categories: int = 20,
                   gen_model: Optional[str] = None,
                   run_model: Optional[str] = None,
                   levels_in_hierarchy: Optional[int] = None,
                   response_column: str = 'category') -> pd.DataFrame:
        current_levels = levels_in_hierarchy or self.default_levels_in_hierarchy

        logger.info("Starting category generation", function="catify", levels=current_levels)
        self.categories_dict = self.generate_categories(df=df, column_to_categorise=column_to_categorise, context_prompt=gen_context, sample_size=sample_size, max_categories=max_categories, model=gen_model, levels_in_hierarchy=current_levels)

        if not self.categories_dict:
            logger.warning("Category generation failed", function="catify")
            df["category_status"] = "Generation Failed"
            return df

        logger.info("Category generation complete", function="catify", categories=json.dumps(self.categories_dict))

        categorised_df = self.run_categorisation(
            df=df,
            categories_dict=self.categories_dict,
            column_to_categorise=column_to_categorise,
            context_prompt=run_context,
            model=run_model,
            levels_in_hierarchy=current_levels,
            response_column=response_column)
        return categorised_df

    def estimate_cost(self,
                      df: pd.DataFrame,
                      gen_model: Optional[str] = None,
                      run_model: Optional[str] = None,
                      sample_size: int = 100,
                      max_tokens_output_per_item: int = 50,
                      levels_in_hierarchy: Optional[int] = None) -> float:
        gen_model_to_use = gen_model or self.default_gen_model
        run_model_to_use = run_model or self.default_run_model
        current_levels = levels_in_hierarchy or self.default_levels_in_hierarchy

        est_gen_prompt_base_len = 500
        sample_text_for_est = " ".join(df[df.columns[0]].dropna().sample(min(sample_size, len(df))).astype(str).tolist())
        input_tokens_gen = len(sample_text_for_est.split()) + est_gen_prompt_base_len
        output_tokens_gen = sample_size * max_tokens_output_per_item * 2
        cost_gen = (input_tokens_gen + output_tokens_gen) * 0.000002

        avg_row_text_len = int(df[df.columns[0]].astype(str).apply(lambda x: len(x.split())).mean())
        est_cat_list_len = sample_size * 10
        est_run_prompt_base_len = 100
        cost_run = len(df) * (avg_row_text_len + est_cat_list_len + est_run_prompt_base_len + max_tokens_output_per_item) * 0.000002

        total_cost = cost_gen + cost_run
        logger.info("Estimated costs", function="estimate_cost", cost_gen=cost_gen, cost_run=cost_run, total_cost=total_cost)
        return total_cost