from typing import Dict, Any, Optional, Union, List
import concurrent.futures
import json
import os
import random
import time
import traceback
from functools import partial
from importlib import reload
from datetime import datetime

import openai
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from app.settings import settings
from app.simple_logger import get_logger

logger = get_logger(__name__)

reload(openai)

# Initialise the clients with None
openrouter_ai_client = None

# connect to openai API if api key in env
if settings.open_router_key:
    openrouter_api_key = settings.open_router_key

    openrouter_ai_client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=openrouter_api_key, 
        base_url="https://openrouter.ai/api/v1"
    )


def get_openrouter_api_key():
    try:
        return openrouter_api_key[-4:]
    except:
        "No open ai key"

def is_valid_placeholder(row):
    """ Placeholder function for checking if the respond from the AI is valid"""
    return True


def get_prompt_example(row):
    json_example = """
{  
    field 1 : value 1,
    field 2 : value 2
}
"""
    prompt = f"""
Summarise the following 

row={row}

into a json object, example being

example={json_example}
"""
    # Obviously don't keep this line!
    raise Exception("Using prompt example")
    return prompt
    
# Count the number of tokens in a string
def get_num_tokens (text: str | list | pd.Series,
                    model: str = 'gpt-4'
                   ) -> int:
    '''
    returns the number of tokens used 
    for a given string, list or 
    pandas series (i.e. column in a df with IO to GPT.)

    if list or pd, the elems have to be str. 

    args:
        :input: str | list | pd.Series - the input to count tokens for
        :model: str - the model to use for the count
    '''
    if isinstance(text, list):
        input = ' '.join(text)
    elif isinstance(text, pd.Series):
        input = ' '.join(text.tolist())
    elif isinstance(text, str):
        input = text
    else:
        raise TypeError('input must be str, list or pd.Series')
    
    # run the tiktoken tokeniser based on model type
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(input))

    return num_tokens


def get_context_window(model_name, client=openrouter_ai_client, default=8192):
    try:
        models = client.models.list()
        for model in models.data:
            if model.id == model_name:
                return model.context_length
    except Exception as e:
        logger.error("Error fetching model info: %s", e)
    return default

def get_open_ai_response(prompt, model="openai/gpt-4o-mini", client=openrouter_ai_client, temperature=0.0, max_tokens=1000):
    if client is None:
        raise Exception("No client")
    
    # Check if prompt is already in messages format
    if isinstance(prompt, list):
        messages = prompt
    else:
        # Convert string to default message format
        messages = [{"role": "user", "content": prompt}]
    
    # Prepare base kwargs
    kwargs = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    response = client.chat.completions.create(**kwargs)
    return response


def get_gpt_text_from_response(res):
    """ Simply returns the response text from gpt"""
    return res.choices[0].message.content.strip()

def get_model_list ():
    #return open_ai_client.models.list()
    return [m.id for m in openrouter_ai_client.models.list().data]

def extract_json_objects(input):
    """
    Function to extract json objects from string.
    It uses stack to handle the nested braces.
    """
    objects = []
    stack = []
    start_pos = 0
    end_pos = 0

    for i, char in enumerate(input):
        if char == '{':
            if not stack:
                start_pos = i
            stack.append(i)
        elif char == '}' and stack:
            stack.pop()
            if not stack:
                end_pos = i + 1 # +1 to include the '}' in the substring
                try:
                    objects.append(json.loads(input[start_pos:end_pos]))
                except json.JSONDecodeError:
                    pass

    return objects



def get_llm_text_response(prompt, model="openai/gpt-4o-mini", max_tokens=1000, max_retries=5,verbose=False, response='str'):
    """
    Get a response using prompt from the specified model. Returns the text from the response.
    """
    retry = 0
    while retry < max_retries:
        try:
            # Prepare base kwargs
            kwargs = {"max_tokens": max_tokens}


            res = get_open_ai_response(prompt, model, **kwargs)
            if verbose:
                logger.debug("Response: %s", res)

            response_text = get_gpt_text_from_response(res)

            if response == 'str':
                return response_text
            elif response == 'dict':
                try:
                    return extract_json_objects(response_text)[0]
                except Exception as e:
                    logger.error("Error parsing a JSON object from this response: \n %s\nError: %s", response_text, e)
                    return None
            elif response == 'list':
                try:
                    parsed = json.loads(response_text)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass

                objects = extract_json_objects(response_text)
                if objects:
                    return objects

                logger.error("Error parsing a JSON list from this response: \n %s", response_text)
                return None
            else:
                raise Exception(f"Invalid response type: {response}")

        except Exception as e:
            retry += 1
            wait_time = 2 ** retry + random.uniform(0, 1)
            if verbose:
                logger.debug(traceback.format_exc())
            logger.error("Rate limit hit or other error: %s. Retrying in %.2f seconds...", e, wait_time)
            time.sleep(wait_time)

    logger.error("Failed to get response after %d retries.", max_retries)
    return None


def get_cost(input_tokens, output_tokens, cached_input_tokens=0, model="openai/gpt-4o-mini"):
    """
    Calculate the cost of a call based on input, output, and cached input tokens,
    using model-specific pricing rates (in USD per 1,000 tokens).

    Parameters:
        input_tokens (int): Total number of input tokens.
        output_tokens (int): Total number of output tokens.
        cached_input_tokens (int): Number of input tokens provided via cache (default: 0).
        model (str): The model name (default: "gpt-4o-mini").
        
    Returns:
        float: The calculated total cost.
        
    Pricing rates (USD per 1K tokens) based on your table:
      - gpt-4.5-preview:        input = $75.00,  cached input = $37.50, output = $150.00
      - gpt-4o:                 input = $2.50,   cached input = $1.25,  output = $10.00
      - gpt-4o-audio-preview:   input = $2.50,   cached input = N/A,    output = $10.00
      - gpt-4o-realtime-preview:input = $5.00,   cached input = $2.50,  output = $20.00
      - gpt-4o-mini:            input = $0.15,   cached input = $0.075, output = $0.60
      - gpt-4o-mini-audio-preview:
                                input = $0.15,   cached input = N/A,    output = $0.60
      - gpt-4o-mini-realtime-preview:
                                input = $0.60,   cached input = $0.30,  output = $2.40
      - o1:                     input = $15.00,  cached input = $7.50,  output = $60.00
      - o3-mini:                input = $1.10,   cached input = $0.55,  output = $4.40
      - o1-mini:                input = $1.10,   cached input = $0.55,  output = $4.40
    """
    # Define pricing for each model.
    pricing = {
        "openai/gpt-4.5-preview": {
            "input": 75.00, "cached_input": 37.50, "output": 150.00
        },
        "openai/gpt-4o": {
            "input": 2.50, "cached_input": 1.25, "output": 10.00
        },
        "openai/gpt-4o-audio-preview": {
            "input": 2.50, "cached_input": None, "output": 10.00
        },
        "openai/gpt-4o-realtime-preview": {
            "input": 5.00, "cached_input": 2.50, "output": 20.00
        },
        "openai/gpt-4o-mini": {
            "input": 0.15, "cached_input": 0.075, "output": 0.60
        },
        "openai/gpt-4o-mini-audio-preview": {
            "input": 0.15, "cached_input": None, "output": 0.60
        },
        "openai/gpt-4o-mini-realtime-preview": {
            "input": 0.60, "cached_input": 0.30, "output": 2.40
        },
        "openai/o1": {
            "input": 15.00, "cached_input": 7.50, "output": 60.00
        },
        "openai/o3-mini": {
            "input": 1.10, "cached_input": 0.55, "output": 4.40
        },
        "openai/o1-mini": {
            "input": 1.10, "cached_input": 0.55, "output": 4.40
        },
    }
    
    if model not in pricing:
        raise ValueError(f"Unknown model: {model}")
        
    model_rates = pricing[model]
    
    # Calculate cost per 1K tokens
    # For non-cached tokens, subtract the cached tokens (but not below zero)
    non_cached_input_tokens = max(0, input_tokens - cached_input_tokens)
    cost = non_cached_input_tokens / 1e6 * model_rates["input"]
    
    # Add cost for cached tokens if applicable (if not, they incur no cost)
    if model_rates.get("cached_input") is not None:
        cost += cached_input_tokens / 1e6 * model_rates["cached_input"]
        
    # Add cost for output tokens
    cost += output_tokens / 1e6 * model_rates["output"]
    
    return cost


def iterate_df_rows(
    df, 
    get_prompt=get_prompt_example, 
    response='str', 
    response_column='gpt_text', 
    is_valid=is_valid_placeholder, 
    max_iterations=5,
    verbose=False, 
    model='openai/gpt-4o-mini', 
    concurrency=30, 
    temperature=0.0, 
    max_tokens=500, 
    output_temp_file=None, 
    output_temp_every_n=10,
    subtract_input_tokens=False,
    drop_meta_columns=True
):
    """ 
    Call this with a dataframe and a prompt generation function. 
    It iterates through each row passing it to the specified prompt generation function.
    If response='dict' it unpacks the response into the data frame, otherwise it unpacks the string into response_column.
    If you specify an is_valid= function, then it checks each row to see if they're valid.
    It will rerun invalid rows up to max_iterations times.
    It uses its own ai_wrapper_success column to mark which rows have been successfully processed.
    If you run this multiple times it will only update rows with ai_wrapper_success==False.
    You can specify an output_temp_file to periodically save progress.
    """    
    
    # Helper function to approximate token count (adjust as needed)
    def count_tokens(text):
        return get_num_tokens(text)
    
    # Initialize counters for tokens
    total_input_tokens = 0
    total_output_tokens = 0
    
    if response != 'str' and response != 'dict':
        raise Exception(f'Faulty response parameter: {response}')
        
    df_result = df.copy()
    if 'ai_wrapper_success' not in df_result.columns:
        df_result['ai_wrapper_success'] = False
    else:
        # Ensure the column is boolean type
        df_result['ai_wrapper_success'] = df_result['ai_wrapper_success'].fillna(False).astype(bool)
        
    if 'iteration_count' not in df_result.columns:
        df_result['iteration_count'] = 0
        
    # Function to process a single row, now also returning token counts.
    def process_row(index, row):
        input_tokens = 0
        try:
            prompt = get_prompt(row)
            input_tokens = count_tokens(prompt)
            local_max_tokens = max_tokens-int(input_tokens*1.1) if subtract_input_tokens else max_tokens
            res = get_llm_text_response(prompt, model=model, max_tokens=max_tokens)
            output_tokens = count_tokens(res) if res is not None else 0
            if res is None:
                return index, None, False, input_tokens, output_tokens
            if verbose:
                logger.debug("Prompt: %s", prompt)
                logger.debug("Response: %s", res)
            
            if response == 'str':
                fields_dict = {response_column: res}
            elif response == 'dict':
                try:
                    fields_dict = extract_json_objects(res)[0]
                except Exception as e:
                    logger.error("Error parsing a JSON from this response: \n %s\nError: %s", res, e)
                    return index, None, False, input_tokens, output_tokens
                    
            # Check if the result is valid
            row_with_result = row.copy()
            for key, value in fields_dict.items():
                row_with_result[key] = value
            
            if is_valid(row_with_result):
                fields_dict.update({'ai_wrapper_success': True})
                return index, fields_dict, True, input_tokens, output_tokens
            else:
                return index, None, False, input_tokens, output_tokens
                
        except Exception as e:
            logger.error("Error processing row %s: %s", index, e, exc_info=True)
            # In case of exception, return the counted input tokens and 0 output tokens.
            return index, None, False, input_tokens, 0
            
    current_iteration = 0
    while current_iteration < max_iterations:
        logger.info("Starting iteration %d", current_iteration + 1)
        
        # Ensure ai_wrapper_success is boolean before filtering
        df_result['ai_wrapper_success'] = df_result['ai_wrapper_success'].fillna(False).astype(bool)
        
        rows_to_process = df_result[
            (~df_result.ai_wrapper_success) & 
            (df_result.iteration_count < max_iterations)
        ]
        
        if len(rows_to_process) == 0:
            logger.info("All rows are either valid or have reached max iterations")
            break
            
        logger.info("Processing %d rows in this iteration", len(rows_to_process))
        
        # Use ThreadPoolExecutor to process rows in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_row, i, row): i for i, row in rows_to_process.iterrows()}
            completed = 0
            futures_iter = concurrent.futures.as_completed(futures)
            if verbose:
                futures_iter = tqdm(futures_iter, total=len(futures))
            for future in futures_iter:
                index = futures[future]
                try:
                    idx, result, success, in_tok, out_tok = future.result()
                    # Update token counters
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    
                    # Update iteration count regardless of success
                    df_result.loc[idx, 'iteration_count'] += 1
                    
                    if success and result is not None:
                        # Update each column individually to avoid dtype issues
                        for col, val in result.items():
                            if col not in df_result.columns:
                                # If the column 'col' is new to df_result:
                                # 1. Assign `None` to df_result.at[idx, col] first.
                                #    This creates the column 'col'.
                                #    The cell (idx, col) gets `None`. Other cells in new col get `NaN`.
                                #    Crucially, the column's dtype will be set to `object`
                                #    (or be compatible with it), allowing it to hold diverse types
                                #    like your list of lists.
                                df_result.at[idx, col] = None 
                            df_result.at[idx, col] = val
                    
                    completed += 1
                    if output_temp_file and (completed % output_temp_every_n == 0):
                        logger.info("Outputting temp file at %d rows: %s", completed, output_temp_file)
                        save_to_file(df_result, output_temp_file)
                except Exception as e:
                    logger.error("Row %s generated an exception: %s", index, e, exc_info=True)
                    
        current_iteration += 1
        
        # Ensure boolean type before calculating statistics
        df_result['ai_wrapper_success'] = df_result['ai_wrapper_success'].fillna(False).astype(bool)
        
        # Log statistics for this iteration
        success_count = len(df_result[df_result.ai_wrapper_success])
        failed_count = len(df_result[~df_result.ai_wrapper_success])
        logger.info("Iteration %d complete: Successfully processed rows: %d | Remaining invalid rows: %d",
                     current_iteration, success_count, failed_count)
        
    # Get the current date and time
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Finished at: %s", formatted_date_time)
    
    # Report token counts at the end
    logger.info("Total input tokens: %d", total_input_tokens)
    logger.info("Total output tokens: %d", total_output_tokens)
    logger.info("Grand total tokens: %d", total_input_tokens + total_output_tokens)

    try:
        logger.info("Cost: $%.4f", get_cost(total_input_tokens, total_output_tokens, model=model))
    except Exception as e:
        logger.error("Exception getting cost: %s", e)
    if drop_meta_columns:
        df_result = df_result.drop(columns=['ai_wrapper_success','iteration_count'])
    return df_result


# Note: save_to_file function should be defined to handle saving the dataframe to a file.
def save_to_file(df, fname):
    logger.info("Saving file: %s", fname)
    if '.csv' in fname:
        return df.to_csv(fname, index=False)
    elif '.feather' in fname:
        return df.to_feather(fname)
    else:
        raise Exception('Unrecognised file format')


import json

# --- The updated get_categories_as_list function ---
def get_categories_as_list(
    categories: Union[str, Dict[str, Any]],
    include_descriptions: bool = False,
    one_level_hierarchy: bool = False # Crucial flag
) -> List[str]:
    """
    Converts a categories dictionary (or its JSON string representation) into a flat list.
    Handles both one-level and two-level category hierarchies based on the 'one_level_hierarchy' flag.

    Args:
        categories (Union[str, Dict[str, Any]]): The categories data.
            If one_level_hierarchy is True, expected format: {"Category1": "Description1", ...}
            If one_level_hierarchy is False, expected format:
                {"MainCategory1": {"Sub1": "Desc1", ...},
                 "MainCategory2": ["SubName1", "SubName2"],  // Subcategories as a list of names
                 "MainCategory3": {} // Main category with no subcategories explicitly defined yet
                }
        include_descriptions (bool, optional): Whether to include descriptions.
            For one-level: "Category: Description"
            For two-level: "Main.Sub: Description" (description of the subcategory)
            Not applicable if subcategories are provided as a list of names.
        one_level_hierarchy (bool): If True, treats the input as a flat
            {category: description} structure. If False, expects a two-level
            {main_category: {sub_category: description}} or
            {main_category: [sub_category_names]} structure.

    Returns:
        List[str]: A flat list of category strings.

    Raises:
        TypeError: If input types or structures are incorrect for the specified hierarchy.
        ValueError: If 'categories' is an invalid JSON string or structure is malformed.
    """
    if isinstance(categories, str):
        try:
            data = json.loads(categories)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string provided for categories: {e}") from e
    elif isinstance(categories, dict):
        data = categories
    else:
        raise TypeError(f"Unexpected category type: {type(categories)}. Expected str or dict.")

    if not isinstance(data, dict):
         raise ValueError(f"Parsed categories data is not a dictionary: {type(data)}")


    category_list = []

    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Category key must be a string, got {type(key)}: {key}")

        if one_level_hierarchy:
            if not isinstance(value, str):
                raise TypeError(
                    f"For one-level hierarchy, value for category '{key}' must be a string description, got {type(value)}."
                )
            entry = key
            if include_descriptions:
                entry += f": {value}"
            category_list.append(entry)
        else: # Two-level hierarchy
            main_category = key
            if isinstance(value, dict): # Subcategories with descriptions
                if not value: # e.g. "MainCategory": {}
                    category_list.append(main_category) # Represents a main category with no specified subcategories
                else:
                    for sub_key, sub_value in value.items():
                        if not isinstance(sub_key, str):
                             raise ValueError(f"Subcategory key for main category '{main_category}' must be a string, got {type(sub_key)}: {sub_key}")
                        if not isinstance(sub_value, str):
                            raise TypeError(
                                f"In two-level hierarchy, description for '{main_category}.{sub_key}' must be a string, got {type(sub_value)}."
                            )
                        entry = f"{main_category}.{sub_key}"
                        if include_descriptions:
                            entry += f": {sub_value}"
                        category_list.append(entry)
            elif isinstance(value, list): # Subcategories as a list of names
                if not value: # e.g. "MainCategory": []
                    category_list.append(main_category)
                else:
                    for sub_name in value:
                        if not isinstance(sub_name, str):
                            raise TypeError(
                                f"In two-level hierarchy, subcategory name in list for '{main_category}' must be a string, got {type(sub_name)}."
                            )
                        category_list.append(f"{main_category}.{sub_name}")
            else: # Value is not a dict or list, which is expected for two-level sub-items
                raise TypeError(
                    f"For two-level hierarchy, value for main category '{main_category}' must be a dict (of subcategories with descriptions) or a list (of subcategory names), got {type(value)}."
                )
    return category_list

# New helper function to process a list of dfs
def process_list_of_samples(list_of_dfs, get_prompt, categories):
    category_subcategory_list = get_categories_as_list(categories)
    # Process all the samples
    output_dfs = []
    for sample_df in list_of_dfs:
        processed_df = iterate_df_rows(
            sample_df,
            get_prompt,
            response='dict',
            model = 'gpt-4o-mini',
            concurrency=30
        )
        # exclude any categories not in the originial list
        processed_df.loc[~processed_df.category.isin(category_subcategory_list),'ai_wrapper_success'] = False
        while len(processed_df[~processed_df.ai_wrapper_success])>0:
            processed_df = iterate_df_rows(
                processed_df,
                get_prompt,
                response='dict',
                model = 'gpt-4o-mini',
                concurrency=30
            )
            processed_df.loc[~processed_df.category.isin(category_subcategory_list),'ai_wrapper_success'] = False
        output_dfs += [processed_df,]
    return output_dfs


def process_single_df_with_vote(df, get_prompt, categories, category_field='category', model='openai/gpt-4o-mini', process_temp_file_prefix=None, load_temp_file_if_exists=False, include_reasons=True):
    """
    Processes a DataFrame by iteratively classifying rows into categories using a specified model, ensuring consistency through multiple runs.

    This function takes a DataFrame and applies a voting mechanism to determine consistent categories for each row. It performs multiple runs of classification to ensure each row has a category label that appears at least twice, helping to mitigate inconsistencies.

    Additionally, it ensures all returned categories are valid by checking against the original category list and reprocessing invalid rows.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be processed.
    - get_prompt (function): A function to generate prompts for each row.
    - categories (list or dict): Categories and subcategories for classification.
    - category_field (str): The column name in the DataFrame where category classifications are stored (default is 'category').
    - model (str): Model name to use for processing (default is 'gpt-4o-mini').
    - process_temp_file_prefix (str, optional): Prefix for saving intermediate results, allowing recovery in case of interruptions.
    - load_temp_file_if_exists (bool, optional): If True, loads previously saved temporary files to resume processing.
    - include_reasons (bool, optional): If True, includes reasons for category classifications in the final output.

    Returns:
    - processed_df_final (pd.DataFrame): A DataFrame with consistent 'category' and 'reason' columns for each row.
    """    
    category_subcategory_list = get_categories_as_list(categories)

    # Initialize a list to track category and reason attempts for each row
    attempts_list = []

    if process_temp_file_prefix is not None:
        process_temp_file_1 = process_temp_file_prefix+'_run_1.csv'
        process_temp_file_2 = process_temp_file_prefix+'_run_2.csv'
    else:
        process_temp_file_1 = None
        process_temp_file_2 = None

    df1 = df.copy()
    df2 = df.copy()

    if load_temp_file_if_exists:
        if os.path.exists(process_temp_file_1):
            df1 = pd.read_csv(process_temp_file_1)
        if os.path.exists(process_temp_file_2):
            df2 = pd.read_csv(process_temp_file_2)
    
    # First processing run
    processed_df_1 = iterate_df_rows(
        df1,
        get_prompt,
        response='dict',
        model=model,
        concurrency=30,
        output_temp_file=process_temp_file_1,
        output_temp_every_n=100
    )
    processed_df_1.loc[~processed_df_1[category_field].isin(category_subcategory_list), 'ai_wrapper_success'] = False
    while len(processed_df_1[~processed_df_1['ai_wrapper_success']]) > 0:
        logger.info("Miscategorised %d", len(processed_df_1[~processed_df_1['ai_wrapper_success']]))
        logger.info("Categories: %s", processed_df_1[category_field].unique())
        #print (processed_df_1[~processed_df_1.ai_wrapper_success])
        processed_df_1 = iterate_df_rows(
            processed_df_1,
            get_prompt,
            response='dict',
            model=model,
            concurrency=30
        )
        processed_df_1.loc[~processed_df_1[category_field].isin(category_subcategory_list), 'ai_wrapper_success'] = False

    attempts_list.append(processed_df_1)
    
    # Second processing run
    processed_df_2 = iterate_df_rows(
        df2,
        get_prompt,
        response='dict',
        model=model,
        concurrency=30,
        output_temp_file=process_temp_file_2, 
        output_temp_every_n=100
    )
    processed_df_2.loc[~processed_df_2[category_field].isin(category_subcategory_list), 'ai_wrapper_success'] = False
    while len(processed_df_2[~processed_df_2['ai_wrapper_success']]) > 0:
        processed_df_2 = iterate_df_rows(
            processed_df_2,
            get_prompt,
            response='dict',
            model=model,
            concurrency=30
        )
        processed_df_2.loc[~processed_df_2[category_field].isin(category_subcategory_list), 'ai_wrapper_success'] = False

    attempts_list.append(processed_df_2)

    # Keep processing until each row's category matches twice
    final_categories = processed_df_1[category_field].copy()
    if include_reasons:
        final_reasons = processed_df_1.reason.copy()
    disagreement_indices = final_categories.index[final_categories != processed_df_2[category_field]].tolist()

    iteration = 0  # To track the number of iterations
    while len(disagreement_indices) > 0:
        iteration += 1
        logger.info("Iteration %d: Disagreement indices - %s", iteration, disagreement_indices)

        # Third and subsequent runs until a category appears twice for all rows
        new_processed_df = iterate_df_rows(
            df.loc[disagreement_indices],
            get_prompt,
            response='dict',
            model=model,
            concurrency=30
        )
        new_processed_df.loc[~new_processed_df[category_field].isin(category_subcategory_list), 'ai_wrapper_success'] = False
        while len(new_processed_df[~new_processed_df['ai_wrapper_success']]) > 0:
            new_processed_df = iterate_df_rows(
                new_processed_df,
                get_prompt,
                response='dict',
                model=model,
                concurrency=30
            )
            new_processed_df.loc[~new_processed_df[category_field].isin(category_subcategory_list), 'ai_wrapper_success'] = False

        attempts_list.append(new_processed_df)

        # Update the disagreement indices by checking if any row does not have a consistent category
        new_disagreement_indices = []
        for idx in disagreement_indices:
            category_attempts = [attempt[category_field][idx] for attempt in attempts_list if idx in attempt.index]
            category_counts = pd.Series(category_attempts).value_counts()
            if category_counts.iloc[0] >= 2:
                final_categories.loc[idx] = category_counts.idxmax()
                # Update the reason from a successful category attempt
                if include_reasons:
                    for attempt in attempts_list:
                        if attempt[category_field][idx] == final_categories.loc[idx]:
                            final_reasons.at[idx] = attempt.reason[idx]
                            break
            else:
                new_disagreement_indices.append(idx)

        # Recalculate disagreement indices for the next iteration
        disagreement_indices = new_disagreement_indices

    # Create the final processed dataframe
    processed_df_final = df.copy()
    processed_df_final[category_field] = final_categories
    if include_reasons:
        processed_df_final['reason'] = final_reasons

    return processed_df_final

