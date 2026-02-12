import os
import json
import re
from typing import Callable, Optional, List, Union

import pandas as pd
from cloudpathlib import AnyPath

import app.ai_wrapper as ai
from app.categorise import add_uncategorised_to_dict, dict_depth
from app.tag_manager import TagManager
from app.simple_logger import get_logger
from app.file_utils import (
    read_feather_from_anypath,
    write_feather_to_anypath,
)

logger = get_logger(__name__)

def derive_region_from_df(df: pd.DataFrame) -> str:
    """
    Derive a region "bucket" name from a dataframe's `tags.location` column.

    Returns "default" when the column is missing or empty. Output is a
    filesystem/identifier-friendly string: lowercase, spaces to '-', and only
    [a-z0-9-_] characters.
    """
    if "tags.location" not in df.columns:
        return "default"
    value_counts = df["tags.location"].value_counts()
    if value_counts.empty:
        return "default"
    top_value = value_counts.reset_index().loc[0]["tags.location"]
    s = str(top_value).strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_]", "", s)
    return str(s) if s else "default"


class PostProcessingPipeline:
    """
    Three-stage orchestration for posts:
    1) Political filter
    2) Category assignment
    3) Tagging with TagManager (stores tag id back onto posts)
    """

    def __init__(
        self,
        categories_path: Union[str, AnyPath],
        tags_path: Union[str, AnyPath],
        intermediary_path: Optional[Union[str, AnyPath]] = None,
    ) -> None:
        self.intermediary_path = AnyPath(intermediary_path) if intermediary_path is not None else None
        self.tags_path = AnyPath(tags_path)
        self.categories_path = AnyPath(categories_path)
        if not self.categories_path.exists():
            raise FileNotFoundError(f"Categories file not found at {self.categories_path}")

    @staticmethod
    def _read_posts_df(path: AnyPath) -> pd.DataFrame:
        """
        Read a dataframe from AnyPath in Feather format.
        """
        return read_feather_from_anypath(path)

    def save_final_df(
        self,
        input_path: Union[str, AnyPath],
        path: Optional[AnyPath] = None,
        overwrite: bool = False,
    ) -> Optional[AnyPath]:
        """
        Save the final (stage 3) dataframe to Feather.

        Call after process(). Uses the last pipeline result (self._df_stage3).
        Default path: "{intermediary_path}/{stem}-final.feather" where stem is derived from input_path.
        Requires intermediary_path to be set when path is None.

        If the output file already exists and overwrite is False, returns None without saving.
        """
        if not hasattr(self, "_df_stage3") or self._df_stage3 is None:
            raise RuntimeError("No pipeline result to save; run process() first.")
        
        if path is None:
            if self.intermediary_path is None:
                raise ValueError("intermediary_path required when path is None")
            input_path_obj = AnyPath(input_path)
            input_name = input_path_obj.name
            stem = input_name.rsplit(".", 1)[0] if "." in input_name else input_name
            output_filename = f"{stem}-final.feather"
            out_path = self.intermediary_path / output_filename
        else:
            out_path = path
        
        if out_path.exists() and not overwrite:
            logger.info("Final output file already exists, skipping save", extra={"path": str(out_path)})
            return None
        
        write_feather_to_anypath(self._df_stage3, out_path)
        logger.info("Saved final df", extra={"path": str(out_path), "rows": len(self._df_stage3)})
        return out_path

    def _run_stage(
        self,
        name: str,
        cache_path: Optional[AnyPath],
        run_fn: Callable[[], pd.DataFrame],
        *,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Run a pipeline stage with optional cache read/write."""
        if use_cache and cache_path and cache_path.exists():
            logger.info(f"{name}: detected existing output, skipping", extra={"path": str(cache_path)})
            return read_feather_from_anypath(cache_path)

        logger.info(f"{name}: starting")
        result = run_fn()

        if use_cache and cache_path:
            write_feather_to_anypath(result, cache_path)
            logger.info(f"{name}: completed and saved", extra={"path": str(cache_path), "rows": len(result)})
        else:
            logger.info(f"{name}: completed", extra={"rows": len(result)})

        return result

    def process(
        self,
        input_path: Union[str, AnyPath],
        columns_for_categorisation: Optional[List[str]] = ["body", "comment_texts"],
        run_context: str = "Local political/community issues",
        save_intermediary_files: bool = False,
        save_final: bool = False,
        overwrite_final: bool = False,
        add_uncategorised_opt: bool = False,
    ) -> pd.DataFrame:
        """
        Run the three-stage pipeline.

        Args:
            input_path: Full path to input posts file in Feather format.
            columns_for_categorisation: Columns to use for categorisation
            run_context: Context description for the pipeline
            save_intermediary_files: If True, save stage outputs as Feather files for checkpointing
            save_final: If True, save the final (stage 3) dataframe after completion
            overwrite_final: If True and save_final is True, overwrite existing final file

        When intermediary_path is set, intermediary and final files are saved/loaded
        from that directory. When intermediary_path is None, save_intermediary_files
        and save_final are silently ignored and the pipeline runs in memory only.

        When save_intermediary_files is True and intermediary_path is set, stage
        Feathers are read/written for checkpointing and resumability.

        When save_final is True and intermediary_path is set, the final (stage 3)
        dataframe is written to "{intermediary_path}/{stem}-final.feather".
        """
        if (save_intermediary_files or save_final) and self.intermediary_path is None:
            raise ValueError(
                "intermediary_path is required when save_intermediary_files or save_final is True"
            )

        input_path_obj = AnyPath(input_path)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input posts file not found: {input_path_obj}")

        can_save = self.intermediary_path is not None
        use_cache = can_save and save_intermediary_files

        def stage_path(suffix: str) -> Optional[AnyPath]:
            if not can_save:
                return None
            return self.intermediary_path / f"{input_path_obj.stem}-{suffix}.feather"

        self._df_stage1 = self._run_stage(
            "Stage 1", stage_path("stage1"),
            lambda: self._stage1_political_filter(self._read_posts_df(input_path_obj)),
            use_cache=use_cache,
        )

        # Filter to only local-political rows for stages 2+
        mask = self._df_stage1["is_localpolitical"].astype(str).str.lower().eq("true")
        df_political = self._df_stage1.loc[mask].copy()
        logger.info(
            "Filtered to local-political rows for stages 2+",
            extra={"total": len(self._df_stage1), "political": len(df_political)},
        )

        self._df_stage2 = self._run_stage(
            "Stage 2", stage_path("stage2"),
            lambda: self._stage2_categorise(
                df_political,
                columns_for_categorisation=columns_for_categorisation,
                context_prompt=run_context,
                add_uncategorised_opt=add_uncategorised_opt,
            ),
            use_cache=use_cache,
        )

        self._df_stage3 = self._run_stage(
            "Stage 3", stage_path("stage3"),
            lambda: self._stage3_tagging(self._df_stage2, run_context),
            use_cache=use_cache,
        )

        self.df_response = self._df_stage3

        if can_save and save_final:
            self.save_final_df(input_path, overwrite=overwrite_final)

        return self._df_stage3

    @staticmethod
    def _get_model(env_var: str, default: str) -> str:
        return os.getenv(env_var, default)

    @staticmethod
    def _stage1_political_filter(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the political filter prompt to add `is_localpolitical`.
        """
        model = PostProcessingPipeline._get_model("POSTS_STAGE1_MODEL", "google/gemini-2.5-flash")
        logger.info("Stage 1: using model", extra={"model": model})

        def get_prompt_is_localpolitical(row: pd.Series) -> str:
            return f"""
    You are a qualitative researcher dealing calmly with every topic.
    Using British English and social norms, please determine if the following post & comments is fits the following filter definition. 
    
    General filter: posts that fit broadly into the following categories: describe a public issue, political issue, social issue, community impact, or request for help/action.

    Specific examples: local services or infrastructure problems; public safety/antisocial behaviour; housing, homelessness, or landlord issues; access to healthcare/social care/benefits; cost-of-living pressures; schools/SEND/childcare; planning/development disputes; environmental concerns; local business/economic strain; discrimination/community tensions; accessibility/transport barriers; or governance, spending, and transparency concerns.

    If it fits the filter, then return only the word 'True' otherwise return 'False'

    Post: \"{row.body}\"
    Comments: \"{row.comment_texts}\"    

    Only return a single word!
    """

        def is_valid_is_localpolitical(row: pd.Series) -> bool:
            if "is_localpolitical" not in row.index:
                return False
            return row["is_localpolitical"] in ["True", "False"]

        df_result = ai.iterate_df_rows(
            df,
            get_prompt=get_prompt_is_localpolitical,
            is_valid=is_valid_is_localpolitical,
            response_column="is_localpolitical",
            model=model,
        )
        return df_result

    def _load_categories(self) -> dict:
        with self.categories_path.open("r") as f:
            return json.load(f)

    def _stage2_categorise(
        self,
        df: pd.DataFrame,
        context_prompt: str,
        columns_for_categorisation: Optional[List[str]],
        add_uncategorised_opt: Optional[bool]=False,
    ) -> pd.DataFrame:
        """
        Categorise posts using categories_to_study.json.
        Only rows flagged as political are sent to the LLM; others get a Non-specific label.
        """
        model = self._get_model("POSTS_STAGE2_MODEL", "google/gemini-2.5-flash")
        columns_to_use = columns_for_categorisation or ["body", "comment_texts"]
        categories_dict = self._load_categories()
        levels = dict_depth(categories_dict)
        is_one_level = levels == 1
        self._region = derive_region_from_df(df)

        categories_dict_local = categories_dict.copy()
        if add_uncategorised_opt:
            categories_dict_local = add_uncategorised_to_dict(
                categories_dict_local, levels_in_hierarchy=levels
            )

        categories_list = ai.get_categories_as_list(
            categories_dict_local,
            include_descriptions=False,
            one_level_hierarchy=is_one_level,
        )

        categories_descriptions = ai.get_categories_as_list(
            categories_dict_local,
            include_descriptions=True,
            one_level_hierarchy=is_one_level,
        )

        logger.info(
            "Stage 2: prepared categories",
            extra={"model": model, "levels": levels, "one_level": is_one_level, "categories_count": len(categories_list)},
        )

        def cat_prompt_func(row: pd.Series) -> str:
            entries = []
            for idx, col in enumerate(columns_to_use):
                val = row.get(col, "")
                try:
                    val = "" if pd.isna(val) else str(val)
                except (ValueError, TypeError):
                    val = str(val)
                entries.append(f"<{col}>{val}</{col}>")
            joined_entries = "\n".join(entries)
            base_prompt = f"""
You are a trained qualitative analyst.
Considering this item:
{joined_entries}
From the following list of categories, please choose the most appropriate one according to the context of {context_prompt} in {self._region}.

category_list = {categories_descriptions}
"""
            if is_one_level:
                base_prompt += "Only respond with the chosen category name!"
            else:
                base_prompt += "Only respond with 'category.subcategory' format!"
            #print (f"<prompt>\n{base_prompt}\n</prompt>")
            return base_prompt

        def is_valid_func(row: pd.Series) -> bool:
            if "category" not in row.index or pd.isna(row["category"]):
                return False
            return str(row["category"]) in categories_list

        df_result = df.copy()

        logger.info(f"Number of rows before categorisation: {len(df_result)}")
        df_result = ai.iterate_df_rows(
            df_result,
            get_prompt=cat_prompt_func,
            is_valid=is_valid_func,
            response_column="category",
            model=model,
        )
        logger.info(f"Number of rows after categorisation: {len(df_result)}")

        # Fill non-political or unprocessed rows
        fallback_value = "Non-specific.Non-specific" if not is_one_level else "Non-specific"
        na_count = df_result["category"].isna().sum()
        logger.info(f"Number of NA categories before fill: {na_count}")
        df_result["category"] = df_result["category"].fillna(fallback_value)

        # Split category into main_category (left of '.') and sub_category (right of '.')
        def _split_category(s: str) -> pd.Series:
            s = str(s).strip() if pd.notna(s) else ""
            if "." in s:
                a, b = s.split(".", 1)
                return pd.Series([a.strip(), b.strip()])
            return pd.Series([s, ""])

        df_result[["main_category", "sub_category"]] = df_result["category"].apply(_split_category)
        return df_result

    def _stage3_tagging(self, df: pd.DataFrame, context_description: str) -> pd.DataFrame:
        """
        Tag posts per category, storing tag id via TagManager.
        """
        model = self._get_model("POSTS_STAGE3_MODEL", "google/gemini-2.5-flash")
        self._region = derive_region_from_df(df)
        logger.info("Stage 3: starting TagManager", extra={"model": model, "tags_path": self.tags_path})
        with TagManager(tags_path=self.tags_path) as tag_manager:

            results: List[pd.DataFrame] = []
            main_categories = df["main_category"].dropna().unique().tolist() if "main_category" in df.columns else [None]
            logger.info("Stage 3: main_category batches", extra={"count": len(main_categories), "main_categories": main_categories})

            for main_category in main_categories:
                subset = df[df["main_category"] == main_category] if main_category is not None else df
                if subset.empty:
                    continue

                # main_category is used for tag records
                main_category = str(main_category).strip() if main_category else "Uncategorised"

                logger.info(
                    "Stage 3: processing main_category subset",
                    extra={"main_category": main_category, "rows": len(subset)},
                )

                tags_record_df = tag_manager.get_tags_for_category(main_category)[["tag", "tag_description"]].copy()

                # Build context and description emphasising locally salient issue and location
                context_label = "locally salient issue"
                if main_category:
                    ctx_desc = f"{context_description} — locally salient {main_category} issue in {self._region}."
                else:
                    ctx_desc = f"{context_description} — locally salient issue in {self._region}"

                processed_subset, _ = self._run_tagging_for_subset(
                    subset,
                    tags_record_df=tags_record_df,
                    context=context_label,
                    context_description=ctx_desc,
                    model=model,
                )

                # Preserve main_category and sub_category from original subset
                if "main_category" not in processed_subset.columns:
                    processed_subset["main_category"] = main_category
                if "sub_category" not in processed_subset.columns and "sub_category" in subset.columns:
                    # Preserve sub_category values by index
                    processed_subset["sub_category"] = subset.loc[processed_subset.index, "sub_category"]

                # Map to TagManager and attach tag_id
                for idx, row in processed_subset.iterrows():
                    tag = row.get("tag")
                    desc = row.get("tag_description", "")
                    if pd.isna(tag):
                        continue

                    existing_mask = tag_manager.df["tag"] == tag
                    if existing_mask.any():
                        tag_id = int(tag_manager.df.loc[existing_mask, "id"].iloc[0])
                        tag_manager.update_tag(tag_id, tag, main_category, desc)
                    else:
                        tag_id = tag_manager.add_new_tag(tag, main_category, desc)
                    processed_subset.at[idx, "tag_id"] = tag_id

                results.append(processed_subset)

            tag_manager.save()

        if results:
            combined = pd.concat(results, axis=0).sort_index()
            # Keep only tag_id on the df; tag/tag_description can be added via TagManager.merge_tags()
            combined = combined.drop(columns=["tag", "tag_description"], errors="ignore")
            return combined
        return df

    def _run_tagging_for_subset(
        self,
        df_subset: pd.DataFrame,
        tags_record_df: pd.DataFrame,
        context: str,
        context_description: str,
        model: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reuse the sequential tagging flow for a subset, keeping it simple.
        """
        from app.tagging_posts import iterate_tagging_posts_sequentially

        processed_df, updated_tags_df = iterate_tagging_posts_sequentially(
            df_subset,
            tags_record_df=tags_record_df,
            context=context,
            context_description=context_description,
            avoid="local, locally salient, issue",
            more_specific_than_column="category",
            response_column="tag",
            response_column_description="tag_description",
            model=model,
            drop_meta_columns=True,
        )
        return processed_df, updated_tags_df


def process_posts(
    input_path: Union[str, AnyPath],
    categories_path: Union[str, AnyPath],
    tags_path: Union[str, AnyPath],
    intermediary_path: Optional[Union[str, AnyPath]] = None,
    columns_for_categorisation: Optional[List[str]] = None,
    add_uncategorised_opt: bool = True,
    run_context: str = "Local political/community issues",
    save_intermediary_files: bool = False,
    save_final: bool = False,
    overwrite_final: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper to run the pipeline without instantiating the class.

    Args:
        input_path: Full path to input posts file in Feather format.
        categories_path: Path to categories JSON file
        tags_path: Base directory for tag records; created if missing.
        intermediary_path: Optional base path for intermediary and final outputs. Required when
            save_intermediary_files or save_final is True.
        columns_for_categorisation: Columns to use for categorisation
        add_uncategorised_opt: Whether to add uncategorised option (passed to stage 2)
        run_context: Context description for the pipeline
        save_intermediary_files: If True, save stage outputs as Feather files
        save_final: If True, save the final dataframe
        overwrite_final: If True, overwrite existing final file
    """
    pipeline = PostProcessingPipeline(
        categories_path=categories_path,
        tags_path=tags_path,
        intermediary_path=intermediary_path,
    )
    return pipeline.process(
        input_path=input_path,
        columns_for_categorisation=columns_for_categorisation,
        run_context=run_context,
        save_intermediary_files=save_intermediary_files,
        save_final=save_final,
        overwrite_final=overwrite_final,
        add_uncategorised_opt=add_uncategorised_opt,
    )
