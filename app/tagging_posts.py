import sys
import pandas as pd
import app.ai_wrapper as ai
import json
import app.google_sheets as gs
from datetime import datetime
from tqdm import tqdm

# TODO, post.body and post.comment_texts should be more freeform in terms of which columns to use.

from importlib import reload
reload(ai)

from app.simple_logger import get_logger

logger = get_logger(__name__)

def update_tags_record_df(df_posts: pd.DataFrame, df_tags_record: pd.DataFrame = None, response_column: str = 'tag', response_column_description: str = 'tag_description') -> pd.DataFrame:
    # Go throught the posts and add them to the tags record.

    # We assume df_tags_record has columns: tag, post_id, timestamp
    # df_posts should have at least: id (post id), tag (or about), and timestamp

    # We're assuming new tags are in df_posts with columns: id, tag/about, timestamp
    # First, rename/standardize for merging
    # Throw exceptions if required columns aren't present
    required_post_cols = {"id", response_column, response_column_description, "timestamp"}
    missing_cols = required_post_cols - set(df_posts.columns)
    if missing_cols:
        raise Exception(f"Input DataFrame df_posts is missing required columns: {missing_cols}")

    posts_tags = df_posts.rename(columns={"id": "post_id"})
    # Ensure the posts_tags DataFrame has a 'timestamp' column (should always be true if input checking above)

    # Only keep necessary columns
    posts_tags = posts_tags.loc[:, [response_column, response_column_description, "post_id", "timestamp"]]

    # Append posts, then trim duplicates so we only keep one row per unique tag/post/timestamp
    subset_cols = [response_column, response_column_description, "post_id", "timestamp"]
    df_tags_record = df_tags_record if df_tags_record is not None else pd.DataFrame(columns=subset_cols)
    df_tags_record = (
        pd.concat([df_tags_record, posts_tags], ignore_index=True)
        .drop_duplicates(subset=subset_cols, keep="first")
        .reset_index(drop=True)
    )

    return df_tags_record


def get_top_tags_from_tags_record(df_tags_record: pd.DataFrame, n_tags: int = 1000, verbose: bool = False, response_column: str = 'tag') -> pd.DataFrame:

    return df_tags_record.groupby(response_column).size().sort_values(ascending=False).head(n_tags).index.tolist()



# I removed this part because it's not needed for the new approach.
# Start by reviewing the current list of tags and descriptions, and only generate a new tag and description if an appropriate tag is not already in the list.
# current_list={json.dumps(df_tags_record[['tag','description']].to_dict(orient='records'), indent=4) if df_tags_record is not None else "Not yet generated"}


def extract_tags_from_posts(df_posts: pd.DataFrame, location: str = None) -> pd.DataFrame:
    def get_prompt_extract_local_issue(row):
        prompt = f"""
From the following post and comments, identify the specific **locally salient entity** being discussed — something that is specific to the {location} 
area (e.g., a local business, person, place, behaviour, or event).

If no clearly identifiable local entity is mentioned, return "Non-specific".

Post: "{row.body}"
Comments: "{row.comment_texts}"

Return your chosen tag and description as a JSON object with the following keys:
- tag
- tag_description (a short description of what or who this refers to, and its local relevance)
Example output:
{json.dumps({
"tag": "Very terse: the specific local entity or topic being discussed",
"tag_description": "A short description of what or who this refers to, and its local relevance",
}, indent=4)}

Only return the JSON object, nothing else.
"""
        return prompt

    def is_valid_extract_local_issue(row):
        if 'tag' not in row.index:
            return False
        if 'tag_description' not in row.index:
            return False
        if row['tag'] == '':
            return False
        if row['tag_description'] == '':
            return False
        return True

    df_posts_with_tags = ai.iterate_df_rows(
        df_posts.drop(columns=['ai_wrapper_success','iteration_count']),
        get_prompt=get_prompt_extract_local_issue,
        is_valid=is_valid_extract_local_issue,
        model='google/gemini-2.5-flash',
        response = 'dict'
    )

    return df_posts_with_tags

def amalgamate_tags(df_posts_with_tags: pd.DataFrame,df_tags_record: pd.DataFrame=None, location: str = None, verbose: bool = False) -> pd.DataFrame:
    prompt = f"""
    You are tagging posts for the {location} area.

    You are a helpful assistant that assesses tags.

    You are to read a list of new tags and their descriptions. 
    There is also a list of existing tags and their descriptions.

    You are to assess the new tags and their descriptions, based on these and the existing tags, you are 
    to come up with a list of new tags to be added to the existing list.

    New tags should be amalgamated together if they are very similar to each other.

    You can change any of the new tags and descriptions to make them more specific, broader or more accurate.

new_tags = {json.dumps(df_posts_with_tags[['tag','tag_description']].to_dict(orient='records'), indent=4)}
existing_tags = {json.dumps(df_tags_record[['tag','tag_description']].to_dict(orient='records'), indent=4) if df_tags_record is not None else "Not yet generated"}


    You are to return the new list of tags and descriptions in the following format:
    {json.dumps([
        {
            "tag": "The new tag",
            "tag_description": "The new tag description"
        }
    ], indent=4)}
    Include one object per tag you want to add.

    Only return the JSON object, nothing else. Do not return any of the existing tags and descriptions.
    """

    if verbose:
        logger.debug("Amalgamate tags prompt", extra={"prompt": prompt[:500]})

    new_tags = ai.get_llm_text_response(
        prompt,
        model='google/gemini-2.5-pro',
        verbose=verbose,
        response='list',
        max_tokens=1000000
    )

    if new_tags is None:
        return pd.DataFrame(columns=['tag', 'tag_description'])

    if isinstance(new_tags, dict):
        new_tags = [new_tags]

    return pd.DataFrame(new_tags, columns=['tag', 'tag_description'])


# So need a new function. Here is the current list of tags for the category. Here is a list of body+comment_texts. Please list any new tags that could be added in [{tag,description}] format


def _sequential_tagging_loop(
    df_posts: pd.DataFrame,
    tags_record_df: pd.DataFrame,
    prompt_fn,
    is_valid_fn,
    response_column: str,
    response_column_description: str,
    add_new_tags: bool = True,
    max_iterations: int = 5,
    verbose: bool = False,
    model: str = 'google/gemini-2.5-flash',
    max_tokens: int = 10000,
    subtract_input_tokens: bool = False,
    drop_meta_columns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core sequential loop: iterate over unfinished rows, call the LLM, validate,
    and optionally grow the tag record.  Shared by iterate_tagging_posts_sequentially
    and tag_with_existing.
    """
    def count_tokens(text):
        return ai.get_num_tokens(text)

    df_result = df_posts.copy()
    if 'ai_wrapper_success' not in df_result.columns:
        df_result['ai_wrapper_success'] = False
    else:
        df_result['ai_wrapper_success'] = df_result['ai_wrapper_success'].fillna(False).astype(bool)

    if 'iteration_count' not in df_result.columns:
        df_result['iteration_count'] = 0

    total_input_tokens = 0
    total_output_tokens = 0

    current_iteration = 0
    while current_iteration < max_iterations:
        if verbose:
            logger.info("Starting iteration", extra={"iteration": current_iteration + 1})

        df_result['ai_wrapper_success'] = df_result['ai_wrapper_success'].fillna(False).astype(bool)
        rows_to_process = df_result[
            (~df_result.ai_wrapper_success) &
            (df_result.iteration_count < max_iterations)
        ]

        if len(rows_to_process) == 0:
            if verbose:
                logger.info("All rows valid or at max iterations")
            break

        rows_count = len(rows_to_process)
        if verbose:
            logger.info("Processing rows", extra={"count": rows_count, "iteration": current_iteration + 1})

        row_iterator = rows_to_process.iterrows()
        if not verbose:
            row_iterator = tqdm(
                row_iterator,
                total=rows_count,
                desc=f"Iteration {current_iteration + 1}",
                leave=False,
                disable=not sys.stdout.isatty(),
            )

        for idx, row in row_iterator:
            prompt = prompt_fn(row, tags_record_df)
            input_tokens = count_tokens(prompt)
            local_max_tokens = max_tokens - int(input_tokens * 1.1) if subtract_input_tokens else max_tokens
            local_max_tokens = max(local_max_tokens, 1)

            try:
                res = ai.get_llm_text_response(prompt, model=model, max_tokens=local_max_tokens, verbose=verbose)
            except Exception as exc:
                if verbose:
                    logger.warning("Error processing row", extra={"index": idx, "error": str(exc)})
                df_result.loc[idx, 'iteration_count'] += 1
                total_input_tokens += input_tokens
                continue

            output_tokens = count_tokens(res) if res is not None else 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            if res is None:
                df_result.loc[idx, 'iteration_count'] += 1
                continue

            if verbose:
                logger.debug("Prompt/response", extra={"prompt": prompt[:200], "response": str(res)[:200]})

            try:
                fields_dict = ai.extract_json_objects(res)[0]
            except Exception as exc:
                if verbose:
                    logger.debug("Error parsing JSON", extra={"response": str(res)[:200], "error": str(exc)})
                df_result.loc[idx, 'iteration_count'] += 1
                continue

            if not isinstance(fields_dict, dict):
                if verbose:
                    logger.debug("Parsed response not a JSON object", extra={"parsed": str(fields_dict)})
                df_result.loc[idx, 'iteration_count'] += 1
                continue

            row_with_result = row.copy()
            for key, value in fields_dict.items():
                row_with_result[key] = value

            if is_valid_fn(row_with_result):
                fields_dict.update({'ai_wrapper_success': True})
                for col, val in fields_dict.items():
                    if col not in df_result.columns:
                        df_result.at[idx, col] = None
                    df_result.at[idx, col] = val

                # Update tag record with newly seen tags
                if {response_column, response_column_description}.issubset(fields_dict.keys()):
                    new_tag = fields_dict[response_column]
                    new_desc = fields_dict[response_column_description]
                    existing_mask = tags_record_df[response_column] == new_tag
                    if existing_mask.any():
                        if new_desc is not None:
                            tags_record_df.loc[existing_mask, response_column_description] = new_desc
                    elif add_new_tags:
                        tags_record_df = pd.concat(
                            [
                                tags_record_df,
                                pd.DataFrame([{response_column: new_tag, response_column_description: new_desc}])
                            ],
                            ignore_index=True
                        )
                    tags_record_df = tags_record_df.drop_duplicates(subset=[response_column], keep='last').reset_index(drop=True)
            else:
                df_result.loc[idx, 'ai_wrapper_success'] = False

            df_result.loc[idx, 'iteration_count'] += 1

        current_iteration += 1

        if verbose:
            success_count = len(df_result[df_result.ai_wrapper_success])
            failed_count = len(df_result[~df_result.ai_wrapper_success])
            logger.info("Iteration complete", extra={"iteration": current_iteration, "success": success_count, "failed": failed_count})

    if verbose:
        now = datetime.now()
        logger.info(
            "Finished",
            extra={
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
        )
        try:
            cost = ai.get_cost(total_input_tokens, total_output_tokens, model=model)
            logger.info("Cost", extra={"cost_usd": cost})
        except Exception as exc:
            logger.warning("Exception getting cost", extra={"error": str(exc)})

    processed_df = df_result
    if drop_meta_columns:
        processed_df = processed_df.drop(columns=['ai_wrapper_success', 'iteration_count'])

    return processed_df, tags_record_df


def iterate_tagging_posts_sequentially(
    df_posts: pd.DataFrame,
    tags_record_df: pd.DataFrame | None = None,
    context: str | None = None,
    context_description: str | None = None,
    avoid: str | None = None,
    more_specific_than_column: str | None = None,
    response_column: str | None = 'tag',
    response_column_description: str | None = None,
    get_prompt=None,
    is_valid=None,
    max_iterations: int = 5,
    verbose: bool = False,
    model: str = 'google/gemini-2.5-flash',
    max_tokens: int = 10000,
    subtract_input_tokens: bool = False,
    drop_meta_columns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequentially iterate through posts, prompting an LLM for tags and updating the tag record.
    Returns a tuple of (processed_posts_df, updated_tags_record_df).
    """
    if response_column_description is None:
        response_column_description = f"{response_column}_description"

    tags_record_df = tags_record_df.copy() if tags_record_df is not None else pd.DataFrame(columns=[response_column, response_column_description])

    def default_get_prompt_extract_local_issue(row, current_tags_record_df):
        existing_tags = current_tags_record_df[[response_column, response_column_description]].to_dict(orient='records') if not current_tags_record_df.empty else []
        more_specific_than_value = None
        if more_specific_than_column:
            try:
                v = row.get(more_specific_than_column, None)
            except Exception:
                v = None
            if v is not None and not pd.isna(v):
                s = str(v).strip()
                if s and s.lower() not in {"nan", "none"} and not s.lower().startswith("non-specific"):
                    more_specific_than_value = s

        avoid_instruction = ""
        if avoid is not None:
            avoid_instruction = f'Avoid using any of the following words/phrases in the tag (they are assumed): {avoid}.'
        elif context is not None:
            avoid_instruction = f'Avoid using the "{context}" or any part of the "{context}" in the tag as that is assumed.'

        specificity_instruction = ""
        if more_specific_than_value:
            specificity_instruction = f"""
Your chosen {context} MUST be more specific than: "{more_specific_than_value}".
It should be a narrower subtopic/entity/issue within that category, not the category name itself.
"""

        avoid_instruction_block = f"\n{avoid_instruction}\n" if avoid_instruction else ""

        tag_example_hint = (
            f"Very terse (<4 words): the {context} being discussed, use spaces to separate words"
            + (f", avoid: {avoid}" if avoid is not None else f", don't use '{context}' in the tag as that is assumed")
        )
        prompt = f"""
From the following post and comments, identify the specific **{context}** being discussed — {context_description}

If no clearly identifiable {context} is mentioned, return "Non-specific".

{specificity_instruction}
{avoid_instruction_block}
Consider using one of the existing tags that have been previously generated:
Existing_tags={json.dumps(existing_tags, indent=4)}

The post and comments are:

Post: "{row.body}"
Comments: "{row.comment_texts}"

Return your chosen {context} as {response_column}, and a description as a JSON object with the following keys:
- {response_column}
- {response_column_description} (the short description of what this refers to, and its relevance to the {context}, use spaces to separate words - you may update the old description if the new description needs to be more general)
Example output:
{json.dumps({
    "{response_column}": tag_example_hint,
    "{response_column_description}": "A short description of what this refers to, and its relevance to the {context}, use spaces to separate words - you may update the old description if the new description needs to be more general",
}, indent=4)}

Only return the JSON object, nothing else.
"""
        return prompt

    def default_is_valid_extract_local_issue(row):
        if response_column not in row.index or response_column_description not in row.index:
            return False
        if row[response_column] == '' or row[response_column_description] == '':
            return False
        return True

    prompt_fn = get_prompt if get_prompt is not None else default_get_prompt_extract_local_issue
    is_valid_fn = is_valid if is_valid is not None else default_is_valid_extract_local_issue

    return _sequential_tagging_loop(
        df_posts=df_posts,
        tags_record_df=tags_record_df,
        prompt_fn=prompt_fn,
        is_valid_fn=is_valid_fn,
        response_column=response_column,
        response_column_description=response_column_description,
        add_new_tags=True,
        max_iterations=max_iterations,
        verbose=verbose,
        model=model,
        max_tokens=max_tokens,
        subtract_input_tokens=subtract_input_tokens,
        drop_meta_columns=drop_meta_columns,
    )


def tag_with_existing(
    df: pd.DataFrame,
    tags_record_df: pd.DataFrame,
    context: str,
    context_description: str,
    avoid: str | None = None,
    more_specific_than_column: str | None = None,
    response_column: str = 'tag',
    response_column_description: str | None = None,
    get_prompt=None,
    is_valid=None,
    max_iterations: int = 5,
    verbose: bool = False,
    model: str = 'google/gemini-2.5-flash',
    max_tokens: int = 10000,
    subtract_input_tokens: bool = False,
    drop_meta_columns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tag posts using only existing tags from tags_record_df.  No new tags are
    created.  A sentinel "No tag" / "No appropriate tag" entry (tag_id = -1)
    is added to the local tag list so the LLM has an explicit escape hatch.

    Returns (processed_df, tags_record_df) where tags_record_df may contain
    updated descriptions for existing tags.
    """
    if response_column_description is None:
        response_column_description = f"{response_column}_description"

    tags_record_copy = tags_record_df.copy() if tags_record_df is not None else pd.DataFrame(columns=[response_column, response_column_description])

    # Build local copy with "No tag" sentinel
    local_tags_df = tags_record_copy.copy()
    no_tag_row = {response_column: "No tag", response_column_description: "No appropriate tag", "tag_id": -1}
    local_tags_df = pd.concat(
        [local_tags_df, pd.DataFrame([no_tag_row])],
        ignore_index=True,
    )

    existing_tag_names = set(local_tags_df[response_column].dropna().astype(str).tolist())

    # -- default prompt (existing-tags-only variant) -----------------------
    def default_get_prompt(row, current_tags_record_df):
        existing_tags = current_tags_record_df[[response_column, response_column_description]].to_dict(orient='records') if not current_tags_record_df.empty else []
        more_specific_than_value = None
        if more_specific_than_column:
            try:
                v = row.get(more_specific_than_column, None)
            except Exception:
                v = None
            if v is not None and not pd.isna(v):
                s = str(v).strip()
                if s and s.lower() not in {"nan", "none"} and not s.lower().startswith("non-specific"):
                    more_specific_than_value = s

        avoid_instruction = ""
        if avoid is not None:
            avoid_instruction = f'Avoid using any of the following words/phrases in the tag (they are assumed): {avoid}.'
        elif context is not None:
            avoid_instruction = f'Avoid using the "{context}" or any part of the "{context}" in the tag as that is assumed.'

        specificity_instruction = ""
        if more_specific_than_value:
            specificity_instruction = f"""
Your chosen {context} MUST be more specific than: "{more_specific_than_value}".
It should be a narrower subtopic/entity/issue within that category, not the category name itself.
"""

        avoid_instruction_block = f"\n{avoid_instruction}\n" if avoid_instruction else ""

        tag_example_hint = (
            f"Very terse (<4 words): the {context} being discussed, use spaces to separate words"
            + (f", avoid: {avoid}" if avoid is not None else f", don't use '{context}' in the tag as that is assumed")
        )
        prompt = f"""
From the following post and comments, identify the specific **{context}** being discussed — {context_description}

You MUST pick one of the existing tags exactly as written. Do not invent new tags.
If no existing tag is appropriate, use "No tag".

{specificity_instruction}
{avoid_instruction_block}
Existing_tags={json.dumps(existing_tags, indent=4)}

The post and comments are:

Post: "{row.body}"
Comments: "{row.comment_texts}"

Return your chosen {context} as {response_column}, and a description as a JSON object with the following keys:
- {response_column}
- {response_column_description} (the short description of what this refers to, and its relevance to the {context}, use spaces to separate words - you may update the old description if the new description needs to be more general)
Example output:
{json.dumps({
    "{response_column}": tag_example_hint,
    "{response_column_description}": "A short description of what this refers to, and its relevance to the {context}, use spaces to separate words - you may update the old description if the new description needs to be more general",
}, indent=4)}

Only return the JSON object, nothing else.
"""
        return prompt

    # -- default validator (must be an existing tag) -----------------------
    def default_is_valid(row):
        if response_column not in row.index or response_column_description not in row.index:
            return False
        if row[response_column] == '' or row[response_column_description] == '':
            return False
        if str(row[response_column]) not in existing_tag_names:
            return False
        return True

    prompt_fn = get_prompt if get_prompt is not None else default_get_prompt
    is_valid_fn = is_valid if is_valid is not None else default_is_valid

    processed_df, updated_tags_df = _sequential_tagging_loop(
        df_posts=df,
        tags_record_df=local_tags_df,
        prompt_fn=prompt_fn,
        is_valid_fn=is_valid_fn,
        response_column=response_column,
        response_column_description=response_column_description,
        add_new_tags=False,
        max_iterations=max_iterations,
        verbose=verbose,
        model=model,
        max_tokens=max_tokens,
        subtract_input_tokens=subtract_input_tokens,
        drop_meta_columns=drop_meta_columns,
    )

    # Strip the "No tag" sentinel before returning
    updated_tags_df = updated_tags_df[updated_tags_df["tag_id"] != -1].reset_index(drop=True)

    return processed_df, updated_tags_df


def generate_new_tags(
    df_posts: pd.DataFrame,
    tags_record_df: pd.DataFrame,
    context: str | None = None,
    context_description: str | None = None,
    avoid: str | None = None,
    response_column: str = 'tag',
    response_column_description: str | None = None,
    verbose: bool = False,
    model: str = 'google/gemini-2.5-pro',
    max_tokens: int = 1000000,
) -> pd.DataFrame:
    """
    Generate new tags from posts that were left untagged by tag_with_existing.

    Examines df_posts for rows where `response_column` is "No tag" (the
    sentinel used by tag_with_existing), builds a single prompt containing
    those posts' body + comments alongside the full existing tag list, and
    asks the LLM to propose new tags.  The target is at least
    max(num_untagged_posts // 3, 1) new tags but no more than 10.

    The newly generated tags are appended to tags_record_df (duplicates
    removed) and the updated DataFrame is returned.

    Parameters
    ----------
    df_posts : pd.DataFrame
        Output of tag_with_existing.  Must contain ``body``,
        ``comment_texts``, and the ``response_column``.
    tags_record_df : pd.DataFrame
        Current tag corpus with at least ``response_column`` and
        ``response_column_description`` columns.
    context / context_description / avoid
        Same meaning as in the other tagging helpers.
    response_column / response_column_description
        Column names used for tags and their descriptions.
    verbose : bool
        If True, log the prompt and response.
    model : str
        LLM model to use for generation.
    max_tokens : int
        Max tokens for the LLM call.

    Returns
    -------
    pd.DataFrame
        Updated ``tags_record_df`` with the new tags appended.
    """
    if response_column_description is None:
        response_column_description = f"{response_column}_description"

    tags_record_df = tags_record_df.copy() if tags_record_df is not None else pd.DataFrame(columns=[response_column, response_column_description])

    # --- Identify untagged posts -----------------------------------------
    if response_column not in df_posts.columns:
        logger.warning("generate_new_tags: response_column '%s' not found in df_posts; nothing to do.", response_column)
        return tags_record_df

    untagged_mask = df_posts[response_column].fillna('').astype(str).str.strip().isin({"No tag", ""})
    df_untagged = df_posts[untagged_mask]

    if df_untagged.empty:
        if verbose:
            logger.info("generate_new_tags: no untagged posts found; nothing to generate.")
        return tags_record_df

    num_untagged = len(df_untagged)
    min_new_tags = max(num_untagged // 3, 1)
    min_new_tags = min(min_new_tags, 10)

    # --- Build the post summaries ----------------------------------------
    post_summaries = []
    for _, row in df_untagged.iterrows():
        body = str(row.get("body", "")).strip()
        comments = str(row.get("comment_texts", "")).strip()
        post_summaries.append({"post": body, "comments": comments})

    # --- Build the existing tags list ------------------------------------
    existing_tags = (
        tags_record_df[[response_column, response_column_description]].to_dict(orient='records')
        if not tags_record_df.empty
        else []
    )

    # --- Context helpers -------------------------------------------------
    context_line = f" for the **{context}** category" if context else ""
    context_desc_line = f" ({context_description})" if context_description else ""
    avoid_line = f"\nAvoid using any of the following words/phrases in the tags (they are assumed): {avoid}." if avoid else ""

    # --- Prompt ----------------------------------------------------------
    prompt = f"""You are a helpful assistant that generates new tags{context_line}{context_desc_line}.

Below is a list of posts and their comments that could NOT be tagged with any existing tag.
There is also a list of existing tags and their descriptions for reference.

Your task:
1. Read through the untagged posts and identify new, distinct topics/entities/issues.
2. Generate **exactly {min_new_tags}** new tags (and short descriptions) that would cover these posts.
3. Do NOT duplicate or repeat any of the existing tags.
4. Each tag should be very terse (< 4 words) and each description should be a short sentence.
5. Amalgamate similar posts into a single tag where appropriate.
{avoid_line}

Untagged posts ({num_untagged} total):
{json.dumps(post_summaries, indent=2)}

Existing tags:
{json.dumps(existing_tags, indent=2)}

Return a JSON array of objects, each with:
- "{response_column}": the new tag
- "{response_column_description}": a short description

Example:
{json.dumps([
    {response_column: "Example tag", response_column_description: "Short description of the tag"},
], indent=4)}

Only return the JSON array, nothing else.
"""

    if verbose:
        logger.debug("generate_new_tags prompt", extra={"prompt": prompt[:500]})

    new_tags = ai.get_llm_text_response(
        prompt,
        model=model,
        verbose=verbose,
        response='list',
        max_tokens=max_tokens,
    )

    if new_tags is None:
        if verbose:
            logger.info("generate_new_tags: LLM returned None; no new tags generated.")
        return tags_record_df

    if isinstance(new_tags, dict):
        new_tags = [new_tags]

    if not isinstance(new_tags, list):
        logger.warning("generate_new_tags: unexpected response type %s", type(new_tags))
        return tags_record_df

    # --- Append to tags_record_df ----------------------------------------
    new_tags_df = pd.DataFrame(new_tags)
    # Keep only the expected columns (LLM may return extras)
    expected_cols = {response_column, response_column_description}
    new_tags_df = new_tags_df[[c for c in new_tags_df.columns if c in expected_cols]]

    if new_tags_df.empty or response_column not in new_tags_df.columns:
        if verbose:
            logger.info("generate_new_tags: LLM response contained no valid tags.")
        return tags_record_df

    tags_record_df = pd.concat([tags_record_df, new_tags_df], ignore_index=True)
    tags_record_df = tags_record_df.drop_duplicates(subset=[response_column], keep='last').reset_index(drop=True)

    if verbose:
        logger.info(
            "generate_new_tags: added %d new tag(s) to tags_record_df (total: %d)",
            len(new_tags_df),
            len(tags_record_df),
        )

    return tags_record_df


def suggest_tags_without_updating_corpus(
    df_posts: pd.DataFrame,
    tags_record_df: pd.DataFrame | None = None,
    context: str | None = None,
    context_description: str | None = None,
    more_specific_than_column: str | None = None,
    response_column: str | None = 'tag',
    response_column_description: str | None = None,
    allow_new_tags: bool = True,
    get_prompt=None,
    is_valid=None,
    max_iterations: int = 5,
    verbose: bool = False,
    model: str = 'google/gemini-2.5-flash',
    max_tokens: int = 500,
    subtract_input_tokens: bool = False,
    drop_meta_columns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate through posts (in parallel via ai.iterate_df_rows), prefer tagging with an
    existing corpus entry, and optionally suggest a new tag without mutating the corpus.
    Set allow_new_tags=False to force selecting from the provided tags list only.
    Flags are set per row to show whether an existing tag was used (`_succesfully_tagged_from_corpus`)
    or a new tag was suggested (`_new_tag_generated`). Returns (processed_posts_df, original_tags_record_df).
    """
    if response_column_description is None:
        response_column_description = f"{response_column}_description"

    tags_record_df = tags_record_df.copy() if tags_record_df is not None else pd.DataFrame(columns=[response_column, response_column_description])
    existing_tags = set(tags_record_df[response_column].dropna().astype(str).tolist()) if not tags_record_df.empty else set()
    if not allow_new_tags and not existing_tags:
        raise ValueError("allow_new_tags is False but no existing tags were provided (include a 'Non-specific' tag if needed).")

    def default_get_prompt(row, current_tags_record_df):
        existing_tags = current_tags_record_df[[response_column, response_column_description]].to_dict(orient='records') if not current_tags_record_df.empty else []
        tag_instruction = (
            "You must pick one of the existing tags exactly as written. Do not invent new tags."
            if not allow_new_tags
            else "Prefer using an existing tag, but you may propose a new tag if none apply."
        )
        more_specific_than_value = None
        if more_specific_than_column:
            try:
                v = row.get(more_specific_than_column, None)
            except Exception:
                v = None
            if v is not None and not pd.isna(v):
                s = str(v).strip()
                if s and s.lower() not in {"nan", "none"} and not s.lower().startswith("non-specific"):
                    more_specific_than_value = s

        specificity_instruction = ""
        if more_specific_than_value:
            specificity_instruction = f"""
Your chosen {response_column} MUST be more specific than: "{more_specific_than_value}".
It should be a narrower subtopic/entity/issue within that category, not the category name itself.
"""
        prompt = f"""
From the following post and comments, identify the specific **{context}** being discussed — {context_description}

If no clearly identifiable {context} is mentioned, return "Non-specific".

{specificity_instruction}
{tag_instruction}
Consider using one of the existing tags that have been previously generated:
Existing_tags={json.dumps(existing_tags, indent=4)}

The post and comments are:

Post: "{row.body}"
Comments: "{row.comment_texts}"

Return your chosen {context} as {response_column}, and a description as a JSON object with the following keys:
- {response_column}
- {response_column_description} (the short description of what this refers to, and its relevance to the {context}, use spaces to separate words - you may update the old description if the new description needs to be more general)
Example output:
{json.dumps({
    f"{response_column}": f"Very terse: the {context} being discussed, use spaces to separate words",
    f"{response_column_description}": f"A short description of what this refers to, and its relevance to the {context}, use spaces to separate words - you may update the old description if the new description needs to be more general",
}, indent=4)}

Only return the JSON object, nothing else.
"""
        return prompt

    def default_is_valid(row):
        if response_column not in row.index or response_column_description not in row.index:
            return False
        if row[response_column] == '' or row[response_column_description] == '':
            return False
        if not allow_new_tags and str(row[response_column]) not in existing_tags:
            return False
        return True

    prompt_fn = get_prompt if get_prompt is not None else default_get_prompt
    is_valid_fn = is_valid if is_valid is not None else default_is_valid

    df_result = df_posts.copy()
    # Preserve existing custom flags if present; otherwise initialise.
    for flag_col in ['_succesfully_tagged_from_corpus', '_new_tag_generated']:
        if flag_col not in df_result.columns:
            df_result[flag_col] = False
        else:
            df_result[flag_col] = df_result[flag_col].fillna(False).astype(bool)

    # Run parallel iterations; keep meta columns for our bookkeeping.
    df_result = ai.iterate_df_rows(
        df_result,
        get_prompt=lambda row: prompt_fn(row, tags_record_df),
        response='dict',
        response_column=response_column,
        is_valid=is_valid_fn,
        max_iterations=max_iterations,
        verbose=verbose,
        model=model,
        max_tokens=max_tokens,
        subtract_input_tokens=subtract_input_tokens,
        drop_meta_columns=False,
    )

    # Bookkeep corpus usage vs new tag suggestions.
    if response_column not in df_result.columns:
        df_result[response_column] = None
    df_result['_succesfully_tagged_from_corpus'] = df_result[response_column].fillna('').astype(str).isin(existing_tags)
    df_result['_new_tag_generated'] = ~df_result['_succesfully_tagged_from_corpus']

    processed_df = df_result
    if drop_meta_columns:
        cols_to_drop = [col for col in ['ai_wrapper_success', 'iteration_count'] if col in processed_df.columns]
        processed_df = processed_df.drop(columns=cols_to_drop)

    return processed_df


