import pandas as pd
import numpy as np
import warnings
from urllib.parse import urlparse
import re
import requests
import time
import concurrent.futures
from tqdm import tqdm
import json
from html import unescape

from app.simple_logger import get_logger

logger = get_logger(__name__)

# A list of all the url paths that facebook uses for internal shares
facebook_internal = ['photo.php', 'groups', 'reel', 'photo', 'story.php',
       'share', 'l.php', 'hashtag', 'events', 'login','profile.php','people','marketplace','watch']

def get_fb_group_account_user_id(row):
    if row.account_username is not None:
        if '/' in row.account_username:
            return row.account_username.split('/')[1]
        else:
            return row.account_username
    else: 
        return row.account_id

def get_top_n_comments_per_post(df, n_comments=3, min_comments=None, post_id_field='post_id', sort_field='comment_likes_count', text_field='comment_text'):
    """
    Get top n comments per post, optionally filtering for posts with at least min_comments.
    
    Parameters:
    - df: DataFrame containing post and comment data
    - n_comments: Number of top comments to return per post (default: 3)
    - min_comments: Minimum number of comments a post must have to be included (default: None)
    
    Returns:
    - DataFrame with post metadata and lists of top comments
    """
    # Sort by 'post_id' and 'comment_likes_count' 
    df_sorted = df.sort_values(by=[post_id_field, sort_field], ascending=[True, False])
    
    # If min_comments is specified, filter posts
    if min_comments is not None:
        # Count comments per post
        comment_counts = df_sorted[post_id_field].value_counts()
        # Get posts that meet the minimum threshold
        valid_posts = comment_counts[comment_counts >= min_comments].index
        # Filter the dataframe
        df_sorted = df_sorted[df_sorted[post_id_field].isin(valid_posts)]
    
    # Get top n comments per post
    top_comments_df = df_sorted.groupby(post_id_field).head(n_comments)
    
    
    # Collect comments into lists
    comment_lists_df = (
        top_comments_df.groupby(post_id_field)[text_field]
        .agg(list)
        .reset_index()
        .rename(columns={text_field: 'comment_texts'})
    )
    
    # Merge metadata with comment lists
    
    return comment_lists_df

def get_n_items_per_group(df, 
                         group_by_field, 
                         n_items=3,
                         how='top',
                         sort_by_field=None,
                         min_items=None,
                         item_field=None,
                         aggregate=True,
                         metadata_fields=None,
                         random_state=None):
    """
    Get n items per group using different selection methods, optionally filtering for groups with at least min_items.
    
    Parameters:
    - df: DataFrame containing grouped data
    - group_by_field: Field(s) to group by
                      Can be a string (single field) or a list of strings (multiple fields)
    - n_items: Number of items to return per group (default: 3)
    - how: Method to select items ('top', 'sample', or 'roulette') (default: 'top')
        - 'top': Select top n items based on sort_by_field
        - 'sample': Randomly sample n items 
        - 'roulette': Weighted random sampling using sort_by_field as weights
    - sort_by_field: Field to sort or weight items by (default: None)
                    Required if how='top' or how='roulette'
    - min_items: Minimum number of items a group must have to be included (default: None)
    - item_field: Field containing items to aggregate into lists (default: None)
                  Required if aggregate=True
    - aggregate: Whether to aggregate items into lists (default: True)
                 If False, returns the selected items without aggregation
    - metadata_fields: List of fields to include as group metadata (default: None)
                       If None, will use all columns from the original DataFrame
                       except item_field (if specified)
    - random_state: Random seed for reproducibility (default: None)
    
    Returns:
    - If aggregate=True: DataFrame with group metadata and lists of selected items
    - If aggregate=False: DataFrame with selected items (not aggregated)
    """
    import numpy as np
    import pandas as pd
    
    # Check if item_field is provided when aggregate=True
    if aggregate and item_field is None:
        raise ValueError("item_field must be provided when aggregate=True")
    
    # Check if sort_by_field is provided when how is 'top' or 'roulette'
    if (how == 'top' or how == 'roulette') and sort_by_field is None:
        raise ValueError(f"sort_by_field must be provided when how='{how}'")
    
    # Convert group_by_field to list if it's a string
    group_by_fields = [group_by_field] if isinstance(group_by_field, str) else group_by_field
    
    # If metadata_fields not specified, use all columns except item_field (if provided)
    if metadata_fields is None:
        exclude_fields = []
        if item_field is not None:
            exclude_fields.append(item_field)
        metadata_fields = [col for col in df.columns if col not in exclude_fields]
    else:
        # Use the provided metadata_fields
        metadata_fields = list(metadata_fields)
        
    # Ensure sort_by_field is included in metadata_fields when it's used
    if sort_by_field is not None and (how == 'top' or how == 'roulette') and sort_by_field not in metadata_fields:
        metadata_fields.append(sort_by_field)
    
    # If min_items is specified, filter groups
    if min_items is not None:
        # Count items per group
        item_counts = df.groupby(group_by_fields).size()
        # Get groups that meet the minimum threshold
        valid_groups = item_counts[item_counts >= min_items].reset_index()[group_by_fields]
        
        # Filter the dataframe - handle both single and multiple group fields
        if len(group_by_fields) == 1:
            valid_values = valid_groups[group_by_fields[0]].tolist()
            df = df[df[group_by_fields[0]].isin(valid_values)]
        else:
            # For multiple fields, create a merge key
            merge_df = df.merge(valid_groups, on=group_by_fields, how='inner')
            df = merge_df
    
    # Create an empty dataframe to store results
    selected_items_df = pd.DataFrame()
    
    # Group the dataframe
    grouped = df.groupby(group_by_fields)
    
    if how == 'top':
        # Sort by group fields and sort field
        sort_columns = group_by_fields + [sort_by_field]
        ascending = [True] * len(group_by_fields) + [False]
        df_sorted = df.sort_values(by=sort_columns, ascending=ascending)
        
        # Get top n items per group
        selected_items_df = df_sorted.groupby(group_by_fields).head(n_items)
        
    elif how == 'sample':
        # Random sampling for each group
        selected_items_df = grouped.apply(
            lambda x: x.sample(min(n_items, len(x)), random_state=random_state)
        ).reset_index(drop=True)
        
    elif how == 'roulette':
        # Function for weighted random sampling
        def weighted_sample(group, n, weight_col, random_state=None):
            if len(group) <= n:
                return group
            
            # Handle negative or zero weights
            weights = group[weight_col].values
            weights = np.maximum(weights, 0)  # Convert negatives to zero
            
            # Count non-zero weights
            non_zero_count = np.count_nonzero(weights)
            
            # If not enough non-zero weights, use top n items
            if non_zero_count < n:
                return group.sort_values(by=weight_col, ascending=False).head(n)
            
            # If all weights are zero, use uniform sampling
            if weights.sum() == 0:
                return group.sample(n, random_state=random_state)
            
            # Roulette wheel selection
            probs = weights / weights.sum()
            selected_indices = np.random.RandomState(random_state).choice(
                len(group), size=n, replace=False, p=probs
            )
            return group.iloc[selected_indices]
        
        # Apply roulette wheel sampling to each group
        selected_items_df = grouped.apply(
            lambda x: weighted_sample(x, min(n_items, len(x)), sort_by_field, random_state)
        ).reset_index(drop=True)
    
    else:
        raise ValueError("'how' must be one of 'top', 'sample', or 'roulette'")
    
    # If aggregate is False, return the selected items without aggregation
    if not aggregate:
        return selected_items_df
    
    # Get unique group metadata
    groups_metadata = selected_items_df[metadata_fields].drop_duplicates(subset=group_by_fields)
    
    # Collect items into lists
    item_lists = (
        selected_items_df.groupby(group_by_fields)[item_field]
        .agg(list)
        .reset_index()
    )
    
    # Merge metadata with item lists
    result_df = pd.merge(groups_metadata, item_lists, on=group_by_fields)
    
    return result_df

def roulette_wheel_selection(df, count_column, n_samples=1):
    # Normalize the 'count' values to get probabilities
    total_count = df[count_column].sum()
    probabilities = df[count_column] / total_count
    
    # Sample with these probabilities
    sampled_df = df.sample(n=n_samples, weights=probabilities, replace=False)
    
    return sampled_df


def balanced_sample(df, column, sample_size=None, power=0.5):
    """
    Sample from dataframe with controlled downweighting of common values in column.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to sample from
    column : str
        Column name containing the identifiers to balance
    sample_size : int or None
        Size of the sample to return, defaults to original df size if None
    power : float
        Controls downweighting strength (0-1):
        - 0: equal probability for all rows (ignores MP frequency)
        - 1: perfect inverse frequency weighting
        - 0.5 (default): square root of inverse frequency
    
    Returns:
    --------
    pandas DataFrame
        Sampled dataframe
    """
    # Calculate frequency of each MP
    id_counts = df[column].value_counts()
    
    # Calculate sampling weights (inverse frequency raised to specified power)
    weights = 1 / (id_counts.pow(power))
    
    # Map weights back to each row
    row_weights = df[column].map(lambda x: weights[x])
    
    # Determine sample size if not specified
    if sample_size is None:
        sample_size = len(df)
    
    # Sample with calculated weights
    return df.sample(n=sample_size, weights=row_weights, replace=False)

def url_extract_first_portion(url):
    """ Uses a regex pattern to match the portion after the first '/' and before any delimiter """
    pattern = r"https?://[^/]+/([^/?#]+)"
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    else:
        return None

def url_extract_second_portion(url):
    """ Uses a regex pattern to match the portion after the second '/' and before the third """
    # Regex pattern to match the second portion between slashes
    pattern = r"https?://[^/]+/[^/]+/([^/?#]+)"
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    else:
        return None

def process_json_with_item_descriptions(json_obj):
    """ Function used by the format_json_text_to_df function to process JSON with item descriptions """
    results = []
    for key1, val1 in json_obj.items():
        results.append((key1, '', ''))
        
        for key2, val2 in val1.items():
            first_elem = True
            
            for key3, val3 in val2.items():
                item_description = val3 if isinstance(val3, str) else ''
                
                if first_elem:
                    results.append((key2, key3, item_description))
                    first_elem = False
                else:
                    results.append(('', key3, item_description))
    return results


def format_json_text_to_df(text, add_root=True):
    """ Takes a JSON text tree and returns a DataFrame of the tree in a human-readable format """
    json_obj = json.loads(text)
    if add_root:
        objs = process_json_with_item_descriptions({'': json_obj})
    else:
        objs = process_json_with_item_descriptions(json_obj)
    df = pd.DataFrame(objs, columns=['Category', 'Subcategory', 'Item Description'])
    return df


def normalize_domain(url):
    """This function takes a URL and returns a normalized domain string.
It strips of www. m. or l. from the front as well.
    """
    if not isinstance(url,str):
        return "<empty>"
    if not urlparse(url).scheme:
        url = 'http://' + url
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    if domain.startswith('m.'):
        domain = domain[2:]
    if domain.startswith('l.'):
        domain = domain[2:]
    return domain

def compare_domains(url1, url2):
    def extract_domain(url):
        # Parse the URL and extract the netloc part, then split to remove possible 'www.'
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Optionally remove 'www.' if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    # Extract domains
    domain1 = extract_domain(url1)
    domain2 = extract_domain(url2)

    # Compare domains
    return domain1 == domain2


SHORTENED_DOMAINS = [
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly"
]

def is_shortened(url):
    if not isinstance(url,str):
        return False
    url_domain = extract_domain(url)
    # Create a regex pattern to match the exact domains
    pattern = r'\b(?:' + '|'.join(re.escape(domain) for domain in SHORTENED_DOMAINS) + r')\b'
    return re.search(pattern, url_domain) is not None

def extract_domain(url):
    """ Function to extract the domain from url """
    try:
        # Ensure the URL is a string
        if not isinstance(url, str):
            return ''
        if url == '<empty>':
            return url
            
        # Replace incorrect 'https://https/' prefix
        url = re.sub(r'^https://https?/', 'https://', url)
        
        # Ensure the URL has the correct format
        if not re.match(r'^https?://', url):
            url = 'https://' + url

        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove scheme or any other incorrect values
        if not domain:
            raise ValueError("Invalid URL, domain could not be extracted.")
        return domain
    except Exception as e:
        logger.warning("Error parsing URL", extra={"url": url[:100] if isinstance(url, str) else None, "error": str(e)})
        raise

def convert_to_list(x):
    if pd.isna(x):
        return None
    try:
        return eval(x)
    except:
        return None

def load_csv_and_detect_column_types(file_path, chunk_size=100000, verbose=False):
    """
    This function loads a CSV file while attempting to convert specific columns based on their names:
    - Columns containing '_id' are treated as strings.
    - Columns containing '_count' are treated as integers.
    - Columns containing '_date' or '_time' are converted to datetime objects.
    
    The function handles bad rows gracefully by replacing invalid data with safe defaults. It also reports the total
    number of bad data cells encountered, and if verbose mode is enabled, it outputs detailed information about those cells.
    """
    def detect_column_type(column_name):
        if "_id" in column_name:
            return str
        elif "_count" in column_name:
            return 'Int64'  # Use pandas nullable integer type
        elif "_date" in column_name or "_time" in column_name:
            return 'datetime'
        return None

    logger.info("Reading CSV to infer column types", extra={"path": file_path})
    df = pd.read_csv(file_path, nrows=0, low_memory=False)
    
    dtype_map = {}
    parse_dates = []
    for col in df.columns:
        col_type = detect_column_type(col)
        if col_type == 'datetime':
            parse_dates.append(col)
        elif col_type:
            dtype_map[col] = col_type
    
    def process_chunk(chunk):
        for col, dtype in dtype_map.items():
            if dtype == 'Int64':
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('Int64')
        return chunk

    logger.info("Loading CSV in chunks", extra={"path": file_path})
    chunks = []
    bad_cells_count = 0
    bad_cells_info = []
    
    for chunk in pd.read_csv(file_path, dtype=dtype_map, parse_dates=parse_dates, 
                             chunksize=chunk_size, low_memory=False):
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)
        
        if verbose:
            for col in dtype_map:
                if dtype_map[col] == 'Int64':
                    invalid_vals = processed_chunk[col].isna()
                    bad_cells_count += invalid_vals.sum()
                    bad_cells_info.extend(processed_chunk[invalid_vals].to_dict(orient='records'))

    final_df = pd.concat(chunks, ignore_index=True)
    
    if verbose and bad_cells_info:
        warnings.warn(f"Verbose mode: Listing bad data cells:\n {bad_cells_info}")
    
    logger.info(
        "Finished processing CSV",
        extra={
            "path": file_path,
            "rows": len(final_df),
            "cells": len(final_df) * len(final_df.columns),
            "bad_cells": bad_cells_count,
        },
    )
    
    return final_df

def get_post_link(row):
    """ Helper function for Facebook posts to get a link to the post """
    account = ''
    if row.account_id is not None:
        account = str(row.account_id).strip('.0')
    else:
        account = row.account_username
    return f"https://www.facebook.com/groups/{account}/posts/{str(row.post_id).strip('.0')}"

def get_group_link(row):
    """ Helper function for Facebook posts to get a link to the group"""
    account = ''
    if row.account_id is not None:
        account = str(row.account_id).strip('.0')
    else:
        account = row.account_username
    return f"https://www.facebook.com/groups/{account}"


def has_text(entry):
    """ If an unspecified entry in a DataFrame has some text in it then this returns True, otherwise it returns False """
    if isinstance(entry,str):
        if len(entry)>0:
            return True
    return False

def clean_and_join(df1, df2, key, suffixes=('_df1', '_df2')):
    """ Function that takes two dataframes and joins them based on key. It makes an effort to clean the key as much as possible as it can be sometimes a little corrupted when loaded from a text file. You should make sure it is loaded properly in the first place."""
    # Make copies of the DataFrames to avoid modifying the original ones
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Step 1: Strip leading and trailing whitespace from the key columns
    df1_copy[key] = df1_copy[key].astype(str).str.strip()
    df2_copy[key] = df2_copy[key].astype(str).str.strip()

    # Step 2: Remove rows where the key column contains null values
    df1_copy = df1_copy.dropna(subset=[key])
    df2_copy = df2_copy.dropna(subset=[key])

    # Step 3: Ensure both key columns are of the same data type (string)
    df1_copy[key] = df1_copy[key].astype(str)
    df2_copy[key] = df2_copy[key].astype(str)

    # Step 4: Perform type checks to confirm both columns are strings
    logger.debug("df1 key types", extra={"value_counts": str(df1_copy[key].apply(type).value_counts())})
    logger.debug("df2 key types", extra={"value_counts": str(df2_copy[key].apply(type).value_counts())})

    # Step 5: Attempt to join the DataFrames on the key column with suffixes for overlapping columns
    try:
        result = df1_copy.set_index(key).join(df2_copy.set_index(key), how='inner', lsuffix=suffixes[0], rsuffix=suffixes[1])
        logger.info("Join successful", extra={"key": key})
        return result
    except Exception as e:
        logger.warning("Join failed", extra={"key": key, "error": str(e)})
        return None
