import os
from glob import glob
from typing import Optional, Set

import pandas as pd

import app.utility_functions as ut
from app.simple_logger import get_logger

logger = get_logger(__name__)


def _debug(verbose: bool, message: str, *args) -> None:
    if verbose:
        logger.debug(message, *args)


def get_all_post_ids(all_posts_df: pd.DataFrame, verbose: bool = False) -> Set:
    """Return a set of post identifiers present in ``all_posts_df``.

    Parameters
    ----------
    all_posts_df:
        DataFrame of posts that should contain a ``post.id`` column.
    verbose:
        When ``True``, emit debug logs about the IDs that are extracted.

    Returns
    -------
    set
        Set of unique post identifiers. Empty set if the column is missing.
    """
    if 'post.id' not in all_posts_df.columns:
        _debug(verbose, "Requested post IDs from DataFrame without 'post.id'")
        return set()

    unique_ids = set(all_posts_df['post.id'])
    _debug(verbose, "Extracted %d unique post IDs", len(unique_ids))
    return unique_ids



def get_posts_with_top_n_comments(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    n_comments: int = 10,
    min_comments: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return posts paired with their top ``n_comments`` comments.

    Parameters
    ----------
    posts_df:
        DataFrame of posts. Must include ``post.id`` (used as the join key).
    comments_df:
        DataFrame of comments. Must include ``post.id`` plus the fields used by
        :func:`utility_functions.get_top_n_comments_per_post` (typically
        ``likes`` and ``body``).
    n_comments:
        Number of comments to keep per post, as defined by
        :func:`utility_functions.get_top_n_comments_per_post`.
    min_comments:
        Minimum number of comments required before a post is included.
    verbose:
        When ``True``, emit detailed debug logs during processing.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame of posts and their top comments. Empty if the inputs
        do not produce any matches.
    """
    if posts_df is None or posts_df.empty:
        logger.info("No posts provided; returning empty DataFrame")
        return pd.DataFrame()

    if 'post.id' not in posts_df.columns:
        logger.info("Posts DataFrame missing 'post.id'; returning empty DataFrame")
        return pd.DataFrame()

    if comments_df is None or comments_df.empty:
        logger.info("No comments provided; returning empty DataFrame")
        return pd.DataFrame()

    if 'post.id' not in comments_df.columns:
        logger.info("Comments DataFrame missing 'post.id'; returning empty DataFrame")
        return pd.DataFrame()

    all_post_ids = get_all_post_ids(posts_df, verbose=verbose)
    if not all_post_ids:
        logger.info("No post IDs found in posts DataFrame; returning empty DataFrame")
        return pd.DataFrame()

    filtered_comments_df = comments_df[comments_df['post.id'].isin(all_post_ids)].copy()
    if filtered_comments_df.empty:
        logger.info("No comments matched provided post IDs; returning empty DataFrame")
        return pd.DataFrame()

    # Remove duplicate comment and post rows prior to merging.
    original_comment_count = len(filtered_comments_df)
    comment_dedupe_subset = [c for c in ('id', 'body') if c in filtered_comments_df.columns]
    if comment_dedupe_subset:
        filtered_comments_df = filtered_comments_df.drop_duplicates(
            ignore_index=True,
            subset=comment_dedupe_subset,
        )
    else:
        filtered_comments_df = filtered_comments_df.drop_duplicates(ignore_index=True)
    if original_comment_count != len(filtered_comments_df):
        _debug(
            verbose,
            "Dropped %d duplicate comments",
            original_comment_count - len(filtered_comments_df),
        )

    original_post_count = len(posts_df)
    post_dedupe_subset = [c for c in ('id', 'body') if c in posts_df.columns]
    if post_dedupe_subset:
        posts_df = posts_df.drop_duplicates(ignore_index=True, subset=post_dedupe_subset)
    else:
        posts_df = posts_df.drop_duplicates(ignore_index=True)
    if original_post_count != len(posts_df):
        _debug(
            verbose,
            "Dropped %d duplicate posts",
            original_post_count - len(posts_df),
        )

    top_comments_df = ut.get_top_n_comments_per_post(
        filtered_comments_df,
        n_comments=n_comments,
        min_comments=min_comments,
        post_id_field='post.id',
        sort_field='likes',
        text_field='body'
    )
    _debug(verbose, "Top comments extraction produced %d rows", len(top_comments_df))

    posts_with_comments_df = pd.merge(posts_df, top_comments_df, on='post.id')
    logger.info(
        "Merged posts and top comments; resulting rows: %d", len(posts_with_comments_df),
    )
    return posts_with_comments_df
