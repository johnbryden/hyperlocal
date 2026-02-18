from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import dotenv
import elasticsearch
import os

from cloudpathlib import AnyPath

import app.file_utils as fu
import app.posts_and_comments as pacs
import app.process_week as pw
from app.simple_logger import get_logger

logger = get_logger(__name__)


def sync_all_downloads(
    data_root: AnyPath,
    target_locations: Sequence[str],
    first_start: datetime | None = None,
    *,
    week_start: int = 0,
    redownload: bool = False,
) -> None:
    """
    Walk week-by-week from *first_start* up to the most recent complete week,
    downloading posts and comments for each location that hasn't been fetched yet.

    If *redownload* is ``True``, existing files are overwritten.
    """
    dotenv.load_dotenv()

    es = elasticsearch.Elasticsearch(
        cloud_id=os.getenv("ES_CLOUD_ID"),
        api_key=os.getenv("API_KEY"),
    )
    logger.info("Created ES client")

    if first_start is None:
        first_start = datetime(2026, 2, 2, tzinfo=timezone.utc)

    _, latest_end = pw.get_most_recent_week(
        tz=timezone.utc, week_start=week_start, weeks_ago=0,
    )

    start_dt = first_start
    while start_dt < latest_end:
        end_dt = start_dt + timedelta(days=7)
        logger.info(
            "Checking week",
            extra={"start": start_dt.strftime("%Y-%m-%d"), "end": end_dt.strftime("%Y-%m-%d")},
        )

        files_need_downloading = redownload

        if not redownload:
            for location in target_locations:
                fout = fu.file_name_to_slug(
                    data_root / f"posts_{location}_{start_dt.strftime('%Y-%m-%d')}_to_{end_dt.strftime('%Y-%m-%d')}.feather"
                )
                if not AnyPath(fout).exists():
                    files_need_downloading = True
                else:
                    logger.info("File already exists", extra={"file": str(fout)})

        if files_need_downloading:
            posts_df = pw.download_entries_for_period(
                es, "dalmation-fb-posts",
                start_dt, end_dt + timedelta(hours=5),
                target_locations=target_locations,
            )
            post_ids = list(set(posts_df["post.id"].values))
            comments_df = pw.download_entries_for_period(
                es, "dalmation-fb-comments",
                start_dt, end_dt,
                post_ids=post_ids,
            )

            logger.info(
                "Downloaded entries",
                extra={"posts": len(posts_df), "comments": len(comments_df)},
            )

            df_all = pacs.get_posts_with_top_n_comments(
                posts_df, comments_df, n_comments=10, min_comments=1,
            )

            for location in target_locations:
                df_loc = df_all[df_all["tags.location"] == location]
                fout = fu.file_name_to_slug(
                    data_root / f"posts_{location}_{start_dt.strftime('%Y-%m-%d')}_to_{end_dt.strftime('%Y-%m-%d')}.feather"
                )
                fu.write_feather_to_anypath(df_loc, fout)

        start_dt = end_dt
