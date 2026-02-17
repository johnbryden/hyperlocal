from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from app.simple_logger import get_logger

logger = get_logger(__name__)


def download_entries_for_period(
    es: Elasticsearch,
    index: str,
    start_date: datetime,
    end_date: datetime,
    target_locations: Sequence[str] | None = None,
    post_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch posts for a given period from Elasticsearch and return a DataFrame.
    """
    if start_date.tzinfo is None or end_date.tzinfo is None:
        raise ValueError("start_date and end_date must be timezone-aware datetimes")
    if end_date <= start_date:
        raise ValueError("end_date must be after start_date")

    filters: list[dict] = [
        {
            "range": {
                "timestamp": {
                    "gte": start_date.isoformat(),
                    "lt": end_date.isoformat(),
                }
            }
        }
    ]

    if target_locations:
        filters.append({"terms": {"tags.location.keyword": list(target_locations)}})

    if post_ids:
        filters.append({"terms": {"post.id.keyword": list(post_ids)}})

    query = {"query": {"bool": {"filter": filters}}}
    hits = list(scan(client=es, index=index, query=query))

    if not hits:
        logger.info("No posts found for period", extra={"index": index, "start": start_date.isoformat(), "end": end_date.isoformat()})
        return pd.DataFrame()

    logger.info("Downloaded posts for period", extra={"index": index, "hits": len(hits)})
    return pd.json_normalize([h["_source"] for h in hits], sep=".")


def get_most_recent_week(
    reference: datetime | date | None = None,
    *,
    weeks_ago: int = 0,
    week_start: int = 0,  # 0=Mon ... 6=Sun
    tz: timezone = timezone.utc,
) -> tuple[datetime, datetime]:
    """
    Return a *complete* week window [start, end) as datetimes.

    Parameters
    ----------
    reference : datetime | date | None
        The reference point (defaults to now in *tz*).
    weeks_ago : int
        0 (default) = most recent complete week,
        1 = the week before that, 2 = two weeks back, etc.
    week_start : int
        Day the week begins: 0=Monday … 6=Sunday.  Default is Monday.
    tz : timezone
        Timezone used when *reference* is None or for the returned datetimes.
    """
    if not (0 <= week_start <= 6):
        raise ValueError("week_start must be in [0, 6] where 0=Monday")
    if weeks_ago < 0:
        raise ValueError("weeks_ago must be >= 0")

    if reference is None:
        reference_date = datetime.now(tz).date()
    elif isinstance(reference, datetime):
        reference_date = reference.astimezone(tz).date()
    else:
        reference_date = reference

    weekday = reference_date.weekday()
    days_into_current_week = (weekday - week_start) % 7
    # Start from the most recent complete week, then step back further.
    days_since_start = 7 + days_into_current_week + 7 * weeks_ago

    period_start_date = reference_date - timedelta(days=days_since_start)
    start = datetime.combine(period_start_date, time.min, tzinfo=tz)
    end = start + timedelta(days=7)
    return start, end