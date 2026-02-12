from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from app.simple_logger import get_logger

logger = get_logger(__name__)


def download_posts_for_period(
    es: Elasticsearch,
    index: str,
    start_date: datetime,
    end_date: datetime,
    target_locations: Sequence[str] | None = None,
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
    week_start: int = 2,  # 0=Mon ... 2=Wed
    tz: timezone = timezone.utc,
) -> tuple[datetime, datetime]:
    """
    Return the most recent *complete* week window [start, end) as datetimes in UTC.

    By default, the week is Wednesday->Tuesday, so this returns the window starting on the
    most recent Wednesday strictly before "today" (UTC), with end = start + 7 days.
    """
    if not (0 <= week_start <= 6):
        raise ValueError("week_start must be in [0, 6] where 0=Monday")

    if reference is None:
        reference_date = datetime.now(tz).date()
    elif isinstance(reference, datetime):
        reference_date = reference.astimezone(tz).date()
    else:
        reference_date = reference

    weekday = reference_date.weekday()
    days_into_current_week = (weekday - week_start) % 7
    # Always use the *previous* complete week (never the current in-progress one).
    days_since_start = 7 + days_into_current_week

    period_start_date = reference_date - timedelta(days=days_since_start)
    start = datetime.combine(period_start_date, time.min, tzinfo=tz)
    end = start + timedelta(days=7)
    return start, end