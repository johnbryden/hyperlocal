import json
import os
import re
import socket
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union

try:
    from cloudpathlib import AnyPath
except ImportError:  # cloudpathlib optional; Path fallback
    AnyPath = Path

import pandas as pd

from app.simple_logger import get_logger

logger = get_logger(__name__)


def merge_tags(
    df: pd.DataFrame,
    tags_path: Union[str, "AnyPath"],
    tag_id_column: str = "tag_id",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function: open a TagManager, merge tag columns into *df*, and close.

    Args:
        df: DataFrame that has a column with tag ids.
        tags_path: Directory for the tag records (e.g. data_root / "tags" / location_slug).
        tag_id_column: Name of the column in *df* that holds the tag id.
        columns: Tag-record columns to add. Defaults to ["tag", "tag_description"].

    Returns:
        A new DataFrame with the requested tag columns merged in.
    """
    with TagManager(tags_path) as tagman:
        return tagman.merge_tags(df, tag_id_column=tag_id_column, columns=columns)


def model_to_slug(model: str) -> str:
    """Turn a model name into a filesystem-safe slug for paths and filenames."""
    s = re.sub(r"[^\w\-.]", "-", model)
    return re.sub(r"-+", "-", s).strip("-").lower() or "model"


class TagManager:
    STALE_LOCK_SECONDS = 3600  # auto-override locks older than 1 hour

    def __init__(
        self,
        tags_path: Union[str, AnyPath],
        stale_lock_timeout: Optional[float] = None,
    ):
        """
        Manage tag records for a region.

        Args:
            tags_path: Directory for tag records (incorporates region); created if missing.
            stale_lock_timeout: Seconds after which an existing lock is considered stale
                and will be forcibly overridden. Defaults to STALE_LOCK_SECONDS (3600).
        """
        self.tags_path = AnyPath(tags_path)
        self.tags_path.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.tags_path / "tag_record.csv"
        self.lock_path = self.tags_path / "tag_record.lock"

        self._lock_acquired: bool = False
        self._lock_blob = None  # GCS blob reference for cloud lock cleanup
        self._stale_lock_timeout = stale_lock_timeout if stale_lock_timeout is not None else self.STALE_LOCK_SECONDS
        self._next_id: int = 0
        self.df: pd.DataFrame = pd.DataFrame()

        self._acquire_lock()
        self._load_or_initialize()
        logger.info("TagManager initialized", extra={"path": str(self.csv_path)})

    def __enter__(self) -> "TagManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """
        Ensure lock cleanup on exit.

        Saves only if the context exits without an exception.
        Returning False propagates any exception from inside the `with` block.
        """
        try:
            if exc_type is None and self._lock_acquired:
                self.save()
        finally:
            self._release_lock()
        return False

    # ------------------------------------------------------------------
    # Locking helpers
    # ------------------------------------------------------------------

    def _is_cloud_path(self) -> bool:
        return hasattr(self.lock_path, "cloud_prefix")

    def _lock_metadata(self) -> str:
        return json.dumps({
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        })

    def _acquire_lock(self) -> None:
        """Create an exclusive lock file to avoid concurrent writers."""
        if self._is_cloud_path():
            self._acquire_lock_cloud()
        else:
            self._acquire_lock_local()

    def _acquire_lock_cloud(self) -> None:
        """Atomic lock on GCS using generation-match precondition.

        ``if_generation_match=0`` tells GCS to accept the upload **only** when
        the object does not yet exist — the cloud equivalent of ``open("x")``.
        """
        from google.cloud import storage as gcs_storage
        from google.api_core.exceptions import PreconditionFailed

        client = gcs_storage.Client()
        bucket = client.bucket(self.lock_path.bucket)
        blob = bucket.blob(self.lock_path.blob)

        try:
            blob.upload_from_string(self._lock_metadata(), if_generation_match=0)
            self._lock_acquired = True
            self._lock_blob = blob
        except PreconditionFailed:
            # Lock already exists on GCS — check staleness
            self._handle_stale_cloud_lock(blob)

    def _handle_stale_cloud_lock(self, blob) -> None:
        """If an existing cloud lock is stale, force-acquire; otherwise raise."""
        try:
            existing = json.loads(blob.download_as_text())
            age = time.time() - existing.get("timestamp", 0)
        except Exception:
            age = float("inf")
            existing = {}

        if age > self._stale_lock_timeout:
            logger.warning(
                "Overriding stale cloud lock",
                extra={"age_s": round(age), "old_meta": existing},
            )
            blob.upload_from_string(self._lock_metadata())
            self._lock_acquired = True
            self._lock_blob = blob
        else:
            raise RuntimeError(
                f"Lock file already exists: {self.lock_path} "
                f"(held by pid={existing.get('pid')} "
                f"on {existing.get('hostname')}, age={age:.0f}s)"
            )

    def _acquire_lock_local(self) -> None:
        """Local-filesystem lock using exclusive create."""
        if self.lock_path.exists():
            raise RuntimeError(f"Lock file already exists: {self.lock_path}")
        try:
            with self.lock_path.open("x") as lock_file:
                lock_file.write(self._lock_metadata())
            self._lock_acquired = True
        except FileExistsError as exc:
            raise RuntimeError(f"Lock file already exists: {self.lock_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to create lock file: {self.lock_path}") from exc

    def _release_lock(self) -> None:
        """Remove the lock file if we created it.

        Catches all exceptions so cleanup never raises (important for
        ``__exit__`` and ``__del__``).
        """
        if not self._lock_acquired:
            return
        self._lock_acquired = False
        try:
            if self._lock_blob is not None:
                self._lock_blob.delete()
                self._lock_blob = None
            else:
                self.lock_path.unlink()
        except Exception:
            pass

    def _load_or_initialize(self) -> None:
        """Load existing CSV or seed with the default row."""
        required_cols = ["id", "tag", "category", "tag_description"]
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"CSV at {self.csv_path} is missing columns: {missing}")
            df["id"] = df["id"].astype(int)
            self.df = df[required_cols].copy()
            self._next_id = int(self.df["id"].max()) + 1 if not self.df.empty else 0
        else:
            self.df = pd.DataFrame(
                [
                    {
                        "id": 0,
                        "tag": "Non-specific",
                        "category": "Non-specific",
                        "tag_description": "Not enough information",
                    }
                ],
                columns=required_cols,
            )
            self._next_id = 1

    def get_tags_for_category(self, category: str) -> pd.DataFrame:
        """Return a copy of rows matching the given category."""
        return self.df[self.df["category"] == category].copy()

    def get_id_to_tag_map(self) -> pd.Series:
        """Return a Series mapping tag id -> tag string (index=id, value=tag)."""
        return self.df.set_index("id")["tag"]

    def merge_tags(
        self,
        df: pd.DataFrame,
        tag_id_column: str = "tag_id",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add tag columns to df by merging on the tag id column.

        Args:
            df: DataFrame that has a column with tag ids (e.g. tag_id).
            tag_id_column: Name of the column in df that holds the tag id.
            columns: Tag-record columns to add (e.g. tag, tag_description, category).
                     Defaults to ["tag", "tag_description"].

        Returns:
            df with the requested tag columns added (left merge on tag_id_column).
        """
        if columns is None:
            columns = ["tag", "tag_description"]
        cols = [c for c in columns if c in self.df.columns]
        merge_df = self.df[["id"] + cols].rename(columns={"id": tag_id_column})
        return df.merge(merge_df, on=tag_id_column, how="left")

    def add_new_tag(self, tag: str, category: str, tag_description: str) -> int:
        """Append a new tag row and return its id."""
        new_id = self._next_id
        new_row = pd.DataFrame(
            [
                {
                    "id": new_id,
                    "tag": tag,
                    "category": category,
                    "tag_description": tag_description,
                }
            ]
        )
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._next_id += 1
        return new_id

    def update_tag(self, id: int, tag: str, category: str, tag_description: str) -> None:
        """Update an existing tag row by id."""
        mask = self.df["id"] == id
        if not mask.any():
            raise KeyError(f"No tag with id {id}")
        self.df.loc[mask, ["tag", "category", "tag_description"]] = [tag, category, tag_description]

    def save(self) -> None:
        """Persist the DataFrame atomically to CSV."""
        if not self._lock_acquired:
            raise RuntimeError("Lock not held; cannot save.")

        logger.debug("Saving tag record", extra={"path": str(self.csv_path), "rows": len(self.df)})
        if hasattr(self.csv_path, "cloud_prefix"):
            # CloudPath: write directly (best effort, not fully atomic on all backends)
            csv_text = self.df.to_csv(index=False)
            self.csv_path.write_text(csv_text)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, dir=self.tags_path, suffix=".tmp"
            ) as tmp_file:
                temp_path = Path(tmp_file.name)
                self.df.to_csv(tmp_file, index=False)

            try:
                temp_path.replace(self.csv_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    def __del__(self):
        """Persist changes and ensure lock cleanup."""
        try:
            if self._lock_acquired:
                try:
                    self.save()
                except Exception:
                    # Best-effort save; avoid raising in destructor
                    pass
            self._release_lock()
        except Exception:
            # Avoid destructor exceptions
            pass

