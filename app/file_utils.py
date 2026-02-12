from __future__ import annotations
from io import BytesIO
import pyarrow.feather as feather
from cloudpathlib import AnyPath
import pandas as pd
import re
from typing import Any

from app.simple_logger import get_logger

logger = get_logger(__name__)


def file_name_to_slug(file_name_or_path) -> str | AnyPath:
    """
    Turn a file name or AnyPath into a filesystem-safe slug for paths and filenames.

    Accepts either a str filename or an AnyPath-like object.
    Returns a slug (str) for string inputs, or an AnyPath for AnyPath-like objects.
    """
    def slugify(name: str) -> str:
        s = re.sub(r"[^\w\-.]", "-", name)
        return re.sub(r"-+", "-", s).strip("-").lower()

    if isinstance(file_name_or_path, AnyPath):
        name = file_name_or_path.name
        slug = slugify(name)
        return file_name_or_path.with_name(slug)
    else:
        file_name = str(getattr(file_name_or_path, "name", file_name_or_path))
        return slugify(file_name)

def maybe_mkdir(dirpath) -> None:
    """
    Best-effort mkdir for local paths.

    For cloud paths, directories are typically virtual so this is a no-op.
    """
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
    except (NotImplementedError, AttributeError):
        return


def write_feather_to_anypath(df: pd.DataFrame, path: AnyPath) -> None:
    """
    Write feather data to a local path or a bucket URI.
    """
    buf = BytesIO()
    feather.write_feather(df, buf)
    buf.seek(0)
    with path.open("wb") as f:
        f.write(buf.read())


def read_feather_from_anypath(path) -> pd.DataFrame:
    """
    Read feather data from a local path or a bucket URI.
    """
    with path.open("rb") as f:
        data = f.read()
    return feather.read_feather(BytesIO(data))


def read_csv_from_anypath(path: AnyPath, **kwargs: Any) -> pd.DataFrame:
    """
    Read CSV data from a local path or a bucket URI.
    """
    with path.open("r") as f:
        return pd.read_csv(f, **kwargs)


def write_csv_to_anypath(df: pd.DataFrame, path: AnyPath, *, index: bool = False, **kwargs: Any) -> None:
    """
    Write CSV data to a local path or a bucket URI.
    """
    with path.open("w") as f:
        df.to_csv(f, index=index, **kwargs)

