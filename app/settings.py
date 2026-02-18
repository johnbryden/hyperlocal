"""
settings.py — single source of truth for config

Design goals
- One import, no surprises: `from settings import settings`
- Works locally and on Cloud Run
- Simple environment variable configuration
- Avoids hard dependency on GOOGLE_CLOUD_PROJECT when ADC can supply it
- Produces full Pub/Sub topic path automatically

Cloud Run note
- You can set env vars in the service config
- For secrets, map Secret Manager versions to env vars at deploy time
"""
from __future__ import annotations

import os
from functools import cached_property, lru_cache
from typing import Optional

from cloudpathlib import AnyPath
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv

# Load .env robustly: search upward from CWD, or honor DOTENV_PATH if provided
dotenv_path = os.environ.get("DOTENV_PATH") or find_dotenv('.env', usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path)

# ----------------------------
# Project detection
# ----------------------------



# ----------------------------
# Settings object
# ----------------------------

class Settings(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Base
    open_router_key: str = Field(None, alias="OPEN_ROUTER_KEY")
    gc_project: Optional[str] = Field(None, alias="GC_PROJECT")
    root_path: str = Field("gs://categorum-test/hyperlocal", alias="ROOT_PATH")
    output_drive_root: str = Field("1am0JHqLbZkMJ87WXI_HWIR9CjueKNwmx", alias="OUTPUT_DRIVE_ROOT")
    target_locations_sheet_id: str = Field("1Bj3syECc8jX9eCGNhutLd-C3QmZzWG1qf6WOhLV0gn4", alias="TARGET_LOCATIONS_SHEET_ID")

    @cached_property
    def root(self) -> AnyPath:
        """root_path as an AnyPath (works for local and GCS paths)."""
        return AnyPath(self.root_path)

    @cached_property
    def data_root(self) -> AnyPath:
        """Convenience: root / 'data'."""
        return self.root / "data"




# public singleton
settings = Settings()
