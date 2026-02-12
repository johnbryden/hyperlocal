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
from functools import lru_cache
from typing import Optional

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
    # Base
    open_router_key: str = Field(None, alias="OPEN_ROUTER_KEY")




# public singleton
settings = Settings()
