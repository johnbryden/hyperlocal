#!/usr/bin/env bash
set -euo pipefail

# Optional: point to a non-default env file
ENV_FILE="${ENV_FILE:-.env}"

# Optional: require a prefix under the bucket root (appended to ROOT_PATH)
SUBPATH="${1:-}"  # e.g. "some/folder" (no leading slash)

# Ensure deps
command -v gcloud >/dev/null || { echo "ERROR: gcloud not found in PATH"; exit 1; }
command -v tree   >/dev/null || { echo "ERROR: tree not found (install: sudo apt-get install -y tree)"; exit 1; }

# Get ROOT_PATH using Pydantic (loads from ENV_FILE)
ROOT_PATH="$(
python3 - <<'PY'
import os, sys
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception as e:
    print("ERROR: Missing dependency. Install: pip install pydantic-settings", file=sys.stderr)
    raise

env_file = os.environ.get("ENV_FILE", ".env")

class Settings(BaseSettings):
    ROOT_PATH: str
    model_config = SettingsConfigDict(env_file=env_file, extra="ignore")

s = Settings()
print(s.ROOT_PATH.rstrip("/") + "/")
PY
)"

# Validate ROOT_PATH format
if [[ "$ROOT_PATH" != gs://* ]]; then
  echo "ERROR: ROOT_PATH must start with gs://  (got: $ROOT_PATH)"
  exit 1
fi

# Build target
TARGET="$ROOT_PATH"
if [[ -n "$SUBPATH" ]]; then
  SUBPATH="${SUBPATH#/}"          # strip leading slash
  TARGET="${ROOT_PATH%/}/$SUBPATH/"
fi

# Extract bucket name (for stripping prefix)
# ROOT_PATH like gs://bucket/prefix/ -> bucket
bucket="$(printf '%s' "$ROOT_PATH" | sed -E 's|^gs://([^/]+)/.*$|\1|')"

echo "$TARGET"

# List recursively, convert to relative paths under the displayed target, then tree
gcloud storage ls -r "${TARGET}**" \
  | sed -E "s|^${TARGET}||" \
  | sed '/^$/d' \
  | tree --fromfile
