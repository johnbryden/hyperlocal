"""
run_process.py — CLI entry-point for the hyperlocal processing pipeline.

Replicates the workflow from test_process.ipynb as a single command:

    python -m app.run_process [--location "Peckham"]

Steps
-----
1. Setup       – load settings, ensure the GCS bucket exists, resolve target locations.
2. Download    – download posts_*.feather for each location/week if not present.
3. Process     – for each posts_*.feather, generate processed_*.feather if not present
                 (PostProcessingPipeline: political filter → categorise → tag).
4. Output      – for each processed_*.feather, generate output_*.feather and the Google
                 Sheet if not present (merge tags; feather in data dir, sheet in
                 <location>/data_sheets/).
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Optional

from cloudpathlib import AnyPath

import app.file_utils as fu
import app.google_sheets as gs
import app.post_processing as pp
import app.tag_manager as tm
from app.settings import settings
from app.simple_logger import get_logger
from app.sync_downloads import sync_all_downloads

logger = get_logger(__name__)

# ── 1. Target locations ─────────────────────────────────────────────────────

def load_target_locations(override: Optional[str] = None) -> list[str]:
    """
    Decide which locations to process.

    If *override* is provided, it is used as the sole location.  Otherwise the
    canonical list is loaded from the Google Sheet whose id lives in
    ``settings.target_locations_sheet_id``.
    """
    if override:
        logger.info("Using location override", extra={"location": override})
        return [override]

    # Load the authoritative location list from the shared Google Sheet.
    logger.info("Loading target locations from Google Sheet",
                extra={"sheet_id": settings.target_locations_sheet_id})
    target_locations_df = gs.load_dataframe_from_google_sheet(
        file_id=settings.target_locations_sheet_id,
    )
    locations = target_locations_df.Locations.tolist()
    logger.info("Loaded target locations", extra={"count": len(locations), "locations": locations})
    return locations


# ── 2. Download ──────────────────────────────────────────────────────────────

def download_posts(
    data_root: AnyPath,
    target_locations: list[str],
    first_start: datetime,
    *,
    redownload: bool = False,
) -> None:
    """
    Ensure weekly post/comment feather files exist for every location.

    Delegates to ``sync_all_downloads`` which walks week-by-week from
    *first_start* up to the most recent complete week, downloading from
    Elasticsearch only for weeks whose files are missing (unless *redownload*
    is ``True``).
    """
    logger.info("Starting download sync",
                extra={"data_root": str(data_root), "first_start": str(first_start),
                       "redownload": redownload})
    sync_all_downloads(data_root, target_locations, first_start=first_start,
                       redownload=redownload)
    logger.info("Download sync complete")


# ── 3. Find & process ───────────────────────────────────────────────────────

def find_unprocessed_files(
    data_root: AnyPath,
    target_locations: list[str] | None = None,
    *,
    rerun: bool = False,
) -> list[tuple[str, str, str, AnyPath]]:
    """
    Scan *data_root* for ``posts_*.feather`` files that don't have a
    corresponding ``processed_*.feather`` yet.

    When *target_locations* is given the search is limited to those locations
    (slugified to match filenames on disk/GCS).  Otherwise every posts file is
    considered.

    If *rerun* is ``True``, every matching posts file is returned regardless of
    whether a processed counterpart already exists.

    Returns a chronologically-sorted list of
    ``(date_from, date_to, location_slug, input_path)`` tuples.
    """
    if target_locations:
        loc_slugs = [fu.file_name_to_slug(loc) for loc in target_locations]
        loc_group = "|".join(re.escape(s) for s in loc_slugs)
        posts_pattern = re.compile(
            rf"^posts_({loc_group})_(\d{{4}}-\d{{2}}-\d{{2}})_to_(\d{{4}}-\d{{2}}-\d{{2}})\.feather$"
        )
    else:
        posts_pattern = re.compile(
            r"^posts_(.+)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.feather$"
        )

    candidates: list[tuple[str, str, str, AnyPath]] = []
    for f in data_root.iterdir():
        m = posts_pattern.match(f.name)
        if not m:
            continue
        location_slug, date_from, date_to = m.groups()
        if not rerun:
            processed_name = f"processed_{location_slug}_{date_from}_to_{date_to}.feather"
            if (data_root / processed_name).exists():
                continue
        candidates.append((date_from, date_to, location_slug, f))

    candidates.sort()
    return candidates


def run_processing_pipeline(
    root_path: AnyPath,
    data_root: AnyPath,
    unprocessed: list[tuple[str, str, str, AnyPath]],
) -> None:
    """
    Run the three-stage ``PostProcessingPipeline`` on each unprocessed file.

    For every file the pipeline performs:
      Stage 1 – political/community relevance filter  (AI)
      Stage 2 – category assignment                   (AI)
      Stage 3 – tag generation via TagManager          (AI)

    The resulting dataframe is saved as ``processed_<slug>_<dates>.feather``
    alongside the original posts file.
    """
    if not unprocessed:
        logger.info("No unprocessed files found — nothing to do")
        return

    logger.info("Unprocessed files to process", extra={"count": len(unprocessed)})
    for _, _, _, path in unprocessed:
        logger.info("  %s", path.name)

    for date_from, date_to, location_slug, input_path in unprocessed:
        logger.info("Processing file", extra={"file": input_path.name})

        # Each location gets its own tag store so tags accumulate per-area.
        pipeline = pp.PostProcessingPipeline(
            categories_path=root_path / "categories_to_study.json",
            tags_path=data_root / "tags" / location_slug,
        )

        processed_df = pipeline.process(input_path, save_intermediary_files=False)

        # Write the processed feather next to the original posts file.
        output_filename = f"processed_{location_slug}_{date_from}_to_{date_to}.feather"
        fu.write_feather_to_anypath(processed_df, data_root / output_filename)
        logger.info("Saved processed file", extra={"file": output_filename,
                                                    "rows": len(processed_df)})


# ── 4. Output to Google Sheets ───────────────────────────────────────────────

def find_processed_files(
    data_root: AnyPath,
    target_locations: list[str],
) -> list[tuple[str, str, str, str, AnyPath]]:
    """
    Find all ``processed_*.feather`` files for the given locations.

    Returns a chronologically-sorted list of
    ``(date_from, date_to, location_display, location_slug, path)`` tuples.
    """
    loc_slugs = [fu.file_name_to_slug(loc) for loc in target_locations]
    loc_group = "|".join(re.escape(s) for s in loc_slugs)
    processed_pattern = re.compile(
        rf"^processed_({loc_group})_(\d{{4}}-\d{{2}}-\d{{2}})_to_(\d{{4}}-\d{{2}}-\d{{2}})\.feather$"
    )
    slug_to_location = dict(zip(loc_slugs, target_locations))
    candidates: list[tuple[str, str, str, str, AnyPath]] = []
    for f in data_root.iterdir():
        m = processed_pattern.match(f.name)
        if not m:
            continue
        location_slug, date_from, date_to = m.groups()
        location_display = slug_to_location[location_slug]
        candidates.append((date_from, date_to, location_display, location_slug, f))
    candidates.sort()
    return candidates


# Column mapping from internal dataframe names to the human-readable names
# used in the output Google Sheet.
FIELD_MAP = {
    "timestamp": "Date posted",
    "url": "Link",
    "body": "Post text",
    "comment_texts": "Comment texts",
    "summary": "Summary",
    "category": "Category",
    "tag": "Tag",
    "tag_description": "Tag description",
    "comments": "# comments",
}


def upload_results_to_sheets(
    data_root: AnyPath,
    target_locations: list[str],
    output_drive_root: str,
) -> None:
    """
    For each processed file, write output feather and/or sheet only when missing.
    Same data in both: merged tags, renamed columns. Feather as
    ``output_<slug>_<date_from>_to_<date_to>.feather`` in *data_root*; sheet in
    ``<output_drive_root>/<location>/data_sheets/``.
    """
    to_upload = find_processed_files(data_root, target_locations)
    if not to_upload:
        logger.info("No processed files found — nothing to upload")
        return

    for date_from, date_to, location_display, location_slug, path in to_upload:
        output_feather_path = data_root / f"output_{location_slug}_{date_from}_to_{date_to}.feather"
        need_feather = not output_feather_path.exists()

        sheet_title = f"output week starting {date_from}"
        out_folder = gs.get_drive_folder_by_name(
            f"{location_display}/data_sheets",
            parent_drive_folder_id=output_drive_root,
            create_if_missing=True,
        )
        folder_id = str(out_folder["id"])
        existing_sheets = gs.list_drive_folder(
            folder_id,
            mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        need_sheet = not any(f.get("name") == sheet_title for f in existing_sheets)

        if not need_feather and not need_sheet:
            logger.info("Output feather and sheet already exist, skipping",
                        extra={"location": location_display, "sheet": sheet_title})
            continue

        df = fu.read_feather_from_anypath(path)
        df = tm.merge_tags(df, data_root / "tags" / location_slug)
        df = pp.summarise_posts(df)
        df_out = df[list(FIELD_MAP.keys())].rename(columns=FIELD_MAP).sort_values(by="Tag")

        if need_feather:
            fu.write_feather_to_anypath(df_out, output_feather_path)
            logger.info("Wrote output feather", extra={"file": output_feather_path.name, "rows": len(df_out)})

        if need_sheet:
            logger.info("Uploading sheet", extra={"location": location_display, "sheet": sheet_title})
            gs.upload_dataframe_to_google_sheet(
                df_out,
                out_folder,
                spreadsheet_title=sheet_title,
                overwrite=False,
                max_column_width=500,
                max_row_height=200,
            )
            logger.info("Uploaded sheet", extra={"location": location_display, "rows": len(df_out)})


# ── CLI entry-point ──────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the hyperlocal processing pipeline end-to-end.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help=(
            "Process a single location instead of the full list from the Google "
            "Sheet. Provide the human-readable name, e.g. 'Peckham'."
        ),
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Re-process posts files even when a processed_* file already exists.",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        default=False,
        help="Re-download posts from Elasticsearch even when local files exist.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Top-level orchestrator.

    1. Parse CLI args and resolve target locations.
    2. Ensure the GCS bucket / local data directory exists.
    3. Download any missing weekly post files from Elasticsearch.
    4. Run the PostProcessingPipeline on every unprocessed file.
    5. Merge tags and upload results to Google Sheets.
    """
    args = parse_args(argv)

    # ── Setup ──────────────────────────────────────────────────────────────
    root_path = AnyPath(settings.root_path)
    data_root = root_path / "data"

    fu.maybe_mkdir(data_root)

    target_locations = load_target_locations(override=args.location)
    logger.info("Target locations", extra={"locations": target_locations})

    # ── Download ───────────────────────────────────────────────────────────
    first_start = datetime(2026, 2, 2, tzinfo=timezone.utc)
    download_posts(data_root, target_locations, first_start,
                   redownload=args.redownload)

    # ── Process ────────────────────────────────────────────────────────────
    to_process = find_unprocessed_files(
        data_root, target_locations, rerun=args.rerun,
    )
    run_processing_pipeline(root_path, data_root, to_process)

    # ── Output ─────────────────────────────────────────────────────────────
    upload_results_to_sheets(
        data_root, target_locations, settings.output_drive_root,
    )

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
