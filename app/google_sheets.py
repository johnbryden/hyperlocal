# Google API Client Libraries for Google Sheets
from pathlib import Path
from typing import Any, Iterable
import re

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import pandas as pd
import numpy as np
import time
from datetime import date, datetime, time as time_obj
from google.auth import default as google_auth_default
from google.auth import exceptions as auth_exceptions

from app.simple_logger import get_logger

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]


class GoogleSheetsError(Exception):
    """Raised when interacting with Google Sheets or Drive fails."""


def _log_credential_identity(credentials, verbose: bool = False) -> None:
    if not verbose:
        return
    email = getattr(credentials, "service_account_email", None)
    quota_project = getattr(credentials, "quota_project_id", None)
    scopes = getattr(credentials, "scopes", None)
    logger.debug(
        "Resolved Google credentials identity",
        extra={
            "service_account_email": email,
            "quota_project_id": quota_project,
            "scopes": scopes,
        },
    )


# --- Settings and DB Connection ---
logger = get_logger(__name__)
MP_FOLDERS_PARENT_ID: str | None = None
_VALUE_UPDATE_ROW_BATCH_SIZE = 2000
_GOOGLE_SHEETS_CELL_LIMIT = 10_000_000
_GOOGLE_SHEETS_IMPORT_SIZE_LIMIT = 20 * 1024 * 1024  # Official Drive conversion limit for Sheets


# --- Helpers -----------------------------------------------------------------


def _coerce_cell_value(value):
    """Convert DataFrame cell values to JSON-serializable primitives."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, time_obj):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return value.isoformat()

    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)

    if isinstance(value, np.ndarray):
        return ", ".join(str(item) for item in value.tolist())

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _quote_sheet_title(title: str) -> str:
    """Escape a sheet title for A1 notation."""
    escaped = title.replace("'", "''")
    return f"'{escaped}'"


def _a1_range(title: str, cell_range: str) -> str:
    """Build A1 notation with a properly escaped sheet title."""
    prefix = _quote_sheet_title(title)
    return f"{prefix}!{cell_range}" if cell_range else prefix


def _column_number_to_letter(number: int) -> str:
    """Convert a 1-based column number into its A1 notation letter."""
    if number < 1:
        raise ValueError("Column number must be at least 1.")
    letters = []
    current = number
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _calculate_column_widths(display_rows: list[list[str]]) -> list[int]:
    """Estimate pixel widths for each column capped at 300."""
    if not display_rows:
        return []

    column_count = len(display_rows[0])
    widths: list[int] = []
    for column_index in range(column_count):
        max_chars = 0
        for row in display_rows:
            cell_value = row[column_index] if column_index < len(row) else ""
            lines = cell_value.splitlines() if cell_value else [""]
            longest_line = max(len(line) for line in lines) if lines else 0
            max_chars = max(max_chars, longest_line)
        estimated_pixels = int(max_chars * 7 + 16) if max_chars else 64
        widths.append(min(300, max(64, estimated_pixels)))
    return widths


def _calculate_row_heights(display_rows: list[list[str]]) -> list[int]:
    """Estimate pixel heights for each row capped at 50."""
    heights: list[int] = []
    for row in display_rows:
        max_lines = 1
        for cell in row:
            if not cell:
                continue
            line_count = len(cell.splitlines())
            max_lines = max(max_lines, line_count if line_count > 0 else 1)
        estimated_pixels = max(21, max_lines * 15)
        heights.append(min(50, estimated_pixels))
    return heights


def _build_dimension_requests(sheet_id: int, dimension: str, sizes: list[int]) -> list[dict]:
    """Group contiguous dimensions with the same size into batched requests."""
    if not sizes:
        return []

    requests: list[dict] = []
    start_index = 0
    total = len(sizes)

    while start_index < total:
        current_size = sizes[start_index]
        end_index = start_index + 1
        while end_index < total and sizes[end_index] == current_size:
            end_index += 1
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": dimension,
                        "startIndex": start_index,
                        "endIndex": end_index,
                    },
                    "properties": {"pixelSize": current_size},
                    "fields": "pixelSize",
                }
            }
        )
        start_index = end_index

    return requests


def _ensure_sheet_grid_size(
    sheets_service: Any,
    spreadsheet_id: str,
    sheet_id: int,
    *,
    min_rows: int,
    min_columns: int,
) -> None:
    """Ensure the sheet grid can fit the provided number of rows and columns."""
    if min_rows <= 0 or min_columns <= 0:
        raise ValueError("Requested grid dimensions must be positive.")

    spreadsheet_metadata = sheets_service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(sheetId,gridProperties(rowCount,columnCount)))",
    ).execute()
    sheets_metadata = spreadsheet_metadata.get("sheets", [])
    if not sheets_metadata:
        raise GoogleSheetsError(
            f"Spreadsheet '{spreadsheet_id}' contains no worksheets."
        )

    total_cells = 0
    target_sheet_props: dict[str, Any] | None = None
    for sheet in sheets_metadata:
        properties = sheet.get("properties", {})
        grid = properties.get("gridProperties", {}) or {}
        rows = grid.get("rowCount") or 0
        columns = grid.get("columnCount") or 0
        total_cells += rows * columns
        if properties.get("sheetId") == sheet_id:
            target_sheet_props = properties

    if target_sheet_props is None:
        raise GoogleSheetsError(
            f"Sheet with ID {sheet_id} not found in spreadsheet '{spreadsheet_id}'."
        )

    grid_properties = target_sheet_props.get("gridProperties", {}) or {}
    current_rows = grid_properties.get("rowCount") or 0
    current_columns = grid_properties.get("columnCount") or 0

    requested_rows = max(min_rows, 1)
    requested_columns = max(min_columns, 1)

    update_requests: list[dict[str, Any]] = []

    # Adjust columns first to free capacity before growing rows.
    if requested_columns != current_columns:
        current_sheet_cells = current_rows * current_columns
        projected_total_cells = (
            total_cells - current_sheet_cells + current_rows * requested_columns
        )
        if projected_total_cells > _GOOGLE_SHEETS_CELL_LIMIT:
            raise GoogleSheetsError(
                "Adjusting the worksheet columns would exceed the Google Sheets "
                f"workbook cell limit of {_GOOGLE_SHEETS_CELL_LIMIT:,} cells. "
                f"Current workbook uses {total_cells:,} cells; "
                f"requested column count requires {current_rows * requested_columns:,} "
                f"cells ({current_rows} rows × {requested_columns} columns). "
                "Consider reducing the dataset or splitting it across multiple sheets."
            )

        update_requests.append(
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {"columnCount": requested_columns},
                    },
                    "fields": "gridProperties.columnCount",
                }
            }
        )

        total_cells = projected_total_cells
        current_columns = requested_columns
        current_sheet_cells = current_rows * current_columns
    else:
        current_sheet_cells = current_rows * current_columns

    if requested_rows != current_rows:
        projected_total_cells = (
            total_cells - current_sheet_cells + requested_rows * current_columns
        )
        if projected_total_cells > _GOOGLE_SHEETS_CELL_LIMIT:
            raise GoogleSheetsError(
                "Uploading the DataFrame would exceed the Google Sheets workbook cell "
                f"limit of {_GOOGLE_SHEETS_CELL_LIMIT:,} cells. "
                f"Current workbook uses {total_cells:,} cells; "
                f"requested sheet requires {requested_rows * current_columns:,} cells "
                f"({requested_rows} rows × {current_columns} columns). "
                "Consider reducing the dataset or splitting it across multiple sheets."
            )

        update_requests.append(
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {"rowCount": requested_rows},
                    },
                    "fields": "gridProperties.rowCount",
                }
            }
        )

        total_cells = projected_total_cells
        current_rows = requested_rows

    if not update_requests:
        return

    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": update_requests},
    ).execute()


def _update_sheet_values_in_chunks(
    sheets_service: Any,
    spreadsheet_id: str,
    sheet_title: str,
    values: list[list[object]],
    batch_size: int = _VALUE_UPDATE_ROW_BATCH_SIZE,
    show_progress: bool = False,
    progress_description: str | None = None,
) -> None:
    """Push values to the sheet in multiple smaller requests to avoid timeouts."""
    if not values:
        return

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    column_count = len(values[0])
    if column_count == 0:
        raise ValueError("Cannot upload values without any columns.")

    end_column_letter = _column_number_to_letter(column_count)
    total_rows = len(values)
    chunk_range = range(0, total_rows, batch_size)
    total_chunks = (total_rows + batch_size - 1) // batch_size

    progress_iter: Iterable[int]
    if show_progress and tqdm is not None:
        progress_iter = tqdm(
            chunk_range,
            total=total_chunks,
            desc=progress_description or "Uploading to Google Sheets",
            unit="chunk",
        )
    else:
        if show_progress and tqdm is None:
            logger.info(
                "Optional dependency 'tqdm' is not installed; "
                "logging chunk progress instead."
            )
        progress_iter = chunk_range

    for chunk_index, start_index in enumerate(progress_iter, start=1):
        chunk = values[start_index : start_index + batch_size]
        if not chunk:
            continue
        start_row_number = start_index + 1  # Sheets use 1-based indices
        end_row_number = start_index + len(chunk)
        range_notation = _a1_range(
            sheet_title,
            f"A{start_row_number}:{end_column_letter}{end_row_number}",
        )
        sheets_service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "valueInputOption": "RAW",
                "data": [
                    {
                        "range": range_notation,
                        "majorDimension": "ROWS",
                        "values": chunk,
                    }
                ],
            },
        ).execute()
        if show_progress and tqdm is None:
            logger.info(
                "%s: chunk %s of %s",
                progress_description or "Uploading to Google Sheets",
                chunk_index,
                total_chunks,
            )


def sanitize_drive_name(name: str) -> str:
    """Collapse whitespace and replace characters that Drive rejects."""
    cleaned = name.strip()
    if not cleaned:
        return "Untitled"
    cleaned = re.sub(r"[<>:\"/\\|?*]", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .")





def _apply_sheet_formatting(
    sheets_service,
    spreadsheet_id: str,
    sheet_id: int,
    column_count: int,
    column_widths: list[int],
    row_heights: list[int],
) -> None:
    """Apply header styling, column widths, and row heights."""
    requests: list[dict] = []

    if column_count:
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": column_count,
                    },
                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                    "fields": "userEnteredFormat.textFormat.bold",
                }
            }
        )

    requests.extend(_build_dimension_requests(sheet_id, "COLUMNS", column_widths))
    requests.extend(_build_dimension_requests(sheet_id, "ROWS", row_heights))

    if not requests:
        return

    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body={"requests": requests}
    ).execute()




# --- Google Sheets API Wiring and Endpoint ---


def _resolve_credentials(scopes: list[str], *, verbose: bool = False):
    """Resolve Google credentials using Application Default Credentials (ADC).

    Ensure the runtime has ADC configured, for example:
        gcloud auth application-default login
        gcloud auth application-default set-quota-project <PROJECT_ID>
    """
    if verbose:
        logger.debug("Resolving Google credentials via Application Default Credentials.")
    try:
        credentials, project_id = google_auth_default(scopes=scopes)
        if getattr(credentials, "requires_scopes", False):
            if verbose:
                logger.debug("ADC credentials require explicit scopes; applying scopes now.")
            credentials = credentials.with_scopes(scopes)
        if verbose:
            logger.info("✅ Successfully initialized credentials via Application Default Credentials.")
            if project_id:
                logger.debug(f"ADC resolved project ID: {project_id}")
        _log_credential_identity(credentials, verbose=verbose)
        return credentials
    except auth_exceptions.DefaultCredentialsError as exc:
        if verbose:
            logger.error("Unable to locate Application Default Credentials.", exc_info=True)
        raise GoogleSheetsError(
            "Google credentials are not configured. Ensure the runtime has a default service account."
        ) from exc

def get_google_sheets_service(verbose: bool = False):
    """Initializes and returns a Google Sheets service client using
    Application Default Credentials (ADC).
    """
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    try:
        credentials = _resolve_credentials(scopes, verbose=verbose)
        service = build("sheets", "v4", credentials=credentials)
        return service

    except Exception as exc:
        if verbose:
            logger.error("A critical error occurred while initializing the Google Sheets service.", exc_info=True)
            logger.error(f"Error details: {str(exc)}")
        raise GoogleSheetsError(f"Failed to initialize Google Sheets service: {exc}") from exc


def get_google_drive_service(
    scopes: Iterable[str] | None = None,
    *,
    verbose: bool = False,
):
    """Initializes and returns a Google Drive service client using ADC."""
    default_scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    requested_scopes = list(scopes) if scopes else default_scopes

    try:
        credentials = _resolve_credentials(requested_scopes, verbose=verbose)
        service = build("drive", "v3", credentials=credentials)
        return service
    except Exception as exc:
        if verbose:
            logger.error(
                "A critical error occurred while initializing the Google Drive service.",
                exc_info=True,
            )
            logger.error(f"Error details: {str(exc)}")
        raise GoogleSheetsError(f"Failed to initialize Google Drive service: {exc}") from exc


def list_drive_folder(
    folder_id: str,
    page_size: int = 100,
    mime_types: Iterable[str] | None = None,
    *,
    verbose: bool = False,
) -> list[dict]:
    """Return metadata for items contained in the specified Drive folder.

    Set mime_types to an iterable (e.g., ["application/vnd.google-apps.folder"])
    to filter results to particular Drive item types.
    """
    if not folder_id:
        raise ValueError("A target Drive folder_id is required.")
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer.")

    service = get_google_drive_service(verbose=verbose)
    query_parts = [f"'{folder_id}' in parents", "trashed = false"]

    if mime_types:
        type_filters = [
            f"mimeType = '{mime_type}'" for mime_type in mime_types if mime_type
        ]
        if type_filters:
            query_parts.append(f"({' or '.join(type_filters)})")

    query = " and ".join(query_parts)
    normalized_page_size = min(page_size, 1000)
    items: list[dict] = []

    try:
        request = service.files().list(
            q=query,
            pageSize=normalized_page_size,
            fields="nextPageToken, files(id, name, mimeType)",
            corpora="allDrives",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )

        while request is not None:
            response = request.execute()
            items.extend(response.get("files", []))
            request = service.files().list_next(request, response)

        return items
    except HttpError as exc:
        if verbose:
            logger.error("Google API responded with an error while listing Drive folder contents.", exc_info=True)
        raise GoogleSheetsError(f"Google API error while listing folder contents: {exc}") from exc
    except Exception as exc:
        if verbose:
            logger.error("Unexpected error occurred while listing Drive folder contents.", exc_info=True)
        raise GoogleSheetsError(f"Unexpected error while listing folder contents: {exc}") from exc


def create_drive_folder(
    name: str,
    parent_folder_id: str | None = None,
    check_if_exists: bool = True,
    *,
    verbose: bool = False,
) -> dict:
    """Create a Drive folder or return an existing one if requested."""
    if not name:
        raise ValueError("A folder name is required.")

    service = get_google_drive_service(verbose=verbose)
    parents = [parent_folder_id] if parent_folder_id else None

    try:
        if check_if_exists:
            escaped_name = name.replace("'", "\\'")
            query_parts = [
                "mimeType = 'application/vnd.google-apps.folder'",
                "trashed = false",
                f"name = '{escaped_name}'",
            ]
            if parent_folder_id:
                query_parts.append(f"'{parent_folder_id}' in parents")

            query = " and ".join(query_parts)
            response = service.files().list(
                q=query,
                pageSize=1,
                fields="files(id, name, mimeType, parents)",
                corpora="allDrives",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            existing = response.get("files", [])
            if existing:
                return existing[0]

        metadata: dict[str, object] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parents:
            metadata["parents"] = parents

        created = service.files().create(
            body=metadata,
            fields="id, name, mimeType, parents",
            supportsAllDrives=True,
        ).execute()
        return created
    except HttpError as exc:
        if verbose:
            logger.error("Google API responded with an error while creating a Drive folder.", exc_info=True)
        raise GoogleSheetsError(f"Google API error while creating folder: {exc}") from exc
    except Exception as exc:
        if verbose:
            logger.error("Unexpected error occurred while creating a Drive folder.", exc_info=True)
        raise GoogleSheetsError(f"Unexpected error while creating folder: {exc}") from exc


def get_drive_folder_by_name(
    folder_name: str,
    *,
    parent_folder_id: str | None = None,
    allow_multiple: bool = False,
    prefer_latest: bool = False,
    verbose: bool = False,
) -> dict[str, object] | list[dict[str, object]]:
    """Retrieve Drive folder metadata by name, optionally scoped to a parent folder.

    Args:
        folder_name: Exact folder name to locate.
        parent_folder_id: Optional parent folder ID to scope the search.
        allow_multiple: When True, return all matching folders sorted by modified time.
        prefer_latest: When True, return the most recently modified folder if duplicates exist.

    Returns:
        A metadata dictionary for the located folder or, when allow_multiple is True,
        a list of metadata dictionaries ordered by most recent modification.

    Raises:
        ValueError: When required parameters are missing or incompatible flags are set.
        GoogleSheetsError: When no folders are found or Google APIs fail.
    """
    if not folder_name:
        raise ValueError("A folder_name is required.")
    if allow_multiple and prefer_latest:
        raise ValueError("allow_multiple and prefer_latest cannot both be True.")

    service = get_google_drive_service(
        scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"],
        verbose=verbose,
    )

    escaped_name = folder_name.replace("'", "\\'")
    query_parts = [
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
        f"name = '{escaped_name}'",
    ]
    if parent_folder_id:
        query_parts.append(f"'{parent_folder_id}' in parents")

    query = " and ".join(query_parts)
    collected: list[dict[str, object]] = []

    try:
        request = service.files().list(
            q=query,
            pageSize=100,
            fields=(
                "nextPageToken, files(id, name, mimeType, parents, modifiedTime, "
                "createdTime, webViewLink)"
            ),
            corpora="allDrives",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            orderBy="modifiedTime desc, createdTime desc",
        )

        while request is not None:
            response = request.execute()
            collected.extend(response.get("files", []))
            request = service.files().list_next(request, response)

        if not collected:
            raise GoogleSheetsError(
                f"No Drive folder named '{folder_name}' found."
                if not parent_folder_id
                else f"No Drive folder named '{folder_name}' found under '{parent_folder_id}'."
            )

        if len(collected) > 1:
            if prefer_latest:
                if verbose:
                    logger.info(
                        "Multiple folders found; returning most recently modified folder.",
                        extra={
                            "parent_folder_id": parent_folder_id,
                            "folder_name": folder_name,
                            "match_count": len(collected),
                        },
                    )
                return collected[0]
            if not allow_multiple:
                if verbose:
                    logger.error(
                        "Multiple folders found when duplicates were not permitted.",
                        extra={
                            "parent_folder_id": parent_folder_id,
                            "folder_name": folder_name,
                            "match_count": len(collected),
                        },
                    )
                raise GoogleSheetsError(
                    f"Multiple folders named '{folder_name}' were found. "
                    "Use allow_multiple=True or prefer_latest=True to disambiguate."
                )
            if verbose:
                logger.info(
                    "Returning all matching folders for duplicate folder name.",
                    extra={
                        "parent_folder_id": parent_folder_id,
                        "folder_name": folder_name,
                        "match_count": len(collected),
                    },
                )
            return collected

        return collected[0]
    except HttpError as exc:
        if verbose:
            logger.error(
                "Google API responded with an error while retrieving folder metadata.",
                exc_info=True,
            )
        raise GoogleSheetsError(
            f"Google API error while retrieving folder metadata: {exc}"
        ) from exc
    except Exception as exc:
        if verbose:
            logger.error(
                "Unexpected error occurred while retrieving folder metadata.",
                exc_info=True,
            )
        raise GoogleSheetsError(
            f"Unexpected error while retrieving folder metadata: {exc}"
        ) from exc


def get_drive_file_by_name(
    folder_id: str,
    file_name: str,
    *,
    mime_types: Iterable[str] | None = None,
    allow_multiple: bool = False,
    prefer_latest: bool = False,
    verbose: bool = False,
) -> dict[str, object] | list[dict[str, object]]:
    """Retrieve Drive file metadata by name within a specific folder.

    Args:
        folder_id: The Drive folder to search within.
        file_name: Exact file name to locate.
        mime_types: Optional iterable of Drive MIME types to filter results.
        allow_multiple: When True, return all matching files sorted by modified time.
        prefer_latest: When True, return the most recently modified file if duplicates exist.

    Returns:
        A metadata dictionary for the located file or, when allow_multiple is True,
        a list of metadata dictionaries ordered by most recent modification.

    Raises:
        ValueError: When required parameters are missing or incompatible flags are set.
        GoogleSheetsError: When no files are found or Google APIs fail.
    """
    if not folder_id:
        raise ValueError("A target Drive folder_id is required.")
    if not file_name:
        raise ValueError("A file_name is required.")
    if allow_multiple and prefer_latest:
        raise ValueError("allow_multiple and prefer_latest cannot both be True.")

    service = get_google_drive_service(
        scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"],
        verbose=verbose,
    )
    escaped_name = file_name.replace("'", "\\'")
    query_parts = [
        f"'{folder_id}' in parents",
        "trashed = false",
        f"name = '{escaped_name}'",
    ]

    if mime_types:
        type_filters = [
            f"mimeType = '{mime_type}'" for mime_type in mime_types if mime_type
        ]
        if type_filters:
            query_parts.append(f"({' or '.join(type_filters)})")

    query = " and ".join(query_parts)
    collected: list[dict[str, object]] = []

    try:
        request = service.files().list(
            q=query,
            pageSize=100,
            fields=(
                "nextPageToken, files(id, name, mimeType, parents, modifiedTime, "
                "createdTime, webViewLink)"
            ),
            corpora="allDrives",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            orderBy="modifiedTime desc, createdTime desc",
        )

        while request is not None:
            response = request.execute()
            collected.extend(response.get("files", []))
            request = service.files().list_next(request, response)

        if not collected:
            raise GoogleSheetsError(
                f"No Drive file named '{file_name}' found in folder '{folder_id}'."
            )

        if len(collected) > 1:
            if prefer_latest:
                if verbose:
                    logger.info(
                        "Multiple files found; returning most recently modified file.",
                        extra={
                            "folder_id": folder_id,
                            "file_name": file_name,
                            "match_count": len(collected),
                        },
                    )
                return collected[0]
            if not allow_multiple:
                if verbose:
                    logger.error(
                        "Multiple files found when duplicates were not permitted.",
                        extra={
                            "folder_id": folder_id,
                            "file_name": file_name,
                            "match_count": len(collected),
                        },
                    )
                raise GoogleSheetsError(
                    f"Multiple files named '{file_name}' found in folder '{folder_id}'. "
                    "Use allow_multiple=True or prefer_latest=True to disambiguate."
                )
            if verbose:
                logger.info(
                    "Returning all matching files for duplicate file name.",
                    extra={
                        "folder_id": folder_id,
                        "file_name": file_name,
                        "match_count": len(collected),
                    },
                )
            return collected

        return collected[0]
    except HttpError as exc:
        if verbose:
            logger.error(
                "Google API responded with an error while retrieving file metadata.",
                exc_info=True,
            )
        raise GoogleSheetsError(
            f"Google API error while retrieving file metadata: {exc}"
        ) from exc
    except Exception as exc:
        if verbose:
            logger.error(
                "Unexpected error occurred while retrieving file metadata.", exc_info=True
            )
        raise GoogleSheetsError(
            f"Unexpected error while retrieving file metadata: {exc}"
        ) from exc


def load_dataframe_from_google_sheet(
    *,
    folder_id: str | None = None,
    file_name: str | None = None,
    file_id: str | None = None,
    sheet_title: str | None = None,
    value_render_option: str = "UNFORMATTED_VALUE",
    date_time_render_option: str = "FORMATTED_STRING",
    allow_multiple: bool = False,
    prefer_latest: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Download a Google Sheet into a pandas DataFrame.

    Args:
        folder_id: Optional Drive folder containing the target spreadsheet.
        file_name: Optional spreadsheet name when folder_id is provided.
        file_id: Optional spreadsheet ID. Takes precedence when supplied.
        sheet_title: Optional worksheet title. Defaults to the first worksheet.
        value_render_option: Google Sheets API value render option.
        date_time_render_option: Google Sheets API date/time render option.
        allow_multiple: When True, allow multiple matches for (folder_id, file_name).
        prefer_latest: When True, pick the most recently modified file if duplicates exist.

    Returns:
        A pandas DataFrame containing the worksheet contents.

    Raises:
        ValueError: When insufficient identifiers are provided.
        GoogleSheetsError: When the Google APIs encounter an error.
    """
    if not file_id:
        if not folder_id or not file_name:
            raise ValueError(
                "Provide either file_id or both folder_id and file_name to identify the spreadsheet."
            )
        metadata = get_drive_file_by_name(
            folder_id=folder_id,
            file_name=file_name,
            allow_multiple=allow_multiple,
            prefer_latest=prefer_latest,
            mime_types=["application/vnd.google-apps.spreadsheet"],
            verbose=verbose,
        )
        if isinstance(metadata, list):
            raise GoogleSheetsError(
                "Multiple spreadsheets matched. Provide a file_id or adjust flags to disambiguate."
            )
        file_id = metadata["id"]  # type: ignore[index]

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ]
    credentials = _resolve_credentials(scopes, verbose=verbose)
    sheets_service = build("sheets", "v4", credentials=credentials)

    try:
        if not sheet_title:
            spreadsheet_metadata = sheets_service.spreadsheets().get(
                spreadsheetId=file_id,
                fields="sheets(properties(title))",
            ).execute()
            sheets = spreadsheet_metadata.get("sheets", [])
            if not sheets:
                raise GoogleSheetsError(
                    f"Spreadsheet '{file_id}' contains no worksheets."
                )
            sheet_title = sheets[0]["properties"]["title"]

        range_name = _quote_sheet_title(sheet_title)
        response = sheets_service.spreadsheets().values().get(
            spreadsheetId=file_id,
            range=range_name,
            valueRenderOption=value_render_option,
            dateTimeRenderOption=date_time_render_option,
        ).execute()
        rows = response.get("values", [])

        if not rows:
            if verbose:
                logger.info(
                    "Requested Google Sheet contained no data.",
                    extra={
                        "file_id": file_id,
                        "sheet_title": sheet_title,
                    },
                )
            return pd.DataFrame()

        max_width = max(len(row) for row in rows)
        normalized_rows = [
            [
                (row[index] if index < len(row) and row[index] != "" else None)
                for index in range(max_width)
            ]
            for row in rows
        ]

        header = [
            str(column) if column not in (None, "") else f"column_{index + 1}"
            for index, column in enumerate(normalized_rows[0])
        ]
        data_rows = [
            row for row in normalized_rows[1:] if any(cell is not None for cell in row)
        ]

        dataframe = pd.DataFrame(data_rows, columns=header)
        if verbose:
            logger.info(
                "Loaded worksheet into DataFrame.",
                extra={
                    "file_id": file_id,
                    "sheet_title": sheet_title,
                    "rows": dataframe.shape[0],
                    "columns": dataframe.shape[1],
                },
            )
        return dataframe
    except HttpError as exc:
        if verbose:
            logger.error(
                "Google API responded with an error while downloading sheet data.",
                exc_info=True,
            )
        raise GoogleSheetsError(
            f"Google API error while downloading sheet data: {exc}"
        ) from exc
    except Exception as exc:
        if verbose:
            logger.error(
                "Unexpected error occurred while downloading sheet data.", exc_info=True
            )
        raise GoogleSheetsError(
            f"Unexpected error while downloading sheet data: {exc}"
        ) from exc


def upload_dataframe_to_google_sheet(
    dataframe: pd.DataFrame,
    folder_id: str,
    spreadsheet_title: str | None = None,
    sheet_title: str | None = None,
    overwrite: bool = True,
    verbose: bool = False,
    show_progress: bool = False,
    value_batch_size: int | None = None,
) -> str:
    """Create a new Google Sheet in the specified Drive folder and upload a DataFrame.

    Args:
        dataframe: The pandas DataFrame to upload. Column names are written as headers.
        folder_id: The Drive folder that should contain the new spreadsheet.
        spreadsheet_title: Optional name for the spreadsheet. Defaults to a timestamped name.
        sheet_title: Optional title for the first worksheet. Defaults to "Sheet1".
        overwrite: Whether to overwrite an existing spreadsheet with the same title in the folder.
            If False and the spreadsheet exists, a GoogleSheetsError is raised.
        show_progress: When True, display a progress bar (tqdm if installed) while uploading rows.
        value_batch_size: Override the number of rows written per request. Larger batches go faster but
            risk timeouts; defaults to 2000 rows.

    Returns:
        The ID of the newly created spreadsheet.

    Raises:
        ValueError: When the dataframe is invalid or required parameters are missing.
        GoogleSheetsError: When Google APIs fail or credentials are unavailable.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("Cannot upload an empty DataFrame.")
    if not folder_id:
        raise ValueError("A target Drive folder_id is required.")

    spreadsheet_title = spreadsheet_title or f"DataFrame Export {time.strftime('%Y%m%d-%H%M%S')}"
    sheet_title = sheet_title or "Sheet1"
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]

    try:
        credentials = _resolve_credentials(scopes, verbose=verbose)
        sheets_service = build("sheets", "v4", credentials=credentials)
        drive_service = build("drive", "v3", credentials=credentials)

        escaped_title = spreadsheet_title.replace("'", "\\'")
        query_parts = [
            f"'{folder_id}' in parents",
            "trashed = false",
            "mimeType = 'application/vnd.google-apps.spreadsheet'",
            f"name = '{escaped_title}'",
        ]
        existing_response = drive_service.files().list(
            q=" and ".join(query_parts),
            pageSize=1,
            fields="files(id, name)",
            corpora="allDrives",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        existing_files = existing_response.get("files", [])

        if existing_files:
            existing_file = existing_files[0]
            if not overwrite:
                if verbose:
                    logger.error(
                        "Spreadsheet already exists and overwrite disabled.",
                        extra={
                            "folder_id": folder_id,
                            "spreadsheet_title": spreadsheet_title,
                            "existing_spreadsheet_id": existing_file.get("id"),
                        },
                    )
                raise GoogleSheetsError(
                    f"Spreadsheet '{spreadsheet_title}' already exists in folder '{folder_id}' "
                    "and overwrite is disabled."
                )

            drive_service.files().delete(
                fileId=existing_file["id"],
                supportsAllDrives=True,
            ).execute()
            if verbose:
                logger.info(
                    "Deleted existing spreadsheet prior to overwrite.",
                    extra={
                        "folder_id": folder_id,
                        "spreadsheet_title": spreadsheet_title,
                        "deleted_spreadsheet_id": existing_file["id"],
                    },
                )

        file_metadata = {
            "name": spreadsheet_title,
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "parents": [folder_id],
        }
        created_file = drive_service.files().create(
            body=file_metadata,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        spreadsheet_id = created_file["id"]

        if sheet_title != "Sheet1":
            rename_request = [
                {
                    "updateSheetProperties": {
                        "properties": {"sheetId": 0, "title": sheet_title},
                        "fields": "title",
                    }
                }
            ]
            sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": rename_request},
            ).execute()

        headers = [str(column) for column in dataframe.columns]
        if not headers:
            raise ValueError("DataFrame must contain at least one column.")
        values: list[list[object]] = [headers]
        display_rows: list[list[str]] = [headers]

        for row in dataframe.itertuples(index=False, name=None):
            cleaned_row = [_coerce_cell_value(cell) for cell in row]
            values.append(cleaned_row)
            display_rows.append(
                ["" if cell is None else str(cell) for cell in cleaned_row]
            )

        column_widths = _calculate_column_widths(display_rows)
        row_heights = _calculate_row_heights(display_rows)

        _ensure_sheet_grid_size(
            sheets_service=sheets_service,
            spreadsheet_id=spreadsheet_id,
            sheet_id=0,
            min_rows=len(values),
            min_columns=len(headers),
        )

        _update_sheet_values_in_chunks(
            sheets_service=sheets_service,
            spreadsheet_id=spreadsheet_id,
            sheet_title=sheet_title,
            values=values,
            show_progress=show_progress,
            progress_description=f"Uploading '{sheet_title}' data",
            batch_size=value_batch_size or _VALUE_UPDATE_ROW_BATCH_SIZE,
        )

        _apply_sheet_formatting(
            sheets_service=sheets_service,
            spreadsheet_id=spreadsheet_id,
            sheet_id=0,
            column_count=len(headers),
            column_widths=column_widths,
            row_heights=row_heights,
        )

        if verbose:
            logger.info(
                "Uploaded DataFrame to Google Sheets.",
                extra={
                    "spreadsheet_id": spreadsheet_id,
                    "rows": dataframe.shape[0],
                    "columns": dataframe.shape[1],
                },
            )
        return spreadsheet_id
    except HttpError as exc:
        if verbose:
            logger.error("Google API responded with an error while uploading data.", exc_info=True)
        raise GoogleSheetsError(f"Google API error while uploading data: {exc}") from exc
    except Exception as exc:
        if verbose:
            logger.error("Unexpected error occurred while uploading data to Google Sheets.", exc_info=True)
        raise GoogleSheetsError(f"Unexpected error while uploading data: {exc}") from exc


def upload_file_to_google_drive(
    file_path: str | Path,
    *,
    folder_id: str | None = None,
    file_name: str | None = None,
    mime_type: str | None = None,
    resumable: bool = True,
    chunk_size: int = 5 * 1024 * 1024,
    show_progress: bool = False,
    verbose: bool = False,
    convert_to_spreadsheet: bool = False,
) -> str:
    """Upload a local file to Google Drive using the resumable upload flow.

    Args:
        file_path: Local path to the file that should be uploaded.
        folder_id: Optional Drive folder that should contain the uploaded file.
        file_name: Optional destination file name; defaults to the source name.
        mime_type: Optional MIME type hint for the uploaded content.
        resumable: When True (default), use the chunked resumable upload flow.
        chunk_size: Chunk size in bytes for resumable uploads; must be a multiple of 256 KiB.
        show_progress: When True, display a tqdm progress bar (if available).
        verbose: When True, emit detailed log messages.
        convert_to_spreadsheet: When True, request Drive to convert the file into a Google Sheet.
            The source file should be CSV or other sheet-compatible content.

    Returns:
        The ID of the created Drive file.

    Raises:
        ValueError: If the path is invalid or parameters are inconsistent.
        GoogleSheetsError: If the Drive API request fails.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")

    resolved_name = sanitize_drive_name(file_name or path.name)
    total_bytes = path.stat().st_size

    if resumable:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive when resumable uploads are enabled.")
        chunk_unit = 256 * 1024
        if chunk_size % chunk_unit != 0:
            raise ValueError("chunk_size must be a multiple of 256 KiB.")

    drive_service = get_google_drive_service(verbose=verbose)

    file_metadata: dict[str, Any] = {"name": resolved_name}
    if folder_id:
        file_metadata["parents"] = [folder_id]
    if convert_to_spreadsheet:
        file_metadata["mimeType"] = "application/vnd.google-apps.spreadsheet"
        if not mime_type:
            mime_type = "text/csv"
        if total_bytes > _GOOGLE_SHEETS_IMPORT_SIZE_LIMIT:
            raise GoogleSheetsError(
                "Cannot convert file to Google Sheet: source file exceeds the Google Sheets "
                "import size limit of ~20 MB. Split the data or keep the file as a CSV instead."
            )

    if resumable:
        media = MediaFileUpload(
            filename=str(path),
            mimetype=mime_type,
            resumable=True,
            chunksize=chunk_size,
        )
    else:
        media = MediaFileUpload(
            filename=str(path),
            mimetype=mime_type,
            resumable=False,
        )

    request = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name",
        supportsAllDrives=True,
    )

    response: dict[str, Any] | None = None
    progress_bar = None

    try:
        if not resumable:
            response = request.execute()
        else:
            if show_progress and tqdm is not None:
                progress_bar = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=f"Uploading {resolved_name}",
                )
            elif show_progress and tqdm is None and verbose:
                logger.info(
                    "Optional dependency 'tqdm' is not installed; logging upload progress."
                )

            while response is None:
                status, response = request.next_chunk(num_retries=3)
                if status and progress_bar is not None:
                    progress_bar.n = int(status.resumable_progress)
                    progress_bar.refresh()
                elif status and show_progress and tqdm is None and verbose:
                    progress_percent = status.progress() * 100 if hasattr(status, "progress") else 0.0
                    logger.info("Upload progress: %.1f%%", progress_percent)

            if progress_bar is not None:
                progress_bar.n = total_bytes
                progress_bar.refresh()
        if response is None:
            raise GoogleSheetsError("Drive upload returned no response payload.")

        file_id = response.get("id")
        if not file_id:
            raise GoogleSheetsError("Drive did not return a file ID after upload.")

        if verbose:
            logger.info(
                "Uploaded file to Google Drive.",
                extra={
                    "file_id": file_id,
                    "file_name": resolved_name,
                    "folder_id": folder_id,
                    "resumable": resumable,
                    "converted_to_sheet": convert_to_spreadsheet,
                },
            )
        return file_id
    except HttpError as exc:
        if verbose:
            logger.error(
                "Google API responded with an error while uploading file to Drive.",
                exc_info=True,
            )
        if (
            convert_to_spreadsheet
            and getattr(exc, "resp", None) is not None
            and getattr(exc.resp, "status", None) == 413
        ):
            raise GoogleSheetsError(
                "Google Sheets import failed: Drive reported that the upload is too large. "
                "Google Sheets can only import CSVs up to ~20 MB and 10 million cells. "
                "Reduce the dataset size, split it across multiple sheets, or keep the file as CSV."
            ) from exc
        raise GoogleSheetsError(f"Google API error while uploading file: {exc}") from exc
    except Exception as exc:
        if verbose:
            logger.error(
                "Unexpected error occurred while uploading file to Google Drive.",
                exc_info=True,
            )
        raise GoogleSheetsError(f"Unexpected error while uploading file: {exc}") from exc
    finally:
        if progress_bar is not None:
            progress_bar.close()

