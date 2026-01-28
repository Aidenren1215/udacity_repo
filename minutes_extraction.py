import re
import os
import fitz  # PyMuPDF
from datetime import datetime

# --------------------------------------------------
# Patterns and constants
# --------------------------------------------------

START_RE = re.compile(
    r"Minutes\s+of\s+the\s*(?:\([^)]*\)\s*)?"
    r"Asset\s+Liability\s+Management\s+Committee\s*\(ALCO\)\s*Meeting",
    re.IGNORECASE
)

END_MARKER = "GROUP BALANCE SHEET STRATEGY"

DATE_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _normalize_text(text: str) -> str:
    # Normalize whitespace to avoid issues with line breaks and spacing
    return re.sub(r"\s+", " ", text).strip()


def _find_start_page(doc: fitz.Document) -> int:
    # Find the first page matching the ALCO minutes title
    for pno in range(doc.page_count):
        text = _normalize_text(doc.load_page(pno).get_text("text"))
        if START_RE.search(text):
            return pno
    raise ValueError(
        "Start page not found: "
        "'Minutes of the (xxxth) Asset Liability Management Committee (ALCO) Meeting' not matched."
    )


def _find_end_marker_page(doc: fitz.Document) -> int:
    # Find the first page containing the end marker
    for pno in range(doc.page_count):
        if END_MARKER in doc.load_page(pno).get_text("text"):
            return pno
    raise ValueError(
        f"End marker page not found: '{END_MARKER}' not found."
    )


def _parse_meeting_date(text: str) -> datetime | None:
    # Parse strict date format: '15 November 2021'
    m = DATE_RE.search(text)
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTHS[m.group(2).lower()]
    year = int(m.group(3))
    return datetime(year, month, day)


def _extract_meeting_date(
    doc: fitz.Document,
    start_page: int,
    max_pages_to_scan: int = 3
) -> datetime:
    # Scan a few pages starting from the start page to find the meeting date
    end = min(doc.page_count, start_page + max_pages_to_scan)
    for pno in range(start_page, end):
        text = _normalize_text(doc.load_page(pno).get_text("text"))
        dt = _parse_meeting_date(text)
        if dt:
            return dt
    raise ValueError(
        "Meeting date not found. Expected format like '15 November 2021'."
    )


# --------------------------------------------------
# Main API
# --------------------------------------------------

def extract_sg_alco_minutes_to_file(
    pdf_path: str,
    output_dir: str,
    max_pages_to_scan_for_date: int = 3,
) -> dict:
    """
    Extract SG ALCO minutes section and save as a PDF file.

    Input:
        pdf_path: path to the source PDF
        output_dir: directory to save the extracted PDF

    Output (dict):
        {
            "output_path": str,
            "filename": str,
            "start_page": int,
            "end_page": int,
            "meeting_date": "YYYY-MM-DD"
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    src = fitz.open(pdf_path)

    # 1. Locate start page
    start_page = _find_start_page(src)

    # 2. Extract meeting date
    meeting_dt = _extract_meeting_date(
        src,
        start_page=start_page,
        max_pages_to_scan=max_pages_to_scan_for_date,
    )
    filename = f"SG_ALCO minutes_{meeting_dt.strftime('%Y_%m_%d')}.pdf"
    output_path = os.path.join(output_dir, filename)

    # 3. Locate end page
    marker_page = _find_end_marker_page(src)
    end_page = marker_page - 1

    if end_page < start_page:
        src.close()
        raise ValueError(
            f"Invalid page range: start_page={start_page}, marker_page={marker_page}"
        )

    # 4. Export pages to file
    dst = fitz.open()
    dst.insert_pdf(src, from_page=start_page, to_page=end_page)
    dst.save(output_path)

    dst.close()
    src.close()

    return {
        "output_path": output_path,
        "filename": filename,
        "start_page": start_page,
        "end_page": end_page,
        "meeting_date": meeting_dt.strftime("%Y-%m-%d"),
    }


# --------------------------------------------------
# Example usage
# --------------------------------------------------
# result = extract_sg_alco_minutes_to_file(
#     pdf_path="/path/to/input.pdf",
#     output_dir="/path/to/output_folder"
# )
# print(result)
