import re
import os
import fitz  # PyMuPDF
from datetime import datetime

# --------------------------------------------------
# Start page patterns
# --------------------------------------------------

# ALCO minutes start page
ALCO_START_RE = re.compile(
    r"Minutes\s+of\s+the\s+"
    r"(?:\d{1,4}(?:st|nd|rd|th)\s+)?"
    r"Asset\s+Liability\s+Management\s+Committee\s*\(ALCO\)\s*Meeting",
    re.IGNORECASE
)

# Sub-committee minutes start page
# "Minutes of the ... ALCO Sub-committee Meeting"  (middle part can be anything or nothing)
SUB_START_RE = re.compile(
    r"Minutes\s+of\s+the\s+.*?"
    r"ALCO\s+Sub-committee\s+Meeting",
    re.IGNORECASE
)

# --------------------------------------------------
# End marker (the page containing it is NOT included; we take previous page)
# --------------------------------------------------
END_MARKER_LINE = "GROUP BALANCE SHEET STRATEGY"

# --------------------------------------------------
# Date pattern (supports: 15 November 2021, 17 Jul 2022, 15 Nov 2021)
# --------------------------------------------------
DATE_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_start_page(doc: fitz.Document, pattern: re.Pattern, start_from: int = 0) -> int:
    for pno in range(start_from, doc.page_count):
        text = _normalize_text(doc.load_page(pno).get_text("text"))
        if pattern.search(text):
            return pno
    raise ValueError("Start page not found for the given pattern.")


def _find_end_marker_page(doc: fitz.Document, start_from: int) -> int:
    """
    Find the first page whose ANY line (after normalize) equals END_MARKER_LINE (case-insensitive).
    """
    target = END_MARKER_LINE
    for pno in range(start_from, doc.page_count):
        raw = doc.load_page(pno).get_text("text")
        for line in raw.splitlines():
            ln = _normalize_text(line)
            if ln and ln.upper() == target:
                return pno
    raise ValueError(f"End marker page not found: exact line '{END_MARKER_LINE}' not found.")


def _parse_meeting_date(text: str) -> datetime | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTHS[m.group(2).lower()]
    year = int(m.group(3))
    return datetime(year, month, day)


def _extract_meeting_date_in_range(doc: fitz.Document, start_page: int, end_page: int) -> datetime:
    """
    Scan pages [start_page..end_page] and return the first date found.
    (You said the meeting date is below the title, but not necessarily on the title page.)
    """
    for pno in range(start_page, end_page + 1):
        raw = doc.load_page(pno).get_text("text")
        # Prefer line-by-line: closer to "below is meeting time"
        for line in raw.splitlines():
            ln = _normalize_text(line)
            if not ln:
                continue
            dt = _parse_meeting_date(ln)
            if dt:
                return dt
        # Fallback: whole page text
        dt = _parse_meeting_date(_normalize_text(raw))
        if dt:
            return dt

    raise ValueError(f"Meeting date not found in pages [{start_page}..{end_page}].")


def extract_sg_alco_and_sub_minutes_to_files(pdf_path: str, output_dir: str) -> dict:
    """
    Input:
      - pdf_path: big PDF containing both minutes
      - output_dir: folder to save extracted PDFs

    Output:
      dict with paths + page ranges + meeting dates.
    """
    os.makedirs(output_dir, exist_ok=True)
    src = fitz.open(pdf_path)

    # 1) Find boundaries
    alco_start = _find_start_page(src, ALCO_START_RE, start_from=0)
    sub_start = _find_start_page(src, SUB_START_RE, start_from=alco_start + 1)

    alco_end = sub_start - 1
    if alco_end < alco_start:
        src.close()
        raise ValueError(f"Invalid ALCO range: start={alco_start}, end={alco_end}.")

    marker_page = _find_end_marker_page(src, start_from=sub_start)
    sub_end = marker_page - 1
    if sub_end < sub_start:
        src.close()
        raise ValueError(f"Invalid Sub-committee range: start={sub_start}, end={sub_end}.")

    # 2) Extract dates (scan within each minutes range)
    alco_dt = _extract_meeting_date_in_range(src, alco_start, alco_end)
    sub_dt = _extract_meeting_date_in_range(src, sub_start, sub_end)

    # 3) Build filenames
    alco_filename = f"SG_ALCO minutes_{alco_dt.strftime('%Y_%m_%d')}.pdf"
    sub_filename = f"SG_ALCO Sub-committee minutes_{sub_dt.strftime('%Y_%m_%d')}.pdf"

    alco_output_path = os.path.join(output_dir, alco_filename)
    sub_output_path = os.path.join(output_dir, sub_filename)

    # 4) Export PDFs
    dst1 = fitz.open()
    dst1.insert_pdf(src, from_page=alco_start, to_page=alco_end)
    dst1.save(alco_output_path)
    dst1.close()

    dst2 = fitz.open()
    dst2.insert_pdf(src, from_page=sub_start, to_page=sub_end)
    dst2.save(sub_output_path)
    dst2.close()

    src.close()

    return {
        "input_path": pdf_path,
        "alco": {
            "output_path": alco_output_path,
            "filename": alco_filename,
            "start_page": alco_start,
            "end_page": alco_end,
            "exported_pages": alco_end - alco_start + 1,
            "meeting_date": alco_dt.strftime("%Y-%m-%d"),
        },
        "sub_committee": {
            "output_path": sub_output_path,
            "filename": sub_filename,
            "start_page": sub_start,
            "end_page": sub_end,
            "exported_pages": sub_end - sub_start + 1,
            "meeting_date": sub_dt.strftime("%Y-%m-%d"),
        },
        "marker_page": marker_page,
    }