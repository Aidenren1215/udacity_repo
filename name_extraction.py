import re
import fitz
from typing import List, Dict, Tuple, Optional


# ============================================================
# Utilities
# ============================================================

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def strip_prefix(name: str) -> str:
    """Remove honorifics like Mr / Ms from the beginning of a name."""
    return re.sub(
        r"^(Mr|Ms|Mrs|Mdm|Dr|Prof)\s+",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()


# ============================================================
# Meeting name detection (STRICT, line-based)
# ============================================================

MAIN_MEETING_RE = re.compile(
    r"""
    ^Minutes\s+of\s+the\s+
    (?:\(\s*\d+(?:st|nd|rd|th)\s*\)\s+)?   # optional (xxxth)
    Asset\s+Liability\s+Management\s+Committee
    (?:\s*\(\s*ALCO\s*\))?                 # optional (ALCO)
    \s+Meeting\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

SUB_MEETING_RE = re.compile(
    r"""
    ^Minutes\s+of\s+the\s+
    (?:\(\s*[^)]*\s*\)\s+)?   # optional parentheses with ANY content
    ALCO\s+Sub\W*Committee\s+Meeting\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def extract_meeting_name(text: str) -> Optional[str]:
    """
    Return the exact meeting-name line if matched; otherwise None.
    """
    if not text:
        return None

    for line in text.splitlines():
        ln = norm_ws(line)
        if not ln:
            continue

        if MAIN_MEETING_RE.match(ln):
            return ln
        if SUB_MEETING_RE.match(ln):
            return ln

    return None


# ============================================================
# Page locator (by meeting-name keywords)
# ============================================================

def find_minutes_pages(doc: fitz.Document) -> List[int]:
    """
    Find pages that contain one of the two valid meeting-name lines.
    """
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if extract_meeting_name(text):
            pages.append(i)
    return pages


# ============================================================
# Name extraction (FIRST COLUMN ONLY)
# ============================================================

def words_to_lines(words: List[Tuple]) -> List[List[Tuple[float, str]]]:
    """
    Group PyMuPDF words into visual lines using (block_no, line_no),
    keeping (x0, text).
    """
    by_line = {}
    for x0, y0, x1, y1, w, bno, lno, wno in words:
        w = norm_ws(w)
        if not w:
            continue
        by_line.setdefault((bno, lno), []).append((x0, w))

    lines = []
    for items in by_line.values():
        items.sort(key=lambda t: t[0])
        lines.append(items)
    return lines


def detect_name_column_boundary(page: fitz.Page) -> float:
    """
    Boundary between first column (full name) and the rest.
    This is template-dependent but stable for your minutes layout.
    """
    return 0.45 * page.rect.width


def extract_names_from_page(page: fitz.Page) -> List[str]:
    """
    Extract participant names from the FIRST column only.
    - Remove honorifics
    - Support multi-word names
    """
    words = page.get_text("words") or []
    lines = words_to_lines(words)

    split_x = detect_name_column_boundary(page)

    names = []
    seen = set()

    for items in lines:
        col1_words = [t for x, t in items if x < split_x]
        if not col1_words:
            continue

        raw_name = norm_ws(" ".join(col1_words))

        # Require honorific in raw data to filter out headers / noise
        if not re.match(r"^(Mr|Ms|Mrs|Mdm|Dr|Prof)\b", raw_name, re.IGNORECASE):
            continue

        name = strip_prefix(raw_name)
        if not name or name in seen:
            continue

        seen.add(name)
        names.append(name)

    return names


# ============================================================
# Final API (THIS is what you call)
# ============================================================

def extract_minutes_names(pdf_path: str) -> List[Dict]:
    """
    End-to-end:
    - Open PDF
    - Locate meeting pages by strict meeting-name keywords
    - Extract meeting_name and participant names only
    """
    doc = fitz.open(pdf_path)
    page_indices = find_minutes_pages(doc)

    results = []

    for pno in page_indices:
        page = doc.load_page(pno)
        text = page.get_text("text") or ""

        results.append({
            "page_index": pno,
            "meeting_name": extract_meeting_name(text),
            "names": extract_names_from_page(page),
        })

    return results


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    pdf_path = "minutes.pdf"
    data = extract_minutes_names(pdf_path)
    for item in data:
        print(item)
