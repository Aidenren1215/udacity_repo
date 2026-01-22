import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import fitz  # PyMuPDF


# ============================================================
# Regex definitions
# ============================================================

# Meeting title line:
# "Minutes of the 311th Asset Liability Management Committee (ALCO) Meeting"
MEETING_TITLE_RE = re.compile(
    r"^Minutes\s+of\s+the\s+.+?\bMeeting\b.*$",
    re.IGNORECASE
)

# Meeting number: 1st / 2nd / 3rd / 311th
MEETING_NUMBER_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# Held-on line
HELD_ON_RE = re.compile(r"\bheld\s+on\s+(.+)$", re.IGNORECASE)

# Person anchor (hard requirement from you)
PERSON_PREFIX_RE = re.compile(r"^(Mr|Ms)\b", re.IGNORECASE)

# Title keyword anchors (used to locate start of title)
TITLE_KEYWORDS = [
    "CEO", "CFO", "CRO", "COO",
    "Head", "Director", "VP", "SVP", "EVP",
    "Treasury", "Finance", "Audit", "Risk",
    "Corporate", "Group", "Global",
    "Services", "Bank", "Office",
    "Chair", "Chairman", "Chairperson",
]


# ============================================================
# Text utilities
# ============================================================

def normalize_spaces(s: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", (s or "")).strip()


# ============================================================
# Meeting-level extraction
# ============================================================

def extract_meeting_title(lines: List[str]) -> Optional[str]:
    """Extract the meeting title line starting with 'Minutes of the'."""
    for ln in lines[:60]:
        if MEETING_TITLE_RE.match(ln):
            return ln
    return None


def extract_meeting_number(meeting_title: Optional[str]) -> Optional[str]:
    """Extract meeting sequence number such as '311th'."""
    if not meeting_title:
        return None
    m = MEETING_NUMBER_RE.search(meeting_title)
    return m.group(0) if m else None


def classify_committee(meeting_title: str) -> Optional[str]:
    """
    Determine committee type based on meeting title content.
    """
    low = meeting_title.lower()

    if "asset liability management committee" in low:
        return "Asset Liability Management Committee (ALCO)"
    if "alco sub-committee" in low:
        return "ALCO Sub-Committee"

    return None


def extract_held_on(lines: List[str]) -> Optional[str]:
    """Extract 'held on ...' payload."""
    for ln in lines[:120]:
        if "held on" in ln.lower():
            m = HELD_ON_RE.search(ln)
            if m:
                return m.group(1).strip().rstrip(".")
            return ln.strip().rstrip(".")
    return None


def parse_datetime_iso(held_on: Optional[str]) -> Optional[str]:
    """
    Parse datetime from strings like:
      'Tuesday, 16 November 2021 at 10:00am'
    """
    if not held_on:
        return None

    m = re.search(
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        held_on,
        re.IGNORECASE,
    )
    if not m:
        return None

    date_part, hh, mm, ampm = m.groups()
    mm = mm or "00"

    try:
        d = datetime.strptime(date_part, "%d %B %Y")
    except ValueError:
        return None

    hour = int(hh)
    minute = int(mm)

    if ampm.lower() == "pm" and hour != 12:
        hour += 12
    if ampm.lower() == "am" and hour == 12:
        hour = 0

    return d.replace(hour=hour, minute=minute).isoformat()


# ============================================================
# Participant extraction (words-based, table-safe)
# ============================================================

def looks_like_title_token(tok: str) -> bool:
    """Check if a token likely belongs to a title."""
    low = tok.lower()
    return any(kw.lower() in low for kw in TITLE_KEYWORDS)


def words_to_lines(words: List[Tuple]) -> List[List[str]]:
    """
    Group PyMuPDF words into logical lines using (block_no, line_no).
    """
    by_line = {}
    for x0, y0, x1, y1, w, bno, lno, wno in words:
        by_line.setdefault((bno, lno), []).append((x0, w))

    lines = []
    for items in by_line.values():
        items.sort(key=lambda t: t[0])
        toks = [normalize_spaces(w) for _, w in items if normalize_spaces(w)]
        if toks:
            lines.append(toks)

    return lines


def split_name_title(tokens: List[str]) -> Optional[Dict[str, str]]:
    """
    Split a tokenized line into name and title using keyword-based title anchor.
    """
    if not tokens:
        return None

    if not PERSON_PREFIX_RE.match(tokens[0]):
        return None

    title_idx = None
    for i in range(1, len(tokens)):
        if looks_like_title_token(tokens[i]):
            title_idx = i
            break

    if title_idx is None:
        name = " ".join(tokens[:3]) if len(tokens) >= 3 else " ".join(tokens)
        title = " ".join(tokens[3:]) if len(tokens) > 3 else ""
    else:
        name = " ".join(tokens[:title_idx])
        title = " ".join(tokens[title_idx:])

    return {"name": name.strip(), "title": title.strip()}


def extract_participants(page: fitz.Page) -> List[Dict[str, str]]:
    """Extract participants from a page using Mr/Ms anchor."""
    words = page.get_text("words") or []
    lines = words_to_lines(words)

    people = []
    seen = set()

    for toks in lines:
        item = split_name_title(toks)
        if not item:
            continue

        key = (item["name"], item["title"])
        if key in seen:
            continue
        seen.add(key)

        people.append(item)

    return people


# ============================================================
# Main entry
# ============================================================

def extract_minutes_two_pages(pdf_path: str) -> Dict:
    """
    Extract ALCO main committee and ALCO sub-committee pages
    based on meeting title lines.
    """
    doc = fitz.open(pdf_path)
    pages_out = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        text = page.get_text("text") or ""
        lines = [normalize_spaces(x) for x in text.splitlines() if normalize_spaces(x)]

        meeting_title = extract_meeting_title(lines)
        if not meeting_title:
            continue

        committee = classify_committee(meeting_title)
        if not committee:
            continue

        meeting_number = extract_meeting_number(meeting_title)
        held_on = extract_held_on(lines)
        dt_iso = parse_datetime_iso(held_on)
        participants = extract_participants(page)

        pages_out.append({
            "meta": {
                "page_index_0based": pno,
                "page_number_1based": pno + 1,
            },
            "committee": committee,
            "meeting_name": meeting_title,
            "meeting_number": meeting_number,
            "held_on": held_on,
            "datetime_iso": dt_iso,
            "participants": participants,
        })

    return {
        "source_pdf": pdf_path,
        "pages": pages_out,
    }


if __name__ == "__main__":
    pdf_path = "meeting_minutes.pdf"  # <-- update path
    result = extract_minutes_two_pages(pdf_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
