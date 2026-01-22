import re
import json
from datetime import datetime
from typing import List, Dict, Optional

import fitz  # PyMuPDF


# -----------------------------
# Configuration
# -----------------------------

# Committee page anchors (case-insensitive substring match)
COMMITTEE_PATTERNS = [
    {
        "pattern": r"asset\s+liability\s+management\s+committee",
        "committee": "Asset Liability Management Committee",
        "page_type": "ALCO_MAIN",
    },
    {
        "pattern": r"alco\s+sub[-\s]?committee",
        "committee": "ALCO Sub-Committee",
        "page_type": "ALCO_SUB",
    },
]

# People line anchor: only extract lines starting with Mr/Ms (as requested)
PERSON_PREFIX_RE = re.compile(r"^\s*(Mr|Ms)\b", re.IGNORECASE)

# Meeting name line: typically starts with "Minutes of the ..."
MEETING_NAME_RE = re.compile(r"^\s*Minutes\s+of\s+the\s+.+?\bMeeting\b.*$", re.IGNORECASE)

# Extract meeting number like 311th / 1st / 2nd / 3rd / 12th
MEETING_NUMBER_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# Time line: "held on Tuesday, 16 November 2021 at 10:00am."
HELD_ON_RE = re.compile(r"\bheld\s+on\s+(.+)$", re.IGNORECASE)

# Keywords to locate the start of the title segment in a person line
TITLE_KEYWORDS = [
    "CEO", "CFO", "CRO", "COO",
    "Head", "Director", "VP", "SVP", "EVP",
    "Treasury", "Finance", "Audit", "Risk",
    "Corporate", "Group", "Global",
    "Services", "Bank", "Office", "International",
    "ALM", "MRM", "MRPA", "GWB", "CFS", "BOS", "NISP",
    "Chair", "Chairman", "Chairperson",
]


# -----------------------------
# Text utilities
# -----------------------------

def normalize_spaces(s: str) -> str:
    """Normalize whitespace to single spaces and trim."""
    return re.sub(r"\s+", " ", s).strip()


def find_committee_page_type(page_text: str) -> Optional[Dict[str, str]]:
    """Detect which committee page this is based on configured patterns."""
    for rule in COMMITTEE_PATTERNS:
        if re.search(rule["pattern"], page_text, flags=re.IGNORECASE):
            return {"committee": rule["committee"], "page_type": rule["page_type"]}
    return None


def extract_meeting_name(lines: List[str]) -> Optional[str]:
    """Extract the meeting name line (Minutes of the ... Meeting)."""
    for ln in lines[:40]:
        if MEETING_NAME_RE.match(ln):
            return ln
    # Fallback: some templates break the line; take the first line containing both "Minutes" and "Meeting"
    for ln in lines[:60]:
        low = ln.lower()
        if low.startswith("minutes") and "meeting" in low:
            return ln
    return None


def extract_meeting_number(meeting_name: Optional[str]) -> Optional[str]:
    """Extract meeting number token like '311th' from meeting name."""
    if not meeting_name:
        return None
    m = MEETING_NUMBER_RE.search(meeting_name)
    return m.group(0) if m else None


def extract_held_on(lines: List[str]) -> Optional[str]:
    """Extract the 'held on ...' line payload (everything after 'held on')."""
    for ln in lines[:80]:
        if "held on" in ln.lower():
            m = HELD_ON_RE.search(ln)
            if m:
                return m.group(1).strip().rstrip(".")
            # If regex fails, still return the whole line as a fallback
            return ln.strip().rstrip(".")
    return None


def parse_datetime_iso(held_on_payload: Optional[str]) -> Optional[str]:
    """
    Parse ISO datetime from held_on payload.
    Expected patterns like:
      'Tuesday, 16 November 2021 at 10:00am'
      'Wednesday, 10 November 2021 at 3pm'
    Returns ISO string without timezone (you can attach timezone later if you want).
    """
    if not held_on_payload:
        return None

    s = held_on_payload.strip().rstrip(".")

    # Try to capture: date + time + am/pm
    # Examples:
    #   16 November 2021 at 10:00am
    #   10 November 2021 at 3pm
    m = re.search(
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        s,
        flags=re.IGNORECASE,
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
    ap = ampm.lower()

    if ap == "pm" and hour != 12:
        hour += 12
    if ap == "am" and hour == 12:
        hour = 0

    dt = d.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return dt.isoformat()


def looks_like_title_token(token: str) -> bool:
    """Check whether a token is likely part of the title (keyword-based)."""
    tok_low = token.lower()
    for kw in TITLE_KEYWORDS:
        if kw.lower() in tok_low:
            return True
    return False


def split_name_title_from_line(line: str) -> Optional[Dict[str, str]]:
    """
    Split a single person line into name and title.

    Strategy:
    - Keep only lines starting with Mr/Ms.
    - Remove leading numbering like '1.'.
    - Find the earliest token index that matches a title keyword.
      Everything before that index -> name
      Everything from that index -> title
    - If no keyword is found, fall back to:
      name = first 3 tokens (Mr/Ms + First + Last), title = rest
    """
    line = normalize_spaces(line)

    # Remove leading numbering (e.g., "10. Mr ...")
    line = re.sub(r"^\s*\d+\.\s*", "", line)

    if not PERSON_PREFIX_RE.match(line):
        return None

    tokens = line.split()
    if len(tokens) < 3:
        return None

    title_idx = None
    for i, tok in enumerate(tokens):
        if looks_like_title_token(tok):
            title_idx = i
            break

    if title_idx is None or title_idx < 3:
        # Fallback: assume name is first 3 tokens: Mr/Ms + First + Last
        name = " ".join(tokens[:3])
        title = " ".join(tokens[3:]) if len(tokens) > 3 else ""
    else:
        name = " ".join(tokens[:title_idx])
        title = " ".join(tokens[title_idx:])

    return {"name": name.strip(), "title": title.strip()}


def extract_participants(lines: List[str]) -> List[Dict[str, str]]:
    """Extract all participants from lines using Mr/Ms anchor."""
    participants = []
    seen = set()

    for ln in lines:
        item = split_name_title_from_line(ln)
        if not item:
            continue

        key = (item["name"], item["title"])
        if key in seen:
            continue
        seen.add(key)

        participants.append(item)

    return participants


# -----------------------------
# Main extraction
# -----------------------------

def extract_two_pages_as_json(pdf_path: str) -> Dict:
    """
    Extract committee pages (ALCO main + ALCO sub-committee) from a meeting minutes PDF.
    Returns a JSON-compatible dict.
    """
    doc = fitz.open(pdf_path)

    pages_out = []
    found_types = set()

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        text = page.get_text("text") or ""
        if not text.strip():
            continue

        meta = find_committee_page_type(text)
        if not meta:
            continue

        # Avoid extracting duplicates if the keywords appear multiple times in the document
        if meta["page_type"] in found_types:
            continue
        found_types.add(meta["page_type"])

        lines = [normalize_spaces(x) for x in text.splitlines()]
        lines = [x for x in lines if x]

        meeting_name = extract_meeting_name(lines)
        meeting_number = extract_meeting_number(meeting_name)
        held_on = extract_held_on(lines)
        dt_iso = parse_datetime_iso(held_on)

        participants = extract_participants(lines)

        pages_out.append({
            "meta": {
                "page_index_0based": pno,
                "page_number_1based": pno + 1,
                "page_type": meta["page_type"],
            },
            "committee": meta["committee"],
            "meeting_name": meeting_name,
            "meeting_number": meeting_number,  # e.g., "311th"
            "held_on": held_on,
            "datetime_iso": dt_iso,
            "participants": participants,
        })

        # Optional early stop if both pages have been found
        if len(found_types) == 2:
            break

    return {
        "source_pdf": pdf_path,
        "pages": sorted(pages_out, key=lambda x: x["meta"]["page_index_0based"]),
    }


if __name__ == "__main__":
    pdf_path = "meeting_minutes.pdf"  # <-- change this
    result = extract_two_pages_as_json(pdf_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
