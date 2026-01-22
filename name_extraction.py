import re
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF


# -----------------------------
# Utilities
# -----------------------------

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _join_words(words: List[str]) -> str:
    return _norm_ws(" ".join(words))

def _strip_prefix(name: str) -> str:
    # Remove leading honorifics you don't want
    return re.sub(r"^(Mr|Ms|Mrs|Mdm|Dr|Prof)\s+", "", name, flags=re.IGNORECASE).strip()


# -----------------------------
# Column detection
# -----------------------------

def _group_words_to_lines(words: List[Tuple]) -> List[List[Tuple[float, str]]]:
    """
    Group PyMuPDF 'words' into visual lines using (block_no, line_no),
    and keep (x0, text) pairs sorted by x0.
    """
    by_line = {}
    for x0, y0, x1, y1, w, bno, lno, wno in (words or []):
        w = _norm_ws(w)
        if not w:
            continue
        by_line.setdefault((bno, lno), []).append((x0, w))

    lines = []
    for items in by_line.values():
        items.sort(key=lambda t: t[0])
        lines.append(items)
    return lines


def _find_three_column_boundaries(
    lines: List[List[Tuple[float, str]]],
    page_width: float,
) -> Optional[Tuple[float, float]]:
    """
    Find split boundaries between 3 columns: [Full Name] [Minutes Name] [Title].

    Strategy:
    - Find the header line that contains tokens like 'Full' and 'Title'
    - Use x positions of these header keywords to infer two split points
    - Return (split1, split2) in absolute x coordinates
    """
    # Try to locate a header line
    best = None
    for items in lines:
        text = " ".join(t for _, t in items).lower()
        if ("full" in text and "title" in text) or ("full" in text and "name" in text and "title" in text):
            best = items
            break

    if not best:
        return None

    # Get approximate x positions for "Full" and "Title"
    x_full = None
    x_title = None

    for x, t in best:
        tl = t.lower()
        if x_full is None and tl in ("full", "fullname"):
            x_full = x
        if x_title is None and tl == "title":
            x_title = x

    # If keywords are not isolated (e.g. "Full Name" as two words), fall back to broader search
    if x_full is None:
        for x, t in best:
            if "full" in t.lower():
                x_full = x
                break
    if x_title is None:
        for x, t in best:
            if "title" in t.lower():
                x_title = x
                break

    if x_full is None or x_title is None:
        return None

    # Heuristic: split1 somewhere between full-name column and middle column,
    # split2 somewhere before title column.
    # We approximate split2 as slightly left of x_title.
    split2 = x_title - 10  # small margin

    # For split1, try to find the header token that corresponds to the middle column
    # (often "First", "Preferred", or "Name")
    x_mid = None
    for x, t in best:
        tl = t.lower()
        if tl in ("first", "preferred"):
            x_mid = x
            break
    if x_mid is None:
        # If we cannot find it, place split1 halfway between x_full and split2
        split1 = (x_full + split2) / 2.0
    else:
        split1 = x_mid - 10

    # Guardrails: make sure splits are within the page width
    split1 = max(0.1 * page_width, min(split1, 0.8 * page_width))
    split2 = max(split1 + 5, min(split2, 0.95 * page_width))

    return split1, split2


def _fallback_three_column_boundaries(page_width: float) -> Tuple[float, float]:
    """
    Fallback if no header is found. This is template-dependent.
    You may tune these ratios based on your minutes layout.
    """
    split1 = 0.45 * page_width
    split2 = 0.70 * page_width
    return split1, split2


# -----------------------------
# Participant extraction
# -----------------------------

def extract_participants_name_title(page: fitz.Page) -> List[Dict[str, str]]:
    """
    Extract participants from a machine-readable minutes page where the attendee list is a 3-column table:
      [Full name] [Name used in minutes] [Title]

    Output:
      [{"name": "<full name without Mr/Ms>", "title": "<title>"}]
    """
    words = page.get_text("words") or []
    lines = _group_words_to_lines(words)

    pw = float(page.rect.width)
    boundaries = _find_three_column_boundaries(lines, pw)
    if boundaries is None:
        split1, split2 = _fallback_three_column_boundaries(pw)
    else:
        split1, split2 = boundaries

    results = []
    seen = set()

    for items in lines:
        # Assign each word to one of three columns by x0
        col1, col2, col3 = [], [], []
        for x, t in items:
            if x < split1:
                col1.append(t)
            elif x < split2:
                col2.append(t)  # ignored
            else:
                col3.append(t)

        full_name_raw = _join_words(col1)
        title = _join_words(col3)

        # We only keep rows that look like a person row:
        # - full name starts with Mr/Ms/... OR contains at least 2 tokens (robust)
        if not full_name_raw:
            continue

        # Many non-person lines exist (headers, section titles). Filter aggressively.
        if not re.match(r"^(Mr|Ms|Mrs|Mdm|Dr|Prof)\b", full_name_raw, flags=re.IGNORECASE):
            # If the table sometimes drops honorific, you can loosen this condition,
            # but from your screenshots honorific is present in col1.
            continue

        name = _strip_prefix(full_name_raw)
        if not name:
            continue

        # Title must come from the third column; if empty, still keep (optional).
        # You said title was completely empty before; now it should be filled if the third column is machine-readable.
        key = (name, title)
        if key in seen:
            continue
        seen.add(key)

        results.append({"name": name, "title": title})

    return results


# -----------------------------
# Meeting header extraction (keep only what you want)
# -----------------------------

def extract_meeting_name(page_text: str) -> Optional[str]:
    """
    Extract the meeting title line like:
      'Minutes of the xxxth Asset Liability Management Committee (ALCO) Meeting'
    Works on text-layer only.
    """
    if not page_text:
        return None
    # Use DOTALL so it survives line breaks; then normalize.
    m = re.search(r"(Minutes\s+of\s+the\s+.*?\bMeeting\b.*)", page_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return _norm_ws(m.group(1))


def extract_held_on(page_text: str) -> Optional[str]:
    """
    Extract held-on text like:
      'Tuesday, 16 November 2021 at 10:00am'
    """
    if not page_text:
        return None
    m = re.search(r"held\s+on\s+(.{0,200}?)(?:\.\s|\.\n|$)", page_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return _norm_ws(m.group(1)).rstrip(".")


def extract_minutes_page_simple(page: fitz.Page) -> Dict:
    """
    Final page-level object you asked for (no meta, no meeting_number, no datetime_iso):
      {
        "meeting_name": "...",
        "held_on": "...",
        "participants": [{"name": "...", "title": "..."}]
      }
    """
    page_text = page.get_text("text") or ""
    meeting_name = extract_meeting_name(page_text)
    held_on = extract_held_on(page_text)
    participants = extract_participants_name_title(page)

    return {
        "meeting_name": meeting_name,
        "held_on": held_on,
        "participants": participants,
    }


doc = fitz.open("minutes.pdf")
page = doc.load_page(target_page_index)
obj = extract_minutes_page_simple(page)
print(obj)