import re
import fitz
from typing import List, Dict, Tuple, Optional


# ============================================================
# Utilities
# ============================================================

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def strip_prefix(name: str) -> str:
    return re.sub(
        r"^(Mr|Ms|Mrs|Mdm|Dr|Prof)\s+",
        "",
        name,
        flags=re.IGNORECASE
    ).strip()


# ============================================================
# Step 1: find target pages by keywords (YOU requested this)
# ============================================================

MEETING_PAGE_RE = re.compile(
    r"Minutes\s+of\s+the\s+.*?\bMeeting\b",
    re.IGNORECASE | re.DOTALL
)

def find_minutes_pages(doc: fitz.Document) -> List[int]:
    """
    Find page indices that contain meeting minutes titles like:
      'Minutes of the xxxth ... Meeting'
    """
    hit_pages = []

    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text") or ""
        if MEETING_PAGE_RE.search(text):
            hit_pages.append(i)

    return hit_pages


# ============================================================
# Step 2: extract meeting header (simple, no extras)
# ============================================================

def extract_meeting_name(text: str) -> Optional[str]:
    m = re.search(
        r"(Minutes\s+of\s+the\s+.*?\bMeeting\b.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return norm_ws(m.group(1)) if m else None

def extract_held_on(text: str) -> Optional[str]:
    m = re.search(
        r"held\s+on\s+(.{0,200}?)(?:\.\s|\.\n|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return norm_ws(m.group(1)).rstrip(".") if m else None


# ============================================================
# Step 3: extract participants (3-column logic, FIXED)
# ============================================================

def words_to_lines(words: List[Tuple]) -> List[List[Tuple[float, str]]]:
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

def detect_column_boundaries(page: fitz.Page) -> Tuple[float, float]:
    """
    Generic fallback for your minutes template.
    Adjust ratios only if your PDF layout changes.
    """
    w = page.rect.width
    return 0.45 * w, 0.72 * w

def extract_participants(page: fitz.Page) -> List[Dict[str, str]]:
    words = page.get_text("words") or []
    lines = words_to_lines(words)

    split1, split2 = detect_column_boundaries(page)

    results = []
    seen = set()

    for items in lines:
        col1, col3 = [], []

        for x, t in items:
            if x < split1:
                col1.append(t)
            elif x >= split2:
                col3.append(t)

        raw_name = norm_ws(" ".join(col1))
        title = norm_ws(" ".join(col3))

        if not raw_name:
            continue

        # Require honorific in raw data to avoid headers
        if not re.match(r"^(Mr|Ms|Mrs|Mdm|Dr|Prof)\b", raw_name, re.IGNORECASE):
            continue

        name = strip_prefix(raw_name)

        key = (name, title)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "name": name,
            "title": title,
        })

    return results


# ============================================================
# Final API: YOU call this, nothing else
# ============================================================

def extract_minutes_from_pdf(pdf_path: str) -> List[Dict]:
    """
    End-to-end:
    - Open PDF
    - Find pages by meeting-title keywords
    - Extract meeting_name, held_on, participants(name, title)
    """
    doc = fitz.open(pdf_path)
    pages = find_minutes_pages(doc)

    outputs = []

    for pno in pages:
        page = doc.load_page(pno)
        text = page.get_text("text") or ""

        outputs.append({
            "page_index": pno,
            "meeting_name": extract_meeting_name(text),
            "held_on": extract_held_on(text),
            "participants": extract_participants(page),
        })

    return outputs


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    pdf_path = "minutes.pdf"
    data = extract_minutes_from_pdf(pdf_path)
    for d in data:
        print(d)
