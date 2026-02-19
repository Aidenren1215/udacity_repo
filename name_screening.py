import re
from typing import Iterable, Dict, Pattern, Optional, Set


# ==========================================================
# 1. Normalization (free-style robust)
# ==========================================================

def _normalize(text: str) -> str:
    """
    Normalize text for robust matching:
    - lowercase
    - remove possessive (wong's -> wong)
    - convert non-letters to space
    - collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# 2. Compile alternation regex safely
# ==========================================================

def _compile_alternation(phrases: Set[str]) -> Optional[Pattern]:
    if not phrases:
        return None

    # sort longer first to avoid partial alternation issues
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    pattern = r"\b(" + "|".join(map(re.escape, sorted_phrases)) + r")\b"
    return re.compile(pattern, re.I)


# ==========================================================
# 3. Build deterministic detectors from attendees list
# ==========================================================

def build_name_screening_detectors(attendees: Iterable[str]) -> Dict[str, Pattern]:
    """
    Build compiled regex patterns for deterministic screening.

    Rules:
      - full name (2-4 words)
      - all bigrams
      - all trigrams
      - title + token
      - CEO / CFO / Head of XXX
    """

    phrase_set: Set[str] = set()

    for name in attendees:
        n = _normalize(name)
        parts = n.split()

        if len(parts) < 2:
            # ignore single-word names (rare edge case)
            continue

        L = len(parts)

        # ---- full name (2-4 words)
        phrase_set.add(" ".join(parts))

        # ---- bigrams
        for i in range(L - 1):
            phrase_set.add(parts[i] + " " + parts[i + 1])

        # ---- trigrams (if length >= 3)
        if L >= 3:
            for i in range(L - 2):
                phrase_set.add(" ".join(parts[i:i + 3]))

    phrase_pat = _compile_alternation(phrase_set)

    # ---- title + any token (e.g., Ms Wong, Mr Tan, Dr Gao)
    title_any_pat = re.compile(
        r"\b(mr|ms|mrs|mdm|madam|dr|prof)\b\s+\b[a-z][a-z]{1,}\b",
        re.I
    )

    # ---- executive titles
    exec_title_pat = re.compile(
        r"\b(ceo|cfo)\b"
        r"|\bhead\s+of\s+[a-z][a-z\s&\-]{1,80}\b",
        re.I
    )

    return {
        "phrase_pat": phrase_pat,
        "title_any_pat": title_any_pat,
        "exec_title_pat": exec_title_pat,
    }


# ==========================================================
# 4. Detection function
# ==========================================================

def detect_sensitive_name(query: str,
                          detectors: Dict[str, Pattern]) -> Dict[str, str]:
    """
    Return structured result for audit/logging.

    Output example:
        {"hit": True, "type": "NAME_PHRASE"}
    """

    q = _normalize(query)

    # Executive title
    if detectors["exec_title_pat"].search(q):
        return {"hit": True, "type": "EXEC_TITLE"}

    # Title + token
    if detectors["title_any_pat"].search(q):
        return {"hit": True, "type": "TITLE_PLUS_NAME"}

    # Full/bigram/trigram name phrase
    if detectors["phrase_pat"] and detectors["phrase_pat"].search(q):
        return {"hit": True, "type": "NAME_PHRASE"}

    return {"hit": False, "type": "NONE"}



if __name__ == "__main__":
    
    attendees = [
        "Helen Wong",
        "Ng We Yuan",
        "Tan Kee Phong",
        "Evon Lee Ee Hoon",
        "Lee Seng Mong Simon",
    ]

    detectors = build_name_screening_detectors(attendees)

    tests = [
        "What did Helen Wong say?",
        "Kee Phong mentioned something.",
        "Ms Wong commented.",
        "What does the CEO think?",
        "We should review tomorrow.",
        "What does wong say?"
    ]

    for t in tests:
        print(t, "->", detect_sensitive_name(t, detectors))