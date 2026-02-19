import re
from typing import Iterable, Dict, Pattern, Optional, Set


# ==========================================================
# 1) Normalization (free-style robust)
# ==========================================================

def _normalize(text: str) -> str:
    """
    Normalize text for robust matching:
    - lowercase
    - remove possessive (wong's -> wong)
    - convert non-letters to spaces
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)   # wong's -> wong
    text = re.sub(r"[^a-z]", " ", text)          # non-letters -> space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compile_alternation(items: Set[str]) -> Optional[Pattern]:
    """
    Build a single regex that matches any item as a whole word/phrase.
    Sort longer first to reduce partial-match edge cases.
    """
    if not items:
        return None
    alts = sorted(items, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, alts)) + r")\b", re.I)


# ==========================================================
# 2) Build deterministic detectors from attendees list
# ==========================================================

def build_name_screening_detectors(attendees: Iterable[str]) -> Dict[str, Pattern]:
    """
    Deterministic screening rules:

    A) Always block if query matches:
       - full name (2~4 words)
       - any adjacent bigram
       - any adjacent trigram

    B) Single-token candidates (for "name token + attribution intent" rule):
       - 2-word name: BOTH tokens
       - 3-word name: FIRST + LAST only
       - 4-word name: FIRST + LAST only

    C) Always block if query matches:
       - CEO / CFO / Head of XXX
       - Mr/Ms/Mdm/Dr/Prof + <token>
    """

    phrase_set: Set[str] = set()
    token_set: Set[str] = set()

    for name in attendees:
        n = _normalize(name)
        parts = n.split()
        L = len(parts)
        if L < 2:
            continue

        # --- Phrase matches: full name + bigrams + trigrams
        phrase_set.add(" ".join(parts))

        for i in range(L - 1):
            phrase_set.add(parts[i] + " " + parts[i + 1])

        if L >= 3:
            for i in range(L - 2):
                phrase_set.add(" ".join(parts[i:i + 3]))

        # --- Your updated single-token selection rule
        if L == 2:
            token_set.add(parts[0])
            token_set.add(parts[1])
        elif L in (3, 4):
            token_set.add(parts[0])
            token_set.add(parts[-1])
        else:
            # You said max is 4; if it ever exceeds, be conservative:
            token_set.add(parts[0])
            token_set.add(parts[-1])

    phrase_pat = _compile_alternation(phrase_set)
    token_pat = _compile_alternation(token_set)

    # --- Executive titles (explicitly includes "Head of XXX")
    exec_title_pat = re.compile(
        r"\b(ceo|cfo)\b"
        r"|\bhead\s+of\s+[a-z][a-z\s&\-]{1,80}\b",
        re.I
    )

    # --- Courtesy titles + token (Ms Wong / Mr Tan / Dr Gao)
    title_any_pat = re.compile(
        r"\b(mr|ms|mrs|mdm|madam|dr|prof)\b\s+\b[a-z][a-z]{1,}\b",
        re.I
    )

    # --- Attribution / speech intent (used to gate single-token blocking)
    attr_pat = re.compile(
        r"\b("
        r"say|said|mention|mentioned|comment|commented|"
        r"reply|replied|respond|responded|"
        r"view|opinion|think|thought|feel|felt"
        r")\b",
        re.I
    )
    ask_pat = re.compile(
        r"\b(what\s+did|what\s+does|what\s+was|who\s+said)\b",
        re.I
    )

    return {
        "phrase_pat": phrase_pat,
        "token_pat": token_pat,
        "exec_title_pat": exec_title_pat,
        "title_any_pat": title_any_pat,
        "attr_pat": attr_pat,
        "ask_pat": ask_pat,
    }


# ==========================================================
# 3) Detection function
# ==========================================================

def detect_sensitive_name(query: str, det: Dict[str, Pattern]) -> Dict[str, str]:
    """
    Decision order:
      1) EXEC_TITLE (CEO/CFO/Head of XXX)
      2) TITLE_PLUS_NAME (Mr/Ms/Dr + token)
      3) NAME_PHRASE (full name/bigram/trigram)
      4) NAME_TOKEN_WITH_ATTR_INTENT (single token gated by attribution intent)
    """
    q = _normalize(query)

    if det["exec_title_pat"].search(q):
        return {"hit": True, "type": "EXEC_TITLE"}

    if det["title_any_pat"].search(q):
        return {"hit": True, "type": "TITLE_PLUS_NAME"}

    if det["phrase_pat"] and det["phrase_pat"].search(q):
        return {"hit": True, "type": "NAME_PHRASE"}

    if (
        (det["attr_pat"].search(q) or det["ask_pat"].search(q))
        and det["token_pat"]
        and det["token_pat"].search(q)
    ):
        return {"hit": True, "type": "NAME_TOKEN_WITH_ATTR_INTENT"}

    return {"hit": False, "type": "NONE"}


# ==========================================================
# 4) Quick self-test (optional)
# ==========================================================

if __name__ == "__main__":
    attendees = [
        "Helen Wong",            # 2 words -> token: helen, wong
        "Ng We Yuan",            # 3 words -> token: ng, yuan
        "Tan Kee Phong",         # 3 words -> token: tan, phong
        "Evon Lee Ee Hoon",      # 4 words -> token: evon, hoon
        "Lee Seng Mong Simon",   # 4 words -> token: lee, simon
        "Gao Wei",               # 2 words -> token: gao, wei
        "Chris Tan Chor",        # 3 words -> token: chris, chor
    ]

    det = build_name_screening_detectors(attendees)

    tests = [
        "what does helen say?",
        "what does wong say?",
        "what does we say?",                 # should not hit token set (we not included for 3-word names)
        "Ms Wong commented on the issue",
        "Head of Treasury said we should proceed",
        "We should review tomorrow",
        "Kee Phong mentioned a point",       # phrase bigram catches "kee phong" even though token set uses tan/phong
        "We Yuan raised a question",         # phrase bigram catches "we yuan" even though token set uses ng/yuan
        "tangent issue",
    ]

    for t in tests:
        print(t, "->", detect_sensitive_name(t, det))