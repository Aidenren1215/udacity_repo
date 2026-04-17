import json
import string


def load_glossary(json_path: str) -> dict[str, str]:
    """
    Load a glossary from a JSON file.

    Expected JSON format:
        {
            "NIM": "Net Interest Margin",
            "LDR": "Loan to Deposit Ratio",
            ...
        }

    Args:
        json_path: Path to the glossary JSON file.

    Returns:
        Mapping of acronym -> definition.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


import string
from rapidfuzz import fuzz


def find_glossary_terms(
    text: str,
    glossary: dict[str, str],
    fuzzy: bool = True,
    fuzzy_threshold: int = 85,
    partial_threshold: int = 90,
    fuzzy_min_length: int = 4,
) -> str:
    """
    Scan text and return matched glossary terms as a newline-separated string.

    Matching has three passes:
      1. Single-word terms, exact match (case-insensitive, punctuation-stripped)
         — direction: query tokens -> glossary.
      2. Single-word terms, fuzzy match on remaining tokens to handle typos
         — direction: query tokens -> glossary, via fuzz.ratio.
      3. Multi-word terms (e.g. "OCBC HK Ltd (Group)") via partial-ratio match
         against the full query, so users can type a shorter form like "OCBC HK"
         and still hit the full term
         — direction: glossary term -> query, via fuzz.partial_ratio.

    Args:
        text: The text to scan (typically the user's query).
        glossary: Mapping of term -> definition.
        fuzzy: Whether to enable fuzzy matching for single-word terms.
        fuzzy_threshold: Similarity score (0-100) for single-word fuzzy match.
        partial_threshold: Similarity score (0-100) for multi-word partial match.
        fuzzy_min_length: Minimum character length for single-word fuzzy matching
            (both the token and the glossary term must meet this length).

    Returns:
        A newline-separated string of "TERM: definition" lines, ready to be
        injected as additional context to the LLM. Empty string if no match.
    """
    # Split glossary into single-word vs multi-word terms
    single_word = {t: d for t, d in glossary.items() if " " not in t.strip()}
    multi_word = {t: d for t, d in glossary.items() if " " in t.strip()}

    # Tokenize the query: split, strip punctuation, uppercase, dedupe
    tokens = {
        token.strip(string.punctuation).upper()
        for token in text.split()
    }
    tokens.discard("")

    # Build a case-insensitive lookup for single-word terms
    upper_single = {t.upper(): (t, d) for t, d in single_word.items()}

    found: dict[str, str] = {}
    matched_tokens: set[str] = set()

    # Pass 1: exact single-word match
    for token in tokens:
        if token in upper_single:
            original, definition = upper_single[token]
            found[original] = definition
            matched_tokens.add(token)

    # Pass 2: fuzzy single-word match on remaining tokens (typo tolerance)
    if fuzzy:
        remaining_tokens = tokens - matched_tokens
        eligible = [
            (t_up, orig, d)
            for t_up, (orig, d) in upper_single.items()
            if len(t_up) >= fuzzy_min_length
        ]
        for token in remaining_tokens:
            if len(token) < fuzzy_min_length:
                continue
            best_score = 0.0
            best_match: tuple[str, str] | None = None
            for t_up, orig, d in eligible:
                score = fuzz.ratio(token, t_up)
                if score > best_score:
                    best_score = score
                    best_match = (orig, d)
            if best_match and best_score >= fuzzy_threshold:
                orig, d = best_match
                found.setdefault(orig, d)

    # Pass 3: partial-ratio match for multi-word terms against the full query
    if multi_word:
        text_upper = text.upper()
        for term, definition in multi_word.items():
            score = fuzz.partial_ratio(term.upper(), text_upper)
            if score >= partial_threshold:
                found.setdefault(term, definition)

    return "\n".join(f"{t}: {d}" for t, d in found.items())