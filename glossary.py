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


def find_glossary_terms(text: str, glossary: dict[str, str]) -> dict[str, str]:
    """
    Scan text and return all glossary terms that appear in it (case-insensitive).

    Matching strategy: split by whitespace, strip punctuation, uppercase each token,
    then check which glossary keys are present. This reliably catches variations
    like "nim", "NIM", "Nim?", "NIM." etc.

    Args:
        text: The text to scan (typically the user's query).
        glossary: Mapping of acronym -> definition.

    Returns:
        A dict containing only the terms found in the text, with their definitions.
    """
    tokens = {
        token.strip(string.punctuation).upper()
        for token in text.split()
    }
    tokens.discard("")

    return {
        term: definition
        for term, definition in glossary.items()
        if term.upper() in tokens
    }