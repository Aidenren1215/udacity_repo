def tenor_to_months(tenor: str) -> int:
    """
    Map tenor bucket string to rollover frequency in months.

    Rules (your business definition):
    - '1W','2W','3W' -> 1  (monthly rollover)
    - '1M'..'11M' -> 1..11
    - '12M' -> 12
    - '1Y' -> 12, '2Y' -> 24, etc.
    - Unknown -> 1 (fallback)
    """
    t = str(tenor).strip().upper()

    # Weeks: treat as monthly rollover
    if re.fullmatch(r"\d+W", t):
        return 1

    # Months
    m = re.fullmatch(r"(\d+)M", t)
    if m:
        n = int(m.group(1))
        return max(n, 1)

    # Years
    y = re.fullmatch(r"(\d+)Y", t)
    if y:
        n = int(y.group(1))
        return 12 * max(n, 1)

    return 1


# quick self-tests
assert tenor_to_months("1W") == 1
assert tenor_to_months("2W") == 1
assert tenor_to_months("3W") == 1
assert tenor_to_months("1M") == 1
assert tenor_to_months("3M") == 3
assert tenor_to_months("1Y") == 12


def sort_months_mon_yy(months: List[str]) -> List[str]:
    """
    Sort month strings in 'Mon-YY' format chronologically, e.g.:
    ['Jan-24', 'Dec-23', 'Feb-24'] -> ['Dec-23', 'Jan-24', 'Feb-24'].

    Assumes months are valid English month abbreviations.
    """
    s = pd.Series([str(m).strip() for m in months], dtype="string")
    # Parse using explicit format to avoid locale ambiguity
    dt = pd.to_datetime(s, format="%b-%y", errors="raise")
    order = dt.sort_values().index
    return s.iloc[order].tolist()