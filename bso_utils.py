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


import pandas as pd
import numpy as np

def build_matured_balance_matrix(
    maturity_df: pd.DataFrame,
    month_col: str = "month",
    tenor_col: str = "Tenor",
    balance_col: str = "Balance $m",
) -> pd.DataFrame:
    """
    Build a matured_balance matrix (index=month, columns=tenor).
    - Month order is chronological using sort_months_mon_yy (Mon-YY format).
    - Values are sums of Balance $m for each (month, tenor).
    - Missing (month, tenor) filled with 0.0.
    """
    df = maturity_df.copy()

    # Normalize dtypes
    df[month_col] = df[month_col].astype(str).str.strip()
    df[tenor_col] = df[tenor_col].astype(str).str.strip()
    df[balance_col] = pd.to_numeric(df[balance_col], errors="coerce").fillna(0.0)

    # Month and tenor universes
    months_raw = df[month_col].dropna().unique().tolist()
    months = sort_months_mon_yy(months_raw)

    # Keep tenor order as first appearance in the data (stable)
    tenors = df[tenor_col].dropna().unique().tolist()

    # Pivot to matrix
    matured_mat = (
        df.pivot_table(
            index=month_col,
            columns=tenor_col,
            values=balance_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=months, columns=tenors, fill_value=0.0)
        .astype(float)
    )

    return matured_mat


# ---- quick test / usage ----
# matured_mat = build_matured_balance_matrix(maturity_df)
# display(matured_mat.head())
# print(matured_mat.index[:5])
# print(matured_mat.columns[:10])


import pandas as pd

def build_inflow_outflow_matrices(
    matured_mat: pd.DataFrame,
    shift_df: pd.DataFrame,
    month_col: str = "month",
    from_col: str = "from_bucket",
    to_col: str = "to_bucket",
    amt_col: str = "amount",
    external_bucket: str = "EXTERNAL",
    strict: bool = True,
    enforce_unique_edge_per_month: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build outflow_mat and inflow_mat with the same shape as matured_mat.

    outflow_mat[m,b] = sum(amount) where month=m and from_bucket=b (b is real tenor)
    inflow_mat[m,b]  = sum(amount) where month=m and to_bucket=b  (b is real tenor)

    Assumptions (per your confirmation):
    - For any (month, from_bucket, to_bucket), shift_df contains at most ONE row.
      If enforce_unique_edge_per_month=True, we validate and raise on duplicates.
    """
    months = list(matured_mat.index.astype(str))
    tenors = list(matured_mat.columns.astype(str))

    df = shift_df.copy()
    df[month_col] = df[month_col].astype(str).str.strip()
    df[from_col] = df[from_col].astype(str).str.strip()
    df[to_col] = df[to_col].astype(str).str.strip()
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    # Keep only positive movements
    df = df[df[amt_col] > 0].copy()

    # --- uniqueness check (your data promise) ---
    if enforce_unique_edge_per_month:
        dup = df.duplicated(subset=[month_col, from_col, to_col], keep=False)
        if dup.any():
            examples = (
                df.loc[dup, [month_col, from_col, to_col, amt_col]]
                .sort_values([month_col, from_col, to_col])
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                "Found duplicate (month, from_bucket, to_bucket) rows in shift plan. "
                f"Examples: {examples}"
            )

    # --- strict validation: unknown months/tenors ---
    if strict:
        unknown_months = sorted(set(df[month_col].unique()) - set(months))
        if unknown_months:
            raise ValueError(f"Shift plan contains months not in maturity ladder: {unknown_months}")

        bad_from = sorted(set(df[df[from_col] != external_bucket][from_col].unique()) - set(tenors))
        bad_to = sorted(set(df[df[to_col] != external_bucket][to_col].unique()) - set(tenors))
        if bad_from:
            raise ValueError(f"Shift plan contains from_bucket not in tenor universe: {bad_from}")
        if bad_to:
            raise ValueError(f"Shift plan contains to_bucket not in tenor universe: {bad_to}")

    # --- build outflow matrix (real tenors only) ---
    outflow = (
        df[df[from_col] != external_bucket]
        .groupby([month_col, from_col], dropna=False)[amt_col]
        .sum()
        .unstack(from_col, fill_value=0.0)
        .reindex(index=months, columns=tenors, fill_value=0.0)
        .astype(float)
    )

    # --- build inflow matrix (real tenors only) ---
    inflow = (
        df[df[to_col] != external_bucket]
        .groupby([month_col, to_col], dropna=False)[amt_col]
        .sum()
        .unstack(to_col, fill_value=0.0)
        .reindex(index=months, columns=tenors, fill_value=0.0)
        .astype(float)
    )

    # --- strict validation: outflow must not exceed matured capacity ---
    if strict:
        violation = (outflow - matured_mat) > 1e-9
        if violation.any().any():
            bad = []
            for m in months:
                for b in tenors:
                    if violation.loc[m, b]:
                        bad.append((m, b, float(outflow.loc[m, b]), float(matured_mat.loc[m, b])))
                        if len(bad) >= 10:
                            break
                if len(bad) >= 10:
                    break
            raise ValueError(
                "Shift plan outflow exceeds matured balance for some (month, tenor). "
                f"Examples (month, tenor, outflow, matured): {bad}"
            )

    return outflow, inflow


# ---- usage ----
# outflow_mat, inflow_mat = build_inflow_outflow_matrices(matured_mat, shift_df, strict=True, enforce_unique_edge_per_month=True)
# display(outflow_mat.head())
# display(inflow_mat.head())


def compute_renewed_balance(
    matured_mat: pd.DataFrame,
    outflow_mat: pd.DataFrame,
    inflow_mat: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Compute renewed_balance matrix.

    renewed_balance[m,b] = matured_balance[m,b]
                            - outflow[m,b]
                            + inflow[m,b]

    renewed_balance represents NEWLY ISSUED / RENEWED FD in that month and tenor.
    It must be >= 0 everywhere if the shift plan is valid.
    """
    # Basic shape check
    assert matured_mat.shape == outflow_mat.shape == inflow_mat.shape

    renewed_mat = matured_mat - outflow_mat + inflow_mat

    if strict:
        # renewed balance must be non-negative
        neg = renewed_mat < -1e-9
        if neg.any().any():
            bad = []
            for m in renewed_mat.index:
                for b in renewed_mat.columns:
                    if neg.loc[m, b]:
                        bad.append(
                            (
                                m,
                                b,
                                float(matured_mat.loc[m, b]),
                                float(outflow_mat.loc[m, b]),
                                float(inflow_mat.loc[m, b]),
                                float(renewed_mat.loc[m, b]),
                            )
                        )
                        if len(bad) >= 10:
                            break
                if len(bad) >= 10:
                    break

            raise ValueError(
                "Renewed balance is negative for some (month, tenor). "
                "This indicates an invalid monthly shift plan.\n"
                "Examples (month, tenor, matured, outflow, inflow, renewed): "
                f"{bad}"
            )

    return renewed_mat


# ---- usage ----
# renewed_mat = compute_renewed_balance(matured_mat, outflow_mat, inflow_mat, strict=True)
# display(renewed_mat.head())


import pandas as pd

def compute_rollover_balance_h12(
    renewed_mat: pd.DataFrame,
    horizon_months: int = 12,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Step 5 (agreed logic): compute rollover_balance within a 12-month horizon.

    - Month horizon comes from the month index of renewed_mat (which in turn comes
      from monthly maturity data). If earliest month is Jan-24 and horizon_months=12,
      we compute/return rollover for Jan-24 .. Dec-24 only.

    - Rollover rule (REPEATED rollover):
        For each tenor bucket b with frequency freq(b) in months,
        renewed_balance[t,b] rolls into months:
            t+freq, t+2*freq, t+3*freq, ...
        as long as those target months are within the horizon window.

      Example: Jan-24 3M rolls into Apr-24, Jul-24, Oct-24 (within Jan..Dec).
      1W/2W/3W map to freq=1 (monthly rollover).
    """
    months_all = list(renewed_mat.index.astype(str))
    tenors = list(renewed_mat.columns.astype(str))

    months_h = months_all[:horizon_months]
    H = len(months_h)

    rollover_mat = pd.DataFrame(0.0, index=months_h, columns=tenors)

    # Index mapping inside horizon
    month_to_i = {m: i for i, m in enumerate(months_h)}

    for b in tenors:
        freq = tenor_to_months(b)  # uses your confirmed mapping
        if freq <= 0:
            # defensive fallback (shouldn't happen for your tenor set)
            continue

        for t in months_h:
            x = float(renewed_mat.loc[t, b])
            if x == 0.0:
                continue

            ti = month_to_i[t]
            k = 1
            while True:
                j = ti + k * freq
                if j >= H:
                    break
                rollover_mat.iat[j, rollover_mat.columns.get_loc(b)] += x
                k += 1

    return rollover_mat, months_h


def build_monthly_fd_table_h12(
    matured_mat: pd.DataFrame,
    renewed_mat: pd.DataFrame,
    rollover_mat_h12: pd.DataFrame,
    months_h: list[str],
) -> pd.DataFrame:
    """
    Build final Monthly FD Table (12-month horizon).

    Output columns:
      - month
      - tenor_bucket
      - current_balance   = matured_balance + rollover_balance
      - proposed_balance  = rollover_balance + renewed_balance

    Notes:
      - rollover_balance is DISPLAY ONLY
      - renewed_balance is the shift result on matured
      - EXTERNAL is NOT included
    """
    tenors = list(matured_mat.columns.astype(str))

    # Align all matrices to the same 12-month horizon
    matured_h = matured_mat.reindex(index=months_h, columns=tenors, fill_value=0.0).astype(float)
    renewed_h = renewed_mat.reindex(index=months_h, columns=tenors, fill_value=0.0).astype(float)
    rollover_h = rollover_mat_h12.reindex(index=months_h, columns=tenors, fill_value=0.0).astype(float)

    current_h = matured_h + rollover_h
    proposed_h = rollover_h + renewed_h

    # Long-form output
    out = pd.DataFrame(
        {
            "month": np.repeat(months_h, len(tenors)),
            "tenor_bucket": tenors * len(months_h),
            "current_balance": current_h.to_numpy().reshape(-1),
            "proposed_balance": proposed_h.to_numpy().reshape(-1),
        }
    )

    return out
