import pandas as pd
from typing import List, Tuple, Dict, Optional

def _uniform_weights(months: List[str]) -> pd.Series:
    n = len(months)
    return pd.Series([1.0 / n] * n, index=months, dtype=float)

def _split_total_by_weights(total: float, w: pd.Series, round_dp: int = 6) -> pd.Series:
    """
    Split 'total' across months by weights w (index=months).
    Ensures exact sum by putting residual into last month.
    """
    months = list(w.index)
    if len(months) == 1:
        return pd.Series([float(total)], index=months, dtype=float)

    # normalize w defensively (in case of tiny float drift)
    s = float(w.sum())
    if s <= 0:
        w = _uniform_weights(months)
    else:
        w = (w / s)

    x = (total * w).round(round_dp)
    # residual to last month to ensure sum equals total exactly
    resid = float(total) - float(x.iloc[:-1].sum())
    x.iloc[-1] = round(resid, round_dp)

    # drop -0.0
    x = x.where(x.abs() > 0, 0.0)
    return x

def build_monthly_baseline_plan_step3(
    general_shift_plan_df: pd.DataFrame,
    capacity_mat: pd.DataFrame,     # index=month (Mon-YY), columns=buckets
    weights_mat: pd.DataFrame,      # same index/cols as capacity_mat
    month_order: Optional[List[str]] = None,  # if None use capacity_mat.index
    from_col: str = "from_bucket",
    to_col: str = "to_bucket",
    type_col: str = "movement_type",
    amt_col: str = "amount",
    external_bucket: str = "EXTERNAL",
    round_dp: int = 6,
    fallback_for_external_in: str = "uniform",  # "uniform" only for now
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Step 3 (baseline):
    - For each general edge (from,to,type,amount), split amount across months.
    - internal & external_out (real -> *): split using weights of from_bucket.
    - external_in (EXTERNAL -> real): split using weights of to_bucket; if to_bucket has zero maturity => fallback.
    - Feasibility check for real from_buckets:
        total_outflow(from_bucket) <= sum_m capacity[from_bucket]
      If any violated, return empty df + diagnostics.

    Returns:
      monthly_plan_df: columns [month, from_bucket, to_bucket, movement_type, amount]
      diagnostics: empty if OK, else reasons keyed by bucket or "GLOBAL"
    """
    diagnostics: Dict[str, str] = {}

    # ---- month order ----
    if month_order is None:
        months = list(capacity_mat.index)
    else:
        months = [str(m).strip() for m in month_order if str(m).strip() in capacity_mat.index]
    if len(months) == 0:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {"GLOBAL": "Empty month horizon."}

    # align matrices
    cap = capacity_mat.reindex(index=months)
    wmat = weights_mat.reindex(index=months)

    # ---- normalize general plan ----
    g = general_shift_plan_df[[from_col, to_col, type_col, amt_col]].copy()
    g[from_col] = g[from_col].astype(str).str.strip()
    g[to_col] = g[to_col].astype(str).str.strip()
    g[type_col] = g[type_col].astype(str).str.strip()
    g[amt_col] = pd.to_numeric(g[amt_col], errors="coerce").fillna(0.0)

    # ---- external direction check (optional but helpful) ----
    has_ext_in = ((g[from_col] == external_bucket) & (g[amt_col] > 0)).any()
    has_ext_out = ((g[to_col] == external_bucket) & (g[amt_col] > 0)).any()
    if has_ext_in and has_ext_out:
        diagnostics["GLOBAL"] = "General shift plan contains BOTH external_in and external_out edges (net-only rule violated)."
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), diagnostics

    # ---- feasibility check for real from_buckets ----
    # total outflow per real bucket = sum of all edges where from_bucket=b (includes internal + external_out)
    real_from = g[(g[from_col] != external_bucket) & (g[amt_col] > 0)].groupby(from_col)[amt_col].sum()
    for b, out_total in real_from.items():
        if b not in cap.columns:
            diagnostics[b] = f"Bucket '{b}' not found in capacity_mat columns."
            continue
        cap_total = float(cap[b].sum())
        if float(out_total) > cap_total + 1e-9:
            diagnostics[b] = f"Infeasible within horizon: total_outflow={float(out_total)} > total_capacity={cap_total}."
    if diagnostics:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), diagnostics

    # ---- split each edge into months ----
    rows = []
    for _, r in g.iterrows():
        total = float(r[amt_col])
        if total <= 0:
            continue

        f = r[from_col]
        t = r[to_col]
        typ = r[type_col]

        # choose weight series
        if f == external_bucket:
            # external_in: use to_bucket weights, fallback if zero-maturity
            if (t in cap.columns) and (float(cap[t].sum()) > 0) and (t in wmat.columns):
                w = wmat[t]
            else:
                # fallback
                if fallback_for_external_in == "uniform":
                    w = _uniform_weights(months)
                else:
                    w = _uniform_weights(months)
        else:
            # internal or external_out: use from_bucket weights
            if f in wmat.columns:
                w = wmat[f]
            else:
                # should not happen if inputs consistent, but keep safe fallback
                w = _uniform_weights(months)

        w = w.reindex(months).fillna(0.0)
        if float(w.sum()) <= 0:
            w = _uniform_weights(months)

        split = _split_total_by_weights(total, w, round_dp=round_dp)

        for m, a in split.items():
            if abs(float(a)) < 1e-12:
                continue
            rows.append({
                "month": m,
                from_col: f,
                to_col: t,
                type_col: typ,
                amt_col: float(a),
            })

    monthly_plan_df = pd.DataFrame(rows, columns=["month", from_col, to_col, type_col, amt_col])

    # ---- final sanity: edge totals must match general plan exactly (within rounding residual already fixed) ----
    # We'll keep it as a helper check you can run yourself; not raising here.
    return monthly_plan_df, diagnostics


def check_step3_edge_totals(
    monthly_plan_df: pd.DataFrame,
    general_shift_plan_df: pd.DataFrame,
    from_col: str = "from_bucket",
    to_col: str = "to_bucket",
    type_col: str = "movement_type",
    amt_col: str = "amount",
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Returns mismatch table for (from,to,type) totals.
    Empty => OK.
    """
    g = general_shift_plan_df[[from_col, to_col, type_col, amt_col]].copy()
    g[amt_col] = pd.to_numeric(g[amt_col], errors="coerce").fillna(0.0)

    m = monthly_plan_df[[from_col, to_col, type_col, amt_col]].copy()
    m[amt_col] = pd.to_numeric(m[amt_col], errors="coerce").fillna(0.0)

    gsum = g.groupby([from_col, to_col, type_col], as_index=False)[amt_col].sum().rename(columns={amt_col: "general_total"})
    msum = m.groupby([from_col, to_col, type_col], as_index=False)[amt_col].sum().rename(columns={amt_col: "monthly_total"})

    merged = gsum.merge(msum, on=[from_col, to_col, type_col], how="outer").fillna(0.0)
    merged["diff"] = merged["monthly_total"] - merged["general_total"]
    mism = merged[merged["diff"].abs() > tol].sort_values("diff", ascending=False)
    return mism


def check_step3_capacity(
    monthly_plan_df: pd.DataFrame,
    capacity_mat: pd.DataFrame,
    from_col: str = "from_bucket",
    amt_col: str = "amount",
    external_bucket: str = "EXTERNAL",
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Check: for each (month, real from_bucket), sum(outflows) <= capacity[month, bucket]
    Returns violations table. Empty => OK.
    """
    df = monthly_plan_df.copy()
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    df = df[df[from_col] != external_bucket]

    out = df.groupby(["month", from_col], as_index=False)[amt_col].sum().rename(columns={amt_col: "monthly_outflow"})
    out["capacity"] = out.apply(lambda r: float(capacity_mat.loc[r["month"], r[from_col]]), axis=1)
    out["excess"] = out["monthly_outflow"] - out["capacity"]
    viol = out[out["excess"] > tol].sort_values(["month", "excess"], ascending=[True, False])
    return viol




monthly_plan_df, diag = build_monthly_baseline_plan_step3(
    general_shift_plan_df=general_shift_plan_df,
    capacity_mat=capacity_mat.iloc[:12],   # 只看未来12个月（你要的 horizon）
    weights_mat=weights_mat.iloc[:12],
    month_order=list(capacity_mat.index[:12]),  # 用你排序后的月序
    from_col="from_bucket",
    to_col="to_bucket",
    type_col="movement_type",
    amt_col="amount",
    external_bucket="EXTERNAL",
    round_dp=6,
    fallback_for_external_in="uniform",
)

print("diagnostics:", diag)
print(monthly_plan_df.head(30))



mism = check_step3_edge_totals(monthly_plan_df, general_shift_plan_df)
print("edge total mismatches:")
print(mism)

viol = check_step3_capacity(monthly_plan_df, capacity_mat.iloc[:12])
print("capacity violations:")
print(viol)