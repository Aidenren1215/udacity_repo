from typing import Dict, List, Optional, Tuple
import pandas as pd

def build_monthly_baseline_plan_step3(
    general_shift_plan_df: pd.DataFrame,
    capacity_mat_full: pd.DataFrame,   # index=month (Mon-YY), columns=buckets (FULL ladder, ordered)
    weights_mat_full: pd.DataFrame,    # index=month (Mon-YY), columns=buckets (FULL weights, ordered)
    horizon: int = 12,                 # NEW: planning horizon in months
    month_order: Optional[List[str]] = None,  # if provided, will be intersected with capacity_mat_full.index then truncated to horizon
    from_col: str = "from_bucket",
    to_col: str = "to_bucket",
    type_col: str = "movement_type",
    amt_col: str = "amount",
    external_bucket: str = "EXTERNAL",
    round_dp: int = 6,
    infeasible_tol: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Step 3 (baseline):
    - internal & external_out (real -> *): split using weights of from_bucket (weights_mat_full) within horizon.
    - external_in (EXTERNAL -> real): split using GLOBAL maturity weights over horizon:
        w_ext[m] ∝ sum_b cap_h[m,b]
      fallback: uniform if global maturity sum is 0.

    Capacity feasibility check:
    - ALWAYS performed on FULL maturity ladder (capacity_mat_full), regardless of horizon.
    """

    diagnostics: Dict[str, str] = {}

    # ---- validate inputs ----
    if capacity_mat_full is None or capacity_mat_full.empty:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": "capacity_mat_full is required and cannot be empty."
        }
    if weights_mat_full is None or weights_mat_full.empty:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": "weights_mat_full is required and cannot be empty."
        }
    if horizon <= 0:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": f"Invalid horizon={horizon}."
        }

    # ---- determine month list (planning horizon) ----
    full_months = list(capacity_mat_full.index)

    if month_order is None:
        months_h = full_months[:horizon]
    else:
        # Keep user's provided order, but only months that exist in full capacity index
        mo = [str(m).strip() for m in month_order]
        months_h = [m for m in mo if m in capacity_mat_full.index][:horizon]

    if len(months_h) == 0:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": "Empty horizon months after truncation."
        }

    # ---- horizon matrices for splitting ----
    cap_h = capacity_mat_full.reindex(index=months_h)
    wmat_h = weights_mat_full.reindex(index=months_h)

    # ---- GLOBAL maturity weights for external_in over horizon ----
    total_by_month = cap_h.sum(axis=1)  # index=month
    if float(total_by_month.sum()) > 0:
        w_ext = (total_by_month / float(total_by_month.sum())).astype(float)
    else:
        w_ext = _uniform_weights(months_h)

    # ---- normalize general plan ----
    g = general_shift_plan_df[[from_col, to_col, type_col, amt_col]].copy()
    g[from_col] = g[from_col].astype(str).str.strip()
    g[to_col] = g[to_col].astype(str).str.strip()
    g[type_col] = g[type_col].astype(str).str.strip()
    g[amt_col] = pd.to_numeric(g[amt_col], errors="coerce").fillna(0.0)

    # ---- external direction check (net-only) ----
    has_ext_in = ((g[from_col] == external_bucket) & (g[amt_col] > 0)).any()
    has_ext_out = ((g[to_col] == external_bucket) & (g[amt_col] > 0)).any()
    if has_ext_in and has_ext_out:
        diagnostics["GLOBAL"] = "General shift plan contains BOTH external_in and external_out edges (net-only rule violated)."
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), diagnostics

    # ---- capacity feasibility check: ALWAYS use FULL ladder ----
    real_from = g[(g[from_col] != external_bucket) & (g[amt_col] > 0)].groupby(from_col)[amt_col].sum()
    for b, out_total in real_from.items():
        b = str(b).strip()
        if b not in capacity_mat_full.columns:
            diagnostics[b] = f"Bucket '{b}' not found in capacity_mat_full columns."
            continue
        cap_total_full = float(capacity_mat_full[b].sum())
        if float(out_total) > cap_total_full + float(infeasible_tol) + 1e-9:
            diagnostics[b] = (
                f"Infeasible w.r.t FULL maturity ladder: total_outflow={float(out_total)} "
                f"> total_capacity_full={cap_total_full} (tol={infeasible_tol})."
            )

    if diagnostics:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), diagnostics

    # ---- split each edge into months (HORIZON ONLY) ----
    rows = []
    for _, r in g.iterrows():
        total = float(r[amt_col])
        if total <= 0:
            continue

        f = str(r[from_col]).strip()
        t = str(r[to_col]).strip()
        typ = str(r[type_col]).strip()

        # external_in uses global curve; others use from_bucket weights
        if f == external_bucket:
            w = w_ext
        else:
            w = wmat_h[f] if f in wmat_h.columns else _uniform_weights(months_h)

        w = w.reindex(months_h).fillna(0.0)
        if float(w.sum()) <= 0:
            w = _uniform_weights(months_h)

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

    # ---- sort by month (Mon-YY), then from/to/type ----
    if not monthly_plan_df.empty:
        monthly_plan_df["month"] = monthly_plan_df["month"].astype(str).str.strip()
        monthly_plan_df["_month_dt"] = pd.to_datetime(monthly_plan_df["month"], format="%b-%y", errors="raise")
        monthly_plan_df = (
            monthly_plan_df
            .sort_values(by=["_month_dt", from_col, to_col, type_col])
            .drop(columns="_month_dt")
            .reset_index(drop=True)
        )

    return monthly_plan_df, diagnostics


from typing import Dict, List, Optional, Tuple
import pandas as pd

def build_monthly_baseline_plan_step3(
    general_shift_plan_df: pd.DataFrame,
    capacity_mat_full: pd.DataFrame,   # index=month (Mon-YY), columns=buckets (FULL ladder, ordered)
    weights_mat_full: pd.DataFrame,    # index=month (Mon-YY), columns=buckets (FULL weights, ordered)
    horizon: int = 12,                 # planning horizon in months
    month_order: Optional[List[str]] = None,
    from_col: str = "from_bucket",
    to_col: str = "to_bucket",
    type_col: str = "movement_type",
    amt_col: str = "amount",
    external_bucket: str = "EXTERNAL",
    round_dp: int = 6,
    infeasible_tol: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Step 3 (baseline):
    - internal & external_out (real -> *): split using weights of from_bucket (weights_mat_full) within horizon.
    - external_in (EXTERNAL -> real): split using GLOBAL maturity weights over horizon:
        w_ext[m] ∝ sum_b cap_h[m,b]
      fallback: uniform if global maturity sum is 0.

    Capacity feasibility check:
    - ALWAYS performed on FULL maturity ladder (capacity_mat_full), regardless of horizon.

    >>> CHANGED <<< (Bugfix):
    - For each REAL from_bucket b, compute alpha[b] = min(1, cap_h_total[b] / out_total[b]).
    - Scale every edge with from_bucket=b by alpha[b] before splitting into months.
      This prevents horizon monthly outflow from exceeding horizon capacity, especially for >1Y buckets.
    - Also round output amount to 1 decimal for readability.
    """

    diagnostics: Dict[str, str] = {}

    # ---- validate inputs ----
    if capacity_mat_full is None or capacity_mat_full.empty:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": "capacity_mat_full is required and cannot be empty."
        }
    if weights_mat_full is None or weights_mat_full.empty:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": "weights_mat_full is required and cannot be empty."
        }
    if horizon <= 0:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": f"Invalid horizon={horizon}."
        }

    # ---- determine month list (planning horizon) ----
    full_months = list(capacity_mat_full.index)

    if month_order is None:
        months_h = full_months[:horizon]
    else:
        mo = [str(m).strip() for m in month_order]
        months_h = [m for m in mo if m in capacity_mat_full.index][:horizon]

    if len(months_h) == 0:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), {
            "GLOBAL": "Empty horizon months after truncation."
        }

    # ---- horizon matrices for splitting ----
    cap_h = capacity_mat_full.reindex(index=months_h)
    wmat_h = weights_mat_full.reindex(index=months_h)

    # ---- GLOBAL maturity weights for external_in over horizon ----
    total_by_month = cap_h.sum(axis=1)  # index=month
    if float(total_by_month.sum()) > 0:
        w_ext = (total_by_month / float(total_by_month.sum())).astype(float)
    else:
        w_ext = _uniform_weights(months_h)

    # ---- normalize general plan ----
    g = general_shift_plan_df[[from_col, to_col, type_col, amt_col]].copy()
    g[from_col] = g[from_col].astype(str).str.strip()
    g[to_col] = g[to_col].astype(str).str.strip()
    g[type_col] = g[type_col].astype(str).str.strip()
    g[amt_col] = pd.to_numeric(g[amt_col], errors="coerce").fillna(0.0)

    # ---- external direction check (net-only) ----
    has_ext_in = ((g[from_col] == external_bucket) & (g[amt_col] > 0)).any()
    has_ext_out = ((g[to_col] == external_bucket) & (g[amt_col] > 0)).any()
    if has_ext_in and has_ext_out:
        diagnostics["GLOBAL"] = "General shift plan contains BOTH external_in and external_out edges (net-only rule violated)."
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), diagnostics

    # ---- capacity feasibility check: ALWAYS use FULL ladder ----
    real_from = g[(g[from_col] != external_bucket) & (g[amt_col] > 0)].groupby(from_col)[amt_col].sum()
    for b, out_total in real_from.items():
        b = str(b).strip()
        if b not in capacity_mat_full.columns:
            diagnostics[b] = f"Bucket '{b}' not found in capacity_mat_full columns."
            continue
        cap_total_full = float(capacity_mat_full[b].sum())
        if float(out_total) > cap_total_full + float(infeasible_tol) + 1e-9:
            diagnostics[b] = (
                f"Infeasible w.r.t FULL maturity ladder: total_outflow={float(out_total)} "
                f"> total_capacity_full={cap_total_full} (tol={infeasible_tol})."
            )

    if diagnostics:
        return pd.DataFrame(columns=["month", from_col, to_col, type_col, amt_col]), diagnostics

    # ============================================================
    # >>> CHANGED <<< compute alpha[b] based on HORIZON capacity
    # ============================================================
    out_by_from = (
        g[(g[from_col] != external_bucket) & (g[amt_col] > 0)]
        .groupby(from_col)[amt_col]
        .sum()
        .to_dict()
    )

    cap_h_total = cap_h.sum(axis=0).to_dict()  # total capacity within horizon
    alpha: Dict[str, float] = {}
    leftover: Dict[str, float] = {}

    for b, out_total in out_by_from.items():
        b = str(b).strip()
        out_total = float(out_total)
        cap_total = float(cap_h_total.get(b, 0.0))

        if cap_total <= 0 and out_total > 0:
            alpha[b] = 0.0
            leftover[b] = out_total
        elif out_total <= cap_total + 1e-9:
            alpha[b] = 1.0
        else:
            alpha[b] = cap_total / out_total  # < 1
            leftover[b] = out_total - cap_total
    # ============================================================

    # ---- split each edge into months (HORIZON ONLY) ----
    rows = []
    for _, r in g.iterrows():
        total = float(r[amt_col])
        if total <= 0:
            continue

        f = str(r[from_col]).strip()
        t = str(r[to_col]).strip()
        typ = str(r[type_col]).strip()

        # ============================================================
        # >>> CHANGED <<< scale REAL from_bucket edges by alpha
        # This prevents horizon monthly outflow exceeding horizon capacity.
        # (external_in remains unchanged: no capacity constraint)
        # ============================================================
        if f != external_bucket:
            total = total * float(alpha.get(f, 1.0))
            if total <= 0:
                continue
        # ============================================================

        # external_in uses global curve; others use from_bucket weights
        if f == external_bucket:
            w = w_ext
        else:
            w = wmat_h[f] if f in wmat_h.columns else _uniform_weights(months_h)

        w = w.reindex(months_h).fillna(0.0)
        if float(w.sum()) <= 0:
            w = _uniform_weights(months_h)

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

    # ============================================================
    # >>> CHANGED <<< readability: keep 1 decimal in output amount
    # (Do it AFTER splitting; do NOT change internal math helpers.)
    # ============================================================
    if not monthly_plan_df.empty:
        monthly_plan_df[amt_col] = monthly_plan_df[amt_col].round(1)
    # ============================================================

    # ---- sort by month (Mon-YY), then from/to/type ----
    if not monthly_plan_df.empty:
        monthly_plan_df["month"] = monthly_plan_df["month"].astype(str).str.strip()
        monthly_plan_df["_month_dt"] = pd.to_datetime(monthly_plan_df["month"], format="%b-%y", errors="raise")
        monthly_plan_df = (
            monthly_plan_df
            .sort_values(by=["_month_dt", from_col, to_col, type_col])
            .drop(columns="_month_dt")
            .reset_index(drop=True)
        )

    # ============================================================
    # >>> CHANGED <<< diagnostics: report beyond-horizon leftover
    # ============================================================
    if leftover:
        diagnostics["BEYOND_HORIZON_LEFTOVER"] = (
            "Some from_buckets have outflow beyond horizon and were scaled by alpha (not executed within horizon): "
            + ", ".join(f"{k}={v:.6f}" for k, v in leftover.items())
        )
    # ============================================================

    return monthly_plan_df, diagnostics