### Step B: Allocate monthly outflow for each real from_bucket (PROPORTIONAL FIRST-PASS ONLY; NO RESIDUAL REPAIR)

You MUST NOT use greedy earliest-first.

Inputs available:
- From maturity ladder: `cap[m,b] = capacity[month=m, bucket=b]`
- From general shift plan: `required_total_outflow[b]` (total outflow from bucket `b` across all outgoing edges)

For each real bucket `b`:

1) Compute total capacity:
- `cap_total[b] = sum_over_months cap[m,b]`

If `cap_total[b] == 0`:
- If `required_total_outflow[b] == 0`: set `monthly_outflow_total[m,b] = 0` for all months.
- Else: infeasible. Still output JSON. Set `global_validation.status="MISMATCH"` and explain in `summary_markdown`.

2) Compute proportional weights (DO NOT optimize; DO NOT iterate):
- `w[m,b] = cap[m,b] / cap_total[b]`
- `sum_over_months w[m,b] == 1`

3) First-pass proportional allocation (one pass only):
- `alloc_raw[m,b] = required_total_outflow[b] * w[m,b]`

4) Capacity clip (one pass only; NO residual redistribution):
- `monthly_outflow_total[m,b] = min(alloc_raw[m,b], cap[m,b])`

5) NO RESIDUAL REPAIR (CRITICAL):
You MUST NOT redistribute residual to other months.
After clipping, it is possible that:
- `sum_over_months monthly_outflow_total[m,b] < required_total_outflow[b]`

If this happens:
- Mark infeasible in `summary_markdown` and `global_validation`:
  - Add mismatch entries for bucket `b` showing required vs achieved totals.
- Continue producing the best-effort monthly plan using the clipped totals.

Output of Step B:
- Bucket-level totals `monthly_outflow_total[m,b]`.

---

### Step C: Split each month’s bucket outflow into outgoing edges (PROPORTIONAL + OVERWRITE-AWARE; ONE PASS)

For each month `m` and each real `from_bucket b`:

Definitions from GENERAL SHIFT PLAN:
- Outgoing edges `e = (b -> t, movement_type)` that exist in the general plan
- `edge_total[e]` = total_amount in general plan for that edge
- `edge_total_sum[b] = sum over outgoing edges from b of edge_total[e]`

1) Allowed edges only:
- Allocate ONLY across edges present in the GENERAL SHIFT PLAN.
- Do NOT invent or remove edges.

2) Apply user overwrites (edge-level fixed amounts):
If user feedback specifies `(m, b -> t)` for an edge that exists:
- Force `amount[m,e] = fixed_value` if feasible.

Feasibility:
- `fixed_value >= 0`
- `fixed_sum = sum of fixed edge amounts from (m,b)` must satisfy:
  - `fixed_sum <= monthly_outflow_total[m,b]`

If infeasible:
- Do NOT apply overwrite.
- Explain briefly in `summary_markdown`.

3) Proportional split of remaining amount (one pass only):
- `remaining = monthly_outflow_total[m,b] - fixed_sum`

Allocate remaining across NON-fixed outgoing edges using general proportions:
- For each non-fixed edge `e` from b:
  - `weight_e = edge_total[e] / sum(edge_total of non-fixed edges from b)`
  - `amount[m,e] = remaining * weight_e`

4) Minimal rounding (NO iterative repair):
- You MUST ensure per (m,b):
  - `sum over edges e from b of amount[m,e] == monthly_outflow_total[m,b]`
- If rounding causes a small residual, place it on ONE edge deterministically:
  1) Edge with largest `edge_total[e]`
  2) Prefer internal over external_out
  3) Lexicographic `to_bucket`

5) Movement type correctness:
- Internal edge → `movement_type="internal"`
- Bucket → EXTERNAL → `movement_type="external_out"`

6) Output:
- One record per `(month, from_bucket, to_bucket, movement_type)` with `amount > 0`
- Do NOT output zero-amount records.

---

### Step D: Allocate EXTERNAL inflows across months and destination buckets (PROPORTIONAL ONE PASS; NO ITERATION)

Applies ONLY if GENERAL SHIFT PLAN contains `external_in` edges (EXTERNAL -> real bucket).
If the general plan has no external_in, skip this step.

Definitions from GENERAL SHIFT PLAN:
- For each EXTERNAL → bucket edge `e`, `edge_total[e]`
- `external_total = sum(edge_total[e])`

1) Direction fixed:
- If general plan contains only `external_in`, output only `external_in`.
- NEVER output both directions.
- NEVER output `EXTERNAL -> EXTERNAL`.

2) Monthly external profile (one pass):
Use real execution volume as weights:
- `real_exec_volume[m] = sum over real buckets b of monthly_outflow_total[m,b]`
- `real_exec_total = sum over months real_exec_volume[m]`

If `real_exec_total > 0`:
- `w_ext[m] = real_exec_volume[m] / real_exec_total`
Else:
- `w_ext[m] = 1 / number_of_months` (even split)

3) Monthly external totals:
- `external_monthly_total[m] = external_total * w_ext[m]`

4) Split into destination buckets (one pass):
For each EXTERNAL → bucket edge `e`:
- `edge_weight[e] = edge_total[e] / external_total`
- `amount[m,e] = external_monthly_total[m] * edge_weight[e]`

5) Minimal rounding (NO iterative repair):
- For each month, ensure:
  - `sum over EXTERNAL -> * == external_monthly_total[m]`
- If rounding residual occurs, place it on the largest external edge for that month.

6) Output:
- One record per `(month, EXTERNAL, to_bucket, external_in)` with `amount > 0`
- Do NOT output zero-amount records.