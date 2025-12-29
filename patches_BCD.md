### Step B: Allocate monthly outflow for each real from_bucket (PROPORTIONAL; CAPACITY AS VALIDATION ONLY)

Assumption (BUSINESS PRECONDITION):
- The GENERAL SHIFT PLAN is executable under the provided maturity ladder capacity.
- Therefore, proportional allocation SHOULD satisfy capacity without any clipping/optimization.

For each real tenor bucket `b`:

1) Required total outflow (from GENERAL SHIFT PLAN):
- `required_total_outflow[b]` = sum of `total_amount` over all GENERAL SHIFT PLAN edges starting from bucket `b`
  (include internal + external_out if applicable).

2) Capacity weights (from maturity ladder):
- `cap[m,b] = capacity[month=m, bucket=b]`
- `cap_total[b] = sum_over_months cap[m,b]`

If `cap_total[b] == 0`:
- If `required_total_outflow[b] == 0`, set `monthly_outflow_total[m,b] = 0` for all months.
- Else, mark infeasible and STOP allocation for this bucket:
  - Record mismatch in `global_validation.bucket_mismatches`
  - Explain briefly in `summary_markdown`

3) Proportional monthly allocation (NO optimization, NO min/clip step):
- Define weights: `w[m,b] = cap[m,b] / cap_total[b]`
- Allocate: `monthly_outflow_total[m,b] = required_total_outflow[b] * w[m,b]`

4) Capacity validation (VALIDATION ONLY; NO repair):
You MUST validate:
- For every month m: `monthly_outflow_total[m,b] <= cap[m,b]`

If any violation occurs:
- Mark infeasible in `global_validation` and `summary_markdown`.
- Do NOT "clip", do NOT redistribute, do NOT optimize.
- Still proceed to produce best-effort outputs, but set `global_validation.status="MISMATCH"`.

Output of Step B:
- Bucket-level totals `monthly_outflow_total[m,b]` for all months and real buckets.

---

### Step C: Split each month’s bucket outflow into outgoing edges (PROPORTIONAL + OVERWRITE-AWARE; NO GLOBAL RE-SOLVING)

For each month `m` and real `from_bucket b`:

Definitions from GENERAL SHIFT PLAN:
- Outgoing edges `e = (b -> t, movement_type)` that exist in the general plan
- `edge_total[e]` = total_amount in general plan for that edge
- `edge_total_sum[b] = sum of edge_total[e]` over edges from b

1) Allowed edges only:
- Allocate ONLY across edges present in the GENERAL SHIFT PLAN.
- Do NOT invent or remove edges.

2) Apply user overwrites (local-only):
If user feedback specifies `(m, b -> t, movement_type)`:
- Force `amount[m,e] = fixed_value` if feasible locally.

Local feasibility (within the same month and from_bucket):
- `fixed_value >= 0`
- `fixed_sum = sum of fixed edge amounts from (m,b)` must satisfy:
  - `fixed_sum <= monthly_outflow_total[m,b]`

If infeasible:
- Reject overwrite and explain briefly in `summary_markdown`.
- Do NOT attempt global re-solving.

3) Proportional split of remaining amount (one pass):
- `remaining = monthly_outflow_total[m,b] - fixed_sum`

Allocate remaining across NON-fixed outgoing edges using GENERAL proportions:
- `weight_e = edge_total[e] / sum(edge_total of non-fixed edges from b)`
- `amount[m,e] = remaining * weight_e`

4) Minimal per-(m,b) reconciliation (no iteration):
- Ensure:
  - `sum over edges e from b of amount[m,e] == monthly_outflow_total[m,b]`
- If a small rounding residual exists, put it on ONE edge deterministically:
  1) largest `edge_total[e]`
  2) prefer internal over external_out
  3) lexicographic `to_bucket`

5) Movement type correctness:
- Internal edge → `movement_type="internal"`
- Bucket → EXTERNAL → `movement_type="external_out"`

---

### Step D: Allocate EXTERNAL inflows across months and destination buckets (PROPORTIONAL; NO ITERATION)

Applies ONLY if GENERAL SHIFT PLAN contains `external_in` edges (EXTERNAL -> real bucket).
If the general plan has no external_in, skip this step.

Definitions from GENERAL SHIFT PLAN:
- For each EXTERNAL → bucket edge `e`, `edge_total[e]`
- `external_total = sum(edge_total[e])`

1) Direction fixed (net-only):
- If general plan has only `external_in`, output ONLY `external_in`.
- If general plan has only `external_out`, output ONLY `external_out`.
- NEVER output both.
- NEVER output `EXTERNAL -> EXTERNAL`.

2) Monthly EXTERNAL profile (simple proportional, one pass):
- `real_exec_volume[m] = sum over real buckets b of monthly_outflow_total[m,b]`
- `real_exec_total = sum over months real_exec_volume[m]`

If `real_exec_total > 0`:
- `w_ext[m] = real_exec_volume[m] / real_exec_total`
Else:
- `w_ext[m] = 1 / number_of_months`

3) Allocate monthly external totals:
- `external_monthly_total[m] = external_total * w_ext[m]`

4) Split into destination buckets (one pass):
For each EXTERNAL → bucket edge `e`:
- `edge_weight[e] = edge_total[e] / external_total`
- `amount[m,e] = external_monthly_total[m] * edge_weight[e]`

5) Minimal per-month reconciliation (no iteration):
- If rounding residual exists in a month, place it on the largest external edge for that month.