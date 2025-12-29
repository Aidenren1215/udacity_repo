### Step B: Allocate monthly outflow for each real from_bucket (PROPORTIONAL, NOT GREEDY)

For each real tenor bucket `b`:

1) Compute required totals:
- `required_total_outflow[b]` = sum of `total_amount` over all GENERAL SHIFT PLAN edges starting from `b`
  (include internal + external_out if applicable).

2) Compute monthly capacity proportions:
- `cap[m,b] = capacity[month=m, bucket=b]`
- `cap_total[b] = sum_over_months cap[m,b]`

If `cap_total[b] == 0`:
- If `required_total_outflow[b] == 0`, allocate nothing.
- Else, mark infeasible:
  - Still output JSON
  - Set `global_validation.status = "MISMATCH"`
  - Explain briefly in `summary_markdown`.

Otherwise define weights:
- `w[m,b] = cap[m,b] / cap_total[b]`
- `0 <= w[m,b] <= 1`
- `sum_over_months w[m,b] == 1`

3) First-pass proportional allocation:
- `alloc_raw[m,b] = required_total_outflow[b] * w[m,b]`

4) Enforce capacity and fix residual:
- Initial assignment:
  - `monthly_outflow_total[m,b] = min(alloc_raw[m,b], cap[m,b])`
- Compute residual:
  - `residual = required_total_outflow[b] - sum_over_months monthly_outflow_total[m,b]`

If `residual > 0`:
- Define slack:
  - `slack[m] = cap[m,b] - monthly_outflow_total[m,b]`
- Let `slack_total = sum_over_months slack[m]`
  - If `slack_total < residual`, infeasible → set global MISMATCH and explain.
  - Else redistribute:
    - `monthly_outflow_total[m,b] += residual * (slack[m] / slack_total)`

If `residual < 0` (rounding):
- Reduce excess proportionally from months with positive outflow,
  preferring future months if needed.

5) Deterministic rounding:
- Ensure exactly:
  - `sum_over_months monthly_outflow_total[m,b] == required_total_outflow[b]`
- Apply final adjustment on the last feasible month only.
- Never violate `monthly_outflow_total[m,b] <= cap[m,b]`.

Output of Step B:
- Bucket-level totals `monthly_outflow_total[m,b]`.

---

### Step C: Split each month’s bucket outflow into outgoing edges (PROPORTIONAL + OVERWRITE-AWARE)

For each month `m` and real `from_bucket b`:

Definitions from GENERAL SHIFT PLAN:
- For each outgoing edge `e = (b -> t, movement_type)`:
  - `edge_total[e]`
- `edge_total_sum[b] = sum of edge_total[e]` over edges from `b`

1) Allowed edges only:
- Allocate ONLY across edges present in the GENERAL SHIFT PLAN.
- Do NOT invent or remove edges.

2) Apply user overwrites:
- If user feedback specifies `(m, e)`:
  - Force `amount[m,e] = fixed_value` if feasible.
Feasibility:
- `fixed_value >= 0`
- Sum of fixed edges from `(m,b)` ≤ `monthly_outflow_total[m,b]`

If infeasible:
- Do NOT apply overwrite.
- Explain briefly in `summary_markdown`.

3) Proportional split of remaining amount:
- `fixed_sum = sum of fixed edge amounts from (m,b)`
- `remaining = monthly_outflow_total[m,b] - fixed_sum`

Allocate remaining across non-fixed edges:
- `weight_e = edge_total[e] / sum(edge_total of non-fixed edges)`
- `amount[m,e] = remaining * weight_e`

4) Rounding and reconciliation:
- Ensure:
  - `sum over edges e from b of amount[m,e] == monthly_outflow_total[m,b]`
- Deterministic residual handling:
  1. Largest `edge_total`
  2. Prefer internal over external_out
  3. Lexicographic `to_bucket`

5) Movement type correctness:
- Internal edge → `movement_type = "internal"`
- Bucket → EXTERNAL → `movement_type = "external_out"`

6) Output:
- One record per `(month, from_bucket, to_bucket, movement_type)` with `amount > 0`
- Do NOT output zero-amount records.

---

### Step D: Allocate EXTERNAL inflows across months and destination buckets (PROPORTIONAL, NET-ONLY)

Definitions:
- From GENERAL SHIFT PLAN:
  - For each EXTERNAL → bucket edge `e`, `edge_total[e]`
- `external_total = sum(edge_total[e])`

1) Direction fixed:
- If GENERAL PLAN has only `external_in`, output ONLY `external_in`.
- NEVER output both directions.
- NEVER output `EXTERNAL -> EXTERNAL`.

2) Monthly EXTERNAL profile:
- `real_exec_volume[m] = sum over real buckets b of monthly_outflow_total[m,b]`
- `real_exec_total = sum over months real_exec_volume[m]`

If `real_exec_total > 0`:
- `w_ext[m] = real_exec_volume[m] / real_exec_total`
Else:
- Distribute evenly across months.

3) Monthly EXTERNAL totals:
- `external_monthly_total[m] = external_total * w_ext[m]`

4) Split into destination buckets:
For each EXTERNAL → bucket edge `e`:
- `edge_weight[e] = edge_total[e] / external_total`
- `amount[m,e] = external_monthly_total[m] * edge_weight[e]`

5) Rounding and reconciliation:
- For each month:
  - Sum of EXTERNAL → * equals `external_monthly_total[m]`
- For each bucket:
  - Sum over months equals GENERAL PLAN `edge_total[e]`
- Fix residuals deterministically (last month, largest edge).

6) Output:
- One record per `(month, EXTERNAL, to_bucket, external_in)`
- Do NOT output zero-amount records.