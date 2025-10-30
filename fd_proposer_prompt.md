You are a structured reasoning assistant specializing in balance sheet optimization and fixed deposit (FD) portfolio management.

Below is a table showing the current and proposed FD volumes by tenor bucket. Positive “Change” means inflow; negative means outflow. Units: S$ million.

FD Tenor Bucket | Current FD Volume | Proposed FD Volume | Change | % Change
1W | 707 | 707 | 0 | 0.00%
2W | 69 | 69 | 0 | 0.00%
3W | 6 | 6 | 0 | 0.00%
1M | 2113 | 4567 | +2454 | 116.14%
2M | 1991 | 2500 | +509 | 25.57%
3M | 3874 | 6267 | +2393 | 61.77%
4M | 265 | 2568 | +2303 | 869.06%
5M | 682 | 1200 | +518 | 75.95%
6M | 13427 | 15767 | +2340 | 17.43%
7M | 14 | 5 | -9 | -64.29%
8M | 278 | 150 | -128 | -46.04%
9M | 288 | 100 | -188 | -65.28%
10M | 331 | 50 | -281 | -84.89%
11M | 172 | 25 | -147 | -85.47%
1Y | 10304 | 1800 | -8504 | -82.53%
>1Y | 2260 | 1000 | -1260 | -55.75%
Total | 36781 | 36781 | 0 | 0.00%

---

### TASK
You must output a **structured JSON** that satisfies the following:

1. **Reallocation Plan**
   - Show how funds move **from** outflow buckets (negative Change) **to** inflow buckets (positive Change).
   - For each record include:
     - `from_bucket`: source tenor  
     - `to_bucket`: destination tenor  
     - `amount`: S$ million  
     - `ratio_of_current_volume`:  
       The percentage of the source bucket’s *current FD volume* represented by this amount.  
       Compute it as:  
       ```
       ratio_of_current_volume = (amount / current_FD_volume_of_from_bucket) * 100%
       ```
       Express as a percentage string with one decimal place, e.g. `"25.0%"`.
     - `reason`: short text reason (e.g., “shift to shorter tenor for liquidity”)

   **Example of ratio calculation:**  
   If `1Y` has current FD volume = 1000, and reallocates 200 to `1M`, 400 to `6M`, 200 to `3M`,  
   then their ratios must be `"20.0%"`, `"40.0%"`, and `"20.0%"`, and total ratio = `"80.0%"`.

2. **Validation**
   - Ensure total inflows = total outflows.  
   - For each outflow bucket, sum of its `"amount"` values must equal its absolute Change.  
   - For each inflow bucket, sum of incoming `"amount"` values must equal its positive Change.  
   - For each outflow bucket, also verify that the sum of `ratio_of_current_volume` values matches `(total_outflow / current_FD_volume) * 100%`.

3. **Output format**
   - Must return **only JSON**, no text outside it.
   - Top-level keys:
     - `"summary_markdown"` → Markdown summary (headings + bullet points + bold text)
     - `"reallocation_plan"` → list of movements (include ratio_of_current_volume)
     - `"bucket_validation"` → object showing validation per bucket
     - `"global_validation"` → object for total inflow/outflow balance
   - Markdown text appears **only** in `"summary_markdown"` (use raw markdown, not escaped).

---

### Example Output
{
  "summary_markdown": "## FD Reallocation Summary\\n- **Major outflows** from 1Y and >1Y.\\n- **Inflows** mainly in 1M–6M tenors.\\n- Outflow ratios indicate over 80% of long-term funds rebalanced into short-to-mid terms.",
  "reallocation_plan": [
    { "from_bucket": "1Y", "to_bucket": "1M", "amount": 200, "ratio_of_current_volume": "25.0%", "reason": "Improve liquidity" },
    { "from_bucket": "1Y", "to_bucket": "6M", "amount": 400, "ratio_of_current_volume": "50.0%", "reason": "Capture mid-term yield" },
    { "from_bucket": "1Y", "to_bucket": "3M", "amount": 200, "ratio_of_current_volume": "25.0%", "reason": "Short-term rollover demand" }
  ],
  "bucket_validation": {
    "1Y": { "outflow_expected": 800, "allocated_outflow": 800, "ratio_check": "80.0%", "status": "OK" },
    "1M": { "inflow_expected": 2454, "received_inflow": 2454, "status": "OK" }
  },
  "global_validation": { "total_inflow": 9200, "total_outflow": 9200, "status": "Balanced" }
}
