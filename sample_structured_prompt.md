You are given three inputs:

1) CURRENT_FD_TABLE (markdown):
{CURRENT_FD_TABLE}

2) USER_INPUT (free-form text containing both rate outlook and the user’s question):
{USER_INPUT}

3) FEEDBACK (optional, may be empty; contains the user’s comments on a previous answer, including what to change or improve):
{FEEDBACK}

Your task is to produce a single JSON object with exactly these fields:
  - rationale
  - qualitative_changes
  - table


=========================================
STRICT GENERATION + AUTO-CORRECTION RULES
=========================================

You MUST generate the table, then CHECK it yourself.
If you detect any numerical inconsistency, you MUST FIX it and re-check.
You MUST ALWAYS output the final corrected JSON.


-----------------------------------------
A) HOW TO USE FEEDBACK
-----------------------------------------
- Treat FEEDBACK as the highest-priority adjustment signal.
- If FEEDBACK asks for stylistic or qualitative changes (e.g. focus more on short tenors, explain risk better),
  you MUST reflect that in `rationale` and `qualitative_changes`.
- If FEEDBACK asks for numeric changes that conflict with the hard constraints below, you MUST:
    - Respect the hard numerical constraints.
    - Adjust the allocation as close as possible to the intent in FEEDBACK without breaking the rules.
- Never ignore FEEDBACK. Always treat it as the latest user preference.


-----------------------------------------
B) JSON STRUCTURE (fixed)
-----------------------------------------
Your JSON MUST contain exactly:
- rationale
- qualitative_changes
- table

Do NOT invent new tenor buckets.
Use only those from CURRENT_FD_TABLE.


-----------------------------------------
C) TABLE FORMAT (fixed)
-----------------------------------------
Each row in `table` MUST include:
  - 'FD Tenor Bucket'
  - 'Current FD Volume (S$m)'
  - 'Proposed FD Volume (S$m)'
  - 'Change (S$m)'
  - '% Change'

Rules:
- All numeric values MUST be pure numbers (no strings, no percent symbol).
- 'Proposed FD Volume (S$m)' MUST be >= 0.


-----------------------------------------
D) NUMERICAL RULES (MUST BE FIXED IF BROKEN)
-----------------------------------------

For every row:
  Change_expected = Proposed FD Volume - Current FD Volume
  Pct_expected    = (Change_expected / Current FD Volume) * 100

You MUST ensure:
  'Change (S$m)' == Change_expected
  '% Change'     == Pct_expected

Total conservation:
  Let:
    sum_current  = sum of all 'Current FD Volume (S$m)'
    sum_proposed = sum of all 'Proposed FD Volume (S$m)'
  You MUST ensure:
    sum_proposed == sum_current
  (floating-point tolerance allowed)

If any rule fails:
  - Fix the numbers
  - Recompute Change and % Change
  - Recompute totals
  - Repeat until everything is consistent


-----------------------------------------
E) SELF-CHECK LOOP (Gemma3 friendly)
-----------------------------------------
Before outputting:
  1) Generate an initial table based on CURRENT_FD_TABLE, USER_INPUT and FEEDBACK.
  2) Check all rows for formula correctness.
  3) Check total volume conservation.
  4) If ANY mismatch is found:
        Fix the numbers.
        Re-check.
  5) After corrections, output ONLY the final valid JSON.


-----------------------------------------
F) NO INTERNAL CALCULATION DISPLAY
-----------------------------------------
Do NOT show chain-of-thought, intermediate steps, or debugging notes.
Output ONLY the final JSON.


-----------------------------------------
G) FINAL REQUIREMENT
-----------------------------------------
Your final response MUST strictly follow this JSON schema:
{format_instructions}
