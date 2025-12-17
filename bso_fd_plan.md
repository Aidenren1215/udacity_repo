You are given three inputs:

1) CURRENT_FD_TABLE (markdown):
{CURRENT_FD_TABLE}

2) USER_INPUT (free-form text containing both rate outlook and the user’s requirements):
{USER_INPUT}

3) FEEDBACK (optional, may be empty; contains the user’s comments on a previous answer, including what to change or improve):
{FEEDBACK}

Your task is to produce a single JSON object with exactly these fields:
  - rationale
  - qualitative_changes
  - table


=========================================
TASK DEFINITION (IMPORTANT)
=========================================

- CURRENT_FD_TABLE represents the current FD position only.
- USER_INPUT contains:
    (a) rate outlook
    (b) the user’s requirements or preferences.

- You MUST proactively design a full set of Proposed FD Volumes
  for EACH tenor bucket based on USER_INPUT.
- This is NOT a pure rebalancing or mechanical adjustment task.
- Even if USER_INPUT does not specify exact numbers,
  you MUST infer a reasonable proposed allocation consistent with:
    - the rate outlook
    - liquidity considerations
    - interest-rate risk management
    - typical banking ALM practices.

- FEEDBACK, if present, refers to requested changes relative to a previous proposal.


=========================================
STRICT GENERATION + AUTO-CORRECTION RULES
=========================================

You MUST generate the table, then CHECK it yourself.
If you detect any numerical inconsistency, you MUST FIX it and re-check.
You MUST ALWAYS output the final corrected JSON.

IMPORTANT:
- Total proposed FD volume may increase or decrease.
- You MUST NOT assume any default constraint on total volume.
- Total-volume constraints apply ONLY if explicitly specified in FEEDBACK.


-----------------------------------------
A) HOW TO USE USER_INPUT AND FEEDBACK
-----------------------------------------

1) USER_INPUT (primary driver)

- USER_INPUT is the primary driver for designing the proposed FD allocation.
- You MUST actively translate the rate outlook and user requirements
  into proposed FD volumes across all tenor buckets.
- Do NOT assume the task is limited to rebalancing existing volumes.
- Directional intent in USER_INPUT (e.g. rising rates, need for liquidity,
  focus on short tenors, duration reduction) MUST be reflected in the table.

2) FEEDBACK (highest-priority override)

- Treat FEEDBACK as the highest-priority adjustment signal.

- Distinguish carefully between the following types of FEEDBACK:

  (a) Qualitative allocation intent  
      (e.g. “focus on short tenors”, “increase short tenors”,
       “de-risk long tenors”, “tilt toward liquidity”)

      • You MUST reflect this intent in:
          - qualitative_changes
          - table (by adjusting proposed volumes)
      • All numerical rules MUST still be satisfied.

  (b) Pure stylistic or explanatory changes  
      (e.g. “improve explanation”, “be more concise”, “change tone”)

      • You MUST update:
          - rationale
          - qualitative_changes
      • You MUST NOT change any numbers in the table.

- If FEEDBACK specifies a constraint on total proposed volume
  (e.g. “total must increase by +200”, “reduce total by 5%”,
   “keep total unchanged”, “net outflow ≤ 100”),
  you MUST enforce it.

- If FEEDBACK does NOT specify any constraint on total proposed volume,
  you MUST NOT impose any total-volume constraint or assume any default threshold.

- If FEEDBACK conflicts with hard numerical rules below,
  you MUST respect the numerical rules and adjust as close as possible
  to the user’s intent.

- Never ignore FEEDBACK.


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
C) qualitative_changes: content + STYLE requirements
-----------------------------------------
The field qualitative_changes MUST be a markdown-formatted bullet list.

Content rules:
- Each bullet should describe how volumes change for one or a small group
  of tenor buckets, and why.
- The bullets MUST be consistent with the final table,
  USER_INPUT and FEEDBACK.

STYLE rules:
- Use a concise, professional banking / risk-management tone.
- Write directly and factually.
- Do NOT use meta-phrases such as “Below are…” or “In this section…”.
- Do NOT apologize or hedge.
- Each bullet should start with the relevant bucket(s) or action, for example:
    - "• 3M: Increased to enhance liquidity in a rising-rate environment."
    - "• >1Y: Reduced to limit duration risk."
- Do NOT mention that this is JSON or that this is a model output.


-----------------------------------------
D) TABLE FORMAT (fixed)
-----------------------------------------
Each row in table MUST include:
  - 'FD Tenor Bucket'
  - 'Current FD Volume (S$m)'
  - 'Proposed FD Volume (S$m)'
  - 'Change (S$m)'
  - '% Change'

Rules:
- All numeric values MUST be pure numbers.
- 'Proposed FD Volume (S$m)' MUST be >= 0.


-----------------------------------------
E) NUMERICAL RULES (MUST BE FIXED IF BROKEN)
-----------------------------------------

For every row:
  Change_expected = Proposed FD Volume - Current FD Volume
  Pct_expected    = (Change_expected / Current FD Volume) * 100

You MUST ensure:
  'Change (S$m)' == Change_expected
  '% Change'     == Pct_expected

Edge cases:
- If Current FD Volume is 0:
    - 'Change (S$m)' must still be correct.
    - Set '% Change' to 0.
    - Do NOT output NaN or Infinity.

Total volume rules:
- Total volume conservation is NOT required.
- If and only if FEEDBACK specifies a total-volume constraint,
  you MUST satisfy it (within reasonable floating-point tolerance).


-----------------------------------------
F) SELF-CHECK LOOP (Gemma3 friendly)
-----------------------------------------
Before outputting:
  1) Generate an initial table based on CURRENT_FD_TABLE, USER_INPUT and FEEDBACK.
  2) Check all rows for formula correctness.
  3) Check for invalid numbers (negative proposed, NaN, Infinity).
  4) If FEEDBACK specifies a total-volume constraint, verify it is satisfied.
  5) If ANY mismatch is found:
        Fix the numbers.
        Re-check.
  6) After corrections, output ONLY the final valid JSON.


-----------------------------------------
G) NO INTERNAL CALCULATION DISPLAY
-----------------------------------------
Do NOT show chain-of-thought, intermediate steps, or debugging notes.
Output ONLY the final JSON.


-----------------------------------------
H) FINAL REQUIREMENT
-----------------------------------------
Your final response MUST strictly follow this JSON schema:
{format_instructions}
