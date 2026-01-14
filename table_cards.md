This file is the **whitelist** of tables/fields the Planner is allowed to use.
If a field/table is NOT listed here, it must be treated as unavailable.

---

## FACT TABLES

### [FACT][MONTHLY] monthly_deposit_snapshot
- table_type: monthly
- description: Month-end deposit snapshot (aggregated)
- time_column: month
- grain: monthly

- metrics:
  - balance            # month-end balance snapshot
  - interest_rate      # snapshot rate; Planner must use balance-weighted average

- dimensions_in_fact:
  - currency            # e.g. SGD, USD
  - segment_code        # internal segment code (use mapping for segment_name)
  - deposit_type_code   # internal deposit type code (use mapping for deposit_type_name)

- join_keys_out:
  - segment_code
  - deposit_type_code

- notes:
  - Use this table for all monthly questions in the POC.

---

## MAPPING TABLES

### [MAPPING] map_segment
- description: segment_code -> segment_name
- mapping_type: code_to_label

- columns:
  - to_mapping_key: segment_code
  - attributes:
    - segment_name

- join_contracts:
  - from_fact_table: monthly_deposit_snapshot
    from_fact_key: segment_code
    to_mapping_key: segment_code
    join_type: left

- planner_usage:
  - can_filter_on (requires join):
    - segment_name
  - can_group_by (requires join):
    - segment_name

- notes:
  - One segment_code maps to one segment_name.

---

### [MAPPING] map_deposit_type
- description: deposit_type_code -> deposit_type_name
- mapping_type: code_to_label

- columns:
  - to_mapping_key: deposit_type_code
  - attributes:
    - deposit_type_name

- join_contracts:
  - from_fact_table: monthly_deposit_snapshot
    from_fact_key: deposit_type_code
    to_mapping_key: deposit_type_code
    join_type: left

- planner_usage:
  - can_filter_on (requires join):
    - deposit_type_name
  - can_group_by (requires join):
    - deposit_type_name

- notes:
  - One deposit_type_code maps to one deposit_type_name.