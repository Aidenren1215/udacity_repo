The ALCO Minutes Agent adopts a role-based access control (RBAC) model using Active Directory (AD) groups to manage user permissions in a consistent and auditable manner. User access is categorised into three roles: GF_AI_CT_Admin, GF_AI_CT_TeamLeads, and GF_AI_CT_Users, each with clearly defined privileges.

Admins have full access to application features, configuration options, and approved data. Team Leads have full UI access with standardised model configurations and the same data and guardrail policies as Admins, but without administrative controls. End users are subject to the strongest guardrails and the most restrictive data access, preventing queries on sensitive leadership-related content or restricted production data.

This access model enforces the principle of least privilege and ensures governance and compliance requirements are consistently applied across the application.


The ALCO Minutes Agent is designed to support regionalized ALCO operations, where each country maintains its own independent set of ALCO minutes. ALCO minutes are segregated by country and managed as separate datasets, reflecting local regulatory, governance, and operational requirements.

Access to each country’s ALCO minutes is controlled via country-specific Active Directory (AD) groups, with authorization owned by the respective local Corporate Treasury (CT) teams. Where explicitly approved by local CT administrators, a single user may be granted access to multiple country-level ALCO minutes through membership in multiple AD groups. This approach preserves strict regional data separation while allowing controlled cross-country access when business needs require it.