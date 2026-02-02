The ALCO Minutes Agent adopts a role-based access control (RBAC) model using Active Directory (AD) groups to manage user permissions in a consistent and auditable manner. User access is categorised into three roles: GF_AI_CT_Admin, GF_AI_CT_TeamLeads, and GF_AI_CT_Users, each with clearly defined privileges.

Admins have full access to application features, configuration options, and approved data. Team Leads have full UI access with standardised model configurations and the same data and guardrail policies as Admins, but without administrative controls. End users are subject to the strongest guardrails and the most restrictive data access, preventing queries on sensitive leadership-related content or restricted production data.

This access model enforces the principle of least privilege and ensures governance and compliance requirements are consistently applied across the application.


The ALCO Minutes Agent is designed with strict country-level segregation, where each country maintains its own independent set of ALCO minutes. To enforce this separation at the data layer, ALCO minutes are indexed into region-specific Elasticsearch sub-indexes, with one sub-index per country or region.

User queries are restricted to the Elasticsearch sub-indexes associated with the user’s Active Directory (AD) group membership, ensuring that users can only retrieve ALCO minutes relevant to their own region. Authorization for each regional dataset is managed by the respective local Corporate Treasury (CT) teams. Where explicitly approved by local CT administrators, a single user may be granted access to multiple regional sub-indexes through membership in multiple AD groups.

This design enforces regional data isolation by default while supporting controlled cross-region access when required.