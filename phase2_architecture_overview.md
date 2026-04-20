# Phase 2 — Balance Sheet Optimization Agent

## Architecture Overview

Phase 2 introduces an agentic balance sheet optimization system that translates macro strategies from Phase 1 into concrete adjustment plans across the bank's FD, loan, and CASA books. The system follows a **supervisor–sub-agent hierarchy**: one orchestrator (BSO) plans and coordinates, five specialized sub-agents each own a domain or capability, and two shared tools handle data access and mechanical computation. This separation keeps each agent focused on a single responsibility, routes all cross-domain coordination through the BSO for predictability, and relies on deterministic tools for any output that must be reproducible.

### Interaction Pattern

Phase 1's macro strategy arrives at the BSO, which decomposes it and dispatches sub-directives to the three proposers in parallel. Each proposer calls the Data Agent to retrieve its current portfolio state, reasons about how to implement its directive, invokes the Volume Adjuster to generate a numerical plan, and returns its proposal to the BSO. The BSO then invokes the Optimization Agent to refine the combined plan and evaluates the result against the baseline balance sheet. If it represents a clear improvement, it is emitted as the approved adjustment plan. The Analytics Agent sits alongside this main flow and can be invoked by the BSO at any point for ad-hoc analysis, such as questions raised during ALCO meetings.

*[draw.io diagram here]*

### Components

**BSO — Balance Sheet Orchestrator**
The commander of the workflow. BSO receives the macro strategy from Phase 1 (e.g. *"interest rates expected to drop over the next six months"*), decomposes it into concrete directives for each product line, dispatches those directives to the relevant proposers, consolidates their proposals, invokes the Optimization Agent for cross-product refinement, and evaluates the final plan against the current baseline before approval.

**FD Proposer**
Owns the Fixed Deposit book. Given a BSO directive, it inspects the current FD portfolio, forms a tenor-restructuring strategy, and returns a concrete adjustment proposal. For example, when rates are expected to fall, it may propose shifting maturing 1-year deposits into ≤3-month tenors to avoid locking depositors into higher-cost long-tenor positions.

**Loan Proposer**
Owns the loan book. Mirrors the FD Proposer on the asset side — for example, extending short-tenor loans into longer tenors when a rate decline is anticipated, in order to lock in current higher yields.

**CASA Proposer**
Owns Current and Savings Accounts. Proposes structural adjustments to CASA composition consistent with the BSO directive, taking into account the behavioural characteristics that distinguish CASA from term products.

**Analytics Agent**
A general-purpose analytical sub-agent, invoked by the BSO when an ad-hoc number or piece of analysis is needed — most notably to support ALCO discussions. It operates independently of the adjustment workflow and does not produce proposals.

**Optimization Agent**
Invoked by the BSO after the three proposers have returned their individual proposals. Its role is cross-product refinement: reconciling the FD, Loan, and CASA proposals into a jointly optimized plan that maximizes NIM under global constraints, rather than treating each book in isolation.

**Data Agent (tool)**
The single point of access to the underlying data warehouse. All sub-agents that need current-state information on FD, loan, or CASA portfolios — volumes, rates, maturity distributions — call the Data Agent rather than querying the database directly. This centralization simplifies governance and data lineage.

**Volume Adjuster (tool)**
A deterministic tool that takes a portfolio table plus user-defined boundary conditions (target tenor shifts, volume caps, minimum liquidity thresholds, etc.) and returns a rebalanced volume plan. It is the mechanical engine behind every proposer's numerical output.
