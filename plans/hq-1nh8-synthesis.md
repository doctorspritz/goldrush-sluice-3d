# Brains Trust Synthesis: hq-1nh8 (Planning Review Protocol)

**Bead:** hq-1nh8 (Brains Trust Planning Review Protocol)
**Round:** 3 (Gemini)
**Status:** Synthesis & Finalization

## 1. Overview

This synthesis builds on the foundational rounds from Codex (Rounds 1-5, Round 2 follow-up) and Claude (Round 1). The goal is to finalize the protocol for multi-model planning review.

## 2. Consensus Points

All models agree on the following core components:
- **Tiered Review:** Rounds scale with priority (P0=5, P1=3, P2=2, P3=1).
- **Artifact Requirements:** Each round must produce/update a plan, risk list, dependencies, and acceptance criteria.
- **Consensus Rule:** 2/3 majority required to proceed; fallback to 2/2 or model+overseer for critical cases.
- **Beads Integration:** Use `bd` to track gaps as blocking sub-beads.
- **Stop-Rule:** Mechanism to skip/abbreviate trivial work.

## 3. Conflict Resolution

### 3.1 Parallel vs. Sequential Review
- **Claude Proposal:** Parallel blind first-round for P0/P1 to avoid anchoring bias.
- **Codex Proposal:** Sequential with "independent concerns" written before reading previous rounds.
- **Resolution:** Adopt the **"Blind Draft"** approach. Each reviewer must write their initial "Investigate & Analyze" phase notes *privately* (or in a hidden section) before reading previous rounds. However, the final "Improve & Vote" phase remains sequential to benefit from prior refinements. Reserve full parallel execution only for "Ultra-Critical" P0+ (as defined by Overseer).

### 3.2 Artifact Versioning
- **Claude Proposal:** Versioned files (`plans/<id>-v1.md`, etc.).
- **Codex Proposal:** Single file (`plans/<id>.md`) with per-round sections/deltas.
- **Resolution:** **Single-file Canonical Plan.** Use `plans/<id>.md` as the living document. History is tracked via Git. Each round appends a "Round X Review" section to the bottom of the file while updating the main plan body. This avoids file sprawl and keeps the "latest" state clear.

### 3.3 Emergency Bypass / Fast-Path
- **Claude Proposal:** Bypass with owner sign-off for urgent fixes.
- **Codex Proposal:** "Fast-path" (1 model + overseer + mandatory post-work review).
- **Resolution:** Adopt **"Fast-Path"**. For critical urgent work (P0), a single model review plus Overseer (human) approval allows implementation to start. A "post-mortem" completion review (hq-gerz) is mandatory within 48 hours to catch any shortcuts taken.

## 4. Final Perspective

### 4.1 Handoff Mechanism
Explicitly adopt `gt hook sling` (or equivalent `bd` notification) as the trigger for the next model. Models should not be expected to poll.

### 4.2 The "Seal of Approval"
The final step of the protocol must be an explicit label: `bd update <id> --label trust-approved`. This provides a machine-readable flag for dispatchers.

### 4.3 Convergence Exit Criteria
A review round is "Complete" when:
1. All artifacts are present in `plans/<id>.md`.
2. All critical blockers (identified in that round) are filed as sub-beads.
3. The model records a vote (APPROVE/REVISE/BLOCK).

## 5. Open Blockers (TBD)

The following sub-beads must be resolved before the protocol is fully operational:

| Bead | Title | Priority |
|------|-------|----------|
| hq-u1n4 | Define consensus thresholds and fallback for missing models | P1 |
| hq-ipjd | Define deadlock criteria and escalation path | P1 |
| hq-zrel | Specify round outputs and checklist | P2 |
| hq-5k1m | Add stop-rule for trivial work and scope calibration | P2 |

## 6. Recommendation

**APPROVE** hq-1nh8 with the integrated refinements from this synthesis. The core protocol is well-defined with:
- Comprehensive documentation in `docs/protocols/brains-trust-planning-review.md`
- Working formula in `mol-brains-trust-plan.formula.toml`
- Sub-tasks filed for remaining refinements

Proceed to implementing the sub-tasks (hq-u1n4, hq-zrel, hq-ipjd, hq-5k1m).

## 7. Votes Summary

| Round | Model | Vote | Key Notes |
|-------|-------|------|-----------|
| 1-5 | Codex | APPROVE with refinements | Filed sub-beads for gaps |
| 1 | Claude | APPROVE with gaps noted | Handoff, versioning, emergency bypass |
| 2 | Codex | APPROVE | Refined Claude's points |
| 3 | Synthesis | **APPROVE** | Conflicts resolved, ready for implementation |

**Consensus: APPROVE** (all models aligned with refinements tracked in sub-beads)
