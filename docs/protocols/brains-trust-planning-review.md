# Brains Trust - Planning Review Protocol

> Multi-model consensus on task planning before work begins.

## Overview

The Brains Trust is a pre-work quality gate that ensures task definitions are
thoroughly vetted before implementation begins. Multiple AI models (Claude,
Codex, Gemini) review and improve work definitions through iterative rounds of
investigation, critique, and refinement.

## Purpose

1. **Prevent underspecified work** - Catch missing requirements, edge cases,
   and ambiguities before a polecat starts coding
2. **Identify hidden dependencies** - Surface prerequisite work that wasn't
   obvious during initial planning
3. **Improve estimates** - Better-specified work leads to more predictable
   completion
4. **Cross-model validation** - Different models catch different issues;
   ensemble reduces blind spots

## Priority Tiers

The depth of review scales with task criticality:

| Priority | Rounds | Rationale |
|----------|--------|-----------|
| P0 (Critical) | 5 | High-stakes work needs exhaustive review |
| P1 (High) | 3 | Important work gets thorough vetting |
| P2 (Medium) | 2 | Standard work gets dual verification |
| P3+ (Low) | 1 | Simple work gets sanity check |

A **round** consists of three sequential passes (Claude -> Codex -> Gemini).
For P0 work, this means 15 total model passes (5 rounds × 3 models).

## Round Mechanics

Each round is strictly sequential (pass-the-parcel): the next model only begins
after the previous model finishes and hands off. There is no parallel review.
To reduce anchoring bias, each model writes its initial "Investigate" +
"Analyze" notes as a blind draft before reading prior rounds, then proceeds to
"Improve" and "Vote" with full context.

Each model pass follows this structure:

### 1. Investigate (Read Phase)
- Read the bead being reviewed (title, description, existing deps)
- Check related code/files mentioned in the bead
- Review any parent epic or blocking dependencies
- Understand the broader context

### 2. Analyze (Think Phase)
- Is the task well-defined? Can a polecat unambiguously implement this?
- Are there missing dependencies?
- Are there unstated assumptions?
- Could this introduce regressions or conflicts?
- Is the scope appropriate (not too large, not too small)?
- Are acceptance criteria clear?

### 3. Improve (Act Phase)
- File new beads for discovered prerequisite work
- Add missing dependencies via `bd dep add`
- Flag gaps in the bead description (via notes or structured feedback)
- Suggest refinements to title/description
- Update priority if mis-assessed

### 4. Vote (Assess Phase)
Record assessment as one of:
- **APPROVE** - Ready for implementation
- **REVISE** - Needs changes before proceeding
- **BLOCK** - Cannot proceed until prerequisites complete

## Consensus Rules

After all rounds complete for a priority tier:

| Outcome | Condition | Action |
|---------|-----------|--------|
| **Proceed** | ≥2/3 APPROVE | Work is trust-approved, can be dispatched |
| **Revise** | <2/3 APPROVE, no BLOCK majority | Return to Mayor for refinement |
| **Deadlock** | 1/3+ BLOCK, no clear consensus | Escalate to Overseer |

### Consensus Calculation

For N total votes (rounds × 3 models):
- Proceed: `APPROVE_count >= ceil(N * 2/3)`
- Deadlock: `BLOCK_count >= ceil(N / 3)` AND no clear majority

## Protocol Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        Mayor triggers review                      │
│                    (creates trust-assessment bead)                │
└──────────────────────────────────────────┬───────────────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────┐
                            │  Determine rounds from   │
                            │  target bead priority    │
                            └────────────┬─────────────┘
                                         │
                                         ▼
                                ┌──────────────┐
                                │ Claude pass  │
                                └──────┬───────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │  Codex pass  │
                                └──────┬───────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │ Gemini pass  │
                                └──────┬───────┘
                                       │
                                       ▼
                            ┌──────────────────────────┐
                            │   Record votes + notes   │
                            │   Check if more rounds   │
                            └────────────┬─────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              more rounds           consensus            deadlock
                    │                reached                  │
                    │                    │                    │
                    ▼                    ▼                    ▼
               [loop back]         ┌──────────┐        ┌──────────────┐
                                   │ APPROVED │        │ Escalate to  │
                                   │ dispatch │        │   Overseer   │
                                   └──────────┘        └──────────────┘
```

## Data Model

### Trust Assessment Bead

A tracking bead is created for each review:

```
Type: trust-assessment
Title: "Trust Review: <target-bead-title>"
Status: in_progress → approved | rejected | escalated

Fields (in notes/description):
  target_bead: <bead-id>
  priority_tier: P0|P1|P2|P3
  total_rounds: 5|3|2|1
  current_round: 1..total_rounds

Votes: (as structured notes)
  - round: 1, model: claude, vote: APPROVE, notes: "..."
  - round: 1, model: codex, vote: REVISE, notes: "Missing dep on auth service"
  - round: 1, model: gemini, vote: APPROVE, notes: "..."
  - round: 2, model: claude, vote: APPROVE, notes: "..."
  ...
```

### Filed Beads

New beads created during review link to the assessment:
```bash
bd create --type=task --title="..." --parent=<assessment-bead>
```

### Dependency Adjustments

Dependencies added during review are logged:
```bash
bd dep add <target-bead> <new-dep>
bd update <assessment-bead> --notes "Added dep: <target-bead> needs <new-dep>"
```

## Integration Points

### Triggering Review

Mayor triggers review when:
1. New P0-P2 work is created
2. Work is about to be dispatched
3. Manual review requested

```bash
# Mayor dispatches review
bd mol pour mol-brains-trust-plan --var target=<bead-id>
```

### Model Execution

Each model pass is executed sequentially:
- **Claude**: Native polecat session with code review molecule
- **Codex**: `/codex` skill invocation
- **Gemini**: MCP tool or API call (TBD based on integration)

### Handoff Mechanism

After each pass, the active model explicitly hooks/sling the next model
(`gt hook sling` or equivalent `bd` notification). This handoff is the trigger;
models should not poll for work.

### Post-Approval

Once approved, the target bead gains a label:
```bash
bd update <target-bead> --label trust-approved
```

This label allows Mayor's dispatch logic to prefer trust-approved work.

## Failure Modes

| Situation | Handling |
|-----------|----------|
| Model timeout | Record as ABSTAIN, continue with other models |
| All models ABSTAIN | Escalate to Overseer |
| Conflicting deps suggested | Flag for human review |
| Model API error | Retry once, then ABSTAIN |

## Configuration

Default config (can be overridden per-rig):

```toml
[brains_trust]
enabled = true
models = ["claude", "codex", "gemini"]
consensus_threshold = 0.67  # 2/3
deadlock_threshold = 0.34   # 1/3 BLOCK triggers escalation

[brains_trust.rounds]
p0 = 5
p1 = 3
p2 = 2
p3 = 1
p4 = 1  # backlog gets minimal review
```

## Example Session

**Target**: `sluice-xyz` - "Add rate limiting to API endpoints"
**Priority**: P1 (3 rounds)

### Round 1 (order: Claude -> Codex -> Gemini)

**Claude**: APPROVE
> Task is well-defined. Suggests adding note about which endpoints.

**Codex**: REVISE
> Files `sluice-abc`: "Need redis connection pool configured first"
> Adds dep: `sluice-xyz` needs `sluice-abc`

**Gemini**: APPROVE
> Agrees with Claude. Notes that middleware approach is standard.

### Round 2 (order: Claude -> Codex -> Gemini)

**Claude**: APPROVE
> Dep added addresses previous concern.

**Codex**: APPROVE
> Redis dep resolved my concern.

**Gemini**: APPROVE
> Ready to implement.

### Result

Round 2: 3/3 APPROVE = 100% approval
Total: 5/6 APPROVE, 1/6 REVISE = 83% approval > 67% threshold

**Outcome**: APPROVED, dispatched to polecat.

## Related

- [Brains Trust - Completion Review Protocol](./brains-trust-completion-review.md) (post-work)
- `mol-brains-trust-plan` formula
- `mol-brains-trust-complete` formula
