# Brains Trust - Completion Review Protocol

> Multi-model review after work is marked done.

## Overview

The completion review is the post-work counterpart to the planning review.
After a polecat marks work complete, the Brains Trust verifies the
implementation before it is merged.

## Purpose

1. **Validate implementation** - Ensure the code matches the task definition
   and acceptance criteria
2. **Catch regressions** - Identify obvious bugs, edge cases, or performance
   regressions before merge
3. **Ensure quality gates** - Confirm tests, docs, and safety checks are
   adequate for the change
4. **Cross-model validation** - Multiple models catch different issues,
   reducing blind spots

## Key Differences from Planning Review

| Aspect | Planning Review | Completion Review |
|--------|-----------------|-------------------|
| **When** | Before work begins | After work is done |
| **Focus** | Task definition | Implementation quality |
| **Checks** | Missing deps, ambiguities | Bugs, regressions, missing tests |
| **Output** | Trust-approved label | Trust-complete label or filed gaps |

## Triggering Review

Completion review is triggered when:
- A polecat marks a bead done and the branch is ready to merge
- A fast-path P0 bypass was used (mandatory within 48 hours)
- Mayor or Overseer requests a post-work review

Mayor (or Refinery) initiates the review:
```bash
bd mol pour mol-brains-trust-complete --var target=<bead-id>

# Dispatcher script (automates all passes and handoffs)
scripts/gt-trust-dispatch --mode complete --target <bead-id>
```

## Minimum Review

Unlike planning review (priority-tiered), completion review requires:
- **Minimum 1 pass from each model** (Claude, Codex, Gemini)
- No priority tiers - all completed work gets reviewed
- Passes are sequential (Claude -> Codex -> Gemini)

To reduce anchoring bias, each model writes its initial Investigate + Analyze
notes before reading prior passes, then proceeds with full context.

## Model Pass Structure

Each model pass follows this structure:

### 1. Investigate (Read Phase)
- Read the target bead (title, description, acceptance criteria)
- Review the plan file (`plans/<id>.md`) if it exists
- Inspect the code changes (diff, PR, or branch)
- Review test results or run relevant tests

### 2. Analyze (Think Phase)
- Does the implementation match the task definition?
- Are there correctness issues or regressions?
- Are edge cases handled?
- Is the code consistent with codebase patterns?
- Are tests and docs adequate?

### 3. Improve (Act Phase)
- File new beads for discovered bugs or missing work
- Add dependencies if fixes must block the merge
- Note any non-blocking improvements

### 4. Vote (Assess Phase)
Record assessment as one of:
- **APPROVE** - Implementation looks correct and complete
- **REVISE** - Issues found; fixes required before merge
- **BLOCK** - Critical flaw or high-risk regression
- **ABSTAIN** - Unable to review (record reason)

Append votes to the assessment bead using the dispatcher helper to avoid
overwriting notes:
```bash
scripts/gt-trust-dispatch append-note <assessment-id> \
  'BT_VOTE round=1 model=<claude|codex|gemini> vote=<APPROVE|REVISE|BLOCK|ABSTAIN> reason="<short>" timeout=<true|false> retry=<0|1>'
```

## Consensus Rules

After all three passes complete:

| Outcome | Condition | Action |
|---------|-----------|--------|
| **Close** | ≥2/3 APPROVE | Mark target trust-complete, close review |
| **Revise** | <2/3 APPROVE, no BLOCK majority | File gaps, reopen work |
| **Deadlock** | ≥1/3 BLOCK | Escalate to Overseer |

Votes are calculated over non-ABSTAIN votes.

## Protocol Flow

```
┌──────────────────────────────────────────────────────────────┐
│              Mayor/Refinery triggers completion review        │
└───────────────────────────────────────┬──────────────────────┘
                                        │
                                        ▼
                               ┌────────────────┐
                               │ Claude review  │
                               └──────┬─────────┘
                                      │
                                      ▼
                               ┌────────────────┐
                               │  Codex review  │
                               └──────┬─────────┘
                                      │
                                      ▼
                               ┌────────────────┐
                               │ Gemini review  │
                               └──────┬─────────┘
                                      │
                                      ▼
                          ┌──────────────────────────┐
                          │  Consensus + outcome     │
                          └────────────┬─────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
             close                 revise                deadlock
                 │                     │                     │
                 ▼                     ▼                     ▼
        trust-complete label    file gaps + reopen     escalate to Overseer
```

## Data Model

### Completion Assessment Bead

A tracking bead is created for each review:

```
Type: trust-assessment
Title: "Completion Review: <target-bead-title>"
Status: in_progress -> approved | rejected | escalated

Fields (in notes/description):
  target_bead: <bead-id>
  review_type: completion
  votes:
    - model: claude, vote: APPROVE, notes: "..."
    - model: codex, vote: REVISE, notes: "..."
    - model: gemini, vote: APPROVE, notes: "..."
  filed_gaps: <bead-ids if any>
```

### Filed Gaps

If reviewers find issues:
```bash
bd create --type=bug --title="Post-review: <issue>" \
  --description="Found during completion review of <original-bead>.

  Issue: <description>
  Location: <file:line if applicable>"

bd dep add <original-bead> <new-gap-bead>
```

## Integration Points

### Triggering Review

- Mayor triggers completion review when a polecat marks work done.
- Refinery may trigger review before merge if no review exists.

### Refinery Merge Gate

Refinery should only merge branches whose target bead has:
- The `trust-complete` label
- No open blocking dependencies

### Fast-Path Requirement

For fast-path P0 work (planning review bypass), completion review must
occur within 48 hours. If it does not, escalate to Overseer.

## Failure Modes

| Situation | Handling |
|----------|----------|
| Model timeout | Record ABSTAIN, continue with other models |
| All models ABSTAIN | Escalate to Overseer |
| Critical regression found | Vote BLOCK, file bug bead, halt merge |
| Tests missing/failed | Vote REVISE, require fixes |
| Conflicting conclusions | Escalate to Overseer |

## Example Session

**Target**: `sluice-xyz` - "Improve water shader bloom"

### Claude
**Vote**: APPROVE
> Implementation matches plan, tests green.

### Codex
**Vote**: REVISE
> Found missing test for shader fallback path. Filed `sluice-abc`.

### Gemini
**Vote**: APPROVE
> Looks correct, agrees with Codex on missing test.

### Result
APPROVE count = 2, REVISE count = 1
Approval rate = 67% (2/3)

**Outcome**: Close review, label `trust-complete`, keep `sluice-abc` as
follow-up dependency if required.

## Related

- [Brains Trust - Planning Review Protocol](./brains-trust-planning-review.md)
- `mol-brains-trust-plan` formula
- `mol-brains-trust-complete` formula
