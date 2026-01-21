# Brains Trust - Completion Review Protocol

> Multi-model review after work is marked done.

## Status: PLACEHOLDER

This protocol is defined by bead `hq-gerz` and depends on the planning review
protocol (`hq-1nh8`) being completed first.

## Overview

The completion review is the POST-work counterpart to the planning review.
After a polecat marks work complete, the Brains Trust verifies the
implementation before it's merged.

## Key Differences from Planning Review

| Aspect | Planning Review | Completion Review |
|--------|-----------------|-------------------|
| **When** | Before work begins | After work is done |
| **Focus** | Task definition | Implementation quality |
| **Checks** | Missing deps, ambiguities | Mistakes, missing pieces |
| **Output** | Trust-approved label | Close or file gaps |

## Minimum Review

Unlike planning review (priority-tiered), completion review requires:
- **Minimum 1 pass from each model** (Claude, Codex, Gemini)
- No priority tiers - all completed work gets reviewed
All passes are sequential (Claude -> Codex -> Gemini) using pass-the-parcel
handoff; no parallel reviews.

## What Models Check

1. Does the implementation match the task definition?
2. Are there obvious bugs or regressions?
3. Are there missing edge cases?
4. Is the code consistent with codebase patterns?
5. Are tests adequate?

## Consensus

Same rules as planning:
- 2/3 consensus to close
- Escalate to overseer on conflict

## Filed Gaps

If reviewers find issues:
```bash
bd create --type=bug --title="Post-review: <issue>" \
  --description="Found during completion review of <original-bead>.

  Issue: <description>
  Location: <file:line if applicable>"
```

## TODO

- [ ] Define full protocol spec (see planning review for template)
- [ ] Create `mol-brains-trust-complete` formula
- [ ] Define integration with refinery merge flow

---

*This is a placeholder. Full implementation tracked by bead `hq-gerz`.*
