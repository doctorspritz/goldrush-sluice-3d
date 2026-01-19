---
status: pending
priority: p2
issue_id: "033"
tags: [code-review, dead-code, cleanup]
dependencies: []
---

# Dead Code: Unused SPH/DFSPH Solvers (~2300 lines)

## Problem Statement

The project has documented that FLIP was chosen over DFSPH (see `docs/solutions/solver-choice-flip-vs-dfsph.md`), yet the SPH/DFSPH implementations remain in the codebase.

## Findings

**Unused files:**
- `crates/game/src/gpu/sph_3d.rs` - 1323 lines (IISPH solver)
- `crates/game/src/gpu/sph_dfsph.rs` - 946 lines (DFSPH solver)

**Total:** ~2300 lines of unused code

**Evidence of non-use:**
- Not referenced in main.rs
- Not used in any active examples
- FLIP is the documented chosen approach

## Proposed Solutions

### Option A: Delete both files (Recommended)
**Pros:** Reduces codebase size, maintenance burden
**Cons:** Lose reference implementations (but preserved in git history)
**Effort:** Trivial
**Risk:** None (git preserves history)

### Option B: Gate behind feature flag
**Pros:** Keep for reference/experimentation
**Cons:** Still needs maintenance
**Effort:** Small

## Acceptance Criteria

- [ ] Remove sph_3d.rs and sph_dfsph.rs
- [ ] Remove from mod.rs exports
- [ ] Verify no compile errors

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by code-simplicity-reviewer and git-history-analyzer agents |
