---
status: complete
priority: p1
issue_id: "014"
tags: [code-review, cleanup, architecture]
dependencies: []
---

# Delete dfsph Crate (Unused Alternative Solver)

## Problem Statement

The `crates/dfsph/` directory contains an alternative CPU-based DFSPH (Divergence-Free SPH) solver that is **never used** in the production game. This crate adds ~543 LOC of unmaintained code that duplicates infrastructure from the `sim` crate.

## Findings

**Location:** `crates/dfsph/`

**Evidence:**
- Not in workspace default-members
- No references from `game` crate
- Duplicates particle physics from `sim` crate
- Has glam version mismatch (0.27 vs 0.24 in some places)
- Previous todo (#005) already suggested merging/removing this crate

**Contents:**
```
crates/dfsph/
  Cargo.toml (12 lines)
  src/
    lib.rs (89 lines)
    simulation.rs (454 lines)
```

## Proposed Solutions

### Option A: Delete Entire Crate (Recommended)
- **Pros:** Clean codebase, removes dead code and version conflicts
- **Cons:** Lose experimental DFSPH implementation
- **Effort:** 5 minutes
- **Risk:** Low (not used in production)

### Option B: Archive to Separate Branch
- **Pros:** Preserve work for future reference
- **Cons:** Still clutters main branch history
- **Effort:** 10 minutes
- **Risk:** Low

## Recommended Action

**Option A** - Delete the entire `crates/dfsph/` directory. If the DFSPH algorithm is needed in future, it can be reimplemented on GPU.

## Technical Details

- **Affected directory:** `crates/dfsph/`
- **Lines removed:** ~543 LOC
- **No production impact:** Crate is not used

## Acceptance Criteria

- [ ] `crates/dfsph/` directory deleted
- [ ] No references to dfsph in workspace Cargo.toml (if any)
- [ ] `cargo build` succeeds
- [ ] Related todo #005 can be closed

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2026-01-03 | Created | Found during GPU-only code audit |

## Resources

- Directory: `crates/dfsph/`
- Related todo: #005-pending-p2-merge-dfsph-into-sim-crate.md
