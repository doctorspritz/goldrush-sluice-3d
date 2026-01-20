---
status: resolved
priority: p2
issue_id: "010"
tags: [code-review, simplification, dfsph]
dependencies: []
---

# Simplification: Remove Dead Vorticity Code

## Problem Statement

DFSPH allocates memory and iterates over particles for vorticity calculation, but the actual computation is **commented out** as "skipped for compilation simplicity". This is dead code that wastes memory and cycles.

## Findings

**Locations:**
- `simulation.rs:15` - Unused `VORTICITY_EPSILON` constant
- `simulation.rs:26` - `vorticities: Vec<f32>` field allocation
- `simulation.rs:57` - Pre-allocation with capacity
- `simulation.rs:127` - Resize in sync function
- `simulation.rs:148` - Unused variable declaration
- `simulation.rs:420-426` - Dead iteration loop

**Dead Code:**
```rust
// Line 15
const VORTICITY_EPSILON: f32 = 30.0;  // Never used

// Lines 420-426
particles_slice.par_iter_mut()
    .zip(vorticities_slice.par_iter_mut())
    .zip(old_pos_slice.par_iter())
    .for_each(|((p, vort), old_pos)| {
        p.velocity = (p.position - *old_pos) / dt;

        let mut curl = 0.0;  // WARNING: unused mut
        // Vorticity calc skipped for compilation simplicity (vars unused)
        *vort = curl;  // Always 0.0
    });
```

**Cost:**
- 20,000 × 4 bytes = **80KB wasted memory** for unused Vec
- Parallel iteration overhead for no-op assignment
- Code complexity from tracking unused field

## Proposed Solutions

### Option A: Delete all vorticity code (Recommended)
- **Pros:** Clean, no dead code, saves memory
- **Cons:** Need to re-add if vorticity needed later
- **Effort:** 15 minutes
- **Risk:** Low

**Lines to Delete:**
- Line 15: `const VORTICITY_EPSILON`
- Line 26: `pub vorticities: Vec<f32>`
- Line 57: Capacity allocation
- Line 127: Resize in sync
- Line 148: Variable declaration
- Lines 420-426: Dead iteration

### Option B: Implement vorticity properly
- **Pros:** Feature complete
- **Cons:** More work, may not be needed
- **Effort:** 2-3 hours
- **Risk:** Medium

### Option C: Gate behind feature flag
- **Pros:** Keeps code for future use
- **Cons:** Complexity for unused feature
- **Effort:** 30 minutes
- **Risk:** Low

## Recommended Action

**Option A** - Delete dead vorticity code. Can be re-added from git history if needed later.

## Technical Details

**Affected File:** `crates/dfsph/src/simulation.rs`

**Lines to Delete:** 15, 26, 57, 127, 148, 420-426

**Net Reduction:** ~15 lines

## Acceptance Criteria

- [x] `VORTICITY_EPSILON` constant deleted
- [x] `vorticities` field removed from struct
- [x] No vorticity-related code in `update()`
- [x] All tests pass
- [x] Memory footprint reduced by 80KB at 20k particles

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Code simplicity review |
| 2025-12-22 | Resolved | Removed all vorticity code: constant, struct field, allocations, and sync calls. Package compiles successfully. |

## Resources

- Code simplicity reviewer analysis
- YAGNI principle
