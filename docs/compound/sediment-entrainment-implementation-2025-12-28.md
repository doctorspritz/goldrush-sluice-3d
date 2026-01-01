# Sediment Entrainment: Completing the Transport Cycle

---
title: "Sediment Entrainment: Completing the Transport Cycle"
category: feature-implementation
component: flip.rs, grid.rs
status: solved
date: 2025-12-28
tags:
  - flip
  - sediment
  - entrainment
  - deposition
  - erosion
related_files:
  - crates/sim/src/flip.rs
  - crates/sim/src/grid.rs
  - plans/sediment-entrainment-implementation.md
  - docs/compound/sediment-deposition-dem-2025-12-28.md
---

## Problem

Deposited sediment was permanent. Once particles settled and converted to solid terrain via `deposit_settled_sediment()`, they could never return to the flow. This created an incomplete sediment transport cycle.

**Missing phase**: suspension → settling → deposition → **entrainment** → suspension

## Solution

Implemented `entrain_deposited_sediment()` function that:
1. Samples velocity above each deposited cell
2. Compares to critical threshold (0.5 cells/frame)
3. Probabilistically removes cells exceeding threshold
4. Spawns 4 sand particles per eroded cell with flow velocity
5. Updates SDF after terrain changes

### Key Code

**grid.rs:419-425** - Clear deposited status:
```rust
pub fn clear_deposited(&mut self, i: usize, j: usize) {
    if i < self.width && j < self.height {
        let idx = self.cell_index(i, j);
        self.solid[idx] = false;
        self.deposited[idx] = false;
    }
}
```

**flip.rs:1553-1663** - Core entrainment function:
```rust
fn entrain_deposited_sediment(&mut self, dt: f32) {
    const CRITICAL_VELOCITY: f32 = 0.5;  // cells/frame - any flow erodes
    const BASE_ENTRAINMENT_RATE: f32 = 1.0;
    const MAX_PROBABILITY: f32 = 0.95;
    const PARTICLES_PER_CELL: usize = 4;  // matches MASS_PER_CELL

    // For each deposited cell...
    // Sample velocity just above: (i+0.5, j-0.5) * cell_size
    // If speed > threshold: probabilistic erosion
    // Spawn particles, clear cell, update SDF
}
```

**flip.rs:233-235** - Integration in update loop:
```rust
self.deposit_settled_sediment(dt);      // Step 8f
self.entrain_deposited_sediment(dt);    // Step 8g - NEW
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Velocity sampling | Half cell above deposit | Flow velocity determines erosion; deposit itself is solid |
| Threshold method | Simple velocity (0.5 cells/frame) | Shields parameter too complex for games |
| Probability function | Linear with excess, capped at 95% | Smooth erosion, prevents sudden mass loss |
| Particles per cell | 4 | Mass conservation (matches deposition) |
| Execution order | After deposition (step 8g) | Prevents same-frame oscillation |

## Tuning Journey

Started conservative, tuned based on visual feedback:

| Parameter | Initial | Final | Change |
|-----------|---------|-------|--------|
| `CRITICAL_VELOCITY` | 15.0 | 0.5 | 30x lower for responsive erosion |
| `BASE_ENTRAINMENT_RATE` | 0.1 | 1.0 | 10x higher |
| `MAX_PROBABILITY` | 0.3 | 0.95 | Nearly instant when threshold exceeded |

## Key Learnings

### 1. Symmetric Operations for Mass Conservation
- Deposition: 4 particles → 1 cell
- Entrainment: 1 cell → 4 particles
- Asymmetry causes gradual mass drift

### 2. Support Checking Prevents Impossible States
```rust
// Don't entrain if it would leave floating cells above
let has_deposit_above = j > 0 && self.grid.is_deposited(i, j - 1);
if has_deposit_above {
    let left_support = i > 0 && self.grid.is_solid(i - 1, j - 1);
    let right_support = i < width - 1 && self.grid.is_solid(i + 1, j - 1);
    if !left_support && !right_support {
        continue;  // Would create floating deposit
    }
}
```

### 3. Order of Operations Matters
- Deposition BEFORE entrainment in each frame
- Prevents same-frame oscillation cycles

### 4. Probabilistic Systems Look More Natural
- Hard thresholds create unnatural sudden changes
- Stochastic erosion with probability cap creates gradual, realistic behavior

### 5. Start Conservative, Tune Visually
- Research gives starting points (Shields parameter ≈ 0.045)
- Gameplay defines final values (0.5 cells/frame worked best)
- Visual feedback essential for tuning

## Edge Cases Handled

| Edge Case | Mitigation |
|-----------|------------|
| Oscillation | Order: entrain AFTER deposit |
| Floating deposits | Support check for lateral support |
| Mass loss | Spawn exactly 4 particles |
| Sudden collapse | MAX_PROBABILITY caps erosion rate |
| SDF consistency | Recompute after terrain changes |

## Related Documentation

- `docs/compound/sediment-deposition-dem-2025-12-28.md` - Prerequisite deposition system
- `plans/sediment-entrainment-implementation.md` - Full implementation plan
- `plans/sediment-entrainment.md` - Physics background (Shields parameter)
- `todos/003-pending-p3-pile-gaps.md` - Related polish item

## Files Modified

| File | Changes |
|------|---------|
| `crates/sim/src/grid.rs:419-425` | Added `clear_deposited()` |
| `crates/sim/src/flip.rs:1553-1663` | Added `entrain_deposited_sediment()` |
| `crates/sim/src/flip.rs:233-235` | Integration call in update loop |

## Commits

- `1063c26` feat(sediment): add entrainment - erode deposited cells with flow
- `66391e0` chore: remove obsolete particle-based entrainment test
