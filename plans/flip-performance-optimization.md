# perf: FLIP Simulation Performance Optimization

## Overview

Performance review findings for the PIC/FLIP fluid simulation. Current performance is 7-18 FPS with 5000 particles - target is 60 FPS.

## Problem Statement

The simulation has multiple performance bottlenecks identified by code review:

1. **Per-frame memory allocations** in `particles_to_grid()` - 4 Vec allocations (~40KB/frame)
2. **Per-frame curl buffer allocation** in `apply_vorticity_confinement()`
3. **Dead code** - `resolve_solid_collision()` function is never called (78 LOC)
4. **Over-complex raycast** - DDA algorithm may be overkill for this use case

## Proposed Optimizations

### Optimization 1: Pre-allocate P2G Buffers

**Current Code** (`flip.rs:119-124`):
```rust
fn particles_to_grid(&mut self) {
    // ALLOCATES EVERY FRAME - 4 vectors!
    let mut u_sum = vec![0.0f32; self.grid.u.len()];
    let mut u_weight = vec![0.0f32; self.grid.u.len()];
    let mut v_sum = vec![0.0f32; self.grid.v.len()];
    let mut v_weight = vec![0.0f32; self.grid.v.len()];
    // ...
}
```

**Proposed Fix**: Move buffers to struct fields, clear with `.fill(0.0)` instead of reallocating.

```rust
pub struct FlipSimulation {
    // ... existing fields ...
    // Pre-allocated P2G transfer buffers
    u_sum: Vec<f32>,
    u_weight: Vec<f32>,
    v_sum: Vec<f32>,
    v_weight: Vec<f32>,
}
```

**Expected Impact**: 10-15% FPS improvement (eliminates ~40KB allocation per frame)

### Optimization 2: Pre-allocate Vorticity Curl Buffer

**Current Code** (`grid.rs:348-350`):
```rust
pub fn apply_vorticity_confinement(&mut self, dt: f32, strength: f32) {
    // ALLOCATES EVERY CALL
    let mut curl = vec![0.0f32; self.width * self.height];
    // ...
}
```

**Proposed Fix**: Add `curl` as a field on Grid struct.

**Expected Impact**: 5-8% FPS improvement

### Optimization 3: Delete Dead Code

**Current Code** (`flip.rs:407-486`):
```rust
/// Resolve collision with solid cells (free function to avoid borrow issues)
fn resolve_solid_collision(...) {
    // 78 lines - NEVER CALLED
}
```

This function was replaced by the raycast-based collision in `advect_particles()` but never deleted.

**Proposed Fix**: Delete the unused function.

**Expected Impact**: Code cleanliness, no runtime impact (but reduces confusion)

### Optimization 4: Simplify Raycast (Optional)

The DDA raycast is 100+ lines and may be over-engineered. Consider:
- Simple step-based collision (move in small increments)
- Only raycast for very fast particles

**Expected Impact**: Simpler code, possibly faster for short rays

## Implementation Order

1. Pre-allocate P2G buffers (highest impact, lowest risk)
2. Pre-allocate curl buffer
3. Delete dead code
4. (Optional) Simplify raycast if still needed

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim/src/flip.rs` | Add P2G buffer fields, modify `particles_to_grid()`, delete `resolve_solid_collision()` |
| `crates/sim/src/grid.rs` | Add `curl` field, modify `apply_vorticity_confinement()` |

## Acceptance Criteria

- [ ] No per-frame allocations in simulation hot path
- [ ] FPS improved to 30+ with 5000 particles
- [ ] No dead code remaining
- [ ] All existing tests still pass
- [ ] Visual quality unchanged

## Risks

- **Buffer size mismatch**: If grid resizes, pre-allocated buffers need resizing too
- **Memory footprint**: Pre-allocation uses more baseline memory (acceptable tradeoff)
