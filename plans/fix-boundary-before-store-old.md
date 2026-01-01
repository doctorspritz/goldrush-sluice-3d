# Plan: Fix Boundary Enforcement Timing

**Date:** 2025-12-27
**Status:** PENDING REVIEW
**Category:** bug-fix, physics, momentum-loss

## Problem Statement

Water flows like honey due to ~2% momentum loss per frame. Root cause identified: asymmetric boundary enforcement in FLIP cycle.

## Root Cause Analysis

Current order in `flip.rs:update()`:

```
1. P2G (particles_to_grid)           → Wall faces get non-zero velocity
2. store_old_velocities              → Samples BEFORE boundary enforcement
3. apply_gravity
4. vorticity_confinement
5. enforce_boundary_conditions       → Zeroes wall faces
6. pressure solve
7. apply_pressure_gradient
8. G2P (grid_to_particles)           → Samples AFTER boundary enforcement
```

The FLIP delta is: `grid_delta = new_velocity - old_grid_velocity`

For particles near walls:
- `old_grid_velocity` includes non-zero wall face contributions (sampled before zeroing)
- `new_velocity` has zero wall face contributions (sampled after zeroing)
- **Result:** Artificial negative delta at boundaries → momentum subtracted

## Evidence

1. Isolated FLIP cycle test (skips forces and boundaries) shows 0.11% loss - acceptable
2. Full simulation shows ~2% loss - the difference is the forces + boundary enforcement
3. Previous documentation notes 62% grid momentum removal by boundary enforcement

## Proposed Fix

**Single change:** Add `enforce_boundary_conditions()` call immediately after P2G, before `store_old_velocities()`.

New order:
```
1. P2G (particles_to_grid)
2. enforce_boundary_conditions       ← ADD HERE (first call)
3. store_old_velocities              → Now samples AFTER boundary enforcement
4. apply_gravity
5. vorticity_confinement
6. enforce_boundary_conditions       → Keep existing (second call, after forces)
7. pressure solve
8. apply_pressure_gradient
9. G2P
```

## Code Change

**File:** `crates/sim/src/flip.rs`
**Location:** Lines 140-143

**Before:**
```rust
// 2. Transfer particle velocities to grid (P2G)
self.particles_to_grid();

// 3. Store old grid velocities for FLIP blending
self.store_old_velocities();
```

**After:**
```rust
// 2. Transfer particle velocities to grid (P2G)
self.particles_to_grid();

// 2b. Zero wall velocities BEFORE storing old grid state
// Without this, store_old captures non-zero wall velocities that get
// zeroed later, causing phantom negative deltas in the FLIP update
self.grid.enforce_boundary_conditions();

// 3. Store old grid velocities for FLIP blending
self.store_old_velocities();
```

## What This Does NOT Touch

- FLIP_RATIO (0.97) - unchanged
- Pressure solver iterations (15) - unchanged
- Vorticity confinement (0.05) - unchanged
- Any damping constants - unchanged
- Sediment physics - unchanged
- Collision handling - unchanged

## Verification Plan

1. Run isolated FLIP cycle test - should still show ~0.1% loss
2. Run full simulation - should show significantly less loss
3. Visual check - water should flow more freely, less honey-like

## Risk Assessment

**Low risk:**
- This is adding a call that already exists elsewhere
- It only affects the timing of when walls are zeroed
- Walls should be zero anyway - this just makes the FLIP delta calculation correct

## Alternative Considered

Could also fix by changing `store_old_velocities()` to skip wall cells, but:
- More complex change
- Requires modifying the kernel sampling
- Current fix is simpler and matches standard FLIP implementations
