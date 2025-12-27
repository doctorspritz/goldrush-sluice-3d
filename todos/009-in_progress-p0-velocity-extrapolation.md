# TODO: Implement Velocity Extrapolation

**Priority:** P0 (Critical - fixes week-long honey-water bug)
**Status:** IN_PROGRESS - Tests written, implementation pending

## Problem
Water flows like honey due to ~2% momentum loss per frame at fluid boundaries. Particles near air/solid cells sample undefined velocities, causing phantom FLIP deltas.

## Solution
Layered wavefront velocity extrapolation (standard FLIP technique used by Houdini).

## Current State

### DONE
1. Research complete - see `docs/research/velocity-extrapolation-research.md`
2. Test design complete - see `docs/research/velocity-extrapolation-test-design.md`
3. Implementation plan with detailed pseudocode - see `plans/velocity-extrapolation-implementation.md`
4. **10 uncheatable tests written** - `crates/sim/tests/velocity_extrapolation_tests.rs`
5. Added `rand_chacha` dev dependency to `crates/sim/Cargo.toml`

### TODO
1. Implement `grid.extrapolate_velocities(max_layers)` in `crates/sim/src/grid.rs`
2. Implement helper: `grid.total_momentum()` for tests
3. Implement `sim.run_isolated_flip_cycle_with_extrapolation(dt)` in `crates/sim/src/flip.rs`
4. Implement helper: `sim.total_particle_momentum()` for tests
5. Update `sim.update()` to call extrapolation at correct points
6. Run tests - they should pass
7. Run real game - verify water flows naturally

## Key Files

### Implementation
- `crates/sim/src/grid.rs` - Add extrapolate_velocities() and helpers
- `crates/sim/src/flip.rs` - Update simulation loop, add test helpers

### Reference
- `plans/velocity-extrapolation-implementation.md` - **DETAILED PSEUDOCODE HERE**
- `crates/sim/tests/velocity_extrapolation_tests.rs` - Tests define expected behavior

## Algorithm Summary

```
extrapolate_velocities(max_layers):
    1. mark_fluid_faces_known() - U/V faces adjacent to fluid cells
    2. for each layer:
       - extrapolate_u_layer() - average of known cardinal neighbors
       - extrapolate_v_layer() - average of known cardinal neighbors
```

## Simulation Loop Change

```rust
// BEFORE store_old_velocities:
self.particles_to_grid();
self.grid.extrapolate_velocities(1);  // ADD THIS
self.store_old_velocities();

// AFTER pressure gradient:
self.grid.apply_pressure_gradient(dt);
self.grid.extrapolate_velocities(1);  // ADD THIS
self.grid_to_particles(dt);
```

## Anti-Cheat Test Design

Tests use:
- Random seeds (can't tune for one case)
- Ratio-based thresholds (can't game absolute values)
- Physical invariants (conservation laws)
- Multiple configurations
- Performance measurement (wall-clock)

**DO NOT modify test thresholds. Fix the implementation.**

## Performance Requirement

- Target: 60fps (16.6ms per frame)
- Budget: <1ms per extrapolation call
- Called twice per frame = <2ms total

## Context for Next Session

The codebase currently has broken uncommitted changes from a failed "water-only" simplification attempt. The implementation should work on top of the current state or after restoring to last commit.

Key insight from research: Standard FLIP solvers extrapolate velocities BEFORE storing old grid state AND after pressure solve. This ensures particles near boundaries sample valid velocities for FLIP delta calculation.
