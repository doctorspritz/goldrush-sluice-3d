# TODO: Implement Velocity Extrapolation

**Priority:** P0 (Critical - fixes week-long honey-water bug)
**Status:** RESOLVED (2026-01-19)

## Resolution

**GPU implementation is complete and working.** The "honey-water" behavior was fixed by the GPU velocity extrapolation in `crates/game/src/gpu/shaders/velocity_extrapolate_3d.wgsl`.

### Evidence
- 7/7 GPU FLIP component tests pass (including hydrostatic equilibrium)
- Main application uses GPU path exclusively (`gpu_flip.step*`)
- 4-pass layered wavefront extrapolation implemented in WGSL shader
- Called in `flip_3d.rs:4230+` after pressure solve, before G2P

### What Was Obsolete
- This TODO referenced `crates/sim/` which no longer exists (replaced by `sim3d`, now archived)
- The tests mentioned (`crates/sim/tests/velocity_extrapolation_tests.rs`) were for the old CPU path
- The CPU path (`sim3d::FlipSimulation3D`) was never used by main application

### Note on CPU Path
- `sim3d::FlipSimulation3D` still exists but lacks velocity extrapolation
- Main app uses GPU path exclusively - CPU `update()` is never called
- sim3d crate retained for shared types (World, ClusterSimulation3D, test utilities, etc.)

---

## Original Problem (For Reference)

Water flows like honey due to ~2% momentum loss per frame at fluid boundaries. Particles near air/solid cells sample undefined velocities, causing phantom FLIP deltas.

## Solution Implemented (GPU)

Layered wavefront velocity extrapolation in `velocity_extrapolate_3d.wgsl`:
1. `init_valid_faces()` - Mark faces adjacent to FLUID cells as valid
2. 4-pass loop:
   - `extrapolate_u()` - Average from valid neighbors
   - `extrapolate_v()` - Average from valid neighbors
   - `extrapolate_w()` - Average from valid neighbors
   - `finalize_valid()` - Promote newly-extrapolated faces

Called once after pressure solve, before G2P (line 4230+ in flip_3d.rs).
