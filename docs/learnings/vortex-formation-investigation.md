# Vortex Formation Investigation

**Date**: 2024-12-21
**Branch**: `feat/viscosity-for-vortex-shedding`
**Status**: Unresolved - velocity vectors still predominantly downward, no visible vortices behind riffles

## Problem Statement

Despite having APIC (Affine Particle-In-Cell), suspension, settling, and higher resolution grid (256x192), we are not seeing swirling vortices forming behind riffles as expected for realistic sluice behavior. Velocity vectors appear to all point downward rather than showing the expected flow patterns.

## Root Cause Hypothesis

The simulation has **conflicting velocity calculations** - multiple systems modifying particle velocity in ways that fight each other, resulting in dominant downward motion with no preserved vorticity.

## What We Tried

### 1. Viscosity Implementation
**Rationale**: Inviscid flow cannot generate vortices naturally - boundary layer separation requires viscosity.

**Changes**:
- Added `use_viscosity` flag and `viscosity` parameter to `FlipSimulation`
- Implemented explicit Euler viscosity diffusion in `Grid::apply_viscosity()`
- Added keyboard controls (I toggle, O/P adjust viscosity)
- Added pre-allocated temp buffers to avoid per-frame allocation (fixed 7fps issue)

**Result**: No visible vortex formation. Viscosity alone doesn't create vortices if other forces dominate.

**Files**: `crates/sim/src/flip.rs`, `crates/sim/src/grid.rs`

---

### 2. Reduced Gravity
**Rationale**: GRAVITY=120 px/s² causes velocity to become predominantly downward within 0.5 seconds, overwhelming horizontal flow.

**Changes**:
- Reduced GRAVITY from 120 to 40, then to 80
- Reduced inlet velocity from (45, 25) to (30, 8) then (40, 15)

**Result**: Vectors still predominantly downward. The problem isn't just gravity magnitude.

**Files**: `crates/sim/src/grid.rs:382`, `crates/game/src/main.rs:121-122`

---

### 3. Wall-Tangent Velocity Projection (v1)
**Rationale**: Water should flow along the sluice floor instead of falling away.

**Changes**:
- Added `apply_wall_tangent_projection()` after G2P transfer
- For particles within 3 cell sizes of wall, project velocity onto tangent plane
- Used SDF gradient for wall normal, quadratic distance falloff

**Result**: Made flow worse - removed the driving force that makes water flow downhill.

**Files**: `crates/sim/src/flip.rs:498-554`

---

### 4. Wall-Tangent Velocity Projection (v2 - Improved)
**Rationale**: Don't project ALL velocity, only redirect INTO-wall velocity to tangential.

**Changes**:
- Only act when v·n < 0 (velocity going into wall)
- Remove 80% of into-wall component
- Add 50% back as tangential (downhill) flow
- Choose downhill tangent direction based on which way has +y component

**Result**: Still no improvement. Vectors still pointing mostly downward.

**Files**: `crates/sim/src/flip.rs:498-554`

---

### 5. Disabled Spray Dampening
**Rationale**: Spray dampening (`velocity.y *= 0.7` when y < -40) kills upward velocity needed for vortex recirculation.

**Changes**:
- Commented out spray dampening in G2P

**Result**: No visible improvement to vortex formation.

**Files**: `crates/sim/src/flip.rs:484-487`

---

## Velocity Modification Points Identified

Particle velocity is modified in **many places** each frame, potentially conflicting:

| Location | What it does | When |
|----------|-------------|------|
| `grid_to_particles` (G2P) | Sets velocity from grid | Always |
| ~~Spray dampening~~ | `vy *= 0.7` if vy < -40 | Disabled |
| `apply_wall_tangent_projection` | Redirects into-wall velocity | Near walls |
| `apply_sediment_forces` | Lerp toward fluid+settling velocity | Sediment only |
| `advect_particles` SDF collision | Removes into-solid component | Near solids |
| `advect_particles` is_solid safety | Pushes out, removes into-solid | In solid cells |
| `advect_particles` bounds | Zeros velocity at edges | At boundaries |
| `push_particles_apart` | Position corrections only | Particle overlap |

## Observations

1. **All velocity vectors point downward** - suggests horizontal velocity is being lost somewhere
2. **Flow should be laminar in non-turbulent areas** - currently looks chaotic/jittery
3. **Water doesn't follow the slope** - particles fall rather than slide along surface
4. **Riffles don't cause flow separation** - no recirculation zones visible

## Untried Approaches

### A. Grid-Level Slope Gravity
Instead of modifying particle velocity, modify how gravity is applied to the grid:
- Near solid cells, apply gravity in the wall-tangent direction
- This would make the entire velocity field slope-following

### B. Remove Conflicting Handlers
Simplify by removing redundant collision handlers:
- Keep only SDF-based collision in advect
- Remove is_solid safety net (if SDF is working)
- Remove wall-tangent projection (let collision handle it)

### C. Increase Vorticity Confinement
Current settings are very conservative (ε=0.05, every 2 frames):
- Try ε=0.2 or higher
- Run every frame
- May need to amplify any vorticity that does form

### D. Check Pressure Solve
The pressure projection removes divergence but might also remove vorticity:
- Try fewer iterations (currently 15)
- Check if pressure gradient is over-correcting

### E. FLIP vs PIC Blend
Currently using pure PIC velocity assignment in G2P:
- Try FLIP blend: `velocity = flip_ratio * (old + delta) + (1-flip_ratio) * new`
- FLIP preserves more detail but can be noisy

### F. Boundary Layer Resolution
Vortex shedding requires resolving the boundary layer:
- May need even higher resolution near riffles
- Or adaptive refinement near obstacles

### G. Reynolds Number Check
Vortex shedding requires Re > 60:
- Calculate actual Re for current flow conditions
- May need to adjust viscosity/velocity/length scale

## Current Parameter State

```rust
// grid.rs
const GRAVITY: f32 = 80.0;

// main.rs
inlet_vx: 40.0
inlet_vy: 15.0
spawn_rate: 1
slope: 0.20
riffle_spacing: 50
riffle_height: 8
riffle_width: 6

// flip.rs
use_viscosity: false (toggleable with I)
viscosity: 1.0
vorticity_confinement: ε=0.05, every 2 frames
pressure_iterations: 15
```

## Next Steps

1. Try approach A (grid-level slope gravity) or B (simplify handlers)
2. Add more diagnostic visualization (vorticity field heatmap?)
3. Consider if the fundamental FLIP approach is correct for this use case
4. Look at reference implementations of sluice/channel flow simulation
