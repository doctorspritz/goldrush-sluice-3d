# Sediment Deposition with DEM Settling

**Date**: 2025-12-28
**Status**: Working, minor polish needed
**Category**: Feature Implementation

## Problem

Sand particles in the FLIP simulation flow in vortices but never accumulate into permanent terrain. Need particles that settle to become "bedload" and change the SDF of the sluice.

## Solution

Two-phase approach:
1. **DEM settling**: Contact forces make particles stack into coherent piles
2. **Deposition**: Stable piles convert to solid terrain cells

### DEM Settling (`apply_dem_settling()` in flip.rs:1199-1385)

Particles near the floor with low velocity get discrete element contact forces:

```rust
// Thresholds
const SETTLING_VEL_THRESHOLD: f32 = 0.3;  // cells/frame
const SETTLING_DIST_CELLS: f32 = 2.0;     // SDF distance to floor
const CONTACT_STIFFNESS: f32 = 2000.0;    // Spring constant
const DAMPING_RATIO: f32 = 0.5;           // Critical damping fraction
const FRICTION_COEFF: f32 = 0.6;          // Coulomb friction
const PARTICLE_RADIUS_CELLS: f32 = 0.5;   // Contact radius
```

**Contact model**: Spring-damper with Coulomb friction
- Repulsion force: `F = k * overlap + c * relative_velocity`
- Friction: Tangential force capped at `μ * normal_force`
- Floor contact: Uses SDF gradient for normal direction

### Deposition (`deposit_settled_sediment()` in flip.rs:1390-1535)

Converts stable particle piles to solid terrain:

```rust
// Criteria for deposition
const VELOCITY_THRESHOLD: f32 = 0.05;  // Very low velocity
const FLOOR_DISTANCE_CELLS: f32 = 1.0; // Must be near floor/existing deposit
const MIN_NEIGHBORS: u16 = 3;          // Must be in packed state
const MASS_PER_CELL: usize = 4;        // Particles needed per cell
```

**Process**:
1. Identify settled particles (low vel, near floor, has neighbors)
2. Count particles per grid cell using atomics
3. Convert cells with ≥4 particles to solid
4. Remove particles in converted cells

### Grid Changes (grid.rs)

Added `deposited: Vec<bool>` field to distinguish deposited sediment from original terrain:

```rust
pub fn set_deposited(&mut self, i: usize, j: usize)  // Marks solid + deposited
pub fn is_deposited(&self, i: usize, j: usize) -> bool
```

### Visualization (main.rs)

Deposited cells render as golden color (180, 140, 60) on top of particles for visibility in all render modes.

## Results

- Particles settle into coherent piles at riffle bases
- Piles convert to solid terrain, changing SDF
- Water flow diverts around new deposits
- Visible in all render modes (Metaball, Mesh, FastRect, etc.)

## Known Issues

1. **Gaps in piles**: Some piles have holes where particles didn't pack tightly enough. May need:
   - Lower `MIN_NEIGHBORS` threshold
   - Smaller particle radius for tighter packing
   - Flood-fill to close gaps after deposition

## Files Modified

- `crates/sim/src/flip.rs` - DEM settling + deposition logic
- `crates/sim/src/grid.rs` - `deposited` field and methods
- `crates/game/src/main.rs` - Golden deposit visualization

## Next Phase: Entrainment

Deposited sediment should be re-entrained when flow velocity exceeds Shields parameter threshold. Research points:
- Shields parameter: τ* = τ / ((ρs - ρ) * g * d)
- Critical Shields ≈ 0.03-0.06 for sand
- Entrainment rate proportional to excess shear
