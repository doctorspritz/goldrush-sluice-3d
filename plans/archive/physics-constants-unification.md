# Physics Constants Unification Plan

## Problem
Multiple gravity and density constants scattered across modules create regression risk.

## Current State (after CA cleanup)

### GRAVITY constants
| File | Line | Value | Usage |
|------|------|-------|-------|
| grid.rs | 383 | 400.0 | Grid velocity update |
| flip.rs | 555 | 400.0 | Sediment buoyancy |
| flip.rs | 729 | 400.0 | Sediment settling |
| pbf.rs | 5 | Vec2(0, 400) | PBF external force |
| **particle.rs** | **189** | **150.0** | **Ferguson-Church settling velocity** |

### DENSITY constants
| File | Line | Value | Usage |
|------|------|-------|-------|
| particle.rs | 188 | WATER_DENSITY = 1.0 | Reference density for SG |
| flip.rs | 963 | REST_DENSITY = 0.8 | Clavet near-pressure target |
| pbf.rs | 7 | REST_DENSITY = 0.05 | PBF constraint (SPH kernel scale) |

## Key Insight
The density values are *intentionally different* - they're tuned for different algorithms:
- `WATER_DENSITY = 1.0` is physical (reference for specific gravity)
- Clavet `REST_DENSITY = 0.8` is algorithmic (particles per kernel)
- PBF `REST_DENSITY = 0.05` is algorithmic (different kernel scale)

The gravity values *should* be the same, except Ferguson-Church uses a lower value (150.0 vs 400.0) which may be intentional tuning or a bug.

## Status: COMPLETED

## Tasks

### Phase 1: Create unified physics module
- [x] Create `crates/sim/src/physics.rs` with:
  ```rust
  /// Simulation gravity (pixels/s²)
  pub const GRAVITY: f32 = 400.0;

  /// Water density (dimensionless, reference for specific gravity)
  pub const WATER_DENSITY: f32 = 1.0;
  ```
- [x] Add module to lib.rs

### Phase 2: Add regression tests BEFORE changing constants
- [x] Test settling velocity ordering: Gold > Magnetite > Sand > Mud (existing tests pass)
- [x] Record current settling velocities for each material (tests cover this)
- [x] Test particle falls under gravity at expected rate

### Phase 3: Wire unified GRAVITY
- [x] grid.rs: replace local const with `physics::GRAVITY`
- [x] flip.rs: replace all local GRAVITY/AIR_GRAVITY/SIMPLE_GRAVITY with `physics::GRAVITY`
- [x] pbf.rs: replace with `Vec2::new(0.0, physics::GRAVITY)`
- [x] particle.rs (Ferguson-Church): unified to `physics::GRAVITY` with adjusted KINEMATIC_VISCOSITY

### Phase 4: Document algorithmic densities
- [x] physics.rs documents WATER_DENSITY and KINEMATIC_VISCOSITY
- Note: Clavet/PBF REST_DENSITY left as-is (intentionally different scales)

### Phase 5: Delete old TODO files
- [x] Removed all 7 todo-*.md files
- [x] Deleted 3,095 lines of dead CA code (cell, chunk, fluid, material, update, water, world, sediment)

## Success Criteria
- Single source of truth for GRAVITY
- Tests prevent accidental regression in settling behavior
- Comments explain why densities differ between algorithms
