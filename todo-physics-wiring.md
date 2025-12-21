# TODO – Wire Unified Gravity/Density into FLIP & PBF

## Problem
FLIP grid, FLIP sediment forces, Ferguson–Church settling, and PBF each hard-code their own gravity and water density values.

## Dependencies
- Depends on `SIM_GRAVITY` / `WATER_DENSITY` being defined in a physics/params module.

## Goals
- Ensure FLIP grid, FLIP sediment, Ferguson–Church settling, and PBF all share consistent gravity and fluid density (or explicitly documented deviations).

## Tasks
- [ ] Replace hard-coded gravity in:
  - [ ] `grid.rs::apply_gravity` (`GRAVITY = 400.0`) with `SIM_GRAVITY`.
  - [ ] `flip.rs::apply_sediment_forces` (`GRAVITY = 400.0`) with `SIM_GRAVITY`.
  - [ ] `pbf.rs` (`GRAVITY: Vec2 = (0, 400)`) to derive from `SIM_GRAVITY`.
- [ ] Reconcile gravity in Ferguson–Church settling:
  - [ ] `particle.rs::ParticleMaterial::settling_velocity` (`GRAVITY = 150.0`) should either:
    - Use `SIM_GRAVITY`, or
    - Use a clearly documented `EFFECTIVE_SETTLING_GRAVITY` tied back to `SIM_GRAVITY`.
- [ ] Replace per-module water density constants where appropriate:
  - [ ] Use shared `WATER_DENSITY` for sediment/particle settling calculations.
  - [ ] Leave legacy CA `WATER_DENSITY = 10.0` in `water.rs` untouched but documented as legacy (see separate CA TODO).
- [ ] Add tests or assertions that:
  - [ ] Gravity used in grid, sediment forces, and PBF are equal (or differ with explicit rationale).
  - [ ] Settling velocities scale correctly if `SIM_GRAVITY` is changed.
