# TODO – DRY Sediment & Material Properties and Parameters

## Problem
Sand/magnetite/gold properties (density, shape factor, friction, Shields critical, etc.) are defined in more than one place, and many simulation knobs for sediment live as magic numbers inside `flip.rs`.

## Dependencies
- Canonical sediment system chosen (or at least decided which enum owns the data).

## Goals
- Have a **single source of truth** for sediment material properties.
- Move key sediment tuning knobs into a structured parameter config.

## Tasks
- [ ] Consolidate material properties:
  - [ ] Ensure density, shape factor, friction, Shields critical, and typical diameters for sand/magnetite/gold are defined in **one** enum (likely `ParticleMaterial`).
  - [ ] For any other enums (`SedimentType`), have them delegate to that enum instead of duplicating values.
- [ ] Extract sediment tuning parameters from `flip.rs`:
  - [ ] `BASE_DRAG_RATE`, `REST_NEIGHBORS`, `MIN_WATER_NEIGHBORS`.
  - [ ] `VELOCITY_SETTLE_THRESHOLD`, `SHIELDS_HYSTERESIS`, `SHEAR_VELOCITY_FACTOR`.
  - [ ] Velocity clamps like `MAX_VELOCITY` in sediment sections.
- [ ] Introduce a `SedimentParams` struct (or embed into `PhysicsParams`):
  - [ ] Store an instance on `FlipSimulation`, with sensible defaults.
  - [ ] Replace hard-coded `const`s inside `flip.rs` with fields from this struct.
- [ ] Add doc comments to `SedimentParams` explaining:
  - [ ] Physical meaning of each parameter.
  - [ ] Expected ranges and qualitative effect when tuning.
