# TODO – Define Unified Physics Constants (Design & Units)

## Problem
There is no single, explicit definition of “simulation gravity” or core fluid units for the modern (FLIP/PBF) pipeline. Gravity and densities are redefined per module.

## Goals
- Specify a canonical gravity and water density for the **new** simulation stack.
- Make the relationship between these values and real-world units explicit.

## Tasks
- [ ] Design a `physics` or `params` module under `crates/sim`:
  - [ ] Define `pub const SIM_GRAVITY: f32` (canonical gravity for FLIP/PBF).
  - [ ] Define `pub const WATER_DENSITY: f32` (baseline fluid density for particle/sediment SG).
  - [ ] Optionally define a `pub struct PhysicsParams` for tunables (viscosity, drag, etc.).
- [ ] Document the intended unit system:
  - [ ] What one “world unit” (pixel) represents in meters (or note it’s abstract).
  - [ ] How `SIM_GRAVITY` relates to real 9.81 m/s² (e.g. “~X px/s² ~= 1 g”).
  - [ ] Distinguish “simulation convenience units” from physically meaningful ones when they differ.
- [ ] Add module-level docs summarizing which subsystems should use these constants (FLIP grid, sediment forces, PBF).
