# TODO – Quarantine Legacy CA Gravity & Densities

## Problem
Legacy CA modules (`update.rs`, `water.rs`, `material.rs`) have their own gravity and density scales that differ from the modern FLIP/PBF stack, but this isn’t clearly called out.

## Goals
- Make it explicit that CA constants are on a separate unit system and must not be treated as canonical for the FLIP/PBF path.
- Reduce the risk of accidentally “fixing” or reusing CA constants when tuning the new pipeline.

## Tasks
- [ ] In each legacy CA module, group and label constants:
  - [ ] `update.rs`: `GRAVITY = 0.2`, `TERMINAL_VELOCITY = 5.0`, `EROSION_THRESHOLD = 1.0`, etc.
  - [ ] `water.rs`: `GRAVITY = 30.0`, `WATER_DENSITY = 10.0`, `DAMPING`, `MIN_FLOW`, etc.
  - [ ] `material.rs`: `density() -> u8`, `spread_rate()`, etc.
- [ ] Add a **prominent comment block** in each of these files:
  - [ ] Mark them as “LEGACY CA TUNING – uses its own units; do not mix with FLIP/PBF constants.”
- [ ] Ensure the new unified physics module is **not** imported by CA modules (to keep separation clear).
- [ ] Optionally, move CA-specific constants into a `legacy` submodule to emphasize their scope.
