# TODO – Document & Test Sediment Physics Model

## Problem
The sediment model (Ferguson–Church settling, Richardson–Zaki hindered settling, Shields-based bedload) is implemented across `particle.rs` and `flip.rs` but not summarized in one place, and tests partially cover it.

## Goals
- Provide a concise, high-level description of the sediment model for future maintainers.
- Strengthen tests to lock in the intended behavior and ordering relationships.

## Tasks
- [ ] Write a brief design doc comment (can be in `flip.rs` or a separate `docs/` file) describing:
  - [ ] Ferguson–Church equation usage and how diameter/density/shape factor feed into it.
  - [ ] Richardson–Zaki hindered settling (including exponent and concentration mapping).
  - [ ] Suspended vs Bedload states and how Shields criterion drives transitions.
- [ ] Review and extend existing tests in `particle.rs` and `flip.rs`:
  - [ ] Assert settling order: Gold > Magnetite > Sand > Mud at equal diameters.
  - [ ] Assert hindered settling reduces velocity as concentration increases, but preserves ordering.
  - [ ] Assert Shields critical values stay in a realistic range and monotone with density.
- [ ] Add tests that tie into the new unified parameters:
  - [ ] Changing `SIM_GRAVITY` or `SedimentParams` values should shift results in predictable directions.
  - [ ] Verify that velocity clamps are not silently masking pathological values (e.g., test near-clamp conditions).
