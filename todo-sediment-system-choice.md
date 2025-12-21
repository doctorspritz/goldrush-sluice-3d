# TODO – Choose Canonical Sediment System (Old Sediment vs FLIP Sediment)

## Problem
Two different sediment systems exist:
- Old drift-flux sediment in `sediment.rs` with `SedimentType`, Rubey settling, its own gravity and densities.
- Modern sediment based on `particle.rs::ParticleMaterial` + `flip.rs` (Ferguson–Church, Shields, bedload states).

## Goals
- Decide which sediment implementation is canonical moving forward.
- Either formally deprecate the other or make it a thin wrapper over the canonical one.

## Tasks
- [ ] Audit current usage:
  - [ ] Identify where `Sediment` / `SedimentType` from `sediment.rs` are still used.
  - [ ] Identify where FLIP particles (`ParticleMaterial`) + `FlipSimulation` are used in the game.
- [ ] Decide on the canonical sediment path:
  - [ ] Likely choose `ParticleMaterial` + `FlipSimulation` as the primary system.
- [ ] If `sediment.rs` is deprecated:
  - [ ] Add a module-level doc comment marking it as legacy.
  - [ ] Stop re-exporting it from `lib.rs` (or move it under a `legacy` namespace).
  - [ ] Plan for removal or archival once nothing depends on it.
- [ ] If `sediment.rs` must stay:
  - [ ] Implement conversions between `SedimentType` and `ParticleMaterial` (e.g., `From<SedimentType> for ParticleMaterial`).
  - [ ] Remove duplicated constants (SG, shape factor, etc.) from `SedimentType` and look them up via `ParticleMaterial` instead.
  - [ ] Update Rubey-based logic to reuse unified gravity and densities from the central physics module.
