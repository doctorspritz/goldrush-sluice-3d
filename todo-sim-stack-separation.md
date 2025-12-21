# TODO – Separate Legacy CA Stack from FLIP/PBF in API & Layout

## Problem
Legacy CA modules and the new FLIP/PBF modules coexist under `crates/sim`, and `lib.rs` re-exports both. It’s not obvious which path is canonical, and some CA modules are still wired into `World`.

## Goals
- Make the separation between “old CA stack” and “new FLIP/PBF stack” clear in:
  - Module layout
  - Public API surface
  - Documentation
- Enable future removal of the CA stack without touching FLIP/PBF.

## Tasks
- [ ] Update `lib.rs` documentation:
  - [ ] Clearly label old CA modules as “legacy” and new FLIP/PBF modules as the primary path.
- [ ] Re-export strategy:
  - [ ] Consider removing direct re-exports of legacy CA types from the top-level, or
  - [ ] Move them under a `legacy` or `ca` namespace to make usage explicit.
- [ ] Directory/layout changes (non-breaking if possible):
  - [ ] Move CA modules (`world.rs`, `update.rs`, `water.rs`, `material.rs`, etc.) into a `legacy` submodule.
  - [ ] Keep FLIP/PBF-related modules (`flip.rs`, `grid.rs`, `particle.rs`, `sluice.rs`, `pbf.rs`) at the top of the crate as the main API.
- [ ] Runtime wiring:
  - [ ] Audit where `World` and `update_chunk` are still used.
  - [ ] Decide whether a FLIP-based world/integration layer is needed to replace CA world logic.
- [ ] Add a short section to the project README (or crate docs) that explains:
  - [ ] The difference between the CA and FLIP/PBF stacks.
  - [ ] Which one the game uses today.
  - [ ] Where new features should be implemented.
