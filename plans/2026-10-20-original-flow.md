# Restore original demo flow

- [x] Compare original `game/src/main.rs` with playable adapter at `79f560f`.
- [x] Restore original 1.5 m by 0.40 m channel, 1 cm grid, riffles, slope, inlet footprint, outlet buffer, four substeps and slope-derived flow drive.
- [x] Preserve previous live saves with an explicit legacy hydraulic profile.
- [x] Verify original geometry, volume/gold conservation and save/restore with real Metal compute.
- [x] Regenerate warmed finite demo and inspect its particle surface in the packaged native app.
- [x] Run controls, targeted Rust checks, cleanup, commit and push PR #24.

The old renderer added procedural waves on a smoothed heightfield. Retain the requested three-dimensional particle surface; do not manufacture hydrodynamics with animated waves. Slope drive is restored for behavioural parity with the original demo, not asserted to be calibrated physical accuracy.
