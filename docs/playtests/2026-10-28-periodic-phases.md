# Periodic conservative water/mineral exchange

The experimental strip now has two-way, locally implicit water/mineral exchange coupled to its pressure projection. `PhaseTransfer3D::new_periodic_z`, `GpuG2p3D::new_periodic_with_metadata` and `GpuFlip3D::new_periodic_conservative` expose this explicitly. The game continues to use its existing mode. This advances the [water-only prototype](2026-10-28-periodic-water.md), while finite plant/bed ownership and periodic grain contacts remain unfinished.

The same backward-Euler phase model as the full-width validation mode forms water and mineral body-force predictors jointly, eliminates drag with pressure, and gathers the matching response. Full gravity remains in the predictors; pressure supplies buoyancy. Water receives the mineral reaction rather than acting as an infinite momentum reservoir. Existing density/diameter relaxation laws remain available; these controls do not calibrate recovery against real placer feed.

## Seam and obstruction ownership

APIC weights, offsets and voxel wall planes stay in their local unwrapped image. Storage addresses wrap only after real voxel reflection. This separates a repeated boundary from a physical obstruction that straddles it. Pure-water normalisation reads canonical face moments; W0 alone writes both mixture, pressure-mobility and auxiliary response seam copies, avoiding an in-place read/write race. Redundant allocated faces do not own extra particle moments.

Home markers, represented volume, blocked-volume retention, immutable voxel tags, carried pressure support and actual-water surface feedback use the same periodic topology. A real air gap across the seam suppresses underfilled surface suction; it is not treated as a missing closed wall. Closed uniform cell quadrature retains represented volume8 at every cell, including the seam.

The runtime preserves both phase mode and topology through reallocation. It checks active density inputs and refuses open-Z, unsupported sorted/gravel configurations and unqualified periodic CPU grain contacts. The water-only constructor retains its unit-marker restrictions; the conservative constructor can carry varying represented marker volumes. This provides operator support, not a finished full-width material/bed model.

## Checks

**41 distinct actual Metal tests and7 CPU phase-equation tests pass.** The six new Metal controls cover:

- Fifteen independent physical-face/image response and adjoint-gather cases at depths2/8/12: ordinary/stiff drag, gravity plus a manufactured mixture impulse, affine co-motion and exact/reserved capacities. Relative momentum and translational drag-energy bounds are checked; canonical and redundant response/mobility copies agree. A deliberately wrong redundant velocity is ignored by gathering.
- A real finite block, including open space beside its corner, translated through the seam at depths8/12. Represented volume remains in admissible cells, and translated grid/gather/affine responses match exactly.
- Closed mixed cell quadrature and complete carried-water pressure support, including the canonical W seam face.
- Four source controls with a water gap at the seam or shifted into the interior. Surface feedback follows actual water markers across the periodic boundary.
- Complete mixed gravity stepping through resource growth, plus exact complete position/velocity/affine continuation between resident and synchronous mixed frames, including growth. These are point/SDF particle controls and do not prove dense grain contact or packed-bed calibration.

Affected regressions retain the existing phase failure/conservation, water/pressure/transfer, wet-feed release, upload/storage and finite screen-to-sluice checks. All five mine scenarios pass, covering machine/excavation, soil and payload transport, ordinary/conservative stepping and live restore. CPU phase equations are separately checked against dense pressure/phase systems. Strict new-target lint and relevant formatting pass. Game lint retains98 existing warnings with zero primary findings on changed spans. No full-workspace gate is claimed.

Two complete150-tick saved-operation comparisons, ordinary and conservative, match the pre-change executable exactly after sorting only unique unordered contact-history keys and JSON object keys. No particle/cell order, floating-point value or signed zero is normalised. Canonical hashes are `9cae5a573699e2aa3ff573d235a4f21661999d6cbabd793590e82edfdf8a8dcd` and `bfd395b2760ee4b2e6a53dcc62fa3aa9dd08545233456e7f3161cfbe2874a08b`, respectively. The ordinary baseline also matches the preceding checkpoint. Four successful temporary states,115.932MB total, were deleted after comparison; hashes, executable identities and small reports remain. No executable was copied to preserve a baseline.

Reproduce the new GPU controls with `cargo +nightly test -p mining-physics --test periodic_phases --profile playable --locked --jobs 1 -- --ignored --test-threads=1 --nocapture`. Run the CPU phase checks with `cargo +nightly test -p mining-physics --lib phase_exchange::tests --profile playable --locked --jobs 1 -- --test-threads=1`.

## Remaining qualification

This stage does not establish periodic DEM contacts, dense packing, real fine-gold recovery, full-width ownership or lateral plant incidents. Periodic game-save selection, inlet/outlet world transfers and rendering are not integrated. No new coupled-strip performance/native/visual result is claimed: the preceding1.47/1.77x ratios describe the water-only benchmark, and cannot be assigned to this mixed solver.

Next qualify finite water/mineral/bed and contact ownership across the actual width, then graded washout, burial/loading/cleanout, off-centre feed, local blockage/diversion/spill and coupled12/8/full-width cost. Follow with saved/world transfers, rendering and sustained native combined operation. The [hybrid specification](../physics/hybrid-sluice.md) and every unfinished requirement in the [complete mining operation](../../plans/2026-10-22-realistic-operation.md) remain active.

Only one task-owned heavy job ran at once. Original local files, stash, installed app and input fixture hashes remain exact; no new crash report or app/executable copies appeared. No installation is claimed. The installed Beads CLI lacks `sync`; the owned issue is updated and262 unrelated JSONL rows remain literal before normal git synchronization.

[Source/check/retention receipt](2026-10-28-periodic-phases.json). Compact local evidence: `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/periodic-phases-20261028`.
