# The Sluice Moose

A Rust mining game and a collection of fluid/granular simulation experiments.

## Unfinished checkpoint

The direct-control slice is work in progress. Validation stopped after a macOS WindowServer watchdog incident on 19 October 2026. The complete mining replay and performance with the original solvers have not been verified. This branch is a recovery checkpoint, not a validated playable release. Do not use the launch or verification commands below on the affected Mac while the incident is being investigated.

## First Bend controls and intended loop

Walk to the excavator, operate its joints to cut a finite paydirt bank, swing a loaded bucket over the hopper, place a sluice, wash the load and lift the mats. The first shift's goal is **5 grams of recovered gold**.

On this Mac, double-click **Play First Bend.command**, or run:

```bash
./scripts/play-mining.sh
```

Requires a native GPU, Cargo and Rust **1.88 or newer**. The launcher uses an installed nightly toolchain when available, otherwise stable. It builds the Bevy executable in the `playable` profile, overriding the older repository-selected compiler.

| Control | On foot | In the excavator |
|---|---|---|
| WASD | Walk | Drive / turn tracks |
| Right mouse drag | Look | Look / orbit |
| Shift | Run | — |
| E / Tab | Enter nearby cab | Tab exits |
| V | First / third person | Cab / external view |
| Left click | Shovel reachable ground | — |
| J | Pour hand bucket | Open and empty machine bucket |
| Q / E | — | Swing upper body |
| R / F | — | Raise / lower boom |
| T / G | — | Extend / tuck stick |
| Y / H | — | Curl / open bucket |
| Space | — | Cut soil at the bucket edge |
| I | — | Toggle engine |

At the plant: **B** previews a sluice. Point near the hopper outlet to snap its inlet; **R/Q** rotate, **[/]** or the wheel adjust slope, left-click confirms, Esc cancels. The preview rejects unsupported, overlapping, out-of-claim and disconnected placements. Walk to the plant and use **P** for the pump, **O** for feed, arrows to change water/feed rates, and **C** to lift the mats after stopping both.

**F1** controls, **F5** save, **F9** restore, **Esc** pause, **Ctrl+Q** quit. macOS saves to `~/Library/Application Support/Sluice Moose/first-bend.json`. `SLUICE_SAVE_PATH` overrides this for isolated tests. Invalid saves are rejected before changing the current shift.

## Scope and physics

- Bevy 0.17.3 supplies the native scene, input, PBR lighting, cascaded shadows, fog, UI and screenshots. Equipment, vegetation and the operator are procedural models; Barlow is bundled under the SIL Open Font License.
- `mining-sim` reuses `sim3d::World` for finite layered terrain and excavation. `mining-physics` calls the original GPU heightfield surface, flux, depth, settling, erosion, sediment-transport and collapse pipeline. Coarse-world gold is a conservative passive tag of the paydirt channel; density separation in the sluice uses the original particle solver. Carried loads, falling loads, hopper, live parcels, tailings and cleanup preserve the gold ledger.
- The excavator is a **constrained kinematic machine** with joint limits, terrain support, bucket capacity and simple obstacle checks. Hydraulic forces, rigid-body tracks, tipping and suspension are future work.
- The sluice calls the existing `game::gpu::flip_3d::GpuFlip3D` FLIP/APIC solver and `sim3d::ClusterSimulation3D` DEM contacts. The original geometry builder supplies collision solids and the visible sluice. Water particles, affine state, sediment motion, DEM contact history, terrain momentum and separate suspension channels are saved. Physics currently uses a separate wgpu 23 compute device while Bevy presents readback through wgpu 26. Weighted parcels are not a calibrated recovery predictor; realtime performance, bed packing, clogging and screening need further work.
- The scene is a 32 m surface claim on a 25 cm terrain grid. Caves, truck fleets, purchased assets, audio and a broader economy are outside this slice.

The existing applications remain available:

```bash
cargo +nightly run -p game --release
cargo +nightly run -p game --example detail_zone --release
```

## Verify

```bash
cargo +nightly test -p mining-sim --release
cargo +nightly test -p mining --bin mining
cargo +nightly run -p mining-physics --example operator_replay --profile playable
cargo +nightly fmt --package mining --package mining-sim --package mining-physics --check
```

The operator replay is intended to use the same domain actions and continuous joint axes as interactive play to dig two bucket loads, deliver them, walk around equipment, install the sluice, wash the material and collect the mats. It does not inject material or teleport equipment. Its complete run against the original GPU solvers is still pending. A targeted original-pipeline test passed before the incident, exercising 1,440 FLIP substeps, 120 terrain steps and live save/restore; this does not establish full-loop or performance acceptance.

Rendered verification opens its own native window:

```bash
SLUICE_SAVE_PATH=/tmp/first-bend-validation-save.json ./scripts/play-mining.sh \
  --validate /tmp/first-bend-validation --benchmark-seconds 60
```

It writes five screenshots, `operation.json` and `render-timing.json`, and restores a saved shift during processing. A failure writes `failure.txt` and returns an unsuccessful exit. The operation uses one simulation step per fixed update with four FLIP substeps; the benchmark runs at normal speed, excludes five seconds of warmup and reports wall-frame timing. `--capture /tmp/site.png` captures the initial scene and exits.
