# The Sluice Moose

A Rust mining game and a collection of fluid/granular simulation experiments.

## First Bend prototype

The direct-control slice now completes a verified first shift with the original solvers. Development resumed on 20 October after the previous recovery checkpoint. Builds use one compiler job by default; the game uses one bounded simulation worker and limited Bevy task pools. Operation, native timing and limitations are recorded in [the playtest report](docs/playtests/2026-10-20.md).

## First Bend controls and loop

Walk to the excavator, operate its joints to cut a finite paydirt bank, swing a loaded bucket over the hopper, place a sluice, wash the load and lift the mats. The first shift's goal is **5 grams of recovered gold**.

On this Mac, double-click **Play First Bend.command**, or run:

```bash
./scripts/play-mining.sh
```

Requires a native GPU, Cargo and Rust **1.88 or newer**. The launcher uses an installed nightly toolchain when available, otherwise stable. It builds the Bevy executable in the `playable` profile with one compiler job. `SLUICE_BUILD_JOBS` overrides that limit.

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
- The sluice calls the existing `game::gpu::flip_3d::GpuFlip3D` FLIP/APIC solver and `sim3d::ClusterSimulation3D` DEM contacts. The original geometry builder supplies collision solids and the visible sluice. Water particles, affine state, sediment motion, DEM contact history, terrain momentum and separate suspension channels are saved. Physics uses a separate wgpu 23 compute device on one worker while Bevy presents immutable snapshots through wgpu 26. The worker allows one active input packet and one pending packet, with no burst of overdue ticks. Its two FLIP substeps keep advective CFL below 0.5 at the original solver’s 1 m/s clamp. Sediment drag and buoyancy use an exact relaxation step, so prescribed settling speeds do not change with substep count; CPU DEM owns contact friction. Weighted parcels are not a calibrated recovery predictor; realtime performance, bed packing, clogging and screening need further work.
- The scene is a 32 m surface claim on a 25 cm terrain grid. Caves, truck fleets, purchased assets, audio and a broader economy are outside this slice.

The existing applications remain available:

```bash
cargo +nightly run -p game --release
cargo +nightly run -p game --example detail_zone --release
```

## Verify

```bash
cargo +nightly test -p mining -p mining-sim -p mining-physics --profile playable --jobs 1 --locked
cargo +nightly clippy -p mining -p mining-physics -p mining-sim --all-targets --jobs 1 --no-deps -- -D warnings
cargo +nightly run -p mining-physics --example operator_replay --profile playable --jobs 1
cargo +nightly fmt --package mining --package mining-sim --package mining-physics --check
```

The operator replay uses the same domain actions and continuous joint axes as interactive play to dig two bucket loads, deliver them, walk around equipment, install the sluice, wash the material and collect the mats. It does not inject material or teleport equipment. The native replay completed on 20 October with 0.560 m³ excavated and processed, 8.208 g recovered, 0.400 m³ discharged as tailings and live save/restore. Rendering during washing averaged 57.5 FPS at 1440×900 on the M2 Max; simulation advanced at 0.39× real time under sustained plant load. See [playtest evidence](docs/playtests/2026-10-20.md). This is a playable prototype; real-time physics throughput and recovery calibration remain open work.

Rendered verification opens its own native window:

```bash
SLUICE_SAVE_PATH=/tmp/first-bend-validation-save.json ./scripts/play-mining.sh \
  --validate /tmp/first-bend-validation --benchmark-seconds 60
```

It writes five screenshots, `operation.json` and `render-timing.json`, and restores a saved shift during processing. A failure writes `failure.txt` and returns an unsuccessful exit. The operation runs one fixed 1/60-second domain step with two FLIP substeps per worker request. If physics takes longer, the simulation slows without blocking the camera or accumulating work. The report includes frame timing during active washing, simulation speed relative to wall time, and a separate post-cleanup benchmark excluding five seconds of warmup. `--capture /tmp/site.png` captures the initial scene and exits.

To inspect an existing live shift without replaying the excavation:

```bash
./scripts/play-mining.sh --load /tmp/first-bend-validation-save.json --capture /tmp/live-sluice.png
```

The capture uses actual saved particle state, writes a timing receipt and closes its window. Sluice water renders as GPU-instanced sphere splats, including detached droplets, with curved depth for occlusion. One batch and reusable buffers cap rendering at 65,536 sampled blobs while every particle remains in the original solver and save. The final standalone live-shift check averaged 49.6 FPS; the earlier full-operation timing used the previous water presentation. Terrain flow continues to use the original GPU heightfield simulation.

Build a locally signed macOS application (fonts and scene assets are embedded):

```bash
./scripts/package-mining-macos.sh
```

The app is written to the shared target directory under `bundles/First Bend.app`. This is a local development bundle, not a notarized distribution release.
