# The Sluice Moose

A Rust mining game and a collection of fluid/granular simulation experiments.

## First Bend prototype

The direct-control prototype uses the original water and granular solvers. The [staged operation plan](plans/2026-10-22-realistic-operation.md) covers realistic earthworks, pads, mud, screening and settling pools. The [earthwork protocol](docs/physics/earthworks.md) describes finite swept bucket cuts, constrained soil resistance and their current limits; the earlier complete mining shift is a historical checkpoint in [the first-shift report](docs/playtests/2026-10-20.md). The expanded full operation remains under development. Builds use one compiler job by default; the game uses one bounded simulation worker and limited Bevy task pools.

The [environment-detail report](docs/playtests/2026-10-21-environment-detail.md) records shared finer terrain/water rendering and material detail. The [shallow-water report](docs/playtests/2026-10-21-shallow-water.md) covers millimetre-scale visibility and FPS. The [channel/river surface report](docs/playtests/2026-10-21-water-surface.md), [outlet report](docs/playtests/2026-10-21-water-outlet.md), [original-flow restoration](docs/playtests/2026-10-20-original-flow.md) and [riffle dynamics report](docs/playtests/2026-10-20-riffle-dynamics.md) record earlier flow and presentation work and its measured limits.

The [shoreline report](docs/playtests/2026-10-21-shoreline.md) records full-resolution river/terrain contact, pixel-sized edge blending, native comparisons and its added GPU memory cost. The physical solvers and accepted particle/outlet optics remain unchanged; larger bank contours still need refinement.

The [crest and slurry rendering report](docs/playtests/2026-10-22-crests.md) records the shared water-surface reconstruction, softer muddy transitions, native animation and measured rendering cost. The water, riffle and sediment solvers remain unchanged.

The [earthwork report](docs/playtests/2026-10-22-earthworks.md) records swept excavation, conservative terrain collapse, clean-source dilution, native digging/delivery and live save/restore. Digging with the terrain solver runs near real time; a loaded sluice remains around 0.70× in the latest serial comparison. Full-circuit solid transport and finer physical soil are still open.

**Environmental water now has a complete source/storage/discharge ledger.** Dry/wet digging and dumping preserve water, shared GPU face transfers respect each donor's total volume, and the river drains at its downhill edge. The regenerated demo has a graded return channel connecting the sluice discharge to the river. See the [repair measurements](docs/playtests/2026-10-21-environment-water-fix.md); the [earlier audit](docs/playtests/2026-10-21-environment-water-audit.md) remains a historical record.

## First Bend controls and loop

The demo app opens beside an installed sluice with real water and finite paydirt already flowing. Its bundled shift was warmed by the original solvers for 40 simulated seconds with corrected Coulomb grain contacts and APIC transfers; it still has about 0.81 m³ in the hopper. Use **Demo** to restart that view, **New shift** to begin with an untouched claim, and the visible **Help / Pause** buttons.

The [sluice accuracy laboratory](docs/physics/accuracy.md) provides repeatable size-resolved water/feed experiments and actual cleanup assays. The [measured results](docs/playtests/2026-10-21-sluice-accuracy.md) distinguish outlet loss, residence and collected gold. Real-world recovery remains uncalibrated: the closed momentum budget and dense-bed contact convergence still need work. Water/feed arrows now have fine steps; hold Shift for coarse changes.

New shifts also separate fine soil from sand, and local suspended-material plumes replace the whole-channel dirt tint. Denser visible grain sampling represents the existing weighted material without adding independent DEM bodies. See the [slurry model and its limits](docs/physics/slurry.md) and [native measurements](docs/playtests/2026-10-21-slurry.md).

For a fresh mining shift, walk to the excavator, cut finite paydirt, swing a loaded bucket over the hopper, place a sluice, wash and lift the mats. The first shift's goal is **5 grams of recovered gold**.

On this Mac, double-click **Play First Bend.command**, or run:

```bash
./scripts/play-mining.sh
# Start the full mining loop instead of the running demo:
./scripts/play-mining.sh --new-shift
```

Click the scene to capture the mouse, then move it to look around. The capture click does not dig. **Esc** frees the pointer for the toolbar; Esc again pauses. Right-drag also looks while the pointer is free. WASD, mouse motion and Q/E cab swing follow the camera's screen directions. Short taps are buffered until the worker accepts them, and help/focus loss stop movement.

Requires a native GPU, Cargo and Rust **1.88 or newer**. The launcher uses an installed nightly toolchain when available, otherwise stable. It builds the Bevy executable in the `playable` profile with one compiler job. `SLUICE_BUILD_JOBS` overrides that limit.

| Control | On foot | In the excavator |
|---|---|---|
| WASD | Walk | Drive / turn tracks |
| Mouse after capture / right-drag | Look | Look / orbit |
| Shift | Run | — |
| E / Tab | Enter nearby cab | Tab exits |
| V | First / third person | Cab / external view |
| Left click | Shovel reachable ground | — |
| J | Pour hand bucket | Open and empty machine bucket |
| Q / E | — | Swing upper body |
| R / F | — | Raise / lower boom |
| T / G | — | Extend / tuck stick |
| Y / H | — | Curl / open bucket |
| Space / left mouse held | Shovel on click | Cut soil at the bucket edge |
| I | — | Toggle engine |

At the plant: **B** previews a sluice. Point near the hopper outlet to snap its inlet; **R/Q** rotate, **[/]** or the wheel adjust slope, left-click confirms, Esc cancels. The preview rejects unsupported, overlapping, out-of-claim and disconnected placements. Walk to the plant and use **P** for the pump, **O** for feed, arrows to change water/feed rates, and **C** to lift the mats after stopping both.

**F1** controls, **F5** save, **F9** restore, **Esc** release mouse / pause, **Ctrl+Q** quit. macOS saves to `~/Library/Application Support/Sluice Moose/first-bend.json`. The flowing demo saves separately as `flowing-demo.json` in the same directory. `SLUICE_SAVE_PATH` overrides either path for isolated tests. Invalid saves are rejected before changing the current shift.

## Scope and physics

- Bevy 0.17.3 supplies the native scene, input, PBR lighting, cascaded shadows, fog, UI and screenshots. Equipment, vegetation and the operator are procedural models; Barlow is bundled under the SIL Open Font License.
- `mining-sim` reuses `sim3d::World` for finite layered terrain and excavation. `mining-physics` calls the original GPU heightfield surface, flux, depth, settling, erosion, sediment-transport and collapse pipeline. Coarse-world gold is a conservative passive tag of the paydirt channel; density separation in the sluice uses the original particle solver. Carried loads, falling loads, hopper, live parcels, tailings and cleanup preserve the gold ledger.
- The excavator is a **constrained kinematic machine**. Its moving floor/tooth geometry cuts finite layered soil; passive/bedrock contact, capacity and a generic soil force/power envelope constrain movement. Actual hydraulic leverage, failure wedges, rigid-body tracks, tipping, saturation and compaction remain future work. The physical soil grid is still 25 cm.
- The sluice calls the existing `game::gpu::flip_3d::GpuFlip3D` FLIP/APIC solver and `sim3d::ClusterSimulation3D` DEM contacts. The original geometry builder supplies collision solids and the visible sluice. Water particles, affine state, sediment motion, DEM contact history, terrain momentum and separate suspension channels are saved. Physics uses a separate wgpu 23 compute device on one worker while Bevy presents immutable snapshots through wgpu 26. The worker allows one active input packet and one pending packet, with no burst of overdue ticks. New shifts run four FLIP/DEM substeps per 1/30-second fluid update, retaining 1 cm cells and 80/40 pressure iterations. Density repair preserves momentum and the velocity safety limit follows CFL. Grain diameter and density set exact drag/buoyancy relaxation, local immersion sets wet/dry friction, and settled packet volume feeds back into water resistance/obstruction. Weighted parcels are not a calibrated recovery predictor; realtime performance, bed packing, clogging and screening need further work.
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
cargo +nightly run -p mining-physics --example operator_replay --profile playable --jobs 1 -- --earthworks-only
cargo +nightly fmt --package mining --package mining-sim --package mining-physics --check
```

The current `--earthworks-only` replay drives continuous joint axes to scoop and deliver real terrain, without injected material or teleported equipment. The native mode adds rendered evidence and live save/restore. It verifies excavation/delivery, not the full wash/cleanup loop. At the small default sluice's 0.1 L/s feed, a 0.58 m³ load takes about 97 simulated minutes to wash; plant scaling and long-operation pacing remain open. The full native replay completed on 20 October under the previous excavation model; its [historical evidence](docs/playtests/2026-10-20.md) does not establish completion with the current digging model. Real-time physics throughput, phase exchange, recovery calibration and the full staged goal remain open.

Rendered verification opens its own native window:

```bash
SLUICE_SAVE_PATH=/tmp/first-bend-validation-save.json ./scripts/play-mining.sh \
  --validate /tmp/first-bend-validation --benchmark-seconds 60
```

It writes five screenshots, `operation.json` and `render-timing.json`, and restores a saved shift during processing. A failure writes `failure.txt` and returns an unsuccessful exit. Each bounded worker packet runs at most two already-due fixed 1/60-second domain steps; new shifts update fluid on alternate domain ticks using four substeps. Older saves retain their original fluid cadence. If physics takes longer, the simulation slows without blocking the camera or accumulating work. The report includes frame timing during active washing, simulation speed relative to wall time, and a separate post-cleanup benchmark excluding five seconds of warmup. `--capture /tmp/site.png` captures the initial scene and exits.

To inspect an existing live shift without replaying the excavation:

```bash
./scripts/play-mining.sh --load /tmp/first-bend-validation-save.json --capture /tmp/live-sluice.png
```

The capture uses actual saved particle state, writes a timing receipt and closes its window. Sluice water uses particle-neighbourhood anisotropic kernels, ray/ellipsoid depth reconstruction, bilateral smoothing and a volume-normalized optical thickness pass. Scene depth keeps water behind equipment; detached spray remains separate. Reusable buffers cap rendering at 65,536 sampled kernels while every physical particle stays in the solver/save or the explicit exterior outflow state. River and runoff use connected, shore-clipped surface cells reconstructed from the original terrain-flow solver, with velocity-driven optical ripples and the saved suspension concentration.

New shifts retain the standalone sluice profile: 1 cm cells, a 1.5 m × 0.40 m channel with the original riffles, about 9.8° slope, and four 120 Hz substeps per fluid update. Water leaves at the lip with its actual velocity, follows saved ballistic free flight, and transfers water plus horizontal momentum into terrain at its swept impact. The original FLIP/APIC pressure and DEM contacts own the channel; the original conservative terrain solver owns river/runoff. The exterior falling sheet has no separate pressure or secondary-splash solve yet. New granular shifts use vertical gravity and inlet momentum; older saves retain their extra slope-derived drive and explicit legacy grid/particle volumes. Bounded render prediction and optical animation stop on pause/load.

Measure physical throughput separately from rendered FPS:

```bash
./scripts/play-mining.sh --flow-benchmark /tmp/first-bend-flow.json --benchmark-seconds 60
```

This excludes five seconds of warm-up, records a minute of simulation/wall speed and frame timing, then saves the measured state as `first-bend-flow.save.json` for a matching headless probe. It returns failure below 0.98x simulation speed, 50 mean render FPS, or if water rendering/conservation fails. The current result is recorded in the report above; passing unit tests alone does not establish real-time performance.

To exercise the native keyboard/mouse and toolbar handlers, including cab controls and live F5/F9:

```bash
SLUICE_SAVE_PATH=/tmp/first-bend-controls-save.json ./scripts/play-mining.sh \
  --check-controls /tmp/first-bend-controls.json --capture /tmp/flowing-demo.png
```

These are synthetic Bevy inputs in the native app, not OS key injection. The scripted check runs without desktop focus or mouse capture, writes a receipt for each checked behaviour, captures the running demo and exits. Normal-play focus/cursor handling is covered separately by the input-system regressions. The demo fixture is regenerated with `mining-physics/examples/make_demo.rs`; its source and procedure are documented under `crates/mining/assets/demo/`.


Build a locally signed macOS application (fonts and scene assets are embedded):

```bash
./scripts/package-mining-macos.sh
```

The app is written to the shared target directory under `bundles/First Bend.app`. This is a local development bundle, not a notarized distribution release.
