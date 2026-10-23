A complete retained-soil mechanical owner now preserves live contact history, the actual controller, finite clod custody and application handle maps through clone and saved continuation. It remains **experimental**: loaded timestep refinement fails and positive coupled-energy residuals remain unresolved. The default game retains its existing mechanical model. This checkpoint advances Stage 2 and leaves the full mining-operation goal open.

The shared numerical step is unchanged after substituting handle storage; gameplay and the experiment now share the exact equipment-box generator. All 201 audited rendering, water and terrain source files match `2453215`. Physical grids, water/riffle/terrain equations, 1,040 bank clods, contact law, default 240 Hz clock and four TGS/sixteen PGS stages are not reduced.

The [numerical boundary](../physics/contact-history.md) is 1,749,508 bytes before stepping and 15,620,067 bytes after 24 steps in the actual bank scene. A [complete owner checkpoint](../physics/retained-soil.md) is 16,538,128 bytes at the tested 32-step boundary. Its bounded binary format stores engine state plus canonical controller/soil JSON, handle maps and environment geometry. Input/geometry refusal preserves the prior bytes; corrupt owner/configuration and truncated/trailing data are rejected. This engine-produced format is not admitted into Session saves.

All 19 complete canonical machine/soil checkpoints match between the uninterrupted 18-second retained run and a run restored at every second (18 actual restorations). Initial and warmed numerical-world tests also compare every rigid pose/velocity, generalized velocity, raw/solver manifolds and nonzero cached normal/friction impulses. Separate canonical validation matches the corrected cold mechanical scene against the original recorded controller, root/link state and every clod, retaining exact f64 material/mass/energy fields. The ordinary Session replay independently matches all 19 original complete saved states byte for byte.

The initial isolated comparison omitted the plant boxes, changing approximate peak load from the original 4.219 L to 11.953 L. Those timings are excluded below. The corrected fixture shares the actual game geometry and recovers the original 4.219 L result. The interrupted isolated candidate is recorded as interrupted, not passed.

| Same 1,080 commands, 18 physical seconds,actual equipment geometry | Hz | Mechanical step time | Simulation/step-wall | Peak estimated contents | Maximum positive coupled residual |
|---|---:|---:|---:|---:|---:|
| Rebuilt world | 240 | 161.65 s | 0.1114× | 4.219 L | 353.01 J |
| Retained world | 240 | 113.49 s | 0.1586× | 1.406 L | 2899.93 J |
| Retained, 18 restores | 240 | 111.30 s | 0.1617× | 1.406 L | 2899.93 J |
| Retained, refined | 480 | 226.28 s | 0.0795× | 0.000 L | 1278.24 J |

The retained 240 Hz sample uses 29.79% less step time (1.424× throughput), but it follows different contact/loaded trajectories. This is not a numerically equivalent optimization, a native FPS measurement or a game speedup. Step times exclude checkpoint work; startup/checkpoint/loop totals are separate, and the optional final diagnostic boundary file follows the loop total.

The retained 240/480 Hz root path differs by up to 8.11 cm, with peak contents changing from 1.406 L to zero. The containment estimate is coarse clod geometry, not a calibrated yield measurement. A single stroke does not establish bulk convergence or permit attribution solely to soil instead of actuator/contact coupling. These failures prevent adoption.

Two-second, zero-command controls provide a useful next lead:

| Same initial bank and equipment; no engine supply | Rebuilt maximum positive residual | Retained maximum positive residual |
|---|---:|---:|
| Engine off, elastic actuators | 65.03 J | 838.47 J |
| Engine off, free hinges | 65.47 J | 64.87 J |

Removing passive elastic/damping actuators greatly reduces the retained excess in this sample. It also changes the motion and contacts; this points toward coupled actuator/contact discretization rather than proving one specific fault. Contact stabilization and full energy closure still require investigation. [The maintained input generator](../../scripts/make-soil-passive-tapes.py) reproduces both tested diagnostic tapes exactly, verified by their decompressed SHA256 hashes. It never edits the original fixture or an active save.

Independent free-impact controls retain the existing 0.003 Ns linear, 0.03 Nms angular and 0.01 J energy bounds at 240/480 Hz. Measured linear errors are 1.46e-5/1.63e-5 Ns; angular errors 0.002896/0.001189 Nms; energy falls 2.430/2.436 J. The half-second powered wrap control uses the existing machine-level 2 mm centroid, 1 Ns linear and 5 Nms intrinsic-angular budgets: centroid drift is 0.115 mm and intrinsic angular momentum 1.429 Nms. An initial one-second attempt used small-impact absolute tolerances and failed; that diagnostic is retained, and no stricter powered convergence is claimed. The live engine chart stays intact while the controller reference is lifted to it; tested save continuation crosses pi without rewriting body poses.

**238 maintained checks pass; 93 remain ignored in that default invocation.** Separately, 26 feature-enabled library checks and five explicitly executed real-GPU combined-pipeline checks pass. Other ignored suites are not claimed. Strict default and featured all-target Clippy, maintained-package formatting, whitespace and signed macOS packaging pass. CI runs the complete feature library suite; remote CI results are not claimed here.

The native app passes 18 synthetic ECS control/save checks. Frozen bank and sluice-water RGB match the preceding build exactly, with matching physical clock, solver, particle-track, flow, machine and material fields. Native bank and water images were inspected. Old saved models 7/8/9/10 each retain six canonical continuations (24 states) against their retained reference hashes. Manual OS input, a new evolving native playthrough and a native throughput improvement are not claimed. [The receipt](2026-10-24-retained-soil.json) contains hashes, inputs, measurements and scopes.

Heavy jobs ran sequentially with one compiler or simulation job; no new panic-full/WindowServer report appeared. Scoped cleanup leaves no owned game/build/encoder, browser session or browser-test process; 14 shared Playwright MCP and 56 Node processes remain untouched. Original dirty-checkout files and the user stash are preserved. Installed Beads does not support `bd sync`; only touched project rows are exported for commit/push.

Next work is the coupled elastic-actuator/contact energy and timestep error, then safe changing-geometry/material transactions, complete Session admission and native throughput. Wet/cohesive mud, fine continuous activation, compaction, pads, wet shaker decks, buildup, constructed settling/water circuits, calibration and all remaining Stages 2–6 requirements remain open. The retained owner and negative evidence do not complete those stages.
