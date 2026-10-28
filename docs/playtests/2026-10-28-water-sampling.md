# Full-width water-marker sampling

28 October 2026. Baseline `b4fd3e47c4ac`. This is a saved, opt-in water-quadrature mode, with its remaining qualification tracked by `menace-7rn`. The full realistic mining-operation goal remains active. The installed app and the eight-marker default are unchanged.

Four markers per reference cell each own twice the original marker volume; two each own four times it. Finite intake/feed credits, channel/airborne water, returned/terrain water and optical reconstruction use that volume. Mineral volume/mass and physical diameter retain their original reference, preventing accidental weakening of dirt/gold reaction. The full 38 cm channel, 1 cm grid, pressure schedule, riffles and grain contacts remain in place. The [model and source-build usage](../physics/water-marker-sampling.md) explain the explicit version-15 save envelope and runtime policy.

## Verification

- 52 affected CPU tests, 83 app tests and 14 actual Metal tests pass. Two existing app profiling tests remain ignored and are not counted as passes. Four strict scoped Clippy commands and formatting of all 13 changed Rust files pass; the dependency's existing unused-parameter-constructor warning is unchanged.
- Co-located 8/4/2 quadrature controls preserve physical carrier volume and actual sand/gold response. The real screen/sluice/outlet runs 480 fixed ticks per sampling, with finite water/gold/dry-material budgets. A further 24 fixed ticks compare two reconstructed hydraulic adapters after exact owned-state save/restore; this is not a claim of uninterrupted warm-cache equivalence for the new modes.
- Ordinary and explicit conservative reference operation each match the complete pre-change 150-tick canonical state exactly. Only JSON object keys and unordered contact-history key order are normalized; particle/cell order and floating-point values are retained.
- Two frozen reference captures match the complete pre-change PNG bytes exactly. Four live sluice and four live outlet frames pass readiness, ownership and advancing-clock checks. An additional evolved eight-marker outlet capture passes. Actual images were inspected.

## Measured operation

Each native measurement is one sequential 20-second window after five seconds of native warmup, loaded from that sampling's own finite eight-second evolved plant. Both use conservative phase exchange. This is a short comparison, not sustained or repeated performance qualification.

| Native measurement | Eight markers/cell | Four markers/cell |
|---|---:|---:|
| Mean render FPS | 56.35 | 104.93 |
| Physical seconds per wall second | 0.42983 | 0.48158 |
| p95 frame time | 33.93 ms | 15.00 ms |
| Mean surface-worker reconstruction | 111.34 ms | 113.33 ms |
| Real-time acceptance | **Fails** | **Fails** |

Higher FPS does not make the hydraulic simulation run at real time. Surface reconstruction remains slow, and the four-marker window records seven presentation-buffer underruns. These are follow-up work, not successful real-time acceptance.

At the same eight-second headless observation, eight/four marker cases have mean depths 24.68/24.39 mm and downstream speeds 0.38556/0.39503 m/s. Riffle-pocket reverse fractions are 9.3756%/9.4043%; slow fractions are 19.82%/20.27%. Pumped and processed volumes agree, while sparse settled-parcel distributions differ. These checks retain visible/physical riffle response but do not calibrate recovery, dense packing or clogging.

The separate three-mode integration control samples flow statistics after its 24-tick continuation, at 8.4 simulated seconds. Its two-marker case is about 16.8% shallower and 22.1% faster than eight markers. It remains a coarse investigation setting. No new default follows from its passing conservation checks.

## Visual findings and remaining scope

The four-marker channel remains a connected muddy sheet with mainly downstream motion. The outlet still has beading and stepped silhouettes. The evolved eight-marker outlet is a descriptive comparison at a different clock; it does not establish visual equivalence between sampling modes. Full visual acceptance, surge/stop/drying, off-centre or dense loading, graded washout/burial/clogging/cleanout, uninterrupted new-mode continuation and empirical optics/recovery calibration remain open in `menace-7rn`.

Reduced-width vertical water still requires full-width lane/head, bed/contact and world/save/render integration. Excavation, pads, mud, plant modification, settling ponds, tailings and flood/failure chains remain subject to the full [operation plan](../../plans/2026-10-22-realistic-operation.md). This stage does not close those requirements.

## Storage and safety

All temporary headless and native-final replay dumps were removed after verification: 206.6 MB. Duplicate stills were removed after exact comparison. Compact reports, hashes and distinct inspection images remain; no app copies were made. Original dirty files, fixture, installed executable and stash retain their protected hashes. Heavy jobs ran serially; no new crash report was observed. Early toolchain/formatting/type and one-frame-capture setup failures were corrected; the listed final commands are authoritative.

The installed Beads CLI lacks `sync`; the attempted command is recorded. Only the parent and follow-up issue rows are exported, preserving 262 unrelated rows. The [receipt](2026-10-28-water-sampling.json) records numerical/native results, source hashes and scope. Local evidence is `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/water-ownership-20261028`.
