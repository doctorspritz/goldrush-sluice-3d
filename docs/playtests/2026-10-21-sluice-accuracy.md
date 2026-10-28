# Sluice accuracy: contact corrections and a reproducible laboratory

The sluice now has repeatable physical experiments, corrected grain contacts and a corrected APIC transfer stencil. This is a numerical accuracy milestone, **not a calibrated gold-recovery predictor**. The existing 1 cm FLIP/APIC water, original riffles, terrain flow and finite material accounting remain in use.

## Corrections

- Coulomb wall/pair impulses include contact-point rotation and effective mass. New shifts use them; saves without the new model field retain legacy damping.
- CPU grain contacts and GPU water now sample the same cell-centred SDF. The old CPU offset was 5 mm. Contact normals use the derivative of that interpolant.
- GPU point collision no longer removes grain impact velocity before finite-radius CPU DEM handles it.
- A single grain uses its physical radius; packed-volume inflation applies only to weighted bulk parcels.
- Quadratic particle/grid transfers use their complete support. A manufactured constant field previously generated an artificial APIC gradient around 0.56 /s. The corrected GPU transfer returns constant velocity and zero gradient within a 0.0005 /s floating-point tolerance.
- Pump-off cleanup collects settled material over the mats, leaving moving and upstream parcels resident. Fine water/feed controls use 0.25 L/s and 0.025 L/s steps; Shift retains coarse adjustment.

The [accuracy protocol](../physics/accuracy.md) documents the equations, commands, primary references, limitations and staged shaker-deck validation. The [machine-readable receipt](2026-10-21-sluice-accuracy.json) preserves measured conditions, source/binary hashes, assays and diagnostic failures.

## Actual GPU experiments

Each final trial ran 75 simulated seconds: 8 s water warmup, 12 s gravel loading, 30 s with tracers and feed, 10 s clear-water flushing, then 15 s pump-off. Each size used 24 physical gold grains and 24 equal-sized quartz grains. Feed was explicitly specified gold-free laboratory gravel, not gameplay yield. All four cases achieved their requested water/feed rates within 2% and passed water/gold/finite-state checks.

| Case | Water | 80 µm gold lost | 150 µm | 300 µm | 500 µm |
|---|---:|---:|---:|---:|---:|
| Low | 1.25 L/s | 8.3% | 0% | 0% | 0% |
| Nominal | 3.75 L/s | 50.0% | 12.5% | 4.2% | 0% |
| High | 7.50 L/s | 100% | 16.7% | 4.2% | 0% |
| Bare channel | 3.75 L/s | 100% | 100% | 100% | 100% |

Solid feed was 0.1 L/s in every final case. These deterministic releases demonstrate flow-sensitive loss and a riffle/control difference. They do not establish confidence intervals, equilibrium recovery, or matched experimental accuracy.

After correcting cleanup, the final saved states were assayed again **without advancing physics**. The original receipts and saves were preserved.

| Case | 80 µm gold collected | 150 µm | 300 µm | 500 µm |
|---|---:|---:|---:|---:|
| Low | 75.0% | 33.3% | 4.2% | 4.2% |
| Nominal | 20.8% | 75.0% | 83.3% | 87.5% |
| High | 0% | 83.3% | 95.8% | 100% |
| Bare channel | Empty mats | Empty mats | Empty mats | Empty mats |

Uncollected, unlost material remains resident. In particular, low flow does not imply successful cleanup simply because gold has not reached the outlet. The old raw receipts show cleanup refused when any grain remained moving; the separate reassessment records the corrected settled-mat operation.

## Remaining physics failures

The closed zero-gravity full-step momentum diagnostic is **not passing conservation**. Relative residuals were 1.62% for its water control, 10.35% for the mixture at 120 Hz and 6.80% at 240 Hz. Because the water control also has error and the probe includes transfer and pressure, this does not isolate drag as the cause. Attribute the residual, then implement conservative local phase exchange and porosity-aware equations; do not fit recovery percentages around it.

No final trial had water or grain centres more than 2 mm inside the solid SDF. That is weaker than nonpenetrating finite-radius contacts: the low-flow dense bed had maximum grain **surface** penetration of 2.96 mm. Nominal/high maxima were approximately 0.33/0.19 mm. Dense-bed contact convergence and boundary geometry remain open.

Earlier seven-case experiments included heavy feed and a parcel-resolution pair, but used an intermediate solver before the SDF/APIC corrections. They are preserved as exploratory evidence and do not complete the final convergence matrix. Exact experimental geometry/matting, slope/feed sweeps, flake drag, long-duration loading and independent holdouts remain required. Shaker-deck work has a documented physical validation sequence and its own issue; no validated shaker is claimed here.

## Verification and delivery

All 182 focused tests passed: 113 sim3d library, 29 domain, 15 physics (with GPU tests explicitly included) and 25 native-app tests. Actual GPU contact ownership, constant transfer, fluid transfer equivalence and complete pipeline/live restore are covered. The signed app also passed 18 synthetic native input/toolbar/save/restore checks; this does not test OS keyboard delivery or replace a manual playtest. The broad sim3d integration selection was deliberately stopped before completion; it is not counted as passing. Focused Clippy completed successfully; the game dependency still emits its existing unused-constructor warning.

The refreshed finite demo contains 106,084 channel water particles and 1,358 grains at 40.0167 simulated seconds. Its compressed asset SHA-256 is `da842b36dfbcce024917458526366953a442df72fa820c2308c6b9f3d7cebb6f`. The live native capture shows the reconstructed channel water and airborne discharge; the existing voxel ramp steps remain visible. Full experimental saves and native images are kept with the local artifact checkpoint.

On the Apple M2 Max, the final 20.0335-second native benchmark averaged **48.54 FPS and 0.6506× physical simulation speed** (13.0333 simulated seconds). Water residual was -8.111 mL, sluice residual zero and gold error approximately 3.2e-12 g. Both the 50 FPS and 0.98× throughput gates correctly failed. This was not a matched performance comparison against the preceding revision; host load and the evolving granular bed differ. No water/riffle resolution was reduced. The first throughput launch exited without a receipt and is explicitly unverified; the subsequent complete measurement supplies these results.

Builds and simulations ran sequentially with one compiler worker, alongside at most one existing external browser-test batch. No personal browser was used.
