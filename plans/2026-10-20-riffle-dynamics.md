# Flow, sediment and riffle dynamics

User report: water is slow, gold/dirt representation is wrong and riffle behaviour is missing. Continue from the pushed original-flow checkpoint `05cffd3` on `codex/direct-mining`; preserve dirty root checkout and original terrain/FLIP/APIC/DEM solvers.

- [x] Measure actual simulated time separately from renderer FPS; isolate expensive GPU stages.
- [x] Remove unconditional density-repair velocity damping; batch ordered GPU work and pressure sweeps, preserving reference results.
- [x] Keep water resident across fluid substeps and copy only DEM contact ranges between them; retain grid resolution and pressure iterations.
- [x] Replace two-litre marker parcels with smaller weighted sediment and physical grain size/density settling; correct mass/volume weighting, local wetness and bed obstruction.
- [x] Use actual grain-size meshes and depth/turbidity-dependent water shading.
- [x] Compare riffles, a bare-channel control and doubled flow using finite gold/quartz tracers, water recirculation and ledgers.
- [x] Finish CPU/GPU regression checks, warm and package the finite flowing demo.
- [x] Inspect the native image, controls, shader health and physical/render time separately.
- [x] Record evidence and limits, update Beads, commit and push PR #24; verify backup/app against pushed code.

The tracer experiment measures eight seconds after injection into a warmed channel. Remaining in the channel is not final recovery. No target recovery probability is introduced. A 1 cm fluid grid and representative DEM packets do not resolve individual sand pores or validate industrial recovery. Broader calibration and sustained real-time performance remain tracked independently.
