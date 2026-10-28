# Complete periodic water experiment and boundary coverage fix

`GpuFlip3D::new_periodic_z` now runs the complete experimental water-only loop: periodic APIC transfers, pressure projection, cell marking, density correction, collocated cell-centred curl/confinement, boundary enforcement, interpolated SDF collision and particle advection. Storage growth retains its topology. This advances the [particle-transfer prerequisite](2026-10-28-periodic-transfer.md) to evolving water over riffles. It is not selected by the game.

The constructor requires positive X/Y and even Z depth of at least two. This pilot accepts unit-volume water markers and ordinary unsorted transfers. It refuses open-Z/sorted/bed configurations; conservative phase geometry and solids ownership remain unfinished. Ordinary three-dimensional reference topology/settings remain available.

A shared defect was also corrected: both W-boundary passes launched `ceil((depth+1)/8)` workgroups while the shader processes four Z faces per workgroup. They now launch with a divisor of four. The actual old-dispatch negative control leaves an upper-quarter face at velocity1 after two half-slip passes; the corrected complete solver produces0.25 at depths12/38. This fix affects the ordinary solver as well as the experimental strip.

## Physical and integration checks

**35 distinct actual Metal tests pass on Apple M2 Max.** Seven new controls cover independent periodic neighbour/density and positive-face displacement oracles, analytically known collocated MAC curl, confinement with real air/solid seam neighbours, SDF interpolation, repeated boundary crossings, evolving riffle water without entry into solid voxels, repeated-domain agreement and storage growth. Tiled evolution uses 32 substeps at depths8/12. Resident-versus-synchronous stepping compares complete position/velocity/affine arrays exactly over six four-substep frames. The remaining affected controls cover pressure/transfers/phase conservation, wet-feed water release, finite screen-to-sluice transport, uploads/storage and all five existing excavation/machine/soil/payload/live-restore scenarios.

Strict new-test/example Clippy and formatting pass. Changed host spans and the new API are formatted; whole legacy game formatting is not claimed. Game-library lint retains98 existing warnings with zero primary findings on changed spans. These are affected-surface checks, not a full-workspace gate. The initial one-ULP uniform-flow assertion and test-style lint finding were corrected, with failed logs preserved locally.

## Evolving water cost

This uses the actual reference centre-section floor/riffle/outlet geometry extruded across each periodic width,1cm spacing,162x52 XY cells,80 pressure and40 density sweeps, and a1/240-second substep. Each finite water pulse starts with eight markers per occupied cell. Domain exits transfer to a benchmark-only finite count ledger; live plus retired counts always equal initial counts. There is no incoming feed, mineral/bed coupling, world simulation or rendering.

All three workloads are constructed and warmed before six forward/reverse paired cycles in order38/12/8/8/12/38. Each block has64 warm-up and16 measured substeps. The resident cadence uploads/reads once per four substeps, using the existing in-place GPU path between observations. A synchronous control uploads/reads every substep. Both retire exits at the same four-substep boundary. Complete-array equality is separately tested; final exact-bit diagnostic fingerprints and all recorded state statistics agree across cadences for every width. The diagnostic hash is non-cryptographic and is not a conservation proof.

| Periodic width | Initial markers | Resident median ms/substep | Water-only physical seconds / wall second | Median paired width factor |
|---|---:|---:|---:|---:|
|38cm|172,672|5.422|0.769|1.00|
|12cm|54,528|2.831|1.472|1.922|
|8cm|36,352|2.356|1.769|2.297|

Paired ranges are1.894–1.968 for12cm and2.263–2.375 for8cm. Synchronous medians are13.625/8.544/9.404ms; those water-only ratios remain below1. Final resident live/retired counts are165,733+6,939;52,389+2,139;34,865+1,487. No water marker disappears from the benchmark ledger.

Mean downstream speeds are0.51804/0.51736/0.51647m/s. Backflow fractions differ:2.794/2.407/2.883percent, and outlet counts per centimetre are402.132/396.667/402.000. Similar mean flow is not enough to establish equivalent fine-gold recovery or lateral blockage. This short finite pulse does not prove steady-state riffle behaviour, real plant incidents or a native game speedup. The water-only physical ratios exclude Session/plant/terrain/rendering costs and use the declared pilot substep. The earlier synchronous run without an explicit exit ledger is retained as preliminary evidence, not used for the final factors.

Reproduce: `cargo +nightly run -p mining-physics --example periodic_water_probe --profile playable --locked --jobs 1 -- OUTPUT.json 6 --resident`; omit `--resident` for the synchronous control. Run the new GPU checks with `cargo +nightly test -p mining-physics --test periodic_water --profile playable --locked --jobs 1 -- --ignored --test-threads=1 --nocapture`.

## Ordinary mode and next integration

The complete ordinary150-tick replay intentionally changes after the shared boundary correction: water markers116,221→116,044, mean downstream speed0.389419→0.391886m/s, and depth0.0254871→0.0254482m. Its canonical state is not bit-exact with the older build. Whole-water error remains approximately1.0132e-7m3; gold error is about-5.0e-12g, with606 grains and unchanged measured bulk/dry-mass accounting errors. The old reference snapshot is retained. The roughly30MB generated replay state was deleted after comparison, preserving its hashes and metrics.

Next implement required periodic conservative phase geometry and width-consistent water/mineral/bed ownership, then graded gold washout/loading/cleaning, local off-centre feed/clogging/lateral diversion, saved continuation and rendering/world transfers. Compare12/8/full width under those regimes before selecting a playable default, and profile the remaining Session/plant/world costs. Every unfinished requirement in the [complete mining operation](../../plans/2026-10-22-realistic-operation.md) and [hybrid specification](../physics/hybrid-sluice.md) remains active.

One task-owned heavy job ran at a time; no app/executable copies or native/browser captures were created. Original local files, stash, installed game and input fixture hashes remain exact; no new crash report appeared. No installation or new visual qualification is claimed. The installed Beads CLI lacks `sync`; only the owned active issue row is exported, preserving262 unrelated rows before normal git synchronization.

[Qualification receipt](2026-10-28-periodic-water.json). Compact local evidence: `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/periodic-water-20261028`.
