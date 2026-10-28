# Periodic pressure projection prerequisite

The experimental periodic-Z pressure operator is implemented and tested. It connects lateral pressure neighbours and MAC face flux across the strip seam; it does not yet make the complete FLIP/APIC pipeline periodic. Ordinary constructors keep the specialization disabled. The playable mode and installed app are unchanged.

`GpuPressure3D::new_periodic_z` requires a positive even lateral cell count, because an odd periodic ring is incompatible with the red/black ordering. Pressure stencil codes/indices, face mobility, divergence, gradient and residual diagnostics share the same topology. The W0 face owns the seam. One gradient invocation writes both W0 and its redundant depth-plane copy; the upper invocation returns, preventing a read/write race. The periodic mode uses width-independent SOR 1.5 for these initial controls; this is not an operating optimum or a calibrated fluid model. Open Z boundaries cannot also be periodic. Original constructors/settings remain the reference.

**19 actual Metal tests pass.** The three new tests cover 36 manufactured projections across plain/weighted coefficients, mixed X/Y boundaries, asymmetric air/fluid and solid/fluid seam pairs, poisoned redundant velocity/coefficient inputs, and all three dispatch schedules. An independent oracle visits each physical face once and checks the stored equation and corrected velocities. Closed constant-pressure gauge is retained; an incompatible closed source remains visible as a large residual. Invalid odd/degenerate depths are refused. At 1/7/80 sweeps, 8- and 12-cell pressure strips equal four tiled periods bit-for-bit. Other checks retain ordinary pressure convergence/schedules, transfer/momentum, phase conservation, wet feed, storage restore and finite screen processing.

Strict selected adapter/example/test Clippy, formatting and changed-span game lint pass. The game library retains 98 existing warnings, with zero primary diagnostics on changed lines. The new API and new Rust targets are formatted; no whole-game formatting claim. Initial strict lint found two test boolean-style findings, corrected without suppressions. This is an affected-surface check, not a full-workspace suite.

Three complete 150-tick ordinary-mode saved states have identical canonical SHA-256 values after ordering only unique exported contact-history keys. No particle/cell order, floating-point value, signed zero or scalar is normalized. Short A/B/B/A headless timing varies across Session and solver work; no ordinary-mode performance gain is inferred from it.

## Component timing

The probe uses the reference channel's repeated centre section, floor/riffles/inlet/exit geometry and a synthetic four-cell water layer. It measures divergence, 80 pressure sweeps and gradient with device timestamps. There are no evolved particles, free surface, mineral interactions or rendered replicas. All three widths use the same operator and SOR, with matching relative stored-pressure residuals of approximately 0.005861; this reported value is not a new acceptance limit for complete fluid simulation.

All workloads are constructed first, warmed for eight forward/reverse cycles, then measured for twelve cycles in width order 38/12/8/8/12/38. Each width has 24 measured samples. Absolute times vary substantially, so paired per-cycle factors are reported as well.

| Periodic width | Median device projection ms | Median paired speed factor versus 38 cm | Paired factor range |
|---|---:|---:|---:|
| 38 cm | 3.214 | 1.00 | — |
| 12 cm | 1.863 | 1.76 | 1.49–2.61 |
| 8 cm | 1.711 | 1.96 | 1.63–2.96 |

Every paired cycle favours the narrower projection in this run. These are component measurements, not a full water/game speedup or physical real-time result. The initial separately warmed ordering had strong first/last-reference drift; its data and source remain in the local evidence, and it is not used for the speed factors above.

Reproduce the component run with `cargo +nightly run -p mining-physics --example periodic_pressure_probe --profile playable --locked --jobs 1 -- OUTPUT.json 12`. Run the new actual-GPU controls with `cargo +nightly test -p mining-physics --test periodic_projection --profile playable --locked --jobs 1 -- --ignored --test-threads=1 --nocapture`.

GLM reviewed exact snippets verified publicly downloadable without authentication at `d93931b`, with tools denied. It highlighted canonical indices/coefficients, stale seam writers and closed compatibility. Root corrected its physical-depth/workgroup-rounding distinction and its assumption that every wrapped neighbour is fluid; real air and solid types are retained. No unpublished source was sent, and root implemented/validated the changes.

No new native FPS, visual comparison, packaging or installation is claimed in this prerequisite. Scoped cleanup runs; only one task-owned heavy process ran at a time. Original nine dirty files, stash and installed app remain exact, with no new panic/WindowServer report. Only the hybrid prototype issue changes; unrelated tracker rows remain literal.

[Receipt with source hashes, checks, exact states and timing samples](2026-10-28-periodic-projection.json). Full local evidence is under `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/periodic-projection-20261028`.

Next implement periodic P2G/G2P/APIC transfer, cell/density/vorticity handling, collisions/advection and the complete 12/8/full-width water benchmark. Then qualify width-consistent water/mineral ownership, graded gold, local clogging, lateral diversion and rendering. The [hybrid specification](../physics/hybrid-sluice.md) and [complete mining operation](../../plans/2026-10-22-realistic-operation.md) remain active; the projection prerequisite does not complete them.
