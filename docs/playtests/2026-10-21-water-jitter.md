# Stable water presentation

Water previously extrapolated the latest snapshot and reset its prediction at each arrival. A gap larger than two domain ticks disabled prediction. Index-based thinning also changed most selected particles when the water vector compacted, rebuilding different surface kernels and highlights.

## Changes

- Water has persistent presentation IDs, retained through outlet transfer and saves. Legacy saves receive IDs once without changing positions, velocities, affine state or water volume.
- Bottom-k hashed IDs select at most 65,536 particles. Removing one particle retains the other selected identities; a newly selected replacement fades in.
- A six-snapshot render history maintains a monotonic playback clock with a small adaptive delay. It interpolates known endpoints instead of repeatedly resetting forward predictions. Pause, restored generations, cleanup and placement changes clear or freeze the appropriate history.
- Positions, velocity, kernel axes, normals, planarity and local extinction blend between matching endpoints. Eigenvector signs/permutations are aligned, and interpolated axes remain orthogonal. Births/deaths grow or shrink their kernels. Procedural ripples use the continuous presentation clock and blended velocity.
- One render-only background thread reconstructs optics/kernels and matches particles. At most two jobs are outstanding. The drawing thread copies one bounded source snapshot and uploads prepared endpoints when they change; the GPU interpolates between uploads. Shutdown drains the bounded replies.

The physical FLIP/APIC, riffle geometry, 1 cm grid, 120 Hz fluid/contact substeps, pressure settings, DEM and terrain-flow stages are preserved. IDs are metadata; rendering neither edits particle trajectories nor adds physical water.

## Live native checks

Both sequential runs start from the same bundled, already-flowing Alluvial demo and camera, with five seconds of warmup and twenty seconds measured. The bed develops during each run and shared workstation activity is uncontrolled, so this is an observed comparison rather than a controlled hardware speedup estimate.

| Measurement | Previous signed app | Final signed app |
|---|---:|---:|
| Mean render FPS | 45.80 | 56.56 |
| 95th-percentile frame time | 63.88 ms | 30.68 ms |
| Physical seconds / wall second | 0.7271 | 0.7295 |
| Whole-water residual | -10.614 mL | -10.431 mL |
| Sluice-water residual | 0 | 0 |
| Gold residual | 5.91e-12 g | 5.46e-12 g |

Final source capture on the drawing thread averages **0.333 ms**, maximum **0.916 ms**. Background reconstruction and matching average 25.220 ms; the 89.817 ms maximum remains evidence of occasional scheduling/work spikes. The bounded history recorded nine underrun frames across the complete app run, including startup. Interpolation reduces update popping but cannot hide an arbitrarily long stall.

The superseded synchronous implementation averaged 27.80 FPS, with 21.639 ms surface reconstruction blocking drawing. Its failed receipt is retained in `intermediate-synchronous/`; it is not the delivered app. An earlier background trial is in `intermediate-async/`.

The final rendering gate passes 50 FPS. The combined benchmark correctly exits with failure because physical throughput is below 0.98x. Real-time simulation work remains open in `sluice-realtime`.

## Verification and limits

The affected suites pass **69 tests**. They include irregular/batched snapshot timing, bounded background work and shutdown, eigenframe continuity, stable selection after particle removal, identity transfer, legacy migration, save/restore, conservation and input regressions. Ten GPU-specific tests remain explicitly ignored in this invocation; the native checks exercise the actual Metal solver and rendering pipeline. Focused strict Clippy, changed-file formatting and signed packaging pass. The game dependency retains its existing unused-constructor warning.

The native control replay covers 18 keyboard/mouse/toolbar, pause/resume, demo reset and live save/restore checks. Inputs are synthetic Bevy events; OS focus/key delivery is not exercised. Live channel, outlet and visible river captures were inspected; still images alone do not establish subjective animation quality. The [numeric receipt](2026-10-21-water-jitter.json) records source hashes, native measurements and artifact paths.

Rendering uses a small delay behind authoritative state. It improves presentation without establishing calibrated slurry/recovery accuracy or resolving the existing interphase-momentum budget.
