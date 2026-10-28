# Periodic particle transfer prerequisite

The experimental repeated-Z strip now has periodic particle-to-grid and grid-to-particle transfers, including local APIC velocity gradients. These APIs complement the earlier [periodic projection](2026-10-28-periodic-projection.md). The complete water solver is not periodic yet and the playable/default mode remains unchanged.

`GpuP2g3D::new_periodic_z` uses ordinary atomic scatter. `GpuG2p3D::new_periodic_z` supports ordinary water and supported sediment-carrier gathering. Both require at least two physical Z cells. Pressure still separately requires an even depth. Tiled/cell-centric scatter and conservative phase-exchange geometry have no periodic implementation here; the experimental gather exposes no shared conservative-phase metadata.

Storage addresses wrap, while quadratic B-spline distances and APIC offsets use the original local stencil coordinates. Wrapping the offsets themselves would invent large affine velocities across the seam. W0 owns the physical seam; the redundant allocated depth face receives the same canonical accumulated velocity and support data. Each divide invocation reads immutable accumulators independently, avoiding a cross-thread read/copy race. Water-only and mixture/carrier/support outputs agree across the seam. Home counts and sediment vorticity reads wrap using floor/modulo, including just-negative and above-period coordinates; actual position advection/wrapping remains a later stage.

## Verification

**19 actual Metal checks pass on Apple M2 Max:** five new transfer controls and fourteen affected regressions. Independent oracles enumerate physical faces and their periodic images, rather than copying the shader's nearest-node/stencil algorithm.

- Nine scatter cases at depths 2/8/12 check quantised mass/momentum, separate water/solid home counts and paired water ranges. Aggregate conserved quantities stay within the declared fixed-point rounding bound.
- Twelve divide cases exercise ordinary/paired water, fresh and poisoned/stale output buffers. Both stored seam faces have bit-exact velocity/support values; unsupported faces reset. CPU/GPU division comparisons allow four f32 epsilon units because Metal may use reciprocal multiplication.
- Twenty-seven constant, X/Y-affine and Z-periodic field/PIC/FLIP combinations check 216 particle velocity/affine states against double-precision image sums. Redundant W input faces contain deliberately wrong values. Velocity tolerance is 2e-6 m/s; affine tolerance is 2e-4 per second.
- At depths 8/12, four repeated domains produce bit-exact scatter velocities and gathered velocity/affine state. A two-particle sediment control distinguishes supported stationary water from missing components and samples nonzero vorticity correctly on either side of the seam.
- Ordinary pressure, phase-exchange conservation/failure controls, momentum, particle uploads, wet-feed water release, storage growth and restored continuation retain their existing checks.

Strict Clippy passes for the new test target. Changed-span game-library lint has zero primary findings; 98 pre-existing warnings remain. All three changed/new Rust files pass formatting. These are affected-surface checks, not a full-workspace gate. Initial one-ULP and test-loop lint findings were corrected and their failed logs retained locally.

One complete ordinary-mode replay (30 warm-up plus 120 measured ticks) matches the retained reference after sorting only unique unordered contact-history keys. No particle/cell arrays, floating-point values or signed zero are normalised. Both canonical SHA-256 values are `65cfe7297c15460ff10ec9a7bfa3cad4cda65bf02929e4e13c1080dbd677bc6d`. The approximately 30 MB generated state was deleted after verification; its hashes and small timing/accounting report remain. This single replay is compatibility evidence, not a performance comparison. No new native capture, FPS, installation, full-fluid speedup or physical calibration is claimed.

Reproduce the new GPU controls with `cargo +nightly test -p mining-physics --test periodic_transfer --profile playable --locked --jobs 1 -- --ignored --test-threads=1 --nocapture` and strict lint with `cargo +nightly clippy -p mining-physics --test periodic_transfer --profile playable --locked --jobs 1 --no-deps -- -D warnings`.

Only one task-owned heavy job ran at a time, using the existing playable cache. No app or executable copies were created. Protected original files, stash, installed game and input fixture hashes remain exact; no new panic/WindowServer report appeared. The installed Beads version no longer offers `bd sync`; the service issue is updated and only its owned JSONL row is exported for normal git synchronization. The 262 unrelated tracker rows remain literal.

## Next integration stage

Make cell marking, density correction, vorticity production, boundary/SDF handling and advection agree with the periodic topology; implement the required periodic conservative-phase geometry before using that path. Then run complete 12/8/38-width water comparisons, followed by width-consistent water/mineral/bed ownership, graded gold response, local clogging/off-centre feed/lateral diversion and rendered copies. This prerequisite does not complete the [hybrid sluice](../physics/hybrid-sluice.md) or the [full mining operation](../../plans/2026-10-22-realistic-operation.md).

[Source/check/retention receipt](2026-10-28-periodic-transfer.json). Local evidence: `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/periodic-transfer-20261028`.
