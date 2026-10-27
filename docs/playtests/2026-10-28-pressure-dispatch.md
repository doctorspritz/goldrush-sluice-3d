# Active pressure scheduling investigation

Active-cell pressure lists preserve the simulation but do not establish a speed improvement on this Metal GPU. All production changes are rejected and restored exactly to `a3c4f53`; the rebuilt probe executable is byte-identical. Keep the existing solver and installed app.

Five dispatch modes were measured: atomic linear lists with indirect dispatch; ordinary-index lists with fixed or indirect dispatch; and packed coordinates plus per-workgroup chunk allocation with either dispatch. Counters reset per solve, parity uses physical coordinates, masks use GPU-current flags, and argument construction ends before indirect consumption. Capability/resource failure falls back to the complete dense solve; no dispatch clamping, resolution reduction, dropped particles or reduced sweeps is allowed.

Each timestamp sample runs the same real connected-plant save for 30 warmup plus 60 measured fixed ticks. Pressure columns include list construction and argument work where applicable. Values are milliseconds per fixed tick; the physical ratio excludes rendering.

| Sample | Density pressure ms | Velocity pressure ms | Physical seconds/wall second |
|---|---:|---:|---:|
|reference-device|1.257|2.279|0.6408|
|candidate-device|1.887|3.583|0.5937|
|reference-device-v2|1.342|2.456|0.6313|
|fixed-device-v2|1.461|2.615|0.6182|
|indirect-device-v2|1.811|3.413|0.6014|
|reference-device-v3|1.308|2.376|0.6329|
|fixed-device-v3|1.643|2.655|0.6276|
|indirect-device-v3|2.007|3.512|0.5971|
|reference-last-device-v3|1.528|2.938|0.5835|

Shared workload/clock drift affects the samples: the last reference is slower across unrelated GPU stages too. Fixed dispatch narrows the gap but has no established benefit. Indirect dispatch consistently adds material pressure cost. Numerical equality alone does not justify adopting these variants.

**Four actual Metal tests pass on the restored solver**, including the independent face-flux residual/convergence oracle. The new `pressure_domains` regression compares 112 entire stored fields bit-for-bit across original, packed and cached schedules. It changes domains on live warm solvers, covers empty/one-parity/full regions, partial grid tiles and unit domains, boundary flags, and changing mobility readiness. Eight whole saved-state comparisons across candidate/restored runs are exact after sorting only unique exported contact-history keys; particles, cells, stocks and scalars remain unnormalized. Strict selected Clippy, test formatting and restored probe build/replay pass. This is an affected-surface check; no full-workspace or new native FPS/visual claim.

GLM 5.3 completed the bounded review of exact snippets verified publicly downloadable without authentication at the baseline. Tools were denied. Root checked parity/reset/coherence/indirect-use suggestions and authored the experiments. Its dispatch-clamping suggestion was rejected because it could skip physical cells; its description of the existing packed mapping was corrected. No unpublished code was sent.

An unintended uncached dev-profile compile was stopped by its verified owned process group before using the existing playable cache; it is not counted as passed. Initial strict Clippy found a test style issue, corrected and rerun successfully. No new panic/WindowServer report appeared. Only one task-owned heavy process ran at once. Scoped cleanup reports no automation browsers/tests and four unrelated Node processes left alone. Original nine dirty files, stash and signed installed app remain exact. Only `sluice-realtime` changes in the tracker; 261 unrelated rows remain literal. Beads 1.0.4 has no sync command; the scoped row lands with this checkpoint.

[Receipt](2026-10-28-pressure-dispatch.json), [compact experiments, exact controls and timing reports](2026-10-28-pressure-dispatch-experiments.zip). Full local saves/logs remain under `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/pressure-active-20261027`. Reproduce restored checks with `cargo +nightly test -p mining-physics --test pressure_domains --test pressure_schedule --test pressure_convergence --profile playable --locked --jobs 1 -- --ignored --test-threads=1 --nocapture`. The same plant save is `fixtures/modern.save.json` in the preceding hopper-stock evidence ZIP; probes use `SLUICE_PROFILE_GPU_TIMESTAMPS=1 .../flow_probe INPUT 60 OUTPUT 30`.

The full [realistic operation](../../plans/2026-10-22-realistic-operation.md) remains active. Physical real time is still unproved and previously failed; this investigation does not relax water/riffle/terrain fidelity or the unfinished digging, mud, plant, pond, tailings and failure requirements. Next investigate remaining batched GPU transfer/readback and CPU costs rather than repeat these rejected list schedules.
