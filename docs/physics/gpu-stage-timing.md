The sluice profiler measures device time inside the existing compute passes. It retains the ordinary ordered command buffer and copies the timestamp queries once after each complete FLIP substep. Collection follows the particle readback that the solver already requires. There are no marker dispatches or per-stage CPU/GPU fences.

```bash
SLUICE_PROFILE_GPU_TIMESTAMPS=1 cargo +nightly run \
  -p mining-physics --example flow_probe --profile playable --locked --jobs 1 \
  -- INPUT.save.json 120 OUTPUT.json 12
```

The arguments specify the input save, measured domain ticks, output report and warm-up ticks. `gpu_device_stage_ms` reports mean compute-pass milliseconds per **domain tick**, including the existing fluid cadence and substeps. Repeated labels are summed. The main production path includes scatter/divides, density repair, divergence, velocity-pressure sweeps, gradients, G2P and collision. Sorting and other optional diagnostic paths are not a comprehensive device trace.

The probe's separate `admission` object records numerical capacity/grain-limit blocked ticks and nominal/admitted pump water. These read-only counters distinguish an apparently fast but starved plant from successful sustained processing; they do not alter admission or physical state.

Device pass time excludes buffer clears/copies, queueing, CPU encoding/uploads, required readback, DEM and terrain work. It is not frame time, and its sum must not be treated as a complete wall-time decomposition. Normal execution remains the throughput reference. The separate `serialized_stage_ms` report is the old `SLUICE_PROFILE_STAGES` diagnostic, which inserts blocking fences; the two modes are mutually exclusive. Timing observations are evidence about this workload and adapter, not universal performance guarantees.

The profiler requests `TIMESTAMP_QUERY` only when explicitly enabled and rejects unsupported adapters. Normal play requests no timestamp feature and allocates no timer buffers. A fixed 128-query allocation holds at most 64 pass pairs. The resolve/staging buffers are each 1 KiB, one batch is mapped at a time, and collection has a bounded diagnostic timeout. Query reuse requires collection. Zero-only batches and backwards intervals fail rather than produce apparently valid timings.

Real-device controls cover direct API counters, ordered substantial work, repeated query/readback reuse and the final production solver on young and accumulated-bed saves. Tested encoder-boundary and side-band-marker prototypes produced invalid intervals on the local adapter and were rejected. Only timestamps attached directly to the actual compute passes are shipped. See the [measured stage report](../playtests/2026-10-22-gpu-throughput.md).

The production P2G scatter also skips atomic additions whose **already quantised integer value is zero**. The fixed-point scale remains 100,000; interpolation, mass/volume weights, counts, array layout and subsequent passes are unchanged. Integer zero is an exact additive identity, and scatter does not consume the atomic return value. Both momentum and weight writes are checked independently; phase-count increments remain.

`SLUICE_REFERENCE_P2G_ATOMICS=1` compiles the untiled scatter with those writes retained for A/B controls. The tiled shader receives no unrecognised override. Real-GPU tests compare every raw integer sum, weight and phase-count entry for stationary/moving mixtures, legacy volume weights, tiny gold, domain-edge support, odd MAC dimensions, a partial workgroup, paired transfers and water-only transfers. Full fixed-step save comparisons additionally check against the earlier production shader; only unordered contact-map export arrays are sorted, with no particle reordering or float tolerance.

This optimization reduces a measured GPU stage; it does not establish real-time operation, conservative interphase momentum, calibrated packed-bed rheology or the complete realistic mining game. Dense DEM contacts and other fluid work remain substantial costs. Preserve those model and acceptance requirements when selecting the next optimization.
