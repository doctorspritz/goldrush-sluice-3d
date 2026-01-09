<!-- TODO: Review for current GPU architecture -->

# perf: Next optimization target for GPU FLIP (1M particles)

## Summary
- The next bottleneck is CPU <-> GPU synchronization and transfer, driven by per-frame particle uploads/downloads and blocking readbacks.
- At 1M particles, the current data movement alone is hundreds of MB per frame and forces CPU/GPU stalls via `device.poll(Maintain::Wait)`.
- Once transfers are removed, the dominant GPU costs will be P2G atomic scatter, G2P gather, and pressure iterations.

## Evidence in code
- `GpuFlip3D::step` reads back `position_delta_buffer` via `read_buffer_vec4` and blocks on `device.poll(Maintain::Wait)` before applying deltas on CPU. `crates/game/src/gpu/flip_3d.rs`
- `GpuG2p3D::download` copies velocities + C matrix rows to staging buffers and maps them with `device.poll(Maintain::Wait)` (four maps per frame). `crates/game/src/gpu/g2p_3d.rs`
- Optional SDF collision reads back GPU positions to CPU after compute. `crates/game/src/gpu/flip_3d.rs`
- P2G and G2P each upload full particle state every step, constructing new `Vec` buffers (`positions_padded`, `velocities_padded`, `c_col*`) every frame. `crates/game/src/gpu/p2g_3d.rs`, `crates/game/src/gpu/g2p_3d.rs`
- Large per-frame buffer clears use `queue.write_buffer` with freshly allocated zero `Vec`s for `u_sum`, `u_weight`, `v_sum`, `v_weight`, `w_sum`, `w_weight`, and `particle_count`. `crates/game/src/gpu/p2g_3d.rs`
- Multiple command encoders and `queue.submit` calls per step add CPU overhead and prevent batching. `crates/game/src/gpu/flip_3d.rs`

## Bandwidth rough cut (1M particles)
Per particle per frame (vec4 = 16 bytes):
- P2G upload: 5 * 16 = 80 bytes (pos, vel, 3x C rows)
- G2P upload: 5 * 16 = 80 bytes
- G2P download: 4 * 16 = 64 bytes (vel + 3x C rows)
- Density correction readback: 1 * 16 = 16 bytes
- Optional SDF position readback: 1 * 16 = 16 bytes

Total (no SDF): ~240 MB per frame at 1M particles. At 60 FPS that is ~14 GB/s before any grid traffic, and every readback forces CPU/GPU sync via `device.poll(Maintain::Wait)`.

## Recommendations (priority order)
1. Move particle ownership to GPU and eliminate per-frame upload/download.
   - Keep positions/velocities/C matrices on GPU buffers; only read back for debug snapshots or export.
   - Update rendering to consume GPU buffers directly (storage or vertex buffers) instead of CPU instance builds.
2. Apply density correction and SDF collision fully on GPU.
   - Modify `density_correct_3d.wgsl` to write corrected positions directly into the positions buffer, removing `position_delta_buffer` readback.
   - Keep SDF resident on GPU and update only when it changes.
3. Remove blocking readbacks or make them async.
   - Replace `device.poll(Maintain::Wait)` with async readbacks and a one-frame-late consumer.
   - Gate all readbacks behind a debug flag.
4. Eliminate per-frame CPU zeroing of large buffers.
   - Clear `u_sum`, `u_weight`, `v_sum`, `v_weight`, `w_sum`, `w_weight`, and `particle_count` on GPU via a compute pass or `encoder.clear_buffer`, not `queue.write_buffer` with new zero `Vec`s.
5. Consolidate command encoders.
   - Record P2G -> BC -> grid copy -> forces -> pressure -> G2P -> density projection in a single encoder per step where possible.
6. Shader-level follow-ups once transfers are gone.
   - P2G atomic scatter: investigate binning or per-cell shared memory reduction to reduce atomic contention at high density.
   - G2P gather: precompute 1D weights per component (like P2G) and reduce branch-heavy bounds checks.
   - Pressure solve: switch to multigrid/PCG or lower iteration count; current red-black GS iterations will dominate at larger grids.

## Suggested next experiment
- Disable CPU readbacks (G2P download + density correction readback), keep particles on GPU, and render from GPU buffers.
- Measure FPS at 200k/500k/1M to confirm transfer bottleneck removal before tuning shader kernels.
