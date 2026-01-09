<!-- STATUS: Work In Progress -->

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
\n\n---\n\n# Safe GPU Transfer Optimization (Merged)\n\n_The following was merged from safe-gpu-optimization.md_\n
# Plan: Safe GPU Transfer Optimization

## Problem Analysis

Current per-frame GPU transfers in `flip_3d.rs`:

### Uploads (CPU → GPU) - Every Frame:
1. `upload_cell_types` - Full grid cell types
2. `p2g.upload_particles` - positions, velocities, C matrices (5 buffers)
3. `g2p.upload_particles` - SAME data again (5 buffers) ← DUPLICATE!
4. `sdf_buffer` - Full SDF grid ← STATIC DATA, NEVER CHANGES!
5. Various params buffers (small, fine)

### Downloads (GPU → CPU) - Every Frame:
1. `position_delta_buffer` - Density correction deltas
2. `g2p.download` - velocities, C matrices (4 buffers)
3. `positions_buffer` - Final positions

### What's Wasteful:
1. **SDF uploaded every frame** - It's computed once from static geometry
2. **P2G and G2P upload same particle data** - Should share buffers
3. **Solid cell types rebuilt every frame** - Only fluid cells change

### What MUST Stay (FLIP/APIC correctness):
- Velocity download - Next frame's P2G needs updated velocities
- C matrix download - APIC angular momentum preservation
- Position download - CPU emitter and exit detection need positions
- Cell types upload - Fluid cells change as particles move

## Safe Optimizations

### 1. Cache SDF Buffer (Easy, Big Win)

The SDF is computed once from static solid geometry and never changes.

**Changes:**
- Add `sdf_cached: bool` field to GpuFlip3D
- Add `upload_sdf()` method called once at creation
- In `step()`, skip SDF upload if already cached

**Files:** `crates/game/src/gpu/flip_3d.rs`

### 2. Cache Solid Cell Types (Medium, Good Win)

Solid cells (type=2) never change. Only fluid cells (type=1) change based on particle positions.

**Changes:**
- Add `solid_cell_mask: Vec<bool>` to GpuFlip3D
- Compute once at creation from initial cell_types
- In `step()`, only update fluid cell markers

**Files:** `crates/game/src/gpu/flip_3d.rs`, `crates/game/examples/box_3d_test.rs`

### 3. Share Particle Buffers Between P2G and G2P (Medium)

Currently P2G creates its own particle buffers and G2P creates its own.
Both upload the same data. G2P should read directly from P2G's buffers.

**Changes:**
- G2P constructor takes P2G's particle buffers as input
- Remove duplicate buffer creation in G2P
- G2P's bind group references P2G's buffers

**Files:** `crates/game/src/gpu/p2g_3d.rs`, `crates/game/src/gpu/g2p_3d.rs`, `crates/game/src/gpu/flip_3d.rs`

### 4. Optimize Cell Type Computation in Example (Easy)

The example rebuilds cell_types by iterating ALL grid cells every frame.
Should cache solid mask and only iterate particles for fluid cells.

**Changes in box_3d_test.rs:**
```rust
// Once at init:
self.solid_mask = compute_solid_mask(&self.sim.grid);

// Each frame:
self.cell_types.copy_from_slice(&self.solid_mask); // Start with solids
for p in &self.sim.particles.list {
    // Mark fluid cells from particles
}
```

## What NOT To Do

1. ❌ Remove velocity downloads - Breaks FLIP momentum transfer
2. ❌ Remove C matrix downloads - Breaks APIC angular momentum
3. ❌ Remove position downloads - CPU emitter needs current positions
4. ❌ "GPU-owned particles" without GPU-side emitter - Breaks particle spawning

## Implementation Order

1. **Cache SDF** - Simplest, guaranteed safe, easy to verify
2. **Share P2G/G2P buffers** - Eliminates 5 buffer copies per frame
3. **Cache solid cells** - Reduces CPU work in cell type rebuild
4. **Verify** - Run box_3d_test for 5+ minutes, check velocities are non-zero

## Testing

After each change:
```bash
cargo run --release --example box_3d_test
```

Verify:
- AvgVel is NON-ZERO (water flowing)
- FPS is same or better
- Visual: water flows, doesn't become jello
- Run for at least 2 minutes to catch delayed issues

## Expected Results

| Optimization | Expected FPS Gain |
|-------------|-------------------|
| Cache SDF | +5-10% |
| Share P2G/G2P buffers | +10-15% |
| Cache solid cells | +5% |
| **Total** | **+20-30%** |

Conservative estimate: 55 FPS → 65-70 FPS
