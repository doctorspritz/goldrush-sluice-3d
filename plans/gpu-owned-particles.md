# Plan: GPU-Owned Particles & Direct Rendering

## Goal
Eliminate CPU-GPU particle transfers by keeping particles on GPU and rendering directly from GPU buffers.

## Current Problem
Every frame:
- P2G uploads 5 buffers (pos, vel, 3x C rows)
- G2P downloads 4 buffers (vel, 3x C rows)
- Density correction reads back position deltas
- Blocking `device.poll(Maintain::Wait)` stalls pipeline

At 1M particles this is ~240 MB/frame transfer.

## Solution

### 1. Keep particles GPU-resident

Modify `GpuFlip3D` to own particle buffers that persist across frames:
- `positions_buffer` - already exists in G2P, make it the single source of truth
- `velocities_buffer` - already exists in G2P
- `c_col0/1/2_buffer` - already exist in G2P

Remove per-frame uploads in P2G - instead, P2G reads from the same buffers G2P writes to.

### 2. Modify step() signature

Change from:
```rust
pub fn step(&self, ..., positions: &mut [Vec3], velocities: &mut [Vec3], c_matrices: &mut [Mat3], ...)
```

To:
```rust
pub fn step(&self, device: &Device, queue: &Queue, cell_types: &[u32], sdf: Option<&[f32]>, dt: f32, gravity: f32, flow_accel: f32, pressure_iters: u32)
```

Particles stay on GPU. Only upload cell_types and SDF when they change.

### 3. Add particle spawn/read methods

```rust
impl GpuFlip3D {
    /// Upload initial particles (call once at start)
    pub fn upload_particles(&self, queue: &Queue, positions: &[Vec3], velocities: &[Vec3], c_matrices: &[Mat3]);

    /// Get particle count
    pub fn particle_count(&self) -> usize;

    /// Read positions back (for debug/export only)
    pub fn download_positions(&self, device: &Device, queue: &Queue) -> Vec<Vec3>;
}
```

### 4. Direct GPU rendering

Modify the renderer to draw particles directly from GPU buffer:

```rust
// In render pass, bind positions_buffer as vertex/storage buffer
render_pass.set_vertex_buffer(1, gpu_flip.positions_buffer().slice(..));
```

The particle shader reads vec4 positions directly - no CPU instance buffer needed.

### 5. Remove blocking readbacks

Delete or gate behind debug flag:
- `read_buffer_vec4` for position deltas
- `g2p.download()`
- Position readback in SDF collision path

## Files to Modify

### `crates/game/src/gpu/flip_3d.rs`
- Change `step()` to not take particle slices
- Add `upload_particles()`, `particle_count()`, `positions_buffer()` methods
- Remove `read_buffer_vec4` call for density correction (apply on GPU)
- Store particle_count as field

### `crates/game/src/gpu/g2p_3d.rs`
- Remove `download()` method or make it debug-only
- Expose `positions_buffer` and `velocities_buffer` for rendering

### `crates/game/src/gpu/p2g_3d.rs`
- Modify to read from shared particle buffers instead of uploading each frame
- Change `upload()` to just set params, not copy particle data

### `crates/game/src/gpu/shaders/density_correct_3d.wgsl`
- Write corrected positions directly to positions buffer (not separate delta buffer)

### `crates/game/examples/benchmark.rs`
- Update to use new API: upload_particles once, then step() without particle args
- Read back only for final verification

### `crates/game/examples/box_3d_test.rs`
- Update to new API
- Render directly from GPU buffer (or keep CPU path for now with periodic readback)

## Testing

```bash
cargo build --release --example benchmark
cargo run --release --example benchmark
```

Should see significant FPS improvement at 100k+ particles.

## Expected Result

- Eliminate ~240 MB/frame CPU-GPU transfer at 1M particles
- Remove blocking `device.poll(Maintain::Wait)` calls
- 2-5x FPS improvement at high particle counts
