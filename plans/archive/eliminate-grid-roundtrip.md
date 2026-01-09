# Plan: Eliminate Grid CPU-GPU Round-Trip

## Problem
Every frame downloads 6 grid buffers (~2MB) to CPU then re-uploads to G2P. This blocks the GPU pipeline and kills performance.

Current flow in `flip_3d.rs` lines 1149-1177:
```rust
// BAD: Downloads grids to CPU
self.p2g.download(device, queue, &mut grid_u, &mut grid_v, &mut grid_w);
Self::read_buffer(device, queue, &self.grid_u_old_buffer, &mut grid_u_old);
// ... then re-uploads to G2P
self.g2p.upload(queue, positions, velocities, c_matrices, &grid_u, ...);
```

## Solution
G2P should reference P2G's grid buffers directly via constructor injection. No downloads needed.

## Changes Required

### 1. `crates/game/src/gpu/g2p_3d.rs`

Modify constructor to accept external grid buffer references:

```rust
pub fn new(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    depth: u32,
    max_particles: usize,
    // NEW: Accept P2G's grid buffers directly
    grid_u_buffer: &wgpu::Buffer,
    grid_v_buffer: &wgpu::Buffer,
    grid_w_buffer: &wgpu::Buffer,
    grid_u_old_buffer: &wgpu::Buffer,
    grid_v_old_buffer: &wgpu::Buffer,
    grid_w_old_buffer: &wgpu::Buffer,
) -> Self {
```

- Remove internal grid buffer creation (lines ~132-175)
- Store references or use the passed buffers directly in bind group creation
- Since wgpu buffers can't be stored as references, create bind group using the passed buffers

Add `upload_particles` method that only uploads particle data (no grids):

```rust
pub fn upload_particles(
    &self,
    queue: &wgpu::Queue,
    positions: &[Vec3],
    velocities: &[Vec3],
    c_matrices: &[Mat3],
    cell_size: f32,
    dt: f32,
) -> u32 {
    // Upload only particle positions, velocities, C matrices
    // Grid data already on GPU from P2G
    // Return particle count
}
```

### 2. `crates/game/src/gpu/flip_3d.rs`

In `new()`, pass P2G's grid buffers to G2P constructor:

```rust
let g2p = GpuG2p3D::new(
    device,
    width,
    height,
    depth,
    max_particles,
    &p2g.grid_u_buffer,
    &p2g.grid_v_buffer,
    &p2g.grid_w_buffer,
    &grid_u_old_buffer,
    &grid_v_old_buffer,
    &grid_w_old_buffer,
);
```

In `step()`, replace the download/upload cycle (lines ~1149-1177) with:

```rust
// Grid data stays on GPU - just upload particle data
let g2p_count = self.g2p.upload_particles(
    queue,
    positions,
    velocities,
    c_matrices,
    self.cell_size,
    dt,
);
```

Also remove debug readbacks (lines ~1222-1267) - they're blocking calls that kill performance.

## Testing

After changes:
1. `cargo build --release` should succeed
2. `cargo run --release --example box_3d_test` should show improved FPS
3. Particles should still behave correctly (flow, collision)

## Expected Result

Eliminating 6 buffer downloads per frame should give 3-5x FPS improvement.
