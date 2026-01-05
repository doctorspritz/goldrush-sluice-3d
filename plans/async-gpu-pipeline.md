# Plan: Async GPU Pipeline (Deep Optimization)

## Goal
Get 100k+ particles at 60 FPS by eliminating GPU stalls.

## Current Bottleneck Analysis

Per-frame blocking operations:
1. `queue.write_buffer()` - blocks until upload complete
2. `queue.submit()` - blocks until GPU finishes
3. `buffer.slice().map_async()` + `device.poll(Wait)` - blocks for download

These create a serial pipeline: CPU waits → GPU works → CPU waits → repeat.

## Solution: Async Double-Buffered Pipeline

### Architecture

```
Frame N:   [Upload N] [GPU Compute N] [Download N]
Frame N+1:            [Upload N+1] [GPU Compute N+1] [Download N+1]

With double buffering:
Frame N:   [Upload N] [GPU Compute N] ............... [Use Results N]
Frame N+1:            [Upload N+1] [GPU Compute N+1] [Download N]
                                                      ↑ non-blocking!
```

CPU works on frame N+1 while GPU finishes frame N. One frame of latency, but no stalls.

### Implementation

#### 1. Double Buffer Structure

In `flip_3d.rs`, add:
```rust
struct FrameBuffers {
    positions: Arc<wgpu::Buffer>,
    velocities: Arc<wgpu::Buffer>,
    c_col0: Arc<wgpu::Buffer>,
    c_col1: Arc<wgpu::Buffer>,
    c_col2: Arc<wgpu::Buffer>,
    // Staging buffers for async readback
    positions_staging: wgpu::Buffer,
    velocities_staging: wgpu::Buffer,
    c_staging: [wgpu::Buffer; 3],
}

pub struct GpuFlip3D {
    // ... existing fields ...
    frame_buffers: [FrameBuffers; 2],
    current_frame: usize,
    pending_readback: Option<usize>,
}
```

#### 2. Async Readback

Replace blocking `download()` with:
```rust
fn start_async_readback(&mut self, encoder: &mut wgpu::CommandEncoder, frame_idx: usize) {
    let fb = &self.frame_buffers[frame_idx];
    encoder.copy_buffer_to_buffer(&fb.positions, 0, &fb.positions_staging, 0, size);
    encoder.copy_buffer_to_buffer(&fb.velocities, 0, &fb.velocities_staging, 0, size);
    // ... C matrices ...
}

fn poll_readback(&mut self, device: &wgpu::Device) -> Option<ReadbackData> {
    if let Some(frame_idx) = self.pending_readback {
        let fb = &self.frame_buffers[frame_idx];
        // Non-blocking poll
        device.poll(wgpu::Maintain::Poll);

        // Check if mapped
        if fb.positions_staging.slice(..).get_mapped_range().is_ok() {
            // Read data, unmap, return
            self.pending_readback = None;
            return Some(data);
        }
    }
    None
}
```

#### 3. Frame Flow

```rust
pub fn step_async(&mut self, ...) {
    let write_frame = self.current_frame;
    let read_frame = 1 - self.current_frame;

    // 1. Check if previous frame's readback is ready (non-blocking)
    if let Some(data) = self.poll_readback(device) {
        // Update CPU-side particle data from previous frame
        self.apply_readback(data, positions, velocities, c_matrices);
    }

    // 2. Upload current frame's data
    self.upload_particles(queue, write_frame, positions, velocities, c_matrices);

    // 3. Run GPU compute
    let mut encoder = device.create_command_encoder(...);
    self.run_simulation(&mut encoder, write_frame, ...);

    // 4. Start async readback for current frame
    self.start_async_readback(&mut encoder, write_frame);
    queue.submit([encoder.finish()]);

    // 5. Request async map
    self.frame_buffers[write_frame].positions_staging.slice(..).map_async(Read, ...);
    self.pending_readback = Some(write_frame);

    // 6. Swap frames
    self.current_frame = read_frame;
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `crates/game/src/gpu/flip_3d.rs` | Add double buffering, async readback |
| `crates/game/examples/industrial_sluice.rs` | Use new `step_async()` API |

### Testing

```bash
cargo run --release --example industrial_sluice
```

Verify:
- FPS significantly higher (target: 50+ at 100k particles)
- AvgVel still non-zero (physics works)
- One frame of visual latency is acceptable

### Expected Results

| Particles | Current FPS | Expected FPS |
|-----------|-------------|--------------|
| 50,000 | ~32 | 55+ |
| 100,000 | ~20 | 45+ |
| 200,000 | ~12 | 30+ |

The key insight: GPU is already fast, we're just waiting for it unnecessarily.
