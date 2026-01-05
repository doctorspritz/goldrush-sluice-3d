# Plan: Headless GPU Benchmark

## Goal
Create a headless benchmark that measures GPU FLIP simulation performance without requiring a window.

## Implementation

Create `crates/game/examples/benchmark.rs`:

```rust
//! Headless GPU FLIP Benchmark
//!
//! Measures simulation performance without rendering.
//! Run with: cargo run --example benchmark --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use std::time::Instant;

const GRID_WIDTH: usize = 32;
const GRID_HEIGHT: usize = 32;
const GRID_DEPTH: usize = 16;
const CELL_SIZE: f32 = 0.05;
const MAX_PARTICLES: usize = 50000;
const WARMUP_FRAMES: u32 = 50;
const BENCHMARK_FRAMES: u32 = 200;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // Create headless wgpu instance
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");

    println!("GPU: {}", adapter.get_info().name);
    println!("Grid: {}x{}x{}", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH);

    // Create GPU simulation
    let mut gpu = GpuFlip3D::new(&device, GRID_WIDTH as u32, GRID_HEIGHT as u32, GRID_DEPTH as u32, MAX_PARTICLES);

    // Spawn particles
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();

    // Spawn water block
    let spawn_count = 5000;
    for i in 0..spawn_count {
        let x = (i % 20) as f32 * CELL_SIZE * 0.5 + CELL_SIZE * 5.0;
        let y = (i / 20 % 20) as f32 * CELL_SIZE * 0.5 + CELL_SIZE * 10.0;
        let z = (i / 400) as f32 * CELL_SIZE * 0.5 + CELL_SIZE * 5.0;
        positions.push(Vec3::new(x, y, z));
        velocities.push(Vec3::ZERO);
        c_matrices.push(Mat3::ZERO);
    }

    // Create simple cell types (all fluid interior for benchmark)
    let cell_count = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
    let cell_types: Vec<u8> = (0..cell_count)
        .map(|idx| {
            let i = idx % GRID_WIDTH;
            let j = (idx / GRID_WIDTH) % GRID_HEIGHT;
            let k = idx / (GRID_WIDTH * GRID_HEIGHT);
            if i == 0 || i == GRID_WIDTH - 1 || j == 0 || j == GRID_HEIGHT - 1 || k == 0 || k == GRID_DEPTH - 1 {
                0 // solid
            } else {
                1 // fluid
            }
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;
    let flow_accel = 0.0;
    let pressure_iters = 50;

    println!("Particles: {}", positions.len());
    println!("Warming up ({} frames)...", WARMUP_FRAMES);

    // Warmup
    for _ in 0..WARMUP_FRAMES {
        gpu.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &cell_types,
            dt,
            gravity,
            flow_accel,
            pressure_iters,
        );
    }

    println!("Benchmarking ({} frames)...", BENCHMARK_FRAMES);

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCHMARK_FRAMES {
        gpu.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &cell_types,
            dt,
            gravity,
            flow_accel,
            pressure_iters,
        );
    }
    let elapsed = start.elapsed();

    let fps = BENCHMARK_FRAMES as f64 / elapsed.as_secs_f64();
    let ms_per_frame = elapsed.as_millis() as f64 / BENCHMARK_FRAMES as f64;

    println!("\n=== RESULTS ===");
    println!("Frames: {}", BENCHMARK_FRAMES);
    println!("Total time: {:.2}s", elapsed.as_secs_f64());
    println!("FPS: {:.1}", fps);
    println!("ms/frame: {:.2}", ms_per_frame);
}
```

## Dependencies

Add to `crates/game/Cargo.toml` if not present:
```toml
pollster = "0.3"
```

## Testing

```bash
cargo run --example benchmark --release
```

Should output FPS and ms/frame metrics.
