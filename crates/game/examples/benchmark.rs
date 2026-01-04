//! GPU FLIP 3D Benchmark
//!
//! Run with: cargo run --release --example benchmark

use std::time::Instant;

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

const GRID_WIDTH: u32 = 64;
const GRID_HEIGHT: u32 = 32;
const GRID_DEPTH: u32 = 32;
const CELL_SIZE: f32 = 0.05;
const MAX_PARTICLES: usize = 200_000;
const TARGET_PARTICLES: usize = 150_000;

const WARMUP_STEPS: u32 = 5;
const BENCH_STEPS: u32 = 60;
const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = -9.8;

fn main() {
    println!("=== GPU FLIP 3D Benchmark ===");
    println!(
        "Grid: {}x{}x{}, cell_size: {}",
        GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE
    );
    println!("Target particles: {}", TARGET_PARTICLES);

    let (device, queue) = create_device();
    let mut gpu_flip = GpuFlip3D::new(
        &device,
        GRID_WIDTH,
        GRID_HEIGHT,
        GRID_DEPTH,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    let (positions, velocities, c_matrices) = spawn_particles();
    gpu_flip.upload_particles(&queue, &positions, &velocities, &c_matrices);

    let cell_types = build_cell_types();

    for _ in 0..WARMUP_STEPS {
        gpu_flip.step(
            &device,
            &queue,
            &cell_types,
            None,
            DT,
            GRAVITY,
            0.0,
            30,
        );
    }

    let start = Instant::now();
    for _ in 0..BENCH_STEPS {
        gpu_flip.step(
            &device,
            &queue,
            &cell_types,
            None,
            DT,
            GRAVITY,
            0.0,
            30,
        );
    }
    let elapsed = start.elapsed();
    let ms_per_step = elapsed.as_secs_f64() * 1000.0 / BENCH_STEPS as f64;
    let steps_per_sec = BENCH_STEPS as f64 / elapsed.as_secs_f64();

    println!(
        "Steps: {}, avg: {:.3} ms/step, {:.1} steps/sec",
        BENCH_STEPS, ms_per_step, steps_per_sec
    );

    let final_positions = gpu_flip.download_positions(&device, &queue);
    let finite_count = final_positions.iter().filter(|p| p.is_finite()).count();
    println!(
        "Final positions: {} / {} finite",
        finite_count,
        final_positions.len()
    );
    println!("=== Done ===");
}

fn create_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(async {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter")
    });

    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;
    limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

    pollster::block_on(async {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Benchmark Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device")
    })
}

fn build_cell_types() -> Vec<u32> {
    let width = GRID_WIDTH as usize;
    let height = GRID_HEIGHT as usize;
    let depth = GRID_DEPTH as usize;
    let mut cell_types = vec![1u32; width * height * depth];

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                if i == 0 || j == 0 || k == 0 || i + 1 == width || j + 1 == height || k + 1 == depth {
                    let idx = k * width * height + j * width + i;
                    cell_types[idx] = 2;
                }
            }
        }
    }

    cell_types
}

fn spawn_particles() -> (Vec<Vec3>, Vec<Vec3>, Vec<Mat3>) {
    let target = TARGET_PARTICLES.min(MAX_PARTICLES);
    let mut positions = Vec::with_capacity(target);
    let mut velocities = Vec::with_capacity(target);
    let mut c_matrices = Vec::with_capacity(target);

    let max_i = GRID_WIDTH.saturating_sub(2) as usize;
    let max_j = (GRID_HEIGHT / 2).max(2) as usize;
    let max_k = GRID_DEPTH.saturating_sub(2) as usize;

    'outer: for k in 1..max_k {
        for j in 1..max_j {
            for i in 1..max_i {
                let pos = Vec3::new(
                    (i as f32 + 0.5) * CELL_SIZE,
                    (j as f32 + 0.5) * CELL_SIZE,
                    (k as f32 + 0.5) * CELL_SIZE,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);

                if positions.len() >= target {
                    break 'outer;
                }
            }
        }
    }

    (positions, velocities, c_matrices)
}
