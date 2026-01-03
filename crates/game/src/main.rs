//! Goldrush Fluid Miner - Headless 2D GPU Simulation Driver
//!
//! Runs GPU compute-only FLIP/PIC updates without rendering or CPU solvers.

mod gpu;

use crate::gpu::GpuFlip2D;
use std::time::Instant;

const GRID_WIDTH: u32 = 160;
const GRID_HEIGHT: u32 = 96;
const CELL_SIZE: f32 = 0.02;
const MAX_PARTICLES: usize = 160_000;

const DT: f32 = 1.0 / 60.0;
const STEPS: u32 = 600;
const PRESSURE_ITERS: u32 = 60;
const GRAVITY: f32 = -9.8;

fn main() {
    env_logger::init();

    let (device, queue) = init_gpu();
    let mut sim = GpuFlip2D::new(&device, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, MAX_PARTICLES);

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    spawn_block(
        &mut positions,
        &mut velocities,
        [CELL_SIZE * 6.0, CELL_SIZE * 6.0],
        (32, 24),
        CELL_SIZE * 0.9,
        [1.5, 0.0],
    );
    sim.upload_particles(&queue, &positions, &velocities);

    println!(
        "GPU-only 2D sim start: {} particles, grid {}x{}",
        positions.len(),
        GRID_WIDTH,
        GRID_HEIGHT
    );

    let start = Instant::now();
    for step in 0..STEPS {
        sim.step(&device, &queue, DT, GRAVITY, PRESSURE_ITERS);
        device.poll(wgpu::Maintain::Wait);

        if step % 60 == 0 {
            let elapsed = start.elapsed().as_secs_f32();
            println!("step {:4} | t={:5.2}s", step, elapsed);
        }
    }

    println!("GPU-only 2D sim done in {:.2}s", start.elapsed().as_secs_f32());
}

fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to find GPU adapter");

    println!("Using GPU: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    (device, queue)
}

fn spawn_block(
    positions: &mut Vec<[f32; 2]>,
    velocities: &mut Vec<[f32; 2]>,
    origin: [f32; 2],
    dims: (u32, u32),
    spacing: f32,
    initial_velocity: [f32; 2],
) {
    for j in 0..dims.1 {
        for i in 0..dims.0 {
            if positions.len() >= MAX_PARTICLES {
                return;
            }
            let pos = [
                origin[0] + i as f32 * spacing,
                origin[1] + j as f32 * spacing,
            ];
            positions.push(pos);
            velocities.push(initial_velocity);
        }
    }
}
