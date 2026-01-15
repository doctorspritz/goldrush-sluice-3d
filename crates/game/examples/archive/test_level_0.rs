//! Level 0: Dam Break Test
//!
//! Pure FLIP water simulation - a block of water collapses in a box.
//! Tests basic P2G, pressure solve, G2P cycle.
//!
//! Run with: cargo run --example test_level_0 --release

use game::gpu::flip_3d::GpuFlip3D;
use game::test_harness::{levels::Level0DamBreak, Invariant, SimTest, TestMetrics};
use glam::{Mat3, Vec3};
use pollster::block_on;
use std::time::Instant;

// Test configuration
const CELL_SIZE: f32 = 0.05;
const GRID_WIDTH: usize = 32;
const GRID_HEIGHT: usize = 32;
const GRID_DEPTH: usize = 32;
const MAX_PARTICLES: usize = 50_000;

const GRAVITY: f32 = -9.8;
const PRESSURE_ITERS: u32 = 40;

fn main() {
    let test = Level0DamBreak;

    println!("\n{}", "=".repeat(60));
    println!(" {}", test.name());
    println!("{}", "=".repeat(60));
    println!("\n{}\n", test.description());

    // Initialize GPU
    let (device, queue) = block_on(init_gpu());

    // Create FLIP simulation
    let mut flip = GpuFlip3D::new(
        &device,
        GRID_WIDTH as u32,
        GRID_HEIGHT as u32,
        GRID_DEPTH as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Create initial dam of water particles (quarter of the domain)
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    // Spawn water block in corner
    let particles_per_cell = 8;
    let spacing = CELL_SIZE / (particles_per_cell as f32).cbrt();

    for k in 2..GRID_DEPTH / 2 {
        for j in 2..GRID_HEIGHT * 3 / 4 {
            for i in 2..GRID_WIDTH / 2 {
                for pi in 0..2 {
                    for pj in 0..2 {
                        for pk in 0..2 {
                            let x = (i as f32 + 0.25 + pi as f32 * 0.5) * CELL_SIZE;
                            let y = (j as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE;
                            let z = (k as f32 + 0.25 + pk as f32 * 0.5) * CELL_SIZE;

                            positions.push(Vec3::new(x, y, z));
                            velocities.push(Vec3::ZERO);
                            densities.push(1.0); // Water density
                            c_matrices.push(Mat3::ZERO);
                        }
                    }
                }
            }
        }
    }

    let initial_count = positions.len();
    println!("Initial particles: {}", initial_count);

    // Build cell types (all air except walls)
    let cell_count = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
    let mut cell_types = vec![0u32; cell_count]; // 0 = air

    // Mark floor as solid (j = 0)
    for k in 0..GRID_DEPTH {
        for i in 0..GRID_WIDTH {
            let idx = k * GRID_WIDTH * GRID_HEIGHT + 0 * GRID_WIDTH + i;
            cell_types[idx] = 2; // Solid
        }
    }

    // Create SDF (simple box - distance from walls)
    let mut sdf = vec![1.0f32; cell_count];
    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;

                // Distance from each wall
                let dist_floor = j as f32 * CELL_SIZE;
                let dist_ceiling = (GRID_HEIGHT - 1 - j) as f32 * CELL_SIZE;
                let dist_left = i as f32 * CELL_SIZE;
                let dist_right = (GRID_WIDTH - 1 - i) as f32 * CELL_SIZE;
                let dist_front = k as f32 * CELL_SIZE;
                let dist_back = (GRID_DEPTH - 1 - k) as f32 * CELL_SIZE;

                let min_dist = dist_floor
                    .min(dist_ceiling)
                    .min(dist_left)
                    .min(dist_right)
                    .min(dist_front)
                    .min(dist_back);

                sdf[idx] = min_dist - CELL_SIZE * 0.5; // Negative inside walls
            }
        }
    }

    // Run simulation
    let invariants = test.invariants();
    let total_frames = test.run_frames();
    let dt = 1.0 / 60.0;

    let mut metrics = TestMetrics {
        frame_count: 0,
        particle_count_start: initial_count as u32,
        particle_count_end: initial_count as u32,
        ..Default::default()
    };

    let mut failures: Vec<String> = Vec::new();
    let start = Instant::now();

    println!("\nRunning {} frames...", total_frames);

    for frame in 0..total_frames {
        // Step simulation
        flip.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            Some(&sdf),
            None, // No bed height
            dt,
            GRAVITY,
            0.0, // No flow acceleration
            PRESSURE_ITERS,
        );

        // Update metrics
        metrics.frame_count = frame + 1;
        metrics.particle_count_end = positions.len() as u32;

        // Check for NaN
        for pos in &positions {
            if pos.x.is_nan() || pos.y.is_nan() || pos.z.is_nan() {
                metrics.nan_detected = true;
                break;
            }
        }

        // Track max velocity
        for vel in &velocities {
            let speed = vel.length();
            if speed > metrics.max_velocity {
                metrics.max_velocity = speed;
            }
        }

        // Check particles in solid (floor)
        for pos in &positions {
            if pos.y < CELL_SIZE {
                metrics.particles_in_solid += 1;
            }
        }

        // Check invariants periodically
        if frame % 30 == 0 || frame == total_frames - 1 {
            for inv in &invariants {
                if let Err(e) = inv.check(&metrics, None) {
                    let msg = format!("Frame {}: {}", frame, e);
                    if !failures.contains(&msg) {
                        failures.push(msg.clone());
                        eprintln!("FAIL: {}", msg);
                    }
                }
            }

            // Progress report
            let elapsed = start.elapsed().as_secs_f32();
            let fps = frame as f32 / elapsed.max(0.001);
            print!(
                "\rFrame {}/{} | FPS: {:.1} | Particles: {} | Max vel: {:.2}",
                frame,
                total_frames,
                fps,
                positions.len(),
                metrics.max_velocity
            );
        }
    }

    metrics.elapsed_seconds = start.elapsed().as_secs_f32();
    println!("\n");

    // Final report
    println!("{}", "=".repeat(60));
    if failures.is_empty() {
        println!(" PASS: {}", test.name());
    } else {
        println!(" FAIL: {}", test.name());
        println!("\nFailures:");
        for f in &failures {
            println!("  - {}", f);
        }
    }
    println!("{}", "=".repeat(60));

    println!("\nMetrics:");
    println!("  Frames: {}", metrics.frame_count);
    println!(
        "  Particles: {} -> {}",
        metrics.particle_count_start, metrics.particle_count_end
    );
    println!("  Max velocity: {:.2} m/s", metrics.max_velocity);
    println!("  NaN detected: {}", metrics.nan_detected);
    println!("  Particles in solid: {}", metrics.particles_in_solid);
    println!(
        "  Elapsed: {:.2}s ({:.1} FPS)",
        metrics.elapsed_seconds,
        metrics.frame_count as f32 / metrics.elapsed_seconds
    );

    if !failures.is_empty() {
        std::process::exit(1);
    }
}

async fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find a suitable GPU adapter");

    // Request higher limits for compute shaders
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;
    limits.max_storage_buffer_binding_size = 1024 * 1024 * 256; // 256 MB

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device")
}
