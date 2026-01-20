//! Level 2: Flow Over Riffles Test
//!
//! Water flows over riffle obstacles using SDF collision.
//! Tests solid boundary enforcement with complex geometry.
//!
//! Run with: cargo run --example test_level_2 --release

use game::gpu::flip_3d::GpuFlip3D;
use game::sluice_geometry::SluiceConfig;
use game::test_harness::{levels::Level2Riffles, SimTest, TestMetrics};
use glam::{Mat3, Vec3};
use pollster::block_on;
use std::time::Instant;

// Test configuration - smaller scale for faster testing
const CELL_SIZE: f32 = 0.02;
const GRID_WIDTH: usize = 80; // 1.6m
const GRID_HEIGHT: usize = 30; // 0.6m
const GRID_DEPTH: usize = 20; // 0.4m
const MAX_PARTICLES: usize = 100_000;

const GRAVITY: f32 = -9.8;
const FLOW_ACCEL: f32 = 1.5; // Downstream acceleration
const PRESSURE_ITERS: u32 = 60;

// Emission
const EMIT_RATE: usize = 100; // Particles per frame
const EMIT_START_X: f32 = 0.1; // Where to emit (left side)

fn main() {
    let test = Level2Riffles;

    println!("\n{}", "=".repeat(60));
    println!(" {}", test.name());
    println!("{}", "=".repeat(60));
    println!("\n{}\n", test.description());

    // Initialize GPU
    let (device, queue) = block_on(init_gpu());

    // Build sluice geometry with riffles
    let sluice_config = SluiceConfig {
        grid_width: GRID_WIDTH,
        grid_height: GRID_HEIGHT,
        grid_depth: GRID_DEPTH,
        cell_size: CELL_SIZE,
        floor_height_left: 8, // Sloped floor
        floor_height_right: 3,
        riffle_spacing: 12,
        riffle_height: 3,
        riffle_thickness: 2,
        riffle_start_x: 15,
        riffle_end_pad: 10,
        wall_margin: 5,
        exit_width_fraction: 0.8,
        exit_height: 6,
        ..Default::default()
    };

    // Build cell types and SDF from config
    let cell_count = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
    let mut cell_types = vec![0u32; cell_count]; // 0 = air
    let mut sdf = vec![1.0f32; cell_count];

    // Mark solid cells and compute SDF
    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;

                if sluice_config.is_solid(i, j, k) {
                    cell_types[idx] = 2; // Solid
                    sdf[idx] = -CELL_SIZE; // Inside solid
                } else {
                    // Compute distance to nearest solid (simplified SDF)
                    let mut min_dist = f32::MAX;

                    // Check neighbors for solid cells
                    for dk in -2i32..=2 {
                        for dj in -2i32..=2 {
                            for di in -2i32..=2 {
                                let ni = (i as i32 + di) as usize;
                                let nj = (j as i32 + dj) as usize;
                                let nk = (k as i32 + dk) as usize;

                                if ni < GRID_WIDTH && nj < GRID_HEIGHT && nk < GRID_DEPTH {
                                    if sluice_config.is_solid(ni, nj, nk) {
                                        let dist = ((di * di + dj * dj + dk * dk) as f32).sqrt()
                                            * CELL_SIZE;
                                        min_dist = min_dist.min(dist);
                                    }
                                }
                            }
                        }
                    }

                    sdf[idx] = if min_dist < f32::MAX {
                        min_dist
                    } else {
                        CELL_SIZE * 2.0
                    };
                }
            }
        }
    }

    // Count solid cells for verification
    let solid_count = cell_types.iter().filter(|&&t| t == 2).count();
    println!("Sluice geometry: {} solid cells", solid_count);

    // Create FLIP simulation
    let mut flip = GpuFlip3D::new(
        &device,
        GRID_WIDTH as u32,
        GRID_HEIGHT as u32,
        GRID_DEPTH as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Particle storage
    let mut positions: Vec<Vec3> = Vec::new();
    let mut velocities: Vec<Vec3> = Vec::new();
    let mut densities: Vec<f32> = Vec::new();
    let mut c_matrices: Vec<Mat3> = Vec::new();

    // Run simulation
    let invariants = test.invariants();
    let total_frames = test.run_frames();
    let dt = 1.0 / 60.0;

    let mut metrics = TestMetrics::default();
    let mut failures: Vec<String> = Vec::new();
    let start = Instant::now();

    println!(
        "Running {} frames with continuous emission...\n",
        total_frames
    );

    for frame in 0..total_frames {
        // Emit new particles at inlet
        if positions.len() < MAX_PARTICLES - EMIT_RATE {
            emit_particles(
                &mut positions,
                &mut velocities,
                &mut densities,
                &mut c_matrices,
                &sluice_config,
                EMIT_RATE,
            );
        }

        // Remove particles that exit the domain
        remove_exited_particles(
            &mut positions,
            &mut velocities,
            &mut densities,
            &mut c_matrices,
            &sluice_config,
        );

        if positions.is_empty() {
            continue;
        }

        // Track initial count for this frame
        if frame == 0 {
            metrics.particle_count_start = positions.len() as u32;
        }

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
            FLOW_ACCEL,
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

        // Check particles in solid
        for (i, pos) in positions.iter().enumerate() {
            let ix = (pos.x / CELL_SIZE) as usize;
            let iy = (pos.y / CELL_SIZE) as usize;
            let iz = (pos.z / CELL_SIZE) as usize;

            if ix < GRID_WIDTH && iy < GRID_HEIGHT && iz < GRID_DEPTH {
                let idx = iz * GRID_WIDTH * GRID_HEIGHT + iy * GRID_WIDTH + ix;
                if idx < sdf.len() && sdf[idx] < -CELL_SIZE * 0.5 {
                    metrics.particles_in_solid += 1;
                }
            }
        }

        // Check invariants periodically
        if frame % 60 == 0 || frame == total_frames - 1 {
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
            let fps = (frame + 1) as f32 / elapsed.max(0.001);
            print!(
                "\rFrame {}/{} | FPS: {:.1} | Particles: {} | Max vel: {:.2} | In solid: {}   ",
                frame + 1,
                total_frames,
                fps,
                positions.len(),
                metrics.max_velocity,
                metrics.particles_in_solid
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

fn emit_particles(
    positions: &mut Vec<Vec3>,
    velocities: &mut Vec<Vec3>,
    densities: &mut Vec<f32>,
    c_matrices: &mut Vec<Mat3>,
    config: &SluiceConfig,
    count: usize,
) {
    let start_y = config.floor_height_left as f32 * config.cell_size + config.cell_size * 2.0;
    let end_y = start_y + config.cell_size * 6.0;
    let start_z = config.cell_size * 2.0;
    let end_z = (config.grid_depth - 2) as f32 * config.cell_size;

    for _ in 0..count {
        let x = EMIT_START_X + rand_f32() * config.cell_size * 2.0;
        let y = start_y + rand_f32() * (end_y - start_y);
        let z = start_z + rand_f32() * (end_z - start_z);

        positions.push(Vec3::new(x, y, z));
        velocities.push(Vec3::new(0.5, 0.0, 0.0)); // Initial downstream velocity
        densities.push(1.0); // Water
        c_matrices.push(Mat3::ZERO);
    }
}

fn remove_exited_particles(
    positions: &mut Vec<Vec3>,
    velocities: &mut Vec<Vec3>,
    densities: &mut Vec<f32>,
    c_matrices: &mut Vec<Mat3>,
    config: &SluiceConfig,
) {
    let exit_x = (config.grid_width - 2) as f32 * config.cell_size;

    let mut i = 0;
    while i < positions.len() {
        if positions[i].x > exit_x || positions[i].y < 0.0 {
            positions.swap_remove(i);
            velocities.swap_remove(i);
            densities.swap_remove(i);
            c_matrices.swap_remove(i);
        } else {
            i += 1;
        }
    }
}

fn rand_f32() -> f32 {
    // Simple LCG for reproducible randomness
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
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

    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;
    limits.max_storage_buffer_binding_size = 1024 * 1024 * 256;

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
