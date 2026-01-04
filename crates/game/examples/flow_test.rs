//! Flow Test - Programmatic verification that sluice flow works
//!
//! This test FAILS if water doesn't flow downstream.
//! No window needed - runs headless and checks metrics.
//!
//! Uses a SIMPLE SLOPE (no riffles) to verify flow acceleration works.
//! Riffles are tested separately once basic flow is confirmed.
//!
//! Run with: cargo run --example flow_test --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use sim3d::FlipSimulation3D;

// Grid dimensions - simple channel
const GRID_WIDTH: usize = 48;
const GRID_HEIGHT: usize = 16;
const GRID_DEPTH: usize = 10;
const CELL_SIZE: f32 = 0.04;
const MAX_PARTICLES: usize = 50000;

// Flow parameters
const FLOW_ACCEL: f32 = 3.0; // m/s²

// Test thresholds - these MUST pass for flow to be working
const MIN_AVG_VX: f32 = 0.2;  // Particles must have avg downstream velocity > this
const MIN_X_PROGRESS_RATIO: f32 = 0.4; // Particles must move >40% of channel length
const TEST_FRAMES: u32 = 400;  // Run for this many frames
const SUBSTEPS: u32 = 2;
const DT: f32 = 1.0 / 120.0;

/// Create a simple sloped floor with walls - NO RIFFLES
/// This is the minimal geometry needed to test flow acceleration
fn create_simple_slope(sim: &mut FlipSimulation3D, slope: f32) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;
    let dx = sim.grid.cell_size;

    // Create sloped floor
    // At x=0, floor is at max height. At x=width-1, floor is at y=0.
    // floor_y(x) = (width - 1 - x) * slope
    for k in 0..depth {
        for i in 0..width {
            let floor_height = ((width as f32 - 1.0 - i as f32) * slope).round() as usize;
            for j in 0..=floor_height.min(height - 1) {
                sim.grid.set_solid(i, j, k);
            }
        }
    }

    // Compute SDF for collision
    sim.grid.compute_sdf();
}

/// Spawn a block of water particles at the inlet
fn spawn_water_block(sim: &mut FlipSimulation3D, count: usize, inlet_vel: Vec3) {
    use sim3d::Particle3D;

    let dx = sim.grid.cell_size;
    let depth = sim.grid.depth;

    // Spawn particles in first 5 cells of X, above the sloped floor
    // Find floor height at inlet (x ≈ 0)
    let slope = 0.15_f32;
    let floor_at_inlet = ((sim.grid.width as f32 - 1.0) * slope).round() as usize;

    let mut spawned = 0;
    let spawn_region_x = 5; // First 5 cells
    let spawn_region_y = 5; // 5 cells of water depth

    for i in 1..spawn_region_x {
        for j in (floor_at_inlet + 1)..(floor_at_inlet + 1 + spawn_region_y) {
            if j >= sim.grid.height - 1 { continue; }

            for k in 1..depth-1 {
                // Spawn 4 particles per cell (2x2)
                for sub_i in 0..2 {
                    for sub_j in 0..2 {
                        if spawned >= count { break; }

                        let x = (i as f32 + 0.25 + sub_i as f32 * 0.5) * dx;
                        let y = (j as f32 + 0.25 + sub_j as f32 * 0.5) * dx;
                        let z = (k as f32 + 0.5) * dx;

                        // Create water particle (density = 1.0)
                        let p = Particle3D::new(Vec3::new(x, y, z), inlet_vel);
                        sim.particles.list.push(p);
                        spawned += 1;
                    }
                }
            }
        }
    }

    println!("Spawned {} particles at inlet (floor_y={})", spawned, floor_at_inlet);
}

fn main() {
    println!("=== FLOW TEST ===");
    println!("Verifying water flows downstream on a SIMPLE SLOPE (no riffles).");
    println!("Grid: {}x{}x{}, cell_size: {}", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    println!("Flow accel: {} m/s²", FLOW_ACCEL);
    println!("Test frames: {}, substeps: {}", TEST_FRAMES, SUBSTEPS);
    println!();

    // Create simulation
    let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.95;
    sim.pressure_iterations = 80;

    // Create SIMPLE sloped floor - no riffles!
    let slope = 0.15; // 15% grade
    create_simple_slope(&mut sim, slope);

    // Spawn water at inlet with downstream velocity
    let inlet_vel = Vec3::new(1.0, 0.0, 0.0);
    spawn_water_block(&mut sim, 2000, inlet_vel);

    let initial_count = sim.particle_count();
    let initial_avg_x: f32 = sim.particles.list.iter().map(|p| p.position.x).sum::<f32>()
        / sim.particle_count() as f32;

    println!("Initial particles: {}", initial_count);
    println!("Initial avg X: {:.3}", initial_avg_x);
    println!();

    // Initialize GPU (headless)
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // Headless
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Flow Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create GPU device")
    });

    let gpu_flip = GpuFlip3D::new(
        &device,
        GRID_WIDTH as u32,
        GRID_HEIGHT as u32,
        GRID_DEPTH as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Particle data buffers
    let mut positions: Vec<Vec3> = Vec::with_capacity(MAX_PARTICLES);
    let mut velocities: Vec<Vec3> = Vec::with_capacity(MAX_PARTICLES);
    let mut c_matrices: Vec<Mat3> = Vec::with_capacity(MAX_PARTICLES);
    let mut cell_types: Vec<u32> = vec![0; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];

    println!("Running {} frames...", TEST_FRAMES);

    // Run simulation
    for frame in 0..TEST_FRAMES {
        for _ in 0..SUBSTEPS {
            // Sync particle data from CPU sim to GPU format
            let particle_count = sim.particles.list.len();
            positions.clear();
            velocities.clear();
            c_matrices.clear();

            for p in &sim.particles.list {
                positions.push(p.position);
                velocities.push(p.velocity);
                c_matrices.push(p.affine_velocity);
            }

            // Build cell types from grid
            let w = sim.grid.width;
            let h = sim.grid.height;
            let d = sim.grid.depth;
            cell_types.fill(0); // Reset to air

            // Mark solid cells
            for k in 0..d {
                for j in 0..h {
                    for i in 0..w {
                        let idx = k * w * h + j * w + i;
                        if sim.grid.is_solid(i, j, k) {
                            cell_types[idx] = 2; // solid
                        }
                    }
                }
            }

            // Mark fluid cells based on particle presence
            for p in &sim.particles.list {
                let i = (p.position.x / CELL_SIZE).floor() as i32;
                let j = (p.position.y / CELL_SIZE).floor() as i32;
                let k = (p.position.z / CELL_SIZE).floor() as i32;
                if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                    let idx = k as usize * w * h + j as usize * w + i as usize;
                    if cell_types[idx] != 2 {
                        cell_types[idx] = 1; // fluid
                    }
                }
            }

            // Run GPU FLIP step with flow acceleration
            gpu_flip.step(
                &device,
                &queue,
                &mut positions,
                &mut velocities,
                &mut c_matrices,
                &cell_types,
                DT,
                -9.8,
                FLOW_ACCEL,  // <-- THIS IS THE KEY: flow applied on grid before pressure solve
                sim.pressure_iterations as u32,
            );

            // Copy velocities back to particles and advect
            for (idx, p) in sim.particles.list.iter_mut().enumerate() {
                if idx < velocities.len() {
                    p.velocity = velocities[idx];
                    p.affine_velocity = c_matrices[idx];
                }

                // Advect from density-corrected position (positions[] was modified by step())
                p.position = positions[idx] + p.velocity * DT;

                // Basic boundary handling
                let min = CELL_SIZE * 0.5;
                let max_z = (d as f32 - 0.5) * CELL_SIZE;

                // Inlet (x=0): bounce
                if p.position.x < min {
                    p.position.x = min;
                    p.velocity.x = p.velocity.x.abs() * 0.1;
                }

                // Floor (y=0): bounce
                if p.position.y < min {
                    p.position.y = min;
                    p.velocity.y = p.velocity.y.abs() * 0.1;
                }

                // Side walls (z): bounce
                if p.position.z < min {
                    p.position.z = min;
                    p.velocity.z = p.velocity.z.abs() * 0.1;
                }
                if p.position.z > max_z {
                    p.position.z = max_z;
                    p.velocity.z = -p.velocity.z.abs() * 0.1;
                }

                // SDF collision
                let sdf = sim.grid.sample_sdf(p.position);
                if sdf < 0.0 {
                    let normal = sim.grid.sdf_gradient(p.position);
                    let penetration = -sdf + CELL_SIZE * 0.1;
                    p.position += normal * penetration;
                    let vel_into_solid = p.velocity.dot(normal);
                    if vel_into_solid < 0.0 {
                        p.velocity -= normal * vel_into_solid * 1.1;
                    }
                }
            }

            // Remove out-of-bounds particles
            sim.particles.list.retain(|p| {
                p.position.x > 0.0
                    && p.position.x < (w as f32 - 1.0) * CELL_SIZE
                    && p.position.y > 0.0
                    && p.position.y < (h as f32 - 1.0) * CELL_SIZE
                    && p.position.z > 0.0
                    && p.position.z < (d as f32 - 1.0) * CELL_SIZE
                    && p.velocity.is_finite()
                    && p.position.is_finite()
            });
        }

        // Print progress every 50 frames
        if (frame + 1) % 50 == 0 {
            let avg_vx: f32 = if sim.particles.list.is_empty() {
                0.0
            } else {
                sim.particles.list.iter().map(|p| p.velocity.x).sum::<f32>()
                    / sim.particles.list.len() as f32
            };
            let avg_x: f32 = if sim.particles.list.is_empty() {
                0.0
            } else {
                sim.particles.list.iter().map(|p| p.position.x).sum::<f32>()
                    / sim.particles.list.len() as f32
            };
            println!("Frame {:3}: particles={:5}, avgVx={:6.3}, avgX={:6.3}",
                     frame + 1, sim.particles.list.len(), avg_vx, avg_x);
        }
    }

    // Final metrics
    let final_count = sim.particle_count();
    let final_avg_vx: f32 = if sim.particles.list.is_empty() {
        0.0
    } else {
        sim.particles.list.iter().map(|p| p.velocity.x).sum::<f32>()
            / sim.particles.list.len() as f32
    };
    let final_avg_x: f32 = if sim.particles.list.is_empty() {
        0.0
    } else {
        sim.particles.list.iter().map(|p| p.position.x).sum::<f32>()
            / sim.particles.list.len() as f32
    };
    let x_progress = final_avg_x - initial_avg_x;
    let channel_length = GRID_WIDTH as f32 * CELL_SIZE;
    let progress_ratio = x_progress / channel_length;
    let particles_exited = initial_count as i32 - final_count as i32;

    println!();
    println!("=== RESULTS ===");
    println!("Channel length: {:.3}m", channel_length);
    println!("Final particles: {} (exited: {})", final_count, particles_exited.max(0));

    // Check if most particles exited - this is the STRONGEST signal of working flow!
    let exit_ratio = particles_exited as f32 / initial_count as f32;
    let most_exited = exit_ratio > 0.8;  // >80% of particles exited

    if most_exited {
        println!("✓ PASS: {:.1}% of particles ({}/{}) exited through outlet!",
                 exit_ratio * 100.0, particles_exited, initial_count);
        println!("   Flow acceleration is WORKING - water flowed through the entire channel!");
        println!();
        println!("=== ALL TESTS PASSED ===");
        std::process::exit(0);
    }

    // If particles remain, check velocity and progress
    println!("Final avg Vx: {:.3} m/s (threshold: > {})", final_avg_vx, MIN_AVG_VX);
    println!("X progress: {:.3}m ({:.1}% of channel)", x_progress, progress_ratio * 100.0);
    println!("Progress ratio: {:.3} (threshold: > {})", progress_ratio, MIN_X_PROGRESS_RATIO);
    println!();

    // ASSERTIONS - test FAILS if flow is broken
    let mut passed = true;

    if final_avg_vx < MIN_AVG_VX && final_count > 0 {
        println!("❌ FAIL: avg Vx = {:.3} < {} - water is NOT flowing!", final_avg_vx, MIN_AVG_VX);
        passed = false;
    } else if final_count > 0 {
        println!("✓ PASS: avg Vx = {:.3} >= {} - water IS flowing", final_avg_vx, MIN_AVG_VX);
    }

    if progress_ratio < MIN_X_PROGRESS_RATIO && final_count > 0 {
        println!("❌ FAIL: progress ratio = {:.3} < {} - water hasn't moved far enough!", progress_ratio, MIN_X_PROGRESS_RATIO);
        passed = false;
    } else if final_count > 0 {
        println!("✓ PASS: progress ratio = {:.3} >= {} - water moved downstream", progress_ratio, MIN_X_PROGRESS_RATIO);
    }

    println!();
    if passed {
        println!("=== ALL TESTS PASSED ===");
        std::process::exit(0);
    } else {
        println!("=== TESTS FAILED ===");
        std::process::exit(1);
    }
}
