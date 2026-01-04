//! Riffle Diagnostic - Track EXACTLY where particles go
//!
//! This test runs a sluice with riffles and tracks:
//! - Where particles get stuck
//! - Where particles exit
//! - Velocity distribution at each X position
//!
//! Run with: cargo run --example riffle_diagnostic --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig};

// Match gold_sluice_3d settings
const GRID_WIDTH: usize = 48;
const GRID_HEIGHT: usize = 20;
const GRID_DEPTH: usize = 12;
const CELL_SIZE: f32 = 0.04;
const MAX_PARTICLES: usize = 80000;

const FLOW_ACCEL: f32 = 3.0;
const TEST_FRAMES: u32 = 300;
const DT: f32 = 1.0 / 60.0;

fn main() {
    println!("=== RIFFLE DIAGNOSTIC ===");
    println!("Grid: {}x{}x{}, cell_size: {}", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    println!("Channel length: {:.3}m", GRID_WIDTH as f32 * CELL_SIZE);
    println!();

    // Create simulation with same config as gold_sluice_3d
    let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.95;
    sim.pressure_iterations = 80;

    let sluice_config = SluiceConfig {
        slope: 0.20,
        slick_plate_len: 6,
        riffle_spacing: 8,
        riffle_height: 3,
        riffle_width: 1,
    };

    create_sluice(&mut sim, &sluice_config);

    // Print riffle positions
    println!("=== RIFFLE GEOMETRY ===");
    println!("Slope: {:.0}%", sluice_config.slope * 100.0);
    println!("Slick plate length: {} cells", sluice_config.slick_plate_len);
    println!("Riffle spacing: {} cells", sluice_config.riffle_spacing);
    println!("Riffle height: {} cells ({:.3}m)", sluice_config.riffle_height, sluice_config.riffle_height as f32 * CELL_SIZE);

    // Find riffle X positions
    let mut riffle_x = sluice_config.slick_plate_len;
    let mut riffle_positions = Vec::new();
    while riffle_x + sluice_config.riffle_width < GRID_WIDTH - 2 {
        let floor_y = ((GRID_WIDTH - 1 - riffle_x) as f32 * sluice_config.slope) as usize;
        let riffle_top_y = floor_y + sluice_config.riffle_height;
        println!("Riffle at x={}: floor_y={}, top_y={} (world y={:.3}m)",
                 riffle_x, floor_y, riffle_top_y, riffle_top_y as f32 * CELL_SIZE);
        riffle_positions.push(riffle_x);
        riffle_x += sluice_config.riffle_spacing;
    }
    println!();

    // Print actual spawn height (matches spawn_inlet_water logic)
    let first_riffle_x = sluice_config.slick_plate_len;
    let first_riffle_floor_y = ((GRID_WIDTH - 1 - first_riffle_x) as f32 * sluice_config.slope) as usize;
    let first_riffle_top_y = first_riffle_floor_y + sluice_config.riffle_height;
    let spawn_y_base = (first_riffle_top_y as f32 + 2.0) * CELL_SIZE;
    println!("First riffle top: y={} cells ({:.3}m)", first_riffle_top_y, first_riffle_top_y as f32 * CELL_SIZE);
    println!("Spawn height: y={:.3}m (2 cells above riffle top)", spawn_y_base);
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
                compatible_surface: None,
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
                    label: Some("Riffle Diagnostic Device"),
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

    // Tracking stats
    let mut total_spawned: u32 = 0;
    let mut total_exited: u32 = 0;
    let mut x_histogram: Vec<u32> = vec![0; GRID_WIDTH];

    println!("=== RUNNING SIMULATION ===");
    println!("Frame | Particles | Spawned | Exited | Avg X | Avg Vx | Max Vx | Min Y");
    println!("------|-----------|---------|--------|-------|--------|--------|------");

    for frame in 0..TEST_FRAMES {
        // Spawn water at inlet (every 4 frames like gold_sluice_3d)
        if frame % 4 == 0 && sim.particles.len() < MAX_PARTICLES {
            let before = sim.particles.len();
            let flow_velocity = Vec3::new(0.5, 0.0, 0.0);
            spawn_inlet_water(&mut sim, &sluice_config, 20, flow_velocity);
            total_spawned += (sim.particles.len() - before) as u32;
        }

        // Sync particle data to GPU
        let particle_count = sim.particles.list.len();
        positions.clear();
        velocities.clear();
        c_matrices.clear();

        for p in &sim.particles.list {
            positions.push(p.position);
            velocities.push(p.velocity);
            c_matrices.push(p.affine_velocity);
        }

        // Build cell types
        let w = sim.grid.width;
        let h = sim.grid.height;
        let d = sim.grid.depth;
        cell_types.fill(0);

        for k in 0..d {
            for j in 0..h {
                for i in 0..w {
                    let idx = k * w * h + j * w + i;
                    if sim.grid.is_solid(i, j, k) {
                        cell_types[idx] = 2;
                    }
                }
            }
        }

        for p in &sim.particles.list {
            let i = (p.position.x / CELL_SIZE).floor() as i32;
            let j = (p.position.y / CELL_SIZE).floor() as i32;
            let k = (p.position.z / CELL_SIZE).floor() as i32;
            if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                let idx = k as usize * w * h + j as usize * w + i as usize;
                if cell_types[idx] != 2 {
                    cell_types[idx] = 1;
                }
            }
        }

        // Debug: count fluid cells
        let fluid_cell_count: u32 = cell_types.iter().filter(|&&t| t == 1).map(|_| 1).sum();

        // Store original positions to measure density projection effect
        let positions_before: Vec<Vec3> = positions.clone();

        // Run GPU step
        gpu_flip.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &cell_types,
            DT,
            -9.8,
            FLOW_ACCEL,
            sim.pressure_iterations as u32,
        );

        // Measure density projection effect (position corrections)
        let mut max_delta = 0.0f32;
        let mut sum_delta_y = 0.0f32;
        for (i, pos) in positions.iter().enumerate() {
            let delta = *pos - positions_before[i];
            let delta_mag = delta.length();
            if delta_mag > max_delta {
                max_delta = delta_mag;
            }
            sum_delta_y += delta.y;
        }

        // Debug output on first few frames
        if frame < 5 {
            println!("Frame {}: {} particles, {} fluid cells, max_delta={:.6}, avg_delta_y={:.6}",
                     frame, particle_count, fluid_cell_count, max_delta,
                     if particle_count > 0 { sum_delta_y / particle_count as f32 } else { 0.0 });
        }

        // Sync back and advect
        // IMPORTANT: positions[] was modified by density projection, use it as the base for advection
        for (idx, p) in sim.particles.list.iter_mut().enumerate() {
            if idx < velocities.len() {
                p.velocity = velocities[idx];
                p.affine_velocity = c_matrices[idx];
            }

            // Advect from density-corrected position (positions[] was modified by step())
            p.position = positions[idx] + p.velocity * DT;

            // SDF collision
            let sdf = sim.grid.sample_sdf(p.position);
            if sdf < 0.0 {
                let grad = sim.grid.sdf_gradient(p.position);
                p.position -= grad * sdf * 1.1;

                let vel_into = p.velocity.dot(grad);
                if vel_into < 0.0 {
                    p.velocity -= grad * vel_into;
                }
            }

            // Track exits
            let max_x = GRID_WIDTH as f32 * CELL_SIZE;
            if p.position.x >= max_x {
                total_exited += 1;
                p.position.x = 1000.0; // Mark for removal
            }
        }

        // Remove exited particles
        sim.particles.list.retain(|p| p.position.x < 100.0);

        // Compute stats
        if (frame + 1) % 20 == 0 {
            let count = sim.particles.len();
            let (avg_x, avg_vx, max_vx, min_y) = if count > 0 {
                let sum_x: f32 = sim.particles.list.iter().map(|p| p.position.x).sum();
                let sum_vx: f32 = sim.particles.list.iter().map(|p| p.velocity.x).sum();
                let max_vx: f32 = sim.particles.list.iter().map(|p| p.velocity.x).fold(f32::MIN, f32::max);
                let min_y: f32 = sim.particles.list.iter().map(|p| p.position.y).fold(f32::MAX, f32::min);
                (
                    sum_x / count as f32,
                    sum_vx / count as f32,
                    max_vx,
                    min_y,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            println!(
                "{:5} | {:9} | {:7} | {:6} | {:5.3} | {:6.3} | {:6.3} | {:5.3}",
                frame + 1, count, total_spawned, total_exited, avg_x, avg_vx, max_vx, min_y
            );
        }
    }

    println!();
    println!("=== FINAL ANALYSIS ===");

    // Build histogram of particle X positions
    x_histogram.fill(0);
    for p in &sim.particles.list {
        let cell_x = (p.position.x / CELL_SIZE).floor() as usize;
        if cell_x < GRID_WIDTH {
            x_histogram[cell_x] += 1;
        }
    }

    println!("\n=== PARTICLE DISTRIBUTION (by X cell) ===");
    println!("Cell X | Count | Notes");
    println!("-------|-------|------");
    for (x, count) in x_histogram.iter().enumerate() {
        let note = if riffle_positions.contains(&x) {
            "RIFFLE"
        } else if x == 0 {
            "INLET"
        } else if x == GRID_WIDTH - 1 {
            "OUTLET"
        } else {
            ""
        };

        if *count > 0 || note.len() > 0 {
            println!("{:6} | {:5} | {}", x, count, note);
        }
    }

    // Summary
    let final_count = sim.particles.len();
    let retention_rate = if total_spawned > 0 {
        (final_count as f32 + total_exited as f32) / total_spawned as f32 * 100.0
    } else {
        0.0
    };

    println!();
    println!("=== SUMMARY ===");
    println!("Total spawned: {}", total_spawned);
    println!("Total exited: {}", total_exited);
    println!("Final count: {}", final_count);
    println!("Retention rate: {:.1}%", retention_rate);

    // Check for pooling behind riffles
    let mut pooling = Vec::new();
    for (idx, &riffle_x) in riffle_positions.iter().enumerate() {
        // Check 2 cells before riffle (where water should pool)
        let pool_x = if riffle_x > 1 { riffle_x - 1 } else { 0 };
        let pool_count = x_histogram[pool_x];
        if pool_count > 10 {
            pooling.push((idx, riffle_x, pool_count));
        }
    }

    if pooling.is_empty() {
        println!("\nWARNING: No significant pooling behind riffles!");
        println!("Water should accumulate behind riffles before overflowing.");
    } else {
        println!("\nPooling detected behind {} riffles:", pooling.len());
        for (idx, riffle_x, count) in pooling {
            println!("  Riffle {} (x={}): {} particles pooled", idx, riffle_x, count);
        }
    }

    // Find where particles are stuck
    let stuck_threshold = total_spawned / 10; // 10% of spawned
    let mut stuck_regions = Vec::new();
    for x in 0..GRID_WIDTH {
        if x_histogram[x] as u32 > stuck_threshold {
            stuck_regions.push(x);
        }
    }

    if !stuck_regions.is_empty() {
        println!("\nHigh particle density regions (>{} particles):", stuck_threshold);
        for x in stuck_regions {
            let is_riffle = riffle_positions.contains(&x);
            let after_riffle = riffle_positions.iter().any(|&rx| x == rx + 1);
            let note = if is_riffle {
                "AT RIFFLE"
            } else if after_riffle {
                "JUST AFTER RIFFLE"
            } else if x < sluice_config.slick_plate_len {
                "INLET REGION"
            } else {
                ""
            };
            println!("  x={}: {} particles {}", x, x_histogram[x], note);
        }
    }

    // Final verdict
    println!();
    let avg_vx: f32 = if sim.particles.len() > 0 {
        sim.particles.list.iter().map(|p| p.velocity.x).sum::<f32>() / sim.particles.len() as f32
    } else {
        0.0
    };

    if total_exited > total_spawned / 2 {
        println!("RESULT: FLOW IS WORKING - >50% of particles exited");
    } else if avg_vx > 0.2 {
        println!("RESULT: FLOW IS MOVING - avg Vx = {:.3} m/s", avg_vx);
    } else {
        println!("RESULT: FLOW IS BLOCKED - avg Vx = {:.3} m/s, few exits", avg_vx);
        println!();
        println!("DIAGNOSIS NEEDED:");
        println!("  1. Are riffles too tall for water to overflow?");
        println!("  2. Is flow_accel being applied correctly?");
        println!("  3. Is pressure solver zeroing outlet velocity?");
    }
}
