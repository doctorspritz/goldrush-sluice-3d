//! Headless DEM settling diagnostic test
//!
//! Tests particle behavior numerically without visual rendering:
//! - Measures horizontal spread (angle of repose)
//! - Checks settling velocity convergence
//! - Verifies particles don't stack vertically
//!
//! Run with: cargo run --example dem_settling_diagnostic -p game --release

use game::gpu::dem::GpuDemSolver;
use glam::Vec2;
use sim::particle::{Particle, ParticleMaterial, Particles};
use sim::flip::FlipSimulation;
use std::sync::Arc;

const CELL_SIZE: f32 = 2.0;
const WIDTH: usize = 160;
const HEIGHT: usize = 120;
const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = 200.0;  // Positive = downward (y increases downward)

fn main() {
    pollster::block_on(run_diagnostic());
}

async fn run_diagnostic() {
    println!("=== DEM Settling Diagnostic ===\n");

    // Create headless GPU context
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

    // Request higher limits for DEM solver (needs 10+ storage buffers)
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("DEM Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    println!("GPU: {}", adapter.get_info().name);

    // Create simulation
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    build_box(&mut sim);

    // Create GPU DEM solver
    let gpu = GpuContextHeadless { device: Arc::new(device), queue: Arc::new(queue) };
    let mut dem = GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 10_000);

    // Test 1: Single column pour - should spread laterally
    println!("TEST 1: Single column pour (expect lateral spread)");
    test_column_pour(&mut sim, &mut dem, &gpu).await;

    // Reset
    sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    build_box(&mut sim);

    // Test 2: Settling velocity - should converge to zero
    println!("\nTEST 2: Settling velocity convergence");
    test_settling_velocity(&mut sim, &mut dem, &gpu).await;

    println!("\n=== Diagnostic Complete ===");
}

struct GpuContextHeadless {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

fn build_box(sim: &mut FlipSimulation) {
    let w = sim.grid.width;
    let h = sim.grid.height;

    // Floor (2 cells thick)
    for i in 0..w {
        sim.grid.set_solid(i, h - 1);
        sim.grid.set_solid(i, h - 2);
    }

    // Left wall
    for j in 0..h {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(1, j);
    }

    // Right wall
    for j in 0..h {
        sim.grid.set_solid(w - 1, j);
        sim.grid.set_solid(w - 2, j);
    }

    sim.grid.compute_sdf();

    // Debug: print some SDF values
    let center_i = w / 2;
    println!("  SDF debug (center column i={}):", center_i);
    for j in [0, 10, 50, h - 5, h - 3, h - 2, h - 1] {
        let idx = j * w + center_i;
        let sdf_val = sim.grid.sdf[idx];
        let world_y = j as f32 * CELL_SIZE;
        println!("    j={:3} (y={:5.0}): sdf={:8.2}", j, world_y, sdf_val);
    }
}

async fn test_column_pour(sim: &mut FlipSimulation, dem: &mut GpuDemSolver, gpu: &GpuContextHeadless) {
    let center_x = (WIDTH as f32 / 2.0) * CELL_SIZE;
    let drop_y = 20.0 * CELL_SIZE;  // Near top
    let spacing = CELL_SIZE * 0.7;

    // Pour 200 particles in a narrow column (3 particles wide)
    // NO artificial jitter - shader now has surface roughness
    let num_particles = 200;
    let particles_per_row = 3;
    let rows = num_particles / particles_per_row;

    for row in 0..rows {
        for col in 0..particles_per_row {
            let x = center_x + (col as f32 - 1.0) * spacing;
            let y = drop_y + row as f32 * spacing;
            sim.particles.list.push(Particle::new(
                Vec2::new(x, y),
                Vec2::ZERO,
                ParticleMaterial::Sand,
            ));
        }
    }

    // Debug: show particle radius for collision detection
    let test_radius = sim::particle::ParticleMaterial::Sand.typical_diameter() * 0.35 * CELL_SIZE;
    println!("  Spawned {} particles at x={:.0}, collision_radius={:.2}", sim.particles.len(), center_x, test_radius);

    // Run simulation for 300 frames (5 seconds)
    let frames = 300;
    for frame in 0..frames {
        sim.grid.compute_sdf();
        dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
        dem.execute_headless(&gpu.device, &gpu.queue, &mut sim.particles, CELL_SIZE, DT, GRAVITY, -1.0);

        if frame % 60 == 0 {
            let stats = compute_pile_stats(&sim.particles);
            println!("  frame {}: x_spread={:.1}, y_range={:.1}, avg_speed={:.2}",
                     frame, stats.x_spread, stats.y_range, stats.avg_speed);
        }
    }

    // Final analysis
    let stats = compute_pile_stats(&sim.particles);
    println!("\n  RESULT:");
    println!("    Horizontal spread: {:.1} (expect > 50 for good angle of repose)", stats.x_spread);
    println!("    Vertical range: {:.1}", stats.y_range);
    println!("    Final avg speed: {:.3} (expect < 1.0 for settled)", stats.avg_speed);

    let aspect_ratio = stats.x_spread / stats.y_range.max(1.0);
    println!("    Aspect ratio (spread/height): {:.2} (expect > 1.0 for pile, not tower)", aspect_ratio);

    if stats.x_spread < 30.0 {
        println!("    WARNING: Particles stacking vertically - no lateral spread!");
    }
    if stats.avg_speed > 2.0 {
        println!("    WARNING: Particles still moving - not settled!");
    }
}

async fn test_settling_velocity(sim: &mut FlipSimulation, dem: &mut GpuDemSolver, gpu: &GpuContextHeadless) {
    // Drop a single layer of particles
    let y = 50.0 * CELL_SIZE;
    let spacing = CELL_SIZE * 0.8;

    for i in 0..20 {
        let x = (40.0 + i as f32 * 2.0) * CELL_SIZE;
        sim.particles.list.push(Particle::new(
            Vec2::new(x, y),
            Vec2::ZERO,
            ParticleMaterial::Sand,
        ));
    }

    println!("  Spawned {} particles at y={:.0}", sim.particles.len(), y);

    // Track velocity convergence
    let mut prev_avg_speed = f32::MAX;
    let mut settled_frames = 0;

    for frame in 0..600 {
        sim.grid.compute_sdf();
        dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
        dem.execute_headless(&gpu.device, &gpu.queue, &mut sim.particles, CELL_SIZE, DT, GRAVITY, -1.0);

        let stats = compute_pile_stats(&sim.particles);

        if frame % 30 == 0 {
            println!("  frame {}: avg_speed={:.3}, max_speed={:.3}",
                     frame, stats.avg_speed, stats.max_speed);
        }

        // Check if settled
        if stats.avg_speed < 0.5 && stats.max_speed < 2.0 {
            settled_frames += 1;
            if settled_frames >= 30 {
                println!("\n  SETTLED at frame {} (avg_speed={:.3})", frame, stats.avg_speed);
                break;
            }
        } else {
            settled_frames = 0;
        }

        prev_avg_speed = stats.avg_speed;
    }

    let stats = compute_pile_stats(&sim.particles);
    println!("\n  RESULT:");
    println!("    Final avg speed: {:.4}", stats.avg_speed);
    println!("    Final max speed: {:.4}", stats.max_speed);

    if stats.avg_speed > 1.0 {
        println!("    FAIL: Particles still vibrating!");
    } else {
        println!("    PASS: Particles settled");
    }
}

struct PileStats {
    x_min: f32,
    x_max: f32,
    x_spread: f32,
    y_min: f32,
    y_max: f32,
    y_range: f32,
    avg_speed: f32,
    max_speed: f32,
}

fn compute_pile_stats(particles: &Particles) -> PileStats {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut total_speed = 0.0;
    let mut max_speed = 0.0f32;
    let mut count = 0;

    for p in particles.iter() {
        if !p.is_sediment() {
            continue;
        }
        x_min = x_min.min(p.position.x);
        x_max = x_max.max(p.position.x);
        y_min = y_min.min(p.position.y);
        y_max = y_max.max(p.position.y);

        let speed = p.velocity.length();
        total_speed += speed;
        max_speed = max_speed.max(speed);
        count += 1;
    }

    if count == 0 {
        return PileStats {
            x_min: 0.0, x_max: 0.0, x_spread: 0.0,
            y_min: 0.0, y_max: 0.0, y_range: 0.0,
            avg_speed: 0.0, max_speed: 0.0,
        };
    }

    PileStats {
        x_min,
        x_max,
        x_spread: x_max - x_min,
        y_min,
        y_max,
        y_range: y_max - y_min,
        avg_speed: total_speed / count as f32,
        max_speed,
    }
}
