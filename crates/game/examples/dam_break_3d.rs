//! 3D Dam Break Simulation Example
//!
//! Demonstrates the 3D FLIP/APIC fluid simulation with a classic dam break scenario.
//! Water column collapses under gravity and splashes against walls.
//!
//! Run with: cargo run --example dam_break_3d --release

use glam::Vec3;
use sim3d::FlipSimulation3D;
use std::time::Instant;

// Grid dimensions
const GRID_WIDTH: usize = 32;
const GRID_HEIGHT: usize = 24;
const GRID_DEPTH: usize = 16;
const CELL_SIZE: f32 = 0.1;

// Simulation parameters
const DT: f32 = 1.0 / 120.0; // 120 Hz physics
const TOTAL_TIME: f32 = 3.0; // Simulate 3 seconds

fn main() {
    println!("=== 3D Dam Break Simulation ===\n");

    // Create simulation
    let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);

    // Configure physics
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 50;

    // Report grid dimensions
    let (min, max) = sim.world_bounds();
    println!(
        "Grid: {}x{}x{} cells ({:.2}x{:.2}x{:.2} world units)",
        GRID_WIDTH,
        GRID_HEIGHT,
        GRID_DEPTH,
        max.x - min.x,
        max.y - min.y,
        max.z - min.z
    );

    // Spawn dam break: water column in corner
    // Fill cells (1-8, 1-16, 1-8) with particles
    let particles_per_cell = 4; // 2x2x1 particles per cell
    let mut spawn_count = 0;

    for i in 1..9 {
        for j in 1..17 {
            for k in 1..9 {
                for pi in 0..2 {
                    for pj in 0..2 {
                        let pos = Vec3::new(
                            (i as f32 + 0.25 + pi as f32 * 0.5) * CELL_SIZE,
                            (j as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE,
                            (k as f32 + 0.5) * CELL_SIZE,
                        );
                        sim.spawn_particle(pos);
                        spawn_count += 1;
                    }
                }
            }
        }
    }

    println!("Spawned {} particles", spawn_count);
    println!("Simulating {} seconds at {:.0} Hz...\n", TOTAL_TIME, 1.0 / DT);

    // Run simulation
    let total_steps = (TOTAL_TIME / DT) as usize;
    let start = Instant::now();

    let report_interval = total_steps / 10; // Report 10 times

    for step in 0..total_steps {
        sim.update(DT);

        // Report progress
        if step % report_interval == 0 || step == total_steps - 1 {
            let time = step as f32 * DT;

            // Compute statistics
            let particles = &sim.particles.list;
            let count = particles.len();

            if count > 0 {
                let avg_y: f32 = particles.iter().map(|p| p.position.y).sum::<f32>() / count as f32;
                let avg_vel: f32 =
                    particles.iter().map(|p| p.velocity.length()).sum::<f32>() / count as f32;
                let max_vel = particles
                    .iter()
                    .map(|p| p.velocity.length())
                    .fold(0.0f32, f32::max);

                // Find bounding box
                let min_x = particles.iter().map(|p| p.position.x).fold(f32::MAX, f32::min);
                let max_x = particles.iter().map(|p| p.position.x).fold(f32::MIN, f32::max);
                let min_y = particles.iter().map(|p| p.position.y).fold(f32::MAX, f32::min);
                let max_y = particles.iter().map(|p| p.position.y).fold(f32::MIN, f32::max);
                let min_z = particles.iter().map(|p| p.position.z).fold(f32::MAX, f32::min);
                let max_z = particles.iter().map(|p| p.position.z).fold(f32::MIN, f32::max);

                println!(
                    "t={:.2}s: {} particles, avg_y={:.3}, avg_vel={:.2}, max_vel={:.2}",
                    time, count, avg_y, avg_vel, max_vel
                );
                println!(
                    "         bounds: x=[{:.2}, {:.2}], y=[{:.2}, {:.2}], z=[{:.2}, {:.2}]",
                    min_x, max_x, min_y, max_y, min_z, max_z
                );
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\n=== Simulation Complete ===");
    println!("Total time: {:.2}s", elapsed.as_secs_f32());
    println!(
        "Performance: {:.1} steps/sec ({:.2} ms/step)",
        total_steps as f32 / elapsed.as_secs_f32(),
        elapsed.as_secs_f32() * 1000.0 / total_steps as f32
    );
    println!("Final particle count: {}", sim.particle_count());

    // Validate behavior
    let final_avg_y: f32 = sim.particles.list.iter().map(|p| p.position.y).sum::<f32>()
        / sim.particle_count() as f32;

    if final_avg_y < 0.5 {
        println!("\n✓ Dam collapsed as expected (water settled to floor)");
    } else {
        println!(
            "\n✗ Unexpected: water avg_y={:.3} (expected < 0.5)",
            final_avg_y
        );
    }

    // Check for velocity explosion
    let max_vel = sim
        .particles
        .list
        .iter()
        .map(|p| p.velocity.length())
        .fold(0.0f32, f32::max);

    if max_vel < 20.0 {
        println!("✓ Velocities stable (max={:.2})", max_vel);
    } else {
        println!("✗ Velocity explosion detected (max={:.2})", max_vel);
    }
}
