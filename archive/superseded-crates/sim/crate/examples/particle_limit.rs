//! Find the particle limit for 60 FPS
//!
//! Run with: cargo run --release --example particle_limit -p sim

use sim::FlipSimulation;
use std::time::Instant;

fn main() {
    const DT: f32 = 1.0 / 60.0;
    const FRAMES_PER_TEST: usize = 60; // 1 second per test

    println!("Finding particle limit for 60 FPS target...\n");

    // Test different grid sizes and particle counts
    let tests = [
        // (width, height, cell_size, max_particles, description)
        (128, 128, 2.0, 10000, "Sluice-size (256x256 px)"),
        (256, 192, 2.0, 20000, "Mining screen (512x384 px)"),
        (512, 384, 2.0, 50000, "Large mining (1024x768 px)"),
        (256, 256, 4.0, 30000, "Coarse grid (1024x1024 px)"),
    ];

    println!("{:30} | {:>8} | {:>8} | {:>6} | {:>8}",
        "Configuration", "Grid", "Particles", "FPS", "Status");
    println!("{:-<30}-+-{:-<8}-+-{:-<8}-+-{:-<6}-+-{:-<8}",
        "", "", "", "", "");

    for (width, height, cell_size, target_particles, desc) in tests {
        let mut sim = FlipSimulation::new(width, height, cell_size);

        // Create simple container
        for i in 0..width {
            sim.grid.set_solid(i, height - 1);
        }
        for j in 0..height {
            sim.grid.set_solid(0, j);
            sim.grid.set_solid(width - 1, j);
        }

        // Spawn particles in batches until we hit target or slowdown
        let mut current_particles = 0;
        let mut last_fps = 999.0;

        while current_particles < target_particles && last_fps > 60.0 {
            // Add batch of particles
            let batch_size = (target_particles / 10).max(500);
            let spawn_y = 20.0;
            for i in 0..batch_size {
                let x = 20.0 + (i % (width - 4)) as f32 * cell_size;
                let y = spawn_y + (i / (width - 4)) as f32 * cell_size;
                sim.spawn_water(x, y, 0.0, 0.0, 1);
            }
            current_particles = sim.particles.len();

            // Measure FPS
            let start = Instant::now();
            for _ in 0..FRAMES_PER_TEST {
                sim.update(DT);
            }
            let elapsed = start.elapsed();
            let avg_frame_ms = elapsed.as_secs_f64() * 1000.0 / FRAMES_PER_TEST as f64;
            last_fps = 1000.0 / avg_frame_ms;
        }

        let status = if last_fps >= 60.0 { "OK" } else { "LIMIT" };
        println!("{:30} | {:>4}x{:<3} | {:>8} | {:>6.0} | {:>8}",
            desc, width, height, current_particles, last_fps, status);
    }

    println!("\n=== Summary ===");
    println!("The particle limit depends on grid size and particle density.");
    println!("For real-time mining, recommend:");
    println!("  - Active zone: 256x192 grid, up to ~15,000 particles");
    println!("  - Use dormant zones for large world");
}
