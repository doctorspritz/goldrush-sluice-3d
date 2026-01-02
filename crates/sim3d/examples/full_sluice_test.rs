//! Full-scale sluice test matching the visual simulation parameters
//!
//! Run with: cargo run --example full_sluice_test -p sim3d --release

use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig, Vec3};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            FULL-SCALE SLUICE TEST (Visual Params)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Same params as sluice_3d_visual.rs
    let grid_width = 200;
    let grid_height = 80;
    let grid_depth = 40;
    let cell_size = 0.016;
    let world_width = grid_width as f32 * cell_size;

    println!("Grid: {}x{}x{} = {} cells", grid_width, grid_height, grid_depth,
             grid_width * grid_height * grid_depth);
    println!("Cell size: {}m", cell_size);
    println!("World size: {:.2}m x {:.2}m x {:.2}m\n",
             world_width, grid_height as f32 * cell_size, grid_depth as f32 * cell_size);

    let mut sim = FlipSimulation3D::new(grid_width, grid_height, grid_depth, cell_size);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 100;

    let sluice_config = SluiceConfig {
        slope: 0.10,
        slick_plate_len: 20,
        riffle_spacing: 16,
        riffle_height: 5,
        riffle_width: 2,
    };
    create_sluice(&mut sim, &sluice_config);

    // First riffle position
    let first_riffle_x = sluice_config.slick_plate_len as f32 * cell_size;
    println!("First riffle at x = {:.3}m", first_riffle_x);
    println!("Riffles every {:.3}m", sluice_config.riffle_spacing as f32 * cell_size);

    // Spawn initial water
    let inlet_vel = Vec3::new(1.5, 0.0, 0.0);
    spawn_inlet_water(&mut sim, &sluice_config, 2000, inlet_vel);
    println!("Initial particles: {}", sim.particle_count());
    println!("Inlet velocity: {} m/s\n", inlet_vel.x);

    let dt = 1.0 / 120.0;
    let substeps = 2;
    let total_frames = 120; // 1 second of simulation

    println!("Running {} frames ({:.1}s) with {} substeps...\n",
             total_frames, total_frames as f32 / 120.0, substeps);

    println!("{:>6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8}",
             "Frame", "Count", "MaxX", "AvgVelX", "VelMag", "Time(ms)", "Exited");
    println!("{}", "-".repeat(70));

    let mut total_exited = 0usize;
    for frame in 0..total_frames {
        // Spawn more water
        if frame % 2 == 0 && sim.particle_count() < 50000 {
            spawn_inlet_water(&mut sim, &sluice_config, 100, inlet_vel);
        }

        let count_before = sim.particle_count();
        let start = Instant::now();
        for _ in 0..substeps {
            sim.update(dt);
        }
        let elapsed = start.elapsed().as_millis();
        let exited_this_frame = count_before.saturating_sub(sim.particle_count());
        total_exited += exited_this_frame;

        // Stats
        if !sim.particles.list.is_empty() {
            let max_x = sim.particles.list.iter().map(|p| p.position.x).fold(0.0f32, f32::max);
            let avg_vel_x = sim.particles.list.iter().map(|p| p.velocity.x).sum::<f32>() / sim.particle_count() as f32;
            let avg_vel_mag = sim.particles.list.iter().map(|p| p.velocity.length()).sum::<f32>() / sim.particle_count() as f32;

            if frame % 20 == 0 || frame == total_frames - 1 {
                let past_riffle = if max_x > first_riffle_x { "✓" } else { "" };
                println!("{:>6} {:>8} {:>7.3}{} {:>8.3} {:>10.3} {:>10} {:>8}",
                         frame, sim.particle_count(), max_x, past_riffle, avg_vel_x, avg_vel_mag, elapsed, total_exited);
            }
        }
    }

    // Final assessment
    let max_x = sim.particles.list.iter().map(|p| p.position.x).fold(0.0f32, f32::max);
    let avg_vel = sim.particles.list.iter().map(|p| p.velocity.x).sum::<f32>() / sim.particle_count() as f32;

    println!("\n=== RESULT ===");
    println!("Total particles exited through outlet: {}", total_exited);

    if max_x > first_riffle_x {
        println!("✅ Water passed first riffle (max_x={:.3}m > riffle@{:.3}m)", max_x, first_riffle_x);
    } else {
        println!("❌ Water STUCK before first riffle (max_x={:.3}m < riffle@{:.3}m)", max_x, first_riffle_x);
    }

    if total_exited > 100 {
        println!("✅ Water is exiting through outlet ({} particles)", total_exited);
    } else if total_exited > 0 {
        println!("⚠️  Some water exiting ({} particles) but flow is weak", total_exited);
    } else {
        println!("❌ NO water exiting - outlet is blocked or flow reversed");
    }

    if avg_vel > 0.3 {
        println!("✅ Good forward velocity (avg={:.3} m/s)", avg_vel);
    } else if avg_vel > 0.0 {
        println!("⚠️  Low forward velocity (avg={:.3} m/s)", avg_vel);
    } else {
        println!("❌ Negative/zero velocity - water flowing BACKWARD (avg={:.3} m/s)", avg_vel);
    }
}
