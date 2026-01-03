//! Headless diagnostic test for sluice flow
//!
//! Run with: cargo run --example sluice_diagnostic -p sim3d --release

use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig, Vec3};

fn main() {
    println!("=== Sluice Flow Diagnostic Test ===\n");

    // Create simulation with same params as visual
    let grid_width = 100;  // Smaller for faster test
    let grid_height = 40;
    let grid_depth = 20;
    let cell_size = 0.025;

    let mut sim = FlipSimulation3D::new(grid_width, grid_height, grid_depth, cell_size);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 50;

    let sluice_config = SluiceConfig {
        slope: 0.10,
        slick_plate_len: 10,
        riffle_spacing: 12,
        riffle_height: 3,
        riffle_width: 1,
    };
    create_sluice(&mut sim, &sluice_config);

    // Spawn initial water
    let inlet_vel = Vec3::new(1.5, 0.0, 0.0);
    spawn_inlet_water(&mut sim, &sluice_config, 500, inlet_vel);

    println!("Grid: {}x{}x{}, cell_size: {}", grid_width, grid_height, grid_depth, cell_size);
    println!("World size: {:.2}m x {:.2}m x {:.2}m",
             grid_width as f32 * cell_size,
             grid_height as f32 * cell_size,
             grid_depth as f32 * cell_size);
    println!("Initial particles: {}", sim.particle_count());
    println!("Slope: {}, inlet floor at Y ~ {:.3}m\n",
             sluice_config.slope,
             (grid_width - 1) as f32 * sluice_config.slope * cell_size);

    // Get initial stats
    let initial_avg_x = avg_x(&sim);
    let initial_avg_y = avg_y(&sim);
    println!("Initial: avg_x={:.4}, avg_y={:.4}", initial_avg_x, initial_avg_y);

    // Run simulation
    let dt = 1.0 / 120.0;
    let frames = 300;  // 2.5 seconds of simulation

    for frame in 0..frames {
        // Spawn more water periodically
        if frame % 10 == 0 && sim.particle_count() < 5000 {
            spawn_inlet_water(&mut sim, &sluice_config, 50, inlet_vel);
        }

        sim.update(dt);

        // Remove particles that exit
        let before = sim.particle_count();
        sim.particles.list.retain(|p| {
            p.position.x > 0.0 &&
            p.position.x < (grid_width as f32 - 0.5) * cell_size &&
            p.position.y > 0.0 &&
            p.position.y < (grid_height as f32 - 0.5) * cell_size &&
            p.position.z > 0.0 &&
            p.position.z < (grid_depth as f32 - 0.5) * cell_size &&
            p.velocity.is_finite() &&
            p.position.is_finite()
        });
        let exited = before - sim.particle_count();

        // Print diagnostics every 60 frames (0.5 sec)
        if frame % 60 == 0 || frame == frames - 1 {
            let avg_x = avg_x(&sim);
            let avg_y = avg_y(&sim);
            let avg_vel = avg_velocity(&sim);
            let (min_x, max_x) = x_range(&sim);

            println!("Frame {}: particles={}, exited={}", frame, sim.particle_count(), exited);
            println!("  avg_pos: x={:.4}, y={:.4}", avg_x, avg_y);
            println!("  x_range: [{:.4}, {:.4}]", min_x, max_x);
            println!("  avg_vel: ({:.3}, {:.3}, {:.3})", avg_vel.x, avg_vel.y, avg_vel.z);
        }
    }

    // Final analysis
    println!("\n=== Results ===");
    let final_avg_x = avg_x(&sim);
    let final_avg_y = avg_y(&sim);
    let x_moved = final_avg_x - initial_avg_x;
    let y_moved = final_avg_y - initial_avg_y;

    println!("X movement: {:.4}m (should be positive = downstream)", x_moved);
    println!("Y movement: {:.4}m (should be negative = falling)", y_moved);

    if x_moved > 0.1 {
        println!("PASS: Water is flowing downstream (+X)");
    } else if x_moved < -0.01 {
        println!("FAIL: Water is flowing BACKWARDS (-X)!");
    } else {
        println!("UNCLEAR: Water barely moving in X");
    }

    if y_moved < -0.01 {
        println!("PASS: Water is falling (-Y)");
    } else if y_moved > 0.01 {
        println!("FAIL: Water is rising (+Y)!");
    } else {
        println!("Water Y is stable (maybe on floor)");
    }
}

fn avg_x(sim: &FlipSimulation3D) -> f32 {
    if sim.particles.list.is_empty() { return 0.0; }
    sim.particles.list.iter().map(|p| p.position.x).sum::<f32>() / sim.particle_count() as f32
}

fn avg_y(sim: &FlipSimulation3D) -> f32 {
    if sim.particles.list.is_empty() { return 0.0; }
    sim.particles.list.iter().map(|p| p.position.y).sum::<f32>() / sim.particle_count() as f32
}

fn avg_velocity(sim: &FlipSimulation3D) -> Vec3 {
    if sim.particles.list.is_empty() { return Vec3::ZERO; }
    let sum: Vec3 = sim.particles.list.iter().map(|p| p.velocity).sum();
    sum / sim.particle_count() as f32
}

fn x_range(sim: &FlipSimulation3D) -> (f32, f32) {
    if sim.particles.list.is_empty() { return (0.0, 0.0); }
    let min = sim.particles.list.iter().map(|p| p.position.x).fold(f32::MAX, f32::min);
    let max = sim.particles.list.iter().map(|p| p.position.x).fold(f32::MIN, f32::max);
    (min, max)
}
