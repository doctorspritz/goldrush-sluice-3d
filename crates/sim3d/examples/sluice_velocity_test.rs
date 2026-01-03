//! Diagnose why velocity goes to zero in sluice.

use glam::Vec3;
use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig};

fn main() {
    println!("=== SLUICE VELOCITY DIAGNOSTIC ===\n");

    let mut sim = FlipSimulation3D::new(40, 20, 12, 0.05);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 50;

    let config = SluiceConfig {
        slope: 0.08,
        slick_plate_len: 8,
        riffle_spacing: 8,
        riffle_height: 2,
        riffle_width: 1,
    };
    create_sluice(&mut sim, &config);

    // Count solid cells
    let solid_count: usize = sim.grid.solid.iter().filter(|&&s| s).count();
    let total_cells = sim.grid.width * sim.grid.height * sim.grid.depth;
    println!("Grid: {}x{}x{}", sim.grid.width, sim.grid.height, sim.grid.depth);
    println!("Solid cells: {} / {} ({:.1}%)", solid_count, total_cells, 100.0 * solid_count as f32 / total_cells as f32);

    // Spawn initial water
    spawn_inlet_water(&mut sim, &config, 500, Vec3::new(1.5, 0.0, 0.0));
    println!("Spawned {} particles", sim.particle_count());

    // Track velocities over time
    let dt = 1.0 / 60.0;
    for frame in 0..300 {
        // Track grid velocities
        let max_u = sim.grid.u.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
        let max_v = sim.grid.v.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
        let max_w = sim.grid.w.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));

        let avg_vel: Vec3 = sim.particles.list.iter()
            .map(|p| p.velocity)
            .fold(Vec3::ZERO, |a, b| a + b) / sim.particle_count().max(1) as f32;

        let max_vel: f32 = sim.particles.list.iter()
            .map(|p| p.velocity.length())
            .fold(0.0, f32::max);

        // Count particles by state
        let zero_vel_count = sim.particles.list.iter()
            .filter(|p| p.velocity.length() < 0.001)
            .count();

        if frame % 20 == 0 {
            println!(
                "F{:3}: particles={:4}, avgVel=({:6.3},{:6.3},{:6.3}), |max|={:.3}, zero_vel={}, grid_max=({:.3},{:.3},{:.3})",
                frame, sim.particle_count(),
                avg_vel.x, avg_vel.y, avg_vel.z,
                max_vel, zero_vel_count,
                max_u, max_v, max_w
            );
        }

        // Step simulation
        sim.update(dt);

        // Spawn more water
        if frame % 5 == 0 && sim.particle_count() < 2000 {
            spawn_inlet_water(&mut sim, &config, 20, Vec3::new(1.5, 0.0, 0.0));
        }

        // Debug: if all velocities are zero, print what happened
        if max_vel < 0.001 && frame > 30 {
            println!("\n!!! ALL VELOCITIES ZERO at frame {} !!!", frame);

            // Check grid state
            let nonzero_u = sim.grid.u.iter().filter(|&&u| u.abs() > 0.001).count();
            let nonzero_v = sim.grid.v.iter().filter(|&&v| v.abs() > 0.001).count();
            let nonzero_w = sim.grid.w.iter().filter(|&&w| w.abs() > 0.001).count();
            println!("Grid nonzero: u={}, v={}, w={}", nonzero_u, nonzero_v, nonzero_w);

            // Sample particles
            println!("Sample particles:");
            for (i, p) in sim.particles.list.iter().enumerate().take(3) {
                println!("  [{}] pos=({:.2},{:.2},{:.2}), vel=({:.5},{:.5},{:.5})",
                    i, p.position.x, p.position.y, p.position.z,
                    p.velocity.x, p.velocity.y, p.velocity.z);
            }

            break;
        }
    }

    // Final check
    println!("\n=== FINAL STATE ===");
    let nonzero_u = sim.grid.u.iter().filter(|&&u| u.abs() > 0.001).count();
    let nonzero_v = sim.grid.v.iter().filter(|&&v| v.abs() > 0.001).count();
    let nonzero_w = sim.grid.w.iter().filter(|&&w| w.abs() > 0.001).count();
    println!("Grid nonzero velocities: u={}/{}, v={}/{}, w={}/{}",
        nonzero_u, sim.grid.u.len(),
        nonzero_v, sim.grid.v.len(),
        nonzero_w, sim.grid.w.len()
    );

    let avg_vel: Vec3 = sim.particles.list.iter()
        .map(|p| p.velocity)
        .fold(Vec3::ZERO, |a, b| a + b) / sim.particle_count().max(1) as f32;
    println!("Avg particle velocity: ({:.3}, {:.3}, {:.3})", avg_vel.x, avg_vel.y, avg_vel.z);
}
