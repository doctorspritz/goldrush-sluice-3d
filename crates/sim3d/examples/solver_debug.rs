//! Debug why velocities go to zero over time.

use glam::Vec3;
use sim3d::{create_sluice, spawn_inlet_water, CellType, FlipSimulation3D, SluiceConfig};

fn count_fluid_cells(sim: &FlipSimulation3D) -> usize {
    sim.grid.cell_type.iter().filter(|&&t| t == CellType::Fluid).count()
}

fn main() {
    println!("=== VELOCITY ZERO DEBUG ===\n");

    let mut sim = FlipSimulation3D::new(80, 30, 16, 0.025);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 50;

    let config = SluiceConfig {
        slope: 0.08,
        slick_plate_len: 8,
        riffle_spacing: 10,
        riffle_height: 2,
        riffle_width: 1,
    };
    create_sluice(&mut sim, &config);
    spawn_inlet_water(&mut sim, &config, 1000, Vec3::new(2.0, 0.0, 0.0));

    println!("Grid: {}x{}x{}", sim.grid.width, sim.grid.height, sim.grid.depth);
    println!("Initial particles: {}\n", sim.particle_count());

    let dt = 1.0 / 120.0;

    for frame in 0..500 {
        // Track stats BEFORE update
        let avg_vel: Vec3 = sim.particles.list.iter()
            .map(|p| p.velocity)
            .fold(Vec3::ZERO, |a, b| a + b) / sim.particle_count().max(1) as f32;

        let max_vel = sim.particles.list.iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max);

        let zero_count = sim.particles.list.iter()
            .filter(|p| p.velocity.length() < 0.0001)
            .count();

        // Count particles by velocity range
        let slow_count = sim.particles.list.iter()
            .filter(|p| p.velocity.length() >= 0.0001 && p.velocity.length() < 0.1)
            .count();

        if frame % 30 == 0 || max_vel < 0.1 {
            let fluid = count_fluid_cells(&sim);
            let max_u = sim.grid.u.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
            let max_v = sim.grid.v.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));

            println!(
                "F{:3}: particles={:4}, fluid_cells={:4}, avgVel=({:6.3},{:6.3}), |max|={:.4}, zero={}, slow={}, grid_max=({:.2},{:.2})",
                frame, sim.particle_count(), fluid,
                avg_vel.x, avg_vel.y,
                max_vel, zero_count, slow_count,
                max_u, max_v
            );
        }

        // Stop early if all velocities are zero
        if max_vel < 0.0001 && frame > 30 {
            println!("\n!!! ALL VELOCITIES ZERO at frame {} !!!", frame);

            // Debug: check what G2P is returning
            println!("\nDiagnosing G2P for sample particle...");
            if let Some(p) = sim.particles.list.first() {
                println!("  Position: ({:.4}, {:.4}, {:.4})", p.position.x, p.position.y, p.position.z);
                println!("  Velocity: ({:.6}, {:.6}, {:.6})", p.velocity.x, p.velocity.y, p.velocity.z);
                println!("  Affine C: {:?}", p.affine_velocity);

                // Check if near solid
                let (ci, cj, ck) = sim.grid.world_to_cell(p.position);
                if sim.grid.cell_in_bounds(ci, cj, ck) {
                    let idx = sim.grid.cell_index(ci as usize, cj as usize, ck as usize);
                    println!("  Cell: ({},{},{}), type={:?}", ci, cj, ck, sim.grid.cell_type[idx]);

                    // Check nearby U faces
                    let u_idx = sim.grid.u_index(ci as usize, cj as usize, ck as usize);
                    let u_val = sim.grid.u[u_idx];
                    let is_solid = sim.grid.is_u_face_solid(ci, cj, ck);
                    println!("  U face at ({},{},{}): val={:.4}, is_solid={}", ci, cj, ck, u_val, is_solid);
                }
            }
            break;
        }

        // Run simulation step
        sim.update(dt);

        // Spawn more water
        if frame % 20 == 0 && sim.particle_count() < 8000 {
            spawn_inlet_water(&mut sim, &config, 100, Vec3::new(2.0, 0.0, 0.0));
        }
    }

    println!("\nFinal statistics:");
    let avg_vel: Vec3 = sim.particles.list.iter()
        .map(|p| p.velocity)
        .fold(Vec3::ZERO, |a, b| a + b) / sim.particle_count().max(1) as f32;
    println!("Avg velocity: ({:.6}, {:.6}, {:.6})", avg_vel.x, avg_vel.y, avg_vel.z);
}
