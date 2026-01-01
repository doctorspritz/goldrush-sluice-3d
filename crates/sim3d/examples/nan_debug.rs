//! Debug NaN/Inf issues in the simulation.

use glam::Vec3;
use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig};

fn check_nan(name: &str, arr: &[f32]) -> bool {
    for (i, &v) in arr.iter().enumerate() {
        if !v.is_finite() {
            println!("!!! {} has non-finite at index {}: {}", name, i, v);
            return true;
        }
    }
    false
}

fn check_particles(sim: &FlipSimulation3D) -> bool {
    for (i, p) in sim.particles.list.iter().enumerate() {
        if !p.velocity.is_finite() || !p.position.is_finite() {
            println!("!!! Particle {} has non-finite: pos={:?}, vel={:?}", i, p.position, p.velocity);
            return true;
        }
    }
    false
}

fn main() {
    println!("=== NaN DEBUG ===\n");

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
    spawn_inlet_water(&mut sim, &config, 500, Vec3::new(1.5, 0.0, 0.0));

    println!("Initial: {} particles", sim.particle_count());

    let dt = 1.0 / 60.0;
    for frame in 0..200 {
        // Check for NaN/Inf BEFORE update
        let mut has_nan = false;
        has_nan |= check_nan("u", &sim.grid.u);
        has_nan |= check_nan("v", &sim.grid.v);
        has_nan |= check_nan("w", &sim.grid.w);
        has_nan |= check_nan("pressure", &sim.grid.pressure);
        has_nan |= check_nan("divergence", &sim.grid.divergence);
        has_nan |= check_particles(&sim);

        if has_nan {
            println!("NaN detected at frame {}", frame);
            break;
        }

        // Track velocity stats
        let avg_vel: Vec3 = sim.particles.list.iter()
            .map(|p| p.velocity)
            .fold(Vec3::ZERO, |a, b| a + b) / sim.particle_count().max(1) as f32;

        let max_vel = sim.particles.list.iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max);

        let zero_count = sim.particles.list.iter()
            .filter(|p| p.velocity.length() < 0.0001)
            .count();

        if frame % 20 == 0 || max_vel < 0.01 {
            println!(
                "F{:3}: particles={}, avgVel=({:.3},{:.3},{:.3}), max={:.4}, zero={}",
                frame, sim.particle_count(), avg_vel.x, avg_vel.y, avg_vel.z, max_vel, zero_count
            );
        }

        // Stop if all velocities are zero
        if max_vel < 0.0001 && frame > 10 {
            println!("\n!!! ALL VELOCITIES ZERO at frame {} !!!", frame);

            // Debug: print a few particles
            println!("Sample particles:");
            for (i, p) in sim.particles.list.iter().enumerate().take(5) {
                println!("  [{}] pos=({:.3},{:.3},{:.3}), vel=({:.6},{:.6},{:.6})",
                    i, p.position.x, p.position.y, p.position.z,
                    p.velocity.x, p.velocity.y, p.velocity.z);
            }

            // Check grid
            let nonzero_u = sim.grid.u.iter().filter(|&&v| v.abs() > 0.0001).count();
            let nonzero_v = sim.grid.v.iter().filter(|&&v| v.abs() > 0.0001).count();
            println!("Grid nonzero: u={}/{}, v={}/{}", nonzero_u, sim.grid.u.len(), nonzero_v, sim.grid.v.len());

            break;
        }

        // Update sim
        sim.update(dt);

        // Spawn more water
        if frame % 5 == 0 && sim.particle_count() < 2000 {
            spawn_inlet_water(&mut sim, &config, 20, Vec3::new(1.5, 0.0, 0.0));
        }
    }
}
