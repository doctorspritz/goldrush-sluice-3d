//! Minimal test to diagnose velocity decay.
//!
//! Tests if FLIP transfer preserves velocity in open space.

use glam::Vec3;
use sim3d::FlipSimulation3D;

fn main() {
    println!("=== VELOCITY DECAY DIAGNOSTIC ===\n");

    // Very simple scenario: particles in open box, no gravity
    let mut sim = FlipSimulation3D::new(16, 16, 16, 0.5);
    sim.gravity = Vec3::ZERO; // No gravity
    sim.pressure_iterations = 50;
    sim.flip_ratio = 0.97;

    // Spawn a block of particles moving right
    let initial_vel = Vec3::new(2.0, 0.0, 0.0);
    for i in 4..8 {
        for j in 4..8 {
            for k in 4..8 {
                sim.spawn_particle_with_velocity(
                    Vec3::new(
                        (i as f32 + 0.5) * 0.5,
                        (j as f32 + 0.5) * 0.5,
                        (k as f32 + 0.5) * 0.5,
                    ),
                    initial_vel,
                );
            }
        }
    }

    println!(
        "Initial: {} particles with velocity {:?}",
        sim.particle_count(),
        initial_vel
    );

    // Run simulation and track velocity
    let dt = 1.0 / 60.0;
    for frame in 0..120 {
        // Compute average velocity
        let avg_vel: Vec3 = sim
            .particles
            .list
            .iter()
            .map(|p| p.velocity)
            .fold(Vec3::ZERO, |a, b| a + b)
            / sim.particle_count() as f32;

        let max_vel: f32 = sim
            .particles
            .list
            .iter()
            .map(|p| p.velocity.length())
            .fold(0.0, f32::max);

        if frame % 10 == 0 {
            println!(
                "Frame {:3}: avg_vel=({:7.3}, {:7.3}, {:7.3}), |max|={:.3}",
                frame, avg_vel.x, avg_vel.y, avg_vel.z, max_vel
            );
        }

        sim.update(dt);
    }

    // Final state
    let avg_vel: Vec3 = sim
        .particles
        .list
        .iter()
        .map(|p| p.velocity)
        .fold(Vec3::ZERO, |a, b| a + b)
        / sim.particle_count() as f32;

    println!(
        "\nFinal: avg_vel=({:.3}, {:.3}, {:.3})",
        avg_vel.x, avg_vel.y, avg_vel.z
    );

    // Sample individual particles
    println!("\nSample particles:");
    for (i, p) in sim.particles.list.iter().enumerate().take(5) {
        println!(
            "  [{}] pos=({:.2}, {:.2}, {:.2}), vel=({:.3}, {:.3}, {:.3}), C={:?}",
            i,
            p.position.x,
            p.position.y,
            p.position.z,
            p.velocity.x,
            p.velocity.y,
            p.velocity.z,
            p.affine_velocity
        );
    }

    // Test with 0 pressure iterations
    println!("\n=== TEST 2: No pressure solve ===\n");
    let mut sim2 = FlipSimulation3D::new(16, 16, 16, 0.5);
    sim2.gravity = Vec3::ZERO;
    sim2.pressure_iterations = 0; // Skip pressure
    sim2.flip_ratio = 0.97;

    for i in 4..8 {
        for j in 4..8 {
            for k in 4..8 {
                sim2.spawn_particle_with_velocity(
                    Vec3::new(
                        (i as f32 + 0.5) * 0.5,
                        (j as f32 + 0.5) * 0.5,
                        (k as f32 + 0.5) * 0.5,
                    ),
                    initial_vel,
                );
            }
        }
    }

    for frame in 0..60 {
        let avg_vel: Vec3 = sim2
            .particles
            .list
            .iter()
            .map(|p| p.velocity)
            .fold(Vec3::ZERO, |a, b| a + b)
            / sim2.particle_count() as f32;

        if frame % 10 == 0 {
            println!(
                "Frame {:3}: avg_vel=({:7.3}, {:7.3}, {:7.3})",
                frame, avg_vel.x, avg_vel.y, avg_vel.z
            );
        }

        sim2.update(dt);
    }

    // Test with 100% PIC (flip_ratio = 0)
    println!("\n=== TEST 3: 100% PIC (no FLIP) ===\n");
    let mut sim3 = FlipSimulation3D::new(16, 16, 16, 0.5);
    sim3.gravity = Vec3::ZERO;
    sim3.pressure_iterations = 50;
    sim3.flip_ratio = 0.0; // Pure PIC

    for i in 4..8 {
        for j in 4..8 {
            for k in 4..8 {
                sim3.spawn_particle_with_velocity(
                    Vec3::new(
                        (i as f32 + 0.5) * 0.5,
                        (j as f32 + 0.5) * 0.5,
                        (k as f32 + 0.5) * 0.5,
                    ),
                    initial_vel,
                );
            }
        }
    }

    for frame in 0..60 {
        let avg_vel: Vec3 = sim3
            .particles
            .list
            .iter()
            .map(|p| p.velocity)
            .fold(Vec3::ZERO, |a, b| a + b)
            / sim3.particle_count() as f32;

        if frame % 10 == 0 {
            println!(
                "Frame {:3}: avg_vel=({:7.3}, {:7.3}, {:7.3})",
                frame, avg_vel.x, avg_vel.y, avg_vel.z
            );
        }

        sim3.update(dt);
    }
}
