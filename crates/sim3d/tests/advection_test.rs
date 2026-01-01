//! 3D Advection tests
//!
//! Tests particle advection and boundary enforcement specifically
//! for 3D configurations, focusing on Z-boundary handling.

use sim3d::{FlipSimulation3D, Vec3};

/// Test that particles are properly contained in Z boundaries
#[test]
fn test_z_boundary_containment() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);
    let (min, max) = sim.world_bounds();

    // Spawn particle at center with velocity toward back Z wall
    sim.spawn_particle_with_velocity(
        Vec3::new(2.0, 2.0, 0.5),  // Near back Z wall
        Vec3::new(0.0, 0.0, -10.0),  // Moving into back wall
    );

    // Run several frames
    for _ in 0..20 {
        sim.update(1.0 / 60.0);
    }

    let pos = sim.particles.list[0].position;

    assert!(
        pos.z >= min.z,
        "Particle should not go below Z min boundary. Got z: {}",
        pos.z
    );
}

/// Test that particles are properly contained at front Z boundary
#[test]
fn test_front_z_boundary_containment() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);
    let (min, max) = sim.world_bounds();

    // Spawn particle at center with velocity toward front Z wall
    sim.spawn_particle_with_velocity(
        Vec3::new(2.0, 2.0, max.z - 0.5),  // Near front Z wall
        Vec3::new(0.0, 0.0, 10.0),  // Moving into front wall
    );

    // Disable gravity to isolate Z motion
    sim.gravity = Vec3::ZERO;

    // Run several frames
    for _ in 0..20 {
        sim.update(1.0 / 60.0);
    }

    let pos = sim.particles.list[0].position;

    assert!(
        pos.z <= max.z,
        "Particle should not go above Z max boundary. Got z: {}, max: {}",
        pos.z, max.z
    );
}

/// Test that Z-velocity is properly reflected at Z boundaries
#[test]
fn test_z_velocity_reflection() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Spawn particle moving fast toward back Z wall
    sim.spawn_particle_with_velocity(
        Vec3::new(2.0, 2.0, 0.5),
        Vec3::new(0.0, 0.0, -5.0),
    );

    // Disable gravity
    sim.gravity = Vec3::ZERO;

    // Get initial Z velocity
    let initial_vz = sim.particles.list[0].velocity.z;

    // Run several frames
    for _ in 0..20 {
        sim.update(1.0 / 60.0);
    }

    let final_vz = sim.particles.list[0].velocity.z;

    // Velocity should have bounced (sign changed or at least reduced)
    // With damping, it should be smaller in magnitude
    assert!(
        final_vz.abs() < initial_vz.abs() || final_vz * initial_vz < 0.0,
        "Z velocity should be reflected or damped at boundary. Initial: {}, Final: {}",
        initial_vz, final_vz
    );
}

/// Test that particles don't tunnel through thin geometry in Z
#[test]
fn test_no_z_tunneling() {
    let mut sim = FlipSimulation3D::new(16, 8, 16, 0.25);  // Finer grid
    let (min, max) = sim.world_bounds();

    // Spawn fast-moving particle
    sim.spawn_particle_with_velocity(
        Vec3::new(2.0, 1.0, 1.0),
        Vec3::new(0.0, 0.0, 20.0),  // Very fast Z velocity
    );

    sim.gravity = Vec3::ZERO;

    // Run for a while
    for _ in 0..100 {
        sim.update(1.0 / 120.0);  // Small timestep
    }

    let pos = sim.particles.list[0].position;

    // Particle should still be in bounds
    assert!(
        pos.z >= min.z && pos.z <= max.z,
        "Particle should not tunnel outside Z bounds. Got z: {}, bounds: [{}, {}]",
        pos.z, min.z, max.z
    );
}

/// Test that advection preserves all three position components
#[test]
fn test_advection_updates_all_components() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    let initial_pos = Vec3::new(2.0, 2.0, 2.0);
    let velocity = Vec3::new(1.0, 2.0, 3.0);

    sim.spawn_particle_with_velocity(initial_pos, velocity);
    sim.gravity = Vec3::ZERO;  // Disable gravity for predictable motion

    // Run one step
    sim3d::advection::advect_particles(&mut sim.particles, 1.0 / 60.0);

    let final_pos = sim.particles.list[0].position;
    let expected_displacement = velocity * (1.0 / 60.0);

    // All components should have changed
    let dx = (final_pos.x - initial_pos.x - expected_displacement.x).abs();
    let dy = (final_pos.y - initial_pos.y - expected_displacement.y).abs();
    let dz = (final_pos.z - initial_pos.z - expected_displacement.z).abs();

    assert!(dx < 0.001, "X displacement incorrect: expected {}, got {}", expected_displacement.x, final_pos.x - initial_pos.x);
    assert!(dy < 0.001, "Y displacement incorrect: expected {}, got {}", expected_displacement.y, final_pos.y - initial_pos.y);
    assert!(dz < 0.001, "Z displacement incorrect: expected {}, got {}", expected_displacement.z, final_pos.z - initial_pos.z);
}

/// Test particles falling through 3D volume
#[test]
fn test_gravity_falling_3d() {
    let mut sim = FlipSimulation3D::new(8, 16, 8, 0.5);

    // Spawn particle block near top
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                sim.spawn_particle(Vec3::new(
                    2.0 + i as f32 * 0.3,
                    6.0 + j as f32 * 0.3,
                    2.0 + k as f32 * 0.3,
                ));
            }
        }
    }

    let initial_y: f32 = sim.particles.list.iter()
        .map(|p| p.position.y)
        .sum::<f32>() / sim.particle_count() as f32;

    // Run simulation
    for _ in 0..100 {
        sim.update(1.0 / 60.0);
    }

    let final_y: f32 = sim.particles.list.iter()
        .map(|p| p.position.y)
        .sum::<f32>() / sim.particle_count() as f32;

    assert!(
        final_y < initial_y,
        "Particles should fall due to gravity. Initial Y: {}, Final Y: {}",
        initial_y, final_y
    );
}

/// Test that particles exiting through outlet are removed
#[test]
fn test_outlet_removal() {
    let mut sim = FlipSimulation3D::new(16, 8, 8, 0.5);

    // Spawn particle moving toward outlet (right edge)
    sim.spawn_particle_with_velocity(
        Vec3::new(7.0, 2.0, 2.0),  // Near right edge
        Vec3::new(20.0, 0.0, 0.0),  // Moving right
    );

    let initial_count = sim.particle_count();

    // Run simulation
    for _ in 0..50 {
        sim.update(1.0 / 60.0);
    }

    let final_count = sim.particle_count();

    // Particle should have exited
    assert!(
        final_count < initial_count,
        "Particle should exit through outlet. Initial: {}, Final: {}",
        initial_count, final_count
    );
}

/// Test Z boundary does not remove particles (closed wall, not outlet)
#[test]
fn test_z_boundary_does_not_remove() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Spawn particle near Z boundary moving into it
    sim.spawn_particle_with_velocity(
        Vec3::new(2.0, 2.0, 0.5),
        Vec3::new(0.0, 0.0, -10.0),
    );

    let initial_count = sim.particle_count();

    // Disable gravity
    sim.gravity = Vec3::ZERO;

    // Run simulation
    for _ in 0..50 {
        sim.update(1.0 / 60.0);
    }

    let final_count = sim.particle_count();

    // Particle should NOT be removed (Z is a closed wall)
    assert_eq!(
        final_count, initial_count,
        "Particle should not be removed at Z boundary. Initial: {}, Final: {}",
        initial_count, final_count
    );
}
