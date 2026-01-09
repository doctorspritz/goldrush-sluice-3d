use dfsph::DfsphSimulation;
use sim::physics;
use glam::Vec2;

#[test]
fn test_gravity_application() {
    let mut sim = DfsphSimulation::new(100, 100, 1.0);
    // Use precise spawning
    sim.spawn_particle_internal(Vec2::new(50.0, 50.0), Vec2::ZERO, sim::ParticleMaterial::Water);
    
    let initial_p = sim.particles.list[0];
    assert_eq!(initial_p.velocity, Vec2::ZERO);

    // Run one update step
    let dt = 0.016; 
    sim.update(dt);

    let updated_p = sim.particles.list[0];
    
    // Check Velocity: GRAVITY * dt * damping
    // 250 * 0.016 * 0.999 = 3.996 (damping applied at simulation.rs:190)
    let expected_vy = physics::GRAVITY * dt * 0.999; // Account for damping
    assert!((updated_p.velocity.y - expected_vy).abs() < 0.01,
        "Velocity Y should be approx {}, got {}", expected_vy, updated_p.velocity.y);
    
    // Check Position: Start + Velocity * dt approx
    // 50.0 + ~4.0 * 0.016 = 50.0 + 0.064 = 50.064
    let expected_y = 50.064;
    assert!((updated_p.position.y - expected_y).abs() < 1.0,
        "Position Y should be approx {}, got {}", expected_y, updated_p.position.y);
}

#[test]
fn test_floor_collision() {
    let mut sim = DfsphSimulation::new(100, 100, 2.0);
    
    // Create floor at y=90
    // Grid height is 100*2 = 200.
    // Cell at y=90 is index 45.
    let width_cells = 100;
    for x in 0..width_cells {
        let idx = 45 * width_cells + x;
        sim.grid_solid[idx] = true;
    }
    
    // Spawn particle at 88 (very close to floor)
    // Floor is at 90.
    sim.spawn_particle_internal(Vec2::new(50.0, 88.0), Vec2::new(0.0, 10.0), sim::ParticleMaterial::Water);
    
    // Update multiple frames to push it into floor
    for _ in 0..10 {
        sim.update(0.016);
    }
    
    let p = sim.particles.list[0];
    
    // Should be above 90.
    // With bouncing, it might be anywhere above.
    assert!(p.position.y <= 90.0, "Particle should respect floor at 90.0, got {}", p.position.y);
    assert!(p.position.y > 80.0, "Particle shouldn't fly away, got {}", p.position.y);
    // Assert Velocity
    // If pos reset to old_pos (89.0), then (89.0 - 89.0) / dt == 0.
    assert_eq!(p.velocity, Vec2::ZERO, "Velocity should be zeroed on impact");
}
