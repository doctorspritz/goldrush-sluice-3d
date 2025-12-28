use sim::flip::FlipSimulation;
use sim::particle::{ParticleMaterial, ParticleState};

const DT: f32 = 1.0 / 60.0;
const CELL_SIZE: f32 = 0.5;
const WIDTH: usize = 40;
const HEIGHT: usize = 30;

#[test]
fn test_regression_mixed_simulation() {
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Spawn a mix of particles
    // Water column
    for y in 5..25 {
        for x in 5..15 {
            sim.spawn_water(x as f32 * CELL_SIZE, y as f32 * CELL_SIZE, 0.0, 0.0, 1);
        }
    }
    
    // Sand bed (initially suspended, should settle)
    for x in 15..35 {
         sim.spawn_sand(x as f32 * CELL_SIZE, 5.0 * CELL_SIZE, 0.0, 0.0, 1);
    }
    
    // Extra sand particles (initially suspended)
    for x in 20..30 {
        sim.spawn_sand(x as f32 * CELL_SIZE, 15.0 * CELL_SIZE, 0.0, 0.0, 1);
    }
    
    let initial_count = sim.particles.len();
    
    // 2. Run for 100 frames
    for _ in 0..100 {
        sim.update(DT);
    }
    
    // 3. Verify Conservation and Stability
    
    // Count should be identical (unless they fell out of bounds, but we have walls?)
    // Actually the default sim has walls at x=0, x=width. 
    // Particles might fall out bottom if no floor?
    // sim.classify_cells() marks floor at height-1.
    // So they should stay in bounds.
    
    let final_count = sim.particles.len();
    assert_eq!(initial_count, final_count, "Particle count should be conserved (no leaks).");
    
    // Verify some settled
    let bedload_count = sim.particles.iter()
        .filter(|p| p.state == ParticleState::Bedload)
        .count();
        
    assert!(bedload_count > 0, "Some particles should have settled into Bedload.");
    
    // Verify velocities are not NaN/Infinite (stability)
    for p in sim.particles.iter() {
        assert!(p.velocity.is_finite(), "Particle velocity must be finite.");
        assert!(p.position.is_finite(), "Particle position must be finite.");
    }
}
