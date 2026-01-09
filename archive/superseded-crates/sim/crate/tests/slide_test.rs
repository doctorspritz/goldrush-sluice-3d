//! Regression test for bed particle freezing
//! 
//! Verified behavior:
//! Particles in `ParticleState::Bedload` should NOT be advected, effectively "frozen" in place.
//! This optimization allows for large sediment beds without cost.

use sim::particle::ParticleState;
use sim::FlipSimulation;
use glam::Vec2;

#[test]
fn test_bed_particles_frozen() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Spawn a sediment particle on the floor
    // Give it a non-zero velocity so it WOULd move if not frozen
    let x = WIDTH as f32 * CELL_SIZE * 0.5;
    let y = CELL_SIZE * 1.5; // Just above bottom wall
    
    // Spawn manually to control state
    sim.spawn_sand(x, y, 10.0, 0.0, 1);
    
    // 2. Set state to Bedload
    if let Some(p) = sim.particles.list.first_mut() {
        p.state = ParticleState::Bedload;
        p.velocity = Vec2::new(100.0, 0.0); // High horizontal velocity
    }
    
    let initial_pos = sim.particles.list[0].position;
    
    // 3. Run for a SINGLE frame
    // In frame 1, advection runs BEFORE state update (which might wake it up due to high velocity)
    // So successful freezing means ZERO motion in this frame.
    sim.update(DT);
    
    let final_pos = sim.particles.list[0].position;
    
    // 4. Verification: Position should be UNCHANGED (exact match expected if skipped)
    assert_eq!(initial_pos, final_pos, 
        "Bedload particle moved! initial: {:?}, final: {:?}", initial_pos, final_pos);
        
    // Also verify velocity is preserved (optional, but good for when it wakes up)
    let final_vel = sim.particles.list[0].velocity;
    assert!(final_vel.x > 90.0, "Velocity decayed even though frozen!");
}
