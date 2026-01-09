//! Regression test for particle tunneling
//! 
//! Verified behavior:
//! Particles moving at high velocity (> cell_size per step) must NOT pass through
//! thin walls (1 cell thick).

use sim::FlipSimulation;
use glam::Vec2;

#[test]
fn test_fast_particle_vs_wall() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Create a vertical solid wall (1-cell thick) at x=16
    for j in 0..HEIGHT {
        sim.grid.set_solid(16, j);
    }
    sim.grid.compute_sdf();
    
    // 2. Fire a particle at it at 5x usual max velocity
    // "Usual max" is roughly CFL < 1, so > 1.0 cell/step is "fast"
    // 5 cells per step = 5 * 4.0 = 20.0 distance per step
    // Velocity = distance / dt = 20.0 / (1/60) = 1200.0
    // That seems extreme, let's use the USER's prompt suggestions:
    // "5x your usual max velocity". Engine clamps at 150.0 usually.
    // Let's try 1200.0 (20 cells per frame!) - this GUARANTEES tunneling without micro-steps.
    let velocity = Vec2::new(1200.0, 0.0);
    
    // Spawn at x=5, pointing right (towards wall at 16)
    // Distance to wall = (16-5)*4 = 44 units
    // At 1200u/s, it covers 20u/frame. 
    // Frame 0: 20.0
    // Frame 1: 40.0
    // Frame 2: 60.0 (4 units from wall start)
    // Frame 3: 80.0 (WAY past wall end at 68)
    sim.spawn_water(5.0 * CELL_SIZE, 16.0 * CELL_SIZE, velocity.x, velocity.y, 1);
    
    // Force the velocity on the particle (spawn might randomize or drag)
    sim.particles.list[0].velocity = velocity;
    
    // 3. Run for 200 frames
    // We expect it to hit the wall and stop/slide, effectively staying on the left side (x < 16*CELL_SIZE)
    let wall_x = 16.0 * CELL_SIZE;
    
    for i in 0..200 {
        sim.update(DT);
        let p = &sim.particles.list[0];
        
        // Assert it hasn't tunneled
        // It must be on the left side of the wall (plus margin for the wall thickness itself)
        // Wall is from x=16*4=64 to 68.
        // We allow it to be IN the wall (collision resolution might push it out), but not PAST it.
        assert!(p.position.x < wall_x + CELL_SIZE, 
            "Frame {}: Tunneling detected! Particle at x={} passed wall at {}", 
            i, p.position.x, wall_x);
            
        // Also check if it's way out of bounds (unbounded energy test, partial)
        assert!(p.position.x > 0.0, "Particle flew out left side?");
    }
    
    // Pass condition: Particle ends up flush against boundary (or close to it)
    let final_pos = sim.particles.list[0].position.x;
    println!("Final position: {}", final_pos);
    
    // Should be near the wall (e.g. within 1-2 cells of collision point)
    assert!(final_pos > wall_x - CELL_SIZE * 4.0, 
        "Particle stopped too early? Pos: {}", final_pos);
}
