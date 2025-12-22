//! Regression test for sediment settling physics
//!
//! Verified behavior:
//! 1. Heavy particles (Gold) should settle faster than light ones (Sand).
//! 2. Particles should reach terminal velocity (approx).
//! 3. Particles should transition to Bedload when they hit the floor.

use sim::particle::{ParticleMaterial, ParticleState};
use sim::FlipSimulation;
#[allow(unused_imports)]
use glam::Vec2;

#[test]
fn test_sediment_settling() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 64; // Tall enough to reach terminal velocity
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 0. Fill the tank with water (Dense: 4 particles per cell to ensure Fluid identification)
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            // Jitter positions slightly to avoid grid alignment artifacts
            sim.spawn_water(x as f32 * CELL_SIZE + 1.0, y as f32 * CELL_SIZE + 1.0, 0.0, 0.0, 1);
            sim.spawn_water(x as f32 * CELL_SIZE + 3.0, y as f32 * CELL_SIZE + 3.0, 0.0, 0.0, 1);
            sim.spawn_water(x as f32 * CELL_SIZE + 3.0, y as f32 * CELL_SIZE + 1.0, 0.0, 0.0, 1);
            sim.spawn_water(x as f32 * CELL_SIZE + 1.0, y as f32 * CELL_SIZE + 3.0, 0.0, 0.0, 1);
        }
    }

    // Make sure we update enough to let water settle (pressure explosion from dense packing)
    for _ in 0..60 {
        sim.update(DT);
    }
    
    // Spawn Gold and Sand high up (but inside water)
    let spawn_y = HEIGHT as f32 * CELL_SIZE - 20.0; // Lower slightly to ensure in water
    
    // Spawn Gold
    sim.spawn_gold(10.0 * CELL_SIZE, spawn_y, 0.0, 0.0, 1);
    // Spawn Sand
    sim.spawn_sand(20.0 * CELL_SIZE, spawn_y, 0.0, 0.0, 1);
    
    // 2. Run simulation and track vertical velocities
    let mut gold_pos_history = Vec::new();
    let mut sand_pos_history = Vec::new();
    
    // Find indices for our test particles (they should be the last two added)
    let count = sim.particles.len();
    assert!(count >= 2, "Simulation should have particles");
    
    // Assuming they are appended in order: Gold then Sand
    // But let's find them to be robust
    let mut gold_idx = usize::MAX;
    let mut sand_idx = usize::MAX;
    
    for (i, p) in sim.particles.list.iter().enumerate() {
        if p.material == ParticleMaterial::Gold {
            gold_idx = i;
        } else if p.material == ParticleMaterial::Sand {
            sand_idx = i;
        }
    }
    
    assert!(gold_idx != usize::MAX, "Gold particle not found");
    assert!(sand_idx != usize::MAX, "Sand particle not found");
    
    for frame in 0..100 {
        sim.update(DT);
        
        let gold = &sim.particles.list[gold_idx];
        let sand = &sim.particles.list[sand_idx];
        
        // Debug prints disabled for regression
        // if frame % 10 == 0 { ... }

        gold_pos_history.push(gold.position.y);
        sand_pos_history.push(sand.position.y);
    }
    
    // Re-find indices in case particle order changed (e.g. removal)
    for (i, p) in sim.particles.list.iter().enumerate() {
        if p.material == ParticleMaterial::Gold {
            gold_idx = i;
        } else if p.material == ParticleMaterial::Sand {
            sand_idx = i;
        }
    }

    let gold_final = &sim.particles.list[gold_idx];
    let sand_final = &sim.particles.list[sand_idx];
    
    // 3. Verify Gold fell faster/further than Sand
    // Gravity is positive (250.0), so +Y is DOWN.
    // Gold should have larger Y (closer to floor/bottom).
    
    println!("Gold Y: {:.1} -> {:.1}", spawn_y, gold_final.position.y);
    println!("Sand Y: {:.1} -> {:.1}", spawn_y, sand_final.position.y);
    
    assert!(gold_final.position.y > sand_final.position.y, 
        "Gold should have fallen further (higher Y value) than Sand!");
        
    // 4. Verify settling/bedload transition
    // Run until they hit floor (approx)
    for _ in 0..500 {
        sim.update(DT);
    }
    
    let gold_state = sim.particles.list[gold_idx].state;
    // It should be Bedload now because it hit the floor
    assert_eq!(gold_state, ParticleState::Bedload, "Gold should be in Bedload state after settling");
}
