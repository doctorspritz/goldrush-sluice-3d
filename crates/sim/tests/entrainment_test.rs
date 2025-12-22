use sim::flip::FlipSimulation;
use sim::particle::{ParticleMaterial, ParticleState};
use sim::grid::CellType;

const DT: f32 = 1.0 / 60.0;
const CELL_SIZE: f32 = 0.5; // Fine grid
const WIDTH: usize = 40;
const HEIGHT: usize = 30; // Small channel

#[test]
fn test_sediment_entrainment() {
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Setup: Flat bed of sand at bottom
    // Place them explicitly in Bedload state
    let bed_y = 2.0 * CELL_SIZE;
    for i in 5..35 {
        let x = i as f32 * CELL_SIZE;
        sim.spawn_sand(x, bed_y, 0.0, 0.0, 1);
    }
    
    // Force them to be Bedload and updated grid
    sim.update(DT); // Classify and build grid
    for p in &mut sim.particles.list {
        if p.material == ParticleMaterial::Sand {
            p.state = ParticleState::Bedload;
            p.velocity = glam::Vec2::ZERO;
        }
    }

    // 2. Spawn FAST water above the bed
    // Velocity needs to be high enough to trigger Shields critical shear for Sand but NOT Gold.
    // Sand threshold ~24.0, Gold threshold ~88.0
    // Use 60.0 to safely differentiate.
    let water_y_start = 3.0 * CELL_SIZE;
    // let water_y_end = 10.0 * CELL_SIZE;
    let flow_vx = 60.0; // Fast flow, but not super-critical for Gold

    for _ in 0..10 { // Feed water for a few frames
        for j in 0..10 {
           let y = water_y_start + j as f32 * 0.25;
           for i in 0..5 {
               let x = i as f32 * 0.25;
               sim.spawn_water(x, y, flow_vx, 0.0, 1);
           }
        }
    }

    // 3. Run simulation and track how many particles get "lifted" (Suspended)
    // We expect the shear stress from the water to re-suspend the sand.
    let mut entrained_count = 0;
    
    for frame in 0..100 {
        // Continuously inject fast water to maintain shear
        if frame % 5 == 0 {
             for j in 0..10 {
                let y = water_y_start + j as f32 * 0.25;
                sim.spawn_water(1.0, y, flow_vx, 0.0, 1);
            }
        }
        
        sim.update(DT);
        
        let current_entrained = sim.particles.list.iter()
            .filter(|p| p.material == ParticleMaterial::Sand && p.state == ParticleState::Suspended)
            .count();
            
        // println!("Frame {}: Entrained Sand = {}", frame, current_entrained);
        
        if current_entrained > 0 {
            entrained_count = current_entrained;
        }
    }
    
    // 4. Assert that significant entrainment occurred
    assert!(entrained_count > 5, "Fast water failed to entrain sand! Max entrained: {}", entrained_count);
}

#[test]
fn test_gold_requires_higher_shear() {
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Setup: Flat bed of GOLD at bottom
    let bed_y = 2.0 * CELL_SIZE;
    for i in 5..35 {
        let x = i as f32 * CELL_SIZE;
        sim.spawn_gold(x, bed_y, 0.0, 0.0, 1);
    }
    
    // Force them to be Bedload
    sim.update(DT);
    for p in &mut sim.particles.list {
        if p.material == ParticleMaterial::Gold {
            p.state = ParticleState::Bedload;
            p.velocity = glam::Vec2::ZERO;
        }
    }

    // 2. Spawn SAME fast water as previous test (60.0)
    let water_y_start = 3.0 * CELL_SIZE;
    let flow_vx = 60.0; 

    // Initial water burst
    for _ in 0..10 { 
        for j in 0..10 {
           let y = water_y_start + j as f32 * 0.25;
           for i in 0..5 {
               let x = i as f32 * 0.25;
               sim.spawn_water(x, y, flow_vx, 0.0, 1);
           }
        }
    }

    // 3. Run simulation
    let mut entrained_count = 0;
    
    for frame in 0..100 {
        if frame % 5 == 0 {
             for j in 0..10 {
                let y = water_y_start + j as f32 * 0.25;
                sim.spawn_water(1.0, y, flow_vx, 0.0, 1);
            }
        }
        
        sim.update(DT);
        
        let current_entrained = sim.particles.list.iter()
            .filter(|p| p.material == ParticleMaterial::Gold && p.state == ParticleState::Suspended)
            .count();
        
        if current_entrained > 0 {
            entrained_count = current_entrained;
        }
    }
    
    // 4. Assert that Gold was NOT entrained (or significantly less)
    // Sand was > 5 (likely much higher). Gold allow small noise (<= 3).
    assert!(entrained_count <= 3, "Gold entrainment should be minimal! Found {} entrained.", entrained_count);
}
