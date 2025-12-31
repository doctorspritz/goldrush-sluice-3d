//! Phase 3: Full 2D drag coupling tests

use sim::flip::FlipSimulation;
use sim::particle::Particle;
use glam::Vec2;

/// Find a sediment particle (sand or gold) by iterating through all particles
fn find_sediment_particle(sim: &FlipSimulation) -> Option<&Particle> {
    sim.particles.list.iter().find(|p| p.is_sediment())
}

/// Find gold particle specifically
fn find_gold_particle(sim: &FlipSimulation) -> Option<&Particle> {
    sim.particles.list.iter().find(|p| p.material == sim::particle::ParticleMaterial::Gold)
}

/// Find sand particle specifically
fn find_sand_particle(sim: &FlipSimulation) -> Option<&Particle> {
    sim.particles.list.iter().find(|p| p.material == sim::particle::ParticleMaterial::Sand)
}

#[test]
fn test_vertical_drag_applies() {
    // Test that vertical drag is actually being applied by comparing settling
    // in upward-moving water vs still water.
    let mut sim_still = FlipSimulation::new(20, 30, 4.0);
    let mut sim_upward = FlipSimulation::new(20, 30, 4.0);

    // Fill both with water - one still, one moving upward
    for y in 5..25 {
        for x in 5..15 {
            sim_still.spawn_water(x as f32 * 4.0, y as f32 * 4.0, 0.0, 0.0, 1);
            sim_upward.spawn_water(x as f32 * 4.0, y as f32 * 4.0, 0.0, -50.0, 1); // strong upward
        }
    }

    // Place sand particles in both
    sim_still.particles.list.push(Particle::sand(Vec2::new(40.0, 60.0), Vec2::ZERO));
    sim_upward.particles.list.push(Particle::sand(Vec2::new(40.0, 60.0), Vec2::ZERO));

    // Run 30 frames
    for _ in 0..30 {
        sim_still.update(1.0 / 60.0);
        sim_upward.update(1.0 / 60.0);
    }

    let sand_still = find_sand_particle(&sim_still).unwrap();
    let sand_upward = find_sand_particle(&sim_upward).unwrap();

    let vy_still = sand_still.velocity.y;
    let vy_upward = sand_upward.velocity.y;

    println!("Sand in still water: vy = {}", vy_still);
    println!("Sand in upward flow: vy = {}", vy_upward);

    // The particle in upward-moving water should have LESS downward velocity
    // than the one in still water (even if both are still sinking)
    // This proves vertical drag is working.
    assert!(vy_upward < vy_still,
        "Vertical drag should reduce settling rate. Still water vy={}, upward flow vy={}",
        vy_still, vy_upward);
}

#[test]
fn test_horizontal_transport_preserved() {
    let mut sim = FlipSimulation::new(40, 20, 4.0);

    // Create horizontal water flow - larger water region
    for y in 3..17 {
        for x in 2..38 {
            sim.spawn_water(x as f32 * 4.0, y as f32 * 4.0, 60.0, 0.0, 1);
        }
    }

    // Add gold and sand particles inside the water region
    sim.particles.list.push(Particle::gold(Vec2::new(30.0, 40.0), Vec2::ZERO));
    sim.particles.list.push(Particle::sand(Vec2::new(30.0, 42.0), Vec2::ZERO));

    let gold_start_x = find_gold_particle(&sim).unwrap().position.x;
    let sand_start_x = find_sand_particle(&sim).unwrap().position.x;

    // Run 50 frames
    for _ in 0..50 {
        sim.update(1.0 / 60.0);
    }

    let gold = find_gold_particle(&sim);
    let sand = find_sand_particle(&sim);

    assert!(gold.is_some(), "Gold particle should still exist");
    assert!(sand.is_some(), "Sand particle should still exist");

    let gold_end_x = gold.unwrap().position.x;
    let sand_end_x = sand.unwrap().position.x;

    let gold_moved = gold_end_x - gold_start_x;
    let sand_moved = sand_end_x - sand_start_x;

    println!("Gold moved: {}", gold_moved);
    println!("Sand moved: {}", sand_moved);

    // Sand should move horizontally - it has higher drag coefficient
    // Gold has very low drag (high density) so may barely move, that's physically correct
    assert!(sand_moved > 2.0, "Sand should move horizontally with water, got {}", sand_moved);

    // Key physical insight: Sand (lower density) should move MORE than gold (higher density)
    // because sand experiences stronger drag
    assert!(sand_moved > gold_moved,
        "Sand (low density) should move more than gold (high density). Sand={}, Gold={}",
        sand_moved, gold_moved);
}

#[test]
fn test_settling_still_works_in_still_water() {
    let mut sim = FlipSimulation::new(20, 40, 4.0);

    // Fill with STILL water - larger region
    for y in 3..37 {
        for x in 3..17 {
            sim.spawn_water(x as f32 * 4.0, y as f32 * 4.0, 0.0, 0.0, 1);
        }
    }

    // Drop gold particle in the middle of the water
    sim.particles.list.push(Particle::gold(Vec2::new(40.0, 30.0), Vec2::ZERO));

    let start_y = find_gold_particle(&sim).unwrap().position.y;

    // Run 120 frames (2 seconds)
    for _ in 0..120 {
        sim.update(1.0 / 60.0);
    }

    let gold = find_gold_particle(&sim);
    assert!(gold.is_some(), "Gold particle should still exist");
    let end_y = gold.unwrap().position.y;

    println!("Gold y: {} -> {}", start_y, end_y);

    // Gold should have settled (moved down - positive y direction)
    // Using a smaller threshold since we're now using direct settling velocity
    assert!(end_y - start_y > 10.0,
        "Gold should settle in still water, moved {} pixels", end_y - start_y);
}
