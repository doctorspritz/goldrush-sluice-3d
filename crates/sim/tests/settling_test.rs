//! Regression test for sediment settling physics
//!
//! Verified behavior:
//! 1. Sand particles should settle through water.
//! 2. Sand should reach the bottom floor (y position increases).
//!
//! Note: Bedload state transitions are disabled in Phase 2, so we only
//! verify that sand settles and reaches the bottom.

use sim::particle::ParticleMaterial;
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

    // Set up solid floor for particles to land on
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1); // Floor at bottom
    }
    sim.grid.compute_sdf();

    // 0. Fill the tank with water (Dense: 4 particles per cell to ensure Fluid identification)
    // Don't spawn water in the solid floor row
    for y in 0..(HEIGHT - 1) {
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

    // Spawn Sand high up (inside water)
    let spawn_y = HEIGHT as f32 * CELL_SIZE - 20.0; // Lower slightly to ensure in water
    sim.spawn_sand(15.0 * CELL_SIZE, spawn_y, 0.0, 0.0, 1);

    // 2. Run simulation and track vertical positions
    let mut sand_pos_history = Vec::new();

    // Find sand particle
    let count = sim.particles.len();
    assert!(count >= 1, "Simulation should have particles");

    let mut sand_idx = usize::MAX;
    for (i, p) in sim.particles.list.iter().enumerate() {
        if p.material == ParticleMaterial::Sand {
            sand_idx = i;
            break;
        }
    }

    assert!(sand_idx != usize::MAX, "Sand particle not found");

    for _frame in 0..100 {
        sim.update(DT);

        let sand = &sim.particles.list[sand_idx];
        sand_pos_history.push(sand.position.y);
    }

    // Re-find index in case particle order changed (e.g. removal)
    for (i, p) in sim.particles.list.iter().enumerate() {
        if p.material == ParticleMaterial::Sand {
            sand_idx = i;
            break;
        }
    }

    let sand_final = &sim.particles.list[sand_idx];

    // 3. Verify Sand has fallen (Y increases downward with gravity)
    println!("Sand Y: {:.1} -> {:.1}", spawn_y, sand_final.position.y);

    assert!(sand_final.position.y > spawn_y,
        "Sand should have fallen (higher Y value after settling)!");

    // 4. Verify sand reaches the bottom floor
    // Run until they hit floor (approx)
    for _ in 0..500 {
        sim.update(DT);
    }

    // Re-find sand after long simulation
    for (i, p) in sim.particles.list.iter().enumerate() {
        if p.material == ParticleMaterial::Sand {
            sand_idx = i;
            break;
        }
    }

    let sand_after_settling = &sim.particles.list[sand_idx];
    let floor_y = (HEIGHT - 1) as f32 * CELL_SIZE;

    // Sand should be near the floor (within 2 cell heights)
    assert!(
        sand_after_settling.position.y > floor_y - 2.0 * CELL_SIZE,
        "Sand should have settled near floor. Expected y > {:.1}, got {:.1}",
        floor_y - 2.0 * CELL_SIZE,
        sand_after_settling.position.y
    );
}
