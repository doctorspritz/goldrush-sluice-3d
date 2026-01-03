//! Phase 2: Natural stratification tests
//!
//! These tests verify that after removing artificial stratification forces,
//! natural stratification still emerges from Ferguson-Church settling velocity
//! differences between materials of different densities.

use sim::particle::ParticleMaterial;
use sim::FlipSimulation;
use glam::Vec2;

/// Helper to find a particle by material type
fn find_particle_by_material(sim: &FlipSimulation, mat: ParticleMaterial) -> Option<usize> {
    sim.particles
        .list
        .iter()
        .position(|p| p.material == mat)
}

/// Test that settling velocities are correctly ordered by material density.
/// Gold (density 19.3) should settle faster than sand (2.65), which should
/// settle faster than mud (1.8).
#[test]
fn test_settling_velocity_ordering() {
    const WIDTH: usize = 20;
    const HEIGHT: usize = 50;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Set up solid floor for particles to land on
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    sim.grid.compute_sdf();

    // Fill with water (from row 5 to 45)
    for y in 5..45 {
        for x in 5..15 {
            sim.spawn_water(x as f32 * CELL_SIZE + 2.0, y as f32 * CELL_SIZE + 2.0, 0.0, 0.0, 1);
        }
    }

    // Let water settle
    for _ in 0..60 {
        sim.update(DT);
    }

    // Drop gold, sand, mud from same height (y=10 cells = 40.0 world units)
    let spawn_y = 10.0 * CELL_SIZE;
    sim.spawn_gold(8.0 * CELL_SIZE, spawn_y, 0.0, 0.0, 1);
    sim.spawn_sand(10.0 * CELL_SIZE, spawn_y, 0.0, 0.0, 1);

    // Also add a mud particle by directly pushing to particle list
    sim.particles.list.push(sim::particle::Particle::mud(
        Vec2::new(12.0 * CELL_SIZE, spawn_y),
        Vec2::ZERO,
    ));

    // Floor threshold - particles need to reach this Y position
    let floor_threshold = (HEIGHT as f32 - 5.0) * CELL_SIZE;

    println!("Spawn Y: {}, Floor threshold: {}", spawn_y, floor_threshold);

    // Run until particles settle (300 frames = 5 seconds)
    let mut gold_floor_frame = 0;
    let mut sand_floor_frame = 0;
    let mut mud_floor_frame = 0;

    for frame in 1..=300 {
        sim.update(DT);

        // Find particles by material each frame (indices may change)
        if gold_floor_frame == 0 {
            if let Some(idx) = find_particle_by_material(&sim, ParticleMaterial::Gold) {
                let p = &sim.particles.list[idx];
                if p.position.y > floor_threshold {
                    gold_floor_frame = frame;
                    println!("Gold reached floor at frame {}, y={}", frame, p.position.y);
                }
            }
        }
        if sand_floor_frame == 0 {
            if let Some(idx) = find_particle_by_material(&sim, ParticleMaterial::Sand) {
                let p = &sim.particles.list[idx];
                if p.position.y > floor_threshold {
                    sand_floor_frame = frame;
                    println!("Sand reached floor at frame {}, y={}", frame, p.position.y);
                }
            }
        }
        if mud_floor_frame == 0 {
            if let Some(idx) = find_particle_by_material(&sim, ParticleMaterial::Mud) {
                let p = &sim.particles.list[idx];
                if p.position.y > floor_threshold {
                    mud_floor_frame = frame;
                    println!("Mud reached floor at frame {}, y={}", frame, p.position.y);
                }
            }
        }
    }

    // Print final positions for debugging
    if let Some(idx) = find_particle_by_material(&sim, ParticleMaterial::Gold) {
        println!("Gold final y: {}", sim.particles.list[idx].position.y);
    }
    if let Some(idx) = find_particle_by_material(&sim, ParticleMaterial::Sand) {
        println!("Sand final y: {}", sim.particles.list[idx].position.y);
    }
    if let Some(idx) = find_particle_by_material(&sim, ParticleMaterial::Mud) {
        println!("Mud final y: {}", sim.particles.list[idx].position.y);
    }

    println!("Gold floor frame: {}", gold_floor_frame);
    println!("Sand floor frame: {}", sand_floor_frame);
    println!("Mud floor frame: {}", mud_floor_frame);

    // Gold should reach floor first (highest density, fastest settling)
    assert!(
        gold_floor_frame > 0,
        "Gold should reach floor within 300 frames"
    );
    assert!(
        sand_floor_frame > 0,
        "Sand should reach floor within 300 frames"
    );
    assert!(
        gold_floor_frame < sand_floor_frame,
        "Gold (frame {}) should settle before sand (frame {})",
        gold_floor_frame,
        sand_floor_frame
    );

    // Mud may take longer or may not settle in time (it's very light)
    if mud_floor_frame > 0 {
        assert!(
            sand_floor_frame < mud_floor_frame,
            "Sand (frame {}) should settle before mud (frame {})",
            sand_floor_frame,
            mud_floor_frame
        );
    }
}

/// Test that sand does not experience artificial buoyancy.
/// Sand should always sink or stay put, never rise, when placed at rest in water.
#[test]
fn test_no_artificial_buoyancy() {
    const WIDTH: usize = 20;
    const HEIGHT: usize = 30;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Set up solid floor
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    sim.grid.compute_sdf();

    // Fill with water
    for y in 5..25 {
        for x in 5..15 {
            sim.spawn_water(
                x as f32 * CELL_SIZE + 2.0,
                y as f32 * CELL_SIZE + 2.0,
                0.0,
                0.0,
                1,
            );
        }
    }

    // Let water settle
    for _ in 0..60 {
        sim.update(DT);
    }

    // Place sand particle in middle of water column (at rest)
    let spawn_y = 15.0 * CELL_SIZE; // Middle of water column
    sim.spawn_sand(10.0 * CELL_SIZE, spawn_y, 0.0, 0.0, 1);

    let sand_idx = sim
        .particles
        .list
        .iter()
        .position(|p| p.material == ParticleMaterial::Sand)
        .expect("Sand particle not found");

    let initial_y = sim.particles.list[sand_idx].position.y;

    // Track minimum Y position (highest point reached - lower Y values mean higher)
    let mut min_y = initial_y;

    // Run 100 frames
    for _ in 0..100 {
        sim.update(DT);

        if let Some(p) = sim.particles.list.get(sand_idx) {
            if p.position.y < min_y {
                min_y = p.position.y;
            }
        }
    }

    let final_y = sim
        .particles
        .list
        .get(sand_idx)
        .map(|p| p.position.y)
        .unwrap_or(initial_y);

    println!("Sand position: {} -> {} (min: {})", initial_y, final_y, min_y);

    // Sand should never rise significantly above its initial position
    // Allow a small tolerance for pressure fluctuations (2 pixels)
    let tolerance = 2.0;
    assert!(
        min_y >= initial_y - tolerance,
        "Sand should not rise significantly! Initial: {}, Min reached: {}",
        initial_y,
        min_y
    );

    // Sand should have sunk (y increased in y-down coordinates)
    assert!(
        final_y >= initial_y,
        "Sand should sink or stay put, not rise! {} -> {}",
        initial_y,
        final_y
    );
}
