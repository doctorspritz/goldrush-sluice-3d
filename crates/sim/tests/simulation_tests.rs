//! Integration tests for FLIP simulation
//! Run with: cargo test -p sim --release
//!
//! These tests verify critical simulation behaviors:
//! - P1: Particles never spawn inside solids
//! - P2: Pressure solver converges
//! - P3: Sediment separation doesn't panic

use sim::sediment::{Sediment, SedimentType};
use sim::{create_sluice, FlipSimulation};
use glam::Vec2;

/// P1: Spawn should never place particles inside solid cells
#[test]
fn test_spawn_never_in_solid() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create a complex solid pattern
    for i in 0..WIDTH {
        for j in (HEIGHT - 10)..HEIGHT {
            sim.grid.set_solid(i, j);
        }
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    // Internal obstacles
    for i in 10..20 {
        for j in 20..30 {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Try to spawn particles at positions including inside solids
    let spawn_positions = [
        (32.0, 20.0),   // Open area
        (5.0, 180.0),   // Inside floor solid
        (15.0 * CELL_SIZE, 25.0 * CELL_SIZE), // Inside obstacle
    ];

    for (x, y) in spawn_positions {
        sim.spawn_water(x, y, 0.0, 0.0, 10);
    }

    // Verify NO particles ended up inside solids
    for p in sim.particles.iter() {
        let (i, j) = sim.grid.pos_to_cell(p.position);
        assert!(!sim.grid.is_solid(i, j),
            "Particle at ({}, {}) is in solid cell ({}, {})",
            p.position.x, p.position.y, i, j);
        assert!(sim.grid.sample_sdf(p.position) >= 0.0,
            "Particle at ({}, {}) has negative SDF",
            p.position.x, p.position.y);
    }
}

/// P1b: Particles should never penetrate solids during simulation
#[test]
fn test_no_solid_penetration_during_simulation() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 100;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.2, 10, 4, 2);

    // Spawn water
    for i in 0..5 {
        let x = 30.0 + (i % 3) as f32 * 8.0;
        let y = 40.0 + (i / 3) as f32 * 8.0;
        sim.spawn_water(x, y, 30.0, 0.0, 5);
    }

    // Run simulation
    for _ in 0..FRAMES {
        sim.update(DT);

        // Check every particle every frame
        for p in sim.particles.iter() {
            let (i, j) = sim.grid.pos_to_cell(p.position);
            assert!(!sim.grid.is_solid(i, j),
                "Particle penetrated solid during simulation");
        }
    }
}

/// P2: Pressure solver should converge
#[test]
fn test_pressure_solver_convergence() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container with narrow channel
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
        sim.grid.set_solid(i, 0);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    for i in 5..WIDTH-5 {
        sim.grid.set_solid(i, HEIGHT / 2);
    }
    // Small gap
    let gap_idx = sim.grid.cell_index(WIDTH / 2, HEIGHT / 2);
    sim.grid.solid[gap_idx] = false;
    sim.grid.compute_sdf();

    // Spawn water
    for i in 0..5 {
        for j in 0..5 {
            let x = (5 + i) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = (5 + j) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 20.0, 0.0, 1);
        }
    }

    // Run simulation
    for _ in 0..60 {
        sim.update(DT);
    }

    // Check residual divergence
    sim.grid.compute_divergence();
    let mut max_div = 0.0f32;

    for j in 1..HEIGHT - 1 {
        for i in 1..WIDTH - 1 {
            let idx = sim.grid.cell_index(i, j);
            if sim.grid.cell_type[idx] == sim::grid::CellType::Fluid {
                max_div = max_div.max(sim.grid.divergence[idx].abs());
            }
        }
    }

    assert!(max_div < 0.1,
        "Pressure solver did not converge: max_div = {}", max_div);
}

/// P3: Sediment separation should not panic with overlapping particles
#[test]
fn test_sediment_separation_no_panic() {
    let mut sediment = Sediment::new();

    // Add particles at exactly the same position (worst case)
    for _ in 0..10 {
        sediment.spawn(Vec2::new(100.0, 100.0), Vec2::ZERO, SedimentType::QuartzSand, 2.0);
    }

    // Add particles very close together
    for i in 0..10 {
        sediment.spawn(
            Vec2::new(200.0 + i as f32 * 0.001, 100.0),
            Vec2::ZERO,
            SedimentType::QuartzSand,
            2.0,
        );
    }

    // This should not panic
    sediment.separate_particles(2.0);

    // Run multiple times for stability
    for _ in 0..10 {
        sediment.separate_particles(2.0);
    }
}

/// Overlapping particles should be pushed apart
#[test]
fn test_overlapping_particles_separate() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 8.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Floor
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    sim.grid.compute_sdf();

    let center = Vec2::new(WIDTH as f32 * CELL_SIZE / 2.0, HEIGHT as f32 * CELL_SIZE / 2.0 - 20.0);

    // Spawn two overlapping particles
    sim.particles.spawn_water(center.x, center.y, 0.0, 0.0);
    sim.particles.spawn_water(center.x + 0.5, center.y, 0.0, 0.0);

    let initial_dist = (sim.particles.list[0].position - sim.particles.list[1].position).length();
    assert!(initial_dist < 2.5, "Initial particles should be overlapping");

    // Run simulation
    for _ in 0..10 {
        sim.update(1.0 / 60.0);
    }

    let final_dist = (sim.particles.list[0].position - sim.particles.list[1].position).length();
    assert!(final_dist >= 2.0,
        "Particles should be pushed apart: final_dist = {}", final_dist);
}
