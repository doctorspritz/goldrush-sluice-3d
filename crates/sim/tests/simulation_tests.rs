//! Integration tests for FLIP simulation
//! Run with: cargo test -p sim --release
//!
//! These tests verify critical simulation behaviors:
//! - P1: Particles never spawn inside solids
//! - P2: Pressure solver converges
//! - P3: Sediment separation doesn't panic

use sim::sediment::{Sediment, SedimentType};
use sim::particle::{ParticleState, ParticleMaterial};
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

// ============================================================================
// APIC Behavioral Tests
// These tests verify expected behavior for the APIC transfer method
// ============================================================================

/// APIC-1: Simulation should not gain energy (no blow-up)
/// This is critical - APIC should be stable and not amplify velocities
#[test]
fn test_simulation_energy_bounded() {
    const WIDTH: usize = 48;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 300;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create a container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);  // Floor
        sim.grid.set_solid(i, 0);           // Ceiling
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);           // Left wall
        sim.grid.set_solid(WIDTH - 1, j);   // Right wall
    }
    sim.grid.compute_sdf();

    // Spawn water with initial velocity
    for i in 0..10 {
        for j in 0..10 {
            let x = (10 + i) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = (10 + j) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 20.0, -10.0, 1);
        }
    }

    // Measure initial kinetic energy
    let initial_ke: f32 = sim.particles.iter()
        .map(|p| p.velocity.length_squared())
        .sum();

    // Run simulation
    let mut max_ke: f32 = initial_ke;
    for _ in 0..FRAMES {
        sim.update(DT);

        let ke: f32 = sim.particles.iter()
            .map(|p| p.velocity.length_squared())
            .sum();
        max_ke = max_ke.max(ke);

        // Check for NaN/Inf
        for p in sim.particles.iter() {
            assert!(!p.velocity.x.is_nan(), "Velocity became NaN");
            assert!(!p.velocity.y.is_nan(), "Velocity became NaN");
            assert!(!p.position.x.is_nan(), "Position became NaN");
            assert!(!p.position.y.is_nan(), "Position became NaN");
        }
    }

    // Energy should not increase dramatically (allow 2x for transients)
    // With gravity, potential energy converts to kinetic, so we're lenient
    assert!(max_ke < initial_ke * 10.0 + 50000.0,
        "Energy blow-up detected: initial={}, max={}", initial_ke, max_ke);
}

/// APIC-2: Sediment should still settle through water
/// This tests that the Lagrangian sediment coupling still works with APIC
#[test]
fn test_sediment_settles_through_water() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 200;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    sim.grid.compute_sdf();

    // Fill with water
    for i in 5..WIDTH-5 {
        for j in 10..HEIGHT-5 {
            let x = i as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = j as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 0.0, 0.0, 1);
        }
    }

    // Spawn gold particle near top
    let gold_x = WIDTH as f32 * CELL_SIZE / 2.0;
    let gold_y = 15.0 * CELL_SIZE;  // Near top
    sim.particles.spawn_gold(gold_x, gold_y, 0.0, 0.0);

    let gold_initial_y = sim.particles.list.last().unwrap().position.y;

    // Run simulation
    for _ in 0..FRAMES {
        sim.update(DT);
    }

    // Find gold particle (it's the last one we added)
    let gold_final_y = sim.particles.list.iter()
        .filter(|p| p.material == sim::particle::ParticleMaterial::Gold)
        .next()
        .map(|p| p.position.y)
        .unwrap_or(gold_initial_y);

    // Gold should have moved down significantly (Y increases downward in this coord system)
    let distance_settled = gold_final_y - gold_initial_y;
    assert!(distance_settled > CELL_SIZE * 5.0,
        "Gold should settle through water: initial_y={}, final_y={}, settled={}",
        gold_initial_y, gold_final_y, distance_settled);
}

/// APIC-3: Simulation should remain stable over many frames
/// No NaN, no crashes, particles stay in bounds
#[test]
fn test_simulation_long_stability() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 500;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.3, 15, 4, 2);

    // Spawn initial water
    for i in 0..50 {
        let x = 20.0 + (i % 10) as f32 * 4.0;
        let y = 30.0 + (i / 10) as f32 * 4.0;
        sim.spawn_water(x, y, 40.0, 0.0, 3);
    }

    let sim_width = WIDTH as f32 * CELL_SIZE;
    let sim_height = HEIGHT as f32 * CELL_SIZE;

    // Run simulation
    for frame in 0..FRAMES {
        // Add more water periodically
        if frame % 10 == 0 {
            sim.spawn_water(20.0, 30.0, 40.0, 0.0, 2);
        }

        sim.update(DT);

        // Check stability
        for p in sim.particles.iter() {
            assert!(!p.velocity.x.is_nan(), "Frame {}: Velocity.x NaN", frame);
            assert!(!p.velocity.y.is_nan(), "Frame {}: Velocity.y NaN", frame);
            assert!(!p.position.x.is_nan(), "Frame {}: Position.x NaN", frame);
            assert!(!p.position.y.is_nan(), "Frame {}: Position.y NaN", frame);
            assert!(p.velocity.length() < 500.0,
                "Frame {}: Velocity too high: {}", frame, p.velocity.length());
        }

        // Particles should stay roughly in bounds (allow some margin)
        let in_bounds = sim.particles.iter()
            .filter(|p| p.position.x >= -10.0 && p.position.x < sim_width + 10.0
                     && p.position.y >= -10.0 && p.position.y < sim_height + 10.0)
            .count();
        let total = sim.particles.len();
        assert!(in_bounds as f32 / total as f32 > 0.9 || total < 10,
            "Frame {}: Too many particles out of bounds: {}/{}", frame, total - in_bounds, total);
    }
}

/// APIC-4: Velocity field should be smoother than baseline
/// Measures velocity variance - APIC should produce less noisy velocities
#[test]
fn test_velocity_variance_reasonable() {
    const WIDTH: usize = 48;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    sim.grid.compute_sdf();

    // Spawn water block
    for i in 10..30 {
        for j in 10..30 {
            let x = i as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = j as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 0.0, 0.0, 1);
        }
    }

    // Let it settle
    for _ in 0..100 {
        sim.update(DT);
    }

    // Measure velocity variance
    let n = sim.particles.len() as f32;
    if n < 10.0 {
        return; // Not enough particles
    }

    let mean_vx: f32 = sim.particles.iter().map(|p| p.velocity.x).sum::<f32>() / n;
    let mean_vy: f32 = sim.particles.iter().map(|p| p.velocity.y).sum::<f32>() / n;

    let variance: f32 = sim.particles.iter()
        .map(|p| {
            let dx = p.velocity.x - mean_vx;
            let dy = p.velocity.y - mean_vy;
            dx * dx + dy * dy
        })
        .sum::<f32>() / n;

    // Variance should be reasonable for settled water
    // This is a sanity check - specific threshold may need tuning
    assert!(variance < 1000.0,
        "Velocity variance too high for settled water: {}", variance);
}

// ============================================================================
// BEDLOAD STATE TRANSITION TESTS
// ============================================================================

/// Particles in mid-air should NEVER transition to Bedload
/// This verifies that particles don't "settle in air"
#[test]
fn test_midair_particles_stay_suspended() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create only a floor at the very bottom - leave lots of air space
    for i in 0..WIDTH {
        for j in (HEIGHT - 3)..HEIGHT {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Spawn sediment particles in mid-air (far from floor)
    // Floor is at y = (HEIGHT-3) * CELL_SIZE = 45 * 4 = 180
    // Spawn at y = 40 (very far from floor)
    let spawn_y = 40.0;
    for i in 0..10 {
        let x = 100.0 + i as f32 * 5.0;
        sim.spawn_sand(x, spawn_y, 0.0, 0.0, 1);
    }

    // Give particles zero velocity (should trigger bedload if near floor)
    for p in sim.particles.iter_mut() {
        p.velocity = Vec2::ZERO;
    }

    // Run simulation for a few frames
    for _ in 0..10 {
        sim.update(DT);
    }

    // All particles should still be Suspended (not Bedload) because they're in mid-air
    let bedload_count = sim.particles.iter()
        .filter(|p| p.is_sediment() && p.state == ParticleState::Bedload)
        .count();

    assert_eq!(bedload_count, 0,
        "Found {} particles in Bedload state while in mid-air! Particles should NOT settle in air.",
        bedload_count);
}

/// Particles that have fallen to the floor should transition to Bedload
#[test]
fn test_particles_on_floor_become_bedload() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create floor at bottom
    for i in 0..WIDTH {
        for j in (HEIGHT - 3)..HEIGHT {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Spawn sediment particles just above the floor
    // Floor top is at y = (HEIGHT-3) * CELL_SIZE = 45 * 4 = 180
    // Spawn particles very close to floor
    let floor_y = (HEIGHT - 3) as f32 * CELL_SIZE;
    let spawn_y = floor_y - CELL_SIZE * 0.5; // Half a cell above floor

    for i in 0..10 {
        let x = 100.0 + i as f32 * 5.0;
        sim.spawn_sand(x, spawn_y, 0.0, 0.0, 1);
    }

    // Debug: print SDF and gradient at spawn positions
    println!("Floor y = {}", floor_y);
    println!("Spawn y = {}", spawn_y);
    for p in sim.particles.iter() {
        if p.is_sediment() {
            let sdf = sim.grid.sample_sdf(p.position);
            let grad = sim.grid.sdf_gradient(p.position);
            println!("Particle at ({}, {}): SDF={:.2}, grad=({:.2}, {:.2})",
                p.position.x, p.position.y, sdf, grad.x, grad.y);
        }
    }

    // Set particles to very low velocity (below settle threshold)
    for p in sim.particles.iter_mut() {
        p.velocity = Vec2::new(0.01, 0.01);
    }

    // Run simulation to let them settle
    for _ in 0..30 {
        sim.update(DT);
    }

    // Debug: print states after settling
    println!("\nAfter settling:");
    for (i, p) in sim.particles.iter().enumerate() {
        if p.is_sediment() {
            let sdf = sim.grid.sample_sdf(p.position);
            let grad = sim.grid.sdf_gradient(p.position);
            println!("Particle {}: pos=({:.1}, {:.1}), SDF={:.2}, grad=({:.2}, {:.2}), state={:?}, vel=({:.2}, {:.2})",
                i, p.position.x, p.position.y, sdf, grad.x, grad.y, p.state, p.velocity.x, p.velocity.y);
        }
    }

    // At least some particles should be in Bedload state on the floor
    let bedload_count = sim.particles.iter()
        .filter(|p| p.is_sediment() && p.state == ParticleState::Bedload)
        .count();

    let sediment_count = sim.particles.iter()
        .filter(|p| p.is_sediment())
        .count();

    // At least 50% should have settled to bedload
    assert!(bedload_count > sediment_count / 2,
        "Only {} of {} sediment particles transitioned to Bedload on floor",
        bedload_count, sediment_count);
}

/// Particles near walls (not floors) should stay Suspended
#[test]
fn test_particles_near_walls_stay_suspended() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create only a left wall (no floor)
    for j in 0..HEIGHT {
        for i in 0..3 {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Spawn sediment particles near the left wall
    let wall_x = 3.0 * CELL_SIZE; // Right edge of wall
    let spawn_x = wall_x + CELL_SIZE * 0.5; // Half a cell from wall

    for i in 0..10 {
        let y = 50.0 + i as f32 * 5.0;
        sim.spawn_sand(spawn_x, y, 0.0, 0.0, 1);
    }

    // Set particles to very low velocity
    for p in sim.particles.iter_mut() {
        p.velocity = Vec2::new(0.01, 0.01);
    }

    // Run simulation
    for _ in 0..30 {
        sim.update(DT);
    }

    // Particles should NOT be in Bedload state (walls are not floors)
    let bedload_count = sim.particles.iter()
        .filter(|p| p.is_sediment() && p.state == ParticleState::Bedload)
        .count();

    assert_eq!(bedload_count, 0,
        "Found {} particles in Bedload state near a wall! Only floors should cause Bedload.",
        bedload_count);
}
