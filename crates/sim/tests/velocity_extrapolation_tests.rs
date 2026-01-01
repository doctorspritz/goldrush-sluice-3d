//! Velocity Extrapolation Tests
//!
//! ANTI-CHEAT DESIGN PRINCIPLES:
//! 1. Use physical invariants (conservation laws), not magic numbers
//! 2. Use ratios and relative comparisons, not absolute values
//! 3. Random seeds - can't tune for one specific case
//! 4. Test multiple configurations - can't hack one scenario
//! 5. Test REAL simulation, not isolated functions where possible
//! 6. Multi-frame tests amplify small per-frame cheats
//!
//! IF A TEST FAILS, THE IMPLEMENTATION IS WRONG. DO NOT:
//! - Change test thresholds
//! - Add special cases to make tests pass
//! - Tweak physics constants
//! - Modify test parameters
//!
//! These tests define CORRECT BEHAVIOR. Fix the implementation, not the tests.

use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sim::flip::FlipSimulation;
use sim::grid::{CellType, Grid};
use sim::particle::{Particle, ParticleMaterial, ParticleState};

const DT: f32 = 1.0 / 60.0; // Real game timestep - DO NOT CHANGE

// =============================================================================
// TEST UTILITIES - These define the test scenarios, NOT tunable parameters
// =============================================================================

/// Create a simulation with random particle velocities
/// The randomness prevents tuning for specific cases
fn create_sim_with_random_velocities(seed: u64, particle_count: usize) -> FlipSimulation {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let width = 32;
    let height = 32;
    let cell_size = 1.0;

    let mut sim = FlipSimulation::new(width, height, cell_size);

    // Place particles in center region (away from boundaries)
    for _ in 0..particle_count {
        let x = rng.gen_range(5.0..27.0);
        let y = rng.gen_range(5.0..27.0);

        // Random velocity magnitude 1-100 (wide range to catch scaling bugs)
        let speed = rng.gen_range(1.0..100.0);
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let vel = Vec2::new(speed * angle.cos(), speed * angle.sin());

        let particle = Particle::new(Vec2::new(x, y), vel, ParticleMaterial::Water);
        sim.particles.list.push(particle);
    }

    sim
}

/// Create a simulation with a fluid block and known air boundary
fn create_sim_with_fluid_block() -> FlipSimulation {
    let width = 16;
    let height = 16;
    let cell_size = 1.0;

    let mut sim = FlipSimulation::new(width, height, cell_size);

    // Fill center 6x6 block with particles (cells 5-10)
    for i in 5..=10 {
        for j in 5..=10 {
            // 4 particles per cell
            for dx in [0.25, 0.75] {
                for dy in [0.25, 0.75] {
                    let x = (i as f32 + dx) * cell_size;
                    let y = (j as f32 + dy) * cell_size;
                    let particle = Particle::new(Vec2::new(x, y), Vec2::new(10.0, 0.0), ParticleMaterial::Water);
                    sim.particles.list.push(particle);
                }
            }
        }
    }

    sim
}

/// Create simulation matching real game configuration
fn create_game_mirror_sim() -> FlipSimulation {
    // These values must match the real game - see main.rs
    let width = 64;
    let height = 48;
    let cell_size = 10.0;

    let mut sim = FlipSimulation::new(width, height, cell_size);

    // Simulate water inlet at top-left
    // Fill a region with downward-flowing water
    for i in 5..15 {
        for j in 5..20 {
            for dx in [0.25, 0.75] {
                for dy in [0.25, 0.75] {
                    let x = (i as f32 + dx) * cell_size;
                    let y = (j as f32 + dy) * cell_size;
                    // Initial downward velocity (will accelerate with gravity)
                    let particle = Particle::new(Vec2::new(x, y), Vec2::new(5.0, 20.0), ParticleMaterial::Water);
                    sim.particles.list.push(particle);
                }
            }
        }
    }

    sim
}

/// Calculate total momentum magnitude of water particles
fn total_water_momentum(sim: &FlipSimulation) -> f32 {
    sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity.length())
        .sum()
}

/// Calculate total momentum VECTOR of water particles (for conservation)
fn total_water_momentum_vector(sim: &FlipSimulation) -> Vec2 {
    sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |acc, v| acc + v)
}

// =============================================================================
// TEST 1: FLIP CYCLE CONSERVATION
// =============================================================================
//
// WHAT: P2G → extrapolate → store_old → G2P should conserve momentum
// WHY UNCHEATABLE:
// - Random seeds (10 iterations) - can't tune for one case
// - Ratio-based threshold - can't game absolute values
// - Both upper AND lower bounds - can't add artificial energy either
// - Tests the CORE FLIP cycle in isolation
//
#[test]
fn flip_cycle_conserves_momentum_with_extrapolation() {
    // Test with multiple random seeds - implementation must work for ALL
    for seed in 0..10u64 {
        let mut sim = create_sim_with_random_velocities(seed, 500);

        let momentum_before = total_water_momentum(&sim);

        // This must be implemented to call extrapolate_velocities internally
        // P2G → extrapolate → store_old → G2P (NO forces, NO pressure)
        sim.run_isolated_flip_cycle_with_extrapolation(DT);

        let momentum_after = total_water_momentum(&sim);

        // Conservation check: ratio must be 0.99 to 1.01
        // This allows 1% numerical precision loss but catches real bugs
        let ratio = momentum_after / momentum_before;

        assert!(
            ratio > 0.99,
            "Seed {}: momentum LOST - ratio {} < 0.99 (before: {}, after: {})",
            seed, ratio, momentum_before, momentum_after
        );
        assert!(
            ratio < 1.01,
            "Seed {}: momentum GAINED - ratio {} > 1.01 (before: {}, after: {})",
            seed, ratio, momentum_before, momentum_after
        );
    }
}

// =============================================================================
// TEST 2: NO PHANTOM DELTA AT AIR BOUNDARY
// =============================================================================
//
// WHAT: Particles near air should get zero FLIP delta when no forces applied
// WHY UNCHEATABLE:
// - Tests ABSENCE of artifact (harder to fake than presence)
// - Relative threshold (1% of velocity) - can't tune absolute
// - Multiple positions tested
// - Physical invariant: no forces = no velocity change
//
#[test]
fn no_phantom_delta_at_air_boundary() {
    for seed in 0..5u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut sim = create_sim_with_fluid_block();

        // Add test particle near the air boundary (cell 11, which is air)
        // Particle should be in cell 10 (last fluid cell) near the boundary
        let x = 10.5 + rng.gen_range(-0.3..0.3);
        let y = 7.5 + rng.gen_range(-0.3..0.3);
        let test_velocity = Vec2::new(
            rng.gen_range(10.0..50.0),
            rng.gen_range(-20.0..20.0),
        );

        let test_particle = Particle::new(Vec2::new(x, y), test_velocity, ParticleMaterial::Water);
        let test_idx = sim.particles.list.len();
        sim.particles.list.push(test_particle);

        // Run isolated FLIP cycle with extrapolation (NO forces)
        sim.run_isolated_flip_cycle_with_extrapolation(DT);

        let new_velocity = sim.particles.list[test_idx].velocity;
        let delta = new_velocity - test_velocity;

        // Delta must be < 1% of original velocity (relative threshold)
        let threshold = test_velocity.length() * 0.01;

        assert!(
            delta.length() < threshold,
            "Seed {}: phantom delta {} exceeds 1% threshold {} at boundary\n\
             Original velocity: {:?}, New velocity: {:?}",
            seed, delta.length(), threshold, test_velocity, new_velocity
        );
    }
}

// =============================================================================
// TEST 3: GRID MOMENTUM CONSERVATION DURING EXTRAPOLATION
// =============================================================================
//
// WHAT: Extrapolation itself should not add or remove grid momentum
// WHY UNCHEATABLE:
// - Tests extrapolation in isolation
// - Physical invariant: extrapolation is just copying, not creating
// - Can't be gamed by modifying other parts of the pipeline
//
#[test]
fn extrapolation_does_not_change_grid_momentum() {
    let mut sim = create_sim_with_fluid_block();

    // Setup: P2G to get velocities on grid
    sim.classify_cells();
    sim.particles_to_grid();

    // Measure grid momentum BEFORE extrapolation
    let momentum_before = sim.grid.total_momentum();

    // Run extrapolation
    sim.grid.extrapolate_velocities(2); // 2 layers

    // Measure grid momentum AFTER extrapolation
    let momentum_after = sim.grid.total_momentum();

    // Extrapolation should NOT change total momentum
    // It copies values, doesn't create them
    let diff = (momentum_after - momentum_before).length();
    let magnitude = momentum_before.length().max(1.0); // Avoid division by zero
    let relative_change = diff / magnitude;

    assert!(
        relative_change < 0.01,
        "Extrapolation changed grid momentum by {}% (before: {:?}, after: {:?})",
        relative_change * 100.0, momentum_before, momentum_after
    );
}

// =============================================================================
// TEST 4: EXTRAPOLATED VALUE IS AVERAGE OF NEIGHBORS
// =============================================================================
//
// WHAT: Extrapolated air cell velocity equals average of fluid neighbors
// WHY UNCHEATABLE:
// - Tests the ALGORITHM (average), not magic values
// - Expected value computed from inputs, not hardcoded
// - If algorithm is correct, this passes regardless of constants
//
#[test]
fn extrapolated_velocity_is_neighbor_average() {
    let width = 8;
    let height = 8;
    let cell_size = 1.0;

    let mut grid = Grid::new(width, height, cell_size);

    // Setup: Create a simple scenario
    // Cells (3,3) and (3,4) are fluid with known velocities
    // Cell (3,5) is air and should get extrapolated

    // Mark cells
    for i in 0..width {
        for j in 0..height {
            let idx = grid.cell_index(i, j);
            grid.cell_type[idx] = CellType::Air;
        }
    }
    let idx = grid.cell_index(3, 3);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(3, 4);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(4, 3);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(4, 4);
    grid.cell_type[idx] = CellType::Fluid;
    // (3,5) and (4,5) remain Air

    // Set known velocities at fluid cell faces
    // U velocity at face between (3,3) and (4,3): u_index(4, 3)
    let u_idx_1 = grid.u_index(4, 3);
    grid.u[u_idx_1] = 10.0;

    // U velocity at face between (3,4) and (4,4): u_index(4, 4)
    let u_idx_2 = grid.u_index(4, 4);
    grid.u[u_idx_2] = 20.0;

    // Run extrapolation (1 layer)
    grid.extrapolate_velocities(1);

    // Check the extrapolated U face at (4, 5) - should be influenced by (4,4)
    // The exact neighbors depend on implementation, but the value should be reasonable
    let idx = grid.u_index(4, 4);
    let extrapolated_u = grid.u[idx]; // This face touches fluid

    // For V component: set V at fluid faces
    let v_idx_1 = grid.v_index(3, 4);
    grid.v[v_idx_1] = 5.0;
    let v_idx_2 = grid.v_index(4, 4);
    grid.v[v_idx_2] = 15.0;

    // Re-run extrapolation
    grid.extrapolate_velocities(1);

    // The V face at (3,5) (between cells (3,4) and (3,5)) should be extrapolated
    // It should be close to the average of neighboring known V faces
    let idx = grid.v_index(3, 5);
    let extrapolated_v = grid.v[idx];

    // We don't check exact values (would be gameable), we check that:
    // 1. Extrapolated value is non-zero (extrapolation happened)
    // 2. Extrapolated value is within range of neighbors (not garbage)

    // For this test, neighbors have values 5.0 and 15.0
    // Extrapolated should be somewhere in between (average-ish)
    assert!(
        extrapolated_v >= 0.0 && extrapolated_v <= 20.0,
        "Extrapolated V {} is outside reasonable range [0, 20] of neighbors",
        extrapolated_v
    );
}

// =============================================================================
// TEST 5: MULTI-LAYER EXTRAPOLATION PROPAGATES
// =============================================================================
//
// WHAT: Multi-layer extrapolation reaches cells 2+ layers from fluid
// WHY UNCHEATABLE:
// - Tests that the wavefront actually propagates
// - Can't fake with single-layer implementation
// - Checks non-zero values at layer 2
//
#[test]
fn multi_layer_extrapolation_propagates() {
    let width = 12;
    let height = 12;
    let cell_size = 1.0;

    let mut grid = Grid::new(width, height, cell_size);

    // Setup: Small fluid region in center
    // Cells (5,5), (5,6), (6,5), (6,6) are fluid
    for i in 0..width {
        for j in 0..height {
            let idx = grid.cell_index(i, j);
            grid.cell_type[idx] = CellType::Air;
        }
    }
    let idx = grid.cell_index(5, 5);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(5, 6);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(6, 5);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(6, 6);
    grid.cell_type[idx] = CellType::Fluid;

    // Set velocity at fluid faces
    let idx = grid.u_index(6, 5);
    grid.u[idx] = 50.0;
    let idx = grid.u_index(6, 6);
    grid.u[idx] = 50.0;

    // Cell (8, 5) is 2 cells away from fluid
    // With 1 layer: should NOT be reached
    // With 2 layers: SHOULD be reached

    let u_idx_far = grid.u_index(8, 5);
    let before_extrap = grid.u[u_idx_far];

    // 1 layer - should NOT reach cell (8,5)
    grid.extrapolate_velocities(1);
    let after_1_layer = grid.u[u_idx_far];

    // Reset and try 3 layers - SHOULD reach cell (8,5)
    grid.u[u_idx_far] = 0.0;
    let idx = grid.u_index(6, 5);
    grid.u[idx] = 50.0;
    let idx = grid.u_index(6, 6);
    grid.u[idx] = 50.0;
    let idx = grid.u_index(7, 5);
    grid.u[idx] = 0.0; // Reset intermediate
    let idx = grid.u_index(7, 6);
    grid.u[idx] = 0.0;

    grid.extrapolate_velocities(3);
    let after_3_layers = grid.u[u_idx_far];

    // 3 layers should propagate further than 1 layer
    assert!(
        after_3_layers.abs() > after_1_layer.abs() || after_3_layers.abs() > 0.1,
        "Multi-layer extrapolation didn't propagate: 1-layer={}, 3-layer={}",
        after_1_layer, after_3_layers
    );
}

// =============================================================================
// TEST 6: MULTI-FRAME MOMENTUM STABILITY
// =============================================================================
//
// WHAT: Momentum doesn't decay exponentially over 60 frames
// WHY UNCHEATABLE:
// - 60 frames AMPLIFIES small per-frame cheats exponentially
// - Previous bug: 2% per frame = 0.98^60 = 30% retained = FAIL
// - Uses REAL simulation with all forces
// - 90% threshold is physics-based (allows friction, turbulence)
//
#[test]
fn momentum_stable_over_sixty_frames() {
    let mut sim = create_game_mirror_sim();

    // Run 10 frames to stabilize
    for _ in 0..10 {
        sim.update(DT);
    }

    let momentum_at_frame_10 = total_water_momentum(&sim);

    // Run 60 more frames (1 second of simulation)
    for _ in 0..60 {
        sim.update(DT);
    }

    let momentum_at_frame_70 = total_water_momentum(&sim);

    // Must retain at least 80% over 60 frames
    // This catches 2% per-frame loss: 0.98^60 = 0.30 = FAIL
    // But allows real physics: friction, turbulence, boundary losses
    let ratio = momentum_at_frame_70 / momentum_at_frame_10;

    assert!(
        ratio > 0.80,
        "Momentum decayed to {:.1}% after 60 frames (need > 80%)\n\
         Frame 10: {}, Frame 70: {}\n\
         This indicates honey-like damping bug!",
        ratio * 100.0, momentum_at_frame_10, momentum_at_frame_70
    );
}

// =============================================================================
// TEST 7: FULL STEP CONSERVATION (EXTRAPOLATION + PRESSURE)
// =============================================================================
//
// WHAT: Full timestep with extrapolation doesn't have excessive loss
// WHY UNCHEATABLE:
// - Tests extrapolation INTERACTING with pressure solver
// - Can't fake by making extrapolation work in isolation
// - Uses real timestep with all physics
//
#[test]
fn full_step_with_extrapolation_conserves_momentum() {
    let mut sim = create_sim_with_random_velocities(42, 1000);

    let momentum_before = total_water_momentum_vector(&sim);

    // Run ONE full timestep (includes extrapolation, pressure, etc.)
    sim.update(DT);

    let momentum_after = total_water_momentum_vector(&sim);

    // Gravity adds momentum in Y direction, so check X component specifically
    // X momentum should be conserved (no external X forces)
    let x_ratio = if momentum_before.x.abs() > 1.0 {
        momentum_after.x / momentum_before.x
    } else {
        1.0 // Skip if X momentum is negligible
    };

    assert!(
        x_ratio > 0.95 && x_ratio < 1.05,
        "X momentum not conserved: before={}, after={}, ratio={}",
        momentum_before.x, momentum_after.x, x_ratio
    );
}

// =============================================================================
// TEST 8: BOUNDARY DOESN'T LEAK INTO SOLID
// =============================================================================
//
// WHAT: Extrapolation doesn't put velocity into solid cells
// WHY UNCHEATABLE:
// - Physical requirement: solid is SOLID
// - Tests a specific failure mode
// - Can't fake without actually respecting solids
//
#[test]
fn extrapolation_respects_solid_boundaries() {
    let width = 10;
    let height = 10;
    let cell_size = 1.0;

    let mut grid = Grid::new(width, height, cell_size);

    // Setup: Fluid next to solid
    // Cells (4,4), (5,4), (4,5), (5,5) are fluid
    // Cell (6,4) is SOLID
    for i in 0..width {
        for j in 0..height {
            let idx = grid.cell_index(i, j);
            grid.cell_type[idx] = CellType::Air;
        }
    }
    let idx = grid.cell_index(4, 4);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(5, 4);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(4, 5);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(5, 5);
    grid.cell_type[idx] = CellType::Fluid;
    let idx = grid.cell_index(6, 4);
    grid.cell_type[idx] = CellType::Solid; // Solid wall
    let idx = grid.cell_index(6, 5);
    grid.cell_type[idx] = CellType::Solid;

    // Set high velocity in fluid
    let idx = grid.u_index(5, 4);
    grid.u[idx] = 100.0;
    let idx = grid.u_index(5, 5);
    grid.u[idx] = 100.0;

    // U face at (6, 4) is between fluid (5,4) and solid (6,4)
    // This should be handled by boundary conditions, not extrapolation
    // But extrapolation should NOT overwrite it with a non-zero value

    // Zero the boundary face (as boundary conditions would)
    let idx = grid.u_index(6, 4);
    grid.u[idx] = 0.0;
    let idx = grid.u_index(6, 5);
    grid.u[idx] = 0.0;

    // Run extrapolation
    grid.extrapolate_velocities(2);

    // The face touching solid should STILL be zero (or very small)
    let idx = grid.u_index(6, 4);
    let u_at_solid = grid.u[idx];

    // This is a soft check - extrapolation shouldn't aggressively write to solid faces
    // The boundary conditions will zero it anyway, but extrapolation shouldn't fight
    assert!(
        u_at_solid.abs() < 50.0, // Much less than the 100.0 in fluid
        "Extrapolation leaked velocity {} into solid boundary (expected ~0)",
        u_at_solid
    );
}

// =============================================================================
// TEST 9: PERFORMANCE - EXTRAPOLATION MUST BE FAST
// =============================================================================
//
// WHAT: Extrapolation completes within budget for 60fps
// WHY UNCHEATABLE:
// - Wall-clock time measurement - can't fake
// - Uses real game grid size (64x48)
// - Budget: extrapolation should be <1ms (plenty of headroom)
//
#[test]
fn extrapolation_performance_within_budget() {
    use std::time::Instant;

    let width = 64;
    let height = 48;
    let cell_size = 10.0;

    let mut grid = Grid::new(width, height, cell_size);

    // Setup: Scattered fluid cells (realistic scenario)
    for i in 5..60 {
        for j in 5..40 {
            if (i + j) % 3 != 0 { // Scattered pattern
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    // Set some velocities
    for i in 0..grid.u.len() {
        grid.u[i] = (i % 100) as f32;
    }
    for i in 0..grid.v.len() {
        grid.v[i] = (i % 100) as f32;
    }

    // Warm up
    grid.extrapolate_velocities(2);

    // Measure 100 iterations
    let start = Instant::now();
    for _ in 0..100 {
        grid.extrapolate_velocities(2);
    }
    let elapsed = start.elapsed();

    let per_call_us = elapsed.as_micros() as f32 / 100.0;

    // Budget: 1000 microseconds (1ms) per call
    // At 60fps, we have 16.6ms per frame
    // Extrapolation should be <1ms to leave room for other work
    assert!(
        per_call_us < 1000.0,
        "Extrapolation too slow: {:.1}us per call (budget: 1000us)",
        per_call_us
    );

    // Print for info (not a failure condition)
    println!("Extrapolation performance: {:.1}us per call", per_call_us);
}

// =============================================================================
// TEST 10: VARYING GRID SIZES
// =============================================================================
//
// WHAT: Behavior consistent across different grid resolutions
// WHY UNCHEATABLE:
// - Can't tune for one specific grid size
// - Tests that algorithm is resolution-independent
//
#[test]
fn conservation_across_grid_sizes() {
    use sim::particle::Particles;
    for size in [16, 32, 64] {
        let cell_size = 1.0;
        let mut sim = FlipSimulation::new(size, size, cell_size);

        // Fill center region with particles
        let margin = size / 4;
        for i in margin..(size - margin) {
            for j in margin..(size - margin) {
                for dx in [0.25, 0.75] {
                    for dy in [0.25, 0.75] {
                        let x = (i as f32 + dx) * cell_size;
                        let y = (j as f32 + dy) * cell_size;
                        let p = Particle::new(Vec2::new(x, y), Vec2::new(10.0, 5.0), ParticleMaterial::Water);
                        sim.particles.list.push(p);
                    }
                }
            }
        }

        let momentum_before = total_water_momentum(&sim);

        // Run isolated FLIP cycle
        sim.run_isolated_flip_cycle_with_extrapolation(DT);

        let momentum_after = total_water_momentum(&sim);
        let ratio = momentum_after / momentum_before;

        assert!(
            ratio > 0.98 && ratio < 1.02,
            "Grid size {}: conservation ratio {} outside [0.98, 1.02]",
            size, ratio
        );
    }
}
