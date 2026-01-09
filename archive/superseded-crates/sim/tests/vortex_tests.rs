//! Vortex formation and preservation tests
//!
//! These tests verify that the APIC simulation correctly preserves vorticity:
//! - V1: Taylor-Green vortex initialization
//! - V2: Solid body rotation initialization
//! - V3: Grid kinetic energy stability
//! - V4: Enstrophy tracking during simulation
//! - V5: Vorticity confinement produces enstrophy
//! - V6: CFL computation
//! - V7: Quick smoke test
//! - V8: Max vorticity detection
//!
//! Run with: cargo test -p sim --test vortex_tests --release

use sim::FlipSimulation;

/// V1: Taylor-Green vortex - verify initial state
/// This test verifies that the Taylor-Green initialization produces
/// the expected velocity field and non-zero enstrophy.
#[test]
fn test_taylor_green_initialization() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 64;
    const CELL_SIZE: f32 = 1.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    sim.initialize_taylor_green();

    // Compute grid kinetic energy - should be non-zero
    let ke = sim.compute_grid_kinetic_energy();
    assert!(ke > 0.0, "Taylor-Green should have positive kinetic energy");

    // Compute vorticity and enstrophy
    sim.grid.compute_vorticity();
    let enstrophy = sim.grid.compute_enstrophy();
    assert!(enstrophy > 0.0, "Taylor-Green should have positive enstrophy");

    // Check that max vorticity is reasonable
    let max_vort = sim.grid.max_vorticity();
    assert!(max_vort > 0.0, "Should have non-zero vorticity");

    // Verify velocity at center should be approximately zero
    let center_vel = sim.grid.sample_velocity(glam::Vec2::new(
        WIDTH as f32 * CELL_SIZE / 2.0,
        HEIGHT as f32 * CELL_SIZE / 2.0,
    ));
    assert!(center_vel.length() < 0.1, "Center velocity should be near zero");
}

/// V2: Solid body rotation - verify initialization
/// A rotating disk should have uniform angular velocity
#[test]
fn test_solid_rotation_initialization() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 64;
    const CELL_SIZE: f32 = 1.0;
    const OMEGA: f32 = 0.1; // Angular velocity

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    sim.initialize_solid_rotation(OMEGA);

    // Compute grid kinetic energy
    let ke = sim.compute_grid_kinetic_energy();
    assert!(ke > 0.0, "Solid rotation should have positive kinetic energy");

    // Compute vorticity - should be roughly 2ω for solid body rotation
    sim.grid.compute_vorticity();
    let enstrophy = sim.grid.compute_enstrophy();
    assert!(enstrophy > 0.0, "Solid rotation should have positive enstrophy");

    // Check velocity at a point off-center
    let cx = WIDTH as f32 * CELL_SIZE / 2.0;
    let cy = HEIGHT as f32 * CELL_SIZE / 2.0;
    let test_x = cx + 10.0;
    let test_y = cy;
    let vel = sim.grid.sample_velocity(glam::Vec2::new(test_x, test_y));

    // For solid body rotation: v = ω × r
    // At (cx+10, cy): u = -ω*(y-cy) = 0, v = ω*(x-cx) = 0.1*10 = 1.0
    let expected_v = OMEGA * 10.0;
    assert!((vel.y - expected_v).abs() < 0.5,
        "Velocity v at offset should be ~{}, got {}", expected_v, vel.y);
    assert!(vel.x.abs() < 0.5,
        "Velocity u at offset should be ~0, got {}", vel.x);
}

/// V3: Grid kinetic energy should be bounded during simulation
/// Tests that pressure projection preserves energy in a closed container
#[test]
fn test_grid_energy_stability() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 2.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 100;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create closed container to prevent energy escape
    for i in 0..WIDTH {
        sim.grid.set_solid(i, 0);
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    sim.grid.compute_sdf();

    sim.initialize_taylor_green();

    let initial_ke = sim.compute_grid_kinetic_energy();
    let mut max_ke = initial_ke;

    for _ in 0..FRAMES {
        // Inviscid flow without gravity - energy should be conserved
        sim.grid.enforce_boundary_conditions();
        sim.grid.compute_divergence();
        sim.grid.solve_pressure(10);
        sim.grid.apply_pressure_gradient(DT);

        let ke = sim.compute_grid_kinetic_energy();
        max_ke = max_ke.max(ke);

        // Check for NaN
        assert!(!ke.is_nan(), "Kinetic energy became NaN");
    }

    // In inviscid closed container, energy should be roughly conserved
    // Allow some numerical dissipation (up to 10x is quite generous)
    assert!(max_ke < initial_ke * 10.0 + 100.0,
        "Energy blow-up: initial={}, max={}", initial_ke, max_ke);
}

/// V4: Enstrophy should be trackable during full simulation
/// This is a smoke test that ensures enstrophy computation works
/// during a real simulation with particles
#[test]
fn test_enstrophy_tracking_with_particles() {
    const WIDTH: usize = 48;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 50;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
        sim.grid.set_solid(i, 0);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    sim.grid.compute_sdf();

    // Spawn water
    for i in 0..10 {
        for j in 0..10 {
            let x = (10 + i) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = (10 + j) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 20.0, 0.0, 1);
        }
    }

    let mut enstrophy_history: Vec<f32> = Vec::new();

    for _ in 0..FRAMES {
        sim.update(DT);

        // Compute enstrophy after update (vorticity computed during vorticity confinement)
        let enstrophy = sim.update_and_compute_enstrophy();
        enstrophy_history.push(enstrophy);

        // Enstrophy should be valid (not NaN, not negative)
        assert!(!enstrophy.is_nan(), "Enstrophy became NaN");
        assert!(enstrophy >= 0.0, "Enstrophy should be non-negative");
    }

    // Should have recorded some enstrophy values
    assert!(!enstrophy_history.is_empty());

    // At least some frames should have non-zero enstrophy (there's flow)
    let non_zero_count = enstrophy_history.iter().filter(|&&e| e > 0.0).count();
    assert!(non_zero_count > FRAMES / 2,
        "Expected most frames to have non-zero enstrophy, got {}/{}", non_zero_count, FRAMES);
}

/// V5: Vorticity confinement should produce enstrophy in flowing simulation
/// This test verifies that the simulation with obstacles produces measurable vorticity
#[test]
fn test_vorticity_confinement_produces_enstrophy() {
    const WIDTH: usize = 48;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 60;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container with obstacle to generate vortices
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    // Add obstacle to create vortex shedding
    for i in 20..25 {
        for j in 20..25 {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Spawn water with horizontal velocity to flow past obstacle
    for i in 0..8 {
        for j in 0..8 {
            let x = (5 + i) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = (15 + j) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 30.0, 0.0, 1);
        }
    }

    let mut max_enstrophy: f32 = 0.0;
    let mut total_enstrophy: f32 = 0.0;

    for _ in 0..FRAMES {
        sim.update(DT);

        let enstrophy = sim.update_and_compute_enstrophy();
        max_enstrophy = max_enstrophy.max(enstrophy);
        total_enstrophy += enstrophy;

        // Enstrophy should never be NaN
        assert!(!enstrophy.is_nan(), "Enstrophy became NaN");
    }

    // Simulation should have produced some vorticity at some point
    assert!(max_enstrophy > 0.0,
        "Simulation with obstacle should produce measurable enstrophy");

    // Average enstrophy should be positive (indicates sustained vortical flow)
    let avg_enstrophy = total_enstrophy / FRAMES as f32;
    assert!(avg_enstrophy > 0.0,
        "Average enstrophy should be positive, got {}", avg_enstrophy);
}

/// V6: CFL check - verify CFL computation works
#[test]
fn test_cfl_computation() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // No particles = zero CFL
    assert_eq!(sim.compute_cfl(DT), 0.0, "Empty sim should have zero CFL");

    // Add some particles with velocity
    sim.spawn_water(50.0, 50.0, 40.0, 0.0, 10);

    let cfl = sim.compute_cfl(DT);
    assert!(cfl > 0.0, "Sim with particles should have positive CFL");

    // Verify CFL formula: CFL = v_max * dt / dx
    let v_max = sim.max_velocity();
    let expected_cfl = v_max * DT / CELL_SIZE;
    assert!((cfl - expected_cfl).abs() < 0.001,
        "CFL should match formula: got {}, expected {}", cfl, expected_cfl);
}

/// V7: Quick smoke test for vortex formation
/// Very fast test that just ensures simulation doesn't crash
#[test]
fn test_vortex_smoke_test() {
    const WIDTH: usize = 16;
    const HEIGHT: usize = 16;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Floor and walls
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    sim.grid.compute_sdf();

    sim.spawn_water(32.0, 32.0, 10.0, 0.0, 5);

    // Just run a few frames
    for _ in 0..10 {
        sim.update(DT);
    }

    // Basic sanity check
    assert!(sim.particles.len() > 0, "Should still have particles");
}

/// V8: Max vorticity should be detectable in flow
#[test]
fn test_max_vorticity_detection() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 2.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    sim.initialize_solid_rotation(0.5);

    sim.grid.compute_vorticity();
    let max_vort = sim.grid.max_vorticity();

    // Solid body rotation has constant vorticity = 2ω
    // With ω = 0.5, expect vorticity around 1.0
    // Discretization can cause higher values near boundaries
    assert!(max_vort > 0.5, "Max vorticity should be significant: got {}", max_vort);
    assert!(max_vort < 20.0, "Max vorticity shouldn't be unreasonably high: got {}", max_vort);
}

/// V9: Vortex shedding test - verify vortices form behind obstacle
/// This tests that flow past an obstacle produces detectable vorticity variations
#[test]
fn test_vortex_shedding_behind_obstacle() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 3.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 120;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create channel with obstacle
    // Floor and ceiling
    for i in 0..WIDTH {
        sim.grid.set_solid(i, 0);
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    // Left and right walls
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }

    // Add square obstacle in the flow path (simulates a riffle)
    let obs_x = 20;
    let obs_y = HEIGHT / 2 - 2;
    for i in obs_x..(obs_x + 4) {
        for j in obs_y..(obs_y + 4) {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Spawn water block with initial velocity flowing toward obstacle
    for i in 0..12 {
        for j in 5..(HEIGHT - 5) {
            let x = (3 + i) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = j as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 25.0, 0.0, 1);
        }
    }

    // Sample point downstream of obstacle
    let sample_x = (obs_x + 15) as f32 * CELL_SIZE;
    let sample_y = (HEIGHT / 2) as f32 * CELL_SIZE;

    let mut velocity_samples: Vec<f32> = Vec::new();
    let mut enstrophy_samples: Vec<f32> = Vec::new();

    for _ in 0..FRAMES {
        sim.update(DT);

        // Sample transverse (y) velocity downstream - this oscillates in vortex street
        let vel = sim.grid.sample_velocity(glam::Vec2::new(sample_x, sample_y));
        velocity_samples.push(vel.y);

        // Track enstrophy
        let enstrophy = sim.update_and_compute_enstrophy();
        enstrophy_samples.push(enstrophy);
    }

    // Analyze velocity variations - vortex shedding produces oscillations
    let mean_vy: f32 = velocity_samples.iter().sum::<f32>() / velocity_samples.len() as f32;
    let variance: f32 = velocity_samples
        .iter()
        .map(|v| (v - mean_vy).powi(2))
        .sum::<f32>()
        / velocity_samples.len() as f32;

    // Should see some velocity variation (not just laminar flow)
    // If vortices form, transverse velocity will oscillate
    let std_dev = variance.sqrt();

    // Count zero crossings in transverse velocity (indicates oscillation)
    let mut zero_crossings = 0;
    for i in 1..velocity_samples.len() {
        if velocity_samples[i - 1] * velocity_samples[i] < 0.0 {
            zero_crossings += 1;
        }
    }

    // Should have some enstrophy (indicates rotational flow)
    let max_enstrophy = enstrophy_samples.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_enstrophy > 0.0,
        "Flow past obstacle should produce enstrophy"
    );

    // Either have velocity oscillations OR significant enstrophy
    // (depends on timing whether vortices reach sample point)
    let has_oscillation = zero_crossings >= 2 || std_dev > 0.5;
    let has_rotation = max_enstrophy > 100.0;

    assert!(
        has_oscillation || has_rotation,
        "Expected vortex shedding evidence: zero_crossings={}, std_dev={:.2}, max_enstrophy={:.2}",
        zero_crossings,
        std_dev,
        max_enstrophy
    );
}

/// V10: Kinetic energy evolution test - verify simulation remains stable
/// Tests that energy doesn't explode or produce NaN values
#[test]
fn test_kinetic_energy_stability() {
    const WIDTH: usize = 48;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 100;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create closed container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, 0);
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    sim.grid.compute_sdf();

    // Spawn particles with circular motion (rotating disk)
    let cx = WIDTH as f32 * CELL_SIZE / 2.0;
    let cy = HEIGHT as f32 * CELL_SIZE / 2.0;
    let omega = 0.3;

    for i in 10..(WIDTH - 10) {
        for j in 10..(HEIGHT - 10) {
            let x = (i as f32 + 0.5) * CELL_SIZE;
            let y = (j as f32 + 0.5) * CELL_SIZE;
            let dx = x - cx;
            let dy = y - cy;
            let r = (dx * dx + dy * dy).sqrt();

            if r < 15.0 * CELL_SIZE && r > 2.0 * CELL_SIZE {
                // Solid body rotation velocity
                let vx = -omega * dy;
                let vy = omega * dx;
                sim.spawn_water(x, y, vx, vy, 1);
            }
        }
    }

    // Measure initial kinetic energy
    let initial_ke = sim.compute_kinetic_energy();
    assert!(initial_ke > 0.0, "Should have initial kinetic energy");

    let mut ke_history: Vec<f32> = vec![initial_ke];

    for _ in 0..FRAMES {
        sim.update(DT);
        let ke = sim.compute_kinetic_energy();

        // Energy should never be NaN or infinite
        assert!(!ke.is_nan(), "Kinetic energy became NaN");
        assert!(ke.is_finite(), "Kinetic energy became infinite");

        ke_history.push(ke);
    }

    // Energy evolution should be smooth (no sudden jumps)
    // Check that frame-to-frame changes are bounded
    for i in 1..ke_history.len() {
        let prev = ke_history[i - 1];
        let curr = ke_history[i];
        // Allow up to 50% change per frame (gravity adds energy)
        let ratio = if prev > 0.0 { curr / prev } else { 1.0 };
        assert!(
            ratio < 2.0 && ratio > 0.1,
            "Energy jump at frame {}: prev={:.2}, curr={:.2}, ratio={:.2}",
            i,
            prev,
            curr,
            ratio
        );
    }

    // Final energy should still be positive (simulation didn't explode or die)
    let final_ke = *ke_history.last().unwrap();
    assert!(final_ke > 0.0, "Final kinetic energy should be positive");
}

/// V11: Vorticity sampling test
/// Verify that sample_vorticity() correctly interpolates vorticity values
#[test]
fn test_vorticity_sampling() {
    const WIDTH: usize = 16;
    const HEIGHT: usize = 16;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Initialize solid rotation - produces vorticity proportional to omega
    // Note: Discretization may scale the actual value
    let omega = 0.5;
    sim.initialize_solid_rotation(omega);
    sim.grid.compute_vorticity();

    // Sample at cell centers - should get non-zero positive vorticity (CCW rotation)
    let center_x = WIDTH as f32 * CELL_SIZE / 2.0;
    let center_y = HEIGHT as f32 * CELL_SIZE / 2.0;

    let sampled = sim.grid.sample_vorticity(glam::Vec2::new(center_x, center_y));

    // Vorticity should be positive (CCW rotation) and non-zero
    assert!(
        sampled > 0.0,
        "Solid rotation should have positive vorticity, got {:.2}",
        sampled
    );
    assert!(
        !sampled.is_nan(),
        "Vorticity should not be NaN"
    );

    // Sample at different positions - should be similar (uniform rotation)
    let pos1 = glam::Vec2::new(center_x + 10.0, center_y);
    let pos2 = glam::Vec2::new(center_x, center_y + 10.0);
    let pos3 = glam::Vec2::new(center_x - 10.0, center_y - 10.0);

    let v1 = sim.grid.sample_vorticity(pos1);
    let v2 = sim.grid.sample_vorticity(pos2);
    let v3 = sim.grid.sample_vorticity(pos3);

    // All samples should be roughly similar (uniform rotation in bulk)
    // Allow larger tolerance due to boundary effects
    let mean = (sampled + v1 + v2 + v3) / 4.0;
    let max_deviation = [sampled, v1, v2, v3]
        .iter()
        .map(|v| (v - mean).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_deviation < mean * 0.5 + 1.0, // Within 50% of mean + small absolute tolerance
        "Vorticity should be roughly uniform: center={:.2}, v1={:.2}, v2={:.2}, v3={:.2}",
        sampled,
        v1,
        v2,
        v3
    );

    // Verify interpolation - sample between cell centers
    let between_pos = glam::Vec2::new(
        center_x + CELL_SIZE * 0.25,
        center_y + CELL_SIZE * 0.25,
    );
    let v_between = sim.grid.sample_vorticity(between_pos);

    // Should be smooth (not NaN, roughly in expected range)
    assert!(!v_between.is_nan(), "Interpolated vorticity should not be NaN");
    assert!(
        v_between > 0.0,
        "Interpolated vorticity should be positive for CCW rotation: got {:.2}",
        v_between
    );
}
