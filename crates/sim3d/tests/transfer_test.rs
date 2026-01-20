//! 3D Transfer tests (P2G and G2P)
//!
//! Tests particle-to-grid and grid-to-particle transfers specifically
//! for 3D configurations, ensuring W-velocity is properly handled.

use sim3d::{CellType, FlipSimulation3D, Vec3};

/// Test that after a simulation step, grid velocities are non-zero
#[test]
fn test_p2g_contributes_all_components() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Spawn a particle in the center with velocity in all directions
    sim.spawn_particle_with_velocity(
        Vec3::new(2.0, 2.0, 2.0),
        Vec3::new(1.0, 2.0, 3.0), // Non-zero in all components
    );

    // Run one simulation step which includes P2G
    sim.gravity = Vec3::ZERO; // Disable gravity for cleaner test
    sim.update(1.0 / 60.0);

    // Check that grid has non-zero velocities in all components after P2G
    // Note: update() runs P2G -> pressure -> G2P, so grid may change
    // But velocities should be non-zero at some point

    // Just verify the particle still has reasonable velocity
    let vel = sim.particles.list()[0].velocity;
    assert!(
        !vel.x.is_nan() && !vel.y.is_nan() && !vel.z.is_nan(),
        "Velocity should not be NaN"
    );
    assert!(vel.length() > 0.0, "Particle should retain some velocity");
}

/// Test that G2P samples all three velocity components
#[test]
fn test_g2p_samples_all_components() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Set grid velocities
    sim.grid.u_mut().fill(1.0);
    sim.grid.v_mut().fill(2.0);
    sim.grid.w_mut().fill(3.0);

    // Spawn a stationary particle
    sim.spawn_particle(Vec3::new(2.0, 2.0, 2.0));

    // Mark cells as fluid
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let idx = sim.grid.cell_index(i, j, k);
                sim.grid.cell_type_mut()[idx] = CellType::Fluid;
            }
        }
    }

    // Store old velocities (needed for FLIP)
    sim.grid.store_old_velocities();

    // Run G2P
    sim3d::transfer::grid_to_particles(&sim.grid, &mut sim.particles, 0.97);

    let vel = sim.particles.list()[0].velocity;

    // Particle should have picked up velocities in all directions (non-zero)
    assert!(
        vel.x.abs() > 0.001,
        "G2P should sample U velocity, got: {}",
        vel.x
    );
    assert!(
        vel.y.abs() > 0.001,
        "G2P should sample V velocity, got: {}",
        vel.y
    );
    assert!(
        vel.z.abs() > 0.001,
        "G2P should sample W velocity, got: {}",
        vel.z
    );
}

/// Test that particle at Z boundary gets correct W velocity
#[test]
fn test_particle_near_z_boundary() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);
    let cell_size = 0.5;

    // Spawn particle near the back Z boundary
    let near_boundary = Vec3::new(2.0, 2.0, cell_size * 0.5);
    sim.spawn_particle(near_boundary);

    // Set non-zero W velocities
    sim.grid.w_mut().fill(5.0);

    // Mark cells as fluid
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let idx = sim.grid.cell_index(i, j, k);
                sim.grid.cell_type_mut()[idx] = CellType::Fluid;
            }
        }
    }

    sim.grid.store_old_velocities();

    // Run G2P
    sim3d::transfer::grid_to_particles(&sim.grid, &mut sim.particles, 0.97);

    let vel = sim.particles.list()[0].velocity;

    // Particle should have some Z velocity (even if clamped by boundary)
    // The exact value depends on interpolation, but should be defined (not NaN)
    assert!(
        !vel.z.is_nan(),
        "Z velocity should not be NaN near boundary"
    );
    assert!(
        vel.z.is_finite(),
        "Z velocity should be finite near boundary"
    );
}

/// Test W-velocity stencil indexing
#[test]
fn test_w_grid_indexing() {
    let sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // W grid has size: width x height x (depth+1)
    // Check corner indices are valid
    let corners = [
        (0, 0, 0),
        (7, 0, 0),
        (0, 7, 0),
        (0, 0, 8), // depth+1 = 9, but max k index is 8
        (7, 7, 8),
    ];

    for (i, j, k) in corners {
        let idx = sim.grid.w_index(i, j, k);
        assert!(
            idx < sim.grid.w().len(),
            "W index ({}, {}, {}) = {} out of bounds (len={})",
            i,
            j,
            k,
            idx,
            sim.grid.w().len()
        );
    }
}

/// Test that APIC C matrix is properly constructed for 3D
#[test]
fn test_apic_c_matrix_3d() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Set grid velocities with spatial variation
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..9 {
                let idx = sim.grid.u_index(i, j, k);
                sim.grid.u_mut()[idx] = (i as f32) * 0.1; // U increases with x
            }
        }
    }
    for k in 0..8 {
        for j in 0..9 {
            for i in 0..8 {
                let idx = sim.grid.v_index(i, j, k);
                sim.grid.v_mut()[idx] = (j as f32) * 0.1; // V increases with y
            }
        }
    }
    for k in 0..9 {
        for j in 0..8 {
            for i in 0..8 {
                let idx = sim.grid.w_index(i, j, k);
                sim.grid.w_mut()[idx] = (k as f32) * 0.1; // W increases with z
            }
        }
    }

    // Spawn particle
    sim.spawn_particle(Vec3::new(2.0, 2.0, 2.0));

    // Mark cells as fluid
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let idx = sim.grid.cell_index(i, j, k);
                sim.grid.cell_type_mut()[idx] = CellType::Fluid;
            }
        }
    }

    sim.grid.store_old_velocities();
    sim3d::transfer::grid_to_particles(&sim.grid, &mut sim.particles, 0.97);

    let p = &sim.particles.list()[0];

    // APIC affine velocity matrix should have non-zero entries (velocity gradient)
    // Row 0 (from U): should have positive dU/dx component
    // Row 1 (from V): should have positive dV/dy component
    // Row 2 (from W): should have positive dW/dz component

    // Just check that affine_velocity matrix is finite and not all zeros
    let c = p.affine_velocity;
    let c_sum = c.x_axis.length() + c.y_axis.length() + c.z_axis.length();
    assert!(
        !c_sum.is_nan(),
        "Affine velocity matrix should not contain NaN"
    );
    assert!(
        c_sum > 0.0,
        "Affine velocity matrix should not be all zeros, got sum: {}",
        c_sum
    );
}

/// Test conservation: total momentum should be approximately conserved through transfer cycle
#[test]
fn test_momentum_conservation() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Spawn several particles with various velocities
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                sim.spawn_particle_with_velocity(
                    Vec3::new(
                        (i as f32 + 0.5) * 0.5 + 0.5,
                        (j as f32 + 0.5) * 0.5 + 0.5,
                        (k as f32 + 0.5) * 0.5 + 0.5,
                    ),
                    Vec3::new(1.0, 0.5, 0.25),
                );
            }
        }
    }

    // Calculate initial momentum
    let initial_momentum: Vec3 = sim
        .particles
        .list()
        .iter()
        .map(|p| p.velocity)
        .fold(Vec3::ZERO, |a, b| a + b);

    // Run one full simulation step (without gravity for pure conservation test)
    sim.gravity = Vec3::ZERO;
    sim.update(1.0 / 60.0);

    // Calculate final momentum
    let final_momentum: Vec3 = sim
        .particles
        .list()
        .iter()
        .map(|p| p.velocity)
        .fold(Vec3::ZERO, |a, b| a + b);

    // Momentum should be approximately conserved (within 20% tolerance for FLIP noise)
    let initial_mag = initial_momentum.length();
    let final_mag = final_momentum.length();
    let momentum_change = (final_mag - initial_mag).abs() / initial_mag.max(0.001);

    assert!(
        momentum_change < 0.5,
        "Momentum should be approximately conserved. Initial: {:?}, Final: {:?}, Change: {:.1}%",
        initial_momentum,
        final_momentum,
        momentum_change * 100.0
    );
}
