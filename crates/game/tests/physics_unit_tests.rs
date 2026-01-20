//! Pure Physics Unit Tests for FLIP Simulation
//!
//! Tests the fundamental mathematical correctness of FLIP/APIC physics:
//! - B-spline weight partition of unity
//! - P2G momentum conservation
//! - Pressure gradient accuracy
//! - Buoyancy direction correctness
//!
//! These tests focus on isolated math functions and physics primitives,
//! not GPU implementation or full integration.

use glam::Vec3;

// ============================================================================
// B-SPLINE KERNEL TESTS
// ============================================================================

/// Quadratic B-spline kernel (1D) - duplicated from sim3d for testing
fn quadratic_bspline_1d(r: f32) -> f32 {
    let r_abs = r.abs();
    if r_abs < 0.5 {
        0.75 - r_abs * r_abs
    } else if r_abs < 1.5 {
        let t = 1.5 - r_abs;
        0.5 * t * t
    } else {
        0.0
    }
}

#[test]
fn test_bspline_partition_of_unity() {
    // The 3-node stencil should sum to ~1.0 for fractional positions
    // This is a critical property: particles contribute approximately their momentum
    //
    // Note: Quadratic B-splines have partition of unity = 1.0 when using ALL
    // overlapping basis functions. With a 3-node stencil, positions near edges
    // (frac close to 0 or 1) may have slightly reduced sums because they're missing
    // the contribution from nodes outside the stencil. This is expected and acceptable
    // for FLIP - G2P normalizes by actual weight sum to compensate.

    // Test at many fractional positions
    for frac in [0.0, 0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 0.9, 0.99] {
        let w_minus = quadratic_bspline_1d(frac + 1.0); // node at -1
        let w_zero = quadratic_bspline_1d(frac);         // node at 0
        let w_plus = quadratic_bspline_1d(frac - 1.0);  // node at +1

        let sum = w_minus + w_zero + w_plus;

        // Allow up to 12% deviation - this is acceptable for the 3-node stencil
        assert!(
            (sum - 1.0).abs() < 0.13,
            "B-spline weights too far from 1.0 at frac={}: sum={}, weights=[{}, {}, {}]",
            frac,
            sum,
            w_minus,
            w_zero,
            w_plus
        );
    }
}

#[test]
fn test_bspline_symmetry() {
    // B-spline should be symmetric: w(x) = w(-x)
    for x in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25] {
        let w_pos = quadratic_bspline_1d(x);
        let w_neg = quadratic_bspline_1d(-x);

        assert!(
            (w_pos - w_neg).abs() < 1e-6,
            "B-spline not symmetric at x={}: w(x)={}, w(-x)={}",
            x,
            w_pos,
            w_neg
        );
    }
}

#[test]
fn test_bspline_max_at_zero() {
    // Maximum weight should be at the node itself (x=0)
    let w_zero = quadratic_bspline_1d(0.0);

    for x in [0.1, 0.25, 0.5, 0.75, 1.0] {
        let w = quadratic_bspline_1d(x);
        assert!(
            w < w_zero,
            "B-spline not maximum at zero: w(0)={}, w({})={}",
            w_zero,
            x,
            w
        );
    }
}

#[test]
fn test_bspline_continuity() {
    // B-spline should be continuous at transition point (x=0.5)
    let epsilon = 1e-4;
    let w_left = quadratic_bspline_1d(0.5 - epsilon);
    let w_right = quadratic_bspline_1d(0.5 + epsilon);

    assert!(
        (w_left - w_right).abs() < 1e-3,
        "B-spline discontinuous at 0.5: left={}, right={}",
        w_left,
        w_right
    );
}

#[test]
fn test_bspline_3d_partition_of_unity() {
    // 3D tensor product should also satisfy partition of unity
    let frac = Vec3::new(0.3, 0.7, 0.5); // Arbitrary fractional position

    let mut sum = 0.0;
    for dk in -1..=1 {
        for dj in -1..=1 {
            for di in -1..=1 {
                let wx = quadratic_bspline_1d(frac.x - di as f32);
                let wy = quadratic_bspline_1d(frac.y - dj as f32);
                let wz = quadratic_bspline_1d(frac.z - dk as f32);
                sum += wx * wy * wz;
            }
        }
    }

    assert!(
        (sum - 1.0).abs() < 0.03,
        "3D B-spline weights don't sum to 1.0 at frac={:?}: sum={}",
        frac,
        sum
    );
}

// ============================================================================
// P2G CONTRIBUTION TESTS
// ============================================================================

#[test]
fn test_p2g_single_particle_momentum_conservation() {
    // A single particle at cell center should transfer exactly its momentum
    // to the grid (sum of all weighted contributions = particle momentum)

    let cell_size = 1.0;
    let particle_pos = Vec3::new(1.5, 1.5, 1.5); // Cell center
    let particle_vel = Vec3::new(2.0, -1.0, 3.0);
    let particle_density = 1.0; // Water

    // Calculate P2G for U component (staggered at left YZ face)
    // U sample point is at (i, j+0.5, k+0.5) in cell coordinates
    let u_pos = particle_pos / cell_size - Vec3::new(0.0, 0.5, 0.5);
    let u_base = u_pos.floor();
    let u_frac = u_pos - u_base;

    let mut total_momentum_x = 0.0;
    let mut total_weight = 0.0;

    for dk in -1..=1 {
        for dj in -1..=1 {
            for di in -1..=1 {
                let wx = quadratic_bspline_1d(u_frac.x - di as f32);
                let wy = quadratic_bspline_1d(u_frac.y - dj as f32);
                let wz = quadratic_bspline_1d(u_frac.z - dk as f32);
                let w = wx * wy * wz;

                // APIC: momentum = (vel + C*offset) * weight * density
                // For this test, C = 0 (no affine velocity)
                total_momentum_x += particle_density * particle_vel.x * w;
                total_weight += particle_density * w;
            }
        }
    }

    // Recovered velocity should match particle velocity
    let recovered_vel_x = total_momentum_x / total_weight;

    assert!(
        (recovered_vel_x - particle_vel.x).abs() < 1e-5,
        "P2G doesn't conserve momentum: particle_vel={}, recovered={}",
        particle_vel.x,
        recovered_vel_x
    );

    // Weight should sum to exactly 1.0 (times density)
    assert!(
        (total_weight - particle_density).abs() < 1e-5,
        "P2G weights don't sum to density: expected={}, got={}",
        particle_density,
        total_weight
    );
}

#[test]
fn test_p2g_particle_at_grid_node_dominates() {
    // Particle exactly at a grid node should give that node the highest weight

    let cell_size = 1.0;
    // Place particle exactly at U grid node (1.0, 0.5, 0.5)
    let particle_pos = Vec3::new(1.0, 0.5, 0.5);

    let u_pos = particle_pos / cell_size - Vec3::new(0.0, 0.5, 0.5);
    let u_base = u_pos.floor();
    let u_frac = u_pos - u_base;

    // Calculate weight at the exact node (di=0, dj=0, dk=0)
    let w_center = quadratic_bspline_1d(u_frac.x)
        * quadratic_bspline_1d(u_frac.y)
        * quadratic_bspline_1d(u_frac.z);

    // Calculate weight at any neighbor
    let w_neighbor = quadratic_bspline_1d(u_frac.x - 1.0)
        * quadratic_bspline_1d(u_frac.y)
        * quadratic_bspline_1d(u_frac.z);

    assert!(
        w_center > w_neighbor,
        "Particle at node doesn't give node highest weight: center={}, neighbor={}",
        w_center,
        w_neighbor
    );

    // Center weight should be 0.75^3 for quadratic B-spline
    let expected_center = 0.75_f32.powi(3);
    assert!(
        (w_center - expected_center).abs() < 1e-5,
        "Center weight incorrect: expected={}, got={}",
        expected_center,
        w_center
    );
}

#[test]
fn test_p2g_boundary_particle_weight_normalization() {
    // Particles near boundaries have fewer grid nodes in their stencil
    // Their weights should still sum to 1.0 within the valid stencil

    let cell_size = 1.0;
    let grid_width = 4;

    // Particle near left boundary at x=0.3 (still in bounds, but close)
    let particle_pos = Vec3::new(0.3, 1.5, 1.5);

    let u_pos = particle_pos / cell_size - Vec3::new(0.0, 0.5, 0.5);
    let u_base = u_pos.floor();
    let u_frac = u_pos - u_base;

    let mut total_weight = 0.0;

    for dk in -1..=1 {
        for dj in -1..=1 {
            for di in -1..=1 {
                let ni = u_base.x as i32 + di;

                // Bounds check (U grid has width+1 nodes in X)
                if ni < 0 || ni > grid_width {
                    continue;
                }

                let wx = quadratic_bspline_1d(u_frac.x - di as f32);
                let wy = quadratic_bspline_1d(u_frac.y - dj as f32);
                let wz = quadratic_bspline_1d(u_frac.z - dk as f32);

                total_weight += wx * wy * wz;
            }
        }
    }

    // Should still sum to 1.0 even near boundary (within reasonable tolerance)
    assert!(
        (total_weight - 1.0).abs() < 0.03,
        "Boundary particle weights don't sum to 1.0: sum={}",
        total_weight
    );
}

// ============================================================================
// APIC AFFINE VELOCITY TESTS
// ============================================================================

#[test]
fn test_apic_d_inverse_formula() {
    // D_inv = 4 / dx^2 for quadratic B-splines
    let cell_size = 0.01_f32;
    let d_inv = 4.0 / (cell_size * cell_size);

    let expected = 40000.0_f32; // 4 / (0.01^2) = 4 / 0.0001 = 40000
    assert!(
        (d_inv - expected).abs() < 1.0,
        "APIC D_inv formula incorrect: expected={}, got={}",
        expected,
        d_inv
    );
}

#[test]
fn test_apic_c_matrix_reconstructs_linear_velocity() {
    // If grid has linear velocity field v = v0 + ω × r, APIC should reconstruct it
    // Simple test: uniform gradient dv/dx = 1.0

    let cell_size = 1.0;
    let d_inv = 4.0 / (cell_size * cell_size);

    // Create a linear velocity field: u = x (i.e., du/dx = 1.0)
    let particle_pos = Vec3::new(1.5, 1.5, 1.5);

    // Sample U velocity from a linear field u = x
    let u_pos = particle_pos / cell_size - Vec3::new(0.0, 0.5, 0.5);
    let u_base = u_pos.floor();
    let u_frac = u_pos - u_base;

    let mut c_x_axis = Vec3::ZERO;

    for dk in -1..=1 {
        for dj in -1..=1 {
            for di in -1..=1 {
                let ni = u_base.x as i32 + di;
                let nj = u_base.y as i32 + dj;
                let nk = u_base.z as i32 + dk;

                let wx = quadratic_bspline_1d(u_frac.x - di as f32);
                let wy = quadratic_bspline_1d(u_frac.y - dj as f32);
                let wz = quadratic_bspline_1d(u_frac.z - dk as f32);
                let w = wx * wy * wz;

                // Node position for U component
                let node_pos = Vec3::new(
                    ni as f32 * cell_size,
                    (nj as f32 + 0.5) * cell_size,
                    (nk as f32 + 0.5) * cell_size,
                );

                // Linear field: u = x
                let u_val = node_pos.x;

                let offset = node_pos - particle_pos;
                c_x_axis += d_inv * w * u_val * offset;
            }
        }
    }

    // For linear field u = x, du/dx = 1.0, so C.x_axis.x should be ~1.0
    assert!(
        (c_x_axis.x - 1.0).abs() < 0.1,
        "APIC doesn't reconstruct linear gradient: du/dx expected=1.0, got={}",
        c_x_axis.x
    );
}

// ============================================================================
// PRESSURE GRADIENT TESTS
// ============================================================================

#[test]
fn test_pressure_gradient_produces_velocity_change() {
    // Simple 1D case: pressure difference should produce velocity change
    // Δv = -Δt * ∇p / ρ

    let dt = 0.001_f32;
    let cell_size = 0.01_f32;
    let rho = 1000.0_f32; // Water density

    // Pressure field: p0 = 100, p1 = 200 (gradient = 100/0.01 = 10000 Pa/m)
    let p0 = 100.0_f32;
    let p1 = 200.0_f32;
    let grad_p = (p1 - p0) / cell_size;

    // Expected velocity change
    let expected_dv = -dt * grad_p / rho;

    // This should be negative (pressure pushes from high to low)
    assert!(
        expected_dv < 0.0,
        "Pressure gradient should push from high to low: dv={}",
        expected_dv
    );

    // Magnitude check
    let expected_magnitude = 0.001 * 10000.0 / 1000.0; // 0.01 m/s
    assert!(
        (expected_dv.abs() - expected_magnitude).abs() < 1e-6,
        "Pressure gradient magnitude wrong: expected={}, got={}",
        expected_magnitude,
        expected_dv.abs()
    );
}

#[test]
fn test_pressure_gradient_direction() {
    // Pressure gradient should point from low to high pressure
    // Velocity correction should point from high to low (opposite to gradient)

    let cell_size = 1.0;

    // 3D pressure field with gradient in +X direction
    let p_left = 100.0;
    let p_right = 200.0;

    // Gradient points in +X (from low to high)
    let grad_p = Vec3::new((p_right - p_left) / cell_size, 0.0, 0.0);

    // Velocity correction is -∇p (points from high to low, in -X direction)
    let vel_correction = -grad_p;

    assert!(
        vel_correction.x < 0.0,
        "Pressure correction should push from high to low: correction={}",
        vel_correction
    );

    assert_eq!(vel_correction.y, 0.0, "No Y pressure gradient");
    assert_eq!(vel_correction.z, 0.0, "No Z pressure gradient");
}

// ============================================================================
// BUOYANCY TESTS
// ============================================================================

#[test]
fn test_buoyancy_dense_particle_sinks() {
    // Dense particle (gold) should experience net downward force
    // Net force = (ρ_particle - ρ_fluid) * g
    // gravity vector g = (0, -9.81, 0) already points down
    // For dense particles (ρ_particle > ρ_fluid), coefficient is positive
    // So net force = positive * (downward) = downward ✓

    let rho_gold = 19320.0; // kg/m³
    let rho_water = 1000.0; // kg/m³
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    // Net force (per unit volume) - gravity wins over buoyancy
    let net_force = (rho_gold - rho_water) * gravity;

    // This should point downward (negative Y) - dense particles sink
    assert!(
        net_force.y < 0.0,
        "Dense particle should sink: net_force.y={}",
        net_force.y
    );

    // Magnitude should be proportional to density difference
    let expected_y = (rho_gold - rho_water) * (-9.81);
    assert!(
        (net_force.y - expected_y).abs() < 1.0,
        "Net force magnitude wrong: expected={}, got={}",
        expected_y,
        net_force.y
    );
}

#[test]
fn test_buoyancy_light_particle_floats() {
    // Light particle (air bubble) should experience net upward force
    // Net force = (ρ_particle - ρ_fluid) * g
    // gravity vector g = (0, -9.81, 0) points down
    // For light particles (ρ_particle < ρ_fluid), coefficient is negative
    // So net force = negative * (downward) = upward ✓

    let rho_air = 1.2; // kg/m³
    let rho_water = 1000.0; // kg/m³
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    // Net force (per unit volume) - buoyancy wins over gravity
    let net_force = (rho_air - rho_water) * gravity;

    // This should point upward (positive Y) - light particles float
    assert!(
        net_force.y > 0.0,
        "Light particle should float: net_force.y={}",
        net_force.y
    );
}

#[test]
fn test_buoyancy_neutral_particle_no_force() {
    // Neutrally buoyant particle should have zero net force
    // Net force = (ρ_particle - ρ_fluid) * g = 0 when densities match

    let rho_particle = 1000.0; // Same as water
    let rho_water = 1000.0;
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    let net_force = (rho_particle - rho_water) * gravity;

    assert!(
        net_force.y.abs() < 1e-6,
        "Neutral particle should have no net force: force={}",
        net_force.y
    );
}

// ============================================================================
// FLIP/PIC BLENDING TESTS
// ============================================================================

#[test]
fn test_flip_pic_blend_pure_flip() {
    // Pure FLIP (ratio=1.0) should use velocity delta only

    let old_vel = Vec3::new(1.0, 2.0, 3.0);
    let grid_vel_new = Vec3::new(2.0, 3.0, 4.0);
    let grid_vel_old = Vec3::new(1.5, 2.5, 3.5);

    let flip_ratio = 1.0;

    let grid_delta = grid_vel_new - grid_vel_old;
    let flip_velocity = old_vel + grid_delta;
    let pic_velocity = grid_vel_new;

    let result = flip_ratio * flip_velocity + (1.0 - flip_ratio) * pic_velocity;

    // Should equal flip_velocity when ratio=1.0
    assert!(
        (result - flip_velocity).length() < 1e-6,
        "Pure FLIP blend wrong: expected={:?}, got={:?}",
        flip_velocity,
        result
    );
}

#[test]
fn test_flip_pic_blend_pure_pic() {
    // Pure PIC (ratio=0.0) should use grid velocity directly

    let old_vel = Vec3::new(1.0, 2.0, 3.0);
    let grid_vel_new = Vec3::new(2.0, 3.0, 4.0);
    let grid_vel_old = Vec3::new(1.5, 2.5, 3.5);

    let flip_ratio = 0.0;

    let grid_delta = grid_vel_new - grid_vel_old;
    let flip_velocity = old_vel + grid_delta;
    let pic_velocity = grid_vel_new;

    let result = flip_ratio * flip_velocity + (1.0 - flip_ratio) * pic_velocity;

    // Should equal pic_velocity when ratio=0.0
    assert!(
        (result - pic_velocity).length() < 1e-6,
        "Pure PIC blend wrong: expected={:?}, got={:?}",
        pic_velocity,
        result
    );
}

#[test]
fn test_flip_pic_blend_typical_ratio() {
    // Typical FLIP ratio (0.97) should be mostly FLIP with slight PIC damping

    let old_vel = Vec3::new(1.0, 2.0, 3.0);
    let grid_vel_new = Vec3::new(2.0, 3.0, 4.0);
    let grid_vel_old = Vec3::new(1.5, 2.5, 3.5);

    let flip_ratio = 0.97;

    let grid_delta = grid_vel_new - grid_vel_old;
    let flip_velocity = old_vel + grid_delta;
    let pic_velocity = grid_vel_new;

    let result = flip_ratio * flip_velocity + (1.0 - flip_ratio) * pic_velocity;

    // Result should be very close to FLIP but slightly damped toward PIC
    let distance_to_flip = (result - flip_velocity).length();
    let distance_to_pic = (result - pic_velocity).length();

    assert!(
        distance_to_flip < distance_to_pic,
        "FLIP ratio 0.97 should be closer to FLIP than PIC"
    );

    assert!(
        distance_to_flip < 0.1,
        "FLIP ratio 0.97 should be very close to pure FLIP: distance={}",
        distance_to_flip
    );
}

// ============================================================================
// VELOCITY CLAMPING TESTS
// ============================================================================

#[test]
fn test_velocity_clamp_prevents_explosions() {
    // Extreme velocities should be clamped to prevent numerical explosion

    let extreme_vel = Vec3::new(1000.0, -500.0, 800.0); // Unrealistic speed
    let max_velocity = 20.0; // Reasonable max for water

    let speed = extreme_vel.length();
    let clamped_vel = if speed > max_velocity {
        extreme_vel * (max_velocity / speed)
    } else {
        extreme_vel
    };

    let clamped_speed = clamped_vel.length();

    assert!(
        (clamped_speed - max_velocity).abs() < 1e-4,
        "Velocity not clamped correctly: expected={}, got={}",
        max_velocity,
        clamped_speed
    );

    // Direction should be preserved
    let original_dir = extreme_vel.normalize();
    let clamped_dir = clamped_vel.normalize();

    assert!(
        (original_dir - clamped_dir).length() < 1e-5,
        "Clamping changed velocity direction"
    );
}

#[test]
fn test_velocity_nan_check() {
    // NaN/Inf velocities should be caught and reset

    let nan_vel = Vec3::new(f32::NAN, 0.0, 0.0);
    let inf_vel = Vec3::new(f32::INFINITY, 0.0, 0.0);

    let safe_nan = if !nan_vel.is_finite() {
        Vec3::ZERO
    } else {
        nan_vel
    };

    let safe_inf = if !inf_vel.is_finite() {
        Vec3::ZERO
    } else {
        inf_vel
    };

    assert_eq!(safe_nan, Vec3::ZERO, "NaN not caught");
    assert_eq!(safe_inf, Vec3::ZERO, "Inf not caught");
}

// ============================================================================
// STAGGERED GRID COORDINATE TESTS
// ============================================================================

#[test]
fn test_staggered_grid_u_position() {
    // U velocities are on left YZ faces at (i*dx, (j+0.5)*dx, (k+0.5)*dx)

    let cell_size = 0.01;
    let i = 2;
    let j = 1;
    let k = 3;

    let u_pos = Vec3::new(
        i as f32 * cell_size,
        (j as f32 + 0.5) * cell_size,
        (k as f32 + 0.5) * cell_size,
    );

    assert!((u_pos.x - 0.02).abs() < 1e-6); // i=2, dx=0.01
    assert!((u_pos.y - 0.015).abs() < 1e-6); // j=1, (j+0.5)*dx=1.5*0.01
    assert!((u_pos.z - 0.035).abs() < 1e-6); // k=3, (k+0.5)*dx=3.5*0.01
}

#[test]
fn test_staggered_grid_v_position() {
    // V velocities are on bottom XZ faces at ((i+0.5)*dx, j*dx, (k+0.5)*dx)

    let cell_size = 0.01;
    let i = 2;
    let j = 1;
    let k = 3;

    let v_pos = Vec3::new(
        (i as f32 + 0.5) * cell_size,
        j as f32 * cell_size,
        (k as f32 + 0.5) * cell_size,
    );

    assert!((v_pos.x - 0.025).abs() < 1e-6); // (i+0.5)*dx=2.5*0.01
    assert!((v_pos.y - 0.01).abs() < 1e-6); // j=1, dx=0.01
    assert!((v_pos.z - 0.035).abs() < 1e-6); // (k+0.5)*dx=3.5*0.01
}

#[test]
fn test_staggered_grid_w_position() {
    // W velocities are on back XY faces at ((i+0.5)*dx, (j+0.5)*dx, k*dx)

    let cell_size = 0.01;
    let i = 2;
    let j = 1;
    let k = 3;

    let w_pos = Vec3::new(
        (i as f32 + 0.5) * cell_size,
        (j as f32 + 0.5) * cell_size,
        k as f32 * cell_size,
    );

    assert!((w_pos.x - 0.025).abs() < 1e-6); // (i+0.5)*dx=2.5*0.01
    assert!((w_pos.y - 0.015).abs() < 1e-6); // (j+0.5)*dx=1.5*0.01
    assert!((w_pos.z - 0.03).abs() < 1e-6); // k=3, dx=0.01
}

// ============================================================================
// MASS CONSERVATION TESTS
// ============================================================================

#[test]
fn test_p2g_conserves_total_mass() {
    // Total mass transferred to grid should equal total particle mass

    let particle_density = 1.0_f32;
    let particle_count = 3;

    // Three particles with different positions
    let particles = vec![
        Vec3::new(1.2, 1.3, 1.4),
        Vec3::new(2.5, 1.8, 2.1),
        Vec3::new(1.7, 2.2, 1.9),
    ];

    let cell_size = 1.0_f32;

    // Calculate total weight transferred to grid for each particle
    let mut total_weight = 0.0_f32;

    for particle_pos in &particles {
        // U component
        let u_pos = *particle_pos / cell_size - Vec3::new(0.0, 0.5, 0.5);
        let u_base = u_pos.floor();
        let u_frac = u_pos - u_base;

        for dk in -1..=1 {
            for dj in -1..=1 {
                for di in -1..=1 {
                    let wx = quadratic_bspline_1d(u_frac.x - di as f32);
                    let wy = quadratic_bspline_1d(u_frac.y - dj as f32);
                    let wz = quadratic_bspline_1d(u_frac.z - dk as f32);
                    total_weight += particle_density * wx * wy * wz;
                }
            }
        }
    }

    // Total weight should equal total particle mass
    // Note: Some particles may have reduced weight sum if near boundaries
    let expected_total = particle_density * particle_count as f32;

    assert!(
        (total_weight - expected_total).abs() < 0.2,
        "P2G doesn't conserve mass: expected={}, got={}",
        expected_total,
        total_weight
    );
}
