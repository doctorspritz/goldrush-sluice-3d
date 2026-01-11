//! 3D Pressure solver tests
//!
//! Tests divergence computation, pressure solve, and gradient application
//! specifically for 3D grid configurations.

use sim3d::{CellType, FlipSimulation3D, Vec3};

/// Test that divergence of zero velocity field is zero
#[test]
fn test_divergence_zero_velocity() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // All velocities start at zero
    // Mark some cells as fluid
    for k in 2..6 {
        for j in 2..6 {
            for i in 2..6 {
                let idx = sim.grid.cell_index(i, j, k);
                sim.grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    sim3d::pressure::compute_divergence(&mut sim.grid);

    // All divergence should be zero
    for &div in &sim.grid.divergence {
        assert!(
            div.abs() < 1e-6,
            "Divergence should be zero for zero velocity field, got: {}",
            div
        );
    }
}

/// Test that W component (Z-velocity) is properly handled in divergence
#[test]
fn test_divergence_includes_w_component() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Mark center cell as fluid
    let center_idx = sim.grid.cell_index(4, 4, 4);
    sim.grid.cell_type[center_idx] = CellType::Fluid;

    // Set W velocities to create non-zero divergence in Z
    // W faces are at (i, j, k) and (i, j, k+1)
    let w_back = sim.grid.w_index(4, 4, 4);
    let w_front = sim.grid.w_index(4, 4, 5);
    sim.grid.w[w_back] = 0.0;
    sim.grid.w[w_front] = 1.0; // Flow exiting in +Z direction

    sim3d::pressure::compute_divergence(&mut sim.grid);

    // Divergence should be positive (flow exiting)
    let div = sim.grid.divergence[center_idx];
    assert!(
        div > 0.0,
        "Divergence should be positive for outward W flow, got: {}",
        div
    );
}

/// Test pressure solver reduces divergence
#[test]
fn test_pressure_solver_reduces_divergence() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Create a fluid region
    for k in 2..6 {
        for j in 2..6 {
            for i in 2..6 {
                let idx = sim.grid.cell_index(i, j, k);
                sim.grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    // Set some non-zero velocities to create divergence
    for k in 2..6 {
        for j in 2..6 {
            let u_idx = sim.grid.u_index(4, j, k);
            sim.grid.u[u_idx] = 1.0;
        }
    }

    // Compute initial divergence
    sim3d::pressure::compute_divergence(&mut sim.grid);
    let max_div_before: f32 = sim
        .grid
        .divergence
        .iter()
        .map(|d| d.abs())
        .fold(0.0, f32::max);

    // Solve pressure
    sim3d::pressure::solve_pressure_jacobi(&mut sim.grid, 100);
    sim3d::pressure::apply_pressure_gradient(&mut sim.grid);

    // Recompute divergence
    sim3d::pressure::compute_divergence(&mut sim.grid);
    let max_div_after: f32 = sim
        .grid
        .divergence
        .iter()
        .map(|d| d.abs())
        .fold(0.0, f32::max);

    assert!(
        max_div_after < max_div_before * 0.1,
        "Pressure solver should significantly reduce divergence. Before: {}, After: {}",
        max_div_before,
        max_div_after
    );
}

/// Test that Z-boundary conditions are properly enforced
#[test]
fn test_z_boundary_conditions() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Set all W velocities to non-zero
    sim.grid.w.fill(5.0);

    sim3d::pressure::enforce_boundary_conditions(&mut sim.grid);

    // Check that W at k=0 and k=depth are zeroed
    for j in 0..8 {
        for i in 0..8 {
            let idx0 = sim.grid.w_index(i, j, 0);
            let idx_depth = sim.grid.w_index(i, j, 8);

            assert_eq!(sim.grid.w[idx0], 0.0, "W at k=0 should be zero");
            assert_eq!(sim.grid.w[idx_depth], 0.0, "W at k=depth should be zero");
        }
    }

    // Interior W velocities should be unchanged
    let interior_idx = sim.grid.w_index(4, 4, 4);
    assert_eq!(
        sim.grid.w[interior_idx], 5.0,
        "Interior W should be unchanged"
    );
}

/// Test that all three velocity components are affected by pressure solve
/// when we create non-divergence-free flow
#[test]
fn test_pressure_affects_all_components() {
    let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

    // Create fluid region
    for k in 2..6 {
        for j in 2..6 {
            for i in 2..6 {
                let idx = sim.grid.cell_index(i, j, k);
                sim.grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    // Set ASYMMETRIC velocities to create non-zero divergence
    // This ensures the pressure solver must correct them
    // U: only on one side (creating positive divergence)
    for k in 2..6 {
        for j in 2..6 {
            let idx = sim.grid.u_index(5, j, k); // Only set outflow U
            sim.grid.u[idx] = 2.0;
        }
    }
    // V: only on top (creating positive divergence)
    for k in 2..6 {
        for i in 2..6 {
            let idx = sim.grid.v_index(i, 5, k); // Only set outflow V
            sim.grid.v[idx] = 2.0;
        }
    }
    // W: only on front (creating positive divergence)
    for j in 2..6 {
        for i in 2..6 {
            let idx = sim.grid.w_index(i, j, 5); // Only set outflow W
            sim.grid.w[idx] = 2.0;
        }
    }

    // Compute divergence - should be non-zero
    sim3d::pressure::compute_divergence(&mut sim.grid);

    // Solve pressure
    sim3d::pressure::solve_pressure_jacobi(&mut sim.grid, 100);
    sim3d::pressure::apply_pressure_gradient(&mut sim.grid);

    // Inflow velocities should now be non-zero (pressure creates inflow to balance outflow)
    let mut u_inflow_created = false;
    let mut v_inflow_created = false;
    let mut w_inflow_created = false;

    // Check for inflow on the opposite sides
    for k in 2..6 {
        for j in 2..6 {
            if sim.grid.u[sim.grid.u_index(2, j, k)].abs() > 0.01 {
                u_inflow_created = true;
            }
        }
    }
    for k in 2..6 {
        for i in 2..6 {
            if sim.grid.v[sim.grid.v_index(i, 2, k)].abs() > 0.01 {
                v_inflow_created = true;
            }
        }
    }
    for j in 2..6 {
        for i in 2..6 {
            if sim.grid.w[sim.grid.w_index(i, j, 2)].abs() > 0.01 {
                w_inflow_created = true;
            }
        }
    }

    assert!(
        u_inflow_created,
        "Pressure should create U inflow to balance divergence"
    );
    assert!(
        v_inflow_created,
        "Pressure should create V inflow to balance divergence"
    );
    assert!(
        w_inflow_created,
        "Pressure should create W inflow to balance divergence"
    );
}
