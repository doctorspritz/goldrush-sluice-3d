//! Regression tests for module refactoring.
//!
//! These tests capture the EXACT numerical behavior of the current implementation.
//! If any test fails after refactoring, it indicates a behavioral change.
//!
//! Golden values captured on: 2025-12-29
//! Run `cargo test -p sim capture_all_golden -- --ignored --nocapture` to regenerate.

use sim::flip::FlipSimulation;
use sim::grid::{quadratic_bspline, quadratic_bspline_1d};
use sim::particle::{Particle, ParticleMaterial};
use glam::Vec2;

const DT: f32 = 1.0 / 60.0;

// ============= GOLDEN VALUES =============
// P2G
const GOLDEN_P2G_U_SUM: f32 = 90.0000000000;
const GOLDEN_P2G_V_SUM: f32 = 0.0000000000;
const GOLDEN_P2G_U_MAX: f32 = 1.0000000000;
const GOLDEN_P2G_V_MAX: f32 = 0.0000000000;

// Pressure Solve
const GOLDEN_DIV_BEFORE: f32 = 0.0000000000;
const GOLDEN_DIV_AFTER: f32 = 0.0000000000;
const GOLDEN_PRESSURE_SUM: f32 = 0.0000000000;
const GOLDEN_P_MIN: f32 = 0.0000000000;
const GOLDEN_P_MAX: f32 = 0.0000000000;
const GOLDEN_P_AVG: f32 = 0.0000000000;

// FLIP Cycle
const GOLDEN_FLIP_VEL_BEFORE_X: f32 = 64.0000000000;
const GOLDEN_FLIP_VEL_BEFORE_Y: f32 = 0.0000000000;
const GOLDEN_FLIP_VEL_AFTER_X: f32 = 64.0000000000;
const GOLDEN_FLIP_VEL_AFTER_Y: f32 = 0.0000000000;

// Extrapolation
const GOLDEN_EXTRAP_CHANGED_COUNT: usize = 80;
const GOLDEN_EXTRAP_U_SUM_AFTER: f32 = 170.0000000000;

// Vorticity
const GOLDEN_ENSTROPHY: f32 = 1.0000000000;
const GOLDEN_MAX_VORTICITY: f32 = 0.5000000000;
const GOLDEN_TOTAL_VORTICITY: f32 = 4.0000000000;

// Full Step
const GOLDEN_STEP_KE: f32 = 1120.8897705078;
const GOLDEN_STEP_MAX_V: f32 = 5.9184269905;
const GOLDEN_STEP_PARTICLE_COUNT: usize = 64;
const GOLDEN_STEP_POS_SUM_X: f32 = 497.0665588379;
const GOLDEN_STEP_POS_SUM_Y: f32 = 502.2225341797;
const GOLDEN_STEP_VEL_SUM_X: f32 = 64.0000000000;
const GOLDEN_STEP_VEL_SUM_Y: f32 = 373.3334655762;

// Sediment
const GOLDEN_DEPOSITED_CELLS: usize = 0;
const GOLDEN_REMAINING_SAND: usize = 4;

// SDF
const GOLDEN_SDF_AT_BOUNDARY: f32 = -1.0000000000;
const GOLDEN_SDF_FLUID_CENTER: f32 = 7.0000000000;
const GOLDEN_SDF_FLUID_EDGE: f32 = 3.5000000000;
const GOLDEN_SDF_AIR_REGION: f32 = 1.2500000000;
const GOLDEN_SDF_GRAD_X: f32 = 1.0000000000;
const GOLDEN_SDF_GRAD_Y: f32 = 0.0000000000;
const GOLDEN_SDF_FINITE_SUM: f32 = 680.0000000000;

// Interpolation Kernels
const GOLDEN_KERNEL_0: f32 = 0.7500000000;
const GOLDEN_KERNEL_0_5: f32 = 0.5000000000;
const GOLDEN_KERNEL_1: f32 = 0.1250000000;
const GOLDEN_KERNEL_1_5: f32 = 0.0000000000;
const GOLDEN_KERNEL_2D_CENTER: f32 = 0.5625000000;
const GOLDEN_KERNEL_2D_OFFSET: f32 = 0.2500000000;
const GOLDEN_KERNEL_SUM_GRID: f32 = 1.0000000000;

// Tolerance for floating point comparisons
const TOLERANCE: f32 = 1e-4;
const LOOSE_TOLERANCE: f32 = 1e-2;

// ============= TEST SETUP =============

fn create_deterministic_sim() -> FlipSimulation {
    let mut sim = FlipSimulation::new(16, 16, 1.0);
    // Fixed positions - NO randomness via direct particle insertion
    for i in 4..12 {
        for j in 4..12 {
            let x = i as f32 + 0.25;
            let y = j as f32 + 0.25;
            sim.particles.list.push(Particle::new(
                Vec2::new(x, y),
                Vec2::new(1.0, 0.0),
                ParticleMaterial::Water,
            ));
        }
    }
    sim
}

fn create_sediment_sim() -> FlipSimulation {
    let mut sim = FlipSimulation::new(16, 16, 1.0);
    // Water - direct insertion, no randomness
    for i in 4..12 {
        for j in 4..12 {
            sim.particles.list.push(Particle::new(
                Vec2::new(i as f32 + 0.5, j as f32 + 0.5),
                Vec2::ZERO,
                ParticleMaterial::Water,
            ));
        }
    }
    // Sand - direct insertion, no randomness
    for i in 6..10 {
        sim.particles.list.push(Particle::new(
            Vec2::new(i as f32 + 0.5, 10.5),
            Vec2::new(0.0, -1.0),
            ParticleMaterial::Sand,
        ));
    }
    sim
}

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// ============= REGRESSION TESTS =============

/// P2G transfer must produce identical grid velocities
#[test]
fn regression_p2g_transfer() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();

    let u_sum: f32 = sim.grid.u.iter().sum();
    let v_sum: f32 = sim.grid.v.iter().sum();
    let u_max: f32 = sim.grid.u.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_max: f32 = sim.grid.v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        approx_eq(u_sum, GOLDEN_P2G_U_SUM, TOLERANCE),
        "P2G u_sum changed: {} vs golden {}", u_sum, GOLDEN_P2G_U_SUM
    );
    assert!(
        approx_eq(v_sum, GOLDEN_P2G_V_SUM, TOLERANCE),
        "P2G v_sum changed: {} vs golden {}", v_sum, GOLDEN_P2G_V_SUM
    );
    assert!(
        approx_eq(u_max, GOLDEN_P2G_U_MAX, TOLERANCE),
        "P2G u_max changed: {} vs golden {}", u_max, GOLDEN_P2G_U_MAX
    );
    assert!(
        approx_eq(v_max, GOLDEN_P2G_V_MAX, TOLERANCE),
        "P2G v_max changed: {} vs golden {}", v_max, GOLDEN_P2G_V_MAX
    );
}

/// Pressure solver must converge to same solution
#[test]
fn regression_pressure_solve() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();
    sim.grid.compute_divergence();

    let div_before = sim.grid.total_divergence();
    assert!(
        approx_eq(div_before, GOLDEN_DIV_BEFORE, TOLERANCE),
        "Divergence before changed: {} vs golden {}", div_before, GOLDEN_DIV_BEFORE
    );

    sim.grid.solve_pressure(100);
    sim.grid.apply_pressure_gradient(DT);
    sim.grid.compute_divergence();

    let div_after = sim.grid.total_divergence();
    let pressure_sum: f32 = sim.grid.pressure.iter().sum();
    let (p_min, p_max, p_avg) = sim.grid.pressure_stats();

    assert!(
        approx_eq(div_after, GOLDEN_DIV_AFTER, TOLERANCE),
        "Divergence after changed: {} vs golden {}", div_after, GOLDEN_DIV_AFTER
    );
    assert!(
        approx_eq(pressure_sum, GOLDEN_PRESSURE_SUM, LOOSE_TOLERANCE),
        "Pressure sum changed: {} vs golden {}", pressure_sum, GOLDEN_PRESSURE_SUM
    );
    assert!(
        approx_eq(p_min, GOLDEN_P_MIN, LOOSE_TOLERANCE),
        "P_min changed: {} vs golden {}", p_min, GOLDEN_P_MIN
    );
    assert!(
        approx_eq(p_max, GOLDEN_P_MAX, LOOSE_TOLERANCE),
        "P_max changed: {} vs golden {}", p_max, GOLDEN_P_MAX
    );
    assert!(
        approx_eq(p_avg, GOLDEN_P_AVG, LOOSE_TOLERANCE),
        "P_avg changed: {} vs golden {}", p_avg, GOLDEN_P_AVG
    );
}

/// Full FLIP cycle (P2G + store_old + G2P) must preserve behavior
#[test]
fn regression_flip_cycle() {
    let mut sim = create_deterministic_sim();

    let vel_before: Vec2 = sim.particles.iter()
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |a, b| a + b);

    assert!(
        approx_eq(vel_before.x, GOLDEN_FLIP_VEL_BEFORE_X, TOLERANCE),
        "FLIP vel_before.x changed: {} vs golden {}", vel_before.x, GOLDEN_FLIP_VEL_BEFORE_X
    );
    assert!(
        approx_eq(vel_before.y, GOLDEN_FLIP_VEL_BEFORE_Y, TOLERANCE),
        "FLIP vel_before.y changed: {} vs golden {}", vel_before.y, GOLDEN_FLIP_VEL_BEFORE_Y
    );

    sim.run_isolated_flip_cycle_with_extrapolation(DT);

    let vel_after: Vec2 = sim.particles.iter()
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |a, b| a + b);

    assert!(
        approx_eq(vel_after.x, GOLDEN_FLIP_VEL_AFTER_X, TOLERANCE),
        "FLIP vel_after.x changed: {} vs golden {}", vel_after.x, GOLDEN_FLIP_VEL_AFTER_X
    );
    assert!(
        approx_eq(vel_after.y, GOLDEN_FLIP_VEL_AFTER_Y, TOLERANCE),
        "FLIP vel_after.y changed: {} vs golden {}", vel_after.y, GOLDEN_FLIP_VEL_AFTER_Y
    );
}

/// Velocity extrapolation must affect same cells
#[test]
fn regression_extrapolation() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();

    let u_before: Vec<f32> = sim.grid.u.clone();
    sim.grid.extrapolate_velocities(3);

    let changed: usize = sim.grid.u.iter().zip(u_before.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-10)
        .count();
    let u_sum_after: f32 = sim.grid.u.iter().sum();

    assert_eq!(
        changed, GOLDEN_EXTRAP_CHANGED_COUNT,
        "Extrapolation changed count differs: {} vs golden {}", changed, GOLDEN_EXTRAP_CHANGED_COUNT
    );
    assert!(
        approx_eq(u_sum_after, GOLDEN_EXTRAP_U_SUM_AFTER, TOLERANCE),
        "Extrapolation u_sum_after changed: {} vs golden {}", u_sum_after, GOLDEN_EXTRAP_U_SUM_AFTER
    );
}

/// Vorticity computation must match
#[test]
fn regression_vorticity() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();
    sim.grid.compute_vorticity();

    let enstrophy = sim.grid.compute_enstrophy();
    let max_vort = sim.grid.max_vorticity();
    let total_vort = sim.grid.total_absolute_vorticity();

    assert!(
        approx_eq(enstrophy, GOLDEN_ENSTROPHY, TOLERANCE),
        "Enstrophy changed: {} vs golden {}", enstrophy, GOLDEN_ENSTROPHY
    );
    assert!(
        approx_eq(max_vort, GOLDEN_MAX_VORTICITY, TOLERANCE),
        "Max vorticity changed: {} vs golden {}", max_vort, GOLDEN_MAX_VORTICITY
    );
    assert!(
        approx_eq(total_vort, GOLDEN_TOTAL_VORTICITY, TOLERANCE),
        "Total vorticity changed: {} vs golden {}", total_vort, GOLDEN_TOTAL_VORTICITY
    );
}

/// Full simulation step must produce identical output
#[test]
fn regression_full_step() {
    let mut sim = create_deterministic_sim();
    sim.update(DT);

    let ke = sim.compute_kinetic_energy();
    let max_v = sim.max_velocity();
    let particle_count = sim.particles.len();

    let pos_sum: Vec2 = sim.particles.iter()
        .map(|p| p.position)
        .fold(Vec2::ZERO, |a, b| a + b);
    let vel_sum: Vec2 = sim.particles.iter()
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |a, b| a + b);

    assert!(
        approx_eq(ke, GOLDEN_STEP_KE, LOOSE_TOLERANCE),
        "KE changed: {} vs golden {}", ke, GOLDEN_STEP_KE
    );
    assert!(
        approx_eq(max_v, GOLDEN_STEP_MAX_V, LOOSE_TOLERANCE),
        "Max velocity changed: {} vs golden {}", max_v, GOLDEN_STEP_MAX_V
    );
    assert_eq!(
        particle_count, GOLDEN_STEP_PARTICLE_COUNT,
        "Particle count changed: {} vs golden {}", particle_count, GOLDEN_STEP_PARTICLE_COUNT
    );
    assert!(
        approx_eq(pos_sum.x, GOLDEN_STEP_POS_SUM_X, LOOSE_TOLERANCE),
        "Position sum X changed: {} vs golden {}", pos_sum.x, GOLDEN_STEP_POS_SUM_X
    );
    assert!(
        approx_eq(pos_sum.y, GOLDEN_STEP_POS_SUM_Y, LOOSE_TOLERANCE),
        "Position sum Y changed: {} vs golden {}", pos_sum.y, GOLDEN_STEP_POS_SUM_Y
    );
    assert!(
        approx_eq(vel_sum.x, GOLDEN_STEP_VEL_SUM_X, LOOSE_TOLERANCE),
        "Velocity sum X changed: {} vs golden {}", vel_sum.x, GOLDEN_STEP_VEL_SUM_X
    );
    assert!(
        approx_eq(vel_sum.y, GOLDEN_STEP_VEL_SUM_Y, LOOSE_TOLERANCE),
        "Velocity sum Y changed: {} vs golden {}", vel_sum.y, GOLDEN_STEP_VEL_SUM_Y
    );
}

/// Sediment settling must produce same deposits
#[test]
fn regression_sediment_settling() {
    let mut sim = create_sediment_sim();

    for _ in 0..50 {
        sim.update(DT);
    }

    let deposited_count: usize = (0..sim.grid.width).map(|i| {
        (0..sim.grid.height).filter(|&j| sim.grid.is_deposited(i, j)).count()
    }).sum();

    let sand_count = sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Sand)
        .count();

    assert_eq!(
        deposited_count, GOLDEN_DEPOSITED_CELLS,
        "Deposited cell count changed: {} vs golden {}", deposited_count, GOLDEN_DEPOSITED_CELLS
    );
    assert_eq!(
        sand_count, GOLDEN_REMAINING_SAND,
        "Remaining sand count changed: {} vs golden {}", sand_count, GOLDEN_REMAINING_SAND
    );
}

/// Multi-step stability - run 100 steps and verify no NaN/Inf
#[test]
fn regression_stability_100_steps() {
    let mut sim = create_deterministic_sim();

    for step in 0..100 {
        sim.update(DT);

        // Verify no NaN/Inf at each step
        for (i, p) in sim.particles.iter().enumerate() {
            assert!(
                p.position.is_finite(),
                "Step {}: Particle {} position is not finite: {:?}", step, i, p.position
            );
            assert!(
                p.velocity.is_finite(),
                "Step {}: Particle {} velocity is not finite: {:?}", step, i, p.velocity
            );
        }

        // Grid should also be finite
        for (i, &u) in sim.grid.u.iter().enumerate() {
            assert!(u.is_finite(), "Step {}: Grid u[{}] is not finite: {}", step, i, u);
        }
        for (i, &v) in sim.grid.v.iter().enumerate() {
            assert!(v.is_finite(), "Step {}: Grid v[{}] is not finite: {}", step, i, v);
        }
    }

    // After 100 steps, should still have particles
    assert!(sim.particles.len() > 0, "All particles lost after 100 steps");
}

/// SDF computation must produce identical distance field
#[test]
fn regression_sdf_computation() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.grid.compute_sdf();

    // Sample at key positions
    let sdf_at_boundary = sim.grid.sample_sdf(Vec2::new(0.5, 0.5));
    let sdf_fluid_center = sim.grid.sample_sdf(Vec2::new(8.0, 8.0));
    let sdf_fluid_edge = sim.grid.sample_sdf(Vec2::new(4.0, 4.0));
    let sdf_air_region = sim.grid.sample_sdf(Vec2::new(14.0, 14.0));

    // Gradient near boundary
    let grad = sim.grid.sdf_gradient(Vec2::new(1.0, 1.0));

    // Sum of finite SDF values
    let sdf_sum: f32 = sim.grid.sdf.iter().filter(|x| x.is_finite()).sum();

    assert!(
        approx_eq(sdf_at_boundary, GOLDEN_SDF_AT_BOUNDARY, TOLERANCE),
        "SDF at boundary changed: {} vs golden {}", sdf_at_boundary, GOLDEN_SDF_AT_BOUNDARY
    );
    assert!(
        approx_eq(sdf_fluid_center, GOLDEN_SDF_FLUID_CENTER, TOLERANCE),
        "SDF at fluid center changed: {} vs golden {}", sdf_fluid_center, GOLDEN_SDF_FLUID_CENTER
    );
    assert!(
        approx_eq(sdf_fluid_edge, GOLDEN_SDF_FLUID_EDGE, TOLERANCE),
        "SDF at fluid edge changed: {} vs golden {}", sdf_fluid_edge, GOLDEN_SDF_FLUID_EDGE
    );
    assert!(
        approx_eq(sdf_air_region, GOLDEN_SDF_AIR_REGION, TOLERANCE),
        "SDF at air region changed: {} vs golden {}", sdf_air_region, GOLDEN_SDF_AIR_REGION
    );
    assert!(
        approx_eq(grad.x, GOLDEN_SDF_GRAD_X, TOLERANCE),
        "SDF gradient X changed: {} vs golden {}", grad.x, GOLDEN_SDF_GRAD_X
    );
    assert!(
        approx_eq(grad.y, GOLDEN_SDF_GRAD_Y, TOLERANCE),
        "SDF gradient Y changed: {} vs golden {}", grad.y, GOLDEN_SDF_GRAD_Y
    );
    assert!(
        approx_eq(sdf_sum, GOLDEN_SDF_FINITE_SUM, LOOSE_TOLERANCE),
        "SDF sum changed: {} vs golden {}", sdf_sum, GOLDEN_SDF_FINITE_SUM
    );
}

/// Interpolation kernels must remain unchanged (critical for P2G/G2P)
#[test]
fn regression_interpolation_kernels() {
    // 1D kernel values
    let kernel_0 = quadratic_bspline_1d(0.0);
    let kernel_0_5 = quadratic_bspline_1d(0.5);
    let kernel_1 = quadratic_bspline_1d(1.0);
    let kernel_1_5 = quadratic_bspline_1d(1.5);

    // 2D kernel values
    let kernel_2d_center = quadratic_bspline(Vec2::new(0.0, 0.0));
    let kernel_2d_offset = quadratic_bspline(Vec2::new(0.5, 0.5));

    // Kernel sum over 3x3 grid (must be 1.0 for conservation)
    let mut sum = 0.0;
    for i in -1..=1 {
        for j in -1..=1 {
            sum += quadratic_bspline(Vec2::new(i as f32 + 0.25, j as f32 + 0.25));
        }
    }

    assert!(
        approx_eq(kernel_0, GOLDEN_KERNEL_0, TOLERANCE),
        "Kernel(0) changed: {} vs golden {}", kernel_0, GOLDEN_KERNEL_0
    );
    assert!(
        approx_eq(kernel_0_5, GOLDEN_KERNEL_0_5, TOLERANCE),
        "Kernel(0.5) changed: {} vs golden {}", kernel_0_5, GOLDEN_KERNEL_0_5
    );
    assert!(
        approx_eq(kernel_1, GOLDEN_KERNEL_1, TOLERANCE),
        "Kernel(1.0) changed: {} vs golden {}", kernel_1, GOLDEN_KERNEL_1
    );
    assert!(
        approx_eq(kernel_1_5, GOLDEN_KERNEL_1_5, TOLERANCE),
        "Kernel(1.5) changed: {} vs golden {}", kernel_1_5, GOLDEN_KERNEL_1_5
    );
    assert!(
        approx_eq(kernel_2d_center, GOLDEN_KERNEL_2D_CENTER, TOLERANCE),
        "2D kernel center changed: {} vs golden {}", kernel_2d_center, GOLDEN_KERNEL_2D_CENTER
    );
    assert!(
        approx_eq(kernel_2d_offset, GOLDEN_KERNEL_2D_OFFSET, TOLERANCE),
        "2D kernel offset changed: {} vs golden {}", kernel_2d_offset, GOLDEN_KERNEL_2D_OFFSET
    );
    assert!(
        approx_eq(sum, GOLDEN_KERNEL_SUM_GRID, TOLERANCE),
        "Kernel sum over grid changed: {} vs golden {} (must be 1.0 for conservation!)",
        sum, GOLDEN_KERNEL_SUM_GRID
    );
}
