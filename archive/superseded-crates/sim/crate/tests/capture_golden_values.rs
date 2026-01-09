//! Run this ONCE before refactoring to capture golden values.
//!
//! ```bash
//! cargo test -p sim capture_golden -- --nocapture --ignored
//! ```
//!
//! Then copy the printed values into refactor_regression.rs

use sim::flip::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial};
use glam::Vec2;

const DT: f32 = 1.0 / 60.0;

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

#[test]
#[ignore] // Run manually with --ignored flag
fn capture_golden_p2g() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();

    let u_sum: f32 = sim.grid.u.iter().sum();
    let v_sum: f32 = sim.grid.v.iter().sum();
    let u_max: f32 = sim.grid.u.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v_max: f32 = sim.grid.v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n=== P2G Golden Values ===");
    println!("const GOLDEN_P2G_U_SUM: f32 = {:.10};", u_sum);
    println!("const GOLDEN_P2G_V_SUM: f32 = {:.10};", v_sum);
    println!("const GOLDEN_P2G_U_MAX: f32 = {:.10};", u_max);
    println!("const GOLDEN_P2G_V_MAX: f32 = {:.10};", v_max);
}

#[test]
#[ignore]
fn capture_golden_pressure_solve() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();
    sim.grid.compute_divergence();

    let div_before = sim.grid.total_divergence();

    sim.grid.solve_pressure(100);
    sim.grid.apply_pressure_gradient(DT);
    sim.grid.compute_divergence();

    let div_after = sim.grid.total_divergence();
    let pressure_sum: f32 = sim.grid.pressure.iter().sum();
    let (p_min, p_max, p_avg) = sim.grid.pressure_stats();

    println!("\n=== Pressure Solve Golden Values ===");
    println!("const GOLDEN_DIV_BEFORE: f32 = {:.10};", div_before);
    println!("const GOLDEN_DIV_AFTER: f32 = {:.10};", div_after);
    println!("const GOLDEN_PRESSURE_SUM: f32 = {:.10};", pressure_sum);
    println!("const GOLDEN_P_MIN: f32 = {:.10};", p_min);
    println!("const GOLDEN_P_MAX: f32 = {:.10};", p_max);
    println!("const GOLDEN_P_AVG: f32 = {:.10};", p_avg);
}

#[test]
#[ignore]
fn capture_golden_flip_cycle() {
    let mut sim = create_deterministic_sim();

    // Capture particle velocities before FLIP cycle
    let vel_before: Vec2 = sim.particles.iter()
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |a, b| a + b);

    // Run isolated FLIP cycle (P2G + store_old + G2P)
    sim.run_isolated_flip_cycle_with_extrapolation(DT);

    let vel_after: Vec2 = sim.particles.iter()
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |a, b| a + b);

    println!("\n=== FLIP Cycle Golden Values ===");
    println!("const GOLDEN_FLIP_VEL_BEFORE_X: f32 = {:.10};", vel_before.x);
    println!("const GOLDEN_FLIP_VEL_BEFORE_Y: f32 = {:.10};", vel_before.y);
    println!("const GOLDEN_FLIP_VEL_AFTER_X: f32 = {:.10};", vel_after.x);
    println!("const GOLDEN_FLIP_VEL_AFTER_Y: f32 = {:.10};", vel_after.y);
}

#[test]
#[ignore]
fn capture_golden_extrapolation() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();

    let u_before: Vec<f32> = sim.grid.u.clone();

    sim.grid.extrapolate_velocities(3);

    // Count how many values changed
    let changed: usize = sim.grid.u.iter().zip(u_before.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-10)
        .count();

    let u_sum_after: f32 = sim.grid.u.iter().sum();

    println!("\n=== Extrapolation Golden Values ===");
    println!("const GOLDEN_EXTRAP_CHANGED_COUNT: usize = {};", changed);
    println!("const GOLDEN_EXTRAP_U_SUM_AFTER: f32 = {:.10};", u_sum_after);
}

#[test]
#[ignore]
fn capture_golden_vorticity() {
    let mut sim = create_deterministic_sim();
    sim.classify_cells();
    sim.particles_to_grid();

    sim.grid.compute_vorticity();

    let enstrophy = sim.grid.compute_enstrophy();
    let max_vort = sim.grid.max_vorticity();
    let total_vort = sim.grid.total_absolute_vorticity();

    println!("\n=== Vorticity Golden Values ===");
    println!("const GOLDEN_ENSTROPHY: f32 = {:.10};", enstrophy);
    println!("const GOLDEN_MAX_VORTICITY: f32 = {:.10};", max_vort);
    println!("const GOLDEN_TOTAL_VORTICITY: f32 = {:.10};", total_vort);
}

#[test]
#[ignore]
fn capture_golden_full_step() {
    let mut sim = create_deterministic_sim();

    // Run one full timestep
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

    println!("\n=== Full Step Golden Values ===");
    println!("const GOLDEN_STEP_KE: f32 = {:.10};", ke);
    println!("const GOLDEN_STEP_MAX_V: f32 = {:.10};", max_v);
    println!("const GOLDEN_STEP_PARTICLE_COUNT: usize = {};", particle_count);
    println!("const GOLDEN_STEP_POS_SUM_X: f32 = {:.10};", pos_sum.x);
    println!("const GOLDEN_STEP_POS_SUM_Y: f32 = {:.10};", pos_sum.y);
    println!("const GOLDEN_STEP_VEL_SUM_X: f32 = {:.10};", vel_sum.x);
    println!("const GOLDEN_STEP_VEL_SUM_Y: f32 = {:.10};", vel_sum.y);
}

#[test]
#[ignore]
fn capture_golden_sediment_deposit() {
    let mut sim = create_sediment_sim();

    // Run several steps to allow settling
    for _ in 0..50 {
        sim.update(DT);
    }

    let deposited_count: usize = (0..sim.grid.width).map(|i| {
        (0..sim.grid.height).filter(|&j| sim.grid.is_deposited(i, j)).count()
    }).sum();

    let sand_count = sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Sand)
        .count();

    println!("\n=== Sediment Golden Values ===");
    println!("const GOLDEN_DEPOSITED_CELLS: usize = {};", deposited_count);
    println!("const GOLDEN_REMAINING_SAND: usize = {};", sand_count);
}

#[test]
#[ignore]
fn capture_golden_sdf() {
    // Use the standard deterministic sim - it has boundaries and compute_sdf is called in update
    let mut sim = create_deterministic_sim();

    // Classify cells and compute SDF (same as update() does)
    sim.classify_cells();
    sim.grid.compute_sdf();

    // Sample at key positions
    // Boundary cell (should be negative or small)
    let sdf_at_boundary = sim.grid.sample_sdf(Vec2::new(0.5, 0.5));
    // Inside fluid region (should be positive)
    let sdf_fluid_center = sim.grid.sample_sdf(Vec2::new(8.0, 8.0));
    // Near fluid edge
    let sdf_fluid_edge = sim.grid.sample_sdf(Vec2::new(4.0, 4.0));
    // Far from particles (air region)
    let sdf_air_region = sim.grid.sample_sdf(Vec2::new(14.0, 14.0));

    // Gradient near boundary should point inward
    let grad_at_boundary = sim.grid.sdf_gradient(Vec2::new(1.0, 1.0));

    // Sum of all SDF values (stability check)
    let sdf_sum: f32 = sim.grid.sdf.iter().filter(|x| x.is_finite()).sum();

    println!("\n=== SDF Golden Values ===");
    println!("const GOLDEN_SDF_AT_BOUNDARY: f32 = {:.10};", sdf_at_boundary);
    println!("const GOLDEN_SDF_FLUID_CENTER: f32 = {:.10};", sdf_fluid_center);
    println!("const GOLDEN_SDF_FLUID_EDGE: f32 = {:.10};", sdf_fluid_edge);
    println!("const GOLDEN_SDF_AIR_REGION: f32 = {:.10};", sdf_air_region);
    println!("const GOLDEN_SDF_GRAD_X: f32 = {:.10};", grad_at_boundary.x);
    println!("const GOLDEN_SDF_GRAD_Y: f32 = {:.10};", grad_at_boundary.y);
    println!("const GOLDEN_SDF_FINITE_SUM: f32 = {:.10};", sdf_sum);
}

#[test]
#[ignore]
fn capture_golden_interpolation() {
    use sim::grid::{quadratic_bspline_1d, quadratic_bspline};

    // B-spline kernel values at specific offsets
    let kernel_0 = quadratic_bspline_1d(0.0);
    let kernel_0_5 = quadratic_bspline_1d(0.5);
    let kernel_1 = quadratic_bspline_1d(1.0);
    let kernel_1_5 = quadratic_bspline_1d(1.5);

    // 2D kernel at center
    let kernel_2d_center = quadratic_bspline(Vec2::new(0.0, 0.0));
    let kernel_2d_offset = quadratic_bspline(Vec2::new(0.5, 0.5));

    // Verify kernel sums to 1 over support (critical for conservation)
    let mut sum = 0.0;
    for i in -1..=1 {
        for j in -1..=1 {
            sum += quadratic_bspline(Vec2::new(i as f32 + 0.25, j as f32 + 0.25));
        }
    }

    println!("\n=== Interpolation Golden Values ===");
    println!("const GOLDEN_KERNEL_0: f32 = {:.10};", kernel_0);
    println!("const GOLDEN_KERNEL_0_5: f32 = {:.10};", kernel_0_5);
    println!("const GOLDEN_KERNEL_1: f32 = {:.10};", kernel_1);
    println!("const GOLDEN_KERNEL_1_5: f32 = {:.10};", kernel_1_5);
    println!("const GOLDEN_KERNEL_2D_CENTER: f32 = {:.10};", kernel_2d_center);
    println!("const GOLDEN_KERNEL_2D_OFFSET: f32 = {:.10};", kernel_2d_offset);
    println!("const GOLDEN_KERNEL_SUM_GRID: f32 = {:.10};", sum);
}

#[test]
#[ignore]
fn capture_all_golden_values() {
    // Run all captures in sequence
    println!("\n\n========================================");
    println!("        GOLDEN VALUES FOR REFACTOR");
    println!("========================================");
    println!("Copy these into refactor_regression.rs\n");

    capture_golden_p2g();
    capture_golden_pressure_solve();
    capture_golden_flip_cycle();
    capture_golden_extrapolation();
    capture_golden_vorticity();
    capture_golden_full_step();
    capture_golden_sediment_deposit();
    capture_golden_sdf();
    capture_golden_interpolation();
    // Note: advection and spatial_hash use private methods,
    // so they are tested indirectly via full_step and stability tests

    println!("\n========================================\n");
}

// Note: advection and spatial_hash tests removed because they use private methods.
// These are tested indirectly via:
// - regression_full_step (tests entire update() pipeline including advection)
// - regression_stability_100_steps (exercises spatial hash through push_particles_apart)
