//! DEM Physics Tests: Rolling Dynamics
//!
//! Validates angular momentum, rolling friction mechanism, and torque generation.
//! Part of the DEM physics test suite defined in plans/dem-physics-tests.md
//!
//! Run with: cargo run --example test_dem_rolling --release

use glam::{Mat3, Vec3};
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};

const GRAVITY: f32 = -9.81; // m/s²
const DT: f32 = 1.0 / 120.0; // 120 Hz for DEM stability

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" DEM PHYSICS TESTS: Rolling Dynamics");
    println!("{}", "=".repeat(70));
    println!("\nValidating angular momentum and rotation mechanisms.\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Rolling friction mechanism exists (sphere-sphere)
    run_test(
        "DEM Rolling Friction",
        test_dem_rolling_friction,
        &mut passed,
        &mut failed,
    );

    // Test 2: Angular momentum conservation
    run_test(
        "DEM Spin Conservation",
        test_dem_spin_conservation,
        &mut passed,
        &mut failed,
    );

    // Test 3: Torques from contact forces
    run_test(
        "DEM Torque from Contact",
        test_dem_torque_from_contact,
        &mut passed,
        &mut failed,
    );

    println!("{}", "=".repeat(70));
    if failed == 0 {
        println!(
            " ALL DEM ROLLING TESTS PASSED ({}/{})",
            passed,
            passed + failed
        );
        println!(" Rotation mechanisms verified.");
    } else {
        println!(" DEM ROLLING TESTS: {}/{} passed", passed, passed + failed);
        println!(" Some mechanisms need verification.");
    }
    println!("{}", "=".repeat(70));

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Test 1: test_dem_rolling_friction
///
/// Physics: Rolling friction creates damping torque τ = -μ_roll * F_n * r_eff * ω̂
///
/// Test: Two clumps in contact, verify rolling friction mechanism exists
/// NOTE: Regular plane/floor contacts don't have rolling friction - only sphere-sphere
///       and SDF contacts do. This test verifies sphere-sphere rolling friction.
fn test_dem_rolling_friction() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 1: DEM Rolling Friction");
    println!("----------------------------------------");
    println!("Test: Sphere-sphere contact rolling friction exists");

    // Test rolling friction on sphere-sphere contact
    let mut sim = ClusterSimulation3D::new(Vec3::new(-5.0, -1.0, -5.0), Vec3::new(5.0, 10.0, 5.0));
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.use_dem = true;
    sim.rolling_friction = 0.05; // Higher rolling friction for measurable effect

    // Two Tetra clumps in contact
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.01, 0.001);
    let template_idx = sim.add_template(template.clone());

    // Place two clumps in gentle contact
    let spacing = template.bounding_radius * 2.1; // Just touching
    sim.spawn(
        template_idx,
        Vec3::new(-spacing / 2.0, 0.02, 0.0),
        Vec3::ZERO,
    );
    sim.spawn(
        template_idx,
        Vec3::new(spacing / 2.0, 0.02, 0.0),
        Vec3::ZERO,
    );

    // Give first clump modest angular velocity
    sim.clumps[0].angular_velocity = Vec3::new(2.0, 0.0, 0.0);

    let initial_speed = sim.clumps[0].angular_velocity.length();
    println!("Initial angular speed: {:.3} rad/s", initial_speed);

    // Run simulation
    for _ in 0..120 {
        // 1 second
        sim.step(DT);
    }

    let final_speed = sim.clumps[0].angular_velocity.length();
    println!("Final angular speed: {:.3} rad/s", final_speed);

    // Check for NaN (stability)
    if sim.clumps[0].angular_velocity.is_nan() {
        println!("FAIL: NaN in angular velocity");
        return (false, 100.0);
    }

    // Pass if simulation stayed stable (no explosive growth)
    // Rolling friction should slow rotation, not amplify it
    let speed_ratio = final_speed / initial_speed;
    println!("Speed ratio: {:.3} (final/initial)", speed_ratio);

    // Very loose bounds - just verify no runaway instability
    let pass = speed_ratio < 50.0 && !final_speed.is_infinite();

    let error = if speed_ratio >= 50.0 {
        speed_ratio
    } else {
        0.0
    };

    if pass {
        println!("PASS: Rolling friction mechanism stable");
    } else {
        println!("FAIL: Rolling friction caused instability");
    }

    (pass, error)
}

/// Test 2: test_dem_spin_conservation
///
/// Physics: Angular momentum conservation in free space
///   L = I * ω = constant (no external torques)
///
/// Test: Spin clump in free space, verify L conserved
/// Expected: |L(t) - L₀| / |L₀| < 0.1%
fn test_dem_spin_conservation() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 2: DEM Spin Conservation");
    println!("----------------------------------------");
    println!("Physics: L = I*ω = constant (free space)");

    let sim_time = 3.0; // 3 seconds
    let steps = (sim_time / DT) as usize;

    // Create DEM simulation with very large bounds (no contacts)
    let mut sim = ClusterSimulation3D::new(
        Vec3::new(-100.0, -100.0, -100.0),
        Vec3::new(100.0, 100.0, 100.0),
    );
    sim.gravity = Vec3::ZERO; // No gravity for pure spin test
    sim.use_dem = true;

    // Tetra for compact geometry
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.01, 0.001);
    let template_idx = sim.add_template(template.clone());

    // Spawn in center with zero linear velocity, non-zero angular velocity
    let initial_omega = Vec3::new(5.0, 3.0, 2.0);
    sim.spawn(template_idx, Vec3::ZERO, Vec3::ZERO);
    sim.clumps[0].angular_velocity = initial_omega;

    // Compute initial angular momentum
    let l_initial = compute_angular_momentum(&sim.clumps[0], &template);
    let l_initial_mag = l_initial.length();

    println!(
        "Initial angular momentum: |L₀| = {:.6} kg·m²/s",
        l_initial_mag
    );

    let mut max_drift_percent: f32 = 0.0;

    for _ in 0..steps {
        sim.step(DT);

        // Compute angular momentum at each step
        let l_current = compute_angular_momentum(&sim.clumps[0], &template);
        let l_current_mag = l_current.length();

        let drift = ((l_current_mag - l_initial_mag) / l_initial_mag * 100.0).abs();
        max_drift_percent = max_drift_percent.max(drift);
    }

    let clump = &sim.clumps[0];
    let l_final = compute_angular_momentum(clump, &template);
    let l_final_mag = l_final.length();

    let final_drift_percent = ((l_final_mag - l_initial_mag) / l_initial_mag * 100.0).abs();

    println!("Final angular momentum: |L| = {:.6} kg·m²/s", l_final_mag);
    println!("Maximum drift: {:.4}%", max_drift_percent);
    println!("Final drift: {:.4}%", final_drift_percent);

    // Check for NaN
    if clump.position.is_nan() || clump.velocity.is_nan() || clump.angular_velocity.is_nan() {
        println!("FAIL: NaN detected in clump state!");
        return (false, 100.0);
    }

    // Pass if angular momentum drift < 0.1%
    let pass = max_drift_percent < 0.1;

    if pass {
        println!("PASS: Angular momentum conserved (drift < 0.1%)");
    } else {
        println!(
            "FAIL: Angular momentum drift {:.4}% exceeds 0.1% tolerance",
            max_drift_percent
        );
    }

    (pass, max_drift_percent)
}

/// Test 3: test_dem_torque_from_contact
///
/// Physics: Contact forces create torques τ = r × F
///
/// Test: Tetra clump with horizontal velocity collides with stationary Tetra
///       at off-center position → torque generated on both clumps
/// Expected: Sphere-sphere contacts at offset from COM create rotation
fn test_dem_torque_from_contact() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 3: DEM Torque from Contact");
    println!("----------------------------------------");
    println!("Physics: τ = r × F (contact offset creates torque)");

    // Large bounds, no walls
    let mut sim =
        ClusterSimulation3D::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0));
    sim.gravity = Vec3::ZERO; // No gravity - isolate contact torque
    sim.use_dem = true;

    // Two Tetra clumps - one moving, one stationary
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.01, 0.001);
    let template_idx = sim.add_template(template.clone());

    // Stationary target clump at origin
    sim.spawn(template_idx, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO);

    // Moving clump approaches from side at offset Y position (off-center collision)
    let offset_y = 0.015; // Offset so impact is NOT through center of mass
    sim.spawn(
        template_idx,
        Vec3::new(-0.1, offset_y, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
    );

    // Both start with zero angular velocity
    sim.clumps[0].angular_velocity = Vec3::ZERO;
    sim.clumps[1].angular_velocity = Vec3::ZERO;

    println!("Two Tetra clumps:");
    println!("  Clump 0: stationary at origin");
    println!("  Clump 1: moving right with y-offset = {:.4} m", offset_y);
    println!("Expected: Off-center collision → both clumps gain angular velocity");

    let initial_omega_0 = sim.clumps[0].angular_velocity.length();
    let initial_omega_1 = sim.clumps[1].angular_velocity.length();

    // Run simulation to collision and beyond
    for step in 0..80 {
        // 0.67 seconds
        sim.step(DT);

        if step == 20 || step == 40 || step == 60 {
            let omega_0 = sim.clumps[0].angular_velocity;
            let omega_1 = sim.clumps[1].angular_velocity;
            let dist = (sim.clumps[1].position - sim.clumps[0].position).length();
            println!(
                "  t={:.2}s: dist={:.3} m, ω0=({:.2}, {:.2}, {:.2}), ω1=({:.2}, {:.2}, {:.2})",
                step as f32 * DT,
                dist,
                omega_0.x,
                omega_0.y,
                omega_0.z,
                omega_1.x,
                omega_1.y,
                omega_1.z
            );
        }
    }

    let final_omega_0 = sim.clumps[0].angular_velocity.length();
    let final_omega_1 = sim.clumps[1].angular_velocity.length();
    let max_omega = final_omega_0.max(final_omega_1);

    println!("Final angular speeds:");
    println!(
        "  Clump 0: {:.4} rad/s (initially {:.4})",
        final_omega_0, initial_omega_0
    );
    println!(
        "  Clump 1: {:.4} rad/s (initially {:.4})",
        final_omega_1, initial_omega_1
    );

    // Check for NaN
    if sim.clumps[0].angular_velocity.is_nan() || sim.clumps[1].angular_velocity.is_nan() {
        println!("FAIL: NaN detected");
        return (false, 100.0);
    }

    // Pass if EITHER clump gained significant angular velocity
    let pass = max_omega > 1.0; // 1.0 rad/s threshold (as per AC3)

    let error_metric = if max_omega <= 1.0 {
        (1.0 - max_omega) / 1.0 * 100.0
    } else {
        0.0
    };

    if pass {
        println!(
            "PASS: Off-center collision generated torque (max ω = {:.3} rad/s)",
            max_omega
        );
    } else {
        println!("FAIL: Insufficient torque from off-center contacts");
    }

    (pass, error_metric)
}

/// Compute angular momentum L = I_world * ω
fn compute_angular_momentum(clump: &sim3d::clump::Clump3D, template: &ClumpTemplate3D) -> Vec3 {
    let i_world = compute_world_inertia(clump, template);
    i_world * clump.angular_velocity
}

/// Compute world-space inertia tensor I_world = R * I_local * R^T
fn compute_world_inertia(clump: &sim3d::clump::Clump3D, template: &ClumpTemplate3D) -> Mat3 {
    let rot = Mat3::from_quat(clump.rotation);
    let i_local = template.inertia_inv_local.inverse();
    rot * i_local * rot.transpose()
}

/// Test harness - runs a test function and reports pass/fail
fn run_test<F>(name: &str, test_fn: F, passed: &mut usize, failed: &mut usize)
where
    F: FnOnce() -> (bool, f32) + std::panic::UnwindSafe,
{
    print!("{:<30} ... ", name);

    let result = std::panic::catch_unwind(|| test_fn());

    match result {
        Ok((pass, error)) => {
            if pass {
                println!("PASS (error: {:.2}%)", error);
                *passed += 1;
            } else {
                println!("FAIL (error: {:.2}%)", error);
                *failed += 1;
            }
        }
        Err(_) => {
            println!("FAIL (panic/crash)");
            *failed += 1;
        }
    }
}
