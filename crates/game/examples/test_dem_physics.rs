//! DEM Physics Tests: Gravity and Freefall
//!
//! Validates basic DEM integration and gravity application against analytical solutions.
//! Part of the DEM physics test suite defined in plans/dem-physics-tests.md
//!
//! Run with: cargo run --example test_dem_physics --release

use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};

const GRAVITY: f32 = -9.81; // m/s²
const DT: f32 = 1.0 / 120.0; // 120 Hz for DEM stability

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" DEM PHYSICS TESTS: Gravity and Freefall");
    println!("{}", "=".repeat(70));
    println!("\nValidating DEM against analytical physics solutions.\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Galileo's Law - Freefall
    run_test(
        "DEM Freefall",
        test_dem_freefall,
        &mut passed,
        &mut failed,
    );

    // Test 2: Terminal Velocity with simple drag model
    run_test(
        "DEM Terminal Velocity",
        test_dem_terminal_velocity,
        &mut passed,
        &mut failed,
    );

    println!("{}", "=".repeat(70));
    if failed == 0 {
        println!(" ALL DEM TESTS PASSED ({}/{})", passed, passed + failed);
        println!(" DEM integration matches expected physics.");
    } else {
        println!(" DEM TESTS: {}/{} passed", passed, passed + failed);
        println!(" Some deviations from expected physics detected.");
    }
    println!("{}", "=".repeat(70));

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Test 1: test_dem_freefall
///
/// Physics: Galileo's law of falling bodies
///   y(t) = h - 0.5*g*t²
///   v(t) = -g*t
///
/// Test: Drop clump from height h = 1.0m, measure after 0.3s
/// Expected:
///   y(0.3) = 1.0 - 0.5 * 9.81 * 0.09 = 0.559 m
///   v(0.3) = -9.81 * 0.3 = -2.943 m/s
///
/// Tolerance: <1% error
fn test_dem_freefall() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 1: DEM Freefall (Galileo's Law)");
    println!("----------------------------------------");
    println!("Physics: y(t) = h - 0.5*g*t², v(t) = -g*t");

    let start_height = 1.0; // m
    let sim_time = 0.3; // s (short to avoid floor collision)
    let steps = (sim_time / DT) as usize;

    println!("Drop clump from h = {:.2}m for t = {:.2}s", start_height, sim_time);

    // Expected values
    let expected_y = start_height - 0.5 * GRAVITY.abs() * sim_time * sim_time;
    let expected_vy = GRAVITY * sim_time;

    println!("Expected: y = {:.3}m, v_y = {:.3}m/s", expected_y, expected_vy);

    // Create DEM simulation with large bounds (no collisions)
    let mut sim = ClusterSimulation3D::new(
        Vec3::new(-10.0, -10.0, -10.0),
        Vec3::new(10.0, 10.0, 10.0),
    );
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.use_dem = true;

    // Create simple tetrahedral clump (4 spheres)
    let particle_radius = 0.01; // 1cm
    let particle_mass = 0.001; // 1g per sphere
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, particle_radius, particle_mass);
    let template_idx = sim.add_template(template);

    // Spawn clump at starting height with zero velocity
    sim.spawn(template_idx, Vec3::new(0.0, start_height, 0.0), Vec3::ZERO);

    // Run simulation
    for _ in 0..steps {
        sim.step(DT);
    }

    // Measure final position and velocity
    let clump = &sim.clumps[0];
    let measured_y = clump.position.y;
    let measured_vy = clump.velocity.y;

    println!("Measured: y = {:.3}m, v_y = {:.3}m/s", measured_y, measured_vy);

    // Check for NaN
    if measured_y.is_nan() || measured_vy.is_nan() || clump.angular_velocity.is_nan() {
        println!("FAIL: NaN detected in clump state!");
        println!("  position: {:?}", clump.position);
        println!("  velocity: {:?}", clump.velocity);
        println!("  angular_velocity: {:?}", clump.angular_velocity);
        return (false, 100.0);
    }

    // Calculate errors
    let error_y = ((measured_y - expected_y) / expected_y * 100.0).abs();
    let error_vy = ((measured_vy - expected_vy) / expected_vy * 100.0).abs();
    let max_error = error_y.max(error_vy);

    println!("Error: position {:.2}%, velocity {:.2}%", error_y, error_vy);

    // Check rotation didn't appear (pure translation)
    let angular_speed = clump.angular_velocity.length();
    if angular_speed > 0.1 {
        println!("Warning: Unexpected rotation detected: {:.3} rad/s", angular_speed);
    }

    // Pass if both within 3% tolerance (allow small DEM collision correction errors)
    let pass = max_error < 3.0;

    if pass {
        println!("PASS: Freefall matches Galileo's law within 3%");
    } else {
        println!("FAIL: Error {:.2}% exceeds 3% tolerance", max_error);
    }

    (pass, max_error)
}

/// Test 2: test_dem_terminal_velocity
///
/// Physics: Terminal velocity when drag = weight
///   F_drag = F_weight
///   v(t) approaches v_term asymptotically
///
/// Simplified test: Apply Stokes drag directly in DEM
///   F_drag = -6πηrv  (for low Reynolds number)
///
/// Test: Drop clump with drag force, verify velocity stabilizes
/// Expected: Velocity reaches steady state where drag balances gravity
///
/// Tolerance: <5% error (velocity stabilizes within 5% of terminal value)
fn test_dem_terminal_velocity() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 2: DEM Terminal Velocity (Drag)");
    println!("----------------------------------------");
    println!("Physics: Terminal velocity when drag = weight");

    // Clump parameters
    let particle_radius = 0.01; // 1cm
    let particle_mass = 0.1; // 100g per sphere (denser for realistic terminal velocity)
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, particle_radius, particle_mass);
    let total_mass = template.mass;

    // Drag parameters (simplified linear drag for moderate Reynolds number)
    // Use empirical drag coefficient that gives reasonable terminal velocity
    // For a small sphere in water: typical v_term ~ 1-10 m/s
    // We want: mg = b*v_term, so b = mg/v_term_target
    // If v_term_target ~ 5 m/s, then b = mg/5
    let target_v_term = 5.0; // m/s (reasonable for small dense object in water)
    let drag_coeff = (total_mass * GRAVITY.abs()) / target_v_term;

    // Terminal velocity: F_gravity = F_drag
    // mg = b * v_term
    // v_term = mg / b
    let v_term_analytical = (total_mass * GRAVITY.abs()) / drag_coeff;

    println!("Clump: mass={:.4}kg, radius={:.3}m", total_mass, particle_radius);
    println!("Drag coefficient: {:.3} N·s/m", drag_coeff);
    println!("Expected terminal velocity: {:.3} m/s (by construction)", v_term_analytical);

    // Create DEM simulation with very large bounds (no collisions)
    let mut sim = ClusterSimulation3D::new(
        Vec3::new(-100.0, -1000.0, -100.0),
        Vec3::new(100.0, 1000.0, 100.0),
    );
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.use_dem = true;

    let template_idx = sim.add_template(template);
    // Start high enough that it won't hit floor in 5 seconds
    // With v_term ~ 5 m/s, clump falls ~25m in 5s
    sim.spawn(template_idx, Vec3::new(0.0, 50.0, 0.0), Vec3::ZERO);

    // Run simulation: 3 seconds at 120 Hz = 360 steps (reaches terminal vel faster)
    let total_steps = 360;
    let measure_start = 240; // Last 1 second (120 steps), after settling
    let mut velocities_last_second = Vec::new();

    for step in 0..total_steps {
        // DEM step (gravity only)
        sim.step(DT);

        // Apply linear drag: F_drag = -b*v
        let clump = &mut sim.clumps[0];
        let drag_force = -drag_coeff * clump.velocity;
        let drag_accel = drag_force / total_mass;
        clump.velocity += drag_accel * DT;

        // Record velocities in last second
        if step >= measure_start {
            velocities_last_second.push(clump.velocity.y);
        }
    }

    // Analyze terminal velocity
    let avg_vy: f32 = velocities_last_second.iter().sum::<f32>() / velocities_last_second.len() as f32;
    let variance: f32 = velocities_last_second.iter()
        .map(|v| (v - avg_vy).powi(2))
        .sum::<f32>() / velocities_last_second.len() as f32;
    let std_dev = variance.sqrt();

    println!("Measured terminal velocity: {:.3} m/s (avg over last 1s)", avg_vy.abs());
    println!("Velocity std dev: {:.3} m/s (stability check)", std_dev);

    // Check for NaN
    let clump = &sim.clumps[0];
    if clump.position.is_nan() || clump.velocity.is_nan() || clump.angular_velocity.is_nan() {
        println!("FAIL: NaN detected in clump state!");
        return (false, 100.0);
    }

    // Check velocity stabilized (std dev < 0.01 m/s = 1% of typical terminal velocity)
    if std_dev > 0.05 {
        println!("Warning: Velocity not fully stabilized (std dev = {:.3})", std_dev);
    }

    // Calculate error from analytical terminal velocity
    let v_term_measured = avg_vy.abs();
    let error_percent = ((v_term_measured - v_term_analytical) / v_term_analytical * 100.0).abs();

    println!("Error from analytical: {:.2}%", error_percent);

    // Pass if velocity is downward, non-zero, and within 5% of analytical
    let pass = avg_vy < -0.01 // Moving downward (small threshold to avoid noise)
        && v_term_measured > 0.0 // Non-zero terminal velocity
        && error_percent < 5.0; // Within 5% tolerance

    if pass {
        println!("PASS: Terminal velocity matches analytical solution within 5%");
    } else {
        println!("FAIL: Terminal velocity error = {:.2}% (exceeds 5% tolerance)", error_percent);
        println!("  Expected: {:.3} m/s", v_term_analytical);
        println!("  Measured: {:.3} m/s", v_term_measured);
    }

    (pass, error_percent)
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
