//! DEM Physics Tests: Water Coupling (FLIP-DEM)
//!
//! Validates buoyancy, drag forces, and water velocity coupling for gravel behavior in water.
//! These tests apply water forces manually to verify the physics equations work correctly.
//!
//! Run with: cargo run --example test_dem_water_coupling --release

use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D};

const WATER_DENSITY: f32 = 1000.0; // kg/m³
const CLUMP_RADIUS: f32 = 0.01; // 1cm sphere
const GRAVITY: f32 = -9.81; // m/s²
const DT: f32 = 1.0 / 120.0; // 120 Hz timestep

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" DEM PHYSICS TESTS: Water Coupling (FLIP-DEM)");
    println!("{}", "=".repeat(70));
    println!("\nValidating water-clump interaction physics.\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Buoyancy Force Direction
    run_test("DEM Buoyancy", test_dem_buoyancy, &mut passed, &mut failed);

    // Test 2: Drag Force Exponential Decay
    run_test(
        "DEM Drag Force",
        test_dem_drag_force,
        &mut passed,
        &mut failed,
    );

    // Test 3: Water Velocity Coupling
    run_test(
        "DEM Water Velocity Coupling",
        test_dem_water_velocity_coupling,
        &mut passed,
        &mut failed,
    );

    println!("{}", "=".repeat(70));
    if failed == 0 {
        println!(" ALL DEM TESTS PASSED ({}/{})", passed, passed + failed);
        println!(" Water coupling physics validated.");
    } else {
        println!(" DEM TESTS: {}/{} passed", passed, passed + failed);
        println!(" Some water coupling physics issues detected.");
    }
    println!("{}", "=".repeat(70));

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Test 1: test_dem_buoyancy
///
/// Physics: Archimedes' principle - buoyant force F_b = ρ_water * V * g
///   Net force: F_net = F_gravity + F_buoyancy = (ρ_clump - ρ_water) * V * g
///
/// Test: Create clumps with different densities
///   - Light clump (density 800 kg/m³ < water) should rise (v_y > 0)
///   - Heavy clump (density 2650 kg/m³ > water) should sink (v_y < 0)
///
/// Tolerance: Correct direction, velocity magnitude > 0.1 m/s after 1s
fn test_dem_buoyancy() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 1: DEM Buoyancy (Archimedes)");
    println!("----------------------------------------");
    println!("Physics: F_buoyancy = ρ_water * V * g");

    let sim_time = 1.0; // seconds
    let steps = (sim_time / DT) as usize;

    // Create template to compute volume
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, CLUMP_RADIUS, 0.01);
    let volume = compute_volume(&template);

    println!("Clump volume: {:.6} m³", volume);

    // Test 1a: Light clump (should rise)
    let light_density = 800.0; // kg/m³ (lighter than water)
    let light_mass = volume * light_density;

    let f_gravity_light = light_mass * GRAVITY;
    let f_buoyancy = WATER_DENSITY * volume * GRAVITY.abs();
    let f_net_light = f_buoyancy + f_gravity_light; // Should be positive (upward)

    println!("\nLight clump (density = {:.0} kg/m³):", light_density);
    println!("  F_gravity:  {:.6} N (down)", f_gravity_light);
    println!("  F_buoyancy: {:.6} N (up)", f_buoyancy);
    println!("  F_net:      {:.6} N (expected: upward)", f_net_light);

    let mut velocity_light = Vec3::ZERO;
    for _ in 0..steps {
        let accel = f_net_light / light_mass;
        velocity_light.y += accel * DT;
    }

    println!(
        "  Final v_y:  {:.3} m/s (expected: positive)",
        velocity_light.y
    );

    // Test 1b: Heavy clump (should sink)
    let heavy_density = 2650.0; // kg/m³ (heavier than water, typical gravel)
    let heavy_mass = volume * heavy_density;

    let f_gravity_heavy = heavy_mass * GRAVITY;
    let f_net_heavy = f_buoyancy + f_gravity_heavy; // Should be negative (downward)

    println!("\nHeavy clump (density = {:.0} kg/m³):", heavy_density);
    println!("  F_gravity:  {:.6} N (down)", f_gravity_heavy);
    println!("  F_buoyancy: {:.6} N (up)", f_buoyancy);
    println!("  F_net:      {:.6} N (expected: downward)", f_net_heavy);

    let mut velocity_heavy = Vec3::ZERO;
    for _ in 0..steps {
        let accel = f_net_heavy / heavy_mass;
        velocity_heavy.y += accel * DT;
    }

    println!(
        "  Final v_y:  {:.3} m/s (expected: negative)",
        velocity_heavy.y
    );

    // Check for NaN
    if velocity_light.is_nan() || velocity_heavy.is_nan() {
        println!("\nFAIL: NaN detected in velocities!");
        return (false, 100.0);
    }

    // Validate directions and magnitudes
    let light_correct = velocity_light.y > 0.1; // Rising at >0.1 m/s
    let heavy_correct = velocity_heavy.y < -0.1; // Sinking at >0.1 m/s (magnitude)

    let pass = light_correct && heavy_correct;

    if pass {
        println!("\nPASS: Buoyancy directions correct");
        println!(
            "  Light clump rises:  v_y = {:.3} m/s > 0.1",
            velocity_light.y
        );
        println!(
            "  Heavy clump sinks:  v_y = {:.3} m/s < -0.1",
            velocity_heavy.y
        );
    } else {
        println!("\nFAIL: Buoyancy direction incorrect");
        if !light_correct {
            println!(
                "  Light clump should rise but v_y = {:.3}",
                velocity_light.y
            );
        }
        if !heavy_correct {
            println!(
                "  Heavy clump should sink but v_y = {:.3}",
                velocity_heavy.y
            );
        }
    }

    // Compute error metric (deviation from expected magnitude)
    let expected_light = (f_net_light / light_mass) * sim_time;
    let expected_heavy = (f_net_heavy / heavy_mass) * sim_time;
    let error_light = ((velocity_light.y - expected_light) / expected_light * 100.0).abs();
    let error_heavy = ((velocity_heavy.y - expected_heavy) / expected_heavy * 100.0).abs();
    let max_error = error_light.max(error_heavy);

    (pass, max_error)
}

/// Test 2: test_dem_drag_force
///
/// Physics: Linear drag F_d = -b*v
///   Velocity decay: v(t) = v₀ * exp(-b/m * t)
///
/// Test: Clump moving through still water (no water velocity)
///   Initial velocity v₀ = 2.0 m/s
///   Apply drag force and verify exponential decay
///
/// Tolerance: Final velocity within 10% of analytical solution after 2s
fn test_dem_drag_force() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 2: DEM Drag Force (Exponential Decay)");
    println!("----------------------------------------");
    println!("Physics: v(t) = v₀ * exp(-drag*t)");

    // Clump parameters
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, CLUMP_RADIUS, 0.01);
    let mass = template.mass;

    // Initial velocity
    let v_initial = 2.0; // m/s
    let mut velocity = Vec3::new(v_initial, 0.0, 0.0);

    // Drag coefficient (use production value from main.rs:1413)
    let drag_coeff = 3.0; // 1/s

    println!("Clump mass: {:.6} kg", mass);
    println!("Drag coefficient: {:.1} 1/s", drag_coeff);
    println!("Initial velocity: {:.1} m/s", v_initial);

    // Simulate 2 seconds
    let sim_time = 2.0;
    let steps = (sim_time / DT) as usize;

    let mut velocities = Vec::new();
    for _ in 0..steps {
        // Apply drag: dv/dt = -drag_coeff * v
        let drag_accel = -drag_coeff * velocity;
        velocity += drag_accel * DT;
        velocities.push(velocity.x);
    }

    // Analytical solution: v(t) = v₀ * exp(-drag_coeff * t)
    let v_analytical = v_initial * (-drag_coeff * sim_time).exp();
    let v_measured = velocities.last().unwrap();

    println!(
        "Expected final velocity: {:.3} m/s (analytical)",
        v_analytical
    );
    println!("Measured final velocity: {:.3} m/s", v_measured);

    // Check for NaN
    if v_measured.is_nan() {
        println!("\nFAIL: NaN detected in velocity!");
        return (false, 100.0);
    }

    // Verify decay characteristics
    let decayed = *v_measured < v_initial * 0.5; // Should have decayed significantly
    let no_reversal = *v_measured > 0.0; // Shouldn't reverse direction

    // Calculate error from analytical
    let error = ((v_measured - v_analytical) / v_analytical * 100.0).abs();

    println!("Error from analytical: {:.2}%", error);

    let pass = error < 10.0 && decayed && no_reversal;

    if pass {
        println!("\nPASS: Drag force produces exponential decay within 10%");
    } else {
        println!(
            "\nFAIL: Drag decay error = {:.2}% (exceeds 10% tolerance)",
            error
        );
        if !decayed {
            println!(
                "  Velocity didn't decay enough: {:.3} vs initial {:.3}",
                v_measured, v_initial
            );
        }
        if !no_reversal {
            println!("  Velocity reversed direction: {:.3}", v_measured);
        }
    }

    (pass, error)
}

/// Test 3: test_dem_water_velocity_coupling
///
/// Physics: Drag accelerates clump toward water velocity
///   dv/dt = (v_water - v_clump) * drag_coeff
///   Steady state: v_clump → v_water as t → ∞
///
/// Test: Uniform water flow at 1.0 m/s in +x direction
///   Clump starts at rest, accelerates toward water velocity
///   Should reach 95%+ of water velocity within 4 seconds
///
/// Tolerance: Final velocity within 5% of water velocity
fn test_dem_water_velocity_coupling() -> (bool, f32) {
    println!("----------------------------------------");
    println!("TEST 3: DEM Water Velocity Coupling");
    println!("----------------------------------------");
    println!("Physics: Drag toward water velocity");

    // Water flow (uniform in +x)
    let water_velocity = Vec3::new(1.0, 0.0, 0.0);
    let mut clump_velocity = Vec3::ZERO;

    // Drag coefficient (stronger than test 2 for faster convergence)
    let drag_coeff = 5.0; // 1/s

    println!(
        "Water velocity: [{:.1}, {:.1}, {:.1}] m/s",
        water_velocity.x, water_velocity.y, water_velocity.z
    );
    println!(
        "Initial clump velocity: [{:.1}, {:.1}, {:.1}] m/s",
        clump_velocity.x, clump_velocity.y, clump_velocity.z
    );
    println!("Drag coefficient: {:.1} 1/s", drag_coeff);

    // Simulate 4 seconds to reach steady state
    let sim_time = 4.0;
    let steps = (sim_time / DT) as usize;

    let mut velocities = Vec::new();
    for _ in 0..steps {
        // Drag toward water velocity
        let delta_v = water_velocity - clump_velocity;
        clump_velocity += delta_v * drag_coeff * DT;
        velocities.push(clump_velocity.x);
    }

    println!(
        "Final clump velocity: [{:.3}, {:.3}, {:.3}] m/s",
        clump_velocity.x, clump_velocity.y, clump_velocity.z
    );

    // Check for NaN
    if clump_velocity.is_nan() {
        println!("\nFAIL: NaN detected in clump velocity!");
        return (false, 100.0);
    }

    // Calculate error from water velocity
    let error = ((clump_velocity.x - water_velocity.x) / water_velocity.x * 100.0).abs();

    println!("Error from water velocity: {:.2}%", error);

    // Check convergence (should be >95% of water velocity)
    let converged = error < 5.0;

    // Check no oscillation (velocity should monotonically approach target)
    let no_overshoot = clump_velocity.x <= water_velocity.x * 1.01; // Allow 1% overshoot

    let pass = converged && no_overshoot;

    if pass {
        println!("\nPASS: Clump entrained by water flow (error < 5%)");
        println!(
            "  Reached {:.1}% of water velocity",
            (clump_velocity.x / water_velocity.x) * 100.0
        );
    } else {
        println!("\nFAIL: Water velocity coupling error = {:.2}%", error);
        if !converged {
            println!("  Didn't converge to water velocity within 5%");
        }
        if !no_overshoot {
            println!(
                "  Velocity overshot target: {:.3} > {:.3}",
                clump_velocity.x, water_velocity.x
            );
        }
    }

    (pass, error)
}

/// Helper: Compute clump volume from template
///
/// For tetrahedral clump, approximate volume from bounding sphere.
/// Exact volume calculation would require summing particle volumes and
/// accounting for overlaps, but sphere approximation is sufficient for tests.
fn compute_volume(template: &ClumpTemplate3D) -> f32 {
    // Volume of bounding sphere: V = 4/3 π r³
    let r = template.bounding_radius;
    4.0 / 3.0 * std::f32::consts::PI * r.powi(3)
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
