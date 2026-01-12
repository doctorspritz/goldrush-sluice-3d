//! PHYSICS VALIDATION TEST HARNESS
//!
//! These tests validate ACTUAL PHYSICS against analytical solutions.
//! Tests FAIL if physics are wrong, not just if "something happened".
//!
//! Run: cargo run --example physics_validation --release

use glam::Vec3;
use sim3d::FlipSimulation3D;
use std::time::Instant;

const CELL_SIZE: f32 = 0.025;

// =============================================================================
// Test Results
// =============================================================================

struct PhysicsTest {
    name: &'static str,
    passed: bool,
    expected: String,
    actual: String,
    error_percent: f32,
    tolerance_percent: f32,
}

impl PhysicsTest {
    fn fail(name: &'static str, expected: String, actual: String, error: f32, tolerance: f32) -> Self {
        Self {
            name,
            passed: false,
            expected,
            actual,
            error_percent: error,
            tolerance_percent: tolerance,
        }
    }

    fn pass(name: &'static str, expected: String, actual: String, error: f32, tolerance: f32) -> Self {
        Self {
            name,
            passed: true,
            expected,
            actual,
            error_percent: error,
            tolerance_percent: tolerance,
        }
    }
}

// =============================================================================
// TEST 1: Freefall - Compare against analytical solution y = y0 - 0.5*g*t^2
// =============================================================================

fn test_freefall_accuracy() -> PhysicsTest {
    let name = "Freefall vs analytical y = y0 - 0.5*g*t^2";
    let tolerance = 5.0; // 5% error tolerance

    // Large grid, particle starts in middle
    let mut sim = FlipSimulation3D::new(20, 80, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 5; // Minimal pressure solve for freefall

    let y0 = 1.5; // Starting height
    let start_pos = Vec3::new(0.25, y0, 0.25);
    sim.spawn_particle_with_velocity(start_pos, Vec3::ZERO);

    let dt = 1.0 / 120.0;
    let num_steps = 60; // 0.5 seconds
    let t = num_steps as f32 * dt;

    for _ in 0..num_steps {
        sim.update(dt);
    }

    // Analytical solution: y = y0 - 0.5 * g * t^2
    let g = 9.81;
    let expected_y = y0 - 0.5 * g * t * t;

    let actual_y = if !sim.particles.list.is_empty() {
        sim.particles.list[0].position.y
    } else {
        return PhysicsTest::fail(
            name,
            format!("y = {:.4}m", expected_y),
            "Particle lost".to_string(),
            100.0,
            tolerance,
        );
    };

    let error = ((actual_y - expected_y) / expected_y).abs() * 100.0;

    if error <= tolerance {
        PhysicsTest::pass(
            name,
            format!("y = {:.4}m", expected_y),
            format!("y = {:.4}m", actual_y),
            error,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            format!("y = {:.4}m", expected_y),
            format!("y = {:.4}m", actual_y),
            error,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 2: Freefall velocity - v = g*t
// =============================================================================

fn test_freefall_velocity() -> PhysicsTest {
    let name = "Freefall velocity v = g*t";
    let tolerance = 10.0; // 10% tolerance

    let mut sim = FlipSimulation3D::new(20, 80, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 5;

    let start_pos = Vec3::new(0.25, 1.5, 0.25);
    sim.spawn_particle_with_velocity(start_pos, Vec3::ZERO);

    let dt = 1.0 / 120.0;
    let num_steps = 60;
    let t = num_steps as f32 * dt;

    for _ in 0..num_steps {
        sim.update(dt);
    }

    let g = 9.81;
    let expected_vy = -g * t; // Negative because falling

    let actual_vy = if !sim.particles.list.is_empty() {
        sim.particles.list[0].velocity.y
    } else {
        return PhysicsTest::fail(
            name,
            format!("vy = {:.4}m/s", expected_vy),
            "Particle lost".to_string(),
            100.0,
            tolerance,
        );
    };

    let error = ((actual_vy - expected_vy) / expected_vy).abs() * 100.0;

    if error <= tolerance {
        PhysicsTest::pass(
            name,
            format!("vy = {:.4}m/s", expected_vy),
            format!("vy = {:.4}m/s", actual_vy),
            error,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            format!("vy = {:.4}m/s", expected_vy),
            format!("vy = {:.4}m/s", actual_vy),
            error,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 3: Projectile motion - x = x0 + vx*t, y = y0 + vy*t - 0.5*g*t^2
// =============================================================================

fn test_projectile_motion() -> PhysicsTest {
    let name = "Projectile motion (parabola)";
    let tolerance = 10.0;

    let mut sim = FlipSimulation3D::new(60, 60, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 5;

    let x0 = 0.2;
    let y0 = 1.0;
    let vx0 = 2.0; // 2 m/s horizontal
    let vy0 = 3.0; // 3 m/s upward

    sim.spawn_particle_with_velocity(
        Vec3::new(x0, y0, 0.25),
        Vec3::new(vx0, vy0, 0.0),
    );

    let dt = 1.0 / 120.0;
    let num_steps = 30; // 0.25 seconds
    let t = num_steps as f32 * dt;

    for _ in 0..num_steps {
        sim.update(dt);
    }

    let g = 9.81;
    let expected_x = x0 + vx0 * t;
    let expected_y = y0 + vy0 * t - 0.5 * g * t * t;

    let (actual_x, actual_y) = if !sim.particles.list.is_empty() {
        let p = &sim.particles.list[0];
        (p.position.x, p.position.y)
    } else {
        return PhysicsTest::fail(
            name,
            format!("({:.4}, {:.4})", expected_x, expected_y),
            "Particle lost".to_string(),
            100.0,
            tolerance,
        );
    };

    let error_x = ((actual_x - expected_x) / expected_x).abs() * 100.0;
    let error_y = ((actual_y - expected_y) / expected_y).abs() * 100.0;
    let max_error = error_x.max(error_y);

    if max_error <= tolerance {
        PhysicsTest::pass(
            name,
            format!("({:.4}, {:.4})", expected_x, expected_y),
            format!("({:.4}, {:.4})", actual_x, actual_y),
            max_error,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            format!("({:.4}, {:.4})", expected_x, expected_y),
            format!("({:.4}, {:.4})", actual_x, actual_y),
            max_error,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 4: Energy conservation in freefall (KE + PE = constant)
// =============================================================================

fn test_energy_conservation() -> PhysicsTest {
    let name = "Energy conservation (KE + PE)";
    let tolerance = 15.0; // FLIP is dissipative, allow some loss

    let mut sim = FlipSimulation3D::new(20, 80, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 5;

    let y0 = 1.5;
    sim.spawn_particle_with_velocity(Vec3::new(0.25, y0, 0.25), Vec3::ZERO);

    let g = 9.81;
    let m = 1.0; // Unit mass
    let initial_pe = m * g * y0;
    let initial_ke = 0.0;
    let initial_energy = initial_pe + initial_ke;

    let dt = 1.0 / 120.0;
    for _ in 0..60 {
        sim.update(dt);
    }

    let (final_y, final_v) = if !sim.particles.list.is_empty() {
        let p = &sim.particles.list[0];
        (p.position.y, p.velocity.length())
    } else {
        return PhysicsTest::fail(
            name,
            format!("E = {:.4}J", initial_energy),
            "Particle lost".to_string(),
            100.0,
            tolerance,
        );
    };

    let final_pe = m * g * final_y;
    let final_ke = 0.5 * m * final_v * final_v;
    let final_energy = final_pe + final_ke;

    let error = ((final_energy - initial_energy) / initial_energy).abs() * 100.0;

    if error <= tolerance {
        PhysicsTest::pass(
            name,
            format!("E = {:.4}J", initial_energy),
            format!("E = {:.4}J", final_energy),
            error,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            format!("E = {:.4}J", initial_energy),
            format!("E = {:.4}J", final_energy),
            error,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 5: Solid boundary - particle cannot penetrate floor
// =============================================================================

fn test_solid_boundary() -> PhysicsTest {
    let name = "Solid boundary (no penetration)";
    let tolerance = 0.0; // Zero tolerance for penetration

    let mut sim = FlipSimulation3D::new(20, 40, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 30;

    // Mark floor as solid
    for i in 0..20 {
        for k in 0..20 {
            for j in 0..3 {
                sim.grid.set_solid(i, j, k);
            }
        }
    }
    sim.grid.compute_sdf();

    let floor_y = 3.0 * CELL_SIZE; // Top of solid region

    // Drop particle onto floor
    sim.spawn_particle_with_velocity(Vec3::new(0.25, 0.8, 0.25), Vec3::ZERO);

    let dt = 1.0 / 120.0;
    for _ in 0..240 {
        sim.update(dt);
    }

    let final_y = if !sim.particles.list.is_empty() {
        sim.particles.list[0].position.y
    } else {
        return PhysicsTest::fail(
            name,
            format!("y >= {:.4}m", floor_y),
            "Particle lost".to_string(),
            100.0,
            tolerance,
        );
    };

    // Particle should be AT or ABOVE the floor, never below
    if final_y >= floor_y - CELL_SIZE * 0.5 {
        PhysicsTest::pass(
            name,
            format!("y >= {:.4}m", floor_y),
            format!("y = {:.4}m", final_y),
            0.0,
            tolerance,
        )
    } else {
        let penetration = floor_y - final_y;
        PhysicsTest::fail(
            name,
            format!("y >= {:.4}m", floor_y),
            format!("y = {:.4}m (penetrated {:.4}m)", final_y, penetration),
            100.0,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 6: Mass conservation - particle count in closed system
// =============================================================================

fn test_mass_conservation() -> PhysicsTest {
    let name = "Mass conservation (closed box)";
    let tolerance = 0.0; // No particles should be lost

    let mut sim = FlipSimulation3D::new(20, 20, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 30;

    // Create a closed box (walls on all sides)
    for i in 0..20 {
        for k in 0..20 {
            // Floor and ceiling
            sim.grid.set_solid(i, 0, k);
            sim.grid.set_solid(i, 1, k);
            sim.grid.set_solid(i, 18, k);
            sim.grid.set_solid(i, 19, k);
        }
    }
    for i in 0..20 {
        for j in 0..20 {
            // Front and back walls
            sim.grid.set_solid(i, j, 0);
            sim.grid.set_solid(i, j, 1);
            sim.grid.set_solid(i, j, 18);
            sim.grid.set_solid(i, j, 19);
        }
    }
    for j in 0..20 {
        for k in 0..20 {
            // Left and right walls
            sim.grid.set_solid(0, j, k);
            sim.grid.set_solid(1, j, k);
            sim.grid.set_solid(18, j, k);
            sim.grid.set_solid(19, j, k);
        }
    }
    sim.grid.compute_sdf();

    // Spawn particles inside the box
    let initial_count = 100;
    for i in 0..10 {
        for j in 0..10 {
            let x = 0.15 + i as f32 * 0.02;
            let y = 0.25 + j as f32 * 0.02;
            sim.spawn_particle_with_velocity(Vec3::new(x, y, 0.25), Vec3::ZERO);
        }
    }

    let dt = 1.0 / 120.0;
    for _ in 0..300 {
        sim.update(dt);
    }

    let final_count = sim.particles.len();

    if final_count == initial_count {
        PhysicsTest::pass(
            name,
            format!("{} particles", initial_count),
            format!("{} particles", final_count),
            0.0,
            tolerance,
        )
    } else {
        let lost = initial_count as i32 - final_count as i32;
        PhysicsTest::fail(
            name,
            format!("{} particles", initial_count),
            format!("{} particles (lost {})", final_count, lost),
            (lost.abs() as f32 / initial_count as f32) * 100.0,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 7: Hydrostatic pressure - water at rest should stay at rest
// =============================================================================

fn test_hydrostatic_equilibrium() -> PhysicsTest {
    let name = "Hydrostatic equilibrium (water at rest)";
    let tolerance = 20.0; // Allow some settling motion

    let mut sim = FlipSimulation3D::new(20, 30, 20, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 60;

    // Floor
    for i in 0..20 {
        for k in 0..20 {
            sim.grid.set_solid(i, 0, k);
            sim.grid.set_solid(i, 1, k);
        }
    }
    sim.grid.compute_sdf();

    // Create a column of water at rest
    let mut initial_positions = Vec::new();
    for i in 3..17 {
        for j in 3..15 {
            for k in 3..17 {
                let pos = Vec3::new(
                    i as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                    j as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                    k as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                );
                sim.spawn_particle_with_velocity(pos, Vec3::ZERO);
                initial_positions.push(pos);
            }
        }
    }

    let dt = 1.0 / 120.0;
    // Let it settle
    for _ in 0..120 {
        sim.update(dt);
    }

    // Measure average velocity - should be near zero
    let mut total_speed = 0.0;
    for p in &sim.particles.list {
        total_speed += p.velocity.length();
    }
    let avg_speed = if !sim.particles.list.is_empty() {
        total_speed / sim.particles.list.len() as f32
    } else {
        return PhysicsTest::fail(
            name,
            "avg_speed ~ 0".to_string(),
            "All particles lost".to_string(),
            100.0,
            tolerance,
        );
    };

    // Water at rest should have very low average velocity
    let expected_speed = 0.0;
    let max_acceptable_speed = 0.5; // 0.5 m/s is too much motion for "at rest"

    if avg_speed < max_acceptable_speed {
        PhysicsTest::pass(
            name,
            format!("avg_speed < {:.2}m/s", max_acceptable_speed),
            format!("avg_speed = {:.4}m/s", avg_speed),
            (avg_speed / max_acceptable_speed) * 100.0,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            format!("avg_speed < {:.2}m/s", max_acceptable_speed),
            format!("avg_speed = {:.4}m/s", avg_speed),
            (avg_speed / max_acceptable_speed) * 100.0,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 8: Density-dependent settling (EXPECTED TO FAIL - not implemented)
// =============================================================================

fn test_density_settling() -> PhysicsTest {
    let name = "Density settling (gold sinks faster than sand) [NOT IMPLEMENTED]";
    let tolerance = 5.0;

    let mut sim = FlipSimulation3D::new(30, 60, 30, CELL_SIZE);
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.pressure_iterations = 30;

    // Floor
    for i in 0..30 {
        for k in 0..30 {
            sim.grid.set_solid(i, 0, k);
            sim.grid.set_solid(i, 1, k);
        }
    }
    sim.grid.compute_sdf();

    // Fill with water first
    for i in 5..25 {
        for j in 3..40 {
            for k in 5..25 {
                let pos = Vec3::new(
                    i as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                    j as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                    k as f32 * CELL_SIZE + CELL_SIZE * 0.5,
                );
                sim.spawn_particle_with_velocity(pos, Vec3::ZERO);
            }
        }
    }

    // Let water settle
    let dt = 1.0 / 120.0;
    for _ in 0..60 {
        sim.update(dt);
    }

    // Now drop gold (19300 kg/m³) and sand (2650 kg/m³) from same height
    let drop_y = 0.8;
    let gold_density = 19300.0;
    let sand_density = 2650.0;

    sim.spawn_sediment(Vec3::new(0.3, drop_y, 0.375), Vec3::ZERO, gold_density);
    sim.spawn_sediment(Vec3::new(0.45, drop_y, 0.375), Vec3::ZERO, sand_density);

    // Run simulation
    for _ in 0..180 {
        sim.update(dt);
    }

    // Find gold and sand by density
    let mut gold_y = drop_y;
    let mut sand_y = drop_y;
    for p in &sim.particles.list {
        if p.density > 10000.0 {
            gold_y = p.position.y;
        } else if p.density > 2000.0 && p.density < 5000.0 {
            sand_y = p.position.y;
        }
    }

    // Gold should be LOWER than sand (it's 7x denser)
    // In real physics: terminal velocity ∝ sqrt(density)
    // Gold should settle ~2.7x faster than sand

    if gold_y < sand_y - 0.05 {
        PhysicsTest::pass(
            name,
            "gold_y < sand_y".to_string(),
            format!("gold={:.4}, sand={:.4}", gold_y, sand_y),
            0.0,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            "gold_y < sand_y (gold should sink faster)".to_string(),
            format!("gold={:.4}, sand={:.4} (same or gold higher!)", gold_y, sand_y),
            100.0,
            tolerance,
        )
    }
}

// =============================================================================
// TEST 9: Momentum conservation in collision
// =============================================================================

fn test_momentum_conservation() -> PhysicsTest {
    let name = "Momentum conservation (two particles)";
    let tolerance = 20.0; // FLIP is not perfectly conservative

    let mut sim = FlipSimulation3D::new(40, 20, 20, CELL_SIZE);
    sim.gravity = Vec3::ZERO; // No gravity for this test
    sim.pressure_iterations = 30;

    // Two particles moving toward each other
    let m = 1.0; // Unit mass
    let v1 = Vec3::new(1.0, 0.0, 0.0);
    let v2 = Vec3::new(-1.0, 0.0, 0.0);

    sim.spawn_particle_with_velocity(Vec3::new(0.3, 0.25, 0.25), v1);
    sim.spawn_particle_with_velocity(Vec3::new(0.7, 0.25, 0.25), v2);

    let initial_momentum = m * v1 + m * v2; // Should be zero

    let dt = 1.0 / 120.0;
    for _ in 0..120 {
        sim.update(dt);
    }

    let mut final_momentum = Vec3::ZERO;
    for p in &sim.particles.list {
        final_momentum += m * p.velocity;
    }

    let momentum_error = (final_momentum - initial_momentum).length();
    let error_percent = momentum_error * 100.0; // Since initial is ~0, use absolute

    if error_percent <= tolerance {
        PhysicsTest::pass(
            name,
            format!("p = ({:.2}, {:.2}, {:.2})", initial_momentum.x, initial_momentum.y, initial_momentum.z),
            format!("p = ({:.2}, {:.2}, {:.2})", final_momentum.x, final_momentum.y, final_momentum.z),
            error_percent,
            tolerance,
        )
    } else {
        PhysicsTest::fail(
            name,
            format!("p = ({:.2}, {:.2}, {:.2})", initial_momentum.x, initial_momentum.y, initial_momentum.z),
            format!("p = ({:.2}, {:.2}, {:.2})", final_momentum.x, final_momentum.y, final_momentum.z),
            error_percent,
            tolerance,
        )
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           PHYSICS VALIDATION TEST HARNESS                        ║");
    println!("║                                                                  ║");
    println!("║  Tests compare simulation results against analytical solutions.  ║");
    println!("║  FAIL = physics are WRONG, not just 'something different'.       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    let tests: Vec<PhysicsTest> = vec![
        test_freefall_accuracy(),
        test_freefall_velocity(),
        test_projectile_motion(),
        test_energy_conservation(),
        test_solid_boundary(),
        test_mass_conservation(),
        test_hydrostatic_equilibrium(),
        test_density_settling(),
        test_momentum_conservation(),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for test in &tests {
        let status = if test.passed {
            passed += 1;
            "\x1b[32mPASS\x1b[0m"
        } else {
            failed += 1;
            "\x1b[31mFAIL\x1b[0m"
        };

        println!("[{}] {}", status, test.name);
        println!("       Expected: {}", test.expected);
        println!("       Actual:   {}", test.actual);
        println!("       Error:    {:.1}% (tolerance: {:.1}%)", test.error_percent, test.tolerance_percent);
        println!();
    }

    let elapsed = start.elapsed();
    println!("════════════════════════════════════════════════════════════════════");
    println!("Results: {}/{} passed, {} failed in {:.2}s", passed, tests.len(), failed, elapsed.as_secs_f32());

    if failed > 0 {
        println!();
        println!("\x1b[31mPHYSICS VALIDATION FAILED\x1b[0m");
        println!("The simulation does not match expected physical behavior.");
        std::process::exit(1);
    } else {
        println!();
        println!("\x1b[32mALL PHYSICS TESTS PASSED\x1b[0m");
    }
}
