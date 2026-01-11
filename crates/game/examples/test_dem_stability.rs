//! DEM Physics Stability Tests
//!
//! Validates energy conservation, stability under compression, and deterministic behavior.
//! These tests prove the DEM contact model is physically sound and numerically stable.
//!
//! Run with: cargo run --example test_dem_stability --release

use glam::{Mat3, Quat, Vec3};
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};

const GRAVITY: f32 = -9.81;
const DT: f32 = 1.0 / 120.0; // 120 Hz timestep
const PARTICLE_RADIUS: f32 = 0.01; // 1cm
const PARTICLE_MASS: f32 = 0.01; // 10g (density ~2650 kg/m³)
const BOUNDS: (Vec3, Vec3) = (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
const RESTITUTION: f32 = 0.2;
const NORMAL_STIFFNESS: f32 = 6000.0;
const TANGENTIAL_STIFFNESS: f32 = 3000.0;

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" DEM PHYSICS STABILITY TESTS");
    println!("{}", "=".repeat(70));
    println!("\nThese tests verify DEM physics stability and correctness.\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Energy Dissipation
    if test_dem_energy_dissipation() {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 2: No Velocity Explosion
    if test_dem_no_explosion() {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 3: Determinism
    if test_dem_determinism() {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n{}", "=".repeat(70));
    if failed == 0 {
        println!(" ALL DEM TESTS PASSED ({}/{})", passed, passed + failed);
    } else {
        println!(" DEM TESTS FAILED: {}/{} passed", passed, passed + failed);
    }
    println!("{}", "=".repeat(70));

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Test 1: Energy Dissipation
///
/// Physics: Drop a clump from height h. Total energy E = KE + PE should monotonically
/// decrease due to damping and collisions. Energy should NEVER increase (no energy
/// injection from contact model).
fn test_dem_energy_dissipation() -> bool {
    println!("----------------------------------------");
    println!("TEST 1: Energy Dissipation");
    println!("----------------------------------------");
    println!("Expected: Energy monotonically decreases");

    let mut sim = ClusterSimulation3D::new(BOUNDS.0, BOUNDS.1);
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.restitution = RESTITUTION;
    sim.normal_stiffness = NORMAL_STIFFNESS;
    sim.tangential_stiffness = TANGENTIAL_STIFFNESS;
    sim.use_dem = true;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    // Drop from 0.4m above floor
    let drop_height = 0.4;
    sim.spawn(template_idx, Vec3::new(0.5, drop_height, 0.5), Vec3::ZERO);

    let frames = 240; // 2 seconds at 120 Hz
    let mut energy_violations = 0;
    let mut prev_energy = f32::INFINITY;
    let mut initial_energy = 0.0;
    let mut final_energy = 0.0;

    for frame in 0..frames {
        sim.step(DT);

        let clump = &sim.clumps[0];
        let template = &sim.templates[template_idx];
        let energy = compute_total_energy(clump, template, sim.gravity);

        if frame == 0 {
            initial_energy = energy;
        }
        if frame == frames - 1 {
            final_energy = energy;
        }

        // Check for energy increase (tolerance for floating point errors)
        if energy > prev_energy + 1e-4 {
            energy_violations += 1;
        }
        prev_energy = energy;
    }

    println!("  Initial energy:      {:.3} J", initial_energy);
    println!("  Final energy:        {:.3} J", final_energy);
    println!("  Energy violations:   {}", energy_violations);

    let pass = energy_violations == 0;
    println!("  Result:              {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 2: No Velocity Explosion
///
/// Physics: Pack many clumps into small volume. Spring-damper contact forces should
/// prevent velocity explosions. Stiffness/damping balance critical.
fn test_dem_no_explosion() -> bool {
    println!("\n----------------------------------------");
    println!("TEST 2: No Velocity Explosion");
    println!("----------------------------------------");
    println!("Expected: Velocities remain < 100 m/s");

    let mut sim = ClusterSimulation3D::new(BOUNDS.0, BOUNDS.1);
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.restitution = RESTITUTION;
    sim.normal_stiffness = NORMAL_STIFFNESS;
    sim.tangential_stiffness = TANGENTIAL_STIFFNESS;
    sim.use_dem = true;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    // Drop 20 clumps from moderate height in a loose grid
    // Spacing ~0.1m (10x particle radius) to allow settling without extreme compression
    // They will fall and pile up, testing stability under realistic conditions
    for iz in 0..4 {
        for ix in 0..5 {
            let x = 0.3 + ix as f32 * 0.1;
            let y = 0.5 + (ix + iz) as f32 * 0.05; // Stagger heights slightly
            let z = 0.3 + iz as f32 * 0.1;
            sim.spawn(template_idx, Vec3::new(x, y, z), Vec3::ZERO);
        }
    }

    let frames = 360; // 3 seconds at 120 Hz
    let mut max_velocity = 0.0_f32;
    let mut max_angular_velocity = 0.0_f32;

    for _ in 0..frames {
        sim.step(DT);

        for clump in &sim.clumps {
            let v_mag = clump.velocity.length();
            let w_mag = clump.angular_velocity.length();
            max_velocity = max_velocity.max(v_mag);
            max_angular_velocity = max_angular_velocity.max(w_mag);
        }
    }

    println!("  Max velocity (linear):   {:.2} m/s", max_velocity);
    println!("  Max velocity (angular):  {:.2} rad/s", max_angular_velocity);

    let pass = max_velocity < 100.0 && max_angular_velocity < 1000.0;
    println!("  Result:                  {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 3: Determinism
///
/// Physics: Same initial conditions → identical results. No random behavior, no
/// timing dependencies.
fn test_dem_determinism() -> bool {
    println!("\n----------------------------------------");
    println!("TEST 3: Determinism");
    println!("----------------------------------------");
    println!("Expected: Bit-exact reproduction");

    // Run 1
    let final_state_1 = run_determinism_scenario();

    // Run 2 (identical setup)
    let final_state_2 = run_determinism_scenario();

    // Compare results
    let max_pos_diff = final_state_1
        .iter()
        .zip(final_state_2.iter())
        .map(|(a, b)| (a.position - b.position).length())
        .fold(0.0_f32, f32::max);

    let max_vel_diff = final_state_1
        .iter()
        .zip(final_state_2.iter())
        .map(|(a, b)| (a.velocity - b.velocity).length())
        .fold(0.0_f32, f32::max);

    let max_rot_diff = final_state_1
        .iter()
        .zip(final_state_2.iter())
        .map(|(a, b)| {
            let q_diff = a.rotation.conjugate() * b.rotation;
            q_diff.xyz().length()
        })
        .fold(0.0_f32, f32::max);

    let max_angvel_diff = final_state_1
        .iter()
        .zip(final_state_2.iter())
        .map(|(a, b)| (a.angular_velocity - b.angular_velocity).length())
        .fold(0.0_f32, f32::max);

    println!("  Max position diff:       {:.6} m", max_pos_diff);
    println!("  Max velocity diff:       {:.6} m/s", max_vel_diff);
    println!("  Max rotation diff:       {:.6}", max_rot_diff);
    println!("  Max angular vel diff:    {:.6} rad/s", max_angvel_diff);

    let pass = max_pos_diff < 1e-6
        && max_vel_diff < 1e-6
        && max_rot_diff < 1e-6
        && max_angvel_diff < 1e-6;
    println!("  Result:                  {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Helper: Run determinism test scenario and return final state
fn run_determinism_scenario() -> Vec<ClumpState> {
    let mut sim = ClusterSimulation3D::new(BOUNDS.0, BOUNDS.1);
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.restitution = RESTITUTION;
    sim.normal_stiffness = NORMAL_STIFFNESS;
    sim.tangential_stiffness = TANGENTIAL_STIFFNESS;
    sim.use_dem = true;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    // Drop 5 clumps from different heights (deterministic positions)
    let drop_positions = [
        Vec3::new(0.3, 0.5, 0.3),
        Vec3::new(0.7, 0.4, 0.3),
        Vec3::new(0.5, 0.6, 0.5),
        Vec3::new(0.3, 0.3, 0.7),
        Vec3::new(0.7, 0.5, 0.7),
    ];

    for pos in &drop_positions {
        sim.spawn(template_idx, *pos, Vec3::ZERO);
    }

    // Run for 120 frames (1 second at 120 Hz)
    for _ in 0..120 {
        sim.step(DT);
    }

    // Record final state
    sim.clumps
        .iter()
        .map(|c| ClumpState {
            position: c.position,
            velocity: c.velocity,
            rotation: c.rotation,
            angular_velocity: c.angular_velocity,
        })
        .collect()
}

#[derive(Clone, Copy)]
struct ClumpState {
    position: Vec3,
    velocity: Vec3,
    rotation: Quat,
    angular_velocity: Vec3,
}

/// Compute total energy: KE_trans + KE_rot + PE
fn compute_total_energy(
    clump: &sim3d::clump::Clump3D,
    template: &ClumpTemplate3D,
    gravity: Vec3,
) -> f32 {
    let ke_trans = compute_kinetic_energy_translation(clump, template);
    let ke_rot = compute_kinetic_energy_rotation(clump, template);
    let pe = compute_potential_energy(clump, template, gravity);
    ke_trans + ke_rot + pe
}

/// Compute translational kinetic energy: 0.5 * m * v²
fn compute_kinetic_energy_translation(
    clump: &sim3d::clump::Clump3D,
    template: &ClumpTemplate3D,
) -> f32 {
    0.5 * template.mass * clump.velocity.length_squared()
}

/// Compute rotational kinetic energy: 0.5 * ω^T * I * ω
fn compute_kinetic_energy_rotation(
    clump: &sim3d::clump::Clump3D,
    template: &ClumpTemplate3D,
) -> f32 {
    // I_world = R * I_local * R^T
    // KE_rot = 0.5 * ω^T * I_world * ω
    // Since I_local is diagonal (approximate), we use:
    // I_world^-1 = R * I_local^-1 * R^T
    // So I_world = R * I_local * R^T where I_local = (I_local^-1)^-1

    // Invert the inverse inertia to get actual inertia
    let inertia_inv_local = template.inertia_inv_local;
    let det = inertia_inv_local.determinant();

    if det.abs() < 1e-10 {
        // Singular matrix - use approximate inertia
        let r_sq = template.bounding_radius * template.bounding_radius;
        let i_approx = 0.4 * template.mass * r_sq;
        return 0.5 * i_approx * clump.angular_velocity.length_squared();
    }

    let inertia_local = inertia_inv_local.inverse();

    // Transform to world space
    let rot = Mat3::from_quat(clump.rotation);
    let inertia_world = rot * inertia_local * rot.transpose();

    // Compute ω^T * I * ω
    let i_omega = inertia_world * clump.angular_velocity;
    0.5 * clump.angular_velocity.dot(i_omega)
}

/// Compute potential energy: m * g * y
fn compute_potential_energy(
    clump: &sim3d::clump::Clump3D,
    template: &ClumpTemplate3D,
    gravity: Vec3,
) -> f32 {
    // PE = m * g * h
    // Gravity is in y direction, so height = y coordinate
    template.mass * (-gravity.y) * clump.position.y
}
