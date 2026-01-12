// DEM Friction Physics Tests
// Validates static/kinetic friction, wet vs dry behavior using the ACTUAL code path
//
// These tests use collision_response_only() with SDF - the same method used in
// the real simulation. This ensures we're testing what actually runs in production.

use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, SdfParams};

const PARTICLE_RADIUS: f32 = 0.01; // 1cm gravel
const PARTICLE_MASS: f32 = 0.01; // 10g
const DT: f32 = 1.0 / 120.0; // 120 Hz timestep
const GRAVITY: f32 = -9.81;

// Grid dimensions for SDF
const GRID_WIDTH: usize = 32;
const GRID_HEIGHT: usize = 16;
const GRID_DEPTH: usize = 32;
const CELL_SIZE: f32 = 0.05; // 5cm cells

/// Create a simple floor SDF - negative below y=floor_height, positive above
fn create_floor_sdf(floor_height: f32) -> Vec<f32> {
    let mut sdf = vec![0.0; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];

    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let y = j as f32 * CELL_SIZE;
                let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                // Distance to floor plane (positive above, negative below)
                sdf[idx] = y - floor_height;
            }
        }
    }
    sdf
}

/// Helper to run physics step with SDF collision
fn step_with_sdf_collision(sim: &mut ClusterSimulation3D, sdf: &[f32], dt: f32, wet: bool) {
    // Apply gravity manually (collision_response_only doesn't do integration)
    for clump in &mut sim.clumps {
        clump.velocity += sim.gravity * dt;
        clump.position += clump.velocity * dt;
    }

    let sdf_params = SdfParams {
        sdf,
        grid_width: GRID_WIDTH,
        grid_height: GRID_HEIGHT,
        grid_depth: GRID_DEPTH,
        cell_size: CELL_SIZE,
        grid_offset: Vec3::ZERO,
    };

    sim.collision_response_only(dt, &sdf_params, wet);
}

/// Test 1: Static Friction - Clump at rest on floor stays at rest
#[test]
fn test_dem_static_friction() {
    let mut sim = ClusterSimulation3D::new(
        Vec3::ZERO,
        Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        ),
    );
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.floor_friction = 0.6;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    // Create floor at y=0.1
    let floor_height = 0.1;
    let sdf = create_floor_sdf(floor_height);

    // Spawn clump just above floor
    let spawn_pos = Vec3::new(0.8, floor_height + PARTICLE_RADIUS * 2.0, 0.8);
    let clump_idx = sim.spawn(template_idx, spawn_pos, Vec3::ZERO);

    // Run for 2 seconds
    for _ in 0..240 {
        step_with_sdf_collision(&mut sim, &sdf, DT, false); // dry friction
    }

    let final_pos = sim.clumps[clump_idx].position;
    let horizontal_displacement =
        ((final_pos.x - spawn_pos.x).powi(2) + (final_pos.z - spawn_pos.z).powi(2)).sqrt();

    println!("Static friction test:");
    println!("  Initial: {:?}", spawn_pos);
    println!("  Final: {:?}", final_pos);
    println!("  Horizontal displacement: {:.4}m", horizontal_displacement);

    // Should remain nearly stationary horizontally
    assert!(
        horizontal_displacement < 0.02,
        "Clump drifted horizontally: {:.4}m (expected < 0.02m)",
        horizontal_displacement
    );
}

/// Test 2: Kinetic Friction - Sliding clump decelerates
#[test]
fn test_dem_kinetic_friction() {
    let mut sim = ClusterSimulation3D::new(
        Vec3::ZERO,
        Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        ),
    );
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.floor_friction = 0.5;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    let floor_height = 0.1;
    let sdf = create_floor_sdf(floor_height);

    // Spawn and let settle
    let spawn_pos = Vec3::new(0.5, floor_height + PARTICLE_RADIUS * 3.0, 0.8);
    let clump_idx = sim.spawn(template_idx, spawn_pos, Vec3::ZERO);

    // Settle onto floor
    for _ in 0..120 {
        step_with_sdf_collision(&mut sim, &sdf, DT, false);
    }

    // Apply horizontal velocity
    let initial_vel = 1.0;
    sim.clumps[clump_idx].velocity.x = initial_vel;

    // Track deceleration
    let mut velocities = vec![];
    for _ in 0..180 {
        step_with_sdf_collision(&mut sim, &sdf, DT, false);
        velocities.push(sim.clumps[clump_idx].velocity.x);
    }

    let final_vel = sim.clumps[clump_idx].velocity.x;

    println!("Kinetic friction test:");
    println!("  Initial velocity: {:.3} m/s", initial_vel);
    println!("  Final velocity: {:.3} m/s", final_vel);
    println!("  Decelerated: {}", final_vel < initial_vel * 0.5);

    // Should have decelerated significantly
    assert!(
        final_vel < initial_vel * 0.5,
        "Clump didn't decelerate enough: final {:.3} (expected < {:.3})",
        final_vel,
        initial_vel * 0.5
    );
}

/// Test 3: Wet vs Dry Friction - Wet slides farther
#[test]
fn test_dem_wet_vs_dry_friction() {
    let floor_height = 0.1;
    let sdf = create_floor_sdf(floor_height);
    let initial_vel = 1.0;

    // === DRY friction test ===
    let mut sim_dry = ClusterSimulation3D::new(
        Vec3::ZERO,
        Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        ),
    );
    sim_dry.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim_dry.floor_friction = 0.5; // Higher dry friction
    sim_dry.wet_friction = 0.08; // Lower wet friction

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim_dry.add_template(template);

    let spawn_pos = Vec3::new(0.4, floor_height + PARTICLE_RADIUS * 3.0, 0.8);
    let clump_dry = sim_dry.spawn(template_idx, spawn_pos, Vec3::ZERO);

    // Settle
    for _ in 0..120 {
        step_with_sdf_collision(&mut sim_dry, &sdf, DT, false);
    }

    let dry_start_x = sim_dry.clumps[clump_dry].position.x;
    sim_dry.clumps[clump_dry].velocity.x = initial_vel;

    // Slide with DRY friction
    for _ in 0..300 {
        step_with_sdf_collision(&mut sim_dry, &sdf, DT, false); // wet=false
    }
    let dry_distance = (sim_dry.clumps[clump_dry].position.x - dry_start_x).abs();

    // === WET friction test ===
    let mut sim_wet = ClusterSimulation3D::new(
        Vec3::ZERO,
        Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        ),
    );
    sim_wet.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim_wet.floor_friction = 0.5;
    sim_wet.wet_friction = 0.08;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim_wet.add_template(template);

    let clump_wet = sim_wet.spawn(template_idx, spawn_pos, Vec3::ZERO);

    // Settle
    for _ in 0..120 {
        step_with_sdf_collision(&mut sim_wet, &sdf, DT, true); // wet during settling too
    }

    let wet_start_x = sim_wet.clumps[clump_wet].position.x;
    sim_wet.clumps[clump_wet].velocity.x = initial_vel;

    // Slide with WET friction
    for _ in 0..300 {
        step_with_sdf_collision(&mut sim_wet, &sdf, DT, true); // wet=true
    }
    let wet_distance = (sim_wet.clumps[clump_wet].position.x - wet_start_x).abs();

    println!("Wet vs Dry friction test:");
    println!("  Dry friction (μ=0.5): slid {:.4}m", dry_distance);
    println!("  Wet friction (μ=0.08): slid {:.4}m", wet_distance);
    println!(
        "  Ratio wet/dry: {:.2}x",
        wet_distance / dry_distance.max(0.001)
    );

    // Wet should slide significantly farther (at least 2x)
    assert!(
        wet_distance > dry_distance * 2.0,
        "Wet friction didn't slide farther: wet={:.4}m, dry={:.4}m, ratio={:.2}x",
        wet_distance,
        dry_distance,
        wet_distance / dry_distance.max(0.001)
    );
}

/// Test 4: Friction is Finite - Large impulse doesn't cause instant stop
///
/// Note: DEM uses spring-damper friction, not strict Coulomb saturation.
/// This test verifies friction behaves reasonably - slows but doesn't instantly stop.
#[test]
fn test_dem_friction_finite() {
    let mut sim = ClusterSimulation3D::new(
        Vec3::ZERO,
        Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        ),
    );
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);
    sim.floor_friction = 0.6;

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    let floor_height = 0.1;
    let sdf = create_floor_sdf(floor_height);

    // Spawn and settle
    let spawn_pos = Vec3::new(0.5, floor_height + PARTICLE_RADIUS * 3.0, 0.8);
    let clump_idx = sim.spawn(template_idx, spawn_pos, Vec3::ZERO);

    for _ in 0..120 {
        step_with_sdf_collision(&mut sim, &sdf, DT, false);
    }

    // Apply large velocity impulse
    let large_vel = 10.0;
    sim.clumps[clump_idx].velocity.x = large_vel;

    let vel_before = sim.clumps[clump_idx].velocity.x;

    // Single step
    step_with_sdf_collision(&mut sim, &sdf, DT, false);

    let vel_after = sim.clumps[clump_idx].velocity.x;

    println!("Friction finite test:");
    println!("  Velocity: {:.3} -> {:.3} m/s", vel_before, vel_after);
    println!("  Retained: {:.1}%", 100.0 * vel_after / vel_before);

    // Should retain significant velocity (not instant stop)
    assert!(
        vel_after > vel_before * 0.5,
        "Friction too strong: {:.3} -> {:.3} m/s (retained only {:.1}%)",
        vel_before,
        vel_after,
        100.0 * vel_after / vel_before
    );

    // Should have slowed somewhat (friction is working)
    assert!(
        vel_after < vel_before,
        "No friction applied: velocity unchanged at {:.3} m/s",
        vel_after
    );
}
