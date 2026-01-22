//! Pressure Gradient Test - Verify directional no-penetration
//!
//! This test verifies that the pressure gradient shader:
//! 1. Allows velocity AWAY from solids (water can leave solid surface)
//! 2. Blocks velocity INTO solids (no penetration)
//!
//! CRITICAL TEST: V velocity at riffle TOP
//! - Water pooled behind a riffle needs to rise UP to overflow
//! - Current bug: V velocity at riffle top is zeroed (both directions)
//! - Correct: V > 0 (upward, away from solid) should be PRESERVED
//!
//! Run with: cargo run --example pressure_gradient_test --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use sim3d::FlipSimulation3D;

// Small grid for focused testing
const GRID_WIDTH: usize = 12;
const GRID_HEIGHT: usize = 12;
const GRID_DEPTH: usize = 6;
const CELL_SIZE: f32 = 0.1;
const MAX_PARTICLES: usize = 5000;

const DT: f32 = 1.0 / 60.0;

fn main() {
    println!("=== PRESSURE GRADIENT TEST ===");
    println!("Testing V velocity at riffle top (water rising to overflow)");
    println!();

    // Create simulation with a riffle
    let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    sim.gravity = Vec3::ZERO; // No gravity - isolate pressure behavior
    sim.flip_ratio = 0.95;
    sim.pressure_iterations = 50;

    // Geometry:
    // - Floor at y=0
    // - Riffle at x=6, height=4 cells (y=1 to y=4)
    // - Water spawns at y=3-4 (AT riffle level) with UPWARD velocity
    // - Water should be able to rise to y=5+ to overflow

    let riffle_x = 6;
    let riffle_height = 4; // y=1,2,3,4 are solid

    // Floor
    for i in 0..GRID_WIDTH {
        for k in 0..GRID_DEPTH {
            sim.grid.set_solid(i, 0, k);
        }
    }

    // Riffle - vertical wall
    for y in 1..=riffle_height {
        for k in 1..GRID_DEPTH - 1 {
            sim.grid.set_solid(riffle_x, y, k);
        }
    }

    sim.grid.compute_sdf();

    println!("Geometry:");
    println!("  Floor at y=0");
    println!("  Riffle at x={}, y=1 to y={}", riffle_x, riffle_height);
    println!(
        "  Riffle top at y={} (cell {})",
        riffle_height, riffle_height
    );
    println!("  Above riffle (y>{}) is OPEN", riffle_height);
    println!();

    // Spawn particles DIRECTLY ABOVE the riffle (at x=riffle_x)
    // This tests the V face between the solid riffle top and the air above
    // Position: x=6 (at riffle), y=5 (just above riffle top at y=4)

    let mut test_particles = Vec::new();

    for k in 2..4 {
        // Spawn directly above the riffle, with upward velocity
        let pos = Vec3::new(
            (riffle_x as f32 + 0.5) * CELL_SIZE, // AT the riffle x position
            (riffle_height as f32 + 1.5) * CELL_SIZE, // Just above riffle top (y=5.5)
            (k as f32 + 0.5) * CELL_SIZE,
        );
        let vel = Vec3::new(0.0, 1.0, 0.0); // UPWARD velocity
        sim.spawn_particle_with_velocity(pos, vel);
        test_particles.push(sim.particles.list().len() - 1);
    }

    // Also spawn some trying to go DOWN into the riffle (should be blocked)
    let mut downward_particles = Vec::new();

    for k in 2..4 {
        let pos = Vec3::new(
            (riffle_x as f32 + 0.5) * CELL_SIZE,
            (riffle_height as f32 + 1.5) * CELL_SIZE,
            (k as f32 + 0.5) * CELL_SIZE,
        );
        let vel = Vec3::new(0.0, -1.0, 0.0); // DOWNWARD velocity (into riffle)
        sim.spawn_particle_with_velocity(pos, vel);
        downward_particles.push(sim.particles.list().len() - 1);
    }

    // Also spawn particles ABOVE the riffle to verify they're not affected
    let mut above_particles = Vec::new();

    for i in 4..8 {
        for k in 2..4 {
            let pos = Vec3::new(
                (i as f32 + 0.5) * CELL_SIZE,
                (riffle_height as f32 + 2.5) * CELL_SIZE, // Well above riffle
                (k as f32 + 0.5) * CELL_SIZE,
            );
            let vel = Vec3::new(1.0, 0.0, 0.0); // Horizontal velocity
            sim.spawn_particle_with_velocity(pos, vel);
            above_particles.push(sim.particles.list().len() - 1);
        }
    }

    println!("Particles spawned:");
    println!(
        "  Above riffle (upward vel): {} particles",
        test_particles.len()
    );
    println!(
        "  Above riffle (downward vel): {} particles",
        downward_particles.len()
    );
    println!(
        "  Far above riffle (horizontal vel): {} particles",
        above_particles.len()
    );
    println!();

    // Initialize GPU
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Pressure Gradient Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create GPU device")
    });

    let mut gpu_flip = GpuFlip3D::new(
        &device,
        GRID_WIDTH as u32,
        GRID_HEIGHT as u32,
        GRID_DEPTH as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Prepare GPU data
    let mut positions: Vec<Vec3> = sim.particles.list().iter().map(|p| p.position).collect();
    let mut velocities: Vec<Vec3> = sim.particles.list().iter().map(|p| p.velocity).collect();
    let mut c_matrices: Vec<Mat3> = sim
        .particles
        .list()
        .iter()
        .map(|p| p.affine_velocity)
        .collect();
    let densities: Vec<f32> = sim.particles.list().iter().map(|p| p.density).collect();
    let bed_height: Vec<f32> = vec![0.0; GRID_WIDTH * GRID_DEPTH];

    // Build cell types
    let mut cell_types: Vec<u32> = vec![0; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
    let mut solid_count = 0;
    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                if sim.grid.is_solid(i, j, k) {
                    cell_types[idx] = 2; // solid
                    solid_count += 1;
                }
            }
        }
    }

    println!("=== CELL TYPE DEBUG ===");
    println!("Total solid cells from sim.grid: {}", solid_count);

    // Check specific cells around the riffle
    println!("Riffle cells at x={}:", riffle_x);
    println!(
        "Cell index formula: k * {} * {} + j * {} + i",
        GRID_WIDTH, GRID_HEIGHT, GRID_WIDTH
    );
    for y in 0..=riffle_height + 2 {
        let k = 2; // middle of depth
        let idx = k * GRID_WIDTH * GRID_HEIGHT + y * GRID_WIDTH + riffle_x;
        let is_solid = sim.grid.is_solid(riffle_x, y, k);
        let cell_type = cell_types[idx];
        println!(
            "  Cell ({}, {}, {}): idx={}, is_solid={}, cell_type={}",
            riffle_x, y, k, idx, is_solid, cell_type
        );
    }

    // V face at j=5 checks cells [i, 4, k] and [i, 5, k]
    // For the pressure gradient shader to zero V, cell [6, 4, 2] must be solid
    let critical_idx = 2 * GRID_WIDTH * GRID_HEIGHT + 4 * GRID_WIDTH + riffle_x;
    println!("\nCritical check: V face at j=5 tests cell (6, 4, 2)");
    println!(
        "  Index = {}, cell_type[{}] = {}",
        critical_idx, critical_idx, cell_types[critical_idx]
    );
    println!("  Expected: 2 (SOLID)");

    // Mark fluid cells
    for p in sim.particles.list() {
        let i = (p.position.x / CELL_SIZE).floor() as i32;
        let j = (p.position.y / CELL_SIZE).floor() as i32;
        let k = (p.position.z / CELL_SIZE).floor() as i32;
        if i >= 0
            && i < GRID_WIDTH as i32
            && j >= 0
            && j < GRID_HEIGHT as i32
            && k >= 0
            && k < GRID_DEPTH as i32
        {
            let idx = k as usize * GRID_WIDTH * GRID_HEIGHT + j as usize * GRID_WIDTH + i as usize;
            if cell_types[idx] != 2 {
                cell_types[idx] = 1; // fluid
            }
        }
    }

    // Check particle positions
    println!("\nParticle positions:");
    for (idx, &p_idx) in test_particles.iter().enumerate() {
        let p = &sim.particles.list()[p_idx];
        let cell_i = (p.position.x / CELL_SIZE).floor() as usize;
        let cell_j = (p.position.y / CELL_SIZE).floor() as usize;
        let cell_k = (p.position.z / CELL_SIZE).floor() as usize;
        println!(
            "  Upward particle {}: pos=({:.2}, {:.2}, {:.2}) -> cell=({}, {}, {})",
            idx, p.position.x, p.position.y, p.position.z, cell_i, cell_j, cell_k
        );
    }
    println!();

    // Record initial velocities
    let initial_vy_up: f32 =
        test_particles.iter().map(|&i| velocities[i].y).sum::<f32>() / test_particles.len() as f32;

    let initial_vy_down: f32 = downward_particles
        .iter()
        .map(|&i| velocities[i].y)
        .sum::<f32>()
        / downward_particles.len() as f32;

    let initial_vx_above: f32 = above_particles
        .iter()
        .map(|&i| velocities[i].x)
        .sum::<f32>()
        / above_particles.len() as f32;

    println!("Initial velocities:");
    println!(
        "  Upward particles (Vy): {:.3} (rising away from riffle)",
        initial_vy_up
    );
    println!(
        "  Downward particles (Vy): {:.3} (falling toward riffle)",
        initial_vy_down
    );
    println!(
        "  Horizontal particles (Vx): {:.3} (flowing above)",
        initial_vx_above
    );
    println!();

    // Run ONE GPU step
    gpu_flip.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        Some(&bed_height),
        DT,
        0.0, // No gravity
        0.0, // No flow accel
        sim.pressure_iterations as u32,
    );

    println!("=== ANALYSIS ===");
    println!("The shader may be zeroing V at face j=5, but G2P interpolation");
    println!("averages from multiple faces (j=4,5,6,7). Even if face j=5 is zero,");
    println!("faces j=6,7 have velocity, so the particle sees a weighted average.");
    println!();

    // Check final velocities
    let final_vy_up: f32 =
        test_particles.iter().map(|&i| velocities[i].y).sum::<f32>() / test_particles.len() as f32;

    let final_vy_down: f32 = downward_particles
        .iter()
        .map(|&i| velocities[i].y)
        .sum::<f32>()
        / downward_particles.len() as f32;

    let final_vx_above: f32 = above_particles
        .iter()
        .map(|&i| velocities[i].x)
        .sum::<f32>()
        / above_particles.len() as f32;

    println!("After pressure solve:");
    println!(
        "  Upward particles (Vy): {:.3} (was {:.3})",
        final_vy_up, initial_vy_up
    );
    println!(
        "  Downward particles (Vy): {:.3} (was {:.3})",
        final_vy_down, initial_vy_down
    );
    println!(
        "  Horizontal particles (Vx): {:.3} (was {:.3})",
        final_vx_above, initial_vx_above
    );
    println!();

    // TESTS
    println!("=== TEST RESULTS ===");
    let mut passed = true;

    // TEST 1: Upward velocity (AWAY from riffle) should be PRESERVED
    // This is the critical test - water must be able to rise to overflow
    if final_vy_up.abs() < 0.1 {
        println!("FAIL: Upward Vy = {:.3} (KILLED!)", final_vy_up);
        println!("      Water cannot rise to overflow the riffle!");
        println!("      BUG: V is zeroed at solid-adjacent faces regardless of direction");
        passed = false;
    } else if final_vy_up > 0.0 {
        println!("PASS: Upward Vy = {:.3} (preserved)", final_vy_up);
        println!("      Water CAN rise to overflow!");
    } else {
        println!("WARN: Upward Vy = {:.3} (reversed?)", final_vy_up);
    }

    // TEST 2: Downward velocity (INTO riffle) should be BLOCKED
    // This is correct no-penetration behavior
    if final_vy_down < -0.1 {
        println!("WARN: Downward Vy = {:.3} (not blocked)", final_vy_down);
        println!("      Water may penetrate the riffle");
    } else {
        println!(
            "PASS: Downward Vy = {:.3} (blocked or reversed)",
            final_vy_down
        );
        println!("      No-penetration working");
    }

    // TEST 3: Horizontal velocity far above should be preserved
    if final_vx_above.abs() < 0.1 {
        println!("FAIL: Horizontal Vx = {:.3} (KILLED!)", final_vx_above);
        passed = false;
    } else {
        println!("PASS: Horizontal Vx = {:.3} (preserved)", final_vx_above);
    }

    println!();

    if !passed {
        println!("=== TEST FAILED ===");
        println!();
        println!("The pressure gradient shader kills V velocity at the top of riffles.");
        println!("Water cannot rise UP to overflow - it's trapped behind the riffle.");
        println!();
        println!("Root cause in pressure_gradient_3d.wgsl:");
        println!("  if (bottom_type == CELL_SOLID || top_type == CELL_SOLID) {{");
        println!("      grid_v[idx] = 0.0;  // WRONG: kills upward flow too!");
        println!("  }}");
        println!();
        println!("Fix: Only zero velocity going INTO the solid:");
        println!("  if (bottom_type == CELL_SOLID && grid_v[idx] < 0.0) {{");
        println!("      grid_v[idx] = 0.0;  // Block downward into solid");
        println!("  }}");
        println!("  if (top_type == CELL_SOLID && grid_v[idx] > 0.0) {{");
        println!("      grid_v[idx] = 0.0;  // Block upward into solid");
        println!("  }}");
        std::process::exit(1);
    } else {
        println!("=== TEST PASSED ===");
        println!("Directional no-penetration is working correctly.");
        std::process::exit(0);
    }
}
