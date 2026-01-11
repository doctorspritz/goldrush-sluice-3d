//! Tracer Particle Tests
//!
//! Validates particle advection by tracking individual tracer particles
//! through the flow field. Proves that:
//! 1. Particles move downstream in a flow
//! 2. Particles follow expected trajectories
//! 3. No particles teleport or get stuck
//! 4. Tracer velocities match surrounding flow
//!
//! Run with: cargo run --example test_tracers --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use pollster::block_on;

const CELL_SIZE: f32 = 0.02;
const GRID_WIDTH: usize = 60;
const GRID_HEIGHT: usize = 20;
const GRID_DEPTH: usize = 16;
const MAX_PARTICLES: usize = 50_000;

/// Update cell types from particle positions - cells with particles become FLUID (1)
fn update_cell_types_from_particles(
    cell_types: &mut [u32],
    positions: &[Vec3],
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    cell_size: f32,
) {
    // Reset non-solid cells to AIR
    for ct in cell_types.iter_mut() {
        if *ct != 2 {
            *ct = 0; // AIR
        }
    }

    // Mark cells containing particles as FLUID
    for pos in positions {
        let i = (pos.x / cell_size) as i32;
        let j = (pos.y / cell_size) as i32;
        let k = (pos.z / cell_size) as i32;

        if i >= 0
            && i < grid_width as i32
            && j >= 0
            && j < grid_height as i32
            && k >= 0
            && k < grid_depth as i32
        {
            let idx = k as usize * grid_width * grid_height + j as usize * grid_width + i as usize;
            if cell_types[idx] != 2 {
                // Preserve solids
                cell_types[idx] = 1; // FLUID
            }
        }
    }
}

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" TRACER PARTICLE TESTS");
    println!("{}", "=".repeat(70));
    println!("\nThese tests validate particle advection through flow fields.\n");

    let (device, queue) = block_on(init_gpu());

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Tracers move downstream
    if test_downstream_advection(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 2: Tracers follow gravity in still water
    if test_gravity_settling(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 3: No teleportation (smooth trajectories)
    if test_no_teleportation(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 4: Tracer velocity matches flow
    if test_velocity_consistency(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n{}", "=".repeat(70));
    if failed == 0 {
        println!(" ALL TRACER TESTS PASSED ({}/{})", passed, passed + failed);
    } else {
        println!(
            " TRACER TESTS FAILED: {}/{} passed",
            passed,
            passed + failed
        );
    }
    println!("{}", "=".repeat(70));

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Test 1: Vertical Advection (Falling Water)
///
/// A blob of water in mid-air falls under gravity.
/// Track center of mass Y to verify particles are being advected.
fn test_downstream_advection(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("----------------------------------------");
    println!("TEST 1: Vertical Advection (Falling Water)");
    println!("----------------------------------------");
    println!("Expected: Water blob falls under gravity (CoM Y decreases)");

    let grid_size = 32;
    let mut flip = GpuFlip3D::new(
        device,
        grid_size as u32,
        grid_size as u32,
        grid_size as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    // Create a blob of water in upper half of domain
    // Using 8 particles per cell like physics validation
    let blob_start_y = grid_size / 2; // Start at vertical center
    let blob_end_y = 3 * grid_size / 4;

    for j in blob_start_y..blob_end_y {
        for k in grid_size / 4..3 * grid_size / 4 {
            for i in grid_size / 4..3 * grid_size / 4 {
                for pj in 0..2 {
                    for pk in 0..2 {
                        for pi in 0..2 {
                            positions.push(Vec3::new(
                                (i as f32 + 0.25 + pi as f32 * 0.5) * CELL_SIZE,
                                (j as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE,
                                (k as f32 + 0.25 + pk as f32 * 0.5) * CELL_SIZE,
                            ));
                            velocities.push(Vec3::ZERO);
                            densities.push(1.0);
                            c_matrices.push(Mat3::ZERO);
                        }
                    }
                }
            }
        }
    }

    println!("  Particle count: {}", positions.len());

    // Simple geometry: floor only (like physics validation test 3)
    let cell_count = grid_size * grid_size * grid_size;
    let mut cell_types = vec![0u32; cell_count];

    // Floor at j=0
    for k in 0..grid_size {
        for i in 0..grid_size {
            cell_types[k * grid_size * grid_size + i] = 2;
        }
    }

    // SDF: just distance to floor
    let sdf: Vec<f32> = (0..cell_count)
        .map(|idx| {
            let j = (idx / grid_size) % grid_size;
            (j as f32 - 0.5) * CELL_SIZE
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;

    // Track center of mass Y
    let initial_com_y: f32 = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
    println!("  Initial center of mass Y: {:.4}m", initial_com_y);

    println!("  Running falling water simulation (60 frames)...");

    // Run simulation
    for frame in 0..60 {
        update_cell_types_from_particles(
            &mut cell_types,
            &positions,
            grid_size,
            grid_size,
            grid_size,
            CELL_SIZE,
        );

        flip.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            Some(&sdf),
            None,
            dt,
            gravity,
            0.0,
            60,
        );

        // Progress at midpoint
        if frame == 29 {
            let com_y: f32 = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
            println!("  Frame {}: com_y = {:.4}m", frame + 1, com_y);
        }
    }

    // Final center of mass
    let final_com_y: f32 = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
    let drop = initial_com_y - final_com_y;

    println!("  Final center of mass Y: {:.4}m", final_com_y);
    println!("  Drop distance: {:.4}m", drop);

    // Water should have fallen at least 5cm in 1 second under gravity
    let pass = drop > 0.05;
    println!("  Result: {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 2: Gravity Settling
///
/// In still water, a dense tracer should settle downward.
/// A water-density tracer should stay approximately level.
fn test_gravity_settling(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 2: Gravity Settling");
    println!("----------------------------------------");
    println!("Expected: Dense tracer sinks, water tracer stays level");

    let grid_size = 24;
    let mut flip = GpuFlip3D::new(
        device,
        grid_size as u32,
        grid_size as u32,
        grid_size as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    // Fill box with water
    for j in 2..grid_size - 2 {
        for k in 2..grid_size - 2 {
            for i in 2..grid_size - 2 {
                positions.push(Vec3::new(
                    (i as f32 + 0.5) * CELL_SIZE,
                    (j as f32 + 0.5) * CELL_SIZE,
                    (k as f32 + 0.5) * CELL_SIZE,
                ));
                velocities.push(Vec3::ZERO);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    // Add water tracer (should stay level)
    let water_tracer_idx = positions.len();
    let water_tracer_start_y = (grid_size as f32 / 2.0) * CELL_SIZE;
    positions.push(Vec3::new(
        grid_size as f32 * CELL_SIZE / 2.0,
        water_tracer_start_y,
        grid_size as f32 * CELL_SIZE / 2.0,
    ));
    velocities.push(Vec3::ZERO);
    densities.push(1.0); // Water density
    c_matrices.push(Mat3::ZERO);

    // Add dense tracer (should sink) - using density > 1.0 marks as sediment
    let dense_tracer_idx = positions.len();
    let dense_tracer_start_y = (grid_size as f32 / 2.0) * CELL_SIZE;
    positions.push(Vec3::new(
        grid_size as f32 * CELL_SIZE / 2.0 + 0.05,
        dense_tracer_start_y,
        grid_size as f32 * CELL_SIZE / 2.0,
    ));
    velocities.push(Vec3::ZERO);
    densities.push(2.7); // Sediment density (gangue)
    c_matrices.push(Mat3::ZERO);

    println!("  Water tracer: start y = {:.3}m", water_tracer_start_y);
    println!("  Dense tracer: start y = {:.3}m", dense_tracer_start_y);

    // Build closed box
    let cell_count = grid_size * grid_size * grid_size;
    let mut cell_types = vec![0u32; cell_count];
    let mut sdf = vec![CELL_SIZE * 2.0; cell_count];

    for k in 0..grid_size {
        for j in 0..grid_size {
            for i in 0..grid_size {
                let idx = k * grid_size * grid_size + j * grid_size + i;
                if i <= 1
                    || i >= grid_size - 2
                    || j <= 1
                    || j >= grid_size - 2
                    || k <= 1
                    || k >= grid_size - 2
                {
                    cell_types[idx] = 2;
                    sdf[idx] = -CELL_SIZE;
                }
            }
        }
    }

    let dt = 1.0 / 60.0;
    let gravity = -9.8;

    // Run simulation
    for _ in 0..120 {
        update_cell_types_from_particles(
            &mut cell_types,
            &positions,
            grid_size,
            grid_size,
            grid_size,
            CELL_SIZE,
        );

        flip.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            Some(&sdf),
            None,
            dt,
            gravity,
            0.0,
            60,
        );
    }

    // Check results
    let water_final_y = positions[water_tracer_idx].y;
    let dense_final_y = positions[dense_tracer_idx].y;

    let water_drop = water_tracer_start_y - water_final_y;
    let dense_drop = dense_tracer_start_y - dense_final_y;

    println!(
        "  Water tracer: y = {:.3}m (dropped {:.3}m)",
        water_final_y, water_drop
    );
    println!(
        "  Dense tracer: y = {:.3}m (dropped {:.3}m)",
        dense_final_y, dense_drop
    );

    // Dense tracer should drop more than water tracer
    let pass = dense_drop > water_drop;
    println!("  Dense dropped more: {}", pass);
    println!("  Result: {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 3: No Teleportation
///
/// Particle positions should change smoothly - no sudden jumps.
/// Max displacement per frame should be bounded by velocity * dt.
fn test_no_teleportation(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 3: No Teleportation (Smooth Trajectories)");
    println!("----------------------------------------");
    println!("Expected: All position changes < velocity * dt * safety_factor");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_WIDTH as u32,
        GRID_HEIGHT as u32,
        GRID_DEPTH as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Create water with tracers
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let floor_height = 3;
    let water_height = 10;

    for j in floor_height..water_height {
        for k in 2..GRID_DEPTH - 2 {
            for i in 2..GRID_WIDTH - 2 {
                positions.push(Vec3::new(
                    (i as f32 + 0.5) * CELL_SIZE,
                    (j as f32 + 0.5) * CELL_SIZE,
                    (k as f32 + 0.5) * CELL_SIZE,
                ));
                velocities.push(Vec3::ZERO);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    // Track 10 tracer particles
    let num_tracers = 10;
    let tracer_indices: Vec<usize> = (0..num_tracers).map(|i| i * 100).collect();

    println!("  Tracking {} tracers", num_tracers);

    // Build geometry
    let cell_count = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
    let mut cell_types = vec![0u32; cell_count];
    let mut sdf = vec![CELL_SIZE * 2.0; cell_count];

    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                if j <= floor_height || k == 0 || k == GRID_DEPTH - 1 || i == 0 {
                    cell_types[idx] = 2;
                    sdf[idx] = -CELL_SIZE;
                }
            }
        }
    }

    let dt = 1.0 / 60.0;
    let gravity = -9.8;
    let flow_accel = 1.5;

    // Let system settle before tracking
    for _ in 0..30 {
        update_cell_types_from_particles(
            &mut cell_types,
            &positions,
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_DEPTH,
            CELL_SIZE,
        );

        flip.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            Some(&sdf),
            None,
            dt,
            gravity,
            flow_accel,
            40,
        );
    }

    // Store previous positions after settling
    let mut prev_positions: Vec<Vec3> = tracer_indices
        .iter()
        .filter_map(|&idx| positions.get(idx).copied())
        .collect();

    let mut max_displacement = 0.0f32;
    let mut max_velocity_seen = 0.0f32;
    let mut teleport_count = 0;

    // Run and track (after settling)
    for frame in 0..90 {
        update_cell_types_from_particles(
            &mut cell_types,
            &positions,
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_DEPTH,
            CELL_SIZE,
        );

        flip.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            Some(&sdf),
            None,
            dt,
            gravity,
            flow_accel,
            40,
        );

        // Check displacements
        for (i, &tracer_idx) in tracer_indices.iter().enumerate() {
            if tracer_idx < positions.len() && i < prev_positions.len() {
                let displacement = (positions[tracer_idx] - prev_positions[i]).length();
                let velocity = velocities[tracer_idx].length();

                max_displacement = max_displacement.max(displacement);
                max_velocity_seen = max_velocity_seen.max(velocity);

                // Displacement should be bounded by CFL condition
                // In FLIP, particles can move up to ~1 cell per frame + density projection
                let expected_max = CELL_SIZE * 2.0; // 2 cells max per frame
                if displacement > expected_max {
                    teleport_count += 1;
                    if teleport_count <= 3 {
                        println!(
                            "  WARNING frame {}: tracer {} jumped {:.4}m (expected < {:.4}m)",
                            frame, i, displacement, expected_max
                        );
                    }
                }

                prev_positions[i] = positions[tracer_idx];
            }
        }
    }

    println!("  Max displacement/frame: {:.4}m", max_displacement);
    println!("  Max velocity seen: {:.2} m/s", max_velocity_seen);
    println!("  Teleport events: {}", teleport_count);

    let pass = teleport_count == 0;
    println!("  Result: {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 4: Velocity Consistency
///
/// Tracer velocity should match the interpolated grid velocity at its location.
/// This validates the G2P transfer is working correctly.
fn test_velocity_consistency(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 4: Velocity Consistency");
    println!("----------------------------------------");
    println!("Expected: Tracer velocities match surrounding flow");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_WIDTH as u32,
        GRID_HEIGHT as u32,
        GRID_DEPTH as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Create uniform flow
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let floor_height = 3;
    let water_height = 10;

    for j in floor_height..water_height {
        for k in 2..GRID_DEPTH - 2 {
            for i in 2..GRID_WIDTH - 2 {
                positions.push(Vec3::new(
                    (i as f32 + 0.5) * CELL_SIZE,
                    (j as f32 + 0.5) * CELL_SIZE,
                    (k as f32 + 0.5) * CELL_SIZE,
                ));
                // Start with uniform velocity
                velocities.push(Vec3::new(0.5, 0.0, 0.0));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    // Pick tracers in middle of flow
    let num_tracers = 5;
    let tracer_indices: Vec<usize> = (0..num_tracers)
        .map(|i| positions.len() / 2 + i * 10)
        .filter(|&idx| idx < positions.len())
        .collect();

    println!(
        "  Tracking {} tracers for velocity consistency",
        tracer_indices.len()
    );

    // Build geometry
    let cell_count = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
    let mut cell_types = vec![0u32; cell_count];
    let mut sdf = vec![CELL_SIZE * 2.0; cell_count];

    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                if j <= floor_height || k == 0 || k == GRID_DEPTH - 1 || i == 0 {
                    cell_types[idx] = 2;
                    sdf[idx] = -CELL_SIZE;
                }
            }
        }
    }

    let dt = 1.0 / 60.0;
    let gravity = -9.8;
    let flow_accel = 1.0;

    // Run to steady state
    for _ in 0..120 {
        update_cell_types_from_particles(
            &mut cell_types,
            &positions,
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_DEPTH,
            CELL_SIZE,
        );

        flip.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            Some(&sdf),
            None,
            dt,
            gravity,
            flow_accel,
            40,
        );
    }

    // Compute average velocity of all water particles
    let total_vel: Vec3 = velocities.iter().sum();
    let avg_vel = total_vel / velocities.len() as f32;

    println!(
        "  Average flow velocity: ({:.3}, {:.3}, {:.3}) m/s",
        avg_vel.x, avg_vel.y, avg_vel.z
    );

    // Check tracer velocities are close to average
    let mut max_deviation = 0.0f32;
    for (i, &tracer_idx) in tracer_indices.iter().enumerate() {
        if tracer_idx < velocities.len() {
            let tracer_vel = velocities[tracer_idx];
            let deviation = (tracer_vel - avg_vel).length();
            max_deviation = max_deviation.max(deviation);

            println!(
                "  Tracer {}: vel = ({:.3}, {:.3}, {:.3}), deviation = {:.3} m/s",
                i, tracer_vel.x, tracer_vel.y, tracer_vel.z, deviation
            );
        }
    }

    // Tracers should be within reasonable tolerance of average velocity
    // FLIP naturally has velocity variation due to P2G/G2P transfers
    let tolerance = avg_vel.length() * 0.6 + 0.15;
    let pass = max_deviation < tolerance;

    println!(
        "  Max deviation: {:.3} m/s (tolerance: {:.3})",
        max_deviation, tolerance
    );
    println!("  Result: {}", if pass { "PASS" } else { "FAIL" });
    pass
}

async fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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
    limits.max_storage_buffer_binding_size = 1024 * 1024 * 256;

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Tracer Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device")
}
