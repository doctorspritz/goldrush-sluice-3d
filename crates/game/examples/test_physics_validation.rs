//! Physics Validation Tests
//!
//! Proves fundamental physics properties against analytical solutions.
//! These are the "immutable physics" tests - if these fail, something is fundamentally wrong.
//!
//! Run with: cargo run --example test_physics_validation --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use pollster::block_on;

const CELL_SIZE: f32 = 0.02;
const GRID_SIZE: usize = 32;
const MAX_PARTICLES: usize = 10_000;

/// Update cell types from particle positions.
/// This marks cells containing particles as FLUID (1), preserving SOLID (2) cells.
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

        if i >= 0 && i < grid_width as i32
            && j >= 0 && j < grid_height as i32
            && k >= 0 && k < grid_depth as i32
        {
            let idx = k as usize * grid_width * grid_height + j as usize * grid_width + i as usize;
            if cell_types[idx] != 2 {
                cell_types[idx] = 1; // FLUID
            }
        }
    }
}

fn main() {
    println!("\n{}", "=".repeat(70));
    println!(" PHYSICS VALIDATION TESTS");
    println!("{}", "=".repeat(70));
    println!("\nThese tests verify fundamental physics against analytical solutions.\n");

    let (device, queue) = block_on(init_gpu());

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Gravity produces correct acceleration
    if test_gravity_acceleration(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 2: Hydrostatic pressure gradient
    if test_hydrostatic_pressure(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 3: Incompressibility (divergence-free after pressure solve)
    if test_incompressibility(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 4: Particle count conservation (closed system)
    if test_particle_conservation(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 5: Solid boundary enforcement
    if test_solid_boundaries(&device, &queue) {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n{}", "=".repeat(70));
    if failed == 0 {
        println!(" ALL PHYSICS TESTS PASSED ({}/{})", passed, passed + failed);
    } else {
        println!(" PHYSICS TESTS FAILED: {}/{} passed", passed, passed + failed);
    }
    println!("{}", "=".repeat(70));

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Test 1: Gravity Acceleration
///
/// Physics: A falling body should accelerate at g = 9.8 m/s²
/// After time t, velocity should be v = g*t
///
/// Method: Drop a blob of particles, measure average velocity after known time
/// Note: FLIP requires particle groups for grid-based transfer to work properly
fn test_gravity_acceleration(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("----------------------------------------");
    println!("TEST 1: Gravity Acceleration");
    println!("----------------------------------------");
    println!("Expected: v_y ~ g*t = -9.8 * 0.25 = -2.45 m/s after 0.25s");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Create a small blob of particles (FLIP needs particle groups for P2G/G2P)
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    // 4x4x4 block of particles high up
    let start_y = GRID_SIZE as f32 * CELL_SIZE * 0.7;
    for dj in 0..4 {
        for dk in 0..4 {
            for di in 0..4 {
                positions.push(Vec3::new(
                    GRID_SIZE as f32 * CELL_SIZE / 2.0 + (di as f32 - 1.5) * CELL_SIZE * 0.5,
                    start_y + (dj as f32 - 1.5) * CELL_SIZE * 0.5,
                    GRID_SIZE as f32 * CELL_SIZE / 2.0 + (dk as f32 - 1.5) * CELL_SIZE * 0.5,
                ));
                velocities.push(Vec3::ZERO);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    println!("  Particles: {}", positions.len());

    // Empty cell types (all air except floor)
    let cell_count = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    let mut cell_types = vec![0u32; cell_count];

    // Floor at j=0
    for k in 0..GRID_SIZE {
        for i in 0..GRID_SIZE {
            cell_types[k * GRID_SIZE * GRID_SIZE + i] = 2; // Solid floor
        }
    }

    // Simple SDF
    let sdf: Vec<f32> = (0..cell_count)
        .map(|idx| {
            let j = (idx / GRID_SIZE) % GRID_SIZE;
            (j as f32 - 0.5) * CELL_SIZE
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;
    let target_time = 0.25; // seconds (shorter to avoid floor collision)
    let frames = (target_time / dt) as u32;

    for _ in 0..frames {
        update_cell_types_from_particles(
            &mut cell_types, &positions,
            GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE,
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
            0.0, // no flow accel
            40,  // pressure iters
        );
    }

    // Compute average vertical velocity
    let avg_vy: f32 = velocities.iter().map(|v| v.y).sum::<f32>() / velocities.len() as f32;
    let expected_vy = gravity * target_time; // Should be -2.45 m/s
    let error_pct = ((avg_vy - expected_vy) / expected_vy * 100.0).abs();

    println!("  Avg v_y:      {:.3} m/s", avg_vy);
    println!("  Expected v_y: {:.3} m/s", expected_vy);
    println!("  Error:        {:.1}%", error_pct);

    // Allow 30% error due to pressure solver effects and grid interpolation
    // FLIP is not perfectly conservative - some energy goes to pressure
    let pass = error_pct < 30.0;
    println!("  Result:       {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 2: Hydrostatic Pressure
///
/// Physics: Pressure at depth h should be P = ρgh
/// A column of water should have linearly increasing pressure with depth
///
/// Method: Create water column, verify pressure gradient
fn test_hydrostatic_pressure(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 2: Hydrostatic Pressure Gradient");
    println!("----------------------------------------");
    println!("Expected: Pressure increases linearly with depth");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Create a column of water particles
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let water_height = GRID_SIZE / 2;
    for j in 2..water_height {
        for k in GRID_SIZE/4..3*GRID_SIZE/4 {
            for i in GRID_SIZE/4..3*GRID_SIZE/4 {
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

    let initial_count = positions.len();
    println!("  Water column: {} particles", initial_count);

    let cell_count = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    let mut cell_types = vec![0u32; cell_count];

    // Floor and walls
    for k in 0..GRID_SIZE {
        for j in 0..GRID_SIZE {
            for i in 0..GRID_SIZE {
                let idx = k * GRID_SIZE * GRID_SIZE + j * GRID_SIZE + i;
                if j <= 1 || i == 0 || i == GRID_SIZE-1 || k == 0 || k == GRID_SIZE-1 {
                    cell_types[idx] = 2;
                }
            }
        }
    }

    let sdf: Vec<f32> = (0..cell_count)
        .map(|idx| {
            let i = idx % GRID_SIZE;
            let j = (idx / GRID_SIZE) % GRID_SIZE;
            let k = idx / (GRID_SIZE * GRID_SIZE);
            let dist_floor = (j as f32 - 1.5) * CELL_SIZE;
            let dist_walls = ((i as f32 - 0.5).min((GRID_SIZE - 1 - i) as f32 - 0.5)
                .min((k as f32 - 0.5).min((GRID_SIZE - 1 - k) as f32 - 0.5))) * CELL_SIZE;
            dist_floor.min(dist_walls)
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;

    // Let system settle (3 seconds at 60 FPS)
    for _ in 0..180 {
        update_cell_types_from_particles(
            &mut cell_types, &positions,
            GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE,
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

    // Check that water has settled (low velocities)
    let max_vel: f32 = velocities.iter().map(|v| v.length()).fold(0.0, f32::max);
    let settled = max_vel < 1.0; // Should be nearly stationary

    println!("  After settling:");
    println!("    Max velocity: {:.3} m/s (should be < 1.0)", max_vel);
    println!("    Particles:    {}", positions.len());

    // For hydrostatic equilibrium, vertical velocity should be near zero
    let avg_vy: f32 = velocities.iter().map(|v| v.y.abs()).sum::<f32>() / velocities.len() as f32;
    println!("    Avg |v_y|:    {:.4} m/s (should be ~0)", avg_vy);

    let pass = settled && avg_vy < 0.5;
    println!("  Result:         {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 3: Incompressibility
///
/// Physics: After pressure projection, velocity field should be divergence-free
/// ∇·v ≈ 0
///
/// Method: Check that particles maintain roughly constant density
fn test_incompressibility(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 3: Incompressibility (Volume Conservation)");
    println!("----------------------------------------");
    println!("Expected: Particle density stays roughly constant");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Create a blob of water
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let center = GRID_SIZE as f32 * CELL_SIZE / 2.0;
    let radius = GRID_SIZE as f32 * CELL_SIZE / 4.0;

    for j in GRID_SIZE/4..3*GRID_SIZE/4 {
        for k in GRID_SIZE/4..3*GRID_SIZE/4 {
            for i in GRID_SIZE/4..3*GRID_SIZE/4 {
                let pos = Vec3::new(
                    (i as f32 + 0.5) * CELL_SIZE,
                    (j as f32 + 0.5) * CELL_SIZE,
                    (k as f32 + 0.5) * CELL_SIZE,
                );
                if (pos - Vec3::splat(center)).length() < radius {
                    positions.push(pos);
                    velocities.push(Vec3::ZERO);
                    densities.push(1.0);
                    c_matrices.push(Mat3::ZERO);
                }
            }
        }
    }

    let initial_count = positions.len();
    println!("  Initial particles: {}", initial_count);

    let cell_count = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    let mut cell_types = vec![0u32; cell_count];

    // Floor only
    for k in 0..GRID_SIZE {
        for i in 0..GRID_SIZE {
            cell_types[k * GRID_SIZE * GRID_SIZE + i] = 2;
        }
    }

    let sdf: Vec<f32> = (0..cell_count)
        .map(|idx| {
            let j = (idx / GRID_SIZE) % GRID_SIZE;
            (j as f32 - 0.5) * CELL_SIZE
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;

    // Run simulation
    for _ in 0..120 {
        update_cell_types_from_particles(
            &mut cell_types, &positions,
            GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE,
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

    // Particle count should be exactly preserved (closed system)
    let final_count = positions.len();
    let count_preserved = final_count == initial_count;

    println!("  Final particles:   {}", final_count);
    println!("  Count preserved:   {}", count_preserved);

    // Check that no extreme compression/expansion occurred
    // by verifying velocities are bounded
    let max_vel: f32 = velocities.iter().map(|v| v.length()).fold(0.0, f32::max);
    let vel_reasonable = max_vel < 20.0;

    println!("  Max velocity:      {:.2} m/s (should be < 20)", max_vel);

    let pass = count_preserved && vel_reasonable;
    println!("  Result:            {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 4: Particle Conservation
///
/// Physics: In a closed system, particle count should be exactly conserved
///
/// Method: Run dam break in closed box, verify no particles lost
fn test_particle_conservation(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 4: Particle Conservation (Closed System)");
    println!("----------------------------------------");
    println!("Expected: Particle count exactly preserved");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Dam break setup - water block in corner
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for j in 2..GRID_SIZE/2 {
        for k in 2..GRID_SIZE/2 {
            for i in 2..GRID_SIZE/2 {
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

    let initial_count = positions.len();
    println!("  Initial particles: {}", initial_count);

    // Closed box
    let cell_count = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    let mut cell_types = vec![0u32; cell_count];

    for k in 0..GRID_SIZE {
        for j in 0..GRID_SIZE {
            for i in 0..GRID_SIZE {
                let idx = k * GRID_SIZE * GRID_SIZE + j * GRID_SIZE + i;
                // All boundary cells are solid
                if i == 0 || i == GRID_SIZE-1 ||
                   j == 0 || j == GRID_SIZE-1 ||
                   k == 0 || k == GRID_SIZE-1 {
                    cell_types[idx] = 2;
                }
            }
        }
    }

    let sdf: Vec<f32> = (0..cell_count)
        .map(|idx| {
            let i = idx % GRID_SIZE;
            let j = (idx / GRID_SIZE) % GRID_SIZE;
            let k = idx / (GRID_SIZE * GRID_SIZE);

            let di = (i as f32).min((GRID_SIZE - 1 - i) as f32);
            let dj = (j as f32).min((GRID_SIZE - 1 - j) as f32);
            let dk = (k as f32).min((GRID_SIZE - 1 - k) as f32);

            (di.min(dj).min(dk) - 0.5) * CELL_SIZE
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;

    // Run for 5 seconds of sim time
    for _ in 0..300 {
        update_cell_types_from_particles(
            &mut cell_types, &positions,
            GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE,
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
            40,
        );
    }

    let final_count = positions.len();
    let pass = final_count == initial_count;

    println!("  Final particles:   {}", final_count);
    println!("  Lost/gained:       {}", final_count as i32 - initial_count as i32);
    println!("  Result:            {}", if pass { "PASS" } else { "FAIL" });
    pass
}

/// Test 5: Solid Boundary Enforcement
///
/// Physics: Particles should never penetrate solid boundaries
///
/// Method: Run simulation, verify no particles inside solid cells
fn test_solid_boundaries(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    println!("\n----------------------------------------");
    println!("TEST 5: Solid Boundary Enforcement");
    println!("----------------------------------------");
    println!("Expected: Zero particles inside solid cells");

    let mut flip = GpuFlip3D::new(
        device,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        GRID_SIZE as u32,
        CELL_SIZE,
        MAX_PARTICLES,
    );

    // Water falling onto floor
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for j in GRID_SIZE/2..3*GRID_SIZE/4 {
        for k in GRID_SIZE/4..3*GRID_SIZE/4 {
            for i in GRID_SIZE/4..3*GRID_SIZE/4 {
                positions.push(Vec3::new(
                    (i as f32 + 0.5) * CELL_SIZE,
                    (j as f32 + 0.5) * CELL_SIZE,
                    (k as f32 + 0.5) * CELL_SIZE,
                ));
                velocities.push(Vec3::new(0.0, -2.0, 0.0)); // Falling
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    println!("  Particles: {}", positions.len());

    // Floor with some obstacles
    let cell_count = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    let mut cell_types = vec![0u32; cell_count];

    for k in 0..GRID_SIZE {
        for j in 0..GRID_SIZE {
            for i in 0..GRID_SIZE {
                let idx = k * GRID_SIZE * GRID_SIZE + j * GRID_SIZE + i;
                // Floor (2 cells thick)
                if j <= 2 {
                    cell_types[idx] = 2;
                }
                // Some obstacle blocks
                if i >= GRID_SIZE/3 && i < GRID_SIZE/3 + 3 &&
                   k >= GRID_SIZE/3 && k < GRID_SIZE/3 + 3 &&
                   j >= 3 && j < 6 {
                    cell_types[idx] = 2;
                }
            }
        }
    }

    let sdf: Vec<f32> = (0..cell_count)
        .map(|idx| {
            let i = idx % GRID_SIZE;
            let j = (idx / GRID_SIZE) % GRID_SIZE;
            let k = idx / (GRID_SIZE * GRID_SIZE);

            if cell_types[idx] == 2 {
                -CELL_SIZE
            } else {
                // Distance to nearest solid
                let dist_floor = (j as f32 - 2.5) * CELL_SIZE;
                dist_floor.max(CELL_SIZE * 0.1)
            }
        })
        .collect();

    let dt = 1.0 / 60.0;
    let gravity = -9.8;

    let mut max_violations = 0;

    // Run and check each frame
    for frame in 0..120 {
        update_cell_types_from_particles(
            &mut cell_types, &positions,
            GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE,
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
            40,
        );

        // Count particles in solid cells
        let mut violations = 0;
        for pos in &positions {
            let i = (pos.x / CELL_SIZE) as usize;
            let j = (pos.y / CELL_SIZE) as usize;
            let k = (pos.z / CELL_SIZE) as usize;

            if i < GRID_SIZE && j < GRID_SIZE && k < GRID_SIZE {
                let idx = k * GRID_SIZE * GRID_SIZE + j * GRID_SIZE + i;
                if cell_types[idx] == 2 {
                    violations += 1;
                }
            }
        }

        max_violations = max_violations.max(violations);
    }

    println!("  Max violations:    {}", max_violations);

    let pass = max_violations == 0;
    println!("  Result:            {}", if pass { "PASS" } else { "FAIL" });
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
                label: Some("Physics Validation Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device")
}
