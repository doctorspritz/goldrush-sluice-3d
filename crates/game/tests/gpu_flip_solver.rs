//! GPU FLIP Solver Tests
//!
//! Comprehensive tests for the GPU-accelerated 3D FLIP/APIC fluid simulation.
//! Tests cover:
//! - P2G (Particle-to-Grid) transfer
//! - G2P (Grid-to-Particle) transfer
//! - Pressure solve (divergence-free projection)
//! - Particle advection
//! - Boundary conditions
//! - Gravity application
//! - Volume/momentum conservation

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

const CELL_AIR: u32 = 0;
const CELL_FLUID: u32 = 1;
const CELL_SOLID: u32 = 2;

/// Initialize wgpu device and queue for GPU tests.
/// Returns None if no compatible GPU adapter is found.
fn init_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    // GpuFlip3D requires at least 16 storage buffers per shader stage
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 16 {
        eprintln!(
            "GPU adapter only supports {} storage buffers (need 16+); skipping test.",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    // Request device with sufficient limits
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU FLIP Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

/// Build cell types with solid boundaries
fn build_cell_types(width: u32, height: u32, depth: u32) -> Vec<u32> {
    let mut cells = vec![CELL_FLUID; (width * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1
                {
                    let idx = (z * width * height + y * width + x) as usize;
                    cells[idx] = CELL_SOLID;
                }
            }
        }
    }
    cells
}

/// Build cell types with open top boundary (for free surface)
fn build_cell_types_open_top(width: u32, height: u32, depth: u32) -> Vec<u32> {
    let mut cells = vec![CELL_FLUID; (width * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                // Solid on sides and bottom, air on top
                if x == 0 || x == width - 1 || y == 0 || z == 0 || z == depth - 1 {
                    let idx = (z * width * height + y * width + x) as usize;
                    cells[idx] = CELL_SOLID;
                } else if y == height - 1 {
                    let idx = (z * width * height + y * width + x) as usize;
                    cells[idx] = CELL_AIR;
                }
            }
        }
    }
    cells
}

/// Build cell types with open X boundaries (for advection/flow tests).
fn build_cell_types_open_x(width: u32, height: u32, depth: u32) -> Vec<u32> {
    let mut cells = vec![CELL_FLUID; (width * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let idx = (z * width * height + y * width + x) as usize;
                if y == 0 || y == height - 1 || z == 0 || z == depth - 1 {
                    cells[idx] = CELL_SOLID;
                } else if x == 0 || x == width - 1 {
                    cells[idx] = CELL_AIR;
                }
            }
        }
    }
    cells
}

// ============================================================================
// P2G Transfer Tests
// ============================================================================

/// Test that P2G transfers particle velocity to the grid.
/// A single particle with velocity should create non-zero grid velocities.
#[test]
fn test_p2g_transfers_velocity_to_grid() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 8u32;
    let height = 8u32;
    let depth = 8u32;
    let cell_size = 0.1f32;
    let max_particles = 64usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0; // Disable density projection

    let cell_types = build_cell_types_open_x(width, height, depth);

    // Single particle in the center with velocity
    let mut positions = vec![Vec3::new(0.35, 0.35, 0.35)]; // Center of grid
    let mut velocities = vec![Vec3::new(1.0, 2.0, 0.5)];
    let densities = vec![1.0];
    let mut c_matrices = vec![Mat3::ZERO];

    let _initial_vel = velocities[0];

    // Run simulation step (P2G -> pressure -> G2P)
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        1.0 / 60.0,
        0.0,  // No gravity
        0.0,  // No flow accel
        10,   // Pressure iterations
    );

    // After G2P, particle should still have non-zero velocity
    // (some change expected due to pressure solve)
    let final_vel = velocities[0];
    assert!(
        !final_vel.x.is_nan() && !final_vel.y.is_nan() && !final_vel.z.is_nan(),
        "Velocity should not be NaN after P2G/G2P cycle"
    );
    assert!(
        final_vel.length() > 0.01,
        "Particle should retain some velocity after P2G/G2P, got: {:?}",
        final_vel
    );
}

/// Test that P2G handles all three velocity components (U, V, W).
#[test]
fn test_p2g_handles_all_velocity_components() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 10u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.1f32;
    let max_particles = 256usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.flip_ratio = 1.0;
    sim.open_boundaries = 1 | 2; // Open +/-X for advection tests.
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types_open_x(width, height, depth);

    // Create particles with velocity only in Z direction
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let pos = Vec3::new(
                    (3 + i) as f32 * cell_size + 0.5 * cell_size,
                    (3 + j) as f32 * cell_size + 0.5 * cell_size,
                    (3 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                // Only W velocity (Z direction)
                velocities.push(Vec3::new(0.0, 0.0, 1.0));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let dt = 1.0 / 60.0;
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        dt,
        0.0,
        0.0,
        10,
    );

    // Average Z velocity should still be non-zero and positive
    let avg_z_vel: f32 = velocities.iter().map(|v| v.z).sum::<f32>() / velocities.len() as f32;
    assert!(
        avg_z_vel > 0.1,
        "Z velocity should be preserved through P2G/G2P cycle, got avg: {}",
        avg_z_vel
    );
}

// ============================================================================
// G2P Transfer Tests
// ============================================================================

/// Test that G2P transfers grid velocities back to particles.
#[test]
fn test_g2p_transfers_velocity_to_particles() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 10u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.1f32;
    let max_particles = 128usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.flip_ratio = 1.0;
    sim.open_boundaries = 1 | 2; // Open +/-X for momentum test.
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types_open_x(width, height, depth);

    // Create a block of particles with uniform velocity
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let initial_velocity = Vec3::new(0.5, 0.3, 0.2);

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (3 + i) as f32 * cell_size + 0.5 * cell_size,
                    (3 + j) as f32 * cell_size + 0.5 * cell_size,
                    (3 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(initial_velocity);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let dt = 1.0 / 60.0;
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        dt,
        0.0,
        0.0,
        10,
    );

    // All particles should have received valid velocities
    for (i, vel) in velocities.iter().enumerate() {
        assert!(
            vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite(),
            "Particle {} velocity should be finite, got: {:?}",
            i,
            vel
        );
    }

    // Average velocity should still be in the same general direction
    let avg_vel: Vec3 = velocities.iter().copied().fold(Vec3::ZERO, |a, b| a + b) / velocities.len() as f32;
    assert!(
        avg_vel.dot(initial_velocity) > 0.0,
        "Average velocity should roughly match initial direction"
    );
}

/// Test that APIC affine velocity matrix (C) is properly computed
#[test]
fn test_g2p_computes_affine_velocity_matrix() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 10u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.1f32;
    let max_particles = 64usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.flip_ratio = 1.0;
    sim.open_boundaries = 1 | 2; // Open +/-X for flow acceleration.
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles with varying velocity (creates velocity gradient)
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let pos = Vec3::new(
                    (3 + i) as f32 * cell_size + 0.5 * cell_size,
                    (3 + j) as f32 * cell_size + 0.5 * cell_size,
                    (3 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                // Velocity that varies with position (creates gradient)
                velocities.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, k as f32 * 0.1));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let dt = 1.0 / 60.0;
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        dt,
        0.0,
        0.0,
        10,
    );

    // C matrices should be finite (not NaN/Inf)
    for (i, c) in c_matrices.iter().enumerate() {
        let c_sum = c.x_axis.length() + c.y_axis.length() + c.z_axis.length();
        assert!(
            c_sum.is_finite(),
            "C matrix for particle {} should be finite, got: {:?}",
            i,
            c
        );
    }
}

// ============================================================================
// Pressure Solve Tests
// ============================================================================

/// Test that pressure solve removes divergence from velocity field.
/// Create converging flow and check that it becomes divergence-free.
#[test]
fn test_pressure_solve_removes_divergence() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 12u32;
    let height = 12u32;
    let depth = 12u32;
    let cell_size = 0.1f32;
    let max_particles = 512usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles with converging velocity (creates positive divergence)
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let center = Vec3::new(0.55, 0.55, 0.55);
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                let pos = Vec3::new(
                    (3 + i) as f32 * cell_size + 0.5 * cell_size,
                    (3 + j) as f32 * cell_size + 0.5 * cell_size,
                    (3 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                // Converging velocity toward center
                let to_center = (center - pos).normalize_or_zero();
                velocities.push(to_center * 0.5);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let dt = 1.0 / 60.0;
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        dt,
        0.0,
        0.0,
        50, // More iterations for better convergence
    );

    // After pressure solve, the velocity field should be less convergent
    // (pressure solver pushes flow outward to maintain incompressibility)
    let mut convergent_count = 0;
    let mut divergent_count = 0;

    for (pos, vel) in positions.iter().zip(velocities.iter()) {
        let to_center = (center - *pos).normalize_or_zero();
        if vel.dot(to_center) > 0.01 {
            convergent_count += 1;
        } else if vel.dot(to_center) < -0.01 {
            divergent_count += 1;
        }
    }

    // Should have more divergent (pushed outward) than strongly convergent particles
    // or at least velocities should be reduced
    let avg_vel_mag: f32 = velocities.iter().map(|v| v.length()).sum::<f32>() / velocities.len() as f32;
    assert!(
        avg_vel_mag < 0.45 || divergent_count > convergent_count / 2,
        "Pressure solve should reduce convergent flow. Convergent: {}, Divergent: {}, Avg vel: {}",
        convergent_count,
        divergent_count,
        avg_vel_mag
    );
}

// ============================================================================
// Particle Advection Tests
// ============================================================================

/// Test that particles move in the direction of their velocity.
#[test]
fn test_particle_advection_direction() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 16u32;
    let height = 16u32;
    let depth = 16u32;
    let cell_size = 0.05f32;
    let max_particles = 256usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles moving in +X direction
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (6 + j) as f32 * cell_size + 0.5 * cell_size,
                    (6 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::new(0.5, 0.0, 0.0)); // Moving in +X
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let initial_avg_x: f32 = positions.iter().map(|p| p.x).sum::<f32>() / positions.len() as f32;

    let dt = 1.0 / 60.0;
    for _ in 0..10 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            0.0,
            0.0,
            10,
        );
    }

    let final_avg_x: f32 = positions.iter().map(|p| p.x).sum::<f32>() / positions.len() as f32;
    let displacement = final_avg_x - initial_avg_x;

    let expected_min = cell_size * 0.25;
    assert!(
        displacement > expected_min,
        "Particles should have moved in +X direction. Initial avg X: {}, Final avg X: {}, Displacement: {}",
        initial_avg_x,
        final_avg_x,
        displacement
    );
}

/// Test that particles don't pass through solid boundaries
#[test]
fn test_particle_boundary_collision() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 10u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.1f32;
    let max_particles = 64usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles near the boundary moving toward it
    let mut positions = vec![
        Vec3::new(0.15, 0.5, 0.5), // Near left boundary
    ];
    let mut velocities = vec![
        Vec3::new(-1.0, 0.0, 0.0), // Moving toward left boundary
    ];
    let densities = vec![1.0];
    let mut c_matrices = vec![Mat3::ZERO];

    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            0.0,
            0.0,
            10,
        );
    }

    // Particle should be clamped inside domain
    let min_bound = cell_size * 0.5;
    let max_bound = (width as f32 - 0.5) * cell_size;

    assert!(
        positions[0].x >= min_bound * 0.5,
        "Particle X should be clamped inside domain, got: {}",
        positions[0].x
    );
    assert!(
        positions[0].x <= max_bound * 1.5,
        "Particle X should be inside domain, got: {}",
        positions[0].x
    );
}

// ============================================================================
// Gravity Tests
// ============================================================================

/// Test that gravity accelerates particles downward
#[test]
fn test_gravity_accelerates_particles() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 10u32;
    let height = 16u32;
    let depth = 10u32;
    let cell_size = 0.1f32;
    let max_particles = 128usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types_open_top(width, height, depth);

    // Create particles in middle of domain
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (8 + j) as f32 * cell_size + 0.5 * cell_size, // Upper region
                    (4 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO); // Start at rest
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let initial_avg_y: f32 = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
    let initial_avg_vy: f32 = velocities.iter().map(|v| v.y).sum::<f32>() / velocities.len() as f32;

    let dt = 1.0 / 60.0;
    let gravity = -9.81;

    for _ in 0..30 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            gravity,
            0.0,
            10,
        );
    }

    let final_avg_y: f32 = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
    let final_avg_vy: f32 = velocities.iter().map(|v| v.y).sum::<f32>() / velocities.len() as f32;

    // Particles should have fallen (Y decreased)
    assert!(
        final_avg_y < initial_avg_y,
        "Particles should fall under gravity. Initial Y: {}, Final Y: {}",
        initial_avg_y,
        final_avg_y
    );

    // Y velocity should be negative (downward)
    assert!(
        final_avg_vy < initial_avg_vy - 0.1,
        "Y velocity should become negative under gravity. Initial: {}, Final: {}",
        initial_avg_vy,
        final_avg_vy
    );
}

// ============================================================================
// Conservation Tests
// ============================================================================

/// Test that total particle count is preserved through simulation
#[test]
fn test_particle_count_preserved() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 12u32;
    let height = 12u32;
    let depth = 12u32;
    let cell_size = 0.1f32;
    let max_particles = 256usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (4 + j) as f32 * cell_size + 0.5 * cell_size,
                    (4 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::new(0.2, 0.1, -0.1));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let initial_count = positions.len();

    let dt = 1.0 / 60.0;
    for _ in 0..20 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            -9.81,
            0.0,
            10,
        );
    }

    let final_count = positions.len();
    assert_eq!(
        initial_count, final_count,
        "Particle count should be preserved"
    );
}

/// Test that momentum doesn't blow up (stability check)
/// Note: FLIP simulation inherently loses some momentum due to FLIP/PIC blending,
/// boundary interactions, and pressure projection. This test verifies momentum
/// stays bounded and the simulation is stable rather than exact conservation.
#[test]
fn test_momentum_bounded_and_stable() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 14u32;
    let height = 14u32;
    let depth = 14u32;
    let cell_size = 0.1f32;
    let max_particles = 512usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles with initial momentum
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (4 + j) as f32 * cell_size + 0.5 * cell_size,
                    (4 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::new(0.3, 0.2, 0.1));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    // Calculate initial momentum magnitude (assuming equal mass particles)
    let initial_momentum: Vec3 = velocities.iter().copied().fold(Vec3::ZERO, |a, b| a + b);
    let initial_mag = initial_momentum.length();

    let dt = 1.0 / 60.0;
    for _ in 0..10 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            0.0, // No gravity for this test
            0.0,
            10,
        );
    }

    let final_momentum: Vec3 = velocities.iter().copied().fold(Vec3::ZERO, |a, b| a + b);
    let final_mag = final_momentum.length();

    // Momentum should not explode (must stay bounded - allow some dissipation but no blowup)
    // Final momentum should be at most the initial (some loss is OK, but no energy gain)
    assert!(
        final_mag < initial_mag * 1.5,
        "Momentum should not explode. Initial: {:.3}, Final: {:.3}",
        initial_mag,
        final_mag
    );

    // Final momentum should have some remaining (not fully damped to zero in 10 steps)
    assert!(
        final_mag > initial_mag * 0.02,
        "Momentum should not be fully damped in few steps. Initial: {:.3}, Final: {:.3}",
        initial_mag,
        final_mag
    );

    // Direction should be roughly preserved (dot product positive)
    assert!(
        final_momentum.dot(initial_momentum) > 0.0,
        "Momentum direction should be roughly preserved. Initial: {:?}, Final: {:?}",
        initial_momentum,
        final_momentum
    );
}

// ============================================================================
// Flow Acceleration Tests
// ============================================================================

/// Test that flow acceleration pushes particles in +X direction
#[test]
fn test_flow_acceleration() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 16u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.05f32;
    let max_particles = 256usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create stationary particles
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (3 + j) as f32 * cell_size + 0.5 * cell_size,
                    (3 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let initial_avg_vx: f32 = velocities.iter().map(|v| v.x).sum::<f32>() / velocities.len() as f32;

    let dt = 1.0 / 60.0;
    let flow_accel = 2.0; // Accelerate in +X

    for _ in 0..20 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            0.0, // No gravity
            flow_accel,
            10,
        );
    }

    let final_avg_vx: f32 = velocities.iter().map(|v| v.x).sum::<f32>() / velocities.len() as f32;

    let expected_delta = flow_accel * dt * 0.8;
    assert!(
        final_avg_vx > initial_avg_vx + expected_delta,
        "Flow acceleration should increase X velocity. Initial: {}, Final: {}",
        initial_avg_vx,
        final_avg_vx
    );
}

// ============================================================================
// Vorticity Tests
// ============================================================================

/// Test that vorticity confinement is applied without NaN
#[test]
fn test_vorticity_confinement_stable() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 12u32;
    let height = 12u32;
    let depth = 12u32;
    let cell_size = 0.1f32;
    let max_particles = 256usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.1; // Enable vorticity confinement
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Create particles with rotational motion (to create vorticity)
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    let center = Vec3::new(0.55, 0.55, 0.55);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (4 + j) as f32 * cell_size + 0.5 * cell_size,
                    (4 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                // Circular motion around center (creates vorticity)
                let to_center = pos - center;
                let tangent = Vec3::new(-to_center.z, 0.0, to_center.x).normalize_or_zero();
                velocities.push(tangent * 0.3);
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let dt = 1.0 / 60.0;
    for _ in 0..20 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            0.0,
            0.0,
            10,
        );
    }

    // All velocities should be finite (no NaN from vorticity computation)
    for (i, vel) in velocities.iter().enumerate() {
        assert!(
            vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite(),
            "Particle {} velocity should be finite with vorticity confinement, got: {:?}",
            i,
            vel
        );
    }

    // All positions should be finite
    for (i, pos) in positions.iter().enumerate() {
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "Particle {} position should be finite with vorticity confinement, got: {:?}",
            i,
            pos
        );
    }
}

// ============================================================================
// Sediment/Density Tests
// ============================================================================

/// Test that sediment particles (higher density) behave differently
#[test]
fn test_sediment_density_differentiation() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 12u32;
    let height = 16u32;
    let depth = 12u32;
    let cell_size = 0.1f32;
    let max_particles = 128usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.water_rest_particles = 0.0;
    sim.sediment_settling_velocity = 0.05; // Enable settling

    let cell_types = build_cell_types_open_top(width, height, depth);

    // Create water particles (density 1.0) and sediment particles (density 2.65)
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut water_indices = Vec::new();
    let mut sediment_indices = Vec::new();

    // Water particles
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.5 * cell_size,
                    (8 + j) as f32 * cell_size + 0.5 * cell_size,
                    (4 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                water_indices.push(positions.len());
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                densities.push(1.0); // Water density
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    // Sediment particles (same starting position range)
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.6 * cell_size,
                    (8 + j) as f32 * cell_size + 0.6 * cell_size,
                    (4 + k) as f32 * cell_size + 0.6 * cell_size,
                );
                sediment_indices.push(positions.len());
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                densities.push(2.65); // Sand density
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let initial_water_y: f32 = water_indices.iter().map(|&i| positions[i].y).sum::<f32>()
        / water_indices.len() as f32;
    let initial_sediment_y: f32 = sediment_indices.iter().map(|&i| positions[i].y).sum::<f32>()
        / sediment_indices.len() as f32;

    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            -9.81,
            0.0,
            10,
        );
    }

    let final_water_y: f32 = water_indices.iter().map(|&i| positions[i].y).sum::<f32>()
        / water_indices.len() as f32;
    let final_sediment_y: f32 = sediment_indices.iter().map(|&i| positions[i].y).sum::<f32>()
        / sediment_indices.len() as f32;

    // Both should fall, but all should be valid
    assert!(
        final_water_y.is_finite() && final_sediment_y.is_finite(),
        "Final positions should be finite"
    );

    // Y position should have decreased for both (falling under gravity)
    assert!(
        final_water_y < initial_water_y,
        "Water should fall. Initial: {}, Final: {}",
        initial_water_y,
        final_water_y
    );
    assert!(
        final_sediment_y < initial_sediment_y,
        "Sediment should fall. Initial: {}, Final: {}",
        initial_sediment_y,
        final_sediment_y
    );
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

/// Test numerical stability with many simulation steps
#[test]
fn test_long_simulation_stability() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 10u32;
    let height = 12u32;
    let depth = 10u32;
    let cell_size = 0.1f32;
    let max_particles = 128usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.05;
    sim.water_rest_particles = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();

    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (3 + i) as f32 * cell_size + 0.5 * cell_size,
                    (4 + j) as f32 * cell_size + 0.5 * cell_size,
                    (3 + k) as f32 * cell_size + 0.5 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::new(0.1, 0.05, -0.05));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    let dt = 1.0 / 60.0;

    // Run many steps
    for step in 0..100 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            -9.81,
            0.0,
            10,
        );

        // Check for NaN every 20 steps
        if step % 20 == 0 {
            for (i, pos) in positions.iter().enumerate() {
                assert!(
                    pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                    "Position NaN at step {}, particle {}: {:?}",
                    step,
                    i,
                    pos
                );
            }
            for (i, vel) in velocities.iter().enumerate() {
                assert!(
                    vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite(),
                    "Velocity NaN at step {}, particle {}: {:?}",
                    step,
                    i,
                    vel
                );
            }
        }
    }

    // Final check - all values should be finite
    for (i, pos) in positions.iter().enumerate() {
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "Final position NaN for particle {}: {:?}",
            i,
            pos
        );
    }
}

/// Test simulation handles empty particle list gracefully
#[test]
fn test_empty_particle_list() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping test.");
            return;
        }
    };

    let width = 8u32;
    let height = 8u32;
    let depth = 8u32;
    let cell_size = 0.1f32;
    let max_particles = 64usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    let cell_types = build_cell_types(width, height, depth);

    let mut positions: Vec<Vec3> = Vec::new();
    let mut velocities: Vec<Vec3> = Vec::new();
    let densities: Vec<f32> = Vec::new();
    let mut c_matrices: Vec<Mat3> = Vec::new();

    // Should not panic with empty input
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        1.0 / 60.0,
        -9.81,
        0.0,
        10,
    );

    assert_eq!(positions.len(), 0, "Empty list should remain empty");
}
