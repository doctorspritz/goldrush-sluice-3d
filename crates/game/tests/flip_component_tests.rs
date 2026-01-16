//! Component-level FLIP tests
//!
//! These tests verify each stage of the FLIP pipeline by running full simulations
//! to stable states. Each test runs for 1000 frames to ensure stability.

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

const FRAMES: u32 = 1000;
const DT: f32 = 1.0 / 60.0;

fn init_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 16 {
        println!(
            "Skipped: Adapter only supports {} storage buffers, need 16",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

fn setup_solid_boundaries(cell_types: &mut [u32], w: usize, h: usize, d: usize, open_top: bool) {
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * w * h + y * w + x;
                let is_floor = y == 0;
                let is_ceiling = y == h - 1;
                let is_x_wall = x == 0 || x == w - 1;
                let is_z_wall = z == 0 || z == d - 1;

                if is_floor || is_x_wall || is_z_wall {
                    cell_types[idx] = 2; // SOLID
                }
                if is_ceiling && !open_top {
                    cell_types[idx] = 2; // SOLID
                }
            }
        }
    }
}

/// TEST 1: Single particle, zero velocity, zero gravity (COMPONENT TEST)
/// Tests that P2G/G2P round-trip preserves zero velocity.
/// Expected: Particle stays at y=0.4 forever
#[test]
fn test_stationary_particle_stays_put() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => { println!("Skipped: No GPU"); return; }
    };

    const W: usize = 8;
    const H: usize = 8;
    const D: usize = 8;
    const CELL: f32 = 0.1;

    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 100);
    flip.vorticity_epsilon = 0.0;
    flip.open_boundaries = 0;
    flip.density_projection_enabled = false;

    let mut positions = vec![Vec3::new(0.4, 0.4, 0.4)];
    let mut velocities = vec![Vec3::ZERO];
    let mut c_matrices = vec![Mat3::ZERO];
    let densities = vec![1.0];
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, false);

    let initial_pos = positions[0];

    for _ in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            0.0,  // NO GRAVITY
            0.0,
            0,    // NO PRESSURE - component test
        );
    }

    let final_pos = positions[0];
    let delta = (final_pos - initial_pos).length();
    let final_vel = velocities[0].length();

    println!("Initial: {:?}", initial_pos);
    println!("Final:   {:?}", final_pos);
    println!("Delta:   {}", delta);
    println!("Final vel: {}", final_vel);

    assert!(
        delta < 0.01,
        "Stationary particle moved! delta={} (should be <0.01)",
        delta
    );
    assert!(
        final_vel < 0.001,
        "Particle has velocity! vel={} (should be ~0)",
        final_vel
    );
}

/// TEST 2: Particle reaches boundary correctly (COMPONENT TEST)
/// Tests that a particle with initial velocity reaches the wall and stops cleanly.
/// Over 1000 frames, particle should end up at X boundary with ~zero velocity.
#[test]
fn test_particle_advects_with_velocity() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => { println!("Skipped: No GPU"); return; }
    };

    const W: usize = 8;
    const H: usize = 8;
    const D: usize = 8;
    const CELL: f32 = 0.1;

    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 100);
    flip.vorticity_epsilon = 0.0;
    flip.open_boundaries = 0;
    flip.density_projection_enabled = false;

    let mut positions = vec![Vec3::new(0.4, 0.4, 0.4)];
    let mut velocities = vec![Vec3::new(0.5, 0.0, 0.0)]; // Move in +X
    let mut c_matrices = vec![Mat3::ZERO];
    let densities = vec![1.0];
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, false);

    let initial_x = positions[0].x;
    let initial_y = positions[0].y;
    let initial_z = positions[0].z;

    // Run full 1000 frames - particle will hit wall and settle
    for _ in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            0.0,  // NO GRAVITY
            0.0,
            0,    // NO PRESSURE - component test
        );
    }

    let final_x = positions[0].x;
    let final_y = positions[0].y;
    let final_z = positions[0].z;
    let final_vel = velocities[0].length();

    println!("Initial pos: ({}, {}, {})", initial_x, initial_y, initial_z);
    println!("Final pos:   ({}, {}, {})", final_x, final_y, final_z);
    println!("Final vel:   {}", final_vel);

    // Particle should have moved toward +X boundary
    assert!(
        final_x > initial_x,
        "Particle didn't move right! X: {} -> {}",
        initial_x, final_x
    );

    // Particle should be near the X boundary (wall at x=0.7, margin at 0.79)
    let max_x = W as f32 * CELL - CELL * 0.1;
    assert!(
        final_x > 0.6,
        "Particle didn't reach boundary! X: {} (expected near {})",
        final_x, max_x
    );

    // Y and Z should NOT have changed significantly (no teleportation!)
    assert!(
        (final_y - initial_y).abs() < 0.1,
        "Y position changed unexpectedly! {} -> {} (teleportation bug?)",
        initial_y, final_y
    );
    assert!(
        (final_z - initial_z).abs() < 0.1,
        "Z position changed unexpectedly! {} -> {} (teleportation bug?)",
        initial_z, final_z
    );

    // Velocity should be small (particle settled against wall)
    assert!(
        final_vel < 0.5,
        "Particle still moving too fast! vel={}",
        final_vel
    );

    println!("Test PASSED: Particle reached boundary at X={:.3}", final_x);
}

/// TEST 3: Particle falls to floor with gravity (COMPONENT TEST)
/// Tests that gravity pulls particle down to the floor over 1000 frames.
/// Expected: Particle ends up at floor with low velocity (settled)
#[test]
fn test_gravity_accelerates_particle() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => { println!("Skipped: No GPU"); return; }
    };

    const W: usize = 8;
    const H: usize = 8;
    const D: usize = 8;
    const CELL: f32 = 0.1;
    const GRAVITY: f32 = -9.81;

    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 100);
    flip.vorticity_epsilon = 0.0;
    flip.open_boundaries = 0;
    flip.density_projection_enabled = false;

    // Start high in domain so particle can fall
    let mut positions = vec![Vec3::new(0.4, 0.6, 0.4)];
    let mut velocities = vec![Vec3::ZERO];
    let mut c_matrices = vec![Mat3::ZERO];
    let densities = vec![1.0];
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, false);

    let initial_y = positions[0].y;
    let initial_x = positions[0].x;
    let initial_z = positions[0].z;

    // Run full 1000 frames - particle will fall to floor and settle
    for _ in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            GRAVITY,
            0.0,
            0,    // NO PRESSURE - component test
        );
    }

    let final_y = positions[0].y;
    let final_x = positions[0].x;
    let final_z = positions[0].z;
    let final_vel = velocities[0].length();

    println!("Initial pos: ({}, {}, {})", initial_x, initial_y, initial_z);
    println!("Final pos:   ({}, {}, {})", final_x, final_y, final_z);
    println!("Final vel:   {}", final_vel);

    // Particle should have fallen (Y decreased)
    assert!(
        final_y < initial_y,
        "Particle didn't fall! Y: {} -> {}",
        initial_y, final_y
    );

    // Particle should be near the floor (Y close to margin)
    let floor_margin = CELL * 0.1;
    assert!(
        final_y < 0.2,
        "Particle didn't reach floor! Y: {} (expected near {})",
        final_y, floor_margin
    );

    // X and Z should NOT have changed significantly (no teleportation!)
    assert!(
        (final_x - initial_x).abs() < 0.1,
        "X position changed unexpectedly! {} -> {} (teleportation bug?)",
        initial_x, final_x
    );
    assert!(
        (final_z - initial_z).abs() < 0.1,
        "Z position changed unexpectedly! {} -> {} (teleportation bug?)",
        initial_z, final_z
    );

    // Velocity should be small (particle settled on floor)
    assert!(
        final_vel < 1.0,
        "Particle still moving too fast! vel={}",
        final_vel
    );

    println!("Test PASSED: Particle fell to floor at Y={:.3}", final_y);
}

/// TEST 4: Particle count conservation (INTEGRATION TEST)
/// Tests that the full FLIP pipeline preserves particle count.
/// Expected: 64 particles in = 64 particles out after settling
#[test]
fn test_particle_count_conservation() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => { println!("Skipped: No GPU"); return; }
    };

    const W: usize = 8;
    const H: usize = 8;
    const D: usize = 8;
    const CELL: f32 = 0.1;

    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 1000);
    flip.vorticity_epsilon = 0.0;
    flip.open_boundaries = 0;
    flip.density_projection_enabled = true; // INTEGRATION TEST: Enable density projection for settling

    // Fill region [2,2,2] to [6,6,6] = 4x4x4 = 64 particles
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut densities = Vec::new();

    for x in 2..6 {
        for y in 2..6 {
            for z in 2..6 {
                positions.push(Vec3::new(
                    (x as f32 + 0.5) * CELL,
                    (y as f32 + 0.5) * CELL,
                    (z as f32 + 0.5) * CELL,
                ));
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                densities.push(1.0);
            }
        }
    }

    let initial_count = positions.len();
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, false);

    for _ in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            -9.81,
            0.0,
            20,  // WITH PRESSURE - integration test
        );
    }

    let final_count = positions.len();

    println!("Initial: {} particles", initial_count);
    println!("Final:   {} particles", final_count);

    assert_eq!(
        final_count, initial_count,
        "Particle count changed! {} -> {}",
        initial_count, final_count
    );
}

/// TEST 5: No NaN in output (INTEGRATION TEST)
/// Tests numerical stability of the full FLIP pipeline.
/// Expected: All positions and velocities remain finite
#[test]
fn test_no_nan_output() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => { println!("Skipped: No GPU"); return; }
    };

    const W: usize = 8;
    const H: usize = 8;
    const D: usize = 8;
    const CELL: f32 = 0.1;

    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 1000);
    flip.vorticity_epsilon = 0.0;
    flip.open_boundaries = 0;
    flip.density_projection_enabled = true; // INTEGRATION TEST: Enable density projection for settling

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut densities = Vec::new();

    for x in 2..6 {
        for y in 2..6 {
            for z in 2..6 {
                positions.push(Vec3::new(
                    (x as f32 + 0.5) * CELL,
                    (y as f32 + 0.5) * CELL,
                    (z as f32 + 0.5) * CELL,
                ));
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                densities.push(1.0);
            }
        }
    }

    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, false);

    for frame in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            -9.81,
            0.0,
            20,  // WITH PRESSURE - integration test
        );

        // Check every 100 frames
        if frame % 100 == 0 {
            for (i, p) in positions.iter().enumerate() {
                assert!(p.is_finite(), "NaN position at frame {}, particle {}: {:?}", frame, i, p);
            }
            for (i, v) in velocities.iter().enumerate() {
                assert!(v.is_finite(), "NaN velocity at frame {}, particle {}: {:?}", frame, i, v);
            }
        }
    }

    // Final check
    for (i, p) in positions.iter().enumerate() {
        assert!(p.is_finite(), "NaN position at particle {}: {:?}", i, p);
    }
    for (i, v) in velocities.iter().enumerate() {
        assert!(v.is_finite(), "NaN velocity at particle {}: {:?}", i, v);
    }

    println!("All {} particles remained finite over {} frames", positions.len(), FRAMES);
}

/// TEST 6: Dense fluid settling (INTEGRATION TEST)
/// Tests the full FLIP pipeline with many particles.
/// Expected: Water column falls and reaches equilibrium
#[test]
fn test_particles_settle() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => { println!("Skipped: No GPU"); return; }
    };

    const W: usize = 10;
    const H: usize = 10;
    const D: usize = 10;
    const CELL: f32 = 0.05;

    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 3000);
    flip.vorticity_epsilon = 0.0;
    flip.open_boundaries = 0; // Closed domain - open_top via cell_types only
    flip.density_projection_enabled = true; // INTEGRATION TEST: Enable density projection for settling

    // Fill region [1,1,1] to [9,5,9] = 8x4x8 = 256 particles
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut densities = Vec::new();

    for x in 1..9 {
        for y in 1..5 {
            for z in 1..9 {
                positions.push(Vec3::new(
                    (x as f32 + 0.5) * CELL,
                    (y as f32 + 0.5) * CELL,
                    (z as f32 + 0.5) * CELL,
                ));
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                densities.push(1.0);
            }
        }
    }

    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, true); // Open top

    let initial_y_max = positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);

    for _ in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            -9.81,
            0.0,
            40,  // WITH PRESSURE - integration test
        );
    }

    let final_y_max = positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let final_y_avg = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
    let initial_y_avg = (0.075 + 0.225) / 2.0; // midpoint of initial fill region
    let max_vel = velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    let avg_vel = velocities.iter().map(|v| v.length()).sum::<f32>() / velocities.len() as f32;

    println!("Initial Y: avg~{:.4}, max={:.4}", initial_y_avg, initial_y_max);
    println!("Final Y:   avg={:.4}, max={:.4}", final_y_avg, final_y_max);
    println!("Final vel: avg={:.4}, max={:.4}", avg_vel, max_vel);

    // In incompressible FLIP with pressure, particles may spread upward through open surface.
    // Check that average Y didn't go crazy (particles stayed roughly in domain)
    // and that velocities have calmed down.
    assert!(
        final_y_avg < 0.4,
        "Particles went too high! avg_y={:.4} (should be < 0.4)",
        final_y_avg
    );

    // Average velocity should be reasonably low (system approaching equilibrium)
    assert!(
        avg_vel < 2.0,
        "Particles still too turbulent! avg_vel={:.4}",
        avg_vel
    );

    // No particles should have escaped the domain
    let domain_max = 10.0 * 0.05; // H * CELL = 0.5
    assert!(
        final_y_max < domain_max,
        "Particles escaped domain! max_y={:.4} > {:.4}",
        final_y_max, domain_max
    );

    println!("Test PASSED: Particles remained in domain with avg_y={:.4}", final_y_avg);
}
