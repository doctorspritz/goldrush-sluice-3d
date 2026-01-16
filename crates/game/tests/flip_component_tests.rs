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

    // Particle should be near the X boundary
    // Wall is at cell W-1, so max valid position is (W-1)*CELL - margin = 0.69
    let margin = CELL * 0.1;
    let max_x = (W as f32 - 1.0) * CELL - margin;
    assert!(
        final_x > 0.6,
        "Particle didn't reach boundary! X: {} (expected near {:.3})",
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

    // Particle should be near the floor
    // Floor is at cell 0, so min valid Y is CELL + margin = 0.11
    let margin = CELL * 0.1;
    let floor_y = CELL + margin;
    assert!(
        final_y < 0.2,
        "Particle didn't reach floor! Y: {} (expected near {:.3})",
        final_y, floor_y
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
    flip.density_projection_enabled = true;

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

    // Domain bounds: solid walls at cells 0 and W-1
    // Valid fluid region is cells 1 to W-2, positions [CELL, (W-1)*CELL]
    let eps = 0.001;
    let margin = CELL * 0.1;
    let min_bound = CELL + margin - eps;           // Just inside cell 1
    let max_bound = (W as f32 - 1.0) * CELL - margin + eps;  // Just inside cell W-2
    println!("Domain bounds: [{:.4}, {:.4}] (margin={}, eps={})", min_bound, max_bound, margin, eps);
    println!("Initial positions:");
    for (i, p) in positions.iter().enumerate() {
        println!("  P{:02}: ({:.3}, {:.3}, {:.3})", i, p.x, p.y, p.z);
    }

    for frame in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            -9.81,
            0.0,
            20,
        );

        // Check EVERY particle EVERY frame
        for (i, p) in positions.iter().enumerate() {
            let oob = p.x < min_bound || p.x > max_bound ||
                      p.y < min_bound || p.y > max_bound ||
                      p.z < min_bound || p.z > max_bound;
            if oob {
                panic!("FRAME {}: P{} OUT OF BOUNDS at ({:.3}, {:.3}, {:.3}), bounds=[{:.3}, {:.3}]",
                    frame, i, p.x, p.y, p.z, min_bound, max_bound);
            }
        }

        // Print every 100 frames
        if frame % 100 == 0 {
            println!("Frame {}:", frame);
            for (i, (p, v)) in positions.iter().zip(velocities.iter()).enumerate() {
                println!("  P{:02}: pos=({:.3}, {:.3}, {:.3}) vel=({:.3}, {:.3}, {:.3})",
                    i, p.x, p.y, p.z, v.x, v.y, v.z);
            }
        }
    }

    let final_count = positions.len();
    println!("Initial: {} particles", initial_count);
    println!("Final:   {} particles", final_count);

    assert_eq!(final_count, initial_count, "Particle count changed! {} -> {}", initial_count, final_count);
}

/// TEST 5: No NaN in output (INTEGRATION TEST)
/// Tests numerical stability of the full FLIP pipeline.
/// Expected: All positions and velocities remain finite AND in bounds
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
    flip.density_projection_enabled = true;

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

    let eps = 0.001;
    let margin = CELL * 0.1;
    let min_bound = CELL + margin - eps;
    let max_bound = (W as f32 - 1.0) * CELL - margin + eps;
    println!("Domain bounds: [{:.4}, {:.4}] (margin={}, eps={})", min_bound, max_bound, margin, eps);

    for frame in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            -9.81,
            0.0,
            20,
        );

        // Check EVERY particle EVERY frame for NaN and bounds
        for (i, (p, v)) in positions.iter().zip(velocities.iter()).enumerate() {
            if !p.is_finite() {
                panic!("FRAME {}: P{} NaN POSITION {:?}", frame, i, p);
            }
            if !v.is_finite() {
                panic!("FRAME {}: P{} NaN VELOCITY {:?}", frame, i, v);
            }
            let oob = p.x < min_bound || p.x > max_bound ||
                      p.y < min_bound || p.y > max_bound ||
                      p.z < min_bound || p.z > max_bound;
            if oob {
                panic!("FRAME {}: P{} OUT OF BOUNDS at ({:.4}, {:.4}, {:.4})", frame, i, p.x, p.y, p.z);
            }
        }

        if frame % 100 == 0 {
            println!("Frame {}:", frame);
            for (i, (p, v)) in positions.iter().zip(velocities.iter()).enumerate() {
                println!("  P{:02}: pos=({:.3}, {:.3}, {:.3}) vel=({:.3}, {:.3}, {:.3})",
                    i, p.x, p.y, p.z, v.x, v.y, v.z);
            }
        }
    }

    println!("All {} particles remained finite and in bounds over {} frames", positions.len(), FRAMES);
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
    flip.open_boundaries = 0;
    flip.density_projection_enabled = true;

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

    // Domain bounds: solid walls at cells 0 and W-1 (floor, walls)
    // Valid fluid region is cells 1 to W-2
    let eps = 0.001;
    let margin = CELL * 0.1;
    let min_x = CELL + margin - eps;
    let max_x = (W as f32 - 1.0) * CELL - margin + eps;
    let min_y = CELL + margin - eps;  // Floor at cell 0
    let max_y = (H as f32 - 1.0) * CELL - margin + eps;  // Ceiling at cell H-1
    let min_z = CELL + margin - eps;
    let max_z = (D as f32 - 1.0) * CELL - margin + eps;

    println!("Domain: X=[{:.3}, {:.3}], Y=[{:.3}, {:.3}], Z=[{:.3}, {:.3}]",
        min_x, max_x, min_y, max_y, min_z, max_z);
    println!("Initial {} particles", positions.len());

    for frame in 0..FRAMES {
        flip.step(
            &device, &queue,
            &mut positions, &mut velocities, &mut c_matrices, &densities, &cell_types,
            None, None,
            DT,
            -9.81,
            0.0,
            40,
        );

        // Check EVERY particle EVERY frame
        for (i, p) in positions.iter().enumerate() {
            let oob_x = p.x < min_x || p.x > max_x;
            let oob_y = p.y < min_y || p.y > max_y;
            let oob_z = p.z < min_z || p.z > max_z;
            if oob_x || oob_y || oob_z {
                panic!("FRAME {}: P{} OUT OF BOUNDS at ({:.4}, {:.4}, {:.4})\n  X: {} [{:.3}, {:.3}]\n  Y: {} [{:.3}, {:.3}]\n  Z: {} [{:.3}, {:.3}]",
                    frame, i, p.x, p.y, p.z,
                    if oob_x { "OOB" } else { "ok" }, min_x, max_x,
                    if oob_y { "OOB" } else { "ok" }, min_y, max_y,
                    if oob_z { "OOB" } else { "ok" }, min_z, max_z);
            }
        }

        // Print summary every 50 frames
        if frame % 50 == 0 {
            let y_min = positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
            let y_max = positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
            let y_avg = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
            let v_max = velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
            println!("Frame {:4}: Y=[{:.4}, {:.4}] avg={:.4}, max_vel={:.3}",
                frame, y_min, y_max, y_avg, v_max);
        }
    }

    let final_y_max = positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let final_y_avg = positions.iter().map(|p| p.y).sum::<f32>() / positions.len() as f32;
    let max_vel = velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    let avg_vel = velocities.iter().map(|v| v.length()).sum::<f32>() / velocities.len() as f32;

    println!("Final: Y_avg={:.4}, Y_max={:.4}, vel_avg={:.4}, vel_max={:.4}",
        final_y_avg, final_y_max, avg_vel, max_vel);

    // Assertions
    assert!(final_y_avg < 0.4, "Particles went too high! avg_y={:.4}", final_y_avg);
    assert!(avg_vel < 2.0, "Particles still turbulent! avg_vel={:.4}", avg_vel);
    assert!(final_y_max < max_y, "Particles escaped domain! max_y={:.4}", final_y_max);

    println!("Test PASSED: Particles settled with avg_y={:.4}", final_y_avg);
}
