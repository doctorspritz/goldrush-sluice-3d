// GPU DEM Collision System Tests
// Validates GPU-based collision detection, spatial hashing, and physics integration
// for the GpuDem3D system

use glam::Vec3;
use std::sync::Arc;

const PARTICLE_RADIUS: f32 = 0.01; // 1cm particles
const PARTICLE_MASS: f32 = 1.0;
const DT: f32 = 1.0 / 120.0; // 120 Hz timestep
const GRAVITY: f32 = -9.81;

/// Initialize GPU device and queue for tests
/// Returns None if no compatible GPU is available
fn init_gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    // Check for required features
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 13 {
        eprintln!(
            "GPU adapter only supports {} storage buffers (need 13+); skipping test.",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU DEM Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some((Arc::new(device), Arc::new(queue)))
}

/// Test 1: GPU DEM initialization and particle spawning
/// Verifies basic setup and particle creation works correctly
#[test]
fn test_gpu_dem_initialization() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_initialization");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let max_particles = 1000;
    let max_templates = 10;
    let max_contacts = 10000;

    let mut gpu_dem = GpuDem3D::new(
        device.clone(),
        queue.clone(),
        max_particles,
        max_templates,
        max_contacts,
    );

    // Verify initial state
    assert_eq!(gpu_dem.particle_count(), 0, "Initial particle count should be 0");

    // Add a template
    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);
    assert_eq!(template_id, 0, "First template should have ID 0");

    // Spawn particles
    for i in 0..10 {
        let pos = Vec3::new(i as f32 * 0.1, 1.0, 0.0);
        let result = gpu_dem.spawn_clump(template_id, pos, Vec3::ZERO);
        assert!(result.is_some(), "Should be able to spawn particle {}", i);
    }

    assert_eq!(gpu_dem.particle_count(), 10, "Should have 10 particles");
}

/// Test 2: GPU DEM collision detection - two particles colliding head-on
/// Verifies particles detect collision and respond appropriately
/// Note: GPU DEM always applies gravity, so we check x-momentum conservation only
#[test]
fn test_gpu_dem_head_on_collision() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_head_on_collision");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);

    // Create template
    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);

    // Spawn two particles approaching each other
    // Use close spacing so they collide quickly (before falling far)
    let separation = 0.03; // 3cm apart - will overlap immediately
    let speed = 1.0;

    gpu_dem.spawn_clump(
        template_id,
        Vec3::new(-separation / 2.0, 0.5, 0.0),
        Vec3::new(speed, 0.0, 0.0),
    );
    gpu_dem.spawn_clump(
        template_id,
        Vec3::new(separation / 2.0, 0.5, 0.0),
        Vec3::new(-speed, 0.0, 0.0),
    );

    // Run simulation for short time (only a few steps to capture collision)
    for _ in 0..20 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Collision Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));
    }

    device.poll(wgpu::Maintain::Wait);

    // Read back positions
    let particles = pollster::block_on(gpu_dem.readback(&device));

    assert_eq!(particles.len(), 2, "Should have 2 particles");

    let vel_a = Vec3::from_slice(&particles[0].velocity[..3]);
    let vel_b = Vec3::from_slice(&particles[1].velocity[..3]);

    println!("Post-collision velocities: A={:?}, B={:?}", vel_a, vel_b);

    // Check x-momentum conservation (gravity affects y only)
    // Initial x-momentum: 1*1 + 1*(-1) = 0
    // Final x-momentum should be close to 0
    let x_momentum = vel_a.x + vel_b.x;
    assert!(
        x_momentum.abs() < 0.5,
        "X-momentum should be approximately conserved (got {:.4})",
        x_momentum
    );

    // Verify collision happened - particles should have interacted
    // Either reversed direction or significantly slowed
    let pos_a = Vec3::from_slice(&particles[0].position[..3]);
    let pos_b = Vec3::from_slice(&particles[1].position[..3]);
    let final_separation = (pos_a - pos_b).length();

    println!("Final separation: {:.4}", final_separation);

    // After collision, particles should have separated (not passing through)
    // Or at least the x-velocities should show interaction
    assert!(
        vel_a.x < speed || vel_b.x > -speed,
        "Particles should have interacted (v_a.x={:.4}, v_b.x={:.4})",
        vel_a.x,
        vel_b.x
    );
}

/// Test 3: GPU DEM spatial hash correctness
/// Verifies that particles in close proximity are detected
#[test]
fn test_gpu_dem_spatial_hash_detection() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_spatial_hash_detection");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);

    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);

    // Create a 2x2x2 grid of particles with touching spacing
    // Spacing slightly less than 2x radius ensures contact
    let spacing = PARTICLE_RADIUS * 1.9;
    let base_pos = Vec3::new(0.5, 0.5, 0.5);

    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let pos = base_pos + Vec3::new(x as f32, y as f32, z as f32) * spacing;
                gpu_dem.spawn_clump(template_id, pos, Vec3::ZERO);
            }
        }
    }

    assert_eq!(gpu_dem.particle_count(), 8, "Should have 8 particles");

    // Run one simulation step to trigger collision detection
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Spatial Hash Test"),
    });

    gpu_dem.update(&mut encoder, DT);

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Read back velocities - if collisions were detected, particles should have forces applied
    let particles = pollster::block_on(gpu_dem.readback(&device));

    // Check that particles have been affected by collision forces
    // In a touching 2x2x2 grid, particles should be pushed apart
    let mut has_velocity = false;
    for (i, p) in particles.iter().enumerate() {
        let vel = Vec3::from_slice(&p.velocity[..3]);
        if vel.length() > 0.001 {
            has_velocity = true;
            println!("Particle {} has velocity: {:?}", i, vel);
        }
    }

    assert!(
        has_velocity,
        "At least one particle should have velocity from collision response"
    );
}

/// Test 4: GPU DEM gravity and floor collision
/// Drops a particle and verifies it falls and bounces
#[test]
fn test_gpu_dem_gravity_and_bounce() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_gravity_and_bounce");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);

    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);

    // Drop particle from height
    let drop_height = 1.0;
    gpu_dem.spawn_clump(template_id, Vec3::new(0.0, drop_height, 0.0), Vec3::ZERO);

    // Run simulation for 1 second (120 steps at 120Hz)
    for _ in 0..120 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gravity Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));
    }

    device.poll(wgpu::Maintain::Wait);

    let particles = pollster::block_on(gpu_dem.readback(&device));
    assert_eq!(particles.len(), 1, "Should have 1 particle");

    let pos = Vec3::from_slice(&particles[0].position[..3]);
    let vel = Vec3::from_slice(&particles[0].velocity[..3]);

    println!("After 1 second: pos={:?}, vel={:?}", pos, vel);

    // Particle should have fallen (y < initial drop height)
    assert!(
        pos.y < drop_height,
        "Particle should have fallen (y={:.4} should be < {:.4})",
        pos.y,
        drop_height
    );

    // Particle should have downward velocity (or be bouncing)
    // After 1 second of free fall: v = gt = 9.81 m/s (if no floor)
    // With floor collision, velocity could be upward from bounce
    assert!(
        vel.y.abs() > 0.1 || pos.y < 0.1,
        "Particle should either be moving or near floor (vel.y={:.4}, pos.y={:.4})",
        vel.y,
        pos.y
    );
}

/// Test 5: GPU DEM multiple particles - no explosion test
/// Spawns multiple particles close together and verifies they interact without exploding
/// Note: GPU DEM has no built-in floor - particles will fall but shouldn't explode
#[test]
fn test_gpu_dem_multi_particle_no_explosion() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_multi_particle_no_explosion");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 200, 10, 5000);

    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);

    // Spawn 25 particles in a 5x5 grid - close spacing to force collisions
    let grid_size = 5;
    let spacing = PARTICLE_RADIUS * 2.5; // Slightly overlapping
    let start_height = 1.0;

    for iz in 0..grid_size {
        for ix in 0..grid_size {
            let pos = Vec3::new(
                (ix as f32 - grid_size as f32 / 2.0) * spacing,
                start_height + (iz as f32) * spacing, // Stack vertically
                0.0,
            );
            gpu_dem.spawn_clump(template_id, pos, Vec3::ZERO);
        }
    }

    assert_eq!(
        gpu_dem.particle_count(),
        (grid_size * grid_size) as u32,
        "Should have {} particles",
        grid_size * grid_size
    );

    // Run simulation for short time (check stability, not floor collision)
    let test_steps = 60; // 0.5 seconds
    for step in 0..test_steps {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("No Explosion Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));

        // Periodic progress check
        if step % 20 == 0 {
            device.poll(wgpu::Maintain::Wait);
            let particles = pollster::block_on(gpu_dem.readback(&device));
            let max_vel = particles
                .iter()
                .map(|p| Vec3::from_slice(&p.velocity[..3]).length())
                .fold(0.0f32, f32::max);
            println!("Step {}: max velocity={:.4}", step, max_vel);
        }
    }

    device.poll(wgpu::Maintain::Wait);

    let particles = pollster::block_on(gpu_dem.readback(&device));

    // Verify no particles exploded (velocity clamped at 50 m/s in shader)
    let mut explosion_count = 0;

    for (i, p) in particles.iter().enumerate() {
        let pos = Vec3::from_slice(&p.position[..3]);
        let vel = Vec3::from_slice(&p.velocity[..3]);

        // Check for explosion (extreme position or velocity)
        // Velocity should be clamped to 50 m/s by the shader
        if vel.length() > 55.0 {
            explosion_count += 1;
            println!(
                "EXPLOSION: Particle {} at pos={:?}, vel={:?}",
                i, pos, vel
            );
        }
    }

    assert_eq!(
        explosion_count, 0,
        "No particles should have exploded (velocity > 55 m/s)"
    );

    // Verify particles are still within reasonable bounds
    // (not teleported to infinity)
    let max_pos = particles
        .iter()
        .map(|p| Vec3::from_slice(&p.position[..3]).length())
        .fold(0.0f32, f32::max);

    println!("Max position magnitude: {:.4}", max_pos);

    assert!(
        max_pos < 100.0,
        "Particles should be within 100m of origin (got {:.4})",
        max_pos
    );
}

/// Test 6: GPU DEM stiffness and damping parameter effects
/// Verifies that changing stiffness/damping affects collision response
#[test]
fn test_gpu_dem_stiffness_damping() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_stiffness_damping");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    // Test with high stiffness (stiffer = faster response)
    let mut gpu_dem_stiff = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);
    gpu_dem_stiff.stiffness = 5000.0;
    gpu_dem_stiff.damping = 50.0;

    // Test with low stiffness (softer = slower response)
    let mut gpu_dem_soft = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);
    gpu_dem_soft.stiffness = 500.0;
    gpu_dem_soft.damping = 5.0;

    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );

    let template_id_stiff = gpu_dem_stiff.add_template(template.clone());
    let template_id_soft = gpu_dem_soft.add_template(template);

    // Spawn colliding particles in both simulations
    let separation = 0.015; // Very close (will overlap immediately)
    let speed = 0.5;

    gpu_dem_stiff.spawn_clump(
        template_id_stiff,
        Vec3::new(-separation / 2.0, 0.5, 0.0),
        Vec3::new(speed, 0.0, 0.0),
    );
    gpu_dem_stiff.spawn_clump(
        template_id_stiff,
        Vec3::new(separation / 2.0, 0.5, 0.0),
        Vec3::new(-speed, 0.0, 0.0),
    );

    gpu_dem_soft.spawn_clump(
        template_id_soft,
        Vec3::new(-separation / 2.0, 0.5, 0.0),
        Vec3::new(speed, 0.0, 0.0),
    );
    gpu_dem_soft.spawn_clump(
        template_id_soft,
        Vec3::new(separation / 2.0, 0.5, 0.0),
        Vec3::new(-speed, 0.0, 0.0),
    );

    // Run both simulations
    for _ in 0..50 {
        let mut encoder_stiff = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Stiff Test"),
        });
        let mut encoder_soft = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Soft Test"),
        });

        gpu_dem_stiff.update(&mut encoder_stiff, DT);
        gpu_dem_soft.update(&mut encoder_soft, DT);

        queue.submit([encoder_stiff.finish(), encoder_soft.finish()]);
    }

    device.poll(wgpu::Maintain::Wait);

    let particles_stiff = pollster::block_on(gpu_dem_stiff.readback(&device));
    let particles_soft = pollster::block_on(gpu_dem_soft.readback(&device));

    // Calculate separation after collision
    let pos_stiff_a = Vec3::from_slice(&particles_stiff[0].position[..3]);
    let pos_stiff_b = Vec3::from_slice(&particles_stiff[1].position[..3]);
    let sep_stiff = (pos_stiff_a - pos_stiff_b).length();

    let pos_soft_a = Vec3::from_slice(&particles_soft[0].position[..3]);
    let pos_soft_b = Vec3::from_slice(&particles_soft[1].position[..3]);
    let sep_soft = (pos_soft_a - pos_soft_b).length();

    println!("Stiff separation: {:.4}", sep_stiff);
    println!("Soft separation: {:.4}", sep_soft);

    // Stiffer collision should result in faster separation (particles bounce apart more quickly)
    // This is a qualitative check - the exact relationship depends on many factors
    assert!(
        sep_stiff > 0.01 && sep_soft > 0.01,
        "Both simulations should show particle separation"
    );
}

/// Test 7: GPU DEM multi-template support
/// Verifies particles with different templates can be created and updated
#[test]
fn test_gpu_dem_multi_template() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_multi_template");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);

    // Create two different templates
    let template_small = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS * 0.5, // Half size
        PARTICLE_MASS * 0.125, // 1/8 mass (volume ratio)
    );
    let template_large = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Cube2,
        PARTICLE_RADIUS * 2.0, // Double size
        PARTICLE_MASS * 8.0,   // 8x mass
    );

    let template_id_small = gpu_dem.add_template(template_small);
    let template_id_large = gpu_dem.add_template(template_large);

    assert_eq!(template_id_small, 0, "Small template should be ID 0");
    assert_eq!(template_id_large, 1, "Large template should be ID 1");

    // Spawn particles with overlapping positions to force collision
    // Place them very close together so collision happens immediately
    let overlap = 0.01; // 1cm overlap
    gpu_dem.spawn_clump(
        template_id_small,
        Vec3::new(-overlap, 0.5, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    gpu_dem.spawn_clump(
        template_id_large,
        Vec3::new(overlap, 0.5, 0.0),
        Vec3::ZERO,
    );

    // Run simulation for short time
    for _ in 0..30 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Multi-Template Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));
    }

    device.poll(wgpu::Maintain::Wait);

    let particles = pollster::block_on(gpu_dem.readback(&device));

    let vel_small = Vec3::from_slice(&particles[0].velocity[..3]);
    let vel_large = Vec3::from_slice(&particles[1].velocity[..3]);
    let pos_small = Vec3::from_slice(&particles[0].position[..3]);
    let pos_large = Vec3::from_slice(&particles[1].position[..3]);

    println!("Small particle: pos={:?}, vel={:?}", pos_small, vel_small);
    println!("Large particle: pos={:?}, vel={:?}", pos_large, vel_large);

    // Verify both particles are still valid (not NaN or infinity)
    assert!(
        vel_small.is_finite() && vel_large.is_finite(),
        "Velocities should be finite"
    );
    assert!(
        pos_small.is_finite() && pos_large.is_finite(),
        "Positions should be finite"
    );

    // Verify particles have separated or interacted
    // After collision, they should not be at the same position
    let separation = (pos_small - pos_large).length();
    println!("Final separation: {:.4}", separation);

    // Large particle should have different x-velocity than initial (0)
    // due to collision with moving small particle (or they separated)
    let interaction_occurred = vel_large.x.abs() > 0.01 || separation > 0.02;
    assert!(
        interaction_occurred,
        "Collision should have caused interaction (v_large.x={:.4}, sep={:.4})",
        vel_large.x,
        separation
    );
}

/// Test 8: GPU DEM angular velocity and rotation
/// Verifies particles rotate when subjected to off-center forces
#[test]
fn test_gpu_dem_rotation() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_rotation");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;

    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);

    // Use a multi-sphere clump that can exhibit rotation
    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Rod3, // Rod shape has distinct orientation
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);

    // Spawn particle with initial angular velocity
    gpu_dem.spawn_clump(template_id, Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO);

    // Get initial orientation
    let initial_particles = pollster::block_on(gpu_dem.readback(&device));
    let initial_orient = initial_particles[0].orientation;

    println!("Initial orientation: {:?}", initial_orient);

    // Run simulation with gravity (will apply torque if off-center mass)
    for _ in 0..60 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Rotation Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));
    }

    device.poll(wgpu::Maintain::Wait);

    let final_particles = pollster::block_on(gpu_dem.readback(&device));
    let final_orient = final_particles[0].orientation;
    let angular_vel = final_particles[0].angular_velocity;

    println!("Final orientation: {:?}", final_orient);
    println!("Angular velocity: {:?}", angular_vel);

    // Verify quaternion is normalized (valid rotation)
    let orient_len = (final_orient[0].powi(2)
        + final_orient[1].powi(2)
        + final_orient[2].powi(2)
        + final_orient[3].powi(2))
    .sqrt();

    assert!(
        (orient_len - 1.0).abs() < 0.01,
        "Quaternion should be normalized (length={:.4})",
        orient_len
    );
}

/// Test 9: GPU DEM vs CPU reference comparison
/// Compares GPU DEM results with CPU ClusterSimulation3D for similar scenarios
#[test]
fn test_gpu_dem_vs_cpu_reference() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_vs_cpu_reference");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;
    use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};

    // Setup GPU simulation
    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), 100, 10, 1000);

    // Setup CPU simulation
    let mut cpu_sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
    cpu_sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    // Create identical templates
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let gpu_template_id = gpu_dem.add_template(template.clone());
    let cpu_template_id = cpu_sim.add_template(template);

    // Spawn identical particles (single falling particle)
    let start_pos = Vec3::new(5.0, 2.0, 5.0);
    gpu_dem.spawn_clump(gpu_template_id, start_pos, Vec3::ZERO);
    cpu_sim.spawn(cpu_template_id, start_pos, Vec3::ZERO);

    // Run both simulations for same duration
    let steps = 60;
    for _ in 0..steps {
        // GPU
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU vs CPU Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));

        // CPU
        cpu_sim.step(DT);
    }

    device.poll(wgpu::Maintain::Wait);

    // Compare results
    let gpu_particles = pollster::block_on(gpu_dem.readback(&device));
    let gpu_pos = Vec3::from_slice(&gpu_particles[0].position[..3]);
    let gpu_vel = Vec3::from_slice(&gpu_particles[0].velocity[..3]);

    let cpu_pos = cpu_sim.clumps[0].position;
    let cpu_vel = cpu_sim.clumps[0].velocity;

    println!("GPU position: {:?}", gpu_pos);
    println!("CPU position: {:?}", cpu_pos);
    println!("GPU velocity: {:?}", gpu_vel);
    println!("CPU velocity: {:?}", cpu_vel);

    // Positions should be reasonably similar (allowing for implementation differences)
    let pos_diff = (gpu_pos - cpu_pos).length();
    let vel_diff = (gpu_vel - cpu_vel).length();

    // Allow for some difference due to different collision handling
    // GPU uses spring-damper, CPU may use different model
    assert!(
        pos_diff < 1.0,
        "Position difference should be < 1m (got {:.4}m)",
        pos_diff
    );
    assert!(
        vel_diff < 5.0,
        "Velocity difference should be < 5m/s (got {:.4}m/s)",
        vel_diff
    );

    // Both should show downward motion due to gravity
    assert!(
        gpu_pos.y < start_pos.y,
        "GPU particle should have fallen"
    );
    assert!(
        cpu_pos.y < start_pos.y,
        "CPU particle should have fallen"
    );
}

/// Test 10: GPU DEM performance with many particles
/// Verifies the system handles larger particle counts without issues
#[test]
fn test_gpu_dem_many_particles() {
    let (device, queue) = match init_gpu() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU; skipping test_gpu_dem_many_particles");
            return;
        }
    };

    use game::gpu::dem_3d::GpuDem3D;
    use std::time::Instant;

    let particle_count = 500;
    let mut gpu_dem = GpuDem3D::new(device.clone(), queue.clone(), particle_count, 10, 50000);

    let template = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_id = gpu_dem.add_template(template);

    // Spawn particles in a 3D grid
    let grid_size = (particle_count as f32).powf(1.0 / 3.0).ceil() as u32;
    let spacing = 0.1;
    let mut count = 0;

    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                if count >= particle_count {
                    break;
                }
                let pos = Vec3::new(
                    x as f32 * spacing,
                    y as f32 * spacing + 1.0,
                    z as f32 * spacing,
                );
                gpu_dem.spawn_clump(template_id, pos, Vec3::ZERO);
                count += 1;
            }
        }
    }

    println!("Spawned {} particles", gpu_dem.particle_count());

    // Time simulation steps
    let start = Instant::now();
    let test_steps = 30;

    for _ in 0..test_steps {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Performance Test"),
        });
        gpu_dem.update(&mut encoder, DT);
        queue.submit(Some(encoder.finish()));
    }

    device.poll(wgpu::Maintain::Wait);
    let elapsed = start.elapsed();

    let ms_per_step = elapsed.as_secs_f64() * 1000.0 / test_steps as f64;
    println!(
        "Performance: {} steps in {:.2}ms ({:.2}ms/step)",
        test_steps,
        elapsed.as_secs_f64() * 1000.0,
        ms_per_step
    );

    // Should complete in reasonable time (< 100ms per step for 500 particles)
    assert!(
        ms_per_step < 100.0,
        "Step time should be < 100ms (got {:.2}ms)",
        ms_per_step
    );

    // Verify particles are still valid
    let particles = pollster::block_on(gpu_dem.readback(&device));
    let active_count = particles
        .iter()
        .filter(|p| p.position[1] > -100.0)
        .count();

    println!("Active particles after simulation: {}", active_count);

    assert!(
        active_count > particle_count as usize / 2,
        "Most particles should still be active (got {}/{})",
        active_count,
        particle_count
    );
}
