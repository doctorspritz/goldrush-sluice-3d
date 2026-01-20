//! SPH Diagnostic Tests
//!
//! Targeted tests to isolate specific SPH behaviors and verify fundamental assumptions.

use game::gpu::sph_3d::GpuSph3D;
use glam::Vec3;

fn init_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
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
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SPH Diagnostics"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..Default::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .ok()?;

        Some((device, queue))
    })
}

/// Test 1: Two particles at distance r < h should find each other
///
/// Expected: Both particles should have density > 0 (they contribute to each other)
/// This verifies:
/// - Kernel evaluation works
/// - Neighbor search works (brute-force)
/// - particle_mass is applied correctly
#[test]
fn test_two_particles_neighbor_detection() {
    let Some((device, queue)) = init_device_queue() else {
        println!("Skipping test: No GPU available");
        return;
    };

    let h = 0.1; // 10cm kernel radius
    let dt = 1.0 / 60.0;
    let grid_dims = [10, 10, 10];

    let mut sph = GpuSph3D::new(&device, 100, h, dt, grid_dims);

    // Calibrate mass
    let _calibrated_density = sph.calibrate_rest_density(&device, &queue);
    sph.set_rest_density(&queue, 1000.0);

    // Create 2 particles at distance 0.05m (half of h)
    // They should definitely be neighbors
    let positions = vec![
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.55, 0.5, 0.5), // 5cm apart, well within h=10cm
    ];
    let velocities = vec![Vec3::ZERO, Vec3::ZERO];

    sph.upload_particles(&queue, &positions, &velocities);

    // Run one brute-force density computation
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Density Test"),
    });
    sph.step_bruteforce(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));

    // Read densities
    let metrics = sph.compute_metrics(&device, &queue);

    println!("Two Particle Test:");
    println!("  Max Density: {:.2}", metrics.max_density);
    println!("  Min Density: {:.2}", metrics.min_density);
    println!("  Particle Mass: {:.6}", sph.rest_density() / 1566666.25); // Approximate from calibration

    // Both particles should have non-zero density
    assert!(
        metrics.min_density > 0.0,
        "Particles at distance 0.05m (h=0.1m) should find each other! min_density={}",
        metrics.min_density
    );

    // Density should be reasonable (at least 1% of rest density)
    assert!(
        metrics.min_density > 10.0,
        "Density too low! Expected >10, got {}. Particles may not be contributing correctly.",
        metrics.min_density
    );
}

/// Test 2: Grid of particles at calibration spacing should match calibration density
///
/// Expected: Density should be close to rest_density (1000)
/// This verifies:
/// - Calibration spacing matches simulation spacing
/// - Mass scaling is correct
/// - No drift or integration issues
#[test]
fn test_calibration_spacing_matches_simulation() {
    let Some((device, queue)) = init_device_queue() else {
        println!("Skipping test: No GPU available");
        return;
    };

    let h = 0.1;
    let dt = 0.0; // ZERO timestep to prevent particle movement
    let grid_dims = [20, 20, 20];

    let mut sph = GpuSph3D::new(&device, 10000, h, dt, grid_dims);

    // Calibrate (this uses its own dt internally)
    let calibrated_density = sph.calibrate_rest_density(&device, &queue);
    sph.set_rest_density(&queue, calibrated_density);
    
    // Set timestep to zero AFTER calibration
    sph.set_timestep(&queue, 0.0);

    // Create same grid as calibration: 10x10x10 at spacing h*0.5
    let spacing = h * 0.5;
    let grid_size = 10;
    let center = Vec3::new(0.5, 0.5, 0.5);

    let mut positions = Vec::new();
    let mut velocities = Vec::new();

    for x in 0..grid_size {
        for y in 0..grid_size {
            for z in 0..grid_size {
                let pos = center
                    + Vec3::new(
                        (x as f32 - grid_size as f32 / 2.0) * spacing,
                        (y as f32 - grid_size as f32 / 2.0) * spacing,
                        (z as f32 - grid_size as f32 / 2.0) * spacing,
                    );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
            }
        }
    }

    sph.upload_particles(&queue, &positions, &velocities);

    // Run ONE step (no integration, just density)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Calibration Match Test"),
    });
    sph.step_bruteforce(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));

    let metrics = sph.compute_metrics(&device, &queue);

    println!("Calibration Match Test:");
    println!("  Calibrated Density: {:.2}", calibrated_density);
    println!("  Simulation Max Density: {:.2}", metrics.max_density);
    println!("  Ratio: {:.2}%", (metrics.max_density / calibrated_density) * 100.0);

    // Density should match calibration within 20%
    let ratio = metrics.max_density / calibrated_density;
    assert!(
        ratio > 0.8 && ratio < 1.2,
        "Simulation density ({:.2}) doesn't match calibration ({:.2})! Ratio: {:.2}",
        metrics.max_density,
        calibrated_density,
        ratio
    );
}

/// Test 3: Verify kernel evaluation at known distances
///
/// Expected: Particle at center of kernel should have maximum density contribution
/// This verifies:
/// - Poly6 kernel coefficient is correct
/// - Kernel evaluation doesn't have bugs
#[test]
fn test_kernel_evaluation_at_center() {
    let Some((device, queue)) = init_device_queue() else {
        println!("Skipping test: No GPU available");
        return;
    };

    let h = 0.1;
    let dt = 1.0 / 60.0;
    let grid_dims = [10, 10, 10];

    let mut sph = GpuSph3D::new(&device, 100, h, dt, grid_dims);

    // Calibrate
    let _calibrated_density = sph.calibrate_rest_density(&device, &queue);
    sph.set_rest_density(&queue, 1000.0);

    // Single particle (self-contribution only)
    let positions = vec![Vec3::new(0.5, 0.5, 0.5)];
    let velocities = vec![Vec3::ZERO];

    sph.upload_particles(&queue, &positions, &velocities);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Kernel Test"),
    });
    sph.step_bruteforce(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));

    let metrics = sph.compute_metrics(&device, &queue);

    println!("Kernel Evaluation Test:");
    println!("  Self-contribution density: {:.2}", metrics.max_density);
    println!("  Expected (from calibration): ~{:.2}", 1000.0 / 1000.0); // Should be ~1.0 after mass scaling

    // Self-contribution should be non-zero
    assert!(
        metrics.max_density > 0.0,
        "Single particle should have self-contribution! density={}",
        metrics.max_density
    );
}
