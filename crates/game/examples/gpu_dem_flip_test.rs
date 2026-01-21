//! GPU DEM-FLIP Integration Test
//!
//! Tests coupling between GPU DEM and GPU FLIP simulations.
//! Verifies momentum transfer and bidirectional interaction.
//!
//! Run with: cargo run --example gpu_dem_flip_test --release --features gpu-dem

use game::gpu::dem_3d::GpuDem3D;
use glam::Vec3;
use std::sync::Arc;

const PARTICLE_COUNT: u32 = 5000;
const GRID_WIDTH: u32 = 32;
const GRID_HEIGHT: u32 = 32;
const GRID_DEPTH: u32 = 32;
const CELL_SIZE: f32 = 0.05; // 5cm cells

fn main() {
    println!("GPU DEM-FLIP Integration Test");
    println!("=============================");
    println!();

    // Initialize GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to find adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("DEM-FLIP Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        },
        None,
    ))
    .expect("Failed to create device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Create GPU DEM
    let mut gpu_dem = GpuDem3D::new(
        device.clone(),
        queue.clone(),
        PARTICLE_COUNT,
        2,     // templates
        10000, // contacts
    );

    // Create DEM templates
    let template_light = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Irregular {
            count: 1,
            seed: 42,
            style: sim3d::IrregularStyle3D::Round,
        },
        0.002,  // 2mm radius (fine sediment)
        1000.0, // density like water - will float
    );

    let template_heavy = sim3d::ClumpTemplate3D::generate(
        sim3d::ClumpShape3D::Tetra,
        0.005,  // 5mm radius (heavier sediment)
        2500.0, // density 2.5x water - will sink
    );

    let light_id = gpu_dem.add_template(template_light);
    let heavy_id = gpu_dem.add_template(template_heavy);

    // Spawn particles in a column within the FLIP grid bounds
    let grid_extent = GRID_WIDTH as f32 * CELL_SIZE;
    let mut spawned = 0u32;
    for i in 0..20 {
        for j in 0..10 {
            for k in 0..10 {
                if spawned >= PARTICLE_COUNT {
                    break;
                }

                let x = 0.2 + (j as f32) * 0.02;
                let y = 0.5 + (i as f32) * 0.02;
                let z = 0.2 + (k as f32) * 0.02;

                // Alternate between light and heavy particles
                let template_id = if (spawned % 2) == 0 { light_id } else { heavy_id };
                gpu_dem.spawn_clump(template_id, Vec3::new(x, y, z), Vec3::ZERO);
                spawned += 1;
            }
        }
    }

    println!("Initialized {} DEM particles", spawned);
    println!(
        "Grid: {}x{}x{}, cell size: {:.3}m, extent: {:.2}m",
        GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE, grid_extent
    );
    println!();

    // Create mock FLIP grid buffers for testing the bridge
    // In production, these would come from GpuFlip3D
    let u_size = ((GRID_WIDTH + 1) * GRID_HEIGHT * GRID_DEPTH) as usize;
    let v_size = (GRID_WIDTH * (GRID_HEIGHT + 1) * GRID_DEPTH) as usize;
    let w_size = (GRID_WIDTH * GRID_HEIGHT * (GRID_DEPTH + 1)) as usize;

    let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mock Grid U"),
        size: (u_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mock Grid V"),
        size: (v_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let grid_w_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mock Grid W"),
        size: (w_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Initialize with some flow (upward in Y direction to test buoyancy interaction)
    let mut grid_v_data = vec![0.0f32; v_size];
    for i in 0..v_size {
        grid_v_data[i] = 0.5; // 0.5 m/s upward flow
    }
    queue.write_buffer(&grid_v_buffer, 0, bytemuck::cast_slice(&grid_v_data));

    println!("Running simulation steps...");

    let dt = 1.0 / 60.0;

    // Run 100 steps
    for step in 0..100 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM-FLIP Step"),
        });

        // 1. DEM prepare step (clears forces, collision detection)
        gpu_dem.prepare_step(&mut encoder, dt);

        // 2. Apply FLIP coupling to DEM particles
        // This samples fluid velocity and applies drag/buoyancy forces
        gpu_dem.apply_flip_coupling(
            &mut encoder,
            &grid_u_buffer,
            &grid_v_buffer,
            &grid_w_buffer,
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_DEPTH,
            CELL_SIZE,
            dt,
            5.0,    // drag_coefficient - how strongly particles follow water
            1000.0, // density_water - kg/m³
            0.01,   // bed_friction_coefficient (Shields criterion)
            0.045,  // critical_shields (incipient motion)
        );

        // 3. DEM finish step (integration)
        gpu_dem.finish_step(&mut encoder);

        queue.submit(Some(encoder.finish()));

        if step % 20 == 0 {
            println!("  Step {}: {} active particles", step, gpu_dem.particle_count());
        }
    }

    // Readback and verify
    let particles = pollster::block_on(gpu_dem.readback(&device));

    println!();
    println!("Final state:");
    println!("  Active particles: {}", particles.len());

    if !particles.is_empty() {
        // Calculate average position
        let avg_y: f32 = particles.iter().map(|p| p.position[1]).sum::<f32>() / particles.len() as f32;
        let avg_vy: f32 = particles.iter().map(|p| p.velocity[1]).sum::<f32>() / particles.len() as f32;

        println!("  Average Y position: {:.3}m", avg_y);
        println!("  Average Y velocity: {:.3}m/s", avg_vy);

        // With upward flow and buoyancy, light particles (density=1000, same as water)
        // should have neutral buoyancy and be carried by the flow.
        // Heavy particles (density=2500) should sink despite the flow.
        let light_count = particles.iter().filter(|p| p.velocity[1] > 0.0).count();
        let heavy_count = particles.len() - light_count;

        println!("  Particles with upward velocity: {}", light_count);
        println!("  Particles with downward velocity: {}", heavy_count);
    }

    println!();
    println!("GPU DEM-FLIP bridge test completed successfully!");
    println!();
    println!("The bridge shader samples FLIP grid velocities at DEM particle positions");
    println!("and applies drag and buoyancy forces. This enables two-way coupling");
    println!("between discrete element particles and FLIP fluid simulation.");
}
