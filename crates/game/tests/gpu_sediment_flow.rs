use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

const CELL_FLUID: u32 = 1;
const CELL_SOLID: u32 = 2;

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
            label: Some("Sediment Flow Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

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

#[test]
fn sediment_moves_downstream_with_water() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping sediment flow test.");
            return;
        }
    };

    let width = 20u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.05f32;
    let max_particles = 1024usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.sediment_vorticity_lift = 0.0;
    sim.sediment_settling_velocity = 0.0;
    sim.sediment_porosity_drag = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut sediment_indices = Vec::new();

    for i in 0..4 {
        for j in 0..3 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.25 * cell_size,
                    (3 + j) as f32 * cell_size + 0.25 * cell_size,
                    (3 + k) as f32 * cell_size + 0.25 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::new(1.0, 0.0, 0.0));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    for i in 0..4 {
        for j in 0..3 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.6 * cell_size,
                    (3 + j) as f32 * cell_size + 0.6 * cell_size,
                    (3 + k) as f32 * cell_size + 0.6 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                densities.push(2.65);
                c_matrices.push(Mat3::ZERO);
                sediment_indices.push(positions.len() - 1);
            }
        }
    }

    assert!(positions.len() <= max_particles);

    let avg_x_before = sediment_indices
        .iter()
        .map(|&idx| positions[idx].x)
        .sum::<f32>()
        / sediment_indices.len() as f32;
    let avg_y_before = sediment_indices
        .iter()
        .map(|&idx| positions[idx].y)
        .sum::<f32>()
        / sediment_indices.len() as f32;

    let dt = 1.0 / 60.0;
    let pressure_iterations = 10;
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
            pressure_iterations,
        );
    }

    let avg_x_after = sediment_indices
        .iter()
        .map(|&idx| positions[idx].x)
        .sum::<f32>()
        / sediment_indices.len() as f32;
    let avg_y_after = sediment_indices
        .iter()
        .map(|&idx| positions[idx].y)
        .sum::<f32>()
        / sediment_indices.len() as f32;

    let dx = avg_x_after - avg_x_before;
    let dy = avg_y_after - avg_y_before;

    assert!(
        dx > cell_size * 0.5,
        "expected sediment to move downstream, dx={dx:.4}"
    );
    assert!(
        dy.abs() < cell_size * 0.5,
        "expected sediment to stay near its initial height, dy={dy:.4}"
    );
}
