//! Real GPU regression: batching is an ordering change; density repair must
//! not behave as an implicit friction term on a translating fluid parcel.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("real GPU adapter");
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    batch: bool,
    projection: bool,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<Mat3>) {
    let dx = 0.02;
    let mut sim = GpuFlip3D::new(device, 16, 12, 8, dx, 4096);
    sim.batch_submissions = batch;
    sim.open_boundaries = 63;
    sim.density_projection_enabled = projection;
    sim.sediment_rest_particles = 0.0;
    sim.sediment_porosity_drag = 0.0;
    let mut positions = Vec::new();
    for k in 2..6 {
        for j in 2..10 {
            for i in 2..14 {
                for sk in 0..2 {
                    for sj in 0..2 {
                        for si in 0..2 {
                            positions.push(
                                Vec3::new(
                                    i as f32 + 0.25 + si as f32 * 0.5,
                                    j as f32 + 0.25 + sj as f32 * 0.5,
                                    k as f32 + 0.25 + sk as f32 * 0.5,
                                ) * dx,
                            );
                        }
                    }
                }
            }
        }
    }
    let mut weights = vec![1.0; positions.len()];
    weights[0] = 3.5;
    sim.set_particle_volumes(weights);
    let mut velocities = vec![Vec3::new(0.4, 0.0, 0.0); positions.len()];
    let mut affine = vec![Mat3::ZERO; positions.len()];
    let densities = vec![1.0; positions.len()];
    let cells = vec![0; 16 * 12 * 8];
    for _ in 0..4 {
        sim.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut affine,
            &densities,
            &cells,
            None,
            None,
            1.0 / 240.0,
            0.0,
            0.0,
            80,
        );
    }
    // Resident transfers must retain parcel volume in position.w, and contact
    // readback must return the requested suffix, not the start of the buffer.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("volume weight regression"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&sim.positions_buffer(), 0, &staging, 0, 16);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    {
        let bytes = staging.slice(..).get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(values[3], 3.5, "advection destroyed parcel volume weight");
    }
    staging.unmap();
    assert!(sim.request_particle_range(device, queue, 5, 3));
    device.poll(wgpu::Maintain::Wait);
    let mut p = [Vec3::ZERO; 3];
    let mut v = p;
    let mut c = [Mat3::ZERO; 3];
    assert_eq!(sim.try_readback(device, &mut p, &mut v, &mut c), Some(3));
    assert_eq!(p.as_slice(), &positions[5..8]);
    assert_eq!(v.as_slice(), &velocities[5..8]);
    assert_eq!(c.as_slice(), &affine[5..8]);
    (positions, velocities, affine)
}
#[test]
#[ignore = "requires real Metal/Vulkan compute adapter"]
fn batching_preserves_results_and_density_projection_preserves_momentum() {
    let (device, queue) = device();
    let reference = run(&device, &queue, false, true);
    let batched = run(&device, &queue, true, true);
    for (a, b) in reference
        .0
        .iter()
        .chain(&reference.1)
        .zip(batched.0.iter().chain(&batched.1))
    {
        assert!(
            (*a - *b).length() < 1e-5,
            "batched GPU order changed result: {a:?} {b:?}"
        );
    }
    for (a, b) in reference.2.iter().zip(&batched.2) {
        assert!(
            (a.x_axis - b.x_axis).length()
                + (a.y_axis - b.y_axis).length()
                + (a.z_axis - b.z_axis).length()
                < 1e-4
        );
    }
    let no_projection = run(&device, &queue, true, false);
    let mean = |v: &[Vec3]| v.iter().map(|v| v.x).sum::<f32>() / v.len() as f32;
    let projected = mean(&batched.1);
    let unprojected = mean(&no_projection.1);
    println!(
        "Uniform translation momentum: density repair={projected:.6} no repair={unprojected:.6}"
    );
    assert!(
        (projected - unprojected).abs() < 0.005,
        "density repair must not add per-substep drag"
    );
    assert!(projected > 0.35);
}
