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
    let mut sim = GpuFlip3D::new_with_transfer(device, 16, 12, 8, dx, 4096, batch);
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
    let mut densities = vec![1.0; positions.len()];
    let n = densities.len();
    densities[n - 2] = 2.65;
    densities[n - 1] = 19.3;
    sim.upload_grain_diameters(queue, n - 2, &[0.0003, 0.0003]);
    let cells = vec![0; 16 * 12 * 8];
    for _ in 0..4 {
        if batch {
            assert!(sim.prepare_particle_readback(0, positions.len()));
            assert!(!sim.prepare_particle_readback(0, positions.len()));
            sim.step_no_readback(
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
            device.poll(wgpu::Maintain::Wait);
            assert_eq!(
                sim.try_readback(device, &mut positions, &mut velocities, &mut affine),
                Some(n)
            );
            continue;
        }
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

#[test]
#[ignore = "requires real Metal/Vulkan compute adapter"]
fn paired_transfer_and_joined_readback_preserve_nonuniform_mixed_parcels() {
    let (device, queue) = device();
    let run = |optimized: bool| {
        let mut sim = GpuFlip3D::new_with_transfer(&device, 16, 12, 8, 0.02, 4096, optimized);
        sim.batch_submissions = optimized;
        sim.open_boundaries = 63;
        let n = 2048;
        let mut p: Vec<_> = (0..n)
            .map(|i| {
                Vec3::new(
                    0.04 + (i * 17 % 239) as f32 * 0.001,
                    0.04 + (i * 37 % 159) as f32 * 0.001,
                    0.04 + (i * 59 % 79) as f32 * 0.001,
                )
            })
            .collect();
        let mut v: Vec<_> = (0..n)
            .map(|i| {
                Vec3::new(
                    (i as f32 * 0.17).sin() * 0.3,
                    (i as f32 * 0.11).cos() * 0.1,
                    (i as f32 * 0.07).sin() * 0.2,
                )
            })
            .collect();
        let mut c: Vec<_> = (0..n)
            .map(|i| {
                Mat3::from_diagonal(Vec3::new(
                    (i as f32 * 0.31).sin(),
                    (i as f32 * 0.23).cos(),
                    0.2,
                ))
            })
            .collect();
        let densities: Vec<_> = (0..n)
            .map(|i| match i % 5 {
                0 => 19.3,
                1 => 2.65,
                _ => 1.,
            })
            .collect();
        sim.set_particle_volumes((0..n).map(|i| 0.25 + (i % 19) as f32 * 0.2).collect());
        sim.upload_grain_diameters(&queue, 0, &vec![0.0003; n]);
        let cells = vec![0; 16 * 12 * 8];
        for step in 0..3 {
            if optimized {
                // Exercise both a suffix range and the full five-plane mapping.
                let first = if step == 0 { 11 } else { 0 };
                assert!(sim.prepare_particle_readback(first, n - first));
                sim.step_no_readback(
                    &device,
                    &queue,
                    &mut p,
                    &mut v,
                    &mut c,
                    &densities,
                    &cells,
                    None,
                    None,
                    1. / 240.,
                    -9.81,
                    0.,
                    80,
                );
                device.poll(wgpu::Maintain::Wait);
                assert_eq!(
                    sim.try_readback(&device, &mut p[first..], &mut v[first..], &mut c[first..]),
                    Some(n - first)
                );
                if first != 0 {
                    assert!(sim.request_particle_range(&device, &queue, 0, first));
                    device.poll(wgpu::Maintain::Wait);
                    assert_eq!(
                        sim.try_readback(
                            &device,
                            &mut p[..first],
                            &mut v[..first],
                            &mut c[..first]
                        ),
                        Some(first)
                    );
                }
            } else {
                sim.step(
                    &device,
                    &queue,
                    &mut p,
                    &mut v,
                    &mut c,
                    &densities,
                    &cells,
                    None,
                    None,
                    1. / 240.,
                    -9.81,
                    0.,
                    80,
                );
            }
        }
        (p, v, c)
    };
    let reference = run(false);
    let optimized = run(true);
    assert_eq!(reference.0, optimized.0, "mixed parcel positions changed");
    assert_eq!(reference.1, optimized.1, "mixed parcel velocities changed");
    assert_eq!(
        reference.2, optimized.2,
        "mixed parcel APIC matrices changed"
    );
}
