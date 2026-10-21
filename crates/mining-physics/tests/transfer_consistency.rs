//! APIC must reproduce a constant field with zero affine velocity gradient,
//! including particles in the upper half of a grid interval.
use game::gpu::g2p_3d::{GpuG2p3D, SedimentParams3D};
use glam::{Mat3, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn constant_flow_does_not_generate_spurious_affine_motion() {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("real GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let buffer = |bytes: &[u8]| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("constant-flow transfer test"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        )
    };
    let positions: Vec<[f32; 4]> = [0.10, 0.25, 0.49, 0.51, 0.75, 0.90]
        .into_iter()
        .map(|f| [(6. + f) * 0.01, (7. + f) * 0.01, (8. + f) * 0.01, 1.])
        .collect();
    let n = positions.len();
    let p = buffer(bytemuck::cast_slice(&positions));
    let v = buffer(bytemuck::cast_slice(&vec![[0.2_f32; 4]; n]));
    let c0 = buffer(bytemuck::cast_slice(&vec![[0_f32; 4]; n]));
    let c1 = buffer(bytemuck::cast_slice(&vec![[0_f32; 4]; n]));
    let c2 = buffer(bytemuck::cast_slice(&vec![[0_f32; 4]; n]));
    let density = buffer(bytemuck::cast_slice(&vec![1_f32; n]));
    let grid = buffer(bytemuck::cast_slice(&vec![0.2_f32; 17 * 16 * 16]));
    let solver = GpuG2p3D::new(
        &device, 16, 16, 16, n, p, v, c0, c1, c2, density, &grid, &grid, &grid, &grid, &grid,
        &grid, &grid, &grid, &grid, &grid, [0; 3],
    );
    solver.upload_params(
        &queue,
        n as u32,
        0.01,
        1. / 120.,
        0.,
        SedimentParams3D::default(),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode(&mut encoder, n as u32);
    queue.submit(Some(encoder.finish()));
    let mut velocity = vec![Vec3::ZERO; n];
    let mut affine = vec![Mat3::ZERO; n];
    solver.download(&device, &queue, n as u32, &mut velocity, &mut affine);
    for (i, (v, c)) in velocity.iter().zip(affine).enumerate() {
        assert!(
            (*v - Vec3::splat(0.2)).length() < 1e-6,
            "constant velocity at particle {i}: {v:?}"
        );
        assert!(
            // f32 summation is amplified by D^-1=4/dx². This allows only
            // roundoff-scale gradients (5 micrometres/s across a 1 cm cell).
            c.x_axis.length() + c.y_axis.length() + c.z_axis.length() < 5e-4,
            "uniform flow gained a false velocity gradient at particle {i}: {c:?}"
        );
    }
}
