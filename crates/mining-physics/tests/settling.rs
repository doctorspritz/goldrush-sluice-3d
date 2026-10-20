//! Regression for the actual G2P shader, isolated from moving water and walls.
//! A prescribed terminal slip must be a velocity, not an impulse per substep.
use game::gpu::g2p_3d::{GpuG2p3D, SedimentParams3D};
use glam::{Mat3, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a real Metal/Vulkan compute adapter"]
fn terminal_settling_is_independent_of_substep_count() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let run = |dt: f32, steps: usize, diameters: [f32; 3]| {
        let buffer = |label, bytes: &[u8]| {
            Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
            )
        };
        let position = buffer(
            "fixed sample positions",
            bytemuck::cast_slice(&[[0.08f32, 0.08, 0.08, 0.0]; 3]),
        );
        let velocity = buffer(
            "sediment velocities",
            bytemuck::cast_slice(&[[0.0f32; 4]; 3]),
        );
        queue.write_buffer(
            &velocity,
            32,
            bytemuck::cast_slice(&[[0.25f32, 0.0, 0.0, 0.0]]),
        );
        let c0 = buffer("C0", bytemuck::cast_slice(&[[0.0f32; 4]; 3]));
        let c1 = buffer("C1", bytemuck::cast_slice(&[[0.0f32; 4]; 3]));
        let c2 = buffer("C2", bytemuck::cast_slice(&[[0.0f32; 4]; 3]));
        let density = buffer(
            "quartz and gold densities",
            bytemuck::cast_slice(&[2.65f32, 19.3, 1.0]),
        );
        let grid = buffer(
            "still water and unchanged grids",
            bytemuck::cast_slice(&[0.0f32; 576]),
        );
        let solver = GpuG2p3D::new(
            &device, 8, 8, 8, 3, position, velocity, c0, c1, c2, density, &grid, &grid, &grid,
            &grid, &grid, &grid, &grid, &grid, &grid, &grid,
        );
        solver.upload_grain_diameters(&queue, 0, &diameters);
        let params = SedimentParams3D {
            settling_velocity: 0.05,
            gold_settling_velocity: 0.15,
            ..Default::default()
        };
        for _ in 0..steps {
            solver.upload_params(&queue, 3, 0.02, dt, 0.95, params);
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode(&mut encoder, 3);
            queue.submit(Some(encoder.finish()));
        }
        let mut velocities = [Vec3::ZERO; 3];
        let mut affine = [Mat3::ZERO; 3];
        solver.download(&device, &queue, 3, &mut velocities, &mut affine);
        velocities
    };
    let at_120 = run(1.0 / 120.0, 30, [0.0; 3]);
    let at_240 = run(1.0 / 240.0, 60, [0.0; 3]);
    for (a, b) in at_120.iter().take(2).zip(at_240) {
        assert!(
            (*a - b).length() < 2e-6,
            "substep-dependent sediment speed: {a:?} / {b:?}"
        );
    }
    assert!(
        (at_120[0].y + 0.05).abs() < 1e-5,
        "sand must approach its prescribed terminal slip"
    );
    assert!(
        (at_120[1].y + 0.15).abs() < 1e-5,
        "gold must approach its prescribed terminal slip"
    );
    assert!(
        (at_120[2].x - 0.25 * 0.95f32.powi(30)).abs() < 1e-5,
        "water must retain the original FLIP/PIC update"
    );
    assert!((at_240[2].x - 0.25 * 0.95f32.powi(60)).abs() < 1e-5);
    let small = run(1.0 / 120.0, 120, [0.0005, 0.0005, 0.0]);
    let small_fine = run(1.0 / 240.0, 240, [0.0005, 0.0005, 0.0]);
    let coarse = run(1.0 / 120.0, 120, [0.001, 0.001, 0.0]);
    for i in 0..2 {
        assert!(
            (small[i] - small_fine[i]).length() < 1e-5,
            "physical grain slip depends on dt"
        );
        assert!(coarse[i].y < small[i].y, "larger grain must settle faster");
        let relative = if i == 0 { 1.65_f32 } else { 18.3 };
        let d = 0.0005_f32;
        let expected =
            relative * 9.81 * d * d / (18e-6 + (0.75 * relative * 9.81 * d * d * d).sqrt());
        assert!(
            (small[i].y + expected).abs() < 1e-4,
            "size/density settling mismatch"
        );
    }
    assert!(
        small[1].y < small[0].y,
        "same-sized gold must settle faster than quartz"
    );
    let dry = run(1.0 / 120.0, 12, [-0.001, -0.001, 0.0]);
    assert!(
        (dry[0].y + 0.981).abs() < 1e-5 && (dry[1].y + 0.981).abs() < 1e-5,
        "dry grains must free-fall without buoyancy"
    );
    println!("Physical G2P: 0.5mm={small:?}; 1mm={coarse:?}; dry={dry:?}");
    println!("Actual G2P terminal speeds: 120Hz={at_120:?}; 240Hz={at_240:?}");
}
