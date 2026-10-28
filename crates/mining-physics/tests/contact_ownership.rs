//! The GPU must not consume the impact before external DEM applies friction.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

#[test]
#[ignore = "requires real GPU; explicitly run with --ignored"]
fn external_dem_receives_the_uncancelled_grain_impact() {
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
    let n = 12;
    let dx = 0.01;
    let sdf: Vec<f32> = (0..n * n * n)
        .map(|i| ((i / n) % n) as f32 * dx + dx * 0.5 - 0.04)
        .collect();
    let run = |external| {
        let mut sim = GpuFlip3D::new(&device, n as u32, n as u32, n as u32, dx, 16);
        sim.cpu_sediment_contacts = external;
        sim.open_boundaries = 63;
        sim.density_projection_enabled = false;
        sim.vorticity_epsilon = 0.;
        sim.sediment_rest_particles = 0.;
        sim.sediment_porosity_drag = 0.;
        sim.sediment_friction_strength = 0.;
        sim.sediment_vorticity_lift = 0.;
        sim.upload_grain_diameters(&queue, 0, &[-0.001]);
        let mut p = [Vec3::new(0.06, 0.045, 0.06)];
        let mut v = [Vec3::new(0.1, -0.5, 0.)];
        let mut c = [Mat3::ZERO];
        sim.step(
            &device,
            &queue,
            &mut p,
            &mut v,
            &mut c,
            &[2.65],
            &vec![0; n * n * n],
            Some(&sdf),
            None,
            0.02,
            0.,
            0.,
            80,
        );
        (p[0], v[0])
    };
    let dem = run(true);
    let gpu = run(false);
    assert!(
        dem.0.y < 0.04 && dem.1.y < -0.4,
        "DEM needs the actual incoming impact: {dem:?}"
    );
    // G2P may limit speed before this stage; SDF ownership must preserve the
    // resulting incoming velocity and its uncorrected Euler displacement.
    assert!((dem.0.y - (0.045 + dem.1.y * 0.02)).abs() < 1e-7);
    assert!(
        gpu.0.y >= 0.04 && gpu.1.y >= -1e-6,
        "legacy GPU contacts remain available: {gpu:?}"
    );
}
