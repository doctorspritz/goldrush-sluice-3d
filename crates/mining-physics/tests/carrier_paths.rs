//! Exercise carrier-support bindings through every ordinary FLIP transfer path.
//! The raw water P2G field is exactly unit X on its supported faces, so a
//! co-moving grain has an independent constant-translation oracle even at a
//! sparse sheet edge. This checks transport wiring, not riffle qualification.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

#[test]
#[ignore = "requires a real Metal/Vulkan compute adapter; run explicitly with --ignored"]
fn sparse_carrier_survives_all_ordinary_transfer_paths() {
    const N: u32 = 12;
    const H: f32 = 0.01;
    const DT: f32 = 1. / 120.;
    const DIAMETER: f32 = 47e-6;
    const VELOCITY_TOLERANCE: f32 = 2e-5;
    const POSITION_TOLERANCE: f32 = 2e-7;
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("real compute adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();

    let mut source = Vec::new();
    for y in 3..=9 {
        for z in 3..=9 {
            source.push(Vec3::new(4.25, y as f32 + 0.25, z as f32 + 0.25) * H);
        }
    }
    let water_count = source.len();
    for offset in [0., 0.5, 1., 1.25, 1.5] {
        source.push(Vec3::new(4.25 + offset, 6.25, 6.25) * H);
    }
    assert_eq!(source.len(), 54);
    for grain in &source[water_count..] {
        let home = (*grain / H).floor();
        assert!(source[..water_count]
            .iter()
            .any(|water| (home - (*water / H).floor()).abs().max_element() <= 1.));
    }
    let mut densities = vec![1.; source.len()];
    densities[water_count..].fill(2.65);
    let cell_types = vec![0; (N * N * N) as usize];
    let modes = [
        ("paired_unsorted", true, false, false),
        ("separate_unsorted", false, false, false),
        ("paired_sorted", true, true, false),
        ("separate_sorted", false, true, false),
        ("paired_cell_centric_sorted", true, true, true),
        ("separate_cell_centric_sorted", false, true, true),
    ];

    // One device, six sequential short-lived solvers, one physical step each.
    // GpuFlip3D has no simulation clock; the independently predicted Euler
    // displacement proves that the requested DT actually advances the grains.
    for (name, paired, sorted, cell_centric) in modes {
        let mut sim = GpuFlip3D::new_with_transfer(&device, N, N, N, H, 64, paired);
        sim.use_sorted_p2g = sorted;
        sim.use_cell_centric_p2g = cell_centric;
        sim.open_boundaries = 63;
        sim.density_projection_enabled = false;
        sim.vorticity_epsilon = 0.;
        sim.sediment_friction_strength = 0.;
        sim.sediment_vorticity_lift = 0.;
        sim.gold_flake_lift = 0.;
        sim.sediment_rest_particles = 0.;
        sim.sediment_porosity_drag = 0.;
        sim.upload_grain_diameters(&queue, water_count, &[DIAMETER; 5]);
        let mut positions = source.clone();
        let mut velocities = vec![Vec3::X; source.len()];
        let mut affine = vec![Mat3::ZERO; source.len()];
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut affine,
            &densities,
            &cell_types,
            None,
            None,
            DT,
            0.,
            0.,
            0,
        );
        assert!(sim.take_phase_error().is_none(), "{name}: solver refusal");
        assert!(
            positions.iter().chain(&velocities).all(|v| v.is_finite())
                && affine.iter().all(|c| c.is_finite()),
            "{name}: nonfinite full-step state"
        );
        let mut max_velocity_error = 0_f32;
        let mut max_position_error = 0_f32;
        for i in water_count..source.len() {
            let velocity_error = velocities[i].distance(Vec3::X);
            let expected_position = source[i] + Vec3::X * DT;
            let position_error = positions[i].distance(expected_position);
            max_velocity_error = max_velocity_error.max(velocity_error);
            max_position_error = max_position_error.max(position_error);
            assert!(
                velocity_error <= VELOCITY_TOLERANCE,
                "{name}: grain {} lost co-motion: velocity={:?}, error={velocity_error}",
                i - water_count,
                velocities[i]
            );
            assert!(
                positions[i].x > source[i].x && position_error <= POSITION_TOLERANCE,
                "{name}: grain {} did not advance by the requested DT: before={:?}, after={:?}, expected={expected_position:?}, error={position_error}",
                i - water_count,
                source[i],
                positions[i]
            );
        }
        println!(
            "{name}: grains=5, dt={DT}, max_velocity_error={max_velocity_error}, max_position_error={max_position_error}"
        );
    }
}
