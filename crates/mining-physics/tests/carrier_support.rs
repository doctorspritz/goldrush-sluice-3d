//! The water carrier must preserve a constant translation at sparse support.
//! Runs the production water-only P2G and grain G2P; no pressure or wall forces.
use game::gpu::{
    g2p_3d::{GpuG2p3D, SedimentParams3D},
    p2g_3d::GpuP2g3D,
};
use glam::{Mat3, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a real Metal/Vulkan compute adapter; run explicitly with --ignored"]
fn sparse_carrier_preserves_constant_translation() {
    const N: u32 = 12;
    const H: f32 = 0.01;
    const CAPACITY: usize = 512;
    const DIAMETER: f32 = 47e-6;
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
    let positions = buffer("carrier positions", &vec![0; CAPACITY * 16]);
    let velocities = buffer("carrier velocities", &vec![0; CAPACITY * 16]);
    let c0 = buffer("carrier C0", &vec![0; CAPACITY * 16]);
    let c1 = buffer("carrier C1", &vec![0; CAPACITY * 16]);
    let c2 = buffer("carrier C2", &vec![0; CAPACITY * 16]);
    let densities = buffer("carrier densities", &vec![0; CAPACITY * 4]);
    let mut water = GpuP2g3D::new(
        &device,
        N,
        N,
        N,
        CAPACITY,
        false,
        false,
        positions.clone(),
        velocities.clone(),
        densities.clone(),
        c0.clone(),
        c1.clone(),
        c2.clone(),
    );
    let sizes = water.grid_sizes();
    let grid_u = buffer("prescribed mixture U", &vec![0; sizes.0 * 4]);
    let grid_v = buffer("prescribed mixture V", &vec![0; sizes.1 * 4]);
    let grid_w = buffer("prescribed mixture W", &vec![0; sizes.2 * 4]);
    let vorticity = buffer("no vorticity", &vec![0; (N * N * N) as usize * 4]);
    let make_gather = |carrier_support| {
        GpuG2p3D::new_with_metadata(
            &device,
            N,
            N,
            N,
            CAPACITY,
            positions.clone(),
            velocities.clone(),
            c0.clone(),
            c1.clone(),
            c2.clone(),
            densities.clone(),
            &grid_u,
            &grid_v,
            &grid_w,
            &grid_u,
            &grid_v,
            &grid_w,
            &vorticity,
            &water.grid_u_buffer,
            &water.grid_v_buffer,
            &water.grid_w_buffer,
            [0; 3],
            None,
            carrier_support,
        )
    };
    let gather = make_gather(true);
    let legacy = make_gather(false);
    let params = SedimentParams3D {
        gravity: 0.,
        friction_strength: 0.,
        vorticity_lift: 0.,
        gold_flake_lift: 0.,
        ..Default::default()
    };
    let mut maximum_sparse_error = 0_f32;
    let mut maximum_control_error = 0_f32;
    let mut failed_sparse = 0;
    let mut sparse_probes = 0;
    let mut control_probes = 0;
    let mut first_sparse_failure = None;
    let mut legacy_sparse_error = 0_f32;
    for normal_axis in 0..3 {
        let tangent_axes: Vec<_> = (0..3).filter(|a| *a != normal_axis).collect();
        for flow_axis in 0..3 {
            let mut translation = Vec3::ZERO;
            translation[flow_axis] = 1.;
            // Unit velocity makes each supported P2G integer momentum exactly
            // equal to its integer weight. This isolates missing support from
            // fixed-point velocity bias, even as represented water mass varies.
            for mass_scale in [0.125, 1., 8.] {
                for scenario in ["sparse", "dense", "stationary"] {
                    let dense = scenario == "dense";
                    let stationary = scenario == "stationary";
                    let carrier = if stationary { Vec3::ZERO } else { translation };
                    let mut points = Vec::new();
                    for layer in 2..=9 {
                        if !dense && layer != 4 {
                            continue;
                        }
                        for a in 3..=9 {
                            for b in 3..=9 {
                                let mut p = Vec3::ZERO;
                                p[normal_axis] = (layer as f32 + 0.25) * H;
                                p[tangent_axes[0]] = (a as f32 + 0.25) * H;
                                p[tangent_axes[1]] = (b as f32 + 0.25) * H;
                                points.push(p);
                            }
                        }
                    }
                    let nw = points.len();
                    let mut initial = vec![carrier; nw];
                    let mut rho = vec![1.; nw];
                    let mut diameters = vec![0.; nw];
                    for offset in [0., 0.5, 1., 1.25, 1.5] {
                        let mut p = Vec3::splat(6.25 * H);
                        p[normal_axis] = (4.25 + offset) * H;
                        // Every probe is wet under the actual adapter's 27-home-
                        // cell presence rule. Positions stay clear of grid bounds.
                        let home = (p / H).floor();
                        assert!(points[..nw]
                            .iter()
                            .any(|q| (home - (*q / H).floor()).abs().max_element() <= 1.));
                        for sign in [1., -1.] {
                            points.push(p);
                            initial.push(translation);
                            rho.push(2.65);
                            diameters.push(sign * DIAMETER);
                        }
                    }
                    assert!(points.len() <= CAPACITY);
                    water.volume_weights = vec![mass_scale; points.len()];
                    let count = water.upload_particles(
                        &queue,
                        &points,
                        &initial,
                        &rho,
                        &vec![Mat3::ZERO; points.len()],
                        H,
                    );
                    gather.upload_grain_diameters(&queue, 0, &diameters);
                    for (axis, grid, size) in [
                        (0, &grid_u, sizes.0),
                        (1, &grid_v, sizes.1),
                        (2, &grid_w, sizes.2),
                    ] {
                        queue.write_buffer(
                            grid,
                            0,
                            bytemuck::cast_slice(&vec![carrier[axis]; size]),
                        );
                    }
                    gather.upload_params(&queue, count, H, 1. / 120., 0., params);
                    let mut encoder = device.create_command_encoder(&Default::default());
                    water.encode(&mut encoder, count);
                    gather.encode(&mut encoder, count);
                    queue.submit(Some(encoder.finish()));
                    let mut result = vec![Vec3::ZERO; points.len()];
                    let mut affine = vec![Mat3::ZERO; points.len()];
                    gather.download(&device, &queue, count, &mut result, &mut affine);
                    if normal_axis == 0
                        && flow_axis == 0
                        && mass_scale == 1.
                        && scenario == "sparse"
                    {
                        water.upload_particles(
                            &queue,
                            &points,
                            &initial,
                            &rho,
                            &vec![Mat3::ZERO; points.len()],
                            H,
                        );
                        legacy.upload_grain_diameters(&queue, 0, &diameters);
                        legacy.upload_params(&queue, count, H, 1. / 120., 0., params);
                        let mut encoder = device.create_command_encoder(&Default::default());
                        water.encode(&mut encoder, count);
                        legacy.encode(&mut encoder, count);
                        queue.submit(Some(encoder.finish()));
                        let mut old = vec![Vec3::ZERO; points.len()];
                        legacy.download(&device, &queue, count, &mut old, &mut affine);
                        for i in (nw..points.len()).step_by(2) {
                            legacy_sparse_error =
                                legacy_sparse_error.max(old[i].distance(translation));
                        }
                    }
                    for i in nw..points.len() {
                        let dry = diameters[i] < 0.;
                        // In zero gravity, co-motion and dry ballistic motion
                        // remain unit translation. A supported still carrier
                        // damps 47um quartz almost completely within this step.
                        let expected = if stationary && !dry {
                            Vec3::ZERO
                        } else {
                            translation
                        };
                        let error = result[i].distance(expected);
                        assert!(error.is_finite(), "nonfinite carrier output");
                        if scenario == "sparse" && !dry {
                            sparse_probes += 1;
                            maximum_sparse_error = maximum_sparse_error.max(error);
                            if error > 2e-5 {
                                failed_sparse += 1;
                                first_sparse_failure.get_or_insert_with(|| format!(
                                    "normal={normal_axis}, flow={flow_axis}, mass={mass_scale}, position={:?}, got={:?}, expected={expected:?}",
                                    points[i], result[i]));
                            }
                        } else {
                            control_probes += 1;
                            maximum_control_error = maximum_control_error.max(error);
                        }
                    }
                }
            }
        }
    }
    // No water at all is distinct from supported water at rest. For partial
    // support, erase exactly one producer support plane after real P2G; its
    // velocity plane stays zero, as do the supported stationary components.
    for (missing, gravity) in [(7u8, 0.), (7, 9.81), (1, 0.), (2, 9.81), (4, 0.)] {
        let p = Vec3::splat(6.25 * H);
        let nw = usize::from(missing != 7);
        let points = vec![p; nw + 2];
        let initial: Vec<_> = (0..points.len())
            .map(|i| {
                if i < nw {
                    Vec3::ZERO
                } else {
                    Vec3::splat(0.25)
                }
            })
            .collect();
        let mut rho = vec![2.65; points.len()];
        if nw > 0 {
            rho[0] = 1.;
        }
        let mut diameters = vec![0.; nw];
        diameters.extend([DIAMETER, -DIAMETER]);
        water.volume_weights = vec![1.; points.len()];
        let count = water.upload_particles(
            &queue,
            &points,
            &initial,
            &rho,
            &vec![Mat3::ZERO; points.len()],
            H,
        );
        gather.upload_grain_diameters(&queue, 0, &diameters);
        for (grid, size) in [(&grid_u, sizes.0), (&grid_v, sizes.1), (&grid_w, sizes.2)] {
            queue.write_buffer(grid, 0, bytemuck::cast_slice(&vec![0_f32; size]));
        }
        gather.upload_params(
            &queue,
            count,
            H,
            1. / 120.,
            0.,
            SedimentParams3D { gravity, ..params },
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        water.encode(&mut encoder, count);
        if nw > 0 {
            for (axis, grid, size) in [
                (0, &water.grid_u_buffer, sizes.0),
                (1, &water.grid_v_buffer, sizes.1),
                (2, &water.grid_w_buffer, sizes.2),
            ] {
                if missing & (1 << axis) != 0 {
                    encoder.clear_buffer(grid, (size * 4) as u64, Some((size * 4) as u64));
                }
            }
        }
        gather.encode(&mut encoder, count);
        queue.submit(Some(encoder.finish()));
        let mut output = vec![Vec3::ZERO; points.len()];
        let mut affine = vec![Mat3::ZERO; points.len()];
        gather.download(&device, &queue, count, &mut output, &mut affine);
        let mut expected = Vec3::ZERO;
        for axis in 0..3 {
            if missing & (1 << axis) != 0 {
                expected[axis] = 0.25;
            }
        }
        if missing & 2 != 0 {
            expected.y -= gravity * (1. / 120.);
        }
        assert!(
            output[nw].distance(expected) < 2e-5,
            "missing support mask={missing}, gravity={gravity}: {:?} != {expected:?}",
            output[nw]
        );
        let dry_expected = Vec3::splat(0.25) - Vec3::Y * (gravity * (1. / 120.));
        assert_eq!(
            output[nw + 1],
            dry_expected,
            "negative diameter must remain exactly ballistic for mask={missing}"
        );
    }
    println!("carrier probes: sparse={sparse_probes}, controls={control_probes}, failed_sparse={failed_sparse}, max_sparse_error={maximum_sparse_error}, max_control_error={maximum_control_error}, legacy_sparse_error={legacy_sparse_error}");
    assert!(
        legacy_sparse_error > 0.7,
        "legacy negative witness no longer exposes support dilution: {legacy_sparse_error}"
    );
    assert!(
        maximum_control_error < 2e-5,
        "dense, stationary-water or dry control failed: {maximum_control_error}"
    );
    assert!(maximum_sparse_error < 2e-5,
        "unsupported water nodes diluted constant translation: {first_sparse_failure:?}; maximum error={maximum_sparse_error}");
}
