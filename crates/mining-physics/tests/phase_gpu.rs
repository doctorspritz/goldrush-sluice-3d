//! Actual phase transfer -> implicit response -> optional pressure -> G2P.
//! This harness is not the playable worker and does not claim packed-bed validation.
use game::gpu::{
    g2p_3d::{GpuG2p3D, SedimentParams3D},
    phase_transfer_3d::PhaseTransfer3D,
    pressure_3d::GpuPressure3D,
};
use glam::DVec3;
use mining_physics::phase_exchange::{FaceResponse, GrainSample};
use std::sync::Arc;
use wgpu::util::DeviceExt;
fn buffer(device: &wgpu::Device, bytes: &[u8]) -> Arc<wgpu::Buffer> {
    Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }),
    )
}
fn floats(device: &wgpu::Device, values: &[f32]) -> Arc<wgpu::Buffer> {
    buffer(device, bytemuck::cast_slice(values))
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = device.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let values = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    values
}
fn weight(x: f64) -> f64 {
    let a = x.abs();
    if a < 0.5 {
        0.75 - a * a
    } else if a < 1.5 {
        0.5 * (1.5 - a).powi(2)
    } else {
        0.
    }
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn phase_transfer_pressure_and_gather_match_oracle_with_momentum_and_freefall_controls() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let shape = [9u32, 8, 7];
    let [nx, ny, nz] = shape;
    let dx = 0.1f32;
    let dt = 1. / 120.;
    let mut exact_snapshot: Option<Vec<f32>> = None;
    for (label, gravity, rate, pressure, co_moving, capacity, reverse, affine) in [
        ("drag", 0.0_f32, 27., false, false, 128, false, false),
        ("exact-capacity", 0., 27., false, false, 96, false, false),
        ("reserved", 0., 27., false, false, 512, false, false),
        ("reversed", 0., 27., false, false, 128, true, false),
        ("stiff", 0., 1e12, false, false, 128, false, false),
        ("pressure", 0., 27., true, false, 128, false, false),
        ("freefall", -9.81, 27., false, true, 128, false, false),
        ("rotation", 0., 27., false, true, 128, false, true),
        ("overflow", 0., 27., false, false, 128, false, false),
        ("nonfinite", 0., 27., false, false, 128, false, false),
        ("bad-rate", 0., -1., false, false, 128, false, false),
    ] {
        eprintln!("phase control {label}");
        let count = 96usize;
        let mut positions: Vec<[f32; 4]> = (0..count)
            .map(|i| {
                [
                    0.31 + (i % 4) as f32 * 0.024,
                    0.29 + ((i / 4) % 4) as f32 * 0.022,
                    0.27 + ((i / 16) % 3) as f32 * 0.025,
                    0.25 + 0.125 * (i % 3) as f32,
                ]
            })
            .collect();
        let mut rho: Vec<f32> = (0..count)
            .map(|i| {
                if i % 3 != 0 {
                    1.
                } else if i % 2 == 0 {
                    19.3
                } else {
                    2.65
                }
            })
            .collect();
        let mut velocities: Vec<[f32; 4]> = (0..count)
            .map(|i| {
                if co_moving {
                    [0.08, -0.02, 0.03, 0.]
                } else if rho[i] > 1. {
                    [-0.03, 0.07, -0.01, 0.]
                } else {
                    [0.09, -0.01, 0.02, 0.]
                }
            })
            .collect();
        let affine_rows = if affine {
            [[0., -0.4, 0.2], [0.4, 0., -0.3], [-0.2, 0.3, 0.]]
        } else {
            [[0.; 3]; 3]
        };
        if affine {
            for (p, v) in positions.iter().zip(&mut velocities) {
                for a in 0..3 {
                    v[a] += (0..3)
                        .map(|b| affine_rows[a][b] * (p[b] - 0.3))
                        .sum::<f32>();
                }
            }
        }
        if reverse {
            positions.reverse();
            rho.reverse();
            velocities.reverse();
        }
        let mut p_storage = vec![[0.; 4]; capacity];
        p_storage[..count].copy_from_slice(&positions);
        if label == "overflow" {
            p_storage[3][3] = 1e12;
        }
        let mut v_storage = vec![[0.; 4]; capacity];
        v_storage[..count].copy_from_slice(&velocities);
        let p = buffer(&device, bytemuck::cast_slice(&p_storage));
        let v = buffer(&device, bytemuck::cast_slice(&v_storage));
        let rows = std::array::from_fn::<_, 3, _>(|a| {
            let data: Vec<[f32; 4]> = (0..capacity)
                .map(|i| {
                    if i < count {
                        [affine_rows[a][0], affine_rows[a][1], affine_rows[a][2], 0.]
                    } else {
                        [0.; 4]
                    }
                })
                .collect();
            buffer(&device, bytemuck::cast_slice(&data))
        });
        let mut meta = vec![0.; capacity * 4];
        meta[..count].copy_from_slice(&rho);
        if label == "nonfinite" {
            meta[capacity + 3] = f32::NAN;
        }
        let metadata = floats(&device, &meta);
        let densities = floats(&device, &{
            let mut x = vec![1.; capacity];
            x[..count].copy_from_slice(&rho);
            x
        });
        let sizes = [
            ((nx + 1) * ny * nz) as usize,
            (nx * (ny + 1) * nz) as usize,
            (nx * ny * (nz + 1)) as usize,
        ];
        let grids = sizes.map(|n| floats(&device, &vec![0.; n]));
        let old = sizes.map(|n| floats(&device, &vec![0.; n]));
        let vort = floats(&device, &vec![0.; (nx * ny * nz) as usize]);
        let phase = PhaseTransfer3D::new(
            &device,
            shape,
            capacity as u32,
            [&p, &v, &rows[0], &rows[1], &rows[2]],
            &metadata,
            [&grids[0], &grids[1], &grids[2]],
        )
        .unwrap();
        let mut gather = GpuG2p3D::new_with_metadata(
            &device,
            nx,
            ny,
            nz,
            capacity,
            p.clone(),
            v.clone(),
            rows[0].clone(),
            rows[1].clone(),
            rows[2].clone(),
            densities,
            &grids[0],
            &grids[1],
            &grids[2],
            &old[0],
            &old[1],
            &old[2],
            &vort,
            &phase.water[0],
            &phase.water[1],
            &phase.water[2],
            [0; 3],
            Some(metadata.clone()),
        );
        gather.set_conservative_phase_exchange(true);
        let sed = SedimentParams3D {
            drag_coefficient: rate,
            gravity: gravity.abs(),
            ..Default::default()
        };
        phase.prepare(&queue, count as u32, dx, dt, gravity, 8., sed);
        gather.upload_params(&queue, count as u32, dx, dt, 0.97, sed);
        let mut e = device.create_command_encoder(&Default::default());
        phase.encode_begin_step(&mut e);
        phase.encode_cell_volume(&mut e, count as u32);
        phase.encode_transfer(&mut e, count as u32);
        queue.submit(Some(e.finish()));
        let base = grids.each_ref().map(|b| read(&device, &queue, b));
        // An explicit full-gravity water predictor. Runtime integration will use
        // the existing gravity/flow/boundary passes between these two encodes.
        let forced = base
            .iter()
            .enumerate()
            .map(|(axis, plane)| {
                plane
                    .iter()
                    .map(|v| v + if axis == 1 { gravity * dt } else { 0. })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for a in 0..3 {
            queue.write_buffer(&grids[a], 0, bytemuck::cast_slice(&forced[a]));
        }
        let mut e = device.create_command_encoder(&Default::default());
        phase.encode_response(&mut e);
        queue.submit(Some(e.finish()));
        if matches!(label, "overflow" | "nonfinite" | "bad-rate") {
            let status = read(&device, &queue, phase.status_buffer())[0].to_bits();
            assert_ne!(status, 0, "must report {label}");
            let mut e = device.create_command_encoder(&Default::default());
            gather.encode(&mut e, count as u32);
            queue.submit(Some(e.finish()));
            let actual = read(&device, &queue, &v);
            let expected: &[f32] = bytemuck::cast_slice(&v_storage);
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "failed gather must retain velocities {label}"
            );
            for row in &rows {
                let actual = read(&device, &queue, row);
                assert!(
                    actual.iter().all(|v| *v == 0.),
                    "failed gather must retain affine rows"
                );
            }
            eprintln!(
                "{label}: status={status}, gather retained every particle velocity and affine row"
            );
            continue;
        }
        assert_eq!(
            read(&device, &queue, phase.status_buffer())[0].to_bits(),
            0,
            "status {label}"
        );
        let q0 = grids.each_ref().map(|b| read(&device, &queue, b));
        let auxiliary = phase.water.each_ref().map(|b| read(&device, &queue, b));
        let coefficients = phase.mobility.each_ref().map(|b| read(&device, &queue, b));
        if label == "drag" {
            let markers = PhaseTransfer3D::cell_count_offset(shape) as usize;
            let density = PhaseTransfer3D::cell_volume_offset(shape) as usize;
            let cells = (nx * ny * nz) as usize;
            let marker_sum: u32 = auxiliary[0][markers..markers + cells]
                .iter()
                .map(|v| v.to_bits())
                .sum();
            assert_eq!(marker_sum, rho.iter().filter(|r| **r <= 1.).count() as u32);
            let represented: f64 = auxiliary[0][density..density + cells]
                .iter()
                .map(|v| *v as f64)
                .sum();
            let expected: f64 = positions.iter().map(|p| p[3] as f64).sum();
            assert!(
                (represented - expected).abs() < 1e-5,
                "represented total volume {represented} vs {expected}"
            );
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let point =
                            DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5) * dx as f64;
                        let oracle: f64 = positions
                            .iter()
                            .map(|p| {
                                let delta = (point
                                    - DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
                                    / dx as f64;
                                p[3] as f64 * weight(delta.x) * weight(delta.y) * weight(delta.z)
                            })
                            .sum();
                        let index = ((z * ny + y) * nx + x) as usize;
                        assert!(
                            (auxiliary[0][density + index] as f64 - oracle).abs()
                                < 5e-6 * oracle.max(1.),
                            "cell represented volume {index}"
                        );
                    }
                }
            }
        }

        for axis in 0..3 {
            let mut dims = shape;
            dims[axis] += 1;
            let mut face = 0usize;
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let mut point = DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
                        point[axis] -= 0.5;
                        point *= dx as f64;
                        let mut vw = 0.;
                        let mut pw = 0.;
                        let mut grains = Vec::new();
                        for i in 0..count {
                            let delta = (point
                                - DVec3::new(
                                    positions[i][0] as f64,
                                    positions[i][1] as f64,
                                    positions[i][2] as f64,
                                ))
                                / dx as f64;
                            let volume = positions[i][3] as f64
                                * weight(delta.x)
                                * weight(delta.y)
                                * weight(delta.z);
                            if volume == 0. {
                                continue;
                            }
                            if rho[i] <= 1. {
                                vw += volume;
                                let affine_value = (0..3)
                                    .map(|b| affine_rows[axis][b] as f64 * delta[b] * dx as f64)
                                    .sum::<f64>();
                                pw += volume * (velocities[i][axis] as f64 + affine_value);
                            } else {
                                let mut predictor = DVec3::new(
                                    velocities[i][0] as f64,
                                    velocities[i][1] as f64,
                                    velocities[i][2] as f64,
                                );
                                for a in 0..3 {
                                    predictor[a] += (0..3)
                                        .map(|b| affine_rows[a][b] as f64 * delta[b] * dx as f64)
                                        .sum::<f64>();
                                }
                                predictor.y += (gravity * dt) as f64;
                                grains.push(GrainSample {
                                    volume,
                                    density: rho[i] as f64,
                                    relaxation_rate: rate as f64,
                                    velocity: predictor,
                                });
                            }
                        }
                        if vw > 1e-5 {
                            let mut water = DVec3::splat(pw / vw);
                            water[axis] += if axis == 1 { (gravity * dt) as f64 } else { 0. };
                            let response =
                                FaceResponse::new(vw, water, &grains, 8., dt as f64).unwrap();
                            let offset =
                                PhaseTransfer3D::aux_offset(sizes[axis] as u64) as usize + 4 * face;
                            let check = |a: f32, b: f64| {
                                assert!(
                                    (a as f64 - b).abs() < 3e-5 * b.abs().max(1.),
                                    "{label} axis={axis} face={face}: {a} vs {b}"
                                )
                            };
                            check(q0[axis][face], response.mixture_predictor[axis]);
                            check(coefficients[axis][face + 1], response.pressure_mobility);
                            check(auxiliary[axis][offset], response.water_predictor[axis]);
                            check(
                                auxiliary[axis][offset + 1],
                                response.water_pressure_mobility / response.pressure_mobility,
                            );
                            check(
                                auxiliary[axis][offset + 3],
                                response.pressure_scale / response.pressure_mobility,
                            );
                        }
                        face += 1;
                    }
                }
            }
        }
        let mut e = device.create_command_encoder(&Default::default());
        if pressure {
            let solver = GpuPressure3D::new_with_mobility(
                &device,
                nx,
                ny,
                nz,
                &grids[0],
                &grids[1],
                &grids[2],
                phase.mobility.each_ref(),
            );
            let mut types = vec![0; (nx * ny * nz) as usize];
            for pp in &positions {
                let q = pp[..3]
                    .iter()
                    .map(|v| (v / dx).floor() as u32)
                    .collect::<Vec<_>>();
                types[((q[2] * ny + q[1]) * nx + q[0]) as usize] = 1;
            }
            solver.upload_cell_types(&queue, &types, dx, 63);
            solver.clear_pressure(&mut e);
            solver.encode(&mut e, 80);
        }
        gather.encode(&mut e, count as u32);
        queue.submit(Some(e.finish()));
        let result = read(&device, &queue, &v);
        let affine_result = rows.each_ref().map(|b| read(&device, &queue, b));
        if matches!(label, "drag" | "exact-capacity" | "reserved" | "reversed") {
            let mut snapshot = Vec::new();
            for a in 0..3 {
                snapshot.extend_from_slice(&q0[a]);
                snapshot.extend_from_slice(&auxiliary[a]);
                snapshot.extend_from_slice(&coefficients[a]);
            }
            for i in 0..count {
                let i = if reverse { count - 1 - i } else { i };
                snapshot.extend_from_slice(&result[4 * i..4 * i + 4]);
                for row in &affine_result {
                    snapshot.extend_from_slice(&row[4 * i..4 * i + 4]);
                }
            }
            if let Some(expected) = &exact_snapshot {
                assert_eq!(&snapshot, expected, "order/allocation {label}");
            } else {
                exact_snapshot = Some(snapshot);
            }
        }
        let mut angular_before = DVec3::ZERO;
        let mut angular_after = DVec3::ZERO;
        let mut energy_before = 0.;
        let mut energy_after = 0.;
        let d = (dx as f64).powi(2) / 4.;
        let mut before = DVec3::ZERO;
        let mut after = DVec3::ZERO;
        let mut total_mass = 0.;
        for i in 0..count {
            let mass = positions[i][3] as f64 * rho[i] as f64;
            total_mass += mass;
            before += DVec3::new(
                velocities[i][0] as f64,
                velocities[i][1] as f64,
                velocities[i][2] as f64,
            ) * mass;
            after += DVec3::new(
                result[4 * i] as f64,
                result[4 * i + 1] as f64,
                result[4 * i + 2] as f64,
            ) * mass;
            let position = DVec3::new(
                positions[i][0] as f64,
                positions[i][1] as f64,
                positions[i][2] as f64,
            );
            let initial = DVec3::new(
                velocities[i][0] as f64,
                velocities[i][1] as f64,
                velocities[i][2] as f64,
            );
            let final_v = DVec3::new(
                result[4 * i] as f64,
                result[4 * i + 1] as f64,
                result[4 * i + 2] as f64,
            );
            let intrinsic = |rows: [[f64; 3]; 3]| {
                DVec3::new(
                    rows[2][1] - rows[1][2],
                    rows[0][2] - rows[2][0],
                    rows[1][0] - rows[0][1],
                ) * d
            };
            let before_rows = affine_rows.map(|r| r.map(f64::from));
            let after_rows = std::array::from_fn::<_, 3, _>(|a| {
                std::array::from_fn::<_, 3, _>(|b| affine_result[a][4 * i + b] as f64)
            });
            angular_before += (position.cross(initial) + intrinsic(before_rows)) * mass;
            angular_after += (position.cross(final_v) + intrinsic(after_rows)) * mass;
            energy_before += 0.5
                * mass
                * (initial.length_squared()
                    + d * before_rows.iter().flatten().map(|c| c * c).sum::<f64>());
            energy_after += 0.5
                * mass
                * (final_v.length_squared()
                    + d * after_rows.iter().flatten().map(|c| c * c).sum::<f64>());
            if affine {
                for a in 0..3 {
                    for b in 0..3 {
                        assert!(
                            (after_rows[a][b] - before_rows[a][b]).abs() < 2e-5,
                            "rotation C particle={i} row={a} column={b}"
                        );
                    }
                }
            }
            if co_moving {
                for a in 0..3 {
                    let expected = velocities[i][a] + if a == 1 { gravity * dt } else { 0. };
                    assert!(
                        (result[4 * i + a] - expected).abs() < 2e-6,
                        "freefall particle {i} axis {a}"
                    );
                }
            }
        }
        let external = DVec3::new(0., (gravity * dt) as f64 * total_mass, 0.);
        let error = (after - before - external).length();
        if gravity == 0. {
            let angular_error = (angular_after - angular_before).length();
            assert!(
                angular_error < 2e-7 * total_mass,
                "{label} angular momentum {angular_error}"
            );
            assert!(
                energy_after <= energy_before + 2e-7 * total_mass,
                "{label} energy {energy_before} -> {energy_after}"
            );
            eprintln!("{label}: angular residual {angular_error:.9}, energy {energy_before:.9} -> {energy_after:.9}");
        }
        assert!(
            error < 2e-7 * total_mass,
            "{label} momentum error={error} / mass={total_mass}"
        );
        eprintln!("{label}: momentum residual {error:.9} / mass {total_mass:.6}");
        if label == "drag" {
            queue.write_buffer(&v, 0, bytemuck::bytes_of(&f32::NAN));
            let mut encoder = device.create_command_encoder(&Default::default());
            phase.encode_validate_particles(&mut encoder, count as u32);
            queue.submit(Some(encoder.finish()));
            assert_eq!(
                read(&device, &queue, phase.status_buffer())[0].to_bits() & 8,
                8,
                "nonfinite output must be refused before advection"
            );
        }
    }
}
