//! Execute the actual pressure source and projection on a real device.
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}
fn buffer(device: &wgpu::Device, data: &[u8], uniform: bool) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | if uniform {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            },
    })
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    data
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn volume_source_matches_linear_continuity_and_preserves_free_surfaces() {
    let (device, queue) = device();
    let params = buffer(&device, &[0; 32], true);
    let cells = buffer(&device, bytemuck::cast_slice(&[1u32]), false);
    let moving = buffer(&device, &[0; 4], false);
    let divergence = buffer(&device, &[0; 4], false);
    let status = buffer(&device, &[0; 16], false);
    let markers = buffer(&device, bytemuck::cast_slice(&[1u32]), false);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 2, 4, 5, 6, 7].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding == 2 || binding == 4 || binding == 7,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cells.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: moving.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: divergence.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: status.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: markers.as_entire_binding(),
            },
        ],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../game/src/gpu/shaders/density_error_3d.wgsl").into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("add_density_rhs"),
        compilation_options: Default::default(),
        cache: None,
    });
    let run = |open: u32, volume: f32, dt: f32, strength: f32, initial: f32| {
        queue.write_buffer(
            &params,
            0,
            bytemuck::cast_slice(&[
                1u32,
                1,
                1,
                8f32.to_bits(),
                dt.to_bits(),
                1,
                open,
                strength.to_bits(),
            ]),
        );
        queue.write_buffer(&moving, 0, bytemuck::bytes_of(&volume));
        queue.write_buffer(&divergence, 0, bytemuck::bytes_of(&initial));
        queue.write_buffer(&status, 0, &[0; 16]);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        let d = read(&device, &queue, &divergence);
        let s = read(&device, &queue, &status);
        (
            f32::from_ne_bytes(d[..4].try_into().unwrap()),
            u32::from_ne_bytes(s[..4].try_into().unwrap()),
        )
    };
    let mut cases = 0;
    for open in [0u32, 3, 63] {
        // Input already contains conservative boundary transfer. A solid
        // ghost volume must never be added to this actual moving material.
        for ratio in [0.6, 1., 1.25, 2.] {
            for dt in [1f32 / 60., 1. / 120., 1. / 240.] {
                for strength in [0f32, 0.25, 1.] {
                    let volume = (8. * ratio) as f32;
                    let (rhs, flag) = run(open, volume, dt, strength, 3.);
                    assert_eq!(flag, 0);
                    let rate = 3. - rhs as f64;
                    let observed = volume as f64;
                    let predicted = volume as f64 - 8. * dt as f64 * rate;
                    let desired = if open != 0 && observed < 8. {
                        observed
                    } else {
                        observed + (8. - observed) * strength as f64
                    };
                    assert!(
                        (predicted - desired).abs() < 2e-5,
                        "open={open},r={ratio},dt={dt},alpha={strength}: {predicted} vs {desired}"
                    );
                    if open != 0 && ratio < 1. {
                        assert_eq!(rhs, 3., "free-surface suction");
                    }
                    cases += 1;
                }
            }
        }
    }
    // A carried-water halo is not a new occupied volume cell. It must not
    // create suction merely because the pressure stencil now includes it.
    queue.write_buffer(&markers, 0, bytemuck::cast_slice(&[0u32]));
    let (rhs, flag) = run(0, 4., 0.01, 1., 3.);
    assert_eq!((rhs, flag), (3., 0));
    let (rhs, flag) = run(0, 12., 0.01, 1., 3.);
    assert_eq!(flag, 0);
    assert!(
        (rhs + 47.).abs() < 1e-5,
        "halo still corrects positive excess"
    );
    queue.write_buffer(&markers, 0, bytemuck::cast_slice(&[1u32]));
    for (volume, dt, strength) in [
        (0., 0.01, 1.),
        (-1., 0.01, 1.),
        (f32::NAN, 0.01, 1.),
        (8., 0., 1.),
        (8., f32::INFINITY, 1.),
        (8., 0.01, -1.),
        (8., 0.01, 2.),
        (8., 0.01, f32::NAN),
    ] {
        let (rhs, flag) = run(63, volume, dt, strength, 3.);
        assert_eq!(flag, 16);
        assert_eq!(rhs, 3., "invalid source changed input RHS");
    }
    eprintln!("actual volume-source cases={cases}; invalid=8");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn boundary_volume_preserves_mass_order_capacity_and_constant_voxel_states() {
    use game::gpu::phase_transfer_3d::PhaseTransfer3D;
    let (device, queue) = device();
    let shape = [4u32, 4, 4];
    let mut cells = vec![0u32; 64];
    for z in 1..3 {
        for y in 1..3 {
            for x in 1..3 {
                cells[(z * 4 + y) * 4 + x] = 2;
            }
        }
    }
    let mut points = Vec::new();
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                if cells[((z / 2) * 4 + y / 2) * 4 + x / 2] == 2 {
                    continue;
                }
                points.push([
                    (x as f32 + 0.5) * 0.005,
                    (y as f32 + 0.5) * 0.005,
                    (z as f32 + 0.5) * 0.005,
                    0.2 + ((x * 13 + y * 7 + z * 3) % 19) as f32 * 0.1,
                ]);
            }
        }
    }
    let run = |points: &[[f32; 4]], open: u32, padding: usize| {
        let capacity = points.len() + padding;
        let mut padded = vec![[0f32; 4]; capacity];
        padded[..points.len()].copy_from_slice(points);
        let positions = buffer(&device, bytemuck::cast_slice(&padded), false);
        let zero = buffer(&device, &vec![0u8; capacity * 16], false);
        let mut data = vec![0f32; capacity * 4];
        data[..points.len()].fill(1.);
        let metadata = buffer(&device, bytemuck::cast_slice(&data), false);
        let grids = [
            buffer(&device, &[0; 320], false),
            buffer(&device, &[0; 320], false),
            buffer(&device, &[0; 320], false),
        ];
        let types = buffer(&device, bytemuck::cast_slice(&cells), false);
        let distances: Vec<f32> = cells
            .iter()
            .map(|&c| if c == 2 { -1. } else { 1. })
            .collect();
        let sdf = buffer(&device, bytemuck::cast_slice(&distances), false);
        let phase = PhaseTransfer3D::new(
            &device,
            shape,
            capacity as u32,
            [&positions, &zero, &zero, &zero, &zero],
            &metadata,
            grids.each_ref(),
        )
        .unwrap();
        let boundary = phase.boundary_volume(&device, &positions, &metadata, &types, &sdf);
        phase.prepare(
            &queue,
            points.len() as u32,
            0.01,
            1. / 120.,
            0.,
            8.,
            Default::default(),
        );
        phase.set_volume_boundaries(&queue, open);
        let mut encoder = device.create_command_encoder(&Default::default());
        phase.encode_begin_step(&mut encoder);
        phase.encode_cell_volume_with_boundaries(&mut encoder, points.len() as u32, &boundary);
        queue.submit(Some(encoder.finish()));
        let bytes = read(&device, &queue, &phase.water[0]);
        let start = PhaseTransfer3D::cell_volume_offset(shape) as usize * 4;
        let values = bytemuck::cast_slice::<u8, f32>(&bytes[start..start + 64 * 4]).to_vec();
        let status = read(&device, &queue, phase.status_buffer());
        let flag = u32::from_ne_bytes(status[..4].try_into().unwrap());
        (values, flag)
    };
    let weight = |x: f64| {
        let x = x.abs();
        if x < 0.5 {
            0.75 - x * x
        } else if x < 1.5 {
            0.5 * (1.5 - x).powi(2)
        } else {
            0.
        }
    };
    for open in [0u32, 1, 2, 4, 8, 16, 32, 63] {
        let (values, status) = run(&points, open, 0);
        assert_eq!(status, 0);
        let mut expected = vec![0f64; 64];
        let mut lost = 0.;
        for p in &points {
            let q = [
                p[0] as f64 / 0.01f32 as f64 - 0.5,
                p[1] as f64 / 0.01f32 as f64 - 0.5,
                p[2] as f64 / 0.01f32 as f64 - 0.5,
            ];
            let center = q.map(|x| (x + 0.5).floor() as i32);
            let home = [
                (p[0] / 0.01).floor() as usize,
                (p[1] / 0.01).floor() as usize,
                (p[2] / 0.01).floor() as usize,
            ];
            for z in center[2] - 1..=center[2] + 1 {
                for y in center[1] - 1..=center[1] + 1 {
                    for x in center[0] - 1..=center[0] + 1 {
                        let node = [x, y, z];
                        let w = (0..3)
                            .map(|a| weight(q[a] - node[a] as f64))
                            .product::<f64>()
                            * p[3] as f64;
                        let leaves = (0..3).any(|a| {
                            (node[a] < 0 && open & (1 << (a * 2)) != 0)
                                || (node[a] >= 4 && open & (2 << (a * 2)) != 0)
                        });
                        if leaves {
                            lost += w;
                            continue;
                        }
                        let inside = node.iter().all(|&x| (0..4).contains(&x));
                        let index = if inside
                            && cells[(z as usize * 4 + y as usize) * 4 + x as usize] != 2
                        {
                            (z as usize * 4 + y as usize) * 4 + x as usize
                        } else {
                            (home[2] * 4 + home[1]) * 4 + home[0]
                        };
                        expected[index] += w;
                    }
                }
            }
        }
        for (i, (&actual, &expected)) in values.iter().zip(&expected).enumerate() {
            assert!(
                (actual as f64 - expected).abs() < 2e-5,
                "open={open},cell={i}: {actual} vs {expected}"
            );
        }
        let actual = values.iter().map(|&x| x as f64).sum::<f64>();
        let total = points.iter().map(|p| p[3] as f64).sum::<f64>();
        assert!(
            (actual + lost - total).abs() / total < 2e-7,
            "boundary volume lost material"
        );
        let reverse: Vec<_> = points.iter().copied().rev().collect();
        let reordered = run(&reverse, open, 1024);
        assert_eq!(reordered.1, 0);
        assert_eq!(values, reordered.0, "order/reservation altered volume");
    }
    let mut uniform = points.clone();
    for p in &mut uniform {
        p[3] = 1.;
    }
    let (values, status) = run(&uniform, 0, 0);
    assert_eq!(status, 0);
    for (i, &v) in values.iter().enumerate() {
        assert!(
            (v - if cells[i] == 2 { 0. } else { 8. }).abs() < 1e-5,
            "voxel corner biased resting volume: cell={i},value={v}"
        );
    }
    let (cut, status) = run(&[[0.012, 0.012, 0.012, 1.]], 0, 0);
    assert_eq!(status, 0, "positive-SDF cutcell point was refused");
    assert!(
        (cut.iter().sum::<f32>() - 1.).abs() < 1e-6,
        "cutcell volume was not retained"
    );
    assert!(
        cut.iter().zip(&cells).all(|(&v, &c)| c != 2 || v == 0.),
        "cutcell volume stayed in inaccessible pressure cells"
    );
    let (values, status) = run(&[[0.015, 0.015, 0.015, 1.]], 0, 0);
    assert_eq!(status, 32);
    assert!(values.iter().all(|x| x.is_finite()));
    let (_, status) = run(&[[0.0025, 0.0025, 0.0025, 1e8]], 0, 0);
    assert_eq!(status, 1, "folded per-particle range was not refused");
}
