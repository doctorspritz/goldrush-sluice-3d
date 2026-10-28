//! Exact raw-accumulator control for the production fixed-point P2G shader.
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a real GPU; run explicitly with --ignored"]
fn skipping_quantized_zero_preserves_every_accumulator_and_phase_count() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production P2G zero-atomic equivalence"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../game/src/gpu/shaders/p2g_scatter_3d.wgsl").into(),
        ),
    });
    let pipelines: Vec<_> = [0., 1.]
        .into_iter()
        .map(|value| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reference or zero-eliding P2G"),
                layout: None,
                module: &shader,
                entry_point: Some("scatter"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &HashMap::from([("ELIDE_ZERO_ATOMICS".into(), value)]),
                    ..Default::default()
                },
                cache: None,
            })
        })
        .collect();
    const N: usize = 2053;
    let dims = [9usize, 7, 11];
    for motion in [false, true] {
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        let mut rows: [Vec<[f32; 4]>; 3] = std::array::from_fn(|_| Vec::new());
        let mut densities = Vec::<f32>::with_capacity(N);
        for i in 0..N {
            let hash = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
            let mut p = std::array::from_fn::<_, 3, _>(|a| {
                let bits = hash.rotate_left(a as u32 * 9);
                if i % 11 == 0 {
                    if bits & 1 == 0 {
                        -0.003
                    } else {
                        dims[a] as f32 * 0.01 + 0.003
                    }
                } else {
                    (2.1 + (bits % 97) as f32 / 97. * 2.7) * 0.01
                }
            })
            .to_vec();
            let (density, weight) = if i % 13 == 0 {
                (19.3, 0.000002)
            } else if i % 9 == 0 {
                (2.65, 16.)
            } else {
                (1., 1.)
            };
            p.push(if i % 97 == 0 { 0. } else { weight });
            positions.push([p[0], p[1], p[2], p[3]]);
            densities.push(density);
            let v = std::array::from_fn::<_, 4, _>(|a| {
                if a == 3 || !motion {
                    0.
                } else if i % 17 == 0 {
                    (a as f32 - 1.) * 0.000001
                } else {
                    ((hash.rotate_left(a as u32 * 7) % 101) as f32 / 100. - 0.5) * 0.8
                }
            });
            velocities.push(v);
            for (a, row) in rows.iter_mut().enumerate() {
                row.push(std::array::from_fn(|b| {
                    if !motion || b == 3 {
                        0.
                    } else {
                        (((i + a * 3 + b * 7) % 11) as f32 - 5.) * 0.13
                    }
                }));
            }
        }
        let input = |bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let inputs = [
            input(bytemuck::cast_slice(&positions)),
            input(bytemuck::cast_slice(&velocities)),
            input(bytemuck::cast_slice(&rows[0])),
            input(bytemuck::cast_slice(&rows[1])),
            input(bytemuck::cast_slice(&rows[2])),
            input(bytemuck::cast_slice(&densities)),
        ];
        for (include_sediment, paired) in [(true, false), (true, true), (false, false)] {
            let parameters = [
                0.01f32.to_bits(),
                dims[0] as u32,
                dims[1] as u32,
                dims[2] as u32,
                N as u32,
                include_sediment as u32,
                paired as u32,
                0,
            ];
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let cells = dims.iter().product::<usize>();
            let faces = [
                (dims[0] + 1) * dims[1] * dims[2],
                dims[0] * (dims[1] + 1) * dims[2],
                dims[0] * dims[1] * (dims[2] + 1),
            ];
            let sizes: Vec<_> = faces
                .into_iter()
                .flat_map(|n| {
                    let size = if include_sediment {
                        n.div_ceil(64) * 64 + n
                    } else {
                        n
                    };
                    [size, size]
                })
                .chain([cells, cells])
                .collect();
            let outputs: Vec<_> = sizes
                .iter()
                .map(|&n| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: n as u64 * 4,
                        mapped_at_creation: false,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    })
                })
                .collect();
            let bytes = sizes.iter().sum::<usize>() as u64 * 4;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            });
            let mut results = Vec::new();
            for pipeline in &pipelines {
                let mut entries = vec![wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                }];
                for binding in 1..=14 {
                    let buffer = match binding {
                        1..=5 => &inputs[binding - 1],
                        6..=12 => &outputs[binding - 6],
                        13 => &inputs[5],
                        14 => &outputs[7],
                        _ => unreachable!(),
                    };
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    });
                }
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                for output in &outputs {
                    encoder.clear_buffer(output, 0, None);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups((N as u32).div_ceil(128), 1, 1);
                }
                let mut offset = 0;
                for output in &outputs {
                    encoder.copy_buffer_to_buffer(output, 0, &staging, offset, output.size());
                    offset += output.size();
                }
                queue.submit(Some(encoder.finish()));
                let (tx, rx) = std::sync::mpsc::channel();
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                let values: Vec<i32> =
                    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
                staging.unmap();
                results.push(values);
            }
            assert_eq!(
                results[0], results[1],
                "motion={motion}, sediment={include_sediment}, paired={paired}"
            );
            let values = &results[1];
            let water_offset: usize = sizes[..6].iter().sum();
            let water_count = densities.iter().filter(|&&d| d <= 1.).count();
            assert_eq!(
                values[water_offset..water_offset + cells]
                    .iter()
                    .map(|&x| x as usize)
                    .sum::<usize>(),
                water_count
            );
            assert_eq!(
                values[water_offset + cells..]
                    .iter()
                    .map(|&x| x as usize)
                    .sum::<usize>(),
                if include_sediment { N - water_count } else { 0 }
            );
            let mut offset = 0;
            for (i, &size) in sizes[..6].iter().enumerate() {
                let field = &values[offset..offset + size];
                if i % 2 == 1 {
                    assert!(field.iter().any(|&v| v > 0), "empty weight field");
                } else if !motion {
                    assert!(field.iter().all(|&v| v == 0), "resting momentum changed");
                } else {
                    assert!(field.iter().any(|&v| v != 0), "moving momentum vanished");
                }
                offset += size;
            }
        }
    }
}
