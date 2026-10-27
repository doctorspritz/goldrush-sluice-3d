//! Compare complete warm pressure fields across the original, packed and cached schedules.
//! Domains change on the same live solver, including zero cells in either parity.
use game::gpu::pressure_3d::GpuPressure3D;
use wgpu::util::DeviceExt;

fn buffer(device: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("changing pressure domains control"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("changing pressure domains readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let values = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    values
}

#[test]
#[ignore = "requires actual Metal/Vulkan compute; explicitly run with --ignored"]
fn pressure_schedules_preserve_every_cell_through_empty_changed_and_warm_domains() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    // Default limits include only eight storage buffers per shader stage.
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    let mut comparisons = 0;
    for [width, height, depth] in [[1, 1, 1], [17, 13, 9], [32, 8, 4], [9, 7, 5]] {
        let count = (width * height * depth) as usize;
        let sizes = [
            (width + 1) * height * depth,
            width * (height + 1) * depth,
            width * height * (depth + 1),
        ];
        for weighted in [false, true] {
            let velocity: Vec<_> = sizes
                .iter()
                .map(|&n| buffer(&device, &vec![0.; n as usize]))
                .collect();
            let mobility: Vec<_> = sizes
                .iter()
                .enumerate()
                .map(|(axis, &n)| {
                    let mut v = vec![0.];
                    v.extend((0..n).map(|i| {
                        0.05 + 0.95 * ((i as f32 * 0.17 + axis as f32).sin() * 0.5 + 0.5)
                    }));
                    buffer(&device, &v)
                })
                .collect();
            let make = |compact, cached| {
                let mut solver = if weighted {
                    GpuPressure3D::new_with_mobility(
                        &device,
                        width,
                        height,
                        depth,
                        &velocity[0],
                        &velocity[1],
                        &velocity[2],
                        [&mobility[0], &mobility[1], &mobility[2]],
                    )
                } else {
                    GpuPressure3D::new(
                        &device,
                        width,
                        height,
                        depth,
                        &velocity[0],
                        &velocity[1],
                        &velocity[2],
                    )
                };
                solver.compact_iterations = compact;
                solver.cached_stencil = cached;
                solver
            };
            let dense = make(false, false);
            let packed = make(true, false);
            let cached = make(true, true);
            let seed: Vec<_> = (0..count).map(|i| 0.02 * (i as f32 * 0.41).cos()).collect();
            for solver in [&dense, &packed, &cached] {
                queue.write_buffer(&solver.pressure_buffer, 0, bytemuck::cast_slice(&seed));
            }
            for (stage, (open, iterations, ready)) in [
                (0, 1, false),
                (14, 7, true),
                (63, 80, true),
                (63, 1, true),
                (10, 80, false),
                (0, 0, true),
                (0, 7, true),
            ]
            .into_iter()
            .enumerate()
            {
                for m in &mobility {
                    queue.write_buffer(m, 0, bytemuck::bytes_of(&if ready { 1f32 } else { 0f32 }));
                }
                let cells: Vec<_> = (0..count)
                    .map(|i| {
                        let x = i % width as usize;
                        let y = (i / width as usize) % height as usize;
                        let z = i / (width * height) as usize;
                        match stage {
                            0 | 5 | 6 => match (i + stage * 3) % 11 {
                                0 => 0,
                                1 => 2,
                                _ => 1,
                            },
                            1 => {
                                if x < 3 && y < 2 && z == 0 {
                                    1
                                } else {
                                    0
                                }
                            }
                            2 => {
                                if i % 2 == 0 {
                                    0
                                } else {
                                    2
                                }
                            }
                            3 => {
                                if (x + y + z).is_multiple_of(2) {
                                    1
                                } else {
                                    0
                                }
                            }
                            4 => 1,
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                let rhs: Vec<_> = (0..count)
                    .map(|i| 0.03 * ((i * 17 + stage * 7) % 101) as f32 - 1.5)
                    .collect();
                for solver in [&dense, &packed, &cached] {
                    solver.upload_cell_types(&queue, &cells, 0.01, open);
                    queue.write_buffer(&solver.divergence_buffer, 0, bytemuck::cast_slice(&rhs));
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode_iterations_only(&mut encoder, iterations);
                    queue.submit(Some(encoder.finish()));
                }
                let a = read(&device, &queue, &dense.pressure_buffer);
                for (name, candidate) in [("packed", &packed), ("cached", &cached)] {
                    let b = read(&device, &queue, &candidate.pressure_buffer);
                    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
                    assert_eq!(a.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"shape{width}/{height}/{depth} weighted{weighted} {name} stage{stage} open{open} sweeps{iterations}");
                    comparisons += 1;
                }
            }
        }
    }
    assert_eq!(comparisons, 112);
    println!("{comparisons} complete pressure fields match bit-for-bit, including empty parities and live changes");
}
