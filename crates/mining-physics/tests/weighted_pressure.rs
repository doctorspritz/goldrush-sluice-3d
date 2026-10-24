//! Real-device manufactured solutions for every production pressure dispatch.
use game::gpu::pressure_3d::GpuPressure3D;
use wgpu::util::DeviceExt;

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, (n * 4) as u64);
    queue.submit(Some(encoder.finish()));
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
fn buffer(device: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn variable_mobility_matches_manufactured_solution_all_dispatches_and_gpu_header_transition() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    let shape = [7usize, 6, 5];
    let [nx, ny, nz] = shape;
    let dx = 0.03f32;
    let ci = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    for closed in [false, true] {
        let mut types = vec![1u32; nx * ny * nz];
        if !closed {
            types[ci(3, 2, 2)] = 2;
            types[ci(2, 4, 1)] = 2;
            types[ci(1, 1, 3)] = 0;
            types[ci(5, 4, 3)] = 0;
        }
        let pressure: Vec<f32> = (0..types.len())
            .map(|i| {
                if types[i] == 1 {
                    0.001 * ((i as f32 * 0.37).cos() + 0.3 * (i as f32 * 0.11).sin())
                } else {
                    0.
                }
            })
            .collect();
        let cell = |q: [i32; 3]| -> (u32, f32) {
            if (0..3).any(|a| q[a] < 0 || q[a] >= shape[a] as i32) {
                (if closed { 2 } else { 0 }, 0.)
            } else {
                let i = ci(q[0] as usize, q[1] as usize, q[2] as usize);
                (types[i], pressure[i])
            }
        };
        let mut mobility = [Vec::new(), Vec::new(), Vec::new()];
        let mut velocity = [Vec::new(), Vec::new(), Vec::new()];
        for a in 0..3 {
            mobility[a].push(1.);
            let mut face_shape = shape;
            face_shape[a] += 1;
            for z in 0..face_shape[2] {
                for y in 0..face_shape[1] {
                    for x in 0..face_shape[0] {
                        let q = [x as i32, y as i32, z as i32];
                        let mut left = q;
                        left[a] -= 1;
                        let (lt, lp) = cell(left);
                        let (rt, rp) = cell(q);
                        let index = velocity[a].len();
                        let k = if index % 29 == 0 {
                            0.
                        } else {
                            0.025 + 6. * ((index * 17 + 3 * a) % 101) as f32 / 100.
                        };
                        mobility[a].push(k);
                        velocity[a].push(if lt != 2 && rt != 2 {
                            (rp - lp) / dx * k
                        } else {
                            0.
                        });
                    }
                }
            }
        }
        for mode in [2, 1, 0] {
            eprintln!("pressure control closed={closed} mode={mode}");
            let grids = velocity.each_ref().map(|v| buffer(&device, v));
            let coefficients = mobility.each_ref().map(|v| buffer(&device, v));
            let mut solver = GpuPressure3D::new_with_mobility(
                &device,
                nx as u32,
                ny as u32,
                nz as u32,
                &grids[0],
                &grids[1],
                &grids[2],
                coefficients.each_ref(),
            );
            solver.compact_iterations = mode != 0;
            solver.cached_stencil = mode == 2;
            solver.upload_cell_types(&queue, &types, dx, if closed { 0 } else { 63 });
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.clear_pressure(&mut encoder);
            // The separate-pass reference path creates Metal command buffers.
            // Submit bounded batches rather than exhausting its unsubmitted pool.
            solver.encode_divergence_only(&mut encoder);
            queue.submit(Some(encoder.finish()));
            for _ in 0..10 {
                let mut batch = device.create_command_encoder(&Default::default());
                solver.encode_iterations_only(&mut batch, 80);
                queue.submit(Some(batch.finish()));
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode(&mut encoder, 0);
            solver.encode_divergence_only(&mut encoder);
            queue.submit(Some(encoder.finish()));
            let actual = read(&device, &queue, &solver.pressure_buffer, pressure.len());
            let gauge = if closed { actual[0] - pressure[0] } else { 0. };
            let max_error = actual
                .iter()
                .zip(&pressure)
                .zip(&types)
                .filter(|(_, t)| **t == 1)
                .map(|((a, p), _)| (a - p - gauge).abs())
                .fold(0., f32::max);
            let residual = read(&device, &queue, &solver.divergence_buffer, pressure.len())
                .into_iter()
                .map(f32::abs)
                .fold(0., f32::max);
            let max_velocity = grids
                .iter()
                .zip(&velocity)
                .map(|(b, v)| {
                    read(&device, &queue, b, v.len())
                        .into_iter()
                        .map(f32::abs)
                        .fold(0., f32::max)
                })
                .fold(0., f32::max);
            assert!(
                max_error < 1e-7,
                "mode={mode} closed={closed} pressure error={max_error}"
            );
            assert!(
                residual < 2e-4,
                "mode={mode} closed={closed} divergence={residual}"
            );
            assert!(
                max_velocity < 2e-5,
                "mode={mode} closed={closed} velocity={max_velocity}"
            );
        }
        // The same command buffer must support ordinary density repair, then
        // assembled phase mobility. Queue writes alone cannot perform this switch.
        let grids = velocity.each_ref().map(|v| buffer(&device, v));
        let coefficients = mobility.each_ref().map(|v| buffer(&device, v));
        let solver = GpuPressure3D::new_with_mobility(
            &device,
            nx as u32,
            ny as u32,
            nz as u32,
            &grids[0],
            &grids[1],
            &grids[2],
            coefficients.each_ref(),
        );
        solver.upload_cell_types(&queue, &types, dx, if closed { 0 } else { 63 });
        let one = buffer(&device, &[1.]);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&coefficients[0], 0, Some(4));
        solver.clear_pressure(&mut encoder);
        solver.encode_iterations_only(&mut encoder, 3);
        encoder.copy_buffer_to_buffer(&one, 0, &coefficients[0], 0, 4);
        solver.clear_pressure(&mut encoder);
        solver.encode(&mut encoder, 800);
        solver.encode_divergence_only(&mut encoder);
        queue.submit(Some(encoder.finish()));
        let residual = read(&device, &queue, &solver.divergence_buffer, pressure.len())
            .into_iter()
            .map(f32::abs)
            .fold(0., f32::max);
        assert!(residual < 2e-4, "GPU header transition residual={residual}");
    }
}
