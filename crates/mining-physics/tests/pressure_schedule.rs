//! Compare every pressure cell against the original dispatch schedule, including
//! odd grid extents, mixed solids/air and all boundary flags.
use game::gpu::pressure_3d::GpuPressure3D;

#[test]
#[ignore = "requires real Metal/Vulkan compute adapter"]
fn packed_checkerboard_preserves_pressure_with_odd_grid_extents() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("real GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let (width, height, depth) = (17, 13, 9);
    let count = (width * height * depth) as usize;
    let cells: Vec<u32> = (0..count)
        .map(|i| match i % 11 {
            0 => 0,
            1 => 2,
            _ => 1,
        })
        .collect();
    let rhs: Vec<f32> = (0..count)
        .map(|i| ((i * 17 % 101) as f32 - 50.) * 0.01)
        .collect();
    let solve = |packed, cached, iterations, boundaries| {
        let buffer = |n: u32| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pressure regression velocity"),
                size: n as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let u = buffer((width + 1) * height * depth);
        let v = buffer(width * (height + 1) * depth);
        let w = buffer(width * height * (depth + 1));
        let mut pressure = GpuPressure3D::new(&device, width, height, depth, &u, &v, &w);
        pressure.compact_iterations = packed;
        pressure.cached_stencil = cached;
        pressure.upload_cell_types(&queue, &cells, 0.01, boundaries);
        queue.write_buffer(&pressure.divergence_buffer, 0, bytemuck::cast_slice(&rhs));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pressure regression readback"),
            size: count as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        pressure.clear_pressure(&mut encoder);
        pressure.encode_iterations_only(&mut encoder, iterations);
        encoder.copy_buffer_to_buffer(&pressure.pressure_buffer, 0, &staging, 0, count as u64 * 4);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let values =
            bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
        staging.unmap();
        values
    };
    for boundaries in [0, 10, 14, 63] {
        for iterations in [1, 7, 40, 80] {
            let reference = solve(false, false, iterations, boundaries);
            let packed = solve(true, false, iterations, boundaries);
            assert!(
                reference.iter().all(|p| p.is_finite()) && reference.iter().any(|p| p.abs() > 1e-7)
            );
            assert_eq!(
                reference, packed,
                "sweeps={iterations} boundaries={boundaries}"
            );
            let cached = solve(true, true, iterations, boundaries);
            assert_eq!(
                reference, cached,
                "cached stencil: sweeps={iterations} boundaries={boundaries}"
            );
        }
    }
}
