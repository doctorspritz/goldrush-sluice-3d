//! Run the production kernels in isolation: contact must preserve tangential
//! transport, and volume repair must sample its actual staggered face locations.
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("real GPU");
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}

fn run(
    source: &str,
    entry: &str,
    data: Vec<Vec<u8>>,
    writable: &[usize],
    count: u32,
) -> Vec<Vec<f32>> {
    let (device, queue) = device();
    let buffers: Vec<_> = data
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("isolated production water kernel"),
                contents: bytes,
                usage: if i == 0 {
                    wgpu::BufferUsages::UNIFORM
                } else {
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                },
            })
        })
        .collect();
    let entries: Vec<_> = (0..buffers.len())
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: !writable.contains(&i),
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let outputs: Vec<_> = writable
        .iter()
        .map(|&i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: buffers[i].size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect();
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
    }
    for (&i, output) in writable.iter().zip(&outputs) {
        encoder.copy_buffer_to_buffer(&buffers[i], 0, output, 0, output.size());
    }
    queue.submit(Some(encoder.finish()));
    outputs
        .iter()
        .map(|output| {
            let (tx, rx) = std::sync::mpsc::channel();
            output
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let result =
                bytemuck::cast_slice::<u8, f32>(&output.slice(..).get_mapped_range()).to_vec();
            output.unmap();
            result
        })
        .collect()
}
fn bytes<T: bytemuck::Pod>(values: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn volume_repair_samples_the_faces_where_deltas_were_written() {
    let n = 12u32;
    let dx = 0.01f32;
    let p = [0.064f32, 0.072, 0.083, 1.];
    // delta_x[i,j,k] lives at (i+1,j+.5,k+.5)*dx, and cyclically for Y/Z.
    // A linear field exposes offsets in every direction, not just uniform flow.
    let field = |axis: usize| {
        (0..n * n * n)
            .map(|i| {
                let mut q = [
                    (i % n) as f32 + 0.5,
                    ((i / n) % n) as f32 + 0.5,
                    (i / (n * n)) as f32 + 0.5,
                ];
                q[axis] += 0.5;
                (q[0] + 2. * q[1] + 3. * q[2]) * 1e-6
            })
            .collect::<Vec<_>>()
    };
    let result = run(
        include_str!("../../game/src/gpu/shaders/density_correct_3d.wgsl"),
        "correct_positions",
        vec![
            bytes(&[n, n, n, 1, dx.to_bits(), 1f32.to_bits(), 0, 0]),
            bytes(&field(0)),
            bytes(&field(1)),
            bytes(&field(2)),
            bytes(&vec![1u32; (n * n * n) as usize]),
            bytes(&p),
            bytes(&[0.4f32, -0.2, 0.1, 0.]),
            bytes(&[1f32]),
        ],
        &[5, 6],
        1,
    );
    let delta = (p[0] + 2. * p[1] + 3. * p[2]) * 1e-6 / (dx * dx);
    for (axis, &value) in p.iter().take(3).enumerate() {
        assert!(
            (result[0][axis] - value - delta).abs() < 1e-7,
            "face {axis}: {:?}, expected delta {delta}",
            result[0]
        );
    }
    assert_eq!(result[1], [0.4, -0.2, 0.1, 0.]);
}

fn contact(p: [f32; 4], v: [f32; 4], cells: Vec<u32>, open: u32) -> Vec<Vec<f32>> {
    let n = 12u32;
    run(
        include_str!("../../game/src/gpu/shaders/sdf_collision_3d.wgsl"),
        "sdf_collision",
        vec![
            bytes(&[n, n, n, 1, 0.01f32.to_bits(), 0.001f32.to_bits(), open, 0]),
            bytes(&p),
            bytes(&v),
            bytes(&vec![0.1f32; (n * n * n) as usize]),
            bytes(&vec![0f32; (n * n) as usize]),
            bytes(&[1f32]),
            bytes(&cells),
        ],
        &[1, 2],
        1,
    )
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn packed_bed_contact_slides_without_cell_snaps_or_tangential_damping() {
    let cells = (0..12 * 12 * 12)
        .map(|i| if (i / 12) % 12 <= 5 { 2 } else { 0 })
        .collect();
    let result = contact([0.055, 0.0599, 0.055, 1.], [0.4, -0.2, 0.1, 0.], cells, 63);
    assert!(
        (result[0][0] - 0.0554).abs() < 1e-7 && (result[0][2] - 0.0551).abs() < 1e-7,
        "contact snapped along the bed: {:?}",
        result[0]
    );
    assert!(
        result[0][1] >= 0.06 && result[0][1] < 0.0601,
        "contact must stop on the bed face: {:?}",
        result[0]
    );
    assert!(
        (result[1][0] - 0.4).abs() < 1e-7
            && (result[1][2] - 0.1).abs() < 1e-7
            && result[1][1].abs() < 1e-7,
        "contact consumed tangential momentum: {:?}",
        result[1]
    );
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn every_open_face_allows_water_to_leave_without_a_false_sdf_contact() {
    for axis in 0..3 {
        for side in [-1f32, 1.] {
            let mut p = [0.06, 0.06, 0.06, 1.];
            let mut v = [0f32; 4];
            p[axis] = if side < 0. { 0.0001 } else { 0.1199 };
            v[axis] = side * 0.4;
            let result = contact(p, v, vec![0; 12 * 12 * 12], 63);
            assert!(
                (result[0][axis] - p[axis] - v[axis] * 0.001).abs() < 1e-7
                    && (result[1][axis] - v[axis]).abs() < 1e-7,
                "open face {axis}/{side}: {result:?}"
            );
        }
    }
}
