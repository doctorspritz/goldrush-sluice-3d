//! Real GPU controls for the conservative phase operator's signed sums.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn signed_wide_sums_retain_carries_cancellation_and_particle_order() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    let input: Vec<[f32; 4]> = (0..4096)
        .map(|i| {
            [
                4.25,
                -4.25,
                match i % 4 {
                    0 => 1_000_000.,
                    1 => 0.0001,
                    2 => -1_000_000.,
                    _ => -0.00003,
                },
                if i % 2 == 0 { 0.00000001 } else { -0.000000003 },
            ]
        })
        .collect();
    let expected: Vec<i64> = (0..4)
        .map(|axis| {
            input
                .iter()
                .map(|v| (v[axis] * 68719476736.).trunc() as i64)
                .sum()
        })
        .collect();
    let source = format!(
        "{}\n{}",
        include_str!("../../game/src/gpu/shaders/wide_sum.wgsl"),
        r#"
@group(0) @binding(0) var<storage,read> values:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> wide_accumulators:array<WideSum>;
@group(0) @binding(2) var<storage,read_write> decoded:array<f32>;
@compute @workgroup_size(128) fn add(@builtin(global_invocation_id) id:vec3<u32>) {
 let n=arrayLength(&values);if id.x>=n {return;}
 let forward=values[id.x];let reverse=values[n-1u-id.x];
 for(var c=0u;c<4u;c++) {wide_add(c,forward[c]);wide_add(c+4u,reverse[c]);}
}
@compute @workgroup_size(8) fn decode(@builtin(global_invocation_id) id:vec3<u32>) {
 if id.x<8u {decoded[id.x]=wide_read(id.x);}
}"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("signed conservative sums"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sums = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let decoded = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1, 2].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding == 0,
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
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sums.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: decoded.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = |entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let add = pipeline("add");
    let decode = pipeline("decode");
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 96,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&sums, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(&add);
            pass.dispatch_workgroups(32, 1, 1);
            pass.set_pipeline(&decode);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&sums, 0, &read, 0, 64);
        encoder.copy_buffer_to_buffer(&decoded, 0, &read, 64, 32);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        {
            let bytes = read.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&bytes[..64]);
            let floats: &[f32] = bytemuck::cast_slice(&bytes[64..]);
            for i in 0..8 {
                let signed = (((words[i * 2 + 1] as u64) << 32) | words[i * 2] as u64) as i64;
                assert_eq!(signed, expected[i % 4], "signed sum {i}");
                let exact = signed as f64 / 68719476736.;
                assert!((floats[i] as f64 - exact).abs() < 1e-6 * exact.abs().max(1.));
            }
        }
        read.unmap();
    }
}
