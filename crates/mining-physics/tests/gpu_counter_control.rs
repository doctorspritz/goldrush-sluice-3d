//! Direct counter control to distinguish backend support from marker placement.
use game::gpu::timing::FEATURES;
#[test]
#[ignore = "requires real GPU counters"]
fn direct_compute_pass_has_measurable_gpu_counter_span() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    println!(
        "adapter {:?}, features {:?}",
        adapter.get_info(),
        adapter.features() & (FEATURES | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    );
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: FEATURES,
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    println!("timestamp period {} ns", queue.get_timestamp_period());
    let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> data:array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) p:vec3<u32>){data[p.x]+=1u;}".into())});
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let data = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1048576,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: data.as_entire_binding(),
        }],
    });
    let query = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &query,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(4096, 1, 1);
    }
    encoder.resolve_query_set(&query, 0..2, &result, 0);
    encoder.copy_buffer_to_buffer(&result, 0, &read, 0, 16);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..16)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = read.slice(..16).get_mapped_range();
    let ticks: &[u64] = bytemuck::cast_slice(&mapped);
    println!("direct counter ticks {ticks:?}");
    assert!(ticks[1] > ticks[0]);
}
