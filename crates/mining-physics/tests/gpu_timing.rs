//! Real-device validation for the opt-in batched timestamp collector.
use game::gpu::timing::{GpuStageTimer, FEATURES};
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a real timestamp-capable GPU; run explicitly with --ignored"]
fn batched_gpu_timestamps_cover_ordered_work_and_reuse_the_readback() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    assert!(
        adapter.features().contains(FEATURES),
        "timestamp features unavailable on {:?}",
        adapter.get_info()
    );
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: FEATURES,
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    // Use substantial ordered work and require a positive span; a broken
    // marker path must not pass by returning zero durations.
    const WORDS: usize = 64 * 4096;
    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vec![0u32; WORDS]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> data:array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) p:vec3<u32>){ data[p.x] += 1u; }".into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: data.as_entire_binding(),
        }],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: WORDS as u64 * 4,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    let mut timer = GpuStageTimer::new(&device, &queue);
    for labels in [vec!["first", "second"], vec!["reused"], vec!["many"; 48]] {
        let mut encoder = device.create_command_encoder(&Default::default());
        for &label in &labels {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(timer.pass(label)),
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((WORDS / 64) as u32, 1, 1);
            }
        }
        timer.resolve(&mut encoder);
        encoder.copy_buffer_to_buffer(&data, 0, &readback, 0, WORDS as u64 * 4);
        queue.submit(Some(encoder.finish()));
        timer.begin_map();
        let report = timer.collect(&device).unwrap();
        assert_eq!(report.iter().map(|r| r.0).collect::<Vec<_>>(), labels);
        assert!(report.iter().all(|r| r.1.is_finite() && r.1 >= 0.));
        assert!(
            report.iter().map(|r| r.1).sum::<f64>() > 0.,
            "no measurable GPU interval: {report:?}"
        );
    }
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    assert!(
        bytemuck::cast_slice::<u8, u32>(&readback.slice(..).get_mapped_range())
            .iter()
            .all(|&v| v == 51)
    );
    readback.unmap();
}
