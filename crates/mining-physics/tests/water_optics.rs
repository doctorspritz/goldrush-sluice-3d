//! Run the production rendering helpers on a real GPU. These check integrated
//! volume and optical continuity, independently of the particle trajectories.
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap()
}

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let values = bytemuck::cast_slice(&buffer.slice(..).get_mapped_range()).to_vec();
    buffer.unmap();
    values
}

fn execute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: String,
    groups: &[&wgpu::BindGroup],
    layouts: &[&wgpu::BindGroupLayout],
    output: &wgpu::Buffer,
    dispatch: [u32; 2],
) -> Vec<f32> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production water optics"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        for (i, group) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *group, &[]);
        }
        pass.dispatch_workgroups(dispatch[0], dispatch[1], 1);
    }
    encoder.copy_buffer_to_buffer(output, 0, &result, 0, output.size());
    queue.submit(Some(encoder.finish()));
    read(device, &result)
}

fn output(
    device: &wgpu::Device,
    count: u64,
) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: count * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, layout, group)
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn softened_columns_preserve_volume_and_add_across_clipping_planes() {
    let (device, queue) = device();
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[],
    });
    let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &empty,
        entries: &[],
    });
    let (buffer, layout, group) = output(&device, 256 * 256);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_depth.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let q=(vec2<f32>(p.xy)+0.5)/256.*2.-1.;let s=1.-dot(q,q);
 if s<=0. {result[p.y*256u+p.x]=vec4(0.);return;}
 let h=sqrt(s);let a=1./(0.007*0.007);
 result[p.y*256u+p.x]=vec4(column_integral(s,a,-h,h),column_integral(s,a,-h,0.),column_integral(s,a,0.,h),column_integral(s,a,0.,0.));
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&empty_group, &group],
        &[&empty, &layout],
        &buffer,
        [32, 32],
    );
    let volume: f64 = values
        .chunks_exact(4)
        .map(|v| {
            assert!(v.iter().all(|x| x.is_finite() && *x >= 0.));
            assert!((v[0] - v[1] - v[2]).abs() < 1e-7);
            assert_eq!(v[3], 0.);
            v[0] as f64
        })
        .sum::<f64>()
        * (2. * 0.013 / 256.)
        * (2. * 0.019 / 256.);
    let expected = 4. / 3. * std::f64::consts::PI * 0.013 * 0.007 * 0.019;
    assert!(
        (volume / expected - 1.).abs() < 0.0001,
        "{volume} versus {expected}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn overlapping_columns_keep_colour_and_normals_when_the_front_particle_changes() {
    let (device, queue) = device();
    let texture = |format, bytes: &[f32], channels: u32| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(bytes),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8 * 4 * channels),
                rows_per_image: None,
            },
            texture.size(),
        );
        texture.create_view(&Default::default())
    };
    // Every other nearest particle changes from clear to extremely muddy.
    // The column underneath has the same 4 cm water / extinction 50 per metre.
    let raw = texture(
        wgpu::TextureFormat::Rgba32Float,
        &(0..64)
            .flat_map(|i| [1., if i % 2 == 0 { -1. } else { -10001. }, 0., 0.])
            .collect::<Vec<_>>(),
        4,
    );
    let moments = texture(wgpu::TextureFormat::Rg32Float, &[0.04f32, 2.].repeat(64), 2);
    let nearest_normals = texture(
        wgpu::TextureFormat::Rgba32Float,
        &(0..64)
            .flat_map(|i| {
                if i % 2 == 0 {
                    [1., 0., 0., 0.1]
                } else {
                    [0., 0., 1., 0.9]
                }
            })
            .collect::<Vec<_>>(),
        4,
    );
    let normal_moments = texture(
        wgpu::TextureFormat::Rgba32Float,
        &[0., 0.008, 0., 0.01].repeat(64),
        4,
    );
    let mut uniforms = [0f32; 72];
    uniforms[64] = 8.;
    uniforms[65] = 8.;
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[3, 4, 5, 6, 9].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 3 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            },
            count: None,
        }),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&raw),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&moments),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&nearest_normals),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&normal_moments),
            },
        ],
    });
    let (buffer, out_layout, out_group) = output(&device, 64);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let column=optical_column(vec2<f32>(p.xy)+0.5,1.,true);
 let n=reconstructed_normal(vec2<f32>(p.xy)+0.5,1.);
 result[p.y*8u+p.x]=vec4(column.x,n.y,column.y/max(column.x,0.000001),n.w);
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    for value in values.chunks_exact(4) {
        assert!((value[0] - 0.04).abs() < 1e-6);
        assert!((value[1] - 1.).abs() < 1e-6);
        assert!((value[2] - 50.).abs() < 0.001, "colour jumped: {value:?}");
        assert!(
            (value[3] - 0.8).abs() < 1e-6,
            "normal confidence jumped: {value:?}"
        );
    }
}
