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
        .as_chunks::<4>()
        .0
        .iter()
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
fn overlapping_interface_gradient_matches_independent_density_derivative() {
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
    let (buffer, layout, group) = output(&device, 64);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_depth.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let x=(f32(p.y*8u+p.x)-31.5)*0.0003;
 let point=vec3(x,0.006,0.);
 let ax=vec3(0.016,0.,0.);let ay=vec3(0.,0.012,0.);let az=vec3(0.,0.,0.02);
 let a=density_gradient(point,vec3(-0.006,0.,0.),ax,ay,az,0.2);
 let b=density_gradient(point,vec3(0.006,0.,0.),ax,ay,az,0.2);
 result[p.y*8u+p.x]=vec4(a+b,0.);
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&empty_group, &group],
        &[&empty, &layout],
        &buffer,
        [1, 1],
    );
    // Differentiate the scalar kernel independently in f64, including the
    // ellipsoid metric and both overlapping volumes. No nearest normal enters.
    let density = |p: [f64; 3]| {
        [-0.006, 0.006]
            .into_iter()
            .map(|cx| {
                let r2 =
                    ((p[0] - cx) / 0.016).powi(2) + (p[1] / 0.012).powi(2) + (p[2] / 0.02).powi(2);
                6.5625 * 0.2 * (1. - r2).max(0.).powi(3)
            })
            .sum::<f64>()
    };
    let mut previous: Option<[f64; 3]> = None;
    for (i, gradient) in values.as_chunks::<4>().0.iter().enumerate() {
        let p = [(i as f64 - 31.5) * 0.0003, 0.006, 0.];
        let mut direction = [0.; 3];
        for axis in 0..3 {
            let (mut a, mut b) = (p, p);
            a[axis] -= 1e-7;
            b[axis] += 1e-7;
            let derivative = (density(a) - density(b)) / 2e-7;
            assert!((gradient[axis] as f64 - derivative).abs() < 0.001);
            direction[axis] = gradient[axis] as f64;
        }
        let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        direction.iter_mut().for_each(|v| *v /= length);
        assert!(direction[1] > 0.8, "joined crest normal: {direction:?}");
        if let Some(previous) = previous {
            let dot = direction
                .iter()
                .zip(previous)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            assert!(dot > 0.998, "normal discontinuity: {dot}");
        }
        previous = Some(direction);
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn shared_interface_uses_relative_depth_moments_without_moving_spray_or_hidden_water() {
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
    let (buffer, layout, group) = output(&device, 64);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_blur.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let scale=f32(p.y*8u+p.x+1u);
 // A shared front at 3 m, with equal weighted offsets of 2 and 6 mm.
 let moments=(vec2(0.002*0.01,0.01)+vec2(0.006*0.01,0.01))*scale;
 result[p.y*8u+p.x]=vec4(
  shared_surface_depth(vec2(3.,-1.),moments),
  shared_surface_depth(vec2(3.,1.),moments),
  shared_surface_depth(vec2(0.,-1.),moments),
  shared_surface_depth(vec2(3.,-1.),vec2(0.)));
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&empty_group, &group],
        &[&empty, &layout],
        &buffer,
        [1, 1],
    );
    for value in values.as_chunks::<4>().0 {
        assert!((value[0] - 3.004).abs() < 1e-6, "{value:?}");
        assert_eq!(value[1..], [3., 0., 3.]);
    }
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
            .flat_map(|i| {
                [
                    if i == 18 { 2. } else { 1. },
                    if i == 45 {
                        1.
                    } else if i % 2 == 0 {
                        -1.
                    } else {
                        -10001.
                    },
                    0.,
                    0.,
                ]
            })
            .collect::<Vec<_>>(),
        4,
    );
    // A different depth surface and an outlet pixel carry deliberately huge
    // extinction. Neither may contaminate the visible channel's column filter.
    let moments = texture(
        wgpu::TextureFormat::Rg32Float,
        &(0..64)
            .flat_map(|i| {
                if i == 18 || i == 45 {
                    [0.5, 5000.]
                } else {
                    [0.04, 2.]
                }
            })
            .collect::<Vec<_>>(),
        2,
    );
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
    uniforms[37] = 1.;
    // Exercise all four full-resolution pixel phases of the half-size surface
    // buffer, rather than checking only the particle-buffer texel centres.
    uniforms[64] = 16.;
    uniforms[65] = 16.;
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
    let (buffer, out_layout, out_group) = output(&device, 256);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let column=optical_column(vec2<f32>(p.xy)+0.5,1.,true);
 let n=reconstructed_normal(vec2<f32>(p.xy)+0.5,1.);
 result[p.y*16u+p.x]=vec4(column.x,n.y,column.y/max(column.x,0.000001),n.w);
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [2, 2],
    );
    for value in values.as_chunks::<4>().0 {
        assert!((value[0] - 0.04).abs() < 1e-6);
        assert!((value[1] - 1.).abs() < 1e-6);
        assert!((value[2] - 50.).abs() < 0.001, "colour jumped: {value:?}");
        assert!(
            (value[3] - 0.8).abs() < 1e-6,
            "normal confidence jumped: {value:?}"
        );
    }
}
