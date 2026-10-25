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
 result[p.y*8u+p.x]=vec4(a+b,kernel_density(point,vec3(-0.006,0.,0.),ax,ay,az,0.2)+kernel_density(point,vec3(0.006,0.,0.),ax,ay,az,0.2));
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
        assert!((gradient[3] as f64 - density(p)).abs() < 1e-6);
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
fn shared_density_interface_crossings_are_bounded_without_moving_spray_or_empty_water() {
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
 // An independent linear ramp reaches half density at 19 mm. The old
 // two-probe 12 mm interval cannot bracket this connected interface.
 var profile:array<f32,8>;
 for(var i=0u;i<8u;i++) {profile[i]=max(0.5+(PROFILE_DEPTHS[i]-0.019)*50.,0.);}
 var near=vec4(profile[0],profile[1],profile[2],profile[3]);
 var far=vec4(profile[4],profile[5],profile[6],profile[7]);
 // Independent near/deep ramps, two layers (choose the first), vacuum,
 // an already submerged front, and a crossing at the last interval.
 switch p.x {
  case 0u: {
   for(var i=0u;i<8u;i++) {profile[i]=max(0.5+(PROFILE_DEPTHS[i]-0.006)*80.,0.);}
   near=vec4(profile[0],profile[1],profile[2],profile[3]);
   far=vec4(profile[4],profile[5],profile[6],profile[7]);
  }
  case 2u: {near=vec4(0.,0.1,0.6,0.);}
  case 3u: {near=vec4(0.);far=vec4(0.);}
  case 4u: {near=vec4(0.8);}
  case 5u: {near=vec4(0.);far=vec4(0.,0.,0.,1.);}
  default: {}
 }
 result[p.y*8u+p.x]=vec4(
  shared_surface_depth(vec2(3.,-1.),near,far),
  shared_surface_depth(vec2(3.,1.),near,far),
  shared_surface_depth(vec2(0.,-1.),near,far),
  shared_surface_depth(vec2(3.,-1.),vec4(0.),vec4(0.)));
 // Test both sides of the sparse/crossing boundary at 0.5 independently.
 let rho=0.47+f32(p.y*8u+p.x)*0.001;
 let weak=shared_surface_depth(vec2(3.,-1.),vec4(0.),vec4(0.,0.,0.,rho));
 result[p.y*8u+p.x].w=weak;
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
    let mut previous: Option<f32> = None;
    let expected = [3.006, 3.019, 3.0044, 3., 3., 3.043, 3.019, 3.019];
    for (i, value) in values.as_chunks::<4>().0.iter().enumerate() {
        match i % 8 {
            2 => assert!(
                (3.002..=3.005).contains(&value[0]),
                "first layer: {value:?}"
            ),
            5 => assert!(
                (3.035..=3.051).contains(&value[0]),
                "last bracket: {value:?}"
            ),
            _ => assert!(
                (value[0] - expected[i % 8]).abs() < 1e-6,
                "affine crossing: {value:?}"
            ),
        }
        assert_eq!(value[1..3], [3., 0.]);
        assert!((3. ..=3.051001).contains(&value[3]));
        if let Some(last) = previous {
            assert!(
                (value[3] - last).abs() < 0.0005,
                "sparse interface jumped: {last} -> {}",
                value[3]
            );
        }
        previous = Some(value[3]);
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn overlapping_columns_keep_colour_and_normals_when_the_front_particle_changes() {
    let (device, queue) = device();
    let mut textures = Vec::new();
    let mut texture = |format, bytes: &[f32], channels: u32| {
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
        let view = texture.create_view(&Default::default());
        textures.push(texture);
        view
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
    let joined_depth = texture(
        wgpu::TextureFormat::Rgba32Float,
        &[1., 0.04, 2., 2.].repeat(64),
        4,
    );
    let mut uniforms = [0f32; 72];
    uniforms[16] = 1.;
    uniforms[22] = -1.;
    uniforms[25] = 1.;
    uniforms[31] = 1.;
    uniforms[32] = 0.1;
    uniforms[37] = 0.1;
    uniforms[42] = -1.;
    uniforms[47] = 1.;
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
        entries: &[1, 3, 4, 5, 6, 9].map(|binding| wgpu::BindGroupLayoutEntry {
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
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&joined_depth),
            },
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
        source.clone(),
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
    // A subpixel camera shift must not expose alternating source texels as
    // bright crest beads or opaque/clear mud islands. The low-frequency field
    // is constant; independently, its area-weighted extinction is 50 / metre.
    // Exercise both full-resolution phases and both half-resolution phases.
    let raw = [1f32, -1., 0., 0.].repeat(64);
    let moments: Vec<f32> = (0..64)
        .flat_map(|i| [0.04, 0.04 * if i % 2 == 0 { 10. } else { 90. }])
        .collect();
    let normals: Vec<f32> = (0..64)
        .flat_map(|i| [if i % 2 == 0 { 0.006 } else { -0.006 }, 0.008, 0., 0.01])
        .collect();
    for (index, data) in [(0, raw), (1, moments), (3, normals)] {
        queue.write_texture(
            textures[index].as_image_copy(),
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8 * textures[index].format().block_copy_size(None).unwrap()),
                rows_per_image: None,
            },
            textures[index].size(),
        );
    }
    let filtered_moments: Vec<f32> = (0..64)
        .flat_map(|i| [1., 0.04, 0.04 * if i % 2 == 0 { 10. } else { 90. }, 2.])
        .collect();
    queue.write_texture(
        textures[4].as_image_copy(),
        bytemuck::cast_slice(&filtered_moments),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(128),
            rows_per_image: None,
        },
        textures[4].size(),
    );
    let filtered = execute(
        &device,
        &queue,
        source.clone(),
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [2, 2],
    );
    for y in 6..10 {
        for x in 6..10 {
            let value = &filtered[(y * 16 + x) * 4..][..4];
            assert!(
                (value[2] - 50.).abs() < 1.5,
                "texel-phase mud island: {value:?}"
            );
            assert!(value[1] > 0.995, "texel-phase crest highlight: {value:?}");
        }
    }
    // Cross the rounded gather-centre boundary by 0.0002 screen pixels.
    // An entering edge tap used to retain substantial Gaussian weight and
    // clamped border texels were counted twice, switching a whole mud patch.
    let edge_moments: Vec<f32> = (0..64)
        .flat_map(|i| [0.04, 0.04 * if i % 8 == 7 { 500. } else { 10. }])
        .collect();
    queue.write_texture(
        textures[1].as_image_copy(),
        bytemuck::cast_slice(&edge_moments),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(64),
            rows_per_image: None,
        },
        textures[1].size(),
    );
    let packed_edge: Vec<f32> = edge_moments
        .chunks_exact(2)
        .flat_map(|m| [1., m[0], m[1], 2.])
        .collect();
    queue.write_texture(
        textures[4].as_image_copy(),
        bytemuck::cast_slice(&packed_edge),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(128),
            rows_per_image: None,
        },
        textures[4].size(),
    );
    let boundary_source = source.replace(
        "let column=optical_column(vec2<f32>(p.xy)+0.5,1.,true);",
        "let column=optical_column(vec2(7.9999+f32(p.x)*0.0002,8.),1.,true);",
    );
    let boundary = execute(
        &device,
        &queue,
        boundary_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [2, 2],
    );
    assert!(
        (boundary[2] - boundary[6]).abs() < 0.01,
        "gather edge switched a sediment patch: {} -> {}",
        boundary[2],
        boundary[6]
    );

    // Intersect the test camera rays with a known tilted geometric plane.
    // Deliberately biased kernel gradients must not bend a known geometric
    // plane back toward individual particle domes.
    let plane_depth: Vec<f32> = (0..64)
        .flat_map(|i| {
            let x = ((i % 8) as f32 + 0.5) / 8. * 2. - 1.;
            let y = 1. - ((i / 8) as f32 + 0.5) / 8. * 2.;
            [1. / (1. - 0.2 * 0.1 * x + 0.1 * 0.1 * y), 0.04, 2., 2.]
        })
        .collect();
    queue.write_texture(
        textures[4].as_image_copy(),
        bytemuck::cast_slice(&plane_depth),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(128),
            rows_per_image: None,
        },
        textures[4].size(),
    );
    let plane_source = source.replace(
        "vec4(column.x,n.y,column.y/max(column.x,0.000001),n.w)",
        "vec4(n.xyz,n.w)",
    );
    let expected = glam::Vec3::new(0.2, 1., 0.1).normalize();
    // A biased gradient and a missing front gradient must both retain the
    // independently known plane. The latter reproduces a delayed bulk crest.
    for gradient in [[0.006f32, 0.008, 0., 0.01], [0.; 4]] {
        queue.write_texture(
            textures[3].as_image_copy(),
            bytemuck::cast_slice(&gradient.repeat(64)),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: None,
            },
            textures[3].size(),
        );
        let plane = execute(
            &device,
            &queue,
            plane_source.clone(),
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [2, 2],
        );
        for y in 6..10 {
            for x in 6..10 {
                let n = glam::Vec3::from_slice(&plane[(y * 16 + x) * 4..][..3]);
                assert!(
                    n.is_finite() && n.dot(expected) > 0.99999,
                    "shared wave slope lost: {n:?} vs {expected:?}"
                );
            }
        }
    }
    // A moving row must leave a robust fit continuously. The previous hard
    // 35 mm residual cutoff dropped the complete row at once, changing its
    // highlight despite only 4 micrometres of geometric motion.
    let continuity_source = source
        .replace(
            "let n=reconstructed_normal(vec2<f32>(p.xy)+0.5,1.);",
            "let n=reconstructed_normal(vec2(8.,8.),1.);",
        )
        .replace(
            "vec4(column.x,n.y,column.y/max(column.x,0.000001),n.w)",
            "vec4(n.xyz,n.w)",
        );
    let mut normals_at_cutoff = Vec::new();
    for shift in [-0.000002f32, 0.000002] {
        let surface: Vec<f32> = (0..64)
            .flat_map(|i| [if i % 8 == 5 { 1.035 + shift } else { 1. }, 0.04, 2., 2.])
            .collect();
        queue.write_texture(
            textures[4].as_image_copy(),
            bytemuck::cast_slice(&surface),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: None,
            },
            textures[4].size(),
        );
        let values = execute(
            &device,
            &queue,
            continuity_source.clone(),
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [2, 2],
        );
        normals_at_cutoff.push(glam::Vec3::from_slice(&values[..3]));
    }
    println!(
        "normal-fit boundary vector change: {}",
        (normals_at_cutoff[0] - normals_at_cutoff[1]).length()
    );
    assert!(
        (normals_at_cutoff[0] - normals_at_cutoff[1]).length() < 0.0001,
        "normal-fit cutoff popped: {normals_at_cutoff:?}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn connected_surface_filter_suppresses_domes_without_popping_or_changing_outlet() {
    let (device, queue) = device();
    let mut textures = Vec::new();
    for (slot, format, channels) in [
        (0, wgpu::TextureFormat::Rgba32Float, 4),
        (2, wgpu::TextureFormat::Rgba32Float, 4),
        (3, wgpu::TextureFormat::Rgba32Float, 4),
        (4, wgpu::TextureFormat::Rgba32Float, 4),
    ] {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 32,
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
        let values: Vec<f32> = (0..256)
            .flat_map(|i| {
                let (x, y) = (i % 32, i / 32);
                if slot == 0 {
                    let depth = match y {
                        4 => 1.125,
                        5 => 1. / (1. + (x as f32 - 16.) * 0.01),
                        6 if x == 16 => 0.,
                        3 | 7 => {
                            1. / (1. + (x as f32 - 16.) * 0.015) + if x == 16 { 0.02 } else { 0. }
                        }
                        _ if x == 16 => 1.02,
                        _ => 1.,
                    };
                    let path = 0.02 + (x as f32) * 0.001;
                    let beta = if y == 4 {
                        if x % 2 == 0 {
                            10.
                        } else {
                            90.
                        }
                    } else {
                        50.
                    };
                    if (y == 1 || y == 2) && x == 18 {
                        // A neighbouring detached outlet column must not lend
                        // its optical material to channel reconstruction.
                        vec![depth, 0.5, 5000., 0.]
                    } else {
                        vec![depth, path, path * beta, if y == 0 { 0. } else { 2. }]
                    }
                } else {
                    let density = [0., 0.0499, 0.0501, 0.5, 0.5, 0.5, 0.5, 0.25][y];
                    if slot == 2 {
                        vec![0.04, 2., density, density]
                    } else if y == 7 {
                        // Bulk surface selected at 35 mm, after both old probes.
                        if slot == 4 {
                            vec![0., 0., 0.5, 0.]
                        } else {
                            vec![0.; 4]
                        }
                    } else {
                        vec![density; 4]
                    }
                }
            })
            .collect();
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&values),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 4 * channels),
                rows_per_image: None,
            },
            texture.size(),
        );
        textures.push(texture);
    }
    let views: Vec<_> = textures
        .iter()
        .map(|t| t.create_view(&Default::default()))
        .collect();
    let mut uniforms = [0f32; 72];
    uniforms[64] = 32.;
    uniforms[65] = 8.;
    uniforms[66] = 0.045;
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1, 2, 3, 4].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if binding == 1 {
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
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&views[3]),
            },
        ],
    });
    let (buffer, out_layout, out_group) = output(&device, 64);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_blur.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 result[p.y*8u+p.x]=smooth_surface(vec2(16,i32(p.y)),vec2(1,0));
}"#
    );
    let values = execute(
        &device,
        &queue,
        source.clone(),
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    let row = |y: usize| values[y * 8 * 4] as f64;
    // Independent original 15-tap reference: outlet/spray must keep its old
    // response, while bulk water suppresses more of a solitary kernel dome.
    let mut sum = 0.;
    let mut weight = 0.;
    for i in -7i32..=7 {
        let depth = if i == 0 { 1.02f32 as f64 } else { 1. };
        let w =
            (-(i * i) as f64 / 24. - ((depth - 1.02f32 as f64) / (0.045f32 as f64)).powi(2)).exp();
        sum += depth * w;
        weight += w;
    }
    assert!(
        (row(0) - sum / weight).abs() < 1e-6,
        "outlet filter changed"
    );
    assert!(
        row(3) > 1. && row(3) < row(0),
        "connected dome was not reduced: connected={}, flat-outlet={}",
        row(3),
        row(0)
    );
    // Independent old bilateral reference on a steep projective plane with
    // a small kernel cap. Range weighting around centre depth retains the cap.
    let mut old_sum = 0.;
    let mut old_weight = 0.;
    for i in -11i32..=11 {
        let d = 1. / (1. + i as f64 * 0.015) + if i == 0 { 0.02 } else { 0. };
        let w = (-(i * i) as f64 / 48. - ((d - 1.02) / 0.045).powi(2)).exp();
        old_sum += d * w;
        old_weight += w;
    }
    assert!(
        (row(3) - 1.).abs() < 0.8 * (old_sum / old_weight - 1.).abs(),
        "oblique-plane range weighting retained the crest cap"
    );
    assert!((row(2) - row(1)).abs() < 1e-6, "density transition popped");
    assert!((row(4) - 1.125).abs() < 1e-6, "uniform film moved");
    assert!((row(5) - 1.).abs() < 1e-6, "planar wave slope drifted");
    assert_eq!(row(6), 0., "dry coverage was invented");
    for y in [1usize, 2, 3, 5, 7] {
        let v = &values[y * 8 * 4..][..4];
        assert!(
            (v[2] / v[1] - 50.).abs() < 0.0001,
            "homogeneous mixture changed: {v:?}"
        );
    }
    let v = &values[4 * 8 * 4..][..4];
    assert!(
        (v[2] / v[1] - 50.).abs() < 1.,
        "opacity was not filtered with the interface: {v:?}"
    );
    assert_eq!(
        &values[6 * 8 * 4..][..4],
        &[0.; 4],
        "dry pixel acquired optical material"
    );
    assert!(
        (row(7) - row(3)).abs() < 1e-6,
        "delayed bulk crest was filtered as a detached drop"
    );

    // A resolved narrow crest is curvature, not kernel noise. The complete
    // geometry filter must reproduce it rather than widening each shoulder.
    let crest: Vec<f32> = (0..256)
        .flat_map(|i| {
            let x = (i % 32) as f32 - 16.;
            [1. / (1. - 0.0005 * x * x), 0.04, 2., 2.]
        })
        .collect();
    queue.write_texture(
        textures[0].as_image_copy(),
        bytemuck::cast_slice(&crest),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        textures[0].size(),
    );
    let crest_source = source.replace(
        "smooth_surface(vec2(16,i32(p.y)),vec2(1,0))",
        "geometry_surface(vec2(16,4),vec2(1,0))",
    );
    let resolved = execute(
        &device,
        &queue,
        crest_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    println!("resolved crest depth error={}", (resolved[0] - 1.).abs());
    assert!(
        (resolved[0] - 1.).abs() < 0.000002,
        "resolved crest broadened: {}",
        resolved[0]
    );

    // A non-quadratic travelling ridge is resolved geometry too. The broad
    // shoulder model can reproduce a parabola while flattening this wave.
    // Judge the production geometry pass, before the compositor sees it.
    for axis in 0..2 {
        for phase in [0f32, 0.25, 0.5, 0.75] {
            let ridge: Vec<f32> = (0..256)
                .flat_map(|i| {
                    let x = if axis == 0 {
                        (i % 32) as f32 - 16.
                    } else {
                        (i / 32) as f32 - 4.
                    } - phase;
                    let inverse = 1. + 0.02 * (x * std::f32::consts::TAU / 12.).cos();
                    [1. / inverse, 0.04, 10., 2.]
                })
                .collect();
            queue.write_texture(
                textures[0].as_image_copy(),
                bytemuck::cast_slice(&ridge),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * 16),
                    rows_per_image: None,
                },
                textures[0].size(),
            );
            let wave = execute(
                &device,
                &queue,
                source.replace(
                    "smooth_surface(vec2(16,i32(p.y)),vec2(1,0))",
                    if axis == 0 {
                        "geometry_surface(vec2(16,4),vec2(1,0))"
                    } else {
                        "geometry_surface(vec2(16,4),vec2(0,1))"
                    },
                ),
                &[&group, &out_group],
                &[&layout, &out_layout],
                &buffer,
                [1, 1],
            );
            let expected = 1. / (1. + 0.02 * (phase * std::f32::consts::TAU / 12.).cos());
            let error = (wave[0] - expected).abs();
            println!("resolved travelling ridge axis={axis} phase={phase} geometry error={error}m");
            assert!(
                error < 0.00002,
                "resolved travelling ridge rounded: axis={axis}, phase={phase}, error={error}m"
            );
            assert!(
                (wave[2] / wave[1] - 250.).abs() < 0.001,
                "ridge acquired a muddy patch"
            );
        }
    }
    queue.write_texture(
        textures[0].as_image_copy(),
        bytemuck::cast_slice(&crest),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        textures[0].size(),
    );

    let crest_polish = source.replace(
        "smooth_surface(vec2(16,i32(p.y)),vec2(1,0))",
        "optical_surface(vec2(16,4),vec2(1,0))",
    );
    let retained = execute(
        &device,
        &queue,
        crest_polish,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    assert!(
        (retained[0] - 1.).abs() < 0.000002,
        "polish broadened a resolved crest"
    );

    // A resolved ridge must survive, but alternating sub-texel density noise
    // is not a train of real wave crests. Exercise the final production polish
    // independently of the local surface/tangent reconstruction.
    let alias: Vec<f32> = (0..256)
        .flat_map(|i| {
            let inverse = 1. + if i % 2 == 0 { 0.001 } else { -0.001 };
            [1. / inverse, 0.04, 2., 2.]
        })
        .collect();
    queue.write_texture(
        textures[0].as_image_copy(),
        bytemuck::cast_slice(&alias),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        textures[0].size(),
    );
    let alias_source = source.replace(
        "smooth_surface(vec2(16,i32(p.y)),vec2(1,0))",
        "optical_surface(vec2(16,4),vec2(1,0))",
    );
    let polished = execute(
        &device,
        &queue,
        alias_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    println!(
        "sub-texel ripple depth error={} m",
        (polished[0] - 1.).abs()
    );
    assert!(
        (polished[0] - 1.).abs() < 0.000002,
        "sub-texel noise became a visible ripple"
    );

    // Real projection, distant sheet: one half-resolution texel spans 50 mm.
    // Measure the response to an optical impulse independently in metres.
    // A fixed 23-pixel blur spreads that impulse across decimetres of water.
    let impulse: Vec<f32> = (0..256)
        .flat_map(|i| [4., 0.04, if i % 32 == 16 { 2.04 } else { 2. }, 2.])
        .collect();
    queue.write_texture(
        textures[0].as_image_copy(),
        bytemuck::cast_slice(&impulse),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        textures[0].size(),
    );
    for slot in [2, 3] {
        queue.write_texture(
            textures[slot].as_image_copy(),
            bytemuck::cast_slice(&[0.5f32; 1024]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[slot].size(),
        );
    }
    uniforms[32] = 0.2;
    uniforms[37] = 0.2;
    uniforms[42] = -1.;
    uniforms[47] = 1.;
    uniforms[64] = 64.;
    uniforms[65] = 16.;
    queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&uniforms));
    let metric_source = source.replace("vec2(16,i32(p.y))", "vec2(12+i32(p.x),4)");
    let metric = execute(
        &device,
        &queue,
        metric_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    let mut mass = 0f64;
    let mut second = 0f64;
    for x in 0..8 {
        let v = &metric[x * 4..x * 4 + 4];
        assert!(
            (v[0] - 4.).abs() < 0.000001,
            "plane moved under footprint control"
        );
        let response = (v[2] as f64 - 2.).max(0.);
        mass += response;
        second += response * ((x as f64 - 4.) * 0.05).powi(2);
    }
    let spread = (second / mass).sqrt();
    assert!(
        mass > 0.01 && spread < 0.030,
        "connected blur spread across too much water: {spread} m"
    );
    println!("world-space optical impulse spread: {spread} m");
    // A 2.7 mm normal displacement on this grazing plane spans 80 mm of
    // camera depth. It is connected water, not a separate layer. The render
    // footprint contains several samples of the same sheet; its depth and
    // additive mineral column must be reconstructed together.
    uniforms[32] = 0.008;
    uniforms[37] = 0.002;
    queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&uniforms));
    for slot in [2, 3] {
        queue.write_texture(
            textures[slot].as_image_copy(),
            bytemuck::cast_slice(&[0.5f32; 1024]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[slot].size(),
        );
    }
    let grazing: Vec<f32> = (0..256)
        .flat_map(|i| {
            let x = i % 32;
            let z = 1. / (1. + (x as f32 - 16.) * 0.015) + if x == 16 { 0.08 } else { 0. };
            [z, 0.04, 0.04 * if x == 16 { 1000. } else { 50. }, 2.]
        })
        .collect();
    queue.write_texture(
        textures[0].as_image_copy(),
        bytemuck::cast_slice(&grazing),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        textures[0].size(),
    );
    let grazing_values = execute(
        &device,
        &queue,
        source.clone(),
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    let v = &grazing_values[0..4];
    println!(
        "grazing cap depth and extinction: {} m, {} /m",
        v[0],
        v[2] / v[1]
    );
    assert!(
        (v[0] - 1.).abs() < 0.012,
        "grazing millimetre cap was not joined to the sheet: {v:?}"
    );
    // Judge suppression at the final shared optical gather, rather than
    // requiring an intermediate filter to throw the impulse's material away.
    // Run the same six production stages, retaining the unchanged <200/m
    // visible-crest bound alongside the independent integrated-volume checks.
    let (stage_buffer, stage_layout, stage_group) = output(&device, 256);
    let mut staged = grazing.clone();
    for (function, direction) in [
        ("geometry_surface", "vec2(1,0)"),
        ("geometry_surface", "vec2(0,1)"),
        ("smooth_surface", "vec2(1,0)"),
        ("smooth_surface", "vec2(0,1)"),
        ("optical_surface", "vec2(1,0)"),
        ("optical_surface", "vec2(0,1)"),
    ] {
        let shader = include_str!("../../mining/src/water_blur.wgsl");
        let stage_source = format!(
            r#"{shader}

@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {{
 result[p.y*32u+p.x]={function}(vec2<i32>(p.xy),{direction});
}}"#
        );
        staged = execute(
            &device,
            &queue,
            stage_source,
            &[&group, &stage_group],
            &[&layout, &stage_layout],
            &stage_buffer,
            [4, 1],
        );
        queue.write_texture(
            textures[0].as_image_copy(),
            bytemuck::cast_slice(&staged),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[0].size(),
        );
    }
    let raw = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fixed raw grazing envelope"),
        size: textures[0].size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let raw_values: Vec<f32> = grazing
        .chunks_exact(4)
        .flat_map(|v| [v[0], -1., 0., 0.])
        .collect();
    queue.write_texture(
        raw.as_image_copy(),
        bytemuck::cast_slice(&raw_values),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        raw.size(),
    );
    let raw_view = raw.create_view(&Default::default());
    for i in [16, 21, 26, 31] {
        uniforms[i] = 1.;
    }
    queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&uniforms));
    let bindings = [1, 3, 4, 5, 6, 9];
    let gather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &bindings.map(|binding| wgpu::BindGroupLayoutEntry {
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
    let gather_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &gather_layout,
        entries: &bindings.map(|binding| wgpu::BindGroupEntry {
            binding,
            resource: match binding {
                3 => uniform.as_entire_binding(),
                4 => wgpu::BindingResource::TextureView(&raw_view),
                1 => wgpu::BindingResource::TextureView(&views[0]),
                _ => wgpu::BindingResource::TextureView(&views[1]),
            },
        }),
    });
    let gather_source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let pixel=vec2(33.,9.);let depth=sample_depth(pixel).x;
 let column=channel_surface(pixel,depth).column;
 result[p.y*8u+p.x]=vec4(column.x,column.y/max(column.x,0.000001),depth,0.);
}"#
    );
    let final_column = execute(
        &device,
        &queue,
        gather_source,
        &[&gather_group, &out_group],
        &[&gather_layout, &out_layout],
        &buffer,
        [1, 1],
    );
    println!(
        "final grazing optical path={}m, extinction={}/m",
        final_column[0], final_column[1]
    );
    assert!(
        final_column[1] < 200.,
        "final grazing crest retained a mud patch: {:?}",
        &final_column[..4]
    );
    for component in [1, 2] {
        let before: f64 = grazing.chunks_exact(4).map(|v| v[component] as f64).sum();
        let after: f64 = staged.chunks_exact(4).map(|v| v[component] as f64).sum();
        let error = (after / before - 1.).abs();
        println!("staged grazing ray-column component={component} relative sum error={error}");
        assert!(
            error < 0.000002,
            "staged optical exchange changed additive moments: component={component} error={error}"
        );
    }

    // A real 40 cm layer step has flat samples on either side. Its secant is
    // not a water tangent. Both sides must survive without invented depth or
    // transferred suspension; the dry centre must remain empty too.
    for dry in [false, true] {
        let edge: Vec<f32> = (0..256)
            .flat_map(|i| {
                if i % 32 < 16 {
                    [1., 0.04, 2., 2.]
                } else if dry {
                    [0.; 4]
                } else {
                    [1.4, 0.04, 36., 2.]
                }
            })
            .collect();
        queue.write_texture(
            textures[0].as_image_copy(),
            bytemuck::cast_slice(&edge),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[0].size(),
        );
        let edge_source = source.replace("vec2(16,i32(p.y))", "vec2(15+i32(p.x % 2u),4)");
        let values = execute(
            &device,
            &queue,
            edge_source,
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [1, 1],
        );
        assert!(
            (values[0] - 1.).abs() < 0.000001 && (values[2] / values[1] - 50.).abs() < 0.001,
            "front layer borrowed depth or mud: dry={dry}, {:?}",
            &values[..4]
        );
        if dry {
            assert_eq!(&values[4..8], &[0.; 4]);
        } else {
            assert!(
                (values[4] - 1.4).abs() < 0.000001 && (values[6] / values[5] - 900.).abs() < 0.001,
                "back layer borrowed depth or clear water: {:?}",
                &values[4..8]
            );
        }
    }
    // Restore the preceding optical impulse fixture for the dilute transition.
    queue.write_texture(
        textures[0].as_image_copy(),
        bytemuck::cast_slice(&impulse),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        textures[0].size(),
    );
    uniforms[32] = 0.2;
    uniforms[37] = 0.2;
    queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&uniforms));

    // The physical-footprint bound must fade in with connected density; the
    // first nonzero confidence must not switch the whole blur width at once.
    let transition: Vec<f32> = (0..256)
        .flat_map(|i| [if i / 32 < 4 { 0.0499 } else { 0.0501 }; 4])
        .collect();
    for slot in [2, 3] {
        queue.write_texture(
            textures[slot].as_image_copy(),
            bytemuck::cast_slice(&transition),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[slot].size(),
        );
    }
    let transition_source = source.replace("vec2(16,i32(p.y))", "vec2(16,3+i32(p.x % 2u))");
    let transition = execute(
        &device,
        &queue,
        transition_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    for component in 0..4 {
        assert!(
            (transition[component] - transition[4 + component]).abs() < 0.000001,
            "physical blur footprint popped at dilute-film threshold"
        );
    }
    // A finite flat sheet has a constant projected pixel area, so the sum of
    // its ray columns is proportional to optical volume. Filtering must not
    // create water or mineral optical material at a thinning/dry crest edge.
    uniforms[32] = 0.2;
    uniforms[37] = 0.2;
    queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&uniforms));
    for slot in [2, 3] {
        queue.write_texture(
            textures[slot].as_image_copy(),
            bytemuck::cast_slice(&[0.5f32; 1024]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[slot].size(),
        );
    }
    let (mass_buffer, mass_layout, mass_group) = output(&device, 256);
    let mass_source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_blur.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 result[p.y*32u+p.x]=smooth_surface(vec2<i32>(p.xy),vec2(1,0));
}"#
    );
    for phase in [0f32, 0.25, 0.5, 0.75] {
        let sheet: Vec<f32> = (0..256)
            .flat_map(|i| {
                let x = (i % 32) as f32;
                if !(2. ..=20.).contains(&x) {
                    return [0.; 4];
                }
                let impulse = (-((x - 2. - phase) / 0.65).powi(2)).exp();
                let path = 0.002 + 0.06 * impulse;
                [1., path, path * (50. + 300. * impulse), 2.]
            })
            .collect();
        queue.write_texture(
            textures[0].as_image_copy(),
            bytemuck::cast_slice(&sheet),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[0].size(),
        );
        let filtered = execute(
            &device,
            &queue,
            mass_source.clone(),
            &[&group, &mass_group],
            &[&layout, &mass_layout],
            &mass_buffer,
            [4, 1],
        );
        for component in [1, 2] {
            let before: f64 = sheet.chunks_exact(4).map(|v| v[component] as f64).sum();
            let after: f64 = filtered.chunks_exact(4).map(|v| v[component] as f64).sum();
            let error = (after / before - 1.).abs();
            println!("finite-sheet phase={phase} component={component} relative optical-volume error={error}");
            assert!(error<0.000002,"finite edge changed optical material: phase={phase} component={component} before={before} after={after}");
        }
    }

    // Two disconnected films can have the same camera depth but independent
    // turbidity. A dry gap must not exchange their optical material. Include
    // narrow films requiring multiple stencil reflections on both screen axes.
    for axis in 0..2 {
        let ranges = if axis == 0 {
            [(2usize, 6usize), (10, 20)]
        } else {
            [(1, 2), (4, 6)]
        };
        let sheet: Vec<f32> = (0..256)
            .flat_map(|i| {
                let coordinate = if axis == 0 { i % 32 } else { i / 32 };
                match ranges
                    .iter()
                    .position(|&(lo, hi)| (lo..=hi).contains(&coordinate))
                {
                    None => [0.; 4],
                    Some(island) => {
                        let path = 0.002 + 0.003 * (coordinate - ranges[island].0) as f32;
                        [1., path, path * if island == 0 { 50. } else { 900. }, 2.]
                    }
                }
            })
            .collect();
        queue.write_texture(
            textures[0].as_image_copy(),
            bytemuck::cast_slice(&sheet),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[0].size(),
        );
        let source = if axis == 0 {
            mass_source.clone()
        } else {
            mass_source.replace(
                "smooth_surface(vec2<i32>(p.xy),vec2(1,0))",
                "smooth_surface(vec2<i32>(p.xy),vec2(0,1))",
            )
        };
        let filtered = execute(
            &device,
            &queue,
            source,
            &[&group, &mass_group],
            &[&layout, &mass_layout],
            &mass_buffer,
            [4, 1],
        );
        for (index, v) in filtered.chunks_exact(4).enumerate() {
            let coordinate = if axis == 0 { index % 32 } else { index / 32 };
            let island = ranges
                .iter()
                .position(|&(lo, hi)| (lo..=hi).contains(&coordinate));
            if let Some(island) = island {
                let beta = if island == 0 { 50. } else { 900. };
                assert!(
                    (v[2] / v[1] - beta).abs() < 0.001,
                    "dry gap exchanged turbidity: axis={axis} pixel={index} {v:?}"
                );
            } else {
                assert_eq!(v, &[0.; 4], "optical reflection invented wet coverage");
            }
        }
        for (island, &(lo, hi)) in ranges.iter().enumerate() {
            for component in [1, 2] {
                let sum = |data: &[f32]| {
                    data.chunks_exact(4)
                        .enumerate()
                        .filter_map(|(i, v)| {
                            let q = if axis == 0 { i % 32 } else { i / 32 };
                            (lo..=hi).contains(&q).then_some(v[component] as f64)
                        })
                        .sum::<f64>()
                };
                let error = (sum(&filtered) / sum(&sheet) - 1.).abs();
                println!("independent-film axis={axis} island={island} component={component} optical-volume error={error}");
                assert!(error<0.000002,"independent film changed optical material: axis={axis} island={island} component={component}");
            }
        }
    }
    // A new, almost-empty wet texel must not abruptly redistribute the
    // neighbouring established column. This checks the actual production GPU
    // path, independently of the static integrated-volume requirement above.
    let mut birth = Vec::new();
    let mut birth_inputs = Vec::new();
    for new_water in [0f32, 1e-12] {
        let sheet: Vec<f32> = (0..256)
            .flat_map(|i| {
                let x = (i % 32) as f32;
                if x == 1. && new_water > 0. {
                    return [1., new_water, new_water * 50., 1.];
                }
                if !(2. ..=20.).contains(&x) {
                    return [0.; 4];
                }
                let impulse = (-((x - 2.) / 0.65).powi(2)).exp();
                let path = 0.002 + 0.06 * impulse;
                [1., path, path * (50. + 300. * impulse), 2.]
            })
            .collect();
        queue.write_texture(
            textures[0].as_image_copy(),
            bytemuck::cast_slice(&sheet),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            textures[0].size(),
        );
        birth.push(execute(
            &device,
            &queue,
            mass_source.clone(),
            &[&group, &mass_group],
            &[&layout, &mass_layout],
            &mass_buffer,
            [4, 1],
        ));
        birth_inputs.push(sheet);
    }
    for component in [1, 2] {
        let scale = birth_inputs[0]
            .chunks_exact(4)
            .map(|v| v[component])
            .fold(0f32, f32::max);
        let change = birth[0]
            .chunks_exact(4)
            .zip(birth[1].chunks_exact(4))
            .enumerate()
            .filter(|(i, _)| (2..=20).contains(&(i % 32)))
            .map(|(_, (a, b))| (a[component] - b[component]).abs() / scale)
            .fold(0f32, f32::max);
        println!("wet-texel birth component={component} maximum relative existing-column change={change}");
        assert!(change<0.000002,"negligible wet-texel birth popped an established optical column: component={component} change={change}");
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn channel_wall_clipping_keeps_the_visible_slab_and_handles_parallel_rays() {
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
 let i=p.y*8u+p.x;let angle=f32(i)*0.1;
 let axis=vec3(cos(angle),0.,sin(angle));let center=vec3(14.,0.8,17.);
 let plane=vec4(axis/0.19,-dot(axis,center)/0.19);
 let interval=channel_interval(center-axis*0.7,axis,plane);
 let inside=channel_interval(center,vec3(0.,1.,0.),plane);
 let outside=channel_interval(center+axis,vec3(0.,1.,0.),plane);
 result[i]=vec4(interval,select(0.,1.,inside.x<0. && inside.y>0.),select(0.,1.,outside.x>outside.y));
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
        assert!(
            (value[0] - 0.51).abs() < 1e-5 && (value[1] - 0.89).abs() < 1e-5,
            "{value:?}"
        );
        assert_eq!(value[2..], [1., 1.]);
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn shallow_visible_columns_bound_optical_depth_without_changing_concentration() {
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
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let gap=f32(p.y*8u+p.x)*0.001;
 // A 20 mm source column viewed through a shallower geometric interval
 // cannot contain more optical water than fits before the opaque bed.
 let path=visible_column_path(0.02,gap,true);
 result[p.y*8u+p.x]=vec4(exp(-50.*path),path,
  visible_column_path(0.02,gap,false),visible_column_path(0.,gap,true));
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
    for (i, v) in values.as_chunks::<4>().0.iter().enumerate() {
        let visible = (i as f64 * 0.001).min(0.02);
        assert!((v[0] as f64 - (-50. * visible).exp()).abs() < 1e-6);
        assert!((v[1] as f64 - visible).abs() < 1e-7);
        assert!((v[2] - (i as f32 * 0.001).min(0.02)).abs() < 1e-7);
        assert_eq!(v[3], 0.);
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn detached_front_drop_cannot_claim_the_bulk_interface_but_keeps_optical_volume() {
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
    let depth_source = include_str!("../../mining/src/water_depth.wgsl");
    let helpers = &depth_source[depth_source.find("fn kernel_density(").unwrap()
        ..depth_source.find("// Fixed view-depth probes").unwrap()];
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../../mining/src/water_blur.wgsl"),
        helpers,
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
fn layered_profile(radius:f32,drop_support:f32,with_drop:bool)->array<f32,8> {
 var density:array<f32,8>;
 for(var i=0u;i<8u;i++) {
  let p=vec3(0.,0.,1.+PROFILE_DEPTHS[i]);
  density[i]=kernel_density(p,vec3(0.,0.,1.03),vec3(0.03,0.,0.),vec3(0.,0.03,0.),vec3(0.,0.,0.012),0.5);
  if with_drop {density[i]+=interface_density(kernel_density(p,vec3(0.,0.,1.+radius),vec3(radius,0.,0.),vec3(0.,radius,0.),vec3(0.,0.,radius),1.),drop_support);}
 }
 return density;
}
fn crossing(d:array<f32,8>)->f32 {
 return shared_surface_depth(vec2(1.,-1.),vec4(d[0],d[1],d[2],d[3]),vec4(d[4],d[5],d[6],d[7]));
}
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let radius=0.0025+f32(p.y*8u+p.x)*0.00001;
 // A normalized small front drop used to win the half-density crossing,
 // despite being detached from the larger water layer behind it. Execute both
 // the old negative control and the connected-support reconstruction.
 result[p.y*8u+p.x]=vec4(crossing(layered_profile(radius,0.,true)),crossing(layered_profile(radius,1.,true)),crossing(layered_profile(radius,0.,false)),interface_density(6.5625,1.));
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
    for v in values.as_chunks::<4>().0 {
        assert!(
            v[1] < 1.005,
            "old front-drop crossing was not reproduced: {v:?}"
        );
        assert!(
            v[0] > 1.015 && v[0] < 1.03,
            "bulk surface did not survive: {v:?}"
        );
        assert!(
            (v[0] - v[2]).abs() < 1e-6,
            "detached drop changed the bulk interface: {v:?}"
        );
        assert_eq!(v[3], 6.5625, "fully supported water density was changed");
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn inclined_sheet_has_projective_normal_and_uniform_suspension() {
    let (device, queue) = device();
    let texture = |values: Vec<f32>| {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("inclined water control"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            t.as_image_copy(),
            bytemuck::cast_slice(&values),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            t.size(),
        );
        t
    };
    let field = |f: &dyn Fn(f32, f32) -> [f32; 4]| {
        (0..32)
            .flat_map(|y| (0..32).flat_map(move |x| f(x as f32, y as f32)))
            .collect()
    };
    let depth = |x: f32, y: f32| 1. / (1. + 0.015 * (x - 16.) + 0.007 * (y - 16.));
    let raw = texture(field(&|x, y| [depth(x, y), -1., 0., 0.]));
    // A uniform suspension over a varying path must retain the same beta.
    let joined = texture(field(&|x, y| {
        let path = 0.02 + 0.001 * x + 0.0003 * y;
        [depth(x, y), path, path * 250., 2.]
    }));
    let moments = texture(field(&|_, _| [0.04, 10., 0., 0.]));
    let normals = texture(field(&|_, _| [0., 0., 0.8, 1.]));
    let views: Vec<_> = [&joined, &raw, &moments, &normals, &normals]
        .map(|t| t.create_view(&Default::default()))
        .into();
    let mut uniforms = [0f32; 72];
    for i in [16, 21, 26, 31] {
        uniforms[i] = 1.;
    }
    uniforms[32] = 0.2;
    uniforms[37] = 0.2;
    uniforms[42] = -1.;
    uniforms[47] = 1.;
    uniforms[64] = 64.;
    uniforms[65] = 64.;
    uniforms[66] = 0.045;
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bindings = [1, 3, 4, 5, 6, 9];
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &bindings.map(|binding| wgpu::BindGroupLayoutEntry {
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
    let mut index = 0;
    let entries: Vec<_> = bindings
        .into_iter()
        .map(|binding| {
            let resource = if binding == 3 {
                uniform.as_entire_binding()
            } else {
                let view = &views[index];
                index += 1;
                wgpu::BindingResource::TextureView(view)
            };
            wgpu::BindGroupEntry { binding, resource }
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    });
    let (buffer, out_layout, out_group) = output(&device, 64 * 64 * 2);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let pixel=vec2<f32>(p.xy)+0.5;
 let q=pixel*0.5-0.5;
 let depth=1./(1.+0.015*(q.x-16.)+0.007*(q.y-16.));
 let sampled=sample_depth(pixel);
 let joined=channel_surface(pixel,sampled.x);
 let i=(p.y*64u+p.x)*2u;
 result[i]=joined.normal;
 result[i+1u]=vec4(joined.column,joined.column.y/max(joined.column.x,0.000001),sampled.x-depth);
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [8, 8],
    );
    // Analytic world plane: 1.2 X - .56 Y - .989 Z = 1.
    // The sign below faces the eye. It comes from the projection, not a fitted
    // reference or a copy of the production stencil.
    let length = (1.2f64 * 1.2 + 0.56 * 0.56 + 0.989 * 0.989).sqrt();
    let expected = [-1.2 / length, 0.56 / length, 0.989 / length];
    let mut max_error = 0f64;
    for y in 16..48 {
        for x in 16..48 {
            let i = (y * 64 + x) * 8;
            let v = &values[i..i + 8];
            assert!(v.iter().all(|v| v.is_finite()));
            let error = (0..3)
                .map(|a| (v[a] as f64 - expected[a]).powi(2))
                .sum::<f64>()
                .sqrt();
            max_error = max_error.max(error);
            assert!(
                error < 0.00015,
                "inclined sheet acquired a cap: {x},{y}: {v:?}, error={error}"
            );
            assert!(
                (v[6] - 250.).abs() < 0.001,
                "uniform slurry acquired a patch: {v:?}"
            );
            assert!(
                v[7].abs() < 0.000001,
                "upsampling curved the water plane: {x},{y}: {}",
                v[7]
            );
            // Probe all four full-resolution pixel phases, including their
            // offset from the half-size depth texel. A linear path on this
            // sheet must not develop an optical island from range weighting.
            let qx = (x as f64 + 0.5) * 0.5 - 0.5;
            let qy = (y as f64 + 0.5) * 0.5 - 0.5;
            let path = 0.02 + 0.001 * qx + 0.0003 * qy;
            assert!(
                (v[4] as f64 - path).abs() < 0.00002,
                "inclined path acquired a patch: {x},{y}: {v:?}, expected={path}"
            );
        }
    }
    println!("inclined-sheet maximum analytic normal error: {max_error}");

    // A front-kernel switch on a continuous sheet must not switch its whole
    // transported ripple phase. The old nearest-envelope velocity jumps 0.8
    // m/s at this boundary; the joined column must vary continuously instead.
    queue.write_texture(
        raw.as_image_copy(),
        bytemuck::cast_slice(&field(&|x, y| {
            [depth(x, y), -1., if x < 16. { 0.2 } else { 1. }, 0.12]
        })),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        raw.size(),
    );
    let flow_source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let i=p.y*8u+p.x;
 let pixel=vec2(31.94+f32(i)*0.002,32.);
 let joined=channel_surface(pixel,sample_depth(pixel).x);
 let nearest=textureLoad(raw_depths,vec2<i32>(round(pixel*0.5-0.5)),0).ba;
 result[i]=vec4(joined.flow,nearest);
}"#
    );
    let flows = execute(
        &device,
        &queue,
        flow_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    let mut joined_jump = 0f32;
    let mut nearest_jump = 0f32;
    for (i, value) in flows[..64 * 4].chunks_exact(4).enumerate() {
        assert!(value.iter().all(|v| v.is_finite()));
        assert!((0.2..=1.).contains(&value[0]));
        assert!((value[1] - 0.12).abs() < 0.000001);
        if i > 0 {
            joined_jump = joined_jump.max((value[0] - flows[(i - 1) * 4]).abs());
            nearest_jump = nearest_jump.max((value[2] - flows[(i - 1) * 4 + 2]).abs());
        }
    }
    assert!(nearest_jump > 0.79, "front-kernel control did not switch");
    assert!(
        joined_jump < 0.001,
        "transported detail switched: {joined_jump}"
    );
    println!("flow boundary: nearest jump {nearest_jump}, joined {joined_jump} m/s");

    // A resolved curved sheet is not a particle bump. Match its local
    // analytic tangent and a smoothly varying optical path, through subpixel
    // phases, rather than borrowing a plane fitted beyond the crest.
    // Exercise both an axis-aligned crest and one oblique to the screen.
    for (cross, yy) in [(0f32, 0f32), (0.003, 0.0015), (-0.003, 0.0015)] {
        let curved_depth = |x: f32, y: f32| {
            let x = x - 16.;
            1. / (1.
                + 0.015 * x
                + 0.007 * (y - 16.)
                + 0.0015 * x * x
                + cross * x * (y - 16.)
                + yy * (y - 16.) * (y - 16.))
        };
        for (texture, values) in [
            (&raw, field(&|x, y| [curved_depth(x, y), -1., 0., 0.])),
            (
                &joined,
                field(&|x, y| {
                    let path = 0.04 + 0.0015 * (x - 16.) + 0.0006 * (y - 16.);
                    [curved_depth(x, y), path, path * 250., 2.]
                }),
            ),
        ] {
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&values),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * 16),
                    rows_per_image: None,
                },
                texture.size(),
            );
        }
        let curved_source = format!(
            "{}\n{}",
            include_str!("../../mining/src/water_compose.wgsl"),
            r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let pixel=vec2<f32>(p.xy)+vec2(24.5,24.5);
 let q=pixel*0.5-0.5;
 let x=q.x-16.;
 let depth=1./(1.+0.015*x+0.007*(q.y-16.)+0.0015*x*x+CROSS*x*(q.y-16.)+YY*(q.y-16.)*(q.y-16.));
 let joined=channel_surface(pixel,depth);
 let i=(p.y*64u+p.x)*2u;
 result[i]=joined.normal;
 result[i+1u]=vec4(joined.column,joined.column.y/max(joined.column.x,0.000001),depth);
}"#
            .replace("CROSS", &cross.to_string())
            .replace("YY", &yy.to_string())
        );
        let curved = execute(
            &device,
            &queue,
            curved_source,
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [2, 2],
        );
        let point = |px: f64, py: f64| {
            let x = px * 0.5 - 0.5 - 16.;
            let y = py * 0.5 - 0.5 - 16.;
            let depth = 1.
                / (1.
                    + 0.015 * x
                    + 0.007 * y
                    + 0.0015 * x * x
                    + cross as f64 * x * y
                    + yy as f64 * y * y);
            glam::DVec3::new(
                0.2 * (px / 32. - 1.) * depth,
                0.2 * (1. - py / 32.) * depth,
                -depth,
            )
        };
        let mut curved_normal_error = 0f64;
        let mut curved_path_error = 0f64;
        for y in 0..16 {
            for x in 0..16 {
                let i = (y * 64 + x) * 8;
                let v = &curved[i..i + 8];
                let px = x as f64 + 24.5;
                let py = y as f64 + 24.5;
                let dx = point(px + 0.0001, py) - point(px - 0.0001, py);
                let dy = point(px, py + 0.0001) - point(px, py - 0.0001);
                let expected = dy.cross(dx).normalize();
                let error =
                    (glam::DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64) - expected).length();
                curved_normal_error = curved_normal_error.max(error);
                let path = 0.04 + 0.0015 * (px * 0.5 - 16.5) + 0.0006 * (py * 0.5 - 16.5);
                curved_path_error = curved_path_error.max((v[4] as f64 - path).abs() / path);
                assert!(
                    (v[6] - 250.).abs() < 0.001,
                    "curved uniform slurry changed concentration: {v:?}"
                );
            }
        }
        println!("curved sheet cross={cross} yy={yy} max normal error={curved_normal_error}, optical path relative error={curved_path_error}");
        assert!(
            curved_normal_error < 0.001,
            "resolved crest acquired false normals: {curved_normal_error}"
        );
        assert!(
            curved_path_error < 0.002,
            "curved slurry acquired optical patches: {curved_path_error}"
        );
    }

    // A travelling ridge is not quadratic across the distant shoulder stencil.
    // Independently sample a sinusoidal reciprocal-depth surface and compare
    // both its subpixel position and tangent, including an oblique crest.
    for angle in [0f32, 0.57] {
        let direction = (angle.cos(), angle.sin());
        let ridge_depth = |x: f32, y: f32| {
            let phase = 0.6 * (direction.0 * (x - 16.) + direction.1 * (y - 16.));
            1. / (1. + 0.03 * phase.cos())
        };
        for (texture, values) in [
            (&raw, field(&|x, y| [ridge_depth(x, y), -1., 0., 0.])),
            (&joined, field(&|x, y| [ridge_depth(x, y), 0.04, 10., 2.])),
        ] {
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&values),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * 16),
                    rows_per_image: None,
                },
                texture.size(),
            );
        }
        let ridge_source = format!(
            "{}\n{}",
            include_str!("../../mining/src/water_compose.wgsl"),
            r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let pixel=vec2<f32>(p.xy)+vec2(24.5);
 let sampled=sample_depth(pixel);
 let joined=channel_surface(pixel,sampled.x);
 let i=(p.y*64u+p.x)*2u;
 result[i]=joined.normal;
 result[i+1u]=vec4(sampled,joined.column);
}"#
        );
        let ridge = execute(
            &device,
            &queue,
            ridge_source,
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [2, 2],
        );
        let point = |px: f64, py: f64| {
            let x = px * 0.5 - 16.5;
            let y = py * 0.5 - 16.5;
            let phase = 0.6 * (direction.0 as f64 * x + direction.1 as f64 * y);
            let depth = 1. / (1. + 0.03 * phase.cos());
            glam::DVec3::new(
                0.2 * (px / 32. - 1.) * depth,
                0.2 * (1. - py / 32.) * depth,
                -depth,
            )
        };
        let mut depth_error = 0f64;
        let mut normal_error = 0f64;
        for y in 0..16 {
            for x in 0..16 {
                let i = (y * 64 + x) * 8;
                let v = &ridge[i..i + 8];
                let px = x as f64 + 24.5;
                let py = y as f64 + 24.5;
                let expected_point = point(px, py);
                let dx = point(px + 0.0001, py) - point(px - 0.0001, py);
                let dy = point(px, py + 0.0001) - point(px, py - 0.0001);
                let expected_normal = dy.cross(dx).normalize();
                depth_error = depth_error.max((v[4] as f64 + expected_point.z).abs());
                normal_error = normal_error.max(
                    (glam::DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64) - expected_normal)
                        .length(),
                );
                assert!(
                    (v[7] / v[6] - 250.).abs() < 0.001,
                    "ridge changed uniform slurry: {v:?}"
                );
            }
        }
        println!("sinusoidal ridge angle={angle}: depth error={depth_error} m, normal error={normal_error}");
        assert!(
            depth_error < 0.0002,
            "ridge position broadened between texels"
        );
        assert!(
            normal_error < 0.04,
            "ridge tangent borrowed the distant rounded shoulders"
        );
    }

    // Two genuinely separate flat layers must keep their independent normals
    // and optical columns, even when the old opposite-side secant suggested
    // an inclined sheet through the step. Probe exact half-size texel centres
    // so the test does not confuse deliberate coverage antialiasing with bleed.
    for axis in 0..3 {
        let back = |x: f32, y: f32| [x, y, (x + y) * 0.5][axis] >= 16.;
        for (texture, values) in [
            (
                &raw,
                field(&|x, y| {
                    let back = back(x, y);
                    [if back { 1.12 } else { 1. }, -1., 0., 0.]
                }),
            ),
            (
                &joined,
                field(&|x, y| {
                    let back = back(x, y);
                    [
                        if back { 1.12 } else { 1. },
                        0.04,
                        0.04 * if back { 900. } else { 50. },
                        2.,
                    ]
                }),
            ),
        ] {
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&values),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * 16),
                    rows_per_image: None,
                },
                texture.size(),
            );
        }
        let shader = include_str!("../../mining/src/water_compose.wgsl");
        let edge_source = format!(
            r#"{shader}

@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {{
 let index=12u+p.x;
 var texel=select(vec2(index,16u),vec2(16u,index),{axis}u==1u);
 if {axis}u==2u {{texel=vec2(index,index);}}
 let pixel=(vec2<f32>(texel)+0.5)*2.;
 let depth=select(1.,1.12,index>=16u);
 let joined=channel_surface(pixel,sample_depth(pixel).x);
 result[p.x*2u]=joined.normal;
 result[p.x*2u+1u]=vec4(joined.column,joined.column.y/joined.column.x,depth);
}}"#
        );
        let values = execute(
            &device,
            &queue,
            edge_source,
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [1, 1],
        );
        for i in 0..8 {
            let v = &values[i * 8..i * 8 + 8];
            let beta = if i + 12 >= 16 { 900. } else { 50. };
            let normal_error = (v[0] * v[0] + v[1] * v[1] + (v[2] - 1.).powi(2)).sqrt();
            println!(
                "layer edge axis={axis} texel={} normal_error={normal_error} beta={}",
                i + 12,
                v[6]
            );
            assert!(
                (v[6] - beta).abs() < 0.001,
                "separate water layers borrowed mud/clear water: axis={axis} texel={} {v:?}",
                i + 12
            );
            assert!(
                normal_error < 0.00015,
                "flat layer acquired a false crest: axis={axis} texel={} {v:?}",
                i + 12
            );
        }
    }

    // A muddy stripe more than 12 cm away along a grazing sheet must not
    // tint the clear water at this pixel. The old circular image footprint
    // crossed that stripe despite the 30 mm optical reconstruction scale.
    // All of the intervening depths lie exactly on the same geometric plane:
    // a depth-discontinuity rejection cannot hide a footprint error here.
    let grazing_depth = |y: f32| 1. / (1. + 0.04 * (y - 15.75));
    for (texture, values) in [
        (&raw, field(&|_, y| [grazing_depth(y), -1., 0., 0.])),
        (
            &joined,
            field(&|_, y| {
                let beta = if y == 19. { 500. } else { 10. };
                [grazing_depth(y), 0.04, 0.04 * beta, 2.]
            }),
        ),
    ] {
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&values),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            texture.size(),
        );
    }
    let grazing_source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(1) fn probe() {
 let column=channel_surface(vec2(32.5),1.).column;
 result[0]=vec4(column,column.y/column.x,0.);
}"#
    );
    let grazing = execute(
        &device,
        &queue,
        grazing_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    println!(
        "grazing clear-water extinction beside distant mud: {}",
        grazing[2]
    );
    assert!(
        (grazing[2] - 10.).abs() < 0.1,
        "distant mud crossed the world-space filter footprint: {:?}",
        &grazing[..4]
    );

    // At a grazing angle, a bounded world-space footprint resolves almost
    // no horizontal span. Falling back to a kernel gradient turns this exact
    // plane into a round cap. Check all four subpixel phases against the
    // independently known geometric plane, with an intentionally wrong raw
    // particle gradient. The stable fit must recover the unresolved slope.
    let narrow_depth = |x: f32| 1. / (1. / 3. + 0.04 * (x - 15.75));
    for (texture, values) in [
        (&raw, field(&|x, _| [narrow_depth(x), -1., 0., 0.])),
        (&joined, field(&|x, _| [narrow_depth(x), 0.04, 7.2, 2.])),
    ] {
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&values),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            texture.size(),
        );
    }
    let narrow_source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(1) fn probe() {
 let xs=array(32.5,33.5,32.9998,33.0002);
 for(var y=0u;y<2u;y++){for(var x=0u;x<4u;x++){
  let pixel=vec2(xs[x],32.5+f32(y));
  let sampled=sample_depth(pixel);
  let i=(y*4u+x)*2u;
  result[i]=channel_surface(pixel,sampled.x).normal;
  // Independent plane intersection, including all full-size pixel phases.
  let texel_x=pixel.x*0.5-0.5;
  let analytic=1./(1./3.+0.04*(texel_x-15.75));
  result[i+1u]=vec4(sampled,analytic,0.);
 }}
}"#
    );
    let narrow = execute(
        &device,
        &queue,
        narrow_source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [1, 1],
    );
    // Plane 1 = 3.2 X - (1/3 - .01) Z, derived from the projection.
    let expected = glam::Vec3::new(-3.2, 0., 1. / 3. - 0.01).normalize();
    for pair in narrow[..64].chunks_exact(8) {
        let value = &pair[..4];
        assert!(
            (pair[4] - pair[6]).abs() < 0.00001,
            "grazing plane became a depth step: sampled={}, analytic={}",
            pair[4],
            pair[6]
        );
        assert!(
            (pair[5] - 1.).abs() < 0.000001,
            "connected sheet lost coverage"
        );
        let error = (glam::Vec3::from_slice(value) - expected).length();
        assert!(
            error < 0.001,
            "grazing footprint fell back to a particle cap: {value:?}, error={error}"
        );
    }

    // Adjacent connected/outlet columns may be close in camera depth while
    // representing different layers. Upsampling the channel must not pull its
    // interface toward the outlet, even though the old range gate accepts it.
    let layers = field(&|x, _| {
        if x < 16. {
            [1., 0.04, 2., 2.]
        } else {
            [1.06, 0.02, 0., 0.]
        }
    });
    queue.write_texture(
        joined.as_image_copy(),
        bytemuck::cast_slice(&layers),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(32 * 16),
            rows_per_image: None,
        },
        joined.size(),
    );
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let sampled=sample_depth(vec2(30.5+f32(p.x),32.5));
 result[p.y*64u+p.x]=vec4(sampled,0.,0.);
}"#
    );
    let edge = execute(
        &device,
        &queue,
        source.clone(),
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [8, 8],
    );
    for (i, expected) in [[1., 1.], [1., 0.75], [1.045, 1.], [1.06, 1.]]
        .iter()
        .enumerate()
    {
        assert!(
            (edge[i * 4] - expected[0]).abs() < 0.000001,
            "upsampling borrowed another layer: {i}: {:?}",
            &edge[i * 4..i * 4 + 2]
        );
        assert!(
            (edge[i * 4 + 1] - expected[1]).abs() < 0.000001,
            "layer coverage changed"
        );
    }
    // A genuine 40 cm gap between two channel layers must stay sharp. A
    // one-sided secant across this jump would incorrectly fit a steep sheet.
    // Also retain a dry edge: interpolation can cover the last subpixel, but
    // cannot create a new full texel of water beyond it.
    for dry in [false, true] {
        let values = field(&|x, _| {
            if x < 16. {
                [1., 0.04, 2., 2.]
            } else if dry {
                [0.; 4]
            } else {
                [1.4, 0.04, 2., 2.]
            }
        });
        queue.write_texture(
            joined.as_image_copy(),
            bytemuck::cast_slice(&values),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 16),
                rows_per_image: None,
            },
            joined.size(),
        );
        let edge = execute(
            &device,
            &queue,
            source.clone(),
            &[&group, &out_group],
            &[&layout, &out_layout],
            &buffer,
            [8, 8],
        );
        let expected = if dry {
            [[1., 1.], [1., 0.75], [1., 0.25], [0., 0.]]
        } else {
            [[1., 1.], [1., 0.75], [1.4, 0.75], [1.4, 1.]]
        };
        for (i, expected) in expected.iter().enumerate() {
            assert!(
                (edge[i * 4] - expected[0]).abs() < 0.000001,
                "layer/dry edge depth changed: dry={dry}, index={i}, value={:?}",
                &edge[i * 4..i * 4 + 2]
            );
            assert!(
                (edge[i * 4 + 1] - expected[1]).abs() < 0.000001,
                "layer/dry edge coverage changed: dry={dry}, index={i}, value={:?}",
                &edge[i * 4..i * 4 + 2]
            );
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn dielectric_sun_is_finite_reciprocal_and_concentrated_at_reflection() {
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
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let a=f32(p.y*8u+p.x)*0.002;
 let n=vec3(0.,1.,0.);let v=normalize(vec3(0.,1.,1.));
 let l=normalize(vec3(a,1.,-1.));
 result[p.y*8u+p.x]=vec4(channel_sun(n,v,l,0.),channel_sun(n,l,v,0.),channel_sun(n,v,l,0.02),channel_sun(n,v,-l,0.));
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
    let expected_fresnel = 0.02037 + 0.97963 * (1. - std::f64::consts::FRAC_1_SQRT_2).powi(5);
    let expected_mask = 2. / (1. + (1.0004f64).sqrt());
    let expected = 1. / (std::f64::consts::PI * 0.0004) * expected_mask.powi(2) * expected_fresnel
        / (4. * std::f64::consts::FRAC_1_SQRT_2)
        * 0.6;
    assert!(
        (values[0] as f64 / expected - 1.).abs() < 0.001,
        "dielectric peak: {} vs {expected}",
        values[0]
    );
    for (i, v) in values.as_chunks::<4>().0.iter().enumerate() {
        assert!(v.iter().all(|x| x.is_finite() && *x >= 0.));
        let nl = 1. / (2. + (i as f64 * 0.002).powi(2)).sqrt();
        assert!(
            (v[0] as f64 / nl - v[1] as f64 / std::f64::consts::FRAC_1_SQRT_2).abs() < 0.0001,
            "nonreciprocal reflection: {v:?}"
        );
        assert_eq!(v[3], 0.);
        if i > 0 {
            assert!(v[0] <= values[(i - 1) * 4] + 0.00001);
        }
    }
    assert!(
        values[63 * 4] < values[0] * 0.03,
        "sun remained a broad crest cap"
    );
    assert!(
        values[2] < values[0] * 0.05,
        "roughness increased peak energy"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn grazing_profile_reaches_connected_water_without_moving_spray_or_empty_pixels() {
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
 let scale=2.+f32(p.x)*0.5;var density:array<f32,8>;
 // Independent linear density ramp: its half-density crossing is 100 mm
 // behind the envelope. The old 51 mm camera-Z search cannot see it.
 for(var i=0u;i<8u;i++) {density[i]=max(10.*(PROFILE_DEPTHS[i]*scale-0.05),0.);}
 let near=vec4(density[0],density[1],density[2],density[3]);
 let far=vec4(density[4],density[5],density[6],density[7]);
 result[p.y*8u+p.x]=vec4(shared_surface_depth(vec2(1.,-scale),near,far),
  shared_surface_depth(vec2(1.,1.),near,far),
  shared_surface_depth(vec2(1.,-scale),vec4(0.),vec4(0.)),scale);
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
        assert!(
            (value[0] - 1.1).abs() < 0.000002,
            "grazing sheet remained a front blob: {value:?}"
        );
        assert_eq!(value[1], 1., "outlet/spray moved");
        assert_eq!(value[2], 1., "empty density acquired a sheet");
    }
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_depth.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let c=0.05+f32(p.x)*0.12;let ray=vec3(sqrt(1.-c*c),-c,0.);
 result[p.y*8u+p.x]=vec4(channel_profile_scale(ray,vec3(0.,1.,0.),0.8),c,0.,0.);
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
        let expected = (0.8 / value[1].max(0.15)).max(1.);
        assert!(
            (value[0] - expected).abs() < 1e-6,
            "wrong normal-distance search scale: {value:?}"
        );
        assert!(value[0] >= 1. && value[0] <= 0.8 / 0.15 + 1e-6);
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn travelling_channel_detail_is_bounded_filtered_and_continuous_at_phase_resets() {
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
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let i=p.y*8u+p.x;let mode=i%8u;
 let normal=normalize(vec3(0.2,0.95,0.16));
 let world=vec3(34.+f32(i/8u)*0.013,2.,28.);
 var flow=vec2(0.7,0.12);var confidence=1.;var footprint=0.0001;
 if mode==0u {flow=vec2(0.);}
 if mode==1u {confidence=0.;}
 if mode==2u {footprint=1.;}
 let before=channel_ripple_normal(normal,world,flow,2.-0.000001,footprint,confidence);
 let after=channel_ripple_normal(normal,world,flow,2.+0.000001,footprint,confidence);
 result[i]=vec4(after,length(before-after));
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
    let length = (0.2f64.powi(2) + 0.95f64.powi(2) + 0.16f64.powi(2)).sqrt();
    let expected = [0.2 / length, 0.95 / length, 0.16 / length];
    let mut moving_max = 0f64;
    for (i, value) in values.chunks_exact(4).enumerate() {
        assert!(value.iter().all(|v| v.is_finite()));
        let norm = value[..3].iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        assert!((norm - 1.).abs() < 0.000001);
        let difference = (0..3)
            .map(|a| (value[a] as f64 - expected[a]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            difference < 0.25,
            "unbounded capillary normal: {difference}"
        );
        assert!(value[3] < 0.0001, "phase reset popped: {value:?}");
        if i % 8 <= 2 {
            assert!(difference < 0.000001, "inactive detail changed sheet");
        } else {
            moving_max = moving_max.max(difference);
        }
    }
    assert!(moving_max > 0.02, "moving fine detail was not exercised");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn grazing_pixel_filters_only_unresolved_wave_directions() {
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
    let (buffer, layout, group) = output(&device, 16);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_compose.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
fn legacy_wave(q:vec2<f32>,k:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec2<f32> {
 let phase=vec2(dot(k,dx),dot(k,dy));
 return normalize(k)*cos(dot(q,k))*exp(-0.5*dot(phase,phase));
}
@compute @workgroup_size(1) fn probe() {
 let q=vec2(0.);let dx=vec2(0.1,0.);let dy=vec2(0.,0.001);
 result[0]=vec4(channel_wave(q,vec2(0.,140.),dx,dy),channel_wave(q,vec2(140.,0.),dx,dy));
 result[1]=vec4(channel_wave(q,vec2(0.,140.),dy,dx),channel_wave(q,vec2(84.,112.),vec2(0.001,0.),vec2(0.,0.001)));
 result[2]=vec4(channel_wave(vec2(0.013,-0.009),vec2(51.,19.),vec2(0.025,0.004),vec2(-0.003,0.005)),0.,0.);
 result[3]=vec4(legacy_wave(vec2(0.013,-0.009),vec2(51.,19.),vec2(0.025,0.004),vec2(-0.003,0.005)),0.,0.);
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
    let expected = (0.07f64).sin() / 0.07;
    assert!(
        (values[1] as f64 - expected).abs() < 1e-6,
        "resolved transverse detail was erased: {values:?}"
    );
    assert!(
        values[2].abs() < 1e-6 && values[3].abs() < 1e-6,
        "unresolved direction aliased: {values:?}"
    );
    assert!(
        (values[5] - values[1]).abs() < 1e-6,
        "pixel basis changed filtering"
    );
    let diagonal = (0.042f64.sin() / 0.042) * (0.056f64.sin() / 0.056);
    assert!((values[6] as f64 - 0.6 * diagonal).abs() < 1e-6);
    assert!((values[7] as f64 - 0.8 * diagonal).abs() < 1e-6);
    // Independent midpoint integration over the actual pixel, rather than
    // reproducing the shader's closed-form filter. This resolved band lost
    // more than half its contrast under the previous 12x-variance Gaussian.
    let mut integral = 0.;
    for y in 0..128 {
        for x in 0..128 {
            let u = (x as f64 + 0.5) / 128. - 0.5;
            let v = (y as f64 + 0.5) / 128. - 0.5;
            let px = 0.013 + 0.025 * u - 0.003 * v;
            let py = -0.009 + 0.004 * u + 0.005 * v;
            integral += (51. * px + 19. * py).cos() / (128. * 128.);
        }
    }
    let length = (51f64.powi(2) + 19f64.powi(2)).sqrt();
    assert!((values[8] as f64 - integral * 51. / length).abs() < 0.00001);
    assert!((values[9] as f64 - integral * 19. / length).abs() < 0.00001);
    let legacy_error = ((values[12] as f64 - integral * 51. / length).powi(2)
        + (values[13] as f64 - integral * 19. / length).powi(2))
    .sqrt();
    assert!(legacy_error > 0.35, "legacy over-filter was not reproduced");
    eprintln!("Resolved pixel-integrated wave: expected={integral}, legacy_error={legacy_error}");
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn curved_density_profiles_retain_crests_and_dilute_peaks_travel_between_probes() {
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
    let (buffer, layout, group) = output(&device, 256);
    let source = format!(
        "{}\n{}",
        include_str!("../../mining/src/water_blur.wgsl"),
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
// Actual previous linear/argmax policy, retained as a negative control.
fn legacy_profile(d:array<f32,8>)->f32 {
 var peak=0.;var depth=0.;
 for(var i=0u;i<8u;i++) {
  if d[i]>=0.5 {
   if i==0u {return 0.;}
   let t=(0.5-d[i-1u])/max(d[i]-d[i-1u],0.000001);
   return mix(PROFILE_DEPTHS[i-1u],PROFILE_DEPTHS[i],t);
  }
  if d[i]>peak {peak=d[i];depth=PROFILE_DEPTHS[i];}
 }
 return depth*smoothstep(0.05,0.5,peak);
}
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let i=p.y*16u+p.x;
 let radii=array(0.018,0.024,0.030,0.040,0.050,0.060,0.080);
 let amplitudes=array(0.55,0.6,0.75,1.,1.5,2.,3.);
 let radius=radii[min(i/7u,6u)];let amplitude=amplitudes[i%7u];
 var d:array<f32,8>;
 for(var k=0u;k<8u;k++) {
  let q=(PROFILE_DEPTHS[k]-radius)/radius;
  let support=max(1.-q*q,0.);d[k]=amplitude*support*support*support;
 }
 let near=vec4(d[0],d[1],d[2],d[3]);let far=vec4(d[4],d[5],d[6],d[7]);
 let crossing=shared_surface_depth(vec2(3.,-1.),near,far)-3.;
 let old=legacy_profile(d);
 // A translating, positive quadratic density peak has an independent exact
 // maximum. This exposes a probe-index jump without inventing particle motion.
 let centre=0.014+f32(i)*0.018/255.;
 for(var k=0u;k<8u;k++) {d[k]=0.24-50.*pow(PROFILE_DEPTHS[k]-centre,2.);}
 let moving=shared_surface_depth(vec2(3.,-1.),vec4(d[0],d[1],d[2],d[3]),vec4(d[4],d[5],d[6],d[7]))-3.;
 result[i]=vec4(crossing,old,moving,legacy_profile(d));
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&empty_group, &group],
        &[&empty, &layout],
        &buffer,
        [2, 2],
    );
    let radii = [0.018_f64, 0.024, 0.030, 0.040, 0.050, 0.060, 0.080];
    let amplitudes = [0.55_f64, 0.6, 0.75, 1., 1.5, 2., 3.];
    let mut error = 0_f64;
    let mut legacy_error = 0_f64;
    let mut motion_jump = 0_f32;
    let mut legacy_jump = 0_f32;
    for i in 0..256 {
        let v = &values[i * 4..i * 4 + 4];
        if i < 49 {
            // Exact first half-density crossing of a normalized poly6 ray.
            let exact = radii[i / 7] * (1. - (1. - (0.5 / amplitudes[i % 7]).cbrt()).sqrt());
            if exact <= 0.051 {
                error = error.max((v[0] as f64 - exact).abs());
                legacy_error = legacy_error.max((v[1] as f64 - exact).abs());
            } else {
                // An interface beyond the bounded search is unobserved.
                // Require a bounded fallback, never claim exact reconstruction.
                assert!((0. ..=0.051001).contains(&v[0]));
            }
        }
        let centre = 0.014 + i as f64 * 0.018 / 255.;
        let t = (0.24_f64 - 0.05) / 0.45;
        let exact = centre * t * t * (3. - 2. * t);
        assert!(
            (v[2] as f64 - exact).abs() < 0.000002,
            "dilute peak at {i}: {v:?}"
        );
        if i > 0 {
            motion_jump = motion_jump.max((v[2] - values[(i - 1) * 4 + 2]).abs());
            legacy_jump = legacy_jump.max((v[3] - values[(i - 1) * 4 + 3]).abs());
        }
    }
    println!("poly6 profile max error: old={legacy_error}m new={error}m; dilute step: old={legacy_jump}m new={motion_jump}m");
    assert!(
        legacy_error > 0.008,
        "negative control missed rounded crossing"
    );
    assert!(
        error < 0.0015,
        "curved interface interpolation error={error}"
    );
    assert!(
        legacy_jump > 0.002,
        "negative control missed probe-index jump"
    );
    assert!(
        motion_jump < 0.00004,
        "dilute interface jumped={motion_jump}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn close_view_denoises_particle_domes_without_flattening_riffle_waves() {
    let (device, queue) = device();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("metre-scaled crest with particle sampling ripple"),
        size: wgpu::Extent3d {
            width: 256,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // At z=1 each texel spans 2 mm in world space. A 120 mm hydraulic wave
    // must survive while a 12 mm sampling ripple is rejected. Vary its phase
    // in every row; the expected smooth profile is independent of the filter.
    let field: Vec<f32> = (0..8)
        .flat_map(|y| {
            (0..256).flat_map(move |x| {
                let wave = 0.002 * (std::f32::consts::TAU * (x as f32 - 128.) / 60.).cos();
                let noise = 0.001 * (std::f32::consts::TAU * x as f32 / 6. + y as f32 * 0.37).cos();
                [1. / (1. + wave + noise), 0.025, 1.25, 2.]
            })
        })
        .collect();
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&field),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(256 * 16),
            rows_per_image: None,
        },
        texture.size(),
    );
    let view = texture.create_view(&Default::default());
    let mut uniforms = [0f32; 72];
    uniforms[32] = 0.256;
    uniforms[37] = 0.008;
    uniforms[42] = -1.;
    uniforms[47] = 1.;
    uniforms[64] = 512.;
    uniforms[65] = 16.;
    uniforms[66] = 0.045;
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1, 2, 3, 4].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: if binding == 1 {
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
        }),
    });
    let entries: Vec<_> = (0..5)
        .map(|binding| wgpu::BindGroupEntry {
            binding,
            resource: if binding == 1 {
                uniform.as_entire_binding()
            } else {
                wgpu::BindingResource::TextureView(&view)
            },
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    });
    let (buffer, out_layout, out_group) = output(&device, 1024);
    let shader = std::env::var_os("SLUICE_CREST_SHADER")
        .map(|p| std::fs::read_to_string(p).unwrap())
        .unwrap_or_else(|| include_str!("../../mining/src/water_blur.wgsl").to_owned());
    let source = format!(
        "{shader}\n{}",
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {
 let pixel=vec2(i32(p.x)+64,i32(p.y));
 let filtered=geometry_surface(pixel,vec2(1,0));
 result[p.y*128u+p.x]=vec4(1./filtered.r,filtered.gba);
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [16, 1],
    );
    let mut original = 0f64;
    let mut corrected = 0f64;
    let mut mean = 0f64;
    for i in 0..1024 {
        let x = i % 128 + 64;
        let expected = 1. + 0.002 * (std::f64::consts::TAU * (x as f64 - 128.) / 60.).cos();
        original += (field[((i / 128) * 256 + x) * 4].recip() as f64 - expected).powi(2);
        corrected += (values[i * 4] as f64 - expected).powi(2);
        mean += values[i * 4] as f64 - expected;
        assert_eq!(values[i * 4 + 1], 0.025, "fairing changed the water moment");
        assert_eq!(
            values[i * 4 + 2],
            1.25,
            "fairing changed suspended sediment"
        );
        assert_eq!(values[i * 4 + 3], 2., "fairing changed coverage");
    }
    let reduction = (corrected / original).sqrt();
    println!(
        "metre-scaled crest: ripple RMS ratio={reduction}, mean inverse-depth error={}",
        mean / 1024.
    );
    assert!(
        reduction < 0.30,
        "close-view kernel domes survive: {reduction}"
    );
    assert!(
        (mean / 1024.).abs() < 0.00002,
        "fairing shifted the bulk sheet"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn changing_channel_kernels_retain_water_and_mineral_between_snapshots() {
    let (device, queue) = device();
    let mut particles = Vec::<[f32; 56]>::new();
    let mut expected = Vec::new();
    // Radius, principal support lengths and orientation change independently.
    // These are render snapshots of the same parcel, not changes in its mass.
    for case in 0..12 {
        let mut p = [0f32; 56];
        let mut radii = [0.02f32, 0.02];
        let mut lengths = [[0.7f32; 3], [1.3; 3]];
        let mut angles = [0f32, 0.];
        let mut water = [2e-6f32, 2e-6];
        let mut beta = [100f32, 100.];
        let mut kinds = [0f32, 0.];
        match case {
            1 => {
                radii = [0.006, 0.02];
                lengths = [[1.; 3]; 2];
                beta = [50., 500.];
            }
            2 => {
                lengths = [[0.5, 1., 1.7]; 2];
                angles[1] = 70f32.to_radians();
            }
            3 => {
                lengths = [[0.4, 1.1, 1.7], [1.6, 0.5, 1.]];
            }
            4 => {
                water = [1e-6, 4e-6];
                beta = [100., 800.];
            }
            5 => {
                lengths = [[0.5, 1., 1.7]; 2];
            }
            6 => {
                kinds = [2.; 2];
            }
            7 => {
                radii[0] = 0.;
            }
            8 => {
                radii[1] = 0.;
            }
            9 => {
                kinds[1] = 2.;
            }
            10 => {
                water = [0.; 2];
            }
            11 => {
                kinds[0] = 2.;
            }
            _ => {}
        }
        let axis = |end: usize, i: usize| {
            let rotation = glam::Mat3::from_rotation_z(angles[end]);
            rotation.col(i) * lengths[end][i]
        };
        let mut endpoint_density = [0.; 2];
        for end in 0..2 {
            // A zero-radius endpoint is the existing birth/death presentation
            // fade. Its density is retained, rather than normalized to infinity.
            let r = if radii[end] > 0. { radii[end] } else { 0.02 };
            endpoint_density[end] = water[end]
                / (4. / 3.
                    * std::f32::consts::PI
                    * r.powi(3)
                    * lengths[end].iter().product::<f32>());
            let center = if end == 0 { 36 } else { 0 };
            p[center..center + 4].copy_from_slice(&[0.1, 0.2, 0.3, radii[end]]);
            let shape = if end == 0 { 52 } else { 8 };
            p[shape..shape + 4].copy_from_slice(&[
                endpoint_density[end],
                beta[end],
                kinds[end],
                1.,
            ]);
            for i in 0..3 {
                let offset = if end == 0 { 40 + i * 4 } else { 12 + i * 4 };
                p[offset..offset + 3].copy_from_slice(&axis(end, i).to_array());
            }
        }
        particles.push(p);
        for phase in 0..33 {
            let t = phase as f32 / 32.;
            let mix = |a: f32, b: f32| a + (b - a) * t;
            let radius = mix(radii[0], radii[1]);
            let product = (0..3)
                .map(|i| axis(0, i).lerp(axis(1, i), t).length())
                .product::<f32>();
            let paired_channel = kinds == [0.; 2] && radii.iter().all(|r| *r > 0.);
            let volume = if paired_channel {
                mix(water[0], water[1])
            } else {
                4. / 3.
                    * std::f32::consts::PI
                    * radius.powi(3)
                    * product
                    * mix(endpoint_density[0], endpoint_density[1])
            };
            let mineral = if paired_channel {
                mix(water[0] * beta[0], water[1] * beta[1])
            } else {
                volume * mix(beta[0], beta[1])
            };
            let support = std::array::from_fn::<_, 3, _>(|i| {
                if paired_channel {
                    mix(lengths[0][i], lengths[1][i])
                } else {
                    axis(0, i).lerp(axis(1, i), t).length()
                }
            });
            expected.push((case, phase, [volume, mineral, radius, kinds[1]], support));
        }
    }
    let particles: Vec<f32> = particles.into_iter().flatten().collect();
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("independent material snapshot pairs"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            resource: input.as_entire_binding(),
        }],
    });
    let (buffer, out_layout, out_group) = output(&device, expected.len() as u64 * 2);
    let shader = std::env::var("SLUICE_TEMPORAL_SHADER")
        .map(|path| std::fs::read_to_string(path).unwrap())
        .unwrap_or_else(|_| include_str!("../../mining/src/water_depth.wgsl").to_string());
    let source = format!(
        "{shader}\n{}",
        r#"
@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 let p=present_particle(particles[id.x],f32(id.y)/32.);
 let radius=p.center_radius.w;
 let measure=radius*radius*radius*length(p.axis_x.xyz)*length(p.axis_y.xyz)*length(p.axis_z.xyz);
 let water=4./3.*3.14159265359*measure*p.shape.x;
 let index=(id.x*33u+id.y)*2u;
 result[index]=vec4(water,water*p.shape.y,radius,p.shape.z);
 result[index+1u]=vec4(length(p.axis_x.xyz),length(p.axis_y.xyz),length(p.axis_z.xyz),0.);
}"#
    );
    let values = execute(
        &device,
        &queue,
        source,
        &[&group, &out_group],
        &[&layout, &out_layout],
        &buffer,
        [12, 33],
    );
    let mut maximum_water_error = 0f32;
    let mut maximum_mineral_error = 0f32;
    let mut maximum_legacy_path_error = 0f32;
    let mut maximum_support_error = 0f32;
    for (case, phase, want, support) in expected {
        let actual = &values[(case * 33 + phase) * 8..(case * 33 + phase + 1) * 8];
        for i in 0..3 {
            maximum_support_error =
                maximum_support_error.max((actual[4 + i] / support[i] - 1.).abs());
        }
        assert!(
            actual.iter().all(|v| v.is_finite()),
            "case={case} phase={phase}: {actual:?}"
        );
        let water_error = (actual[0] - want[0]).abs() / want[0].max(1e-12);
        let mineral_error = (actual[1] - want[1]).abs() / want[1].max(1e-10);
        assert!((actual[2] - want[2]).abs() < 1e-7, "support radius changed");
        assert_eq!(actual[3], want[3], "particle kind changed");
        if case <= 5 || case == 10 {
            maximum_water_error = maximum_water_error.max(water_error);
            maximum_mineral_error = maximum_mineral_error.max(mineral_error);
        } else {
            maximum_legacy_path_error = maximum_legacy_path_error
                .max(water_error)
                .max(mineral_error);
        }
    }
    println!("TEMPORAL_KERNEL 396 GPU cases: water_relative_error={maximum_water_error} mineral_relative_error={maximum_mineral_error} outlet_fade_transfer_error={maximum_legacy_path_error} principal_support_error={maximum_support_error}");
    assert!(
        maximum_legacy_path_error < 0.000005,
        "accepted outlet/fade/transfer changed"
    );
    assert!(
        maximum_water_error < 0.000005,
        "interpolation pumped visible water into the crest"
    );
    assert!(
        maximum_mineral_error < 0.000005,
        "interpolation invented a muddy optical patch"
    );
    assert!(
        maximum_support_error < 0.000005,
        "rotating principal support contracted between snapshots"
    );
}
