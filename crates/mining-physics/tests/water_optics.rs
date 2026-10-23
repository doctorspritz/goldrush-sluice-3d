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
 var near=vec4(0.);
 var far=vec4(0.1,0.9,1.,1.);
 // Independent near/deep ramps, two layers (choose the first), vacuum,
 // an already submerged front, and a crossing at the last interval.
 switch p.x {
  case 0u: {near=vec4(0.,0.1,0.4,0.8);}
  case 2u: {near=vec4(0.,0.1,0.6,0.);}
  case 3u: {far=vec4(0.);}
  case 4u: {near=vec4(0.8);}
  case 5u: {far=vec4(0.,0.,0.,1.);}
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
        assert!((value[0] - expected[i % 8]).abs() < 1e-6, "{value:?}");
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
fn joined_surface_motion_keeps_the_clipped_optical_column_and_sparse_path() {
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
 // Equal pre-clipped columns must transmit the same light, even if a joined
 // interface moves relative to the bed. The outlet retains its own clip.
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
        assert!((v[0] as f64 - (-1f64).exp()).abs() < 1e-6);
        assert!((v[1] - 0.02).abs() < 1e-7);
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
        source,
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
}
