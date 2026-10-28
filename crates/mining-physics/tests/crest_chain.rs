//! Explicit real-GPU regression for the runtime's complete surface pass schedule.
use wgpu::util::DeviceExt;

#[path = "../../mining/src/water_passes.rs"]
mod water_passes;

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

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

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn complete_surface_chain_preserves_resolved_ridges_and_uniform_slurry() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    let extent = wgpu::Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: 1,
    };
    // Production raw/ping/pong are RGBA32F; this deliberately renders into
    // textures rather than calling an individual helper from a compute shader.
    let textures: Vec<_> = (0..5)
        .map(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("crest-chain raw/ping/pong/columns/density"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        })
        .collect();
    let views: Vec<_> = textures
        .iter()
        .map(|t| t.create_view(&Default::default()))
        .collect();
    let upload = |index: usize, values: &[f32]| {
        queue.write_texture(
            textures[index].as_image_copy(),
            bytemuck::cast_slice(values),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 16),
                rows_per_image: Some(SIZE),
            },
            extent,
        )
    };
    let columns: Vec<_> = (0..SIZE * SIZE).flat_map(|_| [0.04, 10., 0., 0.]).collect();
    upload(3, &columns);
    upload(4, &vec![1.; (SIZE * SIZE * 4) as usize]);
    let mut uniforms = [0f32; 72];
    let projection = glam::Mat4::perspective_rh(0.4, 1., 0.1, 100.);
    uniforms[..16].copy_from_slice(&projection.to_cols_array());
    uniforms[16..32].copy_from_slice(&glam::Mat4::IDENTITY.to_cols_array());
    uniforms[32..48].copy_from_slice(&projection.inverse().to_cols_array());
    uniforms[48..64].copy_from_slice(&glam::Mat4::IDENTITY.to_cols_array());
    // The render targets are half the actual viewport, just as in the app.
    uniforms[64] = (SIZE * 2) as f32;
    uniforms[65] = (SIZE * 2) as f32;
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
            visibility: wgpu::ShaderStages::FRAGMENT,
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
    let groups: Vec<_> = (0..3)
        .map(|source| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[source]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&views[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&views[4]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&views[4]),
                    },
                ],
            })
        })
        .collect();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production water_blur.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../mining/src/water_blur.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let names = [
        "reconstruct",
        "geometry_x",
        "geometry_y",
        "blur_x",
        "blur_y",
        "optical_x",
        "optical_y",
        "visible_optical_x",
        "visible_optical_y",
    ];
    let pipelines: Vec<_> = names
        .iter()
        .map(|name| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        })
        .collect();
    let draw =
        |encoder: &mut wgpu::CommandEncoder, pipeline: usize, source: usize, target: usize| {
            assert_ne!(source, target, "no attachment feedback");
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &views[target],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(names[pipeline]),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipelines[pipeline]);
            pass.set_bind_group(0, &groups[source], &[]);
            pass.draw(0..3, 0..1);
        };
    let run = |schedule: [usize; 6], visible: bool| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIZE * SIZE * 16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        // A positive density profile chooses the analytic raw interface; the
        // production reconstruction also seeds its optics and channel tag.
        draw(&mut encoder, 0, 0, 2);
        for (index, pipeline) in schedule.into_iter().enumerate() {
            draw(&mut encoder, pipeline, 2 - index % 2, 1 + index % 2);
        }
        if visible {
            draw(&mut encoder, 7, 2, 1);
            draw(&mut encoder, 8, 1, 2);
        }
        encoder.copy_texture_to_buffer(
            textures[2].as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 16),
                    rows_per_image: Some(SIZE),
                },
            },
            extent,
        );
        queue.submit(Some(encoder.finish()));
        read(&device, &buffer)
    };
    let production = water_passes::surface_passes(1usize, 2, 5, 6);
    let legacy = [1, 2, 3, 4, 5, 6];
    let mut worst_fixed: f64 = 0.;
    let mut best_negative = f64::INFINITY;
    for axis in 0..2 {
        for phase in [0., 0.25, 0.5, 0.75] {
            let analytic = |coordinate: u32| {
                let x = f64::from(coordinate) - 32. - phase;
                1. / (1. + 0.02 * (x * std::f64::consts::TAU / 12.).cos())
            };
            let raw: Vec<f32> = (0..SIZE * SIZE)
                .flat_map(|i| {
                    let coordinate = if axis == 0 { i % SIZE } else { i / SIZE };
                    [analytic(coordinate) as f32, -1., 0., 0.]
                })
                .collect();
            upload(0, &raw);
            let fixed = run(production, false);
            let old = run(legacy, false);
            let frozen = run(production, true);
            let mut fixed_error: f64 = 0.;
            let mut old_error: f64 = 0.;
            // Every inspected sample is well away from all stencil borders.
            for y in 24..40 {
                for x in 24..40 {
                    let index = ((y * SIZE + x) * 4) as usize;
                    let expected = analytic(if axis == 0 { x } else { y });
                    fixed_error = fixed_error.max((f64::from(fixed[index]) - expected).abs());
                    old_error = old_error.max((f64::from(old[index]) - expected).abs());
                    for values in [&fixed, &old, &frozen] {
                        assert!(values[index..index + 4].iter().all(|v| v.is_finite()));
                        assert!(
                            (values[index + 2] / values[index + 1] - 250.).abs() < 0.002,
                            "uniform slurry became a muddy patch: axis={axis}, phase={phase}"
                        );
                        assert_eq!(values[index + 3], 2., "channel classification changed");
                    }
                    assert_eq!(
                        fixed[index].to_bits(),
                        frozen[index].to_bits(),
                        "visible optics moved geometry"
                    );
                }
            }
            assert!(
                fixed_error < 0.00005,
                "complete chain rounded ridge: axis={axis}, phase={phase}, error={fixed_error}m"
            );
            assert!(old_error > 0.002, "old pass schedule failed to reproduce rounding: axis={axis}, phase={phase}, error={old_error}m");
            worst_fixed = worst_fixed.max(fixed_error);
            best_negative = best_negative.min(old_error);
            println!("crest chain axis={axis} phase={phase}: corrected={fixed_error}m legacy={old_error}m");
        }
    }
    println!("complete render-chain worst corrected={worst_fixed}m; smallest legacy failure={best_negative}m");
}
