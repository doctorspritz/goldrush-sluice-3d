use bytemuck::{Pod, Zeroable};
use std::mem;
use wgpu::util::DeviceExt; // Needed for buffer_init

use super::flip_3d::GpuFlip3D;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FluidUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub texel_size: [f32; 2],
    pub particle_radius: f32, // Radius in world units
    pub blur_depth_falloff: f32,
    pub camera_pos: [f32; 3],
    pub padding: f32,
}

pub struct ScreenSpaceFluidRenderer {
    // Pipelines
    depth_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    compose_pipeline: wgpu::RenderPipeline,

    // Uniforms
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // Textures
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,

    blur_texture_a: Option<wgpu::Texture>, // Input/Output (Ping)
    blur_view_a: Option<wgpu::TextureView>,

    blur_texture_b: Option<wgpu::Texture>, // Input/Output (Pong)
    blur_view_b: Option<wgpu::TextureView>,

    // Bind Groups (created per frame/resize)
    depth_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    compose_bind_group_layout: wgpu::BindGroupLayout,

    // Samplers
    linear_sampler: wgpu::Sampler,

    // Config
    pub particle_radius: f32,
    pub blur_depth_falloff: f32,
}

impl ScreenSpaceFluidRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_depth = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fluid_depth_3d.wgsl").into()),
        });

        let shader_blur = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fluid_blur_3d.wgsl").into()),
        });

        let shader_compose = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Compose Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fluid_compose_3d.wgsl").into()),
        });

        // 1. Uniforms
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Uniforms"),
            size: mem::size_of::<FluidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Uniform Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Assume group(0) is always uniforms for simplicity across shaders
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Uniform BG"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // 2. Bind Group Layouts

        // Depth Pass: Read Storage Buffer (Particles)
        let depth_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Depth BG Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Blur Pass: Read Texture + Sampler, and a direction uniform possibly?
        // Note: My shader had @group(2) for direction.

        let blur_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Blur BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // Texture
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Sampler
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let blur_direction_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fluid Blur Direction Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None, // vec2
                    },
                    count: None,
                }],
            });

        // Compose Pass: Read Depth Texture
        let compose_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Compose BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // 3. Pipelines

        // Depth Pipeline
        let depth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fluid Depth Pipeline Layout"),
                bind_group_layouts: &[&uniform_layout, &depth_bg_layout],
                push_constant_ranges: &[],
            });

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fluid Depth Pipeline"),
            layout: Some(&depth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_depth,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers, point sprite gen
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_depth,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float, // High precision linear depth
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip, // Generating strip
                ..Default::default()
            },
            // We need a depth buffer for Z-testing the spheres against *themselves*
            // However, we are outputting to a COLOR attachment (R32F).
            // Do we need hardware depth testing?
            // Yes, if particles overlap, we want the closest Z.
            // So we need a depth attachment too.
            // For now, simpler approach: enable depth test with a separate depth buffer?
            // Or use MAX blending if we store -depth?
            // Traditionally: Use MIN blending on the R32F target.
            // wgpu::BlendState {
            //     color: wgpu::BlendComponent {
            //         src_factor: One, dst_factor: One, operation: Min
            //     } ...
            // }
            // Note: R32Float blending support depends on features (FLOAT32_FILTERABLE ok, but blending?)
            // Core spec allows blending for float32? R32Float isn't blendable by default on all hardware.
            // BETTER: Use a standard depth buffer attachment for hidden surface removal!
            // But we can't READ that depth buffer easily in next pass as texture without waiting.
            // Wait, we can convert depth attachment to texture.
            //
            // LET'S USE MIN BLENDING (Software min via blending).
            // If R32Float blending is supported...
            // If not, we must use a depth attachment.
            // Let's assume we use a Depth attachment `depth_stencil_attachment` that matches the resolution.
            // And we output the Linear Z to color attachment 0.
            // The hardware depth test will handle the sorting.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // Standard Less for View Space (Negative Z)?
                // Wait. View Space Z is negative. Closest is LARGEST (e.g. -5 > -100).
                // So we want Greater if we store standard View Z.
                // IF we store positive distance (-Z), then Less.
                // Shader returns "linear_depth = view_pos.z + z_sphere". This is negative.
                // So closest is MAX (least negative).
                // So CompareFunction::Greater.
                // AND we need to clear to -Infinity.
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Blur Pipeline
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Blur Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &blur_bg_layout, &blur_direction_layout],
            push_constant_ranges: &[],
        });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fluid Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_blur,
                entry_point: Some("vs_main"), // Fullscreen quad
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_blur,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Compose Pipeline
        let compose_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fluid Compose Pipeline Layout"),
                bind_group_layouts: &[&uniform_layout, &compose_bg_layout],
                push_constant_ranges: &[],
            });

        let compose_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fluid Compose Pipeline"),
            layout: Some(&compose_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_compose,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_compose,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,                                        // Output to screen
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Blend with scene
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None, // No depth testing for fullscreen quad composite?
            // Actually we probably want to write depth? Or just read?
            // We are drawing a quad. It has no depth.
            // If we want the water to be occluded by scene geometry...
            // We should use the SCENE depth buffer.
            // But we are compositing ON TOP.
            // The Z-Test should happen in the fragment shader or via early Z.
            // Complex integration. For now, just overlay.
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            depth_pipeline,
            blur_pipeline,
            compose_pipeline,
            uniform_buffer,
            uniform_bind_group,
            depth_texture: None,
            depth_view: None,
            blur_texture_a: None,
            blur_view_a: None,
            blur_texture_b: None,
            blur_view_b: None,
            depth_bind_group_layout: depth_bg_layout,
            blur_bind_group_layout: blur_bg_layout,
            compose_bind_group_layout: compose_bg_layout,
            linear_sampler,
            particle_radius: 0.1,
            blur_depth_falloff: 0.2, // Sensitivity for edge stopping
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Create full resolution textures for R32F depth
        let desc = wgpu::TextureDescriptor {
            label: Some("Fluid R32F Depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let depth_tex = device.create_texture(&desc);
        self.depth_view = Some(depth_tex.create_view(&Default::default()));
        self.depth_texture = Some(depth_tex);

        let blur_tex_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Blur A"),
            ..desc
        });
        self.blur_view_a = Some(blur_tex_a.create_view(&Default::default()));
        self.blur_texture_a = Some(blur_tex_a);

        let blur_tex_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Blur B"),
            ..desc
        });
        self.blur_view_b = Some(blur_tex_b.create_view(&Default::default()));
        self.blur_texture_b = Some(blur_tex_b);
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        gpu_flip: &GpuFlip3D,
        active_particle_count: u32,
        camera_view: [[f32; 4]; 4],
        camera_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        width: u32,
        height: u32,
    ) {
        if self.depth_view.is_none() || active_particle_count == 0 {
            return;
        }

        // 0. Update Uniforms
        // Calculate inv_proj
        use glam::Mat4;
        let proj_mat = Mat4::from_cols_array_2d(&camera_proj);
        let inv_proj_mat = proj_mat.inverse();

        let uniforms = FluidUniforms {
            view: camera_view,
            proj: camera_proj,
            inv_proj: inv_proj_mat.to_cols_array_2d(),
            texel_size: [1.0 / width as f32, 1.0 / height as f32],
            particle_radius: self.particle_radius,
            blur_depth_falloff: self.blur_depth_falloff,
            camera_pos,
            padding: 0.0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // 1. Pass: Render Particles to Linear Depth
        // We need a hardware depth buffer for sorting. Create ephemeral? or can we reuse one passed in?
        // Since we output clear color, we can just create a temporary texture here or cache one.
        // For efficiency, let's just create one. (Though caching would be better)
        let hw_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fluid HW Depth (Ephemeral)"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let hw_depth_view = hw_depth.create_view(&Default::default());

        {
            let particle_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Fluid Particle BG"),
                layout: &self.depth_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_flip.positions_buffer.as_entire_binding(),
                }],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Depth Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.depth_view.as_ref().expect("Depth view should be initialized"),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: -10000.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }), // Clear to far negative
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &hw_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Standard clear for Less
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &particle_bg, &[]);
            pass.draw(0..4, 0..active_particle_count); // 4 verts per instance
        }

        // Helper for creating direction bind groups
        // We know layout 2 is direction
        let dir_layout = self.blur_pipeline.get_bind_group_layout(2);

        // Pass 2a: Horizontal Blur (Depth -> Blur A)
        let dir_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dir H"),
            contents: bytemuck::cast_slice(&[1.0f32, 0.0f32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dir_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dir H BG"),
            layout: &dir_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dir_h.as_entire_binding(),
            }],
        });

        let blur_bg_h = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Input: Depth"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(self.depth_view.as_ref().expect("Depth view should be initialized")),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Blur H"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.blur_view_a.as_ref().expect("Blur view A should be initialized"),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: -10000.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &blur_bg_h, &[]);
            pass.set_bind_group(2, &dir_h_bg, &[]);
            pass.draw(0..3, 0..1); // Fullscreen tri
        }

        // Pass 2b: Vertical Blur (Blur A -> Blur B)
        // blur_view_b will contain the final smoothed depth
        let dir_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dir V"),
            contents: bytemuck::cast_slice(&[0.0f32, 1.0f32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dir_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dir V BG"),
            layout: &dir_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dir_v.as_entire_binding(),
            }],
        });

        let blur_bg_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Input: H Result"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.blur_view_a.as_ref().expect("Blur view A should be initialized"),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Blur V"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.blur_view_b.as_ref().expect("Blur view B should be initialized"),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: -10000.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &blur_bg_v, &[]);
            pass.set_bind_group(2, &dir_v_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3: Composite (Blur B -> Screen)
        let compose_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compose Input: Smoothed Depth"),
            layout: &self.compose_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.blur_view_b.as_ref().expect("Blur view B should be initialized"),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Overlay on existing scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, // Ignore depth for now (overlay)
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.compose_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &compose_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
