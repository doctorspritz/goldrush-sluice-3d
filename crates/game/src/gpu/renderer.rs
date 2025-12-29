use bytemuck::{Pod, Zeroable};
use sim::particle::{ParticleMaterial, Particles};

use super::GpuContext;

/// Vertex data for instanced particle rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 2],
    color: [f32; 4],
    size: f32,
    _padding: f32,
}

/// Uniforms for the particle shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    projection: [[f32; 4]; 4],
    viewport_size: [f32; 2],
    _padding: [f32; 2],
}

/// GPU-based particle renderer
pub struct ParticleRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    max_particles: usize,
}

impl ParticleRenderer {
    pub fn new(gpu: &GpuContext, max_particles: usize) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle.wgsl").into()),
            });

        // Uniform buffer for projection matrix
        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Uniform Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Particle Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // color
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            // size
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.surface_format(),
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // Instance buffer for particles
        let instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (max_particles * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            instance_buffer,
            uniform_buffer,
            uniform_bind_group,
            max_particles,
        }
    }

    /// Update particle data and render
    pub fn draw(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        particles: &Particles,
        screen_scale: f32,
        base_particle_size: f32,
    ) {
        // Update uniforms with orthographic projection
        let (width, height) = gpu.size;
        let world_width = width as f32 / screen_scale;
        let world_height = height as f32 / screen_scale;

        // Orthographic projection: world coords -> NDC
        let projection = [
            [2.0 / world_width, 0.0, 0.0, 0.0],
            [0.0, 2.0 / world_height, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
        ];

        let uniforms = Uniforms {
            projection,
            viewport_size: [width as f32, height as f32],
            _padding: [0.0; 2],
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Build instance data
        let particle_count = particles.len().min(self.max_particles);
        let mut instances: Vec<ParticleInstance> = Vec::with_capacity(particle_count);

        for i in 0..particle_count {
            let p = &particles.list[i];
            let pos = p.position;
            let material = p.material;
            let size = if p.diameter > 0.0 { p.diameter } else { p.material.typical_diameter() };

            let color = match material {
                ParticleMaterial::Water => [0.2, 0.5, 0.9, 0.8],
                ParticleMaterial::Sand => [0.76, 0.70, 0.50, 1.0],      // Sandy tan
                ParticleMaterial::Magnetite => [0.15, 0.15, 0.15, 1.0], // Dark gray/black
                ParticleMaterial::Gold => [1.0, 0.84, 0.0, 1.0],        // Gold
                ParticleMaterial::Mud => [0.45, 0.35, 0.25, 1.0],       // Muddy brown
            };

            instances.push(ParticleInstance {
                position: [pos.x, pos.y],
                color,
                size: size * base_particle_size * screen_scale,
                _padding: 0.0,
            });
        }

        if instances.is_empty() {
            return;
        }

        // Upload instance data
        gpu.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear - we want to draw over background
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            // Draw 6 vertices (2 triangles = quad) per instance
            render_pass.draw(0..6, 0..instances.len() as u32);
        }
    }
}
