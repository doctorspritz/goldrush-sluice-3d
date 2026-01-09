use bytemuck::{Pod, Zeroable};
use sim::grid::Grid;
use sim::particle::Particles;

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

/// Vertex data for terrain rectangles (supports non-square shapes)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TerrainInstance {
    position: [f32; 2], // Center of rectangle
    color: [f32; 4],
    size_x: f32, // Width
    size_y: f32, // Height
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
    terrain_pipeline: wgpu::RenderPipeline,
    particle_buffer: wgpu::Buffer,
    terrain_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    max_particles: usize,
    max_terrain_rects: usize,
    // Terrain caching - avoid rebuilding static geometry every frame
    cached_terrain_count: usize,
    terrain_dirty: bool,
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

        // Terrain shader and pipeline (solid rectangles)
        let terrain_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Terrain Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
            });

        let terrain_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Terrain Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &terrain_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<TerrainInstance>() as u64,
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
                            // size_x
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // size_y
                            wgpu::VertexAttribute {
                                offset: 28,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &terrain_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.surface_format(),
                        blend: None, // Solid terrain, no blending needed
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
        let particle_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (max_particles * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Terrain buffer (RLE rectangles - much smaller than full grid)
        let max_terrain_rects = 10000; // Enough for RLE terrain
        let terrain_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Instance Buffer"),
            size: (max_terrain_rects * std::mem::size_of::<TerrainInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            terrain_pipeline,
            particle_buffer,
            terrain_buffer,
            uniform_buffer,
            uniform_bind_group,
            max_particles,
            max_terrain_rects,
            cached_terrain_count: 0,
            terrain_dirty: true, // Force initial build
        }
    }

    /// Mark terrain as dirty (call when solid cells change)
    pub fn invalidate_terrain(&mut self) {
        self.terrain_dirty = true;
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
        // Note: wgpu uses top-left origin, so we flip Y
        let projection = [
            [2.0 / world_width, 0.0, 0.0, 0.0],
            [0.0, -2.0 / world_height, 0.0, 0.0],  // Flip Y
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],  // Y offset flipped
        ];

        let uniforms = Uniforms {
            projection,
            viewport_size: [width as f32, height as f32],
            _padding: [0.0; 2],
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Build instance data - ALL particle types
        let mut instances: Vec<ParticleInstance> = Vec::with_capacity(self.max_particles);

        for p in &particles.list {
            // Get color from material (u8 RGBA -> f32 RGBA)
            let rgba = p.material.color();
            let color = [
                rgba[0] as f32 / 255.0,
                rgba[1] as f32 / 255.0,
                rgba[2] as f32 / 255.0,
                rgba[3] as f32 / 255.0,
            ];

            // Size based on material type - use actual particle diameter
            let size = if p.material.is_sediment() {
                p.diameter // Use actual particle diameter for sediment
            } else {
                0.6 // Smaller water particles
            };

            instances.push(ParticleInstance {
                position: [p.position.x, p.position.y],
                color,
                size,
                _padding: 0.0,
            });

            if instances.len() >= self.max_particles {
                break;
            }
        }

        // Debug output
        static mut FRAME: u32 = 0;
        unsafe {
            FRAME += 1;
            if FRAME % 120 == 1 {
                eprintln!("PARTICLES: {} rendered", instances.len());
            }
        }

        if instances.is_empty() {
            return;
        }

        // Upload to particle buffer
        gpu.queue
            .write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(&instances));

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
            render_pass.draw(0..6, 0..instances.len() as u32);
        }
    }

    /// Draw terrain (solid cells) as colored blocks
    /// Uses cached geometry - only rebuilds when terrain_dirty is true
    pub fn draw_terrain(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        grid: &Grid,
        cell_size: f32,
        screen_scale: f32,
    ) {
        // Update uniforms with orthographic projection
        let (width, height) = gpu.size;
        let world_width = width as f32 / screen_scale;
        let world_height = height as f32 / screen_scale;

        let projection = [
            [2.0 / world_width, 0.0, 0.0, 0.0],
            [0.0, -2.0 / world_height, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];

        let uniforms = Uniforms {
            projection,
            viewport_size: [width as f32, height as f32],
            _padding: [0.0; 2],
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Only rebuild terrain buffer if dirty - use RLE to merge horizontal runs
        if self.terrain_dirty {
            let mut instances: Vec<TerrainInstance> = Vec::new();
            let terrain_color = [0.45, 0.35, 0.25, 1.0]; // Lighter brown terrain

            // RLE: scan each row and merge consecutive solid cells into rectangles
            for j in 0..grid.height {
                let mut run_start: Option<usize> = None;

                for i in 0..=grid.width {
                    let is_solid = if i < grid.width {
                        let idx = j * grid.width + i;
                        grid.solid[idx]
                    } else {
                        false // End of row triggers final run
                    };

                    if is_solid {
                        if run_start.is_none() {
                            run_start = Some(i);
                        }
                    } else if let Some(start) = run_start {
                        // End of run - emit rectangle
                        let run_len = i - start;
                        let width = run_len as f32 * cell_size;
                        let height = cell_size;

                        // Rectangle center
                        let x = (start as f32 + run_len as f32 / 2.0) * cell_size;
                        let y = (j as f32 + 0.5) * cell_size;

                        instances.push(TerrainInstance {
                            position: [x, y],
                            color: terrain_color,
                            size_x: width * 1.02, // Slight overlap to avoid gaps
                            size_y: height * 1.02,
                        });

                        run_start = None;
                    }
                }
            }

            // Update cached count and upload to GPU
            self.cached_terrain_count = instances.len().min(self.max_terrain_rects);
            if self.cached_terrain_count > 0 {
                gpu.queue.write_buffer(
                    &self.terrain_buffer,
                    0,
                    bytemuck::cast_slice(&instances[..self.cached_terrain_count]),
                );
            }

            self.terrain_dirty = false;
            eprintln!(
                "TERRAIN: {} rectangles (RLE from {} rows)",
                self.cached_terrain_count,
                grid.height
            );
        }

        if self.cached_terrain_count == 0 {
            return;
        }

        // Render pass using cached data with terrain pipeline
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.terrain_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.terrain_buffer.slice(..));
            render_pass.draw(0..6, 0..self.cached_terrain_count as u32);
        }
    }
}
