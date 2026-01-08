//! Friction-Only Sediment Sluice Demo
//!
//! Clean example using the new sluice_geometry and water_heightfield modules
//! with friction-only sediment physics.
//!
//! Run with: cargo run --example friction_sluice --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder, SluiceVertex};
use game::water_heightfield::{WaterHeightfieldRenderer, WaterRenderConfig, WaterVertex};
use glam::{Mat3, Mat4, Vec3};
use sim3d::FlipSimulation3D;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Grid configuration - shorter sluice with steeper angle
const SLUICE_WIDTH: usize = 80;  // Sluice length in cells
const EXIT_BUFFER: usize = 15;   // Buffer zone past exit for clean outflow
const GRID_WIDTH: usize = SLUICE_WIDTH + EXIT_BUFFER;  // Total simulation width
const GRID_HEIGHT: usize = 40;
const GRID_DEPTH: usize = 20;
const CELL_SIZE: f32 = 0.04;    // Slightly larger cells
const MAX_PARTICLES: usize = 300_000;

// Simulation
const GRAVITY: f32 = -9.8;
const PRESSURE_ITERS: u32 = 80;

// Emission
const WATER_EMIT_RATE: usize = 100;
const SEDIMENT_EMIT_RATE: usize = 30;

// Sediment color (brownish)
const SEDIMENT_COLOR: [f32; 4] = [0.6, 0.4, 0.2, 1.0];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 3],
    color: [f32; 4],
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    // Pipelines
    sluice_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,

    // Buffers
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    sluice_vertex_buffer: wgpu::Buffer,
    sluice_index_buffer: wgpu::Buffer,
    water_vertex_buffer: wgpu::Buffer,
    particle_vertex_buffer: wgpu::Buffer,
    particle_instance_buffer: wgpu::Buffer,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    gpu_flip: Option<GpuFlip3D>,
    sim: FlipSimulation3D,
    sluice_builder: SluiceGeometryBuilder,
    water_renderer: WaterHeightfieldRenderer,

    // Particle data for GPU transfer
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,

    // State
    paused: bool,
    frame: u32,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    water_emit_rate: usize,
    sediment_emit_rate: usize,

    // Timing
    start_time: Instant,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
}

impl App {
    fn new() -> Self {
        // Configure sluice geometry - smooth ramp feed, then riffles
        // Sluice ends before the grid boundary, leaving buffer zone for clean outflow
        let sluice_config = SluiceConfig {
            grid_width: SLUICE_WIDTH,  // Sluice width, not full grid width
            grid_height: GRID_HEIGHT,
            grid_depth: GRID_DEPTH,
            cell_size: CELL_SIZE,
            floor_height_left: 15,      // Higher start (steeper)
            floor_height_right: 3,      // Low end
            riffle_spacing: 8,          // Closer riffles in riffle section
            riffle_height: 2,           // Shorter riffles
            riffle_thickness: 2,
            riffle_start_x: 25,         // Ramp section: first 25 cells are smooth
            riffle_end_pad: 8,
            wall_margin: 4,
            exit_width_fraction: 1.0,   // Full-width exit to avoid corner pressure spikes
            exit_height: 10,            // Tall exit
            ..Default::default()
        };

        let sluice_builder = SluiceGeometryBuilder::new(sluice_config.clone());

        // Create simulation
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.pressure_iterations = PRESSURE_ITERS as usize;

        // Mark solid cells from sluice geometry
        let mut solid_count = 0;
        for (i, j, k) in sluice_builder.solid_cells() {
            sim.grid.set_solid(i, j, k);
            solid_count += 1;
        }
        println!("Sluice: Marked {} solid cells", solid_count);

        // Compute SDF from solid cells
        sim.grid.compute_sdf();

        // Debug: Check SDF values
        let sdf_min = sim.grid.sdf.iter().cloned().fold(f32::INFINITY, f32::min);
        let sdf_max = sim.grid.sdf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sdf_neg_count = sim.grid.sdf.iter().filter(|&&v| v < 0.0).count();
        println!("SDF: min={:.3}, max={:.3}, negative_count={}", sdf_min, sdf_max, sdf_neg_count);

        // Water renderer
        let water_config = WaterRenderConfig {
            subdivisions: 2,
            ..Default::default()
        };
        let water_renderer = WaterHeightfieldRenderer::new(
            GRID_WIDTH,
            GRID_DEPTH,
            CELL_SIZE,
            water_config,
        );

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            sluice_builder,
            water_renderer,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            paused: false,
            frame: 0,
            camera_angle: 0.5,
            camera_pitch: 0.4,
            camera_distance: 4.0,
            mouse_pressed: false,
            last_mouse_pos: None,
            water_emit_rate: WATER_EMIT_RATE,
            sediment_emit_rate: SEDIMENT_EMIT_RATE,
            start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
        }
    }

    fn floor_height_at(&self, x: f32) -> f32 {
        let config = self.sluice_builder.config();
        let t = x / (config.grid_width as f32 * config.cell_size);
        let left = config.floor_height_left as f32 * config.cell_size;
        let right = config.floor_height_right as f32 * config.cell_size;
        left * (1.0 - t) + right * t
    }

    fn flow_accel(&self) -> f32 {
        let config = self.sluice_builder.config();
        let rise = (config.floor_height_left as f32 - config.floor_height_right as f32) * config.cell_size;
        let run = config.grid_width as f32 * config.cell_size;
        let slope = rise / run;
        9.8 * slope
    }

    fn emit_particles(&mut self) {
        if self.paused || self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        let config = self.sluice_builder.config();
        // Emit at upstream (high) end, before the riffles
        let emit_x = 3.0 * config.cell_size;  // Near left wall (upstream)
        let center_z = config.grid_depth as f32 * config.cell_size * 0.5;
        let floor_y = self.floor_height_at(emit_x);
        let drop_height = 4.0 * config.cell_size;

        let spread_z = (config.grid_depth as f32 - 4.0) * config.cell_size * 0.4;

        // Emit water
        for _ in 0..self.water_emit_rate {
            if self.sim.particles.len() >= MAX_PARTICLES {
                break;
            }
            let x = emit_x + (rand_float() - 0.5) * 2.0 * config.cell_size;
            let z = center_z + (rand_float() - 0.5) * spread_z;
            let y = floor_y + drop_height + rand_float() * config.cell_size;
            self.sim.spawn_particle(Vec3::new(x, y, z));
        }

        // Emit sediment
        for _ in 0..self.sediment_emit_rate {
            if self.sim.particles.len() >= MAX_PARTICLES {
                break;
            }
            let x = emit_x + (rand_float() - 0.5) * 2.0 * config.cell_size;
            let z = center_z + (rand_float() - 0.5) * spread_z;
            let y = floor_y + drop_height + rand_float() * config.cell_size;
            self.sim.spawn_sediment(Vec3::new(x, y, z), Vec3::ZERO, 2.5);
        }
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }

        let dt = 1.0 / 60.0;
        let flow_accel = self.flow_accel();

        // Emit every other frame
        if self.frame % 2 == 0 {
            self.emit_particles();
        }

        // Sync to GPU
        self.positions.clear();
        self.velocities.clear();
        self.affine_vels.clear();
        self.densities.clear();

        for p in &self.sim.particles.list {
            self.positions.push(p.position);
            self.velocities.push(p.velocity);
            self.affine_vels.push(p.affine_velocity);
            self.densities.push(p.density);
        }

        // Update cell types
        self.cell_types.clear();
        self.cell_types.resize(GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH, 0);

        // Mark solids from SDF
        for (idx, &sdf_val) in self.sim.grid.sdf.iter().enumerate() {
            if sdf_val < 0.0 {
                self.cell_types[idx] = 2; // Solid
            }
        }

        // Mark fluid cells from particles
        for pos in &self.positions {
            let i = (pos.x / CELL_SIZE) as i32;
            let j = (pos.y / CELL_SIZE) as i32;
            let k = (pos.z / CELL_SIZE) as i32;
            if i >= 0 && i < GRID_WIDTH as i32 && j >= 0 && j < GRID_HEIGHT as i32 && k >= 0 && k < GRID_DEPTH as i32 {
                let idx = k as usize * GRID_WIDTH * GRID_HEIGHT + j as usize * GRID_WIDTH + i as usize;
                if self.cell_types[idx] != 2 {
                    self.cell_types[idx] = 1; // Fluid
                }
            }
        }

        // GPU simulation step
        if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
            let sdf = self.sim.grid.sdf.as_slice();

            gpu_flip.step(
                &gpu.device,
                &gpu.queue,
                &mut self.positions,
                &mut self.velocities,
                &mut self.affine_vels,
                &self.densities,
                &self.cell_types,
                Some(sdf),
                None,
                dt,
                GRAVITY,
                flow_accel,
                PRESSURE_ITERS,
            );

            // Sync back to CPU sim
            for (i, p) in self.sim.particles.list.iter_mut().enumerate() {
                if i < self.positions.len() {
                    p.position = self.positions[i];
                    p.velocity = self.velocities[i];
                    p.affine_velocity = self.affine_vels[i];
                }
            }

            // Remove particles that exited the sluice (not the grid - buffer zone is part of grid)
            // Delete at sluice boundary, not grid boundary, for clean outflow
            let sluice_exit_x = SLUICE_WIDTH as f32 * CELL_SIZE;
            self.sim.particles.list.retain(|p| {
                p.position.x > 0.0 &&
                p.position.x < sluice_exit_x &&  // Delete at sluice exit, not grid edge
                p.position.y > -CELL_SIZE &&
                p.position.y < GRID_HEIGHT as f32 * CELL_SIZE &&
                p.position.z > 0.0 &&
                p.position.z < GRID_DEPTH as f32 * CELL_SIZE
            });
        }

        self.frame += 1;

        // FPS
        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.last_fps_time = now;

            let water_count = self.sim.particles.list.iter().filter(|p| p.density <= 1.0).count();
            let sediment_count = self.sim.particles.list.len() - water_count;
            println!(
                "Frame {} | FPS: {:.1} | Particles: {} (water: {}, sediment: {})",
                self.frame, self.current_fps, self.sim.particles.list.len(), water_count, sediment_count
            );
        }
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        // Update uniforms
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.3,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        let eye = center + Vec3::new(
            self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
        );
        let view_matrix = Mat4::look_at_rh(eye, center, Vec3::Y);
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let proj_matrix = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let view_proj = proj_matrix * view_matrix;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Build water mesh
        let time = self.start_time.elapsed().as_secs_f32();
        let water_particles = self.sim.particles.list.iter()
            .filter(|p| p.density <= 1.0)
            .map(|p| (p.position.to_array(), p.velocity.to_array()));

        // Capture floor height params to avoid borrowing self in closure
        let (floor_height_left, floor_height_right, total_width) = {
            let config = self.sluice_builder.config();
            (
                config.floor_height_left as f32 * config.cell_size,
                config.floor_height_right as f32 * config.cell_size,
                config.grid_width as f32 * config.cell_size,
            )
        };

        self.water_renderer.build_mesh(
            water_particles,
            time,
            |x| {
                let t = x / total_width;
                floor_height_left * (1.0 - t) + floor_height_right * t
            },
        );

        // Upload water vertices
        let water_vertices = self.water_renderer.vertices();
        if !water_vertices.is_empty() {
            gpu.queue.write_buffer(
                &gpu.water_vertex_buffer,
                0,
                bytemuck::cast_slice(water_vertices),
            );
        }

        // Build sediment instances
        let sediment_instances: Vec<ParticleInstance> = self.sim.particles.list.iter()
            .filter(|p| p.density > 1.0)
            .map(|p| ParticleInstance {
                position: p.position.to_array(),
                color: SEDIMENT_COLOR,
            })
            .collect();

        if !sediment_instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.particle_instance_buffer,
                0,
                bytemuck::cast_slice(&sediment_instances),
            );
        }

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1, g: 0.1, b: 0.15, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);

            // Draw sluice
            pass.set_pipeline(&gpu.sluice_pipeline);
            pass.set_vertex_buffer(0, gpu.sluice_vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.sluice_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.sluice_builder.indices().len() as u32, 0, 0..1);

            // Draw water
            if !water_vertices.is_empty() {
                pass.set_pipeline(&gpu.water_pipeline);
                pass.set_vertex_buffer(0, gpu.water_vertex_buffer.slice(..));
                pass.draw(0..water_vertices.len() as u32, 0..1);
            }

            // Draw sediment particles
            if !sediment_instances.is_empty() {
                pass.set_pipeline(&gpu.particle_pipeline);
                pass.set_vertex_buffer(0, gpu.particle_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.particle_instance_buffer.slice(..));
                pass.draw(0..6, 0..sediment_instances.len() as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..wgpu::Limits::default()
                }
                .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Build sluice mesh
        self.sluice_builder.build_mesh(|i, j, k| self.sim.grid.is_solid(i, j, k));
        self.sluice_builder.upload(&device);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: &[0u8; std::mem::size_of::<Uniforms>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Layout"),
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Sluice pipeline
        let sluice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sluice Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[SluiceVertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Water pipeline (same vertex format as sluice)
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<WaterVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Water visible from both sides
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Particle pipeline (instanced quads)
        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_particle"),
                buffers: &[
                    // Quad vertices
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Instance data
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { offset: 0, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { offset: 12, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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

        // Buffers - create from builder data
        let sluice_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Vertices"),
            contents: bytemuck::cast_slice(self.sluice_builder.vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sluice_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Indices"),
            contents: bytemuck::cast_slice(self.sluice_builder.indices()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let water_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Vertices"),
            size: (GRID_WIDTH * GRID_DEPTH * 6 * 4 * std::mem::size_of::<WaterVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Quad for particles
        let quad_verts: [[f32; 2]; 6] = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
            [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        ];
        let particle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Quad"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let particle_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instances"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GPU FLIP
        let gpu_flip = GpuFlip3D::new(
            &device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            config,
            sluice_pipeline,
            water_pipeline,
            particle_pipeline,
            uniform_buffer,
            uniform_bind_group,
            sluice_vertex_buffer,
            sluice_index_buffer,
            water_vertex_buffer,
            particle_vertex_buffer,
            particle_instance_buffer,
        });
        self.gpu_flip = Some(gpu_flip);
        self.window = Some(window);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Friction Sluice - Sediment Test")
                .with_inner_size(winit::dpi::LogicalSize::new(1200, 800));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.init_gpu(window.clone());
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        PhysicalKey::Code(KeyCode::Space) => self.paused = !self.paused,
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.sim.particles.list.clear();
                            self.frame = 0;
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.water_emit_rate = (self.water_emit_rate + 25).min(500);
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.water_emit_rate = self.water_emit_rate.saturating_sub(25);
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.sediment_emit_rate = (self.sediment_emit_rate + 10).min(200);
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.sediment_emit_rate = self.sediment_emit_rate.saturating_sub(10);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        self.camera_angle -= dx as f32 * 0.01;
                        self.camera_pitch = (self.camera_pitch + dy as f32 * 0.01).clamp(-1.4, 1.4);
                    }
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.3).clamp(1.0, 15.0);
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

struct ParticleInput {
    @location(0) quad_pos: vec2<f32>,
    @location(2) position: vec3<f32>,
    @location(3) color: vec4<f32>,
}

@vertex
fn vs_particle(in: ParticleInput) -> VertexOutput {
    var out: VertexOutput;
    let size = 0.008;
    let world_pos = in.position + vec3<f32>(in.quad_pos * size, 0.0);
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
