//! Minimal Sluice Test - Friction-based sediment physics
//!
//! Tests the simplified friction-only sediment model without the bed heightfield system.
//! Water + sediment flow down a sloped channel with riffles.
//!
//! Run with: cargo run --example sluice_friction_test --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Mat4, Vec3};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Grid dimensions
const GRID_WIDTH: usize = 100;
const GRID_HEIGHT: usize = 30;
const GRID_DEPTH: usize = 20;
const CELL_SIZE: f32 = 0.02;
const MAX_PARTICLES: usize = 100_000;

// Sluice geometry
const FLOOR_HEIGHT_LEFT: usize = 8;
const FLOOR_HEIGHT_RIGHT: usize = 3;
const RIFFLE_SPACING: usize = 15;
const RIFFLE_HEIGHT: usize = 2;
const RIFFLE_START_X: usize = 10;

// Physics
const GRAVITY: f32 = -9.81;
const FLOW_ACCEL: f32 = 0.5;
const PRESSURE_ITERS: u32 = 40;

// Emission
const WATER_EMIT_RATE: usize = 80;
const SEDIMENT_EMIT_RATE: usize = 20;
const SEDIMENT_DENSITY: f32 = 2.65;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    gpu_flip: Option<GpuFlip3D>,
    paused: bool,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    frame: u32,
    // CPU-side particle data
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,
    sdf: Vec<f32>,
    water_emit_rate: usize,
    sediment_emit_rate: usize,
    last_frame: Instant,
    fps_history: Vec<f32>,
    rng_state: u64,
    // Camera controls
    keys_held: [bool; 6], // W, A, S, D, Q, E
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            paused: false,
            camera_angle: 0.3,
            camera_pitch: 0.4,
            camera_distance: 3.0,
            frame: 0,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            sdf: Vec::new(),
            water_emit_rate: WATER_EMIT_RATE,
            sediment_emit_rate: 0, // Start with water only - add sediment with Right arrow
            last_frame: Instant::now(),
            fps_history: Vec::new(),
            rng_state: 12345,
            keys_held: [false; 6],
        }
    }

    fn update_cell_types(&mut self) {
        // Reset to air (0) or solid (2)
        for (idx, &sdf_val) in self.sdf.iter().enumerate() {
            self.cell_types[idx] = if sdf_val < 0.0 { 2 } else { 0 };
        }

        // Mark fluid cells (1) based on particle positions
        let w = GRID_WIDTH;
        let h = GRID_HEIGHT;
        let d = GRID_DEPTH;
        for pos in &self.positions {
            let i = (pos.x / CELL_SIZE) as i32;
            let j = (pos.y / CELL_SIZE) as i32;
            let k = (pos.z / CELL_SIZE) as i32;
            if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                let idx = k as usize * w * h + j as usize * w + i as usize;
                if self.cell_types[idx] != 2 {
                    self.cell_types[idx] = 1; // Fluid
                }
            }
        }
    }

    fn update_camera(&mut self, dt: f32) {
        let speed = 2.0 * dt;
        let rot_speed = 1.5 * dt;

        // WASD for rotation
        if self.keys_held[0] {
            self.camera_pitch += rot_speed;
        } // W - pitch up
        if self.keys_held[2] {
            self.camera_pitch -= rot_speed;
        } // S - pitch down
        if self.keys_held[1] {
            self.camera_angle -= rot_speed;
        } // A - rotate left
        if self.keys_held[3] {
            self.camera_angle += rot_speed;
        } // D - rotate right
        if self.keys_held[4] {
            self.camera_distance -= speed;
        } // Q - zoom in
        if self.keys_held[5] {
            self.camera_distance += speed;
        } // E - zoom out

        self.camera_pitch = self.camera_pitch.clamp(-1.4, 1.4);
        self.camera_distance = self.camera_distance.clamp(0.5, 10.0);
    }

    fn rand(&mut self) -> f32 {
        // Simple xorshift
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    fn build_sdf_and_cell_types(&mut self) {
        let cell_count = GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH;
        self.sdf = vec![f32::MAX; cell_count];
        self.cell_types = vec![0u32; cell_count]; // 0 = air, 1 = fluid, 2 = solid

        for k in 0..GRID_DEPTH {
            for j in 0..GRID_HEIGHT {
                for i in 0..GRID_WIDTH {
                    let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;

                    // Sloped floor
                    let t = i as f32 / (GRID_WIDTH - 1) as f32;
                    let floor_j =
                        FLOOR_HEIGHT_LEFT as f32 * (1.0 - t) + FLOOR_HEIGHT_RIGHT as f32 * t;
                    let floor_dist = (j as f32 - floor_j) * CELL_SIZE;

                    // Riffles (small bumps on the floor)
                    let mut riffle_dist = f32::MAX;
                    if i >= RIFFLE_START_X && i < GRID_WIDTH - 5 {
                        let rel_i = i - RIFFLE_START_X;
                        if rel_i % RIFFLE_SPACING < 2 {
                            let riffle_top = floor_j + RIFFLE_HEIGHT as f32;
                            if (j as f32) < riffle_top {
                                riffle_dist = (j as f32 - riffle_top) * CELL_SIZE;
                            }
                        }
                    }

                    // Side walls
                    let wall_dist = ((k as f32).min((GRID_DEPTH - 1 - k) as f32)) * CELL_SIZE;

                    // Combine: negative inside solid
                    let dist = floor_dist.min(riffle_dist).min(wall_dist);
                    self.sdf[idx] = dist;

                    // Mark solid cells
                    if dist < 0.0 {
                        self.cell_types[idx] = 2; // Solid = 2, not 1!
                    }
                }
            }
        }
    }

    fn emit_particles(&mut self) {
        if self.paused {
            return;
        }

        let current_count = self.positions.len();
        if current_count >= MAX_PARTICLES - 500 {
            return;
        }

        // Emission zone: left side, above floor
        let emit_x_min = 2.0 * CELL_SIZE;
        let emit_x_max = 6.0 * CELL_SIZE;
        let emit_y_min = (FLOOR_HEIGHT_LEFT as f32 + 2.0) * CELL_SIZE;
        let emit_y_max = (FLOOR_HEIGHT_LEFT as f32 + 5.0) * CELL_SIZE;
        let emit_z_min = 4.0 * CELL_SIZE;
        let emit_z_max = (GRID_DEPTH as f32 - 4.0) * CELL_SIZE;

        // Emit water
        for _ in 0..self.water_emit_rate {
            let x = emit_x_min + self.rand() * (emit_x_max - emit_x_min);
            let y = emit_y_min + self.rand() * (emit_y_max - emit_y_min);
            let z = emit_z_min + self.rand() * (emit_z_max - emit_z_min);
            self.positions.push(Vec3::new(x, y, z));
            self.velocities.push(Vec3::new(0.3, 0.0, 0.0));
            self.c_matrices.push(Mat3::ZERO);
            self.densities.push(1.0); // Water
        }

        // Emit sediment
        for _ in 0..self.sediment_emit_rate {
            let x = emit_x_min + self.rand() * (emit_x_max - emit_x_min);
            let y = emit_y_min + self.rand() * (emit_y_max - emit_y_min) * 0.5; // Lower
            let z = emit_z_min + self.rand() * (emit_z_max - emit_z_min);
            self.positions.push(Vec3::new(x, y, z));
            self.velocities.push(Vec3::new(0.2, 0.0, 0.0));
            self.c_matrices.push(Mat3::ZERO);
            self.densities.push(SEDIMENT_DENSITY); // Sediment - triggers friction in G2P
        }
    }

    fn remove_escaped_particles(&mut self) {
        let max_x = (GRID_WIDTH as f32 - 2.0) * CELL_SIZE;
        let min_y = 0.5 * CELL_SIZE;

        let mut i = 0;
        while i < self.positions.len() {
            let pos = self.positions[i];
            if pos.x > max_x || pos.y < min_y {
                // Swap remove
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };

        let count = self.positions.len();
        if count == 0 {
            return;
        }

        // Build instance data
        let mut instances: Vec<ParticleInstance> = Vec::with_capacity(count);
        for i in 0..count {
            let pos = self.positions[i];
            let density = self.densities[i];

            // Color: blue for water, brown for sediment
            let color = if density > 1.5 {
                [0.6, 0.4, 0.2, 1.0] // Brown sediment
            } else {
                [0.2, 0.5, 0.9, 0.7] // Blue water
            };

            instances.push(ParticleInstance {
                position: [pos.x, pos.y, pos.z],
                color,
            });
        }

        // Upload instances
        gpu.queue
            .write_buffer(&gpu.instance_buffer, 0, bytemuck::cast_slice(&instances));

        // Camera
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.3,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        let eye = center
            + Vec3::new(
                self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
                self.camera_distance * self.camera_pitch.sin(),
                self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
            );
        let view = Mat4::look_at_rh(eye, center, Vec3::Y);
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            gpu.config.width as f32 / gpu.config.height as f32,
            0.01,
            100.0,
        );
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Render
        let frame = gpu.surface.get_current_texture().unwrap();
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&gpu.render_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
            pass.draw(0..6, 0..instances.len() as u32);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Sluice Friction Test")
                        .with_inner_size(winit::dpi::LogicalSize::new(1200, 800)),
                )
                .unwrap(),
        );
        self.window = Some(window.clone());

        // Initialize GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(
            adapter.request_device(
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
            ),
        )
        .unwrap();

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create render pipeline (simple particle shader)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32x4],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Quad vertices for particle billboards
        let vertices = [
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [1.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0],
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let depth_texture = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Build SDF and cell types
        self.build_sdf_and_cell_types();

        // Create GPU FLIP simulation
        let mut gpu_flip = GpuFlip3D::new(
            &device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        // Configure sediment friction parameters (these are the NEW simplified params)
        gpu_flip.sediment_settling_velocity = 0.05;
        gpu_flip.sediment_friction_threshold = 0.1;
        gpu_flip.sediment_friction_strength = 0.4;
        gpu_flip.sediment_vorticity_lift = 1.5;
        gpu_flip.sediment_vorticity_threshold = 2.0;

        self.gpu_flip = Some(gpu_flip);
        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            config,
            render_pipeline,
            vertex_buffer,
            instance_buffer,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
        });

        println!("=== Sluice Friction Test ===");
        println!("Testing friction-only sediment physics");
        println!("Controls:");
        println!("  WASD: Rotate camera");
        println!("  Q/E: Zoom in/out");
        println!("  Scroll: Zoom");
        println!("  Space: Pause/Resume");
        println!("  Up/Down: Adjust water rate");
        println!("  Left/Right: Adjust sediment rate");
        println!("  Escape: Quit");
        println!("");
        println!("Watch for sediment (brown) to settle behind riffles!");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.keys_held[0] = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.keys_held[1] = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.keys_held[2] = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.keys_held[3] = pressed,
                    PhysicalKey::Code(KeyCode::KeyQ) => self.keys_held[4] = pressed,
                    PhysicalKey::Code(KeyCode::KeyE) => self.keys_held[5] = pressed,
                    _ => {}
                }
                if pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                            println!("Paused: {}", self.paused);
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.water_emit_rate = (self.water_emit_rate + 20).min(300);
                            println!("Water rate: {}", self.water_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.water_emit_rate = self.water_emit_rate.saturating_sub(20);
                            println!("Water rate: {}", self.water_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.sediment_emit_rate = (self.sediment_emit_rate + 10).min(100);
                            println!("Sediment rate: {}", self.sediment_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.sediment_emit_rate = self.sediment_emit_rate.saturating_sub(10);
                            println!("Sediment rate: {}", self.sediment_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 100.0,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.2).clamp(1.0, 10.0);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Track FPS
                if dt > 0.0 {
                    self.fps_history.push(1.0 / dt);
                    if self.fps_history.len() > 60 {
                        self.fps_history.remove(0);
                    }
                }

                // Update camera from WASD/QE
                self.update_camera(dt);

                // Emit new particles
                self.emit_particles();

                // Simulation step
                if !self.paused && self.positions.len() > 0 {
                    // Update cell types from particle positions (critical for pressure solve!)
                    self.update_cell_types();

                    if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                        let sim_dt = 1.0 / 60.0;
                        gpu_flip.step(
                            &gpu.device,
                            &gpu.queue,
                            &mut self.positions,
                            &mut self.velocities,
                            &mut self.c_matrices,
                            &self.densities,
                            &self.cell_types,
                            Some(&self.sdf),
                            None, // No bed heightfield
                            sim_dt,
                            GRAVITY,
                            FLOW_ACCEL,
                            PRESSURE_ITERS,
                        );
                    }
                }

                // Remove escaped particles
                self.remove_escaped_particles();

                self.render();

                // Print stats periodically
                self.frame += 1;
                if self.frame % 120 == 0 {
                    let avg_fps: f32 =
                        self.fps_history.iter().sum::<f32>() / self.fps_history.len().max(1) as f32;
                    let water_count = self.densities.iter().filter(|&&d| d < 1.5).count();
                    let sediment_count = self.densities.iter().filter(|&&d| d >= 1.5).count();
                    println!(
                        "Frame {}: {} total ({} water, {} sediment), {:.1} FPS",
                        self.frame,
                        self.positions.len(),
                        water_count,
                        sediment_count,
                        avg_fps
                    );
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

const PARTICLE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) vertex_pos: vec2<f32>,
    @location(1) instance_pos: vec3<f32>,
    @location(2) instance_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let particle_size = 0.006;

    // Billboard facing camera
    let to_camera = normalize(uniforms.camera_pos - in.instance_pos);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    let up = cross(to_camera, right);

    let world_pos = in.instance_pos
        + right * in.vertex_pos.x * particle_size
        + up * in.vertex_pos.y * particle_size;

    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.instance_color;
    out.uv = in.vertex_pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if (dist > 1.0) {
        discard;
    }
    let alpha = in.color.a * (1.0 - dist * dist);
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
