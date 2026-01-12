//! Washplant Visual - Multi-Stage Processing Plant
//!
//! Renders the full 4-stage washplant (Hopper → Grizzly → Shaker → Sluice)
//! with particle transfers between stages.
//!
//! Run with: cargo run --example washplant_visual --release
//!
//! Controls:
//!   Mouse drag     - Rotate camera
//!   Scroll         - Zoom in/out
//!   WASD           - Move camera horizontally
//!   Shift/Ctrl     - Move camera up/down
//!   SPACE          - Pause/unpause
//!   0-4            - Focus camera (0=overview, 1-4=stage)
//!   E              - Spawn water at hopper inlet
//!   R              - Reset plant

use bytemuck::{Pod, Zeroable};
use game::sluice_geometry::SluiceVertex;
use game::washplant::{PlantConfig, Washplant};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const MAX_PARTICLES: usize = 500_000;
const DT: f32 = 1.0 / 60.0;

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

// Stage colors for visual distinction
const STAGE_COLORS: [[f32; 3]; 4] = [
    [0.8, 0.6, 0.2], // Hopper: gold/brown
    [0.6, 0.7, 0.8], // Grizzly: steel blue
    [0.5, 0.8, 0.5], // Shaker: green
    [0.3, 0.6, 0.9], // Sluice: blue
];

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    plant: Washplant,

    // Camera state
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vec3,

    // Input state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    keys_pressed: KeysPressed,

    frame_count: u64,
}

#[derive(Default)]
struct KeysPressed {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    shift: bool,
    ctrl: bool,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Particle rendering
    particle_pipeline: wgpu::RenderPipeline,
    particle_vertex_buffer: wgpu::Buffer,
    particle_instance_buffer: wgpu::Buffer,

    // Mesh rendering
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_vertex_buffers: Vec<wgpu::Buffer>,
    mesh_index_buffers: Vec<wgpu::Buffer>,
    mesh_index_counts: Vec<u32>,

    // Shared
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    // Depth buffer
    depth_texture: wgpu::TextureView,
}

impl App {
    fn new() -> Self {
        let plant = Washplant::new(PlantConfig::default());

        // Calculate initial camera position to see all stages
        let (eye, center) = plant.camera_params();
        let to_eye = eye - center;
        let distance = to_eye.length();
        let yaw = to_eye.z.atan2(to_eye.x);
        let pitch = (to_eye.y / distance).asin();

        println!("=== Washplant Visual ===");
        println!("Stages: {}", plant.stages.len());
        for (i, stage) in plant.stages.iter().enumerate() {
            let (w, h, d) = stage.grid_size();
            println!(
                "  {}: {} ({}x{}x{} @ {:.3}m)",
                i + 1,
                stage.config.name,
                w,
                h,
                d,
                stage.cell_size()
            );
        }
        println!();
        println!("Controls:");
        println!("  Mouse drag     - Rotate camera");
        println!("  Scroll         - Zoom in/out");
        println!("  WASD           - Move camera horizontally");
        println!("  Shift/Ctrl     - Move camera up/down");
        println!("  SPACE          - Pause/unpause");
        println!("  0-4            - Focus on stage");
        println!("  E              - Spawn water");
        println!("  R              - Reset plant");

        Self {
            window: None,
            gpu: None,
            plant,
            camera_yaw: yaw,
            camera_pitch: pitch.clamp(-1.4, 1.4),
            camera_distance: distance,
            camera_target: center,
            mouse_pressed: false,
            last_mouse_pos: None,
            keys_pressed: KeysPressed::default(),
            frame_count: 0,
        }
    }

    fn reset_plant(&mut self) {
        self.plant = Washplant::new(PlantConfig::default());
        println!("Plant reset");
    }

    fn update_camera_from_keys(&mut self, dt: f32) {
        let speed = 5.0 * dt;

        // Calculate forward/right vectors on XZ plane
        let forward = Vec3::new(-self.camera_yaw.sin(), 0.0, -self.camera_yaw.cos()).normalize();
        let right = Vec3::new(forward.z, 0.0, -forward.x);

        let mut movement = Vec3::ZERO;

        if self.keys_pressed.w {
            movement += forward;
        }
        if self.keys_pressed.s {
            movement -= forward;
        }
        if self.keys_pressed.a {
            movement -= right;
        }
        if self.keys_pressed.d {
            movement += right;
        }
        if self.keys_pressed.shift {
            movement.y += 1.0;
        }
        if self.keys_pressed.ctrl {
            movement.y -= 1.0;
        }

        if movement.length_squared() > 0.0 {
            self.camera_target += movement.normalize() * speed;
        }
    }

    fn camera_position(&self) -> Vec3 {
        self.camera_target
            + Vec3::new(
                self.camera_distance * self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_distance * self.camera_pitch.sin(),
                self.camera_distance * self.camera_yaw.sin() * self.camera_pitch.cos(),
            )
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
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
            )
            .await
            .unwrap();

        // Initialize GPU backends for all stages
        self.plant.init_gpu(&device, &queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create depth texture
        let depth_texture = Self::create_depth_texture(&device, size.width, size.height);

        // Shaders
        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
        });

        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
        });

        // Vertex buffer (quad for particles)
        let vertices = [
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [1.0, -1.0] },
            Vertex { position: [1.0, 1.0] },
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [1.0, 1.0] },
            Vertex { position: [-1.0, 1.0] },
        ];
        let particle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Instance buffer for particles
        let particle_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Particle pipeline
        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
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
                module: &particle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
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

        // Mesh pipeline
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SluiceVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Draw both sides
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

        // Create mesh buffers for each stage
        let mut mesh_vertex_buffers = Vec::new();
        let mut mesh_index_buffers = Vec::new();
        let mut mesh_index_counts = Vec::new();

        for stage in &self.plant.stages {
            let vertices = &stage.vertices;
            let indices = &stage.indices;

            if vertices.is_empty() || indices.is_empty() {
                // Empty placeholder buffers
                mesh_vertex_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Empty Vertex Buffer"),
                    size: 64,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                }));
                mesh_index_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Empty Index Buffer"),
                    size: 64,
                    usage: wgpu::BufferUsages::INDEX,
                    mapped_at_creation: false,
                }));
                mesh_index_counts.push(0);
            } else {
                // Apply world offset to vertices
                let offset = stage.world_offset;
                let transformed_vertices: Vec<SluiceVertex> = vertices
                    .iter()
                    .map(|v| SluiceVertex {
                        position: [
                            v.position[0] + offset.x,
                            v.position[1] + offset.y,
                            v.position[2] + offset.z,
                        ],
                        color: v.color,
                    })
                    .collect();

                mesh_vertex_buffers.push(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Mesh Vertex Buffer"),
                        contents: bytemuck::cast_slice(&transformed_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
                mesh_index_buffers.push(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Mesh Index Buffer"),
                        contents: bytemuck::cast_slice(indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ));
                mesh_index_counts.push(indices.len() as u32);
            }
        }

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            particle_pipeline,
            particle_vertex_buffer,
            particle_instance_buffer,
            mesh_pipeline,
            mesh_vertex_buffers,
            mesh_index_buffers,
            mesh_index_counts,
            uniform_buffer,
            bind_group,
            depth_texture,
        });
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&Default::default())
    }

    fn collect_particle_instances(&self) -> Vec<ParticleInstance> {
        let mut instances = Vec::with_capacity(MAX_PARTICLES);

        for (stage_idx, stage) in self.plant.stages.iter().enumerate() {
            let base_color = STAGE_COLORS.get(stage_idx).copied().unwrap_or([0.5, 0.5, 0.5]);
            let world_offset = stage.world_offset;

            for particle in &stage.sim.particles.list {
                if instances.len() >= MAX_PARTICLES {
                    break;
                }

                let world_pos = particle.position + world_offset;

                let color = if particle.is_sediment() {
                    [base_color[0] * 0.6, base_color[1] * 0.4, base_color[2] * 0.3, 0.95]
                } else {
                    let speed = particle.velocity.length();
                    let t = (speed / 3.0).min(1.0);
                    [
                        base_color[0] * (0.7 + t * 0.3),
                        base_color[1] * (0.7 + t * 0.3),
                        base_color[2] * (0.8 + t * 0.2),
                        0.75,
                    ]
                };

                instances.push(ParticleInstance {
                    position: world_pos.to_array(),
                    color,
                });
            }
        }

        instances
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        self.frame_count += 1;

        // Update camera from keys
        self.update_camera_from_keys(DT);

        // Update simulation (borrow gpu temporarily)
        if !self.plant.paused {
            let gpu = self.gpu.as_ref().unwrap();
            self.plant.tick(DT, Some(&gpu.device), Some(&gpu.queue));

            // Auto-spawn water periodically
            if self.frame_count % 10 == 0 {
                self.plant.spawn_inlet_water(5);
            }
        }

        // Print status every 60 frames
        if self.frame_count % 60 == 0 {
            println!("{}", self.plant.status_string());
        }

        // Collect particles before borrowing gpu
        let instances = self.collect_particle_instances();

        // Camera calculations
        let camera_pos = self.camera_position();
        let view = Mat4::look_at_rh(camera_pos, self.camera_target, Vec3::Y);

        // Now borrow gpu and window for rendering
        let gpu = self.gpu.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();

        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 500.0);
        let view_proj = proj * view;

        // Update uniforms
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Upload particle instances
        if !instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.particle_instance_buffer,
                0,
                bytemuck::cast_slice(&instances),
            );
        }

        // Render
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
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
                ..Default::default()
            });

            // Draw meshes first (opaque)
            pass.set_pipeline(&gpu.mesh_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            for i in 0..gpu.mesh_vertex_buffers.len() {
                if gpu.mesh_index_counts[i] > 0 {
                    pass.set_vertex_buffer(0, gpu.mesh_vertex_buffers[i].slice(..));
                    pass.set_index_buffer(
                        gpu.mesh_index_buffers[i].slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(0..gpu.mesh_index_counts[i], 0, 0..1);
                }
            }

            // Draw particles (transparent)
            pass.set_pipeline(&gpu.particle_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.particle_vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.particle_instance_buffer.slice(..));
            pass.draw(0..6, 0..instances.len() as u32);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Washplant - Multi-Stage FLIP Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        self.camera_yaw += dx as f32 * 0.005;
                        self.camera_pitch = (self.camera_pitch + dy as f32 * 0.005).clamp(-1.4, 1.4);
                    }
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.5).clamp(0.5, 50.0);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.keys_pressed.w = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.keys_pressed.a = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.keys_pressed.s = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.keys_pressed.d = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        self.keys_pressed.shift = pressed
                    }
                    PhysicalKey::Code(KeyCode::ControlLeft | KeyCode::ControlRight) => {
                        self.keys_pressed.ctrl = pressed
                    }
                    _ => {}
                }

                if event.state.is_pressed() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.plant.toggle_pause();
                            println!("Paused: {}", self.plant.paused);
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            self.plant.spawn_inlet_water(50);
                            println!("Spawned 50 water particles at hopper");
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => self.reset_plant(),
                        PhysicalKey::Code(KeyCode::Digit0) => {
                            let (eye, center) = self.plant.camera_params();
                            self.camera_target = center;
                            let to_eye = eye - center;
                            self.camera_distance = to_eye.length();
                            self.camera_yaw = to_eye.z.atan2(to_eye.x);
                            self.camera_pitch = (to_eye.y / self.camera_distance).asin();
                            println!("Camera: Overview");
                        }
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            self.plant.focus_stage(1);
                            let (eye, center) = self.plant.camera_params();
                            self.camera_target = center;
                            let to_eye = eye - center;
                            self.camera_distance = to_eye.length();
                            println!("Camera: Stage 1 (Hopper)");
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            self.plant.focus_stage(2);
                            let (eye, center) = self.plant.camera_params();
                            self.camera_target = center;
                            let to_eye = eye - center;
                            self.camera_distance = to_eye.length();
                            println!("Camera: Stage 2 (Grizzly)");
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            self.plant.focus_stage(3);
                            let (eye, center) = self.plant.camera_params();
                            self.camera_target = center;
                            let to_eye = eye - center;
                            self.camera_distance = to_eye.length();
                            println!("Camera: Stage 3 (Shaker)");
                        }
                        PhysicalKey::Code(KeyCode::Digit4) => {
                            self.plant.focus_stage(4);
                            let (eye, center) = self.plant.camera_params();
                            self.camera_target = center;
                            let to_eye = eye - center;
                            self.camera_distance = to_eye.length();
                            println!("Camera: Stage 4 (Sluice)");
                        }
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.depth_texture =
                        Self::create_depth_texture(&gpu.device, size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

const PARTICLE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) vertex: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let size = 0.015;

    let to_camera = normalize(uniforms.camera_pos - in.position);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, to_camera));
    let up = cross(to_camera, right);

    let world_pos = in.position + right * in.vertex.x * size + up * in.vertex.y * size;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
    out.uv = in.vertex * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2<f32>(0.5));
    let alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

const MESH_SHADER: &str = r#"
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
