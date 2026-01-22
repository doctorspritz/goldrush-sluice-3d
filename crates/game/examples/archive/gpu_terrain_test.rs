//! GPU Vertex Displacement Terrain Rendering Prototype
//!
//! Renders heightfield terrain using a static grid mesh with vertex displacement
//! in the vertex shader. Heights are sampled from GPU buffer - zero CPU mesh work.
//!
//! Run: cargo run --example gpu_terrain_test --release

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use sim3d::{generate_klondike_terrain, TerrainConfig, World};

// Constants
const WORLD_WIDTH: usize = 512;
const WORLD_DEPTH: usize = 512;
const CELL_SIZE: f32 = 1.0;
const MOVE_SPEED: f32 = 30.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

/// Vertex for the terrain grid (just position, no height - that's in the shader)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridVertex {
    position: [f32; 2], // X, Z grid position
}

/// Uniforms passed to shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    cell_size: f32,
    grid_width: u32,
    grid_depth: u32,
    time: f32,
    _pad: u32,
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.cos() * cos_pitch,
            self.pitch.sin(),
            self.yaw.sin() * cos_pitch,
        )
        .normalize()
    }

    fn forward_flat(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    fn right_flat(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }
}

struct InputState {
    keys: HashSet<KeyCode>,
    mouse_look: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Terrain rendering
    pipeline: wgpu::RenderPipeline,
    grid_vertex_buffer: wgpu::Buffer,
    grid_index_buffer: wgpu::Buffer,
    num_indices: u32,
    // Geolgoy buffers (bedrock, paydirt, gravel, overburden, sediment)
    // Geolgoy buffers (bedrock, paydirt, gravel, overburden, sediment)
    bedrock_buffer: wgpu::Buffer,
    paydirt_buffer: wgpu::Buffer,
    gravel_buffer: wgpu::Buffer,
    overburden_buffer: wgpu::Buffer,
    sediment_buffer: wgpu::Buffer,
    water_buffer: wgpu::Buffer,

    // Water rendering pipeline
    water_pipeline: wgpu::RenderPipeline,

    // Uniforms
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    // Depth buffer
    depth_texture: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    world: World,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    start_time: Instant,
    window_size: (u32, u32),
}

impl App {
    fn new() -> Self {
        let config = TerrainConfig::default();
        let world = generate_klondike_terrain(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, &config);

        Self {
            window: None,
            gpu: None,
            world,
            camera: Camera {
                position: Vec3::new(256.0, 100.0, 256.0),
                yaw: -1.57,
                pitch: -0.4,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                last_mouse_pos: None,
            },
            last_frame: Instant::now(),
            start_time: Instant::now(),
            window_size: (1280, 720),
        }
    }

    fn update(&mut self, dt: f32) {
        self.update_camera(dt);
        // Buffers are static in this prototype (no simulation running)
    }

    fn update_camera(&mut self, dt: f32) {
        let speed = self.camera.speed * dt;
        let forward = self.camera.forward_flat();
        let right = self.camera.right_flat();

        if self.input.keys.contains(&KeyCode::KeyW) {
            self.camera.position += forward * speed;
        }
        if self.input.keys.contains(&KeyCode::KeyS) {
            self.camera.position -= forward * speed;
        }
        if self.input.keys.contains(&KeyCode::KeyA) {
            self.camera.position -= right * speed;
        }
        if self.input.keys.contains(&KeyCode::KeyD) {
            self.camera.position += right * speed;
        }
        if self.input.keys.contains(&KeyCode::Space) {
            self.camera.position.y += speed;
        }
        if self.input.keys.contains(&KeyCode::ShiftLeft)
            || self.input.keys.contains(&KeyCode::ShiftRight)
        {
            self.camera.position.y -= speed;
        }
    }

    fn render(&mut self) {
        let Some(gpu) = self.gpu.as_ref() else { return };

        let aspect = self.window_size.0 as f32 / self.window_size.1 as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
        let view = self.camera.view_matrix();
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            cell_size: CELL_SIZE,
            grid_width: WORLD_WIDTH as u32,
            grid_depth: WORLD_DEPTH as u32,
            time: self.start_time.elapsed().as_secs_f32(),
            _pad: 0,
        };

        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let output = match gpu.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(_) => return,
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.7,
                            b: 0.9,
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

            render_pass.set_pipeline(&gpu.pipeline);
            render_pass.set_bind_group(0, &gpu.bind_group, &[]);
            render_pass.set_vertex_buffer(0, gpu.grid_vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(gpu.grid_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..gpu.num_indices, 0, 0..1);

            // Draw Water
            render_pass.set_pipeline(&gpu.water_pipeline);
            render_pass.draw_indexed(0..gpu.num_indices, 0, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("GPU Terrain Prototype")
                        .with_inner_size(PhysicalSize::new(1280, 720)),
                )
                .unwrap(),
        );

        self.window = Some(window.clone());

        // Initialize GPU
        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.window_size = (size.width.max(1), size.height.max(1));
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.depth_texture =
                        create_depth_texture(&gpu.device, size.width.max(1), size.height.max(1));
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = self.last_frame.elapsed().as_secs_f32();
                self.last_frame = Instant::now();
                self.update(dt);
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.input.mouse_look = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if self.input.mouse_look {
                    if let Some(last) = self.input.last_mouse_pos {
                        let dx = (current.0 - last.0) as f32;
                        let dy = (current.1 - last.1) as f32;
                        self.camera.yaw += dx * self.camera.sensitivity;
                        self.camera.pitch =
                            (self.camera.pitch - dy * self.camera.sensitivity).clamp(-1.5, 1.5);
                    }
                }
                self.input.last_mouse_pos = Some(current);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.input.keys.insert(key);
                            if key == KeyCode::Escape {
                                event_loop.exit();
                            }
                        }
                        ElementState::Released => {
                            self.input.keys.remove(&key);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl App {
    async fn init_gpu(&mut self, window: Arc<Window>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .unwrap();
        surface.configure(&device, &config);

        // Create static grid mesh (vertices are just X,Z positions)
        let (grid_vertices, grid_indices) = create_grid_mesh(WORLD_WIDTH, WORLD_DEPTH);
        let num_indices = grid_indices.len() as u32;

        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let grid_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Index Buffer"),
            contents: bytemuck::cast_slice(&grid_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create buffers for all 5 layers
        let buffer_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let bedrock_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bedrock Buffer"),
            contents: bytemuck::cast_slice(&self.world.bedrock_elevation),
            usage: buffer_usage,
        });

        let paydirt_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Paydirt Buffer"),
            contents: bytemuck::cast_slice(&self.world.paydirt_thickness),
            usage: buffer_usage,
        });

        let gravel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gravel Buffer"),
            contents: bytemuck::cast_slice(&self.world.gravel_thickness),
            usage: buffer_usage,
        });

        let overburden_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Overburden Buffer"),
            contents: bytemuck::cast_slice(&self.world.overburden_thickness),
            usage: buffer_usage,
        });

        let sediment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sediment Buffer"),
            contents: bytemuck::cast_slice(&self.world.terrain_sediment),
            usage: buffer_usage,
        });

        let water_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Buffer"),
            contents: bytemuck::cast_slice(&self.world.water_surface),
            usage: buffer_usage,
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout (1 uniform + 5 storage)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bedrock_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: paydirt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gravel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: overburden_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sediment_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: water_buffer.as_entire_binding(),
                },
            ],
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/gpu/shaders/heightfield_render.wgsl").into(),
            ),
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GridVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
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

        // Water Pipeline
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_water"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GridVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_water"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Transparent water doesn't write depth
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let depth_texture = create_depth_texture(&device, size.width.max(1), size.height.max(1));

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            water_pipeline,
            grid_vertex_buffer,
            grid_index_buffer,
            num_indices,
            bedrock_buffer,
            paydirt_buffer,
            gravel_buffer,
            overburden_buffer,
            sediment_buffer,
            water_buffer,
            uniform_buffer,
            bind_group,
            depth_texture,
        });
    }
}

fn create_grid_mesh(width: usize, depth: usize) -> (Vec<GridVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(width * depth);
    let mut indices = Vec::with_capacity((width - 1) * (depth - 1) * 6);

    // Create vertices (just X,Z positions)
    for z in 0..depth {
        for x in 0..width {
            vertices.push(GridVertex {
                position: [x as f32, z as f32],
            });
        }
    }

    // Create indices (two triangles per cell)
    for z in 0..(depth - 1) {
        for x in 0..(width - 1) {
            let tl = (z * width + x) as u32;
            let tr = tl + 1;
            let bl = tl + width as u32;
            let br = bl + 1;

            // Two triangles
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    (vertices, indices)
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
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
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

use wgpu::util::DeviceExt;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
