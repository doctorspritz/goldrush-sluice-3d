//! River Flow Test - Simple Erosion Diagnostics
//!
//! Tests water flow across a sloped terrain to diagnose erosion behavior.
//! Creates a simple channel with continuous water inflow.
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look
//! - 3: Toggle emitter
//! - R: Reset world
//! - ESC: Quit
//!
//! Run: cargo run --example river_test --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use sim3d::{TerrainMaterial, World};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Smaller world for focused testing
const WORLD_WIDTH: usize = 128;
const WORLD_DEPTH: usize = 256;
const CELL_SIZE: f32 = 0.5;
const INITIAL_HEIGHT: f32 = 5.0;

const MOVE_SPEED: f32 = 10.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

const STEPS_PER_FRAME: usize = 10;
const DT: f32 = 0.02;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct WorldVertex {
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

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    vertex_capacity: usize,
    index_capacity: usize,
}

impl Mesh {
    fn new(device: &wgpu::Device, vertices: &[WorldVertex], indices: &[u32], label: &str) -> Self {
        let (vertex_buffer, vertex_capacity) = Self::create_vertex_buffer(device, vertices, label);
        let (index_buffer, index_capacity) = Self::create_index_buffer(device, indices, label);

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            vertex_capacity,
            index_capacity,
        }
    }

    fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[WorldVertex],
        indices: &[u32],
        label: &str,
    ) {
        if !vertices.is_empty() {
            if vertices.len() > self.vertex_capacity {
                let (buffer, capacity) = Self::create_vertex_buffer(device, vertices, label);
                self.vertex_buffer = buffer;
                self.vertex_capacity = capacity;
            } else {
                queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
            }
        }

        if !indices.is_empty() {
            if indices.len() > self.index_capacity {
                let (buffer, capacity) = Self::create_index_buffer(device, indices, label);
                self.index_buffer = buffer;
                self.index_capacity = capacity;
            } else {
                queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
            }
        }

        self.num_indices = indices.len() as u32;
    }

    fn create_vertex_buffer(
        device: &wgpu::Device,
        vertices: &[WorldVertex],
        label: &str,
    ) -> (wgpu::Buffer, usize) {
        let capacity = vertices.len().max(1).next_power_of_two();
        let data = if vertices.is_empty() {
            vec![WorldVertex::default(); 1]
        } else {
            vertices.to_vec()
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", label)),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, capacity)
    }

    fn create_index_buffer(
        device: &wgpu::Device,
        indices: &[u32],
        label: &str,
    ) -> (wgpu::Buffer, usize) {
        let capacity = indices.len().max(1).next_power_of_two();
        let data: Vec<u32> = if indices.is_empty() {
            vec![0]
        } else {
            indices.to_vec()
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", label)),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, capacity)
    }
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
    scroll_delta: f32,
}

struct WaterEmitter {
    position: Vec3,
    rate: f32,
    radius: f32,
    enabled: bool,
}

use game::gpu::heightfield::GpuHeightfield;

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    heightfield: GpuHeightfield,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    world: World,
    emitter: WaterEmitter,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    last_stats: Instant,
    start_time: Instant,
    window_size: (u32, u32),
    terrain_dirty: bool,
    frame_count: u32,
    initial_terrain_height: f32, // Track initial height for erosion measurement
}

impl App {
    fn new() -> Self {
        let world = build_river_world();

        // Measure initial terrain height at river center
        let center_x = WORLD_WIDTH / 2;
        let measure_z = WORLD_DEPTH / 2;
        let initial_height = world.ground_height(center_x, measure_z);

        Self {
            window: None,
            gpu: None,
            world,
            emitter: WaterEmitter {
                position: Vec3::new(
                    (WORLD_WIDTH as f32 * CELL_SIZE) / 2.0,
                    10.0,
                    10.0, // Near top of the slope
                ),
                rate: 20.0, // Moderate flow
                radius: 3.0,
                enabled: true,
            },
            camera: Camera {
                position: Vec3::new(
                    (WORLD_WIDTH as f32 * CELL_SIZE) / 2.0,
                    40.0,
                    (WORLD_DEPTH as f32 * CELL_SIZE) / 2.0,
                ),
                yaw: -std::f32::consts::FRAC_PI_2,
                pitch: -0.5,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                last_mouse_pos: None,
                scroll_delta: 0.0,
            },
            last_frame: Instant::now(),
            last_stats: Instant::now(),
            start_time: Instant::now(),
            window_size: (1280, 720),
            terrain_dirty: true,
            frame_count: 0,
            initial_terrain_height: initial_height,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.update_camera(dt);
        self.frame_count += 1;

        if let Some(gpu) = &mut self.gpu {
            let sim_dt = DT;

            for _ in 0..STEPS_PER_FRAME {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sim Encoder"),
                        });

                gpu.heightfield.update_emitter(
                    &gpu.queue,
                    self.emitter.position.x,
                    self.emitter.position.z,
                    self.emitter.radius,
                    self.emitter.rate,
                    sim_dt,
                    self.emitter.enabled,
                );
                gpu.heightfield.dispatch_emitter(&mut encoder);

                gpu.heightfield.update_params(&gpu.queue, sim_dt);
                gpu.heightfield.dispatch(&mut encoder, sim_dt);

                gpu.queue.submit(Some(encoder.finish()));
            }

            pollster::block_on(gpu.heightfield.download_to_world(
                &gpu.device,
                &gpu.queue,
                &mut self.world,
            ));
        }

        // Print stats every second
        if self.last_stats.elapsed() > Duration::from_secs(1) {
            let water = self.world.total_water_volume();
            let sediment = self.world.total_sediment_volume();

            // Measure current terrain height at river center
            let center_x = WORLD_WIDTH / 2;
            let measure_z = WORLD_DEPTH / 2;
            let current_height = self.world.ground_height(center_x, measure_z);
            let erosion = self.initial_terrain_height - current_height;

            let elapsed = self.start_time.elapsed().as_secs_f32();

            println!(
                "t={:.1}s | Water: {:.2}m³ | Sediment: {:.4}m³ | Erosion@center: {:.4}m | Emitter: {}",
                elapsed, water, sediment, erosion,
                if self.emitter.enabled { "ON" } else { "OFF" }
            );
            self.last_stats = Instant::now();
        }
    }

    fn update_camera(&mut self, dt: f32) {
        let mut direction = Vec3::ZERO;

        if self.input.keys.contains(&KeyCode::KeyW) {
            direction += self.camera.forward_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyS) {
            direction -= self.camera.forward_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyA) {
            direction -= self.camera.right_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyD) {
            direction += self.camera.right_flat();
        }
        if self.input.keys.contains(&KeyCode::Space) {
            direction.y += 1.0;
        }
        if self.input.keys.contains(&KeyCode::ShiftLeft) {
            direction.y -= 1.0;
        }

        if direction.length_squared() > 0.0 {
            self.camera.position += direction.normalize() * self.camera.speed * dt;
        }

        if self.input.scroll_delta != 0.0 {
            let forward = self.camera.forward();
            self.camera.position += forward * self.input.scroll_delta * 2.0;
            self.input.scroll_delta = 0.0;
        }
    }
}

/// Build a simple sloped river channel world
fn build_river_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, INITIAL_HEIGHT);

    let center_x = WORLD_WIDTH / 2;
    let channel_half_width = 8; // 8 cells = 4m wide channel

    // Create sloped terrain with a central channel
    for z in 0..WORLD_DEPTH {
        for x in 0..WORLD_WIDTH {
            let idx = world.idx(x, z);

            // Slope: drops 10m over the length (z direction)
            let slope_factor = z as f32 / WORLD_DEPTH as f32;
            let base_height = INITIAL_HEIGHT + 10.0 * (1.0 - slope_factor);

            // Channel: lower in the middle
            let dist_from_center = (x as i32 - center_x as i32).abs() as f32;
            let in_channel = dist_from_center < channel_half_width as f32;

            let channel_depth = if in_channel {
                // Parabolic channel profile
                let normalized = dist_from_center / channel_half_width as f32;
                2.0 * (1.0 - normalized * normalized)
            } else {
                0.0
            };

            // Set layer thicknesses
            world.bedrock_elevation[idx] = base_height - channel_depth - 3.0;
            world.paydirt_thickness[idx] = 0.5;
            world.gravel_thickness[idx] = 0.5;
            world.overburden_thickness[idx] = 2.0; // Main erodible layer
            world.terrain_sediment[idx] = 0.0;
        }
    }

    println!("=== River Flow Test ===");
    println!(
        "World: {}x{} cells ({:.1}m x {:.1}m)",
        WORLD_WIDTH,
        WORLD_DEPTH,
        WORLD_WIDTH as f32 * CELL_SIZE,
        WORLD_DEPTH as f32 * CELL_SIZE
    );
    println!(
        "Channel: {}m wide, 10m slope",
        channel_half_width as f32 * 2.0 * CELL_SIZE
    );
    println!("Controls: WASD=move, Right-click=look, 3=toggle emitter, ESC=quit");
    println!();

    world
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("River Flow Test - Erosion Diagnostics")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.window_size.0,
                self.window_size.1,
            ));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(async {
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
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
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

            // Create shader
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Terrain Shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

            // Uniforms
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Uniform Layout"),
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

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Uniform Bind Group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
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
                        array_stride: std::mem::size_of::<WorldVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
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
                multiview: None,
                cache: None,
            });

            let (depth_texture, depth_view) =
                create_depth_texture(&device, size.width, size.height);

            let heightfield = GpuHeightfield::new(
                &device,
                self.world.width as u32,
                self.world.depth as u32,
                self.world.cell_size,
                INITIAL_HEIGHT,
                config.format,
            );
            heightfield.upload_from_world(&queue, &self.world);

            GpuState {
                surface,
                device,
                queue,
                config,
                pipeline,
                uniform_buffer,
                bind_group,
                depth_texture,
                depth_view,
                heightfield,
            }
        });

        self.gpu = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    let (tex, view) = create_depth_texture(&gpu.device, size.width, size.height);
                    gpu.depth_texture = tex;
                    gpu.depth_view = view;
                    self.window_size = (size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.input.keys.insert(key);
                        match key {
                            KeyCode::Escape => event_loop.exit(),
                            KeyCode::Digit3 => {
                                self.emitter.enabled = !self.emitter.enabled;
                                println!(
                                    "Emitter: {}",
                                    if self.emitter.enabled { "ON" } else { "OFF" }
                                );
                            }
                            KeyCode::KeyR => {
                                self.world = build_river_world();
                                if let Some(gpu) = &mut self.gpu {
                                    gpu.heightfield = GpuHeightfield::new(
                                        &gpu.device,
                                        self.world.width as u32,
                                        self.world.depth as u32,
                                        self.world.cell_size,
                                        INITIAL_HEIGHT,
                                        gpu.config.format,
                                    );
                                    gpu.heightfield.upload_from_world(&gpu.queue, &self.world);
                                }
                                self.terrain_dirty = true;
                                self.start_time = Instant::now();
                                self.initial_terrain_height =
                                    self.world.ground_height(WORLD_WIDTH / 2, WORLD_DEPTH / 2);
                                println!("World reset!");
                            }
                            _ => {}
                        }
                    } else {
                        self.input.keys.remove(&key);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.input.mouse_look = state == ElementState::Pressed;
                    if !self.input.mouse_look {
                        self.input.last_mouse_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.input.mouse_look {
                    if let Some((last_x, last_y)) = self.input.last_mouse_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.camera.yaw += dx * self.camera.sensitivity;
                        self.camera.pitch =
                            (self.camera.pitch - dy * self.camera.sensitivity).clamp(-1.5, 1.5);
                    }
                    self.input.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.input.scroll_delta += scroll;
            }
            WindowEvent::RedrawRequested => {
                self.update();

                if let (Some(window), Some(gpu)) = (self.window.as_ref(), self.gpu.as_mut()) {
                    let frame = match gpu.surface.get_current_texture() {
                        Ok(f) => f,
                        Err(_) => return,
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // Generate terrain mesh
                    let (terrain_verts, terrain_indices) = generate_terrain_mesh(&self.world);
                    let (water_verts, water_indices) = generate_water_mesh(&self.world);

                    // Update uniforms
                    let aspect = gpu.config.width as f32 / gpu.config.height as f32;
                    let proj =
                        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
                    let view_mat = self.camera.view_matrix();
                    let uniforms = Uniforms {
                        view_proj: (proj * view_mat).to_cols_array_2d(),
                        camera_pos: self.camera.position.to_array(),
                        _pad: 0.0,
                    };
                    gpu.queue
                        .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                    // Create temporary buffers
                    let terrain_vb =
                        gpu.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Terrain VB"),
                                contents: bytemuck::cast_slice(&terrain_verts),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let terrain_ib =
                        gpu.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Terrain IB"),
                                contents: bytemuck::cast_slice(&terrain_indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                    let water_vb =
                        gpu.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Water VB"),
                                contents: bytemuck::cast_slice(&water_verts),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let water_ib =
                        gpu.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Water IB"),
                                contents: bytemuck::cast_slice(&water_indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });

                    let mut encoder =
                        gpu.device
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
                                        r: 0.4,
                                        g: 0.6,
                                        b: 0.9,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &gpu.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            ..Default::default()
                        });

                        pass.set_pipeline(&gpu.pipeline);
                        pass.set_bind_group(0, &gpu.bind_group, &[]);

                        // Draw terrain
                        if !terrain_indices.is_empty() {
                            pass.set_vertex_buffer(0, terrain_vb.slice(..));
                            pass.set_index_buffer(terrain_ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..terrain_indices.len() as u32, 0, 0..1);
                        }

                        // Draw water
                        if !water_indices.is_empty() {
                            pass.set_vertex_buffer(0, water_vb.slice(..));
                            pass.set_index_buffer(water_ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..water_indices.len() as u32, 0, 0..1);
                        }
                    }

                    gpu.queue.submit(Some(encoder.finish()));
                    frame.present();

                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn generate_terrain_mesh(world: &World) -> (Vec<WorldVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..world.depth {
        for x in 0..world.width {
            let height = world.ground_height(x, z);
            let idx = world.idx(x, z);
            let sediment = world.terrain_sediment[idx];
            let sediment_ratio = (sediment / 2.0).min(1.0);

            // Color based on surface material
            let base_color = match world.surface_material(x, z) {
                TerrainMaterial::Dirt => [0.4, 0.3, 0.2],
                TerrainMaterial::Gravel => [0.6, 0.5, 0.2],
                TerrainMaterial::Sand => [0.8, 0.7, 0.5],
                TerrainMaterial::Clay => [0.6, 0.4, 0.3],
                TerrainMaterial::Bedrock => [0.2, 0.2, 0.25],
            };

            let sediment_color = [0.6, 0.5, 0.4];
            let color = [
                base_color[0] * (1.0 - sediment_ratio) + sediment_color[0] * sediment_ratio,
                base_color[1] * (1.0 - sediment_ratio) + sediment_color[1] * sediment_ratio,
                base_color[2] * (1.0 - sediment_ratio) + sediment_color[2] * sediment_ratio,
                1.0,
            ];

            let x0 = x as f32 * world.cell_size;
            let x1 = (x + 1) as f32 * world.cell_size;
            let z0 = z as f32 * world.cell_size;
            let z1 = (z + 1) as f32 * world.cell_size;

            let base = vertices.len() as u32;
            vertices.push(WorldVertex {
                position: [x0, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z1],
                color,
            });
            vertices.push(WorldVertex {
                position: [x0, height, z1],
                color,
            });

            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    (vertices, indices)
}

fn generate_water_mesh(world: &World) -> (Vec<WorldVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..world.depth {
        for x in 0..world.width {
            let depth = world.water_depth(x, z);
            if depth < 0.001 {
                continue;
            }

            let idx = world.idx(x, z);
            let height = world.water_surface[idx];
            let turbidity = world.suspended_sediment[idx];

            let alpha = 0.5 + (depth).min(0.3);
            let brown = turbidity.min(0.5) * 2.0;
            let color = [
                0.2 + brown * 0.4,
                0.4 + brown * 0.2,
                0.8 - brown * 0.4,
                alpha,
            ];

            let x0 = x as f32 * world.cell_size;
            let x1 = (x + 1) as f32 * world.cell_size;
            let z0 = z as f32 * world.cell_size;
            let z1 = (z + 1) as f32 * world.cell_size;

            let base = vertices.len() as u32;
            vertices.push(WorldVertex {
                position: [x0, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z1],
                color,
            });
            vertices.push(WorldVertex {
                position: [x0, height, z1],
                color,
            });

            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    (vertices, indices)
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

const SHADER_SRC: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
