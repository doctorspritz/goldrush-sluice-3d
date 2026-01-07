//! World Heightfield Test
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look
//! - Left Mouse: Dig terrain
//! - Ctrl + Left Mouse: Add material
//! - 1: Add water at cursor
//! - 2: Add muddy water at cursor
//! - R: Reset world
//! - ESC: Quit
//!
//! Run: cargo run --example world_test --release

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

const WORLD_WIDTH: usize = 256;
const WORLD_DEPTH: usize = 256;
const CELL_SIZE: f32 = 1.0;
const INITIAL_HEIGHT: f32 = 10.0;

const DIG_RADIUS: f32 = 3.0;
const DIG_DEPTH: f32 = 0.5;
const ADD_RADIUS: f32 = 3.0;
const ADD_HEIGHT: f32 = 0.5;
const WATER_ADD_VOLUME: f32 = 5.0;

const MOVE_SPEED: f32 = 20.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

const STEPS_PER_FRAME: usize = 10;
const DT: f32 = 0.005;

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
        let capacity = vertices.len().max(1);
        let data = if vertices.is_empty() {
            [WorldVertex::default()]
        } else {
            [vertices[0]]
        };

        let contents = if vertices.is_empty() {
            bytemuck::cast_slice(&data)
        } else {
            bytemuck::cast_slice(vertices)
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", label)),
            contents,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, capacity)
    }

    fn create_index_buffer(
        device: &wgpu::Device,
        indices: &[u32],
        label: &str,
    ) -> (wgpu::Buffer, usize) {
        let capacity = indices.len().max(1);
        let data = if indices.is_empty() { [0u32] } else { [indices[0]] };

        let contents = if indices.is_empty() {
            bytemuck::cast_slice(&data)
        } else {
            bytemuck::cast_slice(indices)
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", label)),
            contents,
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
    left_mouse: bool,
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pos: (f32, f32),
}

struct WaterEmitter {
    position: Vec3,
    rate: f32, // Volume per second
    radius: f32,
    enabled: bool,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    terrain: Mesh,
    water: Mesh,
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
    window_size: (u32, u32),
}

impl App {
    fn new() -> Self {
        let world = build_world();

        Self {
            window: None,
            gpu: None,
            world,
            emitter: WaterEmitter {
                position: Vec3::new(128.0, 30.0, 128.0),
                rate: 100.0,
                radius: 5.0,
                enabled: false,
            },
            camera: Camera {
                position: Vec3::new(128.0, 50.0, 128.0),
                yaw: -1.57,
                pitch: -0.4,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                left_mouse: false,
                last_mouse_pos: None,
                mouse_pos: (0.0, 0.0),
            },
            last_frame: Instant::now(),
            last_stats: Instant::now(),
            window_size: (1280, 720),
        }
    }

    fn reset_world(&mut self) {
        self.world = build_world();
    }

    fn update(&mut self, dt: f32) {
        self.update_camera(dt);

        for _ in 0..STEPS_PER_FRAME {
            if self.emitter.enabled {
                self.world.add_water(self.emitter.position, self.emitter.rate * DT);
            }
            self.world.update(DT);
        }

        if self.input.left_mouse {
            if let Some(hit) = self.raycast_terrain() {
                if self
                    .input
                    .keys
                    .contains(&KeyCode::ControlLeft)
                    || self.input.keys.contains(&KeyCode::ControlRight)
                {
                    self.world
                        .add_material(hit, ADD_RADIUS, ADD_HEIGHT, TerrainMaterial::Gravel);
                } else {
                    self.world.excavate(hit, DIG_RADIUS, DIG_DEPTH);
                }
            }
        }

        if self.last_stats.elapsed() > Duration::from_secs(1) {
            let water = self.world.total_water_volume();
            let sediment = self.world.total_sediment_volume();
            println!("Water volume: {:.2}, sediment volume: {:.2}", water, sediment);
            self.last_stats = Instant::now();
        }
    }

    fn add_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world.add_water(hit, WATER_ADD_VOLUME);
        }
    }

    fn add_muddy_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world
                .add_sediment_water(hit, WATER_ADD_VOLUME, WATER_ADD_VOLUME * 0.1);
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
        if self.input.keys.contains(&KeyCode::ShiftLeft) || self.input.keys.contains(&KeyCode::ShiftRight) {
            direction.y -= 1.0;
        }

        if direction.length_squared() > 0.0 {
            self.camera.position += direction.normalize() * self.camera.speed * dt;
        }

        let world_size = self.world.world_size();
        self.camera.position.x = self.camera.position.x.clamp(0.0, world_size.x);
        self.camera.position.z = self.camera.position.z.clamp(0.0, world_size.z);
        self.camera.position.y = self.camera.position.y.clamp(2.0, world_size.y + 30.0);
    }

    fn screen_to_world_ray(&self, screen_x: f32, screen_y: f32) -> Vec3 {
        let ndc_x = (2.0 * screen_x / self.window_size.0 as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / self.window_size.1 as f32);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let inv_vp = (proj * view).inverse();

        let near = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;

        (far - near).normalize()
    }

    fn projection_matrix(&self) -> Mat4 {
        let aspect = self.window_size.0 as f32 / self.window_size.1 as f32;
        Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 500.0)
    }

    fn raycast_terrain(&self) -> Option<Vec3> {
        let ray_dir = self.screen_to_world_ray(self.input.mouse_pos.0, self.input.mouse_pos.1);
        let ray_origin = self.camera.position;

        let step = 0.5;
        let max_dist = 200.0;

        let mut t = 0.0;
        while t < max_dist {
            let p = ray_origin + ray_dir * t;

            if let Some((x, z)) = self.world.world_to_cell(p) {
                let ground = self.world.ground_height(x, z);
                if p.y <= ground {
                    return Some(Vec3::new(p.x, ground, p.z));
                }
            }

            t += step;
        }

        None
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        self.window_size = (size.width.max(1), size.height.max(1));

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

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("World Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<WorldVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (terrain_vertices, terrain_indices) = build_terrain_mesh(&self.world);
        let (water_vertices, water_indices) = build_water_mesh(&self.world);

        let terrain = Mesh::new(&device, &terrain_vertices, &terrain_indices, "Terrain");
        let water = Mesh::new(&device, &water_vertices, &water_indices, "Water");

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            bind_group,
            terrain,
            water,
        });
    }

    fn render(&mut self) {
        let window = match self.window.as_ref() {
            Some(window) => window.clone(),
            None => return,
        };

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.update(dt);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            _pad: 0.0,
        };

        let Some(gpu) = self.gpu.as_mut() else { return };

        let (terrain_vertices, terrain_indices) = build_terrain_mesh(&self.world);
        let (water_vertices, water_indices) = build_water_mesh(&self.world);

        gpu.terrain
            .update(&gpu.device, &gpu.queue, &terrain_vertices, &terrain_indices, "Terrain");
        gpu.water
            .update(&gpu.device, &gpu.queue, &water_vertices, &water_indices, "Water");

        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let frame_view = output.texture.create_view(&Default::default());

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);

            pass.set_vertex_buffer(0, gpu.terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.terrain.num_indices, 0, 0..1);

            pass.set_vertex_buffer(0, gpu.water.vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.water.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.water.num_indices, 0, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("World Heightfield Test")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.window_size = (size.width.max(1), size.height.max(1));
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.window_size = (size.width.max(1), size.height.max(1));
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.input.keys.insert(key);
                            match key {
                                KeyCode::Escape => event_loop.exit(),
                                KeyCode::KeyR => self.reset_world(),
                                KeyCode::Digit1 => self.add_water_at_cursor(),
                                KeyCode::Digit2 => self.add_muddy_water_at_cursor(),
                                KeyCode::Digit3 => {
                                    if let Some(hit) = self.raycast_terrain() {
                                        self.emitter.position = hit;
                                        self.emitter.enabled = true;
                                        println!("Emitter placed at {:?}", hit);
                                    } else {
                                        self.emitter.enabled = !self.emitter.enabled;
                                        println!("Emitter enabled: {}", self.emitter.enabled);
                                    }
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.input.keys.remove(&key);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Right => {
                    self.input.mouse_look = state == ElementState::Pressed;
                    if !self.input.mouse_look {
                        self.input.last_mouse_pos = None;
                    }
                }
                MouseButton::Left => {
                    self.input.left_mouse = state == ElementState::Pressed;
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_pos = (position.x as f32, position.y as f32);

                if self.input.mouse_look {
                    if let Some((last_x, last_y)) = self.input.last_mouse_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.camera.yaw += dx * self.camera.sensitivity;
                        self.camera.pitch = (self.camera.pitch - dy * self.camera.sensitivity)
                            .clamp(-1.4, 1.4);
                    }
                    self.input.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

fn build_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, INITIAL_HEIGHT);

    // Initial terrain sculpting for interesting flow
    for z in 50..200 {
        for x in 50..200 {
            let idx = world.idx(x, z);
            let dist_x = (x as f32 - 128.0).abs();
            let dist_z = (z as f32 - 128.0).abs();
            let dist = (dist_x.max(dist_z)) / 20.0;
            world.terrain_base[idx] = INITIAL_HEIGHT - 5.0 * (1.0 - dist).max(0.0);
        }
    }

    for z in 120..136 {
        for x in 80..176 {
            let idx = world.idx(x, z);
            world.terrain_sediment[idx] = 3.0;
        }
    }

    world
}

fn build_terrain_mesh(world: &World) -> (Vec<WorldVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(world.width * world.depth * 4);
    let mut indices = Vec::with_capacity(world.width * world.depth * 6);

    for z in 0..world.depth {
        for x in 0..world.width {
            let idx = world.idx(x, z);
            let height = world.ground_height(x, z);
            let sediment = world.terrain_sediment[idx];
            let sediment_ratio = (sediment / 2.0).min(1.0);
            let base_color = match world.terrain_material[idx] {
                TerrainMaterial::Dirt => [0.4, 0.3, 0.2],
                TerrainMaterial::Gravel => [0.5, 0.5, 0.5],
                TerrainMaterial::Sand => [0.8, 0.7, 0.5],
                TerrainMaterial::Clay => [0.6, 0.4, 0.3],
                TerrainMaterial::Bedrock => [0.3, 0.3, 0.35],
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

fn build_water_mesh(world: &World) -> (Vec<WorldVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..world.depth {
        for x in 0..world.width {
            let depth = world.water_depth(x, z);
            if depth < 0.01 {
                continue;
            }

            let idx = world.idx(x, z);
            let height = world.water_surface[idx];
            let turbidity = world.suspended_sediment[idx];

            let alpha = (depth / 2.0).min(0.8);
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

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

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
