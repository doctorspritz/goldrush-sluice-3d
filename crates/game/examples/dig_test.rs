//! Open-Cut Heightfield Sandbox
//!
//! Controls:
//! - WASD: Move
//! - SPACE / SHIFT: Up/Down
//! - Right Mouse: Look
//! - Left Mouse: Dig
//! - Ctrl + Left Mouse: Add material
//! - R: Reset terrain
//! - ESC: Quit
//!
//! Run: cargo run --example dig_test --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use sim3d::Heightfield;
use std::collections::HashSet;
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

const WORLD_WIDTH: f32 = 200.0;
const WORLD_DEPTH: f32 = 200.0;
const WORLD_HEIGHT: f32 = 50.0;
const CELL_SIZE: f32 = 1.0;

const GRID_WIDTH: usize = (WORLD_WIDTH / CELL_SIZE) as usize;
const GRID_DEPTH: usize = (WORLD_DEPTH / CELL_SIZE) as usize;

const INITIAL_HEIGHT: f32 = 20.0;
const DIG_RADIUS: f32 = 3.0;
const DIG_DEPTH: f32 = 1.0;
const ADD_RADIUS: f32 = 3.0;
const ADD_HEIGHT: f32 = 1.0;

const ANGLE_OF_REPOSE_DEG: f32 = 35.0;
const COLLAPSE_STEPS_PER_FRAME: usize = 2;
const COLLAPSE_TRANSFER_FRACTION: f32 = 0.35;
const COLLAPSE_MAX_OUTFLOW_FRACTION: f32 = 0.5;

const MOVE_SPEED: f32 = 18.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainVertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

struct HeightfieldRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl HeightfieldRenderer {
    fn new(device: &wgpu::Device, heightfield: &Heightfield) -> Self {
        let vertices = build_heightfield_vertices(heightfield);
        let indices = build_heightfield_indices(heightfield.width, heightfield.depth);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }

    fn update(&mut self, queue: &wgpu::Queue, heightfield: &Heightfield) {
        let vertices = build_heightfield_vertices(heightfield);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
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
    mouse_pos: (f32, f32),
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    heightfield: Heightfield,
    heightfield_dirty: bool,
    collapse_deltas: Vec<f32>,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    window_size: (u32, u32),
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    terrain: HeightfieldRenderer,
}

impl App {
    fn new() -> Self {
        let heightfield = build_heightfield();
        let collapse_deltas = vec![0.0; heightfield.width * heightfield.depth];

        Self {
            window: None,
            gpu: None,
            heightfield,
            heightfield_dirty: true,
            collapse_deltas,
            camera: Camera {
                position: Vec3::new(WORLD_WIDTH * 0.5, 25.0, WORLD_DEPTH * 0.7),
                yaw: -1.57,
                pitch: -0.25,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                last_mouse_pos: None,
                mouse_pos: (0.0, 0.0),
            },
            last_frame: Instant::now(),
            window_size: (1024, 768),
        }
    }

    fn reset_heightfield(&mut self) {
        self.heightfield = build_heightfield();
        self.heightfield_dirty = true;
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
        if self.input.keys.contains(&KeyCode::ShiftLeft)
            || self.input.keys.contains(&KeyCode::ShiftRight)
        {
            direction.y -= 1.0;
        }

        if direction.length_squared() > 0.0 {
            self.camera.position += direction.normalize() * self.camera.speed * dt;
        }

        self.camera.position.x = self.camera.position.x.clamp(0.0, WORLD_WIDTH);
        self.camera.position.z = self.camera.position.z.clamp(0.0, WORLD_DEPTH);
        self.camera.position.y = self.camera.position.y.clamp(2.0, WORLD_HEIGHT + 30.0);
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

    fn handle_click(&mut self, screen_x: f32, screen_y: f32) {
        let ray_origin = self.camera.position;
        let ray_dir = self.screen_to_world_ray(screen_x, screen_y);
        let add_mode = self.input.keys.contains(&KeyCode::ControlLeft)
            || self.input.keys.contains(&KeyCode::ControlRight);

        if let Some(hit) = self.heightfield.raycast(ray_origin, ray_dir) {
            if add_mode {
                self.apply_brush(hit.x, hit.z, ADD_RADIUS, ADD_HEIGHT);
            } else {
                self.apply_brush(hit.x, hit.z, DIG_RADIUS, -DIG_DEPTH);
            }
            self.heightfield_dirty = true;
        }
    }

    fn apply_brush(&mut self, world_x: f32, world_z: f32, radius: f32, delta: f32) {
        let cell_size = self.heightfield.cell_size;
        let cx = (world_x / cell_size).floor() as i32;
        let cz = (world_z / cell_size).floor() as i32;
        let cell_radius = (radius / cell_size).ceil() as i32;
        let radius_sq = radius * radius;

        for dz in -cell_radius..=cell_radius {
            for dx in -cell_radius..=cell_radius {
                let x = cx + dx;
                let z = cz + dz;
                if x < 0
                    || z < 0
                    || x >= self.heightfield.width as i32
                    || z >= self.heightfield.depth as i32
                {
                    continue;
                }

                let wx = (x as f32 + 0.5) * cell_size;
                let wz = (z as f32 + 0.5) * cell_size;
                let dist_sq = (wx - world_x).powi(2) + (wz - world_z).powi(2);
                if dist_sq > radius_sq {
                    continue;
                }

                let falloff = 1.0 - (dist_sq / radius_sq).sqrt();
                let current = self.heightfield.get_height(x as usize, z as usize);
                let next = (current + delta * falloff).clamp(0.0, WORLD_HEIGHT);
                self.heightfield.set_height(x as usize, z as usize, next);
            }
        }
    }

    fn apply_collapse_step(&mut self) -> bool {
        let width = self.heightfield.width;
        let depth = self.heightfield.depth;
        let cell_size = self.heightfield.cell_size;
        let angle_tan = ANGLE_OF_REPOSE_DEG.to_radians().tan();
        let diag = cell_size * 2.0_f32.sqrt();

        {
            let heights = &self.heightfield.heights;
            let deltas = &mut self.collapse_deltas;
            deltas.fill(0.0);

            for z in 0..depth {
                for x in 0..width {
                    let idx = z * width + x;
                    let h = heights[idx];
                    if h <= 0.0 {
                        continue;
                    }

                    let mut neighbor_idx = [0usize; 8];
                    let mut neighbor_amt = [0.0f32; 8];
                    let mut neighbor_count = 0usize;
                    let mut total_out = 0.0f32;

                    for (dx, dz, dist) in [
                        (1, 0, cell_size),
                        (-1, 0, cell_size),
                        (0, 1, cell_size),
                        (0, -1, cell_size),
                        (1, 1, diag),
                        (1, -1, diag),
                        (-1, 1, diag),
                        (-1, -1, diag),
                    ] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                            continue;
                        }

                        let nidx = nz as usize * width + nx as usize;
                        let nh = heights[nidx];
                        let diff = h - nh;
                        let max_diff = angle_tan * dist;
                        if diff > max_diff {
                            let transfer = COLLAPSE_TRANSFER_FRACTION * (diff - max_diff);
                            neighbor_idx[neighbor_count] = nidx;
                            neighbor_amt[neighbor_count] = transfer;
                            neighbor_count += 1;
                            total_out += transfer;
                        }
                    }

                    if total_out <= 0.0 {
                        continue;
                    }

                    let max_out = h * COLLAPSE_MAX_OUTFLOW_FRACTION;
                    let scale = if total_out > max_out {
                        max_out / total_out
                    } else {
                        1.0
                    };

                    for i in 0..neighbor_count {
                        let transfer = neighbor_amt[i] * scale;
                        let nidx = neighbor_idx[i];
                        deltas[idx] -= transfer;
                        deltas[nidx] += transfer;
                    }
                }
            }
        }

        let mut changed = false;
        for (height, delta) in self
            .heightfield
            .heights
            .iter_mut()
            .zip(self.collapse_deltas.iter())
        {
            if *delta != 0.0 {
                *height = (*height + *delta).clamp(0.0, WORLD_HEIGHT);
                changed = true;
            }
        }

        changed
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
            label: Some("Terrain Shader"),
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
            label: Some("Terrain Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TerrainVertex>() as u64,
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
                            format: wgpu::VertexFormat::Float32x3,
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
                    blend: None,
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

        let terrain = HeightfieldRenderer::new(&device, &self.heightfield);

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            bind_group,
            terrain,
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

        self.update_camera(dt);

        let mut collapse_changed = false;
        for _ in 0..COLLAPSE_STEPS_PER_FRAME {
            if self.apply_collapse_step() {
                collapse_changed = true;
            }
        }

        if collapse_changed {
            self.heightfield_dirty = true;
        }

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            _pad: 0.0,
        };

        let Some(gpu) = self.gpu.as_mut() else { return };

        if self.heightfield_dirty {
            gpu.terrain.update(&gpu.queue, &self.heightfield);
            self.heightfield_dirty = false;
        }
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
            pass.set_index_buffer(
                gpu.terrain.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(0..gpu.terrain.num_indices, 0, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Open-Cut Heightfield Sandbox")
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
                                KeyCode::KeyR => self.reset_heightfield(),
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
                    if state == ElementState::Pressed {
                        let (x, y) = self.input.mouse_pos;
                        self.handle_click(x, y);
                    }
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
                        self.camera.pitch =
                            (self.camera.pitch - dy * self.camera.sensitivity).clamp(-1.4, 1.4);
                    }
                    self.input.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

fn build_heightfield() -> Heightfield {
    let mut heightfield = Heightfield::new(GRID_WIDTH, GRID_DEPTH, CELL_SIZE, INITIAL_HEIGHT);

    for z in 0..GRID_DEPTH {
        for x in 0..GRID_WIDTH {
            let nx = x as f32 / GRID_WIDTH as f32;
            let nz = z as f32 / GRID_DEPTH as f32;
            let slope = (1.0 - nz) * 6.0;
            let noise = (nx * 12.0).sin() * (nz * 10.0).cos() * 1.5;
            let height = (INITIAL_HEIGHT + slope + noise).clamp(0.0, WORLD_HEIGHT);
            heightfield.set_height(x, z, height);
        }
    }

    heightfield
}

fn build_heightfield_vertices(heightfield: &Heightfield) -> Vec<TerrainVertex> {
    let mut vertices = Vec::with_capacity(heightfield.width * heightfield.depth * 4);

    let mut max_height = 0.0f32;
    for h in &heightfield.heights {
        max_height = max_height.max(*h);
    }
    let max_height = max_height.max(1.0);

    for z in 0..heightfield.depth {
        for x in 0..heightfield.width {
            let h = heightfield.get_height(x, z);
            let color_factor = (h / max_height).clamp(0.0, 1.0);
            let color = [
                0.35 + 0.25 * color_factor,
                0.25 + 0.2 * color_factor,
                0.15 + 0.1 * color_factor,
            ];

            let x0 = x as f32 * heightfield.cell_size;
            let x1 = (x + 1) as f32 * heightfield.cell_size;
            let z0 = z as f32 * heightfield.cell_size;
            let z1 = (z + 1) as f32 * heightfield.cell_size;

            vertices.push(TerrainVertex {
                position: [x0, h, z0],
                color,
            });
            vertices.push(TerrainVertex {
                position: [x1, h, z0],
                color,
            });
            vertices.push(TerrainVertex {
                position: [x1, h, z1],
                color,
            });
            vertices.push(TerrainVertex {
                position: [x0, h, z1],
                color,
            });
        }
    }

    vertices
}

fn build_heightfield_indices(width: usize, depth: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity(width * depth * 6);

    for z in 0..depth {
        for x in 0..width {
            let base = (z * width + x) as u32 * 4;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    indices
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
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
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
    return vec4<f32>(in.color, 1.0);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
