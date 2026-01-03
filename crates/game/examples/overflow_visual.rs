//! Visual version of overflow_test - CPU simulation with working parameters
//!
//! Run with: cargo run --example overflow_visual --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
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

// Same parameters as overflow_test.rs - these WORK
const GRID_WIDTH: usize = 48;
const GRID_HEIGHT: usize = 16;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.05;
const MAX_PARTICLES: usize = 50000;

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

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sim: FlipSimulation3D,
    paused: bool,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    frame: u32,
    solid_instances: Vec<ParticleInstance>,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    solid_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn create_sluice_with_tall_riffles(sim: &mut FlipSimulation3D) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    let floor_height_left = 4;
    let floor_height_right = 1;
    let riffle_spacing = 6;
    let riffle_height = 3;
    let riffle_start_x = 8;
    let riffle_end_x = width - 4;

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                let t = i as f32 / (width - 1) as f32;
                let floor_height = floor_height_left as f32 * (1.0 - t) + floor_height_right as f32 * t;
                let floor_j = floor_height as usize;

                let is_riffle = i >= riffle_start_x && i < riffle_end_x &&
                    (i - riffle_start_x) % riffle_spacing < 2 &&
                    j <= floor_j + riffle_height &&
                    j > floor_j;

                let is_boundary =
                    i == 0 ||
                    i == width - 1 ||
                    j <= floor_j ||
                    j == height - 1 ||
                    k == 0 || k == depth - 1 ||
                    is_riffle;

                if is_boundary {
                    sim.grid.set_solid(i, j, k);
                }
            }
        }
    }

    sim.grid.compute_sdf();
}

// Simple LCG for deterministic random-like numbers
fn simple_rand(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) & 0x7FFF) as f32 / 32767.0
}

static mut RAND_SEED: u32 = 12345;

fn emit_particles(sim: &mut FlipSimulation3D, count: usize) {
    let emit_x = 2.0 * CELL_SIZE;
    let center_z = GRID_DEPTH as f32 * CELL_SIZE * 0.5;
    let emit_y = 8.0 * CELL_SIZE;
    let inlet_velocity = Vec3::new(0.5, 0.0, 0.0);

    for _ in 0..count {
        let (r1, r2, r3) = unsafe {
            (simple_rand(&mut RAND_SEED), simple_rand(&mut RAND_SEED), simple_rand(&mut RAND_SEED))
        };
        let x = emit_x + r1 * CELL_SIZE;
        let z = center_z + (r2 - 0.5) * 2.0 * CELL_SIZE;
        let y = emit_y + r3 * 0.5 * CELL_SIZE;

        sim.spawn_particle_with_velocity(Vec3::new(x, y, z), inlet_velocity);
    }
}

impl App {
    fn new() -> Self {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        // Tilted gravity - this is what makes water flow!
        sim.gravity = Vec3::new(0.6, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 100;

        create_sluice_with_tall_riffles(&mut sim);
        let solid_instances = Self::collect_solids(&sim);

        // Initial particles
        emit_particles(&mut sim, 500);

        println!("Overflow Visual - CPU Simulation");
        println!("Grid: {}x{}x{}, cell_size={}", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("Gravity: ({}, {}, {})", sim.gravity.x, sim.gravity.y, sim.gravity.z);
        println!("Controls: SPACE=pause, R=reset, ESC=quit, Drag=rotate, Scroll=zoom");

        Self {
            window: None,
            gpu: None,
            sim,
            paused: false,
            camera_angle: 0.3,
            camera_pitch: 0.4,
            camera_distance: 2.0,
            frame: 0,
            solid_instances,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
        }
    }

    fn collect_solids(sim: &FlipSimulation3D) -> Vec<ParticleInstance> {
        let mut solids = Vec::new();
        let dx = sim.grid.cell_size;
        let w = sim.grid.width;
        let h = sim.grid.height;
        let d = sim.grid.depth;

        for k in 0..d {
            for j in 0..h {
                for i in 0..w {
                    let idx = sim.grid.cell_index(i, j, k);
                    if !sim.grid.solid[idx] {
                        continue;
                    }

                    let is_surface =
                        (i == 0 || !sim.grid.is_solid(i - 1, j, k)) ||
                        (i == w - 1 || !sim.grid.is_solid(i + 1, j, k)) ||
                        (j == 0 || !sim.grid.is_solid(i, j - 1, k)) ||
                        (j == h - 1 || !sim.grid.is_solid(i, j + 1, k)) ||
                        (k == 0 || !sim.grid.is_solid(i, j, k - 1)) ||
                        (k == d - 1 || !sim.grid.is_solid(i, j, k + 1));

                    if !is_surface {
                        continue;
                    }

                    solids.push(ParticleInstance {
                        position: [
                            (i as f32 + 0.5) * dx,
                            (j as f32 + 0.5) * dx,
                            (k as f32 + 0.5) * dx,
                        ],
                        color: [0.45, 0.38, 0.32, 1.0],
                    });
                }
            }
        }
        solids
    }

    fn reset_sim(&mut self) {
        self.sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        self.sim.gravity = Vec3::new(0.6, -9.8, 0.0);
        self.sim.flip_ratio = 0.97;
        self.sim.pressure_iterations = 100;

        create_sluice_with_tall_riffles(&mut self.sim);
        self.solid_instances = Self::collect_solids(&self.sim);
        emit_particles(&mut self.sim, 500);
        self.frame = 0;
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let vertices = [
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [1.0, -1.0] },
            Vertex { position: [1.0, 1.0] },
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [1.0, 1.0] },
            Vertex { position: [-1.0, 1.0] },
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

        let solid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Solid Buffer"),
            size: (10000 * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if !self.solid_instances.is_empty() {
            queue.write_buffer(&solid_buffer, 0, bytemuck::cast_slice(&self.solid_instances));
        }

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
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            instance_buffer,
            solid_buffer,
            uniform_buffer,
            bind_group,
        });
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(window) = &self.window else { return };

        if !self.paused {
            let dt = 1.0 / 60.0;

            // CPU simulation - this is what works!
            self.sim.update(dt);

            // Emit particles
            if self.sim.particle_count() < MAX_PARTICLES - 100 {
                emit_particles(&mut self.sim, 20);
            }

            self.frame += 1;
            self.fps_frame_count += 1;

            let now = Instant::now();
            let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
            if elapsed >= 1.0 {
                self.current_fps = self.fps_frame_count as f32 / elapsed;
                let max_x = self.sim.particles.list.iter()
                    .map(|p| p.position.x)
                    .fold(0.0f32, f32::max);
                let max_y = self.sim.particles.list.iter()
                    .map(|p| p.position.y)
                    .fold(0.0f32, f32::max);
                println!(
                    "Frame {:5} | FPS: {:5.1} | Particles: {:5} | MaxX: {:5.2} | MaxY: {:5.2}",
                    self.frame, self.current_fps, self.sim.particle_count(),
                    max_x / CELL_SIZE, max_y / CELL_SIZE
                );
                self.fps_frame_count = 0;
                self.last_fps_time = now;
            }
        }

        // Camera
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.4,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        let cos_pitch = self.camera_pitch.cos();
        let camera_pos = center
            + Vec3::new(
                self.camera_angle.cos() * cos_pitch * self.camera_distance,
                self.camera_pitch.sin() * self.camera_distance,
                self.camera_angle.sin() * cos_pitch * self.camera_distance,
            );

        let view = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 50.0);
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let water_instances: Vec<ParticleInstance> = self.sim.particles.list.iter()
            .take(MAX_PARTICLES)
            .map(|p| {
                let speed = p.velocity.length();
                let t = (speed / 2.0).min(1.0);
                ParticleInstance {
                    position: p.position.to_array(),
                    color: [0.2 + t * 0.3, 0.5 + t * 0.3, 0.9, 0.8],
                }
            })
            .collect();

        if !water_instances.is_empty() {
            gpu.queue.write_buffer(&gpu.instance_buffer, 0, bytemuck::cast_slice(&water_instances));
        }

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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));

            let solid_count = self.solid_instances.len().min(10000);
            if solid_count > 0 {
                pass.set_vertex_buffer(1, gpu.solid_buffer.slice(..));
                pass.draw(0..6, 0..solid_count as u32);
            }

            if !water_instances.is_empty() {
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                pass.draw(0..6, 0..water_instances.len() as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Overflow Visual - CPU Simulation (WORKING)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());
        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => self.paused = !self.paused,
                        PhysicalKey::Code(KeyCode::KeyR) => self.reset_sim(),
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed {
                        self.last_mouse_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((last_x, last_y)) = self.last_mouse_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.camera_angle += dx * 0.01;
                        self.camera_pitch = (self.camera_pitch - dy * 0.01).clamp(-1.4, 1.4);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.2).clamp(0.5, 10.0);
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

const SHADER: &str = r#"
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
    let size = 0.012;
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

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
