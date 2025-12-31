//! 3D Dam Break - Visual Demo
//!
//! Renders the 3D FLIP simulation in a window with point sprites.
//! Run with: cargo run --example dam_break_3d_visual --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use sim3d::FlipSimulation3D;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const GRID_SIZE: usize = 32;
const CELL_SIZE: f32 = 0.1;
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
    camera_distance: f32,
    camera_height: f32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl App {
    fn new() -> Self {
        let mut sim = FlipSimulation3D::new(GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 40;

        // Spawn dam break: water block in corner
        for i in 1..13 {
            for j in 1..20 {
                for k in 1..13 {
                    for pi in 0..2 {
                        for pj in 0..2 {
                            let pos = Vec3::new(
                                (i as f32 + 0.25 + pi as f32 * 0.5) * CELL_SIZE,
                                (j as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE,
                                (k as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE,
                            );
                            sim.spawn_particle(pos);
                        }
                    }
                }
            }
        }

        println!("Spawned {} particles", sim.particle_count());
        println!("Controls: SPACE=pause, LEFT/RIGHT=rotate, UP/DOWN=zoom, R=reset");

        Self {
            window: None,
            gpu: None,
            sim,
            paused: false,
            camera_angle: 0.5,
            camera_distance: 5.0,
            camera_height: 2.0,
        }
    }

    fn reset_sim(&mut self) {
        self.sim = FlipSimulation3D::new(GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE);
        self.sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        self.sim.flip_ratio = 0.97;
        self.sim.pressure_iterations = 40;

        for i in 1..13 {
            for j in 1..20 {
                for k in 1..13 {
                    for pi in 0..2 {
                        for pj in 0..2 {
                            let pos = Vec3::new(
                                (i as f32 + 0.25 + pi as f32 * 0.5) * CELL_SIZE,
                                (j as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE,
                                (k as f32 + 0.25 + pj as f32 * 0.5) * CELL_SIZE,
                            );
                            self.sim.spawn_particle(pos);
                        }
                    }
                }
            }
        }
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

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // Vertex buffer (quad)
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

        // Instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
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

        // Pipeline
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
            uniform_buffer,
            bind_group,
        });
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(window) = &self.window else { return };

        // Update simulation
        if !self.paused {
            for _ in 0..2 {
                self.sim.update(1.0 / 120.0);
            }
        }

        // Camera
        let center = Vec3::new(
            GRID_SIZE as f32 * CELL_SIZE * 0.5,
            GRID_SIZE as f32 * CELL_SIZE * 0.3,
            GRID_SIZE as f32 * CELL_SIZE * 0.5,
        );
        let camera_pos = center
            + Vec3::new(
                self.camera_angle.cos() * self.camera_distance,
                self.camera_height,
                self.camera_angle.sin() * self.camera_distance,
            );

        let view = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj * view;

        // Update uniforms
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Update instances
        let instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .take(MAX_PARTICLES)
            .map(|p| {
                let speed = p.velocity.length();
                let t = (speed / 5.0).min(1.0);
                ParticleInstance {
                    position: p.position.to_array(),
                    color: [0.2 + t * 0.3, 0.5 + t * 0.3, 0.9, 0.8],
                }
            })
            .collect();

        if !instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.instance_buffer,
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
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
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
            .with_title("3D Dam Break - FLIP Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));

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
                        PhysicalKey::Code(KeyCode::ArrowLeft) => self.camera_angle -= 0.1,
                        PhysicalKey::Code(KeyCode::ArrowRight) => self.camera_angle += 0.1,
                        PhysicalKey::Code(KeyCode::ArrowUp) => self.camera_distance = (self.camera_distance - 0.5).max(2.0),
                        PhysicalKey::Code(KeyCode::ArrowDown) => self.camera_distance += 0.5,
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
    let size = 0.025;

    // Billboard: get camera right and up from view matrix
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
