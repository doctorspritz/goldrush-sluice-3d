//! BASIC TEST: Static water in a closed box.
//! Water spawns, sits there, should NOT move or escape.
//! Press SPACE to pause, R to reset.

use bytemuck::{Pod, Zeroable};
use game::example_utils::{Camera, WgpuContext, create_depth_view, Pos3Color4Vertex};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// Grid: 8x8x8 cells - SMALL AND SIMPLE
const GRID_WIDTH: usize = 8;
const GRID_HEIGHT: usize = 8;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.05;
const MAX_PARTICLES: usize = 2000;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const QUAD_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    point_size: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let clip_center = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    let corner_idx = in.vertex_index % 6u;
    var offset: vec2<f32>;
    switch corner_idx {
        case 0u: { offset = vec2<f32>(-1.0, -1.0); }
        case 1u: { offset = vec2<f32>(-1.0, 1.0); }
        case 2u: { offset = vec2<f32>(1.0, 1.0); }
        case 3u: { offset = vec2<f32>(-1.0, -1.0); }
        case 4u: { offset = vec2<f32>(1.0, 1.0); }
        case 5u: { offset = vec2<f32>(1.0, -1.0); }
        default: { offset = vec2<f32>(0.0, 0.0); }
    }
    let size = uniforms.point_size / 500.0;
    out.clip_position = clip_center + vec4<f32>(offset * size, 0.0, 0.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const LINE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    point_size: f32,
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    point_size: f32,
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    // Window/GPU
    window: Option<Arc<Window>>,
    ctx: Option<WgpuContext>,
    depth_view: Option<wgpu::TextureView>,

    // Pipelines
    point_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,
    vertex_buffer: Option<wgpu::Buffer>,
    line_buffer: Option<wgpu::Buffer>,

    // Simulation
    flip: Option<GpuFlip3D>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,

    // State
    camera: Camera,
    paused: bool,
    frame: u32,
    initial_count: usize,
    dragging: bool,
    last_mouse: (f32, f32),
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            ctx: None,
            depth_view: None,
            point_pipeline: None,
            line_pipeline: None,
            uniform_buffer: None,
            uniform_bind_group: None,
            vertex_buffer: None,
            line_buffer: None,
            flip: None,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            camera: Camera::new(
                0.5,  // yaw
                0.3,  // pitch
                0.6,  // distance
                Vec3::new(0.2, 0.15, 0.2), // target (grid center)
            ),
            paused: false,
            frame: 0,
            initial_count: 0,
            dragging: false,
            last_mouse: (0.0, 0.0),
        }
    }

    fn reset(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();

        // Fill lower half with particles
        for x in 1..(GRID_WIDTH - 1) {
            for y in 1..(GRID_HEIGHT / 2) {
                for z in 1..(GRID_DEPTH - 1) {
                    let p = Vec3::new(
                        (x as f32 + 0.5) * CELL_SIZE,
                        (y as f32 + 0.5) * CELL_SIZE,
                        (z as f32 + 0.5) * CELL_SIZE,
                    );
                    self.positions.push(p);
                    self.velocities.push(Vec3::ZERO);
                    self.c_matrices.push(Mat3::ZERO);
                    self.densities.push(1.0);
                }
            }
        }
        self.initial_count = self.positions.len();
        self.frame = 0;
        println!("RESET: {} particles", self.initial_count);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("STATIC WATER TEST - Should NOT move")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

        // Setup cell_types: ALL boundaries solid
        self.cell_types = vec![0u32; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
        for z in 0..GRID_DEPTH {
            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    let idx = z * GRID_WIDTH * GRID_HEIGHT + y * GRID_WIDTH + x;
                    if x == 0 || x == GRID_WIDTH - 1
                        || y == 0 || y == GRID_HEIGHT - 1
                        || z == 0 || z == GRID_DEPTH - 1
                    {
                        self.cell_types[idx] = 2; // SOLID
                    }
                }
            }
        }

        // FLIP solver
        let mut flip = GpuFlip3D::new(
            &ctx.device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );
        flip.vorticity_epsilon = 0.0;
        flip.open_boundaries = 0; // ALL CLOSED (box)
        flip.flip_ratio = 0.0;    // Pure PIC for maximum damping
        flip.slip_factor = 0.0;   // No-slip at solid boundaries (hydrostatic fix)
        self.flip = Some(flip);

        // Initial particles
        self.reset();

        // Uniform buffer
        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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

        let uniform_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let quad_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Quad Shader"),
            source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()),
        });
        let line_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()),
        });

        // Point pipeline (quads)
        let point_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Quad Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &quad_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Pos3Color4Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &quad_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Line pipeline
        let line_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Pos3Color4Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Vertex buffer
        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertices"),
            size: (MAX_PARTICLES * 6 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Line buffer for bounding box
        let line_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lines"),
            size: (24 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.point_pipeline = Some(point_pipeline);
        self.line_pipeline = Some(line_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.line_buffer = Some(line_buffer);
        self.ctx = Some(ctx);

        println!("=== STATIC WATER TEST ===");
        println!("Grid: {}x{}x{}", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH);
        println!("Particles: {}", self.initial_count);
        println!("Expected: Water sits still, NO movement, NO escape");
        println!("Controls: SPACE=pause, R=reset, Mouse=rotate, Scroll=zoom");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: ElementState::Pressed, .. },
                ..
            } => match code {
                KeyCode::Space => self.paused = !self.paused,
                KeyCode::KeyR => self.reset(),
                KeyCode::Escape => event_loop.exit(),
                _ => {}
            },
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if self.dragging {
                    let dx = x - self.last_mouse.0;
                    let dy = y - self.last_mouse.1;
                    self.camera.handle_mouse_move(dx, dy);
                }
                self.last_mouse = (x, y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.camera.handle_zoom(scroll);
            }
            WindowEvent::RedrawRequested => {
                let ctx = self.ctx.as_ref().unwrap();
                let flip = self.flip.as_mut().unwrap();

                // Physics step
                if !self.paused && !self.positions.is_empty() {
                    flip.step(
                        &ctx.device,
                        &ctx.queue,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.c_matrices,
                        &self.densities,
                        &self.cell_types,
                        None,
                        None,
                        1.0 / 60.0,
                        -9.81,
                        0.0,
                        40,
                    );

                    // NO CPU CLAMP - let FLIP handle boundaries via cell_types
                    self.frame += 1;
                    if self.frame % 60 == 0 {
                        let max_vel = self.velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                        let min_x = self.positions.iter().map(|p| p.x).fold(f32::MAX, f32::min);
                        let max_x = self.positions.iter().map(|p| p.x).fold(f32::MIN, f32::max);
                        let min_y = self.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
                        let max_y_pos = self.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
                        let min_z = self.positions.iter().map(|p| p.z).fold(f32::MAX, f32::min);
                        let max_z = self.positions.iter().map(|p| p.z).fold(f32::MIN, f32::max);
                        println!(
                            "Frame {:4}: n={} vel={:.3} x=[{:.3},{:.3}] y=[{:.3},{:.3}] z=[{:.3},{:.3}]",
                            self.frame, self.positions.len(), max_vel,
                            min_x, max_x, min_y, max_y_pos, min_z, max_z
                        );
                    }
                }

                // Build particle vertices (6 per particle for quads)
                let water_color = [0.2f32, 0.5, 1.0, 0.9];
                let mut vertices: Vec<Pos3Color4Vertex> = Vec::with_capacity(self.positions.len() * 6);
                for pos in &self.positions {
                    for _ in 0..6 {
                        vertices.push(Pos3Color4Vertex {
                            position: [pos.x, pos.y, pos.z],
                            color: water_color,
                        });
                    }
                }

                // Build bounding box lines
                let x_min = CELL_SIZE;
                let x_max = (GRID_WIDTH as f32 - 1.0) * CELL_SIZE;
                let y_min = CELL_SIZE;
                let y_max = (GRID_HEIGHT as f32 - 1.0) * CELL_SIZE;
                let z_min = CELL_SIZE;
                let z_max = (GRID_DEPTH as f32 - 1.0) * CELL_SIZE;
                let lc = [0.5f32, 0.5, 0.5, 1.0]; // Gray lines
                let box_lines = [
                    // Bottom
                    Pos3Color4Vertex { position: [x_min, y_min, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_min, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_min, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_min, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_min, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_min, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_min, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_min, z_min], color: lc },
                    // Top
                    Pos3Color4Vertex { position: [x_min, y_max, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_max, z_min], color: lc },
                    // Verticals
                    Pos3Color4Vertex { position: [x_min, y_min, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_max, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_min, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_min], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_min, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_min, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_min, y_max, z_max], color: lc },
                ];

                // Upload
                if !vertices.is_empty() {
                    ctx.queue.write_buffer(
                        self.vertex_buffer.as_ref().unwrap(),
                        0,
                        bytemuck::cast_slice(&vertices),
                    );
                }
                ctx.queue.write_buffer(
                    self.line_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&box_lines),
                );

                // Camera
                let window = self.window.as_ref().unwrap();
                let size = window.inner_size();
                let aspect = size.width as f32 / size.height as f32;
                let view_proj = self.camera.view_proj_matrix(aspect);
                let uniforms = Uniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    camera_pos: self.camera.position().to_array(),
                    point_size: 8.0,
                };
                ctx.queue.write_buffer(
                    self.uniform_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&[uniforms]),
                );

                // Render
                let output = ctx.surface.get_current_texture().unwrap();
                let view = output.texture.create_view(&Default::default());
                let mut encoder = ctx.device.create_command_encoder(&Default::default());

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: self.depth_view.as_ref().unwrap(),
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    // Draw bounding box
                    pass.set_pipeline(self.line_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, self.uniform_bind_group.as_ref().unwrap(), &[]);
                    pass.set_vertex_buffer(0, self.line_buffer.as_ref().unwrap().slice(..));
                    pass.draw(0..24, 0..1);

                    // Draw particles
                    if !vertices.is_empty() {
                        pass.set_pipeline(self.point_pipeline.as_ref().unwrap());
                        pass.set_bind_group(0, self.uniform_bind_group.as_ref().unwrap(), &[]);
                        pass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
                        pass.draw(0..vertices.len() as u32, 0..1);
                    }
                }

                ctx.queue.submit(std::iter::once(encoder.finish()));
                output.present();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
