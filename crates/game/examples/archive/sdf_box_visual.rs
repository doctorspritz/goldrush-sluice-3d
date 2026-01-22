//! SDF Box Volume Preservation Visual
//!
//! Demonstrates FLIP fluid simulation with solid boundaries defined by SDF.
//! Uses TestBox from sim3d::test_geometry to create a thick-walled container.
//!
//! Key insight: The pressure solver needs cell_types synced with SDF to
//! enforce incompressibility against SDF-defined boundaries.
//!
//! Controls:
//!   SPACE = pause/resume
//!   R = reset (static water block)
//!   F = reset with filling mode (stream of water)
//!   Mouse drag = rotate
//!   Scroll = zoom

use bytemuck::{Pod, Zeroable};
use game::example_utils::{Camera, WgpuContext, create_depth_view, Pos3Color4Vertex};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use sim3d::test_geometry::{TestBox, TestSdfGenerator};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// Grid dimensions
const GRID_WIDTH: u32 = 24;
const GRID_HEIGHT: u32 = 20;
const GRID_DEPTH: u32 = 24;
const CELL_SIZE: f32 = 0.05; // 5cm cells
const MAX_PARTICLES: usize = 10000;

const CELL_SOLID: u32 = 2;
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
    window: Option<Arc<Window>>,
    ctx: Option<WgpuContext>,
    depth_view: Option<wgpu::TextureView>,

    point_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,
    vertex_buffer: Option<wgpu::Buffer>,
    line_buffer: Option<wgpu::Buffer>,

    flip: Option<GpuFlip3D>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,
    sdf_data: Vec<f32>,

    // SDF Box geometry
    box_center: Vec3,
    box_width: f32,
    box_depth: f32,
    box_height: f32,

    camera: Camera,
    paused: bool,
    frame: u32,
    sim_time: f32,
    initial_count: usize,
    initial_height: f32,
    dragging: bool,
    last_mouse: (f32, f32),
    filling_mode: bool,
    emitter_pos: Vec3,
}

impl App {
    fn new() -> Self {
        let domain_center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.5,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );

        // Box geometry - centered in domain
        let box_center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE / 2.0,
            CELL_SIZE * 2.0, // Floor at 2 cells up
            GRID_DEPTH as f32 * CELL_SIZE / 2.0,
        );

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
            sdf_data: Vec::new(),
            box_center,
            box_width: 0.5,
            box_depth: 0.5,
            box_height: 0.5,
            camera: Camera::new(1.2, 0.5, 0.8, domain_center),
            paused: false,
            frame: 0,
            sim_time: 0.0,
            initial_count: 0,
            initial_height: 0.0,
            dragging: false,
            last_mouse: (0.0, 0.0),
            filling_mode: false,
            emitter_pos: Vec3::ZERO, // Set in setup
        }
    }

    fn reset_filling(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();

        self.filling_mode = true;
        self.emitter_pos = Vec3::new(
            self.box_center.x,
            self.box_center.y + self.box_height + 0.12,
            self.box_center.z,
        );

        self.initial_count = 0;
        self.initial_height = 0.0;
        self.frame = 0;
        self.sim_time = 0.0;

        // Re-upload SDF when FLIP is available
        if let Some(flip) = &mut self.flip {
            if let Some(ctx) = &self.ctx {
                flip.upload_sdf_force(&ctx.queue, &self.sdf_data);
            }
        }

        println!("\n=== FILLING MODE ===");
        println!("Emitter at ({:.2}, {:.2}, {:.2})", self.emitter_pos.x, self.emitter_pos.y, self.emitter_pos.z);
    }

    fn setup_sdf_and_cell_types(&mut self) {
        let wall_thickness = CELL_SIZE * 2.0;
        let floor_thickness = CELL_SIZE * 3.0;

        let test_box = TestBox::with_thickness(
            self.box_center,
            self.box_width,
            self.box_depth,
            self.box_height,
            wall_thickness,
            floor_thickness,
        );

        // Generate SDF
        let mut sdf_gen = TestSdfGenerator::new(
            GRID_WIDTH as usize,
            GRID_HEIGHT as usize,
            GRID_DEPTH as usize,
            CELL_SIZE,
            Vec3::ZERO,
        );
        sdf_gen.add_box(&test_box);
        self.sdf_data = sdf_gen.sdf_slice().to_vec();

        // Setup cell_types - sync with SDF for pressure solver
        self.cell_types = vec![0u32; (GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH) as usize];
        let mut solid_count = 0;

        for z in 0..GRID_DEPTH {
            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    let idx = (z * GRID_WIDTH * GRID_HEIGHT + y * GRID_WIDTH + x) as usize;

                    // Domain boundary = solid
                    if x == 0 || x == GRID_WIDTH - 1 || y == 0 || y == GRID_HEIGHT - 1 || z == 0 || z == GRID_DEPTH - 1 {
                        self.cell_types[idx] = CELL_SOLID;
                        solid_count += 1;
                    } else {
                        // SDF < 0 means inside solid
                        if self.sdf_data[idx] < 0.0 {
                            self.cell_types[idx] = CELL_SOLID;
                            solid_count += 1;
                        }
                    }
                }
            }
        }
        println!("Cell types: {} solid cells (from SDF + boundaries)", solid_count);
    }

    fn reset(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();
        self.filling_mode = false;

        // Spawn particles inside the box
        let inner_min_x = self.box_center.x - self.box_width / 2.0 + CELL_SIZE * 0.5;
        let inner_max_x = self.box_center.x + self.box_width / 2.0 - CELL_SIZE * 0.5;
        let inner_min_z = self.box_center.z - self.box_depth / 2.0 + CELL_SIZE * 0.5;
        let inner_max_z = self.box_center.z + self.box_depth / 2.0 - CELL_SIZE * 0.5;
        let floor_y = self.box_center.y + CELL_SIZE * 0.25;
        let fill_height = 0.20; // Fill 20cm high

        let particle_spacing = CELL_SIZE / 2.0;

        let mut x = inner_min_x;
        while x < inner_max_x {
            let mut z = inner_min_z;
            while z < inner_max_z {
                let mut y = floor_y;
                while y < floor_y + fill_height {
                    self.positions.push(Vec3::new(x, y, z));
                    self.velocities.push(Vec3::ZERO);
                    self.c_matrices.push(Mat3::ZERO);
                    self.densities.push(1.0);
                    y += particle_spacing;
                }
                z += particle_spacing;
            }
            x += particle_spacing;
        }

        self.initial_count = self.positions.len();
        self.initial_height = fill_height;
        self.frame = 0;
        self.sim_time = 0.0;

        // Re-upload SDF when FLIP is available
        if let Some(flip) = &mut self.flip {
            if let Some(ctx) = &self.ctx {
                flip.upload_sdf_force(&ctx.queue, &self.sdf_data);
            }
        }

        println!("\n=== STATIC MODE ===");
        println!("Particles: {}", self.initial_count);
        println!("Initial height: {:.3}m", self.initial_height);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("SDF BOX - Volume Preservation with SDF Boundaries")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

        // Setup SDF and cell types BEFORE creating FLIP
        self.setup_sdf_and_cell_types();

        // Create FLIP solver
        let mut flip = GpuFlip3D::new(
            &ctx.device,
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_DEPTH,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        flip.vorticity_epsilon = 0.0;
        flip.water_rest_density = 8.0;
        flip.open_boundaries = 0;
        flip.flip_ratio = 0.95;
        flip.slip_factor = 0.0;
        flip.density_projection_enabled = false;

        // Upload SDF to FLIP
        flip.upload_sdf(&ctx.queue, &self.sdf_data);

        self.flip = Some(flip);
        self.reset();

        // Setup rendering
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

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertices"),
            size: (MAX_PARTICLES * 6 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lines"),
            size: (100 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
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

        println!("\n=== SDF BOX VISUAL ===");
        println!("Grid: {}x{}x{} @ {:.2}m cells", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("Box: {:.2}m x {:.2}m x {:.2}m at ({:.2}, {:.2}, {:.2})",
            self.box_width, self.box_depth, self.box_height,
            self.box_center.x, self.box_center.y, self.box_center.z);
        println!("");
        println!("Controls: SPACE=pause, R=reset (static), F=filling mode, Mouse drag=rotate, Scroll=zoom");
        println!("============================================\n");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: ElementState::Pressed, .. },
                ..
            } => match code {
                KeyCode::Space => {
                    self.paused = !self.paused;
                    println!("{}", if self.paused { "PAUSED" } else { "RUNNING" });
                }
                KeyCode::KeyR => self.reset(),
                KeyCode::KeyF => self.reset_filling(),
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
                let dt = 1.0 / 60.0;

                // Emit particles in filling mode (like a faucet)
                if !self.paused && self.filling_mode && self.positions.len() < MAX_PARTICLES - 20 {
                    // Emit 8 particles per frame in a small cluster
                    let emit_count = 8;
                    let spacing = CELL_SIZE * 0.3;
                    for i in 0..emit_count {
                        // 2x2x2 grid of particles
                        let ox = ((i % 2) as f32 - 0.5) * spacing;
                        let oy = (((i / 2) % 2) as f32 - 0.5) * spacing;
                        let oz = ((i / 4) as f32 - 0.5) * spacing;
                        let pos = self.emitter_pos + Vec3::new(ox, oy, oz);
                        self.positions.push(pos);
                        self.velocities.push(Vec3::new(0.0, -0.5, 0.0)); // Downward initial velocity
                        self.c_matrices.push(Mat3::ZERO);
                        self.densities.push(1.0);
                    }

                    // Track "initial" values for stats as we go
                    if self.frame == 0 {
                        self.initial_count = self.positions.len();
                    }
                }

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
                        dt,
                        -9.81,
                        0.0,
                        40,
                    );

                    self.frame += 1;
                    self.sim_time += dt;

                    // Print stats every second
                    if self.frame % 60 == 0 {
                        let max_vel = self.velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                        let min_y = self.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
                        let max_y = self.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
                        let fluid_height = max_y - min_y;

                        // Count particles inside vs outside the box
                        let box_min_x = self.box_center.x - self.box_width / 2.0;
                        let box_max_x = self.box_center.x + self.box_width / 2.0;
                        let box_min_z = self.box_center.z - self.box_depth / 2.0;
                        let box_max_z = self.box_center.z + self.box_depth / 2.0;
                        let box_floor_y = self.box_center.y;
                        let box_ceil_y = self.box_center.y + self.box_height;

                        let inside_count = self.positions.iter().filter(|p| {
                            p.x >= box_min_x && p.x <= box_max_x &&
                            p.z >= box_min_z && p.z <= box_max_z &&
                            p.y >= box_floor_y && p.y <= box_ceil_y
                        }).count();
                        let outside_count = self.positions.len() - inside_count;

                        // Also get X/Z extents
                        let min_x = self.positions.iter().map(|p| p.x).fold(f32::MAX, f32::min);
                        let max_x = self.positions.iter().map(|p| p.x).fold(f32::MIN, f32::max);
                        let min_z = self.positions.iter().map(|p| p.z).fold(f32::MAX, f32::min);
                        let max_z = self.positions.iter().map(|p| p.z).fold(f32::MIN, f32::max);

                        if self.filling_mode {
                            println!(
                                "t={:5.1}s | n={:5} | Y=[{:.2},{:.2}] h={:.3}m | X=[{:.2},{:.2}] Z=[{:.2},{:.2}]",
                                self.sim_time, self.positions.len(), min_y, max_y, fluid_height, min_x, max_x, min_z, max_z
                            );
                        } else {
                            let height_ratio = fluid_height / self.initial_height;
                            println!(
                                "t={:5.1}s | n={:5} | Y=[{:.2},{:.2}] h={:.3}m ({:3.0}%) | X=[{:.2},{:.2}] Z=[{:.2},{:.2}]",
                                self.sim_time, self.positions.len(), min_y, max_y, fluid_height, height_ratio * 100.0, min_x, max_x, min_z, max_z
                            );
                        }
                    }
                }

                // Build particle vertices - color by velocity
                let mut vertices: Vec<Pos3Color4Vertex> = Vec::with_capacity(self.positions.len() * 6);
                for (i, pos) in self.positions.iter().enumerate() {
                    let vel_mag = self.velocities[i].length();
                    let t = (vel_mag / 0.1).min(1.0);
                    let color = [
                        0.2 + 0.8 * t,
                        0.4 * (1.0 - t),
                        1.0 - 0.8 * t,
                        0.9,
                    ];
                    for _ in 0..6 {
                        vertices.push(Pos3Color4Vertex {
                            position: [pos.x, pos.y, pos.z],
                            color,
                        });
                    }
                }

                // Build box wireframe lines
                let bc = self.box_center;
                let hw = self.box_width / 2.0;
                let hd = self.box_depth / 2.0;
                let h = self.box_height;
                let wc = [0.8f32, 0.6, 0.2, 1.0]; // Orange for SDF box

                let box_lines = [
                    // Floor rectangle
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y, bc.z - hd], color: wc },
                    // Top rectangle
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y + h, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y + h, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y + h, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y + h, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y + h, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y + h, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y + h, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y + h, bc.z - hd], color: wc },
                    // Vertical edges
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y + h, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y + h, bc.z - hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x + hw, bc.y + h, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y, bc.z + hd], color: wc },
                    Pos3Color4Vertex { position: [bc.x - hw, bc.y + h, bc.z + hd], color: wc },
                ];

                // Upload buffers
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
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }),
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

                    // Draw box wireframe
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
