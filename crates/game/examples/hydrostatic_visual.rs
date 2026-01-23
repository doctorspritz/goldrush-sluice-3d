//! Hydrostatic Equilibrium Visual Test
//!
//! Matches test_flip_long_settling EXACTLY:
//! - 16x16x16 grid at 0.05m cells
//! - 512 particles (8x8x8 region, 1 per cell)
//! - flip_ratio = 0.95 (FLIP mode)
//! - density_projection_enabled = false
//! - pressure_iterations = 20
//! - Adaptive velocity damping
//! - Bed: 1 cell thick (matches headless harness)
//!
//! Expected: Water spreads and settles to ~27% of initial height
//! Max velocity drops below 0.03 m/s
//!
//! Controls: SPACE=pause, R=reset, Mouse=rotate, Scroll=zoom

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

// Grid matching test_flip_long_settling EXACTLY
const GRID_WIDTH: usize = 16;
const GRID_HEIGHT: usize = 16;
const GRID_DEPTH: usize = 16;
const CELL_SIZE: f32 = 0.05;    // 5cm cells
// Keep in sync with HYDROSTATIC_BED_HEIGHT_CELLS in headless_harness.rs.
const BED_HEIGHT_CELLS: usize = 1;
const BED_COLOR: [f32; 4] = [0.35, 0.25, 0.15, 1.0];

// Water region: cells [4,12)×[1,9)×[4,12) = 8×8×8 = 512 particles at 1 per cell
const WATER_MIN_X: usize = 4;
const WATER_MAX_X: usize = 12;  // exclusive
const WATER_MIN_Y: usize = 1;
const WATER_MAX_Y: usize = 9;   // exclusive, 8 cells high
const WATER_MIN_Z: usize = 4;
const WATER_MAX_Z: usize = 12;  // exclusive

const PARTICLES_PER_CELL: usize = 1; // 1 particle per cell (matching test)
const WATER_CELLS: usize = (WATER_MAX_X - WATER_MIN_X) * (WATER_MAX_Y - WATER_MIN_Y) * (WATER_MAX_Z - WATER_MIN_Z);
const MAX_PARTICLES: usize = WATER_CELLS * PARTICLES_PER_CELL + 1000;

// Physics constants from test
const FLIP_RATIO: f32 = 0.95;
const PRESSURE_ITERATIONS: u32 = 20;
const SETTLED_VELOCITY_THRESHOLD: f32 = 0.03;
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

/// Setup solid boundary cells - 1 cell thick walls on all sides except top
fn setup_solid_boundaries(cell_types: &mut [u32], w: usize, h: usize, d: usize) {
    const CELL_SOLID: u32 = 2;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * w * h + y * w + x;
                // Solid if at any edge (except top)
                let at_x_edge = x == 0 || x == w - 1;
                let at_y_floor = y < BED_HEIGHT_CELLS;  // Floor only, top is open
                let at_z_edge = z == 0 || z == d - 1;

                if at_x_edge || at_y_floor || at_z_edge {
                    cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }
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
    solid_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,
    vertex_buffer: Option<wgpu::Buffer>,
    line_buffer: Option<wgpu::Buffer>,
    bed_buffer: Option<wgpu::Buffer>,

    flip: Option<GpuFlip3D>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,

    camera: Camera,
    paused: bool,
    frame: u32,
    sim_time: f32,
    initial_count: usize,
    initial_height: f32,
    dragging: bool,
    last_mouse: (f32, f32),
}

impl App {
    fn new() -> Self {
        let domain_center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.5,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        Self {
            window: None,
            ctx: None,
            depth_view: None,
            point_pipeline: None,
            line_pipeline: None,
            solid_pipeline: None,
            uniform_buffer: None,
            uniform_bind_group: None,
            vertex_buffer: None,
            line_buffer: None,
            bed_buffer: None,
            flip: None,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            camera: Camera::new(0.5, 0.4, 0.6, domain_center),
            paused: false,
            frame: 0,
            sim_time: 0.0,
            initial_count: 0,
            initial_height: 0.0,
            dragging: false,
            last_mouse: (0.0, 0.0),
        }
    }

    fn reset(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();

        // Spawn 1 particle per cell - EXACTLY as in test_flip_long_settling
        for cx in WATER_MIN_X..WATER_MAX_X {
            for cy in WATER_MIN_Y..WATER_MAX_Y {
                for cz in WATER_MIN_Z..WATER_MAX_Z {
                    let pos = Vec3::new(
                        (cx as f32 + 0.5) * CELL_SIZE,
                        (cy as f32 + 0.5) * CELL_SIZE,
                        (cz as f32 + 0.5) * CELL_SIZE,
                    );
                    self.positions.push(pos);
                    self.velocities.push(Vec3::ZERO);
                    self.c_matrices.push(Mat3::ZERO);
                    self.densities.push(1.0);
                }
            }
        }

        self.initial_count = self.positions.len();
        self.initial_height = (WATER_MAX_Y - WATER_MIN_Y) as f32 * CELL_SIZE;
        self.frame = 0;
        self.sim_time = 0.0;

        println!("\n=== RESET ===");
        println!("Particles: {} ({} per cell, matching test)", self.initial_count, PARTICLES_PER_CELL);
        println!("Initial height: {:.3}m ({} cells)", self.initial_height, WATER_MAX_Y - WATER_MIN_Y);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("HYDROSTATIC EQUILIBRIUM - Water should become STILL")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

        // Setup cell_types: solid walls on all sides, matching the unit test
        self.cell_types = vec![0u32; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
        setup_solid_boundaries(&mut self.cell_types, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH);

        // FLIP solver with EXACT test_flip_long_settling configuration
        let mut flip = GpuFlip3D::new(
            &ctx.device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        // EXACT settings from test_flip_long_settling
        flip.vorticity_epsilon = 0.0;
        flip.water_rest_density = 1.0;
        flip.open_boundaries = 0;
        flip.flip_ratio = FLIP_RATIO;
        flip.slip_factor = 0.0;
        flip.density_projection_enabled = false;

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

        let solid_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Solid Pipeline"),
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

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertices"),
            size: (MAX_PARTICLES * 6 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lines"),
            size: (48 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed"),
            size: (6 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.point_pipeline = Some(point_pipeline);
        self.line_pipeline = Some(line_pipeline);
        self.solid_pipeline = Some(solid_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.line_buffer = Some(line_buffer);
        self.bed_buffer = Some(bed_buffer);
        self.ctx = Some(ctx);

        println!("\n=== HYDROSTATIC VISUAL TEST ===");
        println!("Configuration matches test_flip_long_settling EXACTLY:");
        println!("  Grid: {}x{}x{} cells @ {:.2}m", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("  Particles: {} (1 per cell)", self.initial_count);
        println!("  flip_ratio: {}", FLIP_RATIO);
        println!("  density_projection: false");
        println!("  pressure_iterations: {}", PRESSURE_ITERATIONS);
        println!("  Boundary: floor + walls, open top");
        println!("");
        println!("EXPECTED BEHAVIOR:");
        println!("  - Water column spreads and flattens");
        println!("  - Settles to ~27% of initial height");
        println!("  - Max velocity drops below {} m/s", SETTLED_VELOCITY_THRESHOLD);
        println!("");
        println!("Controls: SPACE=pause, R=reset, Mouse drag=rotate, Scroll=zoom");
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

                // Physics step - EXACT match to test_flip_long_settling
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
                        PRESSURE_ITERATIONS,
                    );

                    // CPU boundary enforcement - EXACT match to test
                    let floor_y = CELL_SIZE * 1.0;
                    let min_x = CELL_SIZE * 1.0;
                    let max_x = (GRID_WIDTH as f32 - 1.0) * CELL_SIZE;
                    let min_z = CELL_SIZE * 1.0;
                    let max_z = (GRID_DEPTH as f32 - 1.0) * CELL_SIZE;
                    let ceiling_y = (GRID_HEIGHT as f32 - 0.5) * CELL_SIZE;

                    // Adaptive velocity damping - EXACT match to test
                    let damping_base = 0.005;
                    let damping_growth = (self.sim_time / 60.0).min(1.0) * 0.015;
                    let damping = 1.0 - (damping_base + damping_growth);

                    for (pos, vel) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
                        *vel *= damping;

                        if pos.y < floor_y { pos.y = floor_y; vel.y = vel.y.abs().min(0.1); }
                        if pos.x < min_x { pos.x = min_x; vel.x = vel.x.abs().min(0.1); }
                        if pos.x > max_x { pos.x = max_x; vel.x = -vel.x.abs().min(0.1); }
                        if pos.z < min_z { pos.z = min_z; vel.z = vel.z.abs().min(0.1); }
                        if pos.z > max_z { pos.z = max_z; vel.z = -vel.z.abs().min(0.1); }
                        if pos.y > ceiling_y { pos.y = ceiling_y; vel.y = -vel.y.abs().min(0.1); }
                    }

                    self.frame += 1;
                    self.sim_time += dt;

                    // Print stats every second
                    if self.frame % 60 == 0 {
                        let max_vel = self.velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                        let min_y = self.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
                        let max_y = self.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
                        let fluid_height = max_y - min_y;
                        let height_ratio = fluid_height / self.initial_height;

                        let status = if max_vel < SETTLED_VELOCITY_THRESHOLD {
                            "SETTLED"
                        } else if max_vel < 0.05 {
                            "settling..."
                        } else {
                            "active"
                        };

                        println!(
                            "t={:5.1}s | max_vel={:.4} | height={:.3}m ({:4.0}%) | y=[{:.3},{:.3}] | {}",
                            self.sim_time, max_vel, fluid_height, height_ratio * 100.0,
                            min_y, max_y, status
                        );
                    }
                }

                // Color particles by velocity magnitude
                let mut vertices: Vec<Pos3Color4Vertex> = Vec::with_capacity(self.positions.len() * 6);
                for (i, pos) in self.positions.iter().enumerate() {
                    let vel_mag = self.velocities[i].length();
                    // Color: blue = still, red = moving
                    let t = (vel_mag / 0.1).min(1.0);
                    let color = [
                        0.2 + 0.8 * t,      // R: increases with velocity
                        0.3 * (1.0 - t),    // G: decreases with velocity
                        1.0 - 0.8 * t,      // B: decreases with velocity
                        0.85,
                    ];
                    for _ in 0..6 {
                        vertices.push(Pos3Color4Vertex {
                            position: [pos.x, pos.y, pos.z],
                            color,
                        });
                    }
                }

                // Build bounding box lines (domain boundaries)
                let x_max = GRID_WIDTH as f32 * CELL_SIZE;
                let y_max = GRID_HEIGHT as f32 * CELL_SIZE;
                let z_max = GRID_DEPTH as f32 * CELL_SIZE;
                let lc = [0.4f32, 0.4, 0.4, 1.0];
                let wc = [0.3f32, 0.5, 0.7, 0.5]; // Water level indicator
                let water_y = WATER_MAX_Y as f32 * CELL_SIZE; // Top of water region

                let box_lines = [
                    // Bottom face
                    Pos3Color4Vertex { position: [0.0, 0.0, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, 0.0, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, 0.0, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, 0.0, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_max, 0.0, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, 0.0, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, 0.0, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, 0.0, 0.0], color: lc },
                    // Top face
                    Pos3Color4Vertex { position: [0.0, y_max, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, y_max, 0.0], color: lc },
                    // Vertical edges
                    Pos3Color4Vertex { position: [0.0, 0.0, 0.0], color: lc },
                    Pos3Color4Vertex { position: [0.0, y_max, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, 0.0, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, 0.0], color: lc },
                    Pos3Color4Vertex { position: [x_max, 0.0, z_max], color: lc },
                    Pos3Color4Vertex { position: [x_max, y_max, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, 0.0, z_max], color: lc },
                    Pos3Color4Vertex { position: [0.0, y_max, z_max], color: lc },
                    // Water level indicator
                    Pos3Color4Vertex { position: [0.0, water_y, 0.0], color: wc },
                    Pos3Color4Vertex { position: [x_max, water_y, 0.0], color: wc },
                    Pos3Color4Vertex { position: [x_max, water_y, 0.0], color: wc },
                    Pos3Color4Vertex { position: [x_max, water_y, z_max], color: wc },
                    Pos3Color4Vertex { position: [x_max, water_y, z_max], color: wc },
                    Pos3Color4Vertex { position: [0.0, water_y, z_max], color: wc },
                    Pos3Color4Vertex { position: [0.0, water_y, z_max], color: wc },
                    Pos3Color4Vertex { position: [0.0, water_y, 0.0], color: wc },
                ];

                let bed_top = BED_HEIGHT_CELLS as f32 * CELL_SIZE;
                let bed_vertices = [
                    Pos3Color4Vertex { position: [0.0, bed_top, 0.0], color: BED_COLOR },
                    Pos3Color4Vertex { position: [x_max, bed_top, 0.0], color: BED_COLOR },
                    Pos3Color4Vertex { position: [x_max, bed_top, z_max], color: BED_COLOR },
                    Pos3Color4Vertex { position: [0.0, bed_top, 0.0], color: BED_COLOR },
                    Pos3Color4Vertex { position: [x_max, bed_top, z_max], color: BED_COLOR },
                    Pos3Color4Vertex { position: [0.0, bed_top, z_max], color: BED_COLOR },
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
                ctx.queue.write_buffer(
                    self.bed_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&bed_vertices),
                );

                // Camera
                let window = self.window.as_ref().unwrap();
                let size = window.inner_size();
                let aspect = size.width as f32 / size.height as f32;
                let view_proj = self.camera.view_proj_matrix(aspect);
                let uniforms = Uniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    camera_pos: self.camera.position().to_array(),
                    point_size: 6.0,
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
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
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
                    pass.draw(0..32, 0..1);

                    pass.set_pipeline(self.solid_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, self.uniform_bind_group.as_ref().unwrap(), &[]);
                    pass.set_vertex_buffer(0, self.bed_buffer.as_ref().unwrap().slice(..));
                    pass.draw(0..6, 0..1);

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
