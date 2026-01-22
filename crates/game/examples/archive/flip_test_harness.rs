//! Visual test harness for FLIP component tests.
//!
//! Usage:
//!   cargo run --example flip_test_harness --release
//!
//! Controls:
//!   1-6   = Switch between tests
//!   SPACE = Pause/Resume
//!   R     = Reset current test
//!   ESC   = Exit
//!   Mouse = Rotate camera
//!   Scroll = Zoom

use bytemuck::{Pod, Zeroable};
use game::example_utils::{Camera, WgpuContext, create_depth_view, Pos3Color4Vertex};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_PARTICLES: usize = 10000;

/// Test config file paths
const TEST_CONFIGS: &[&str] = &[
    "scenarios/flip_tests/test_01_stationary.json",
    "scenarios/flip_tests/test_02_advection.json",
    "scenarios/flip_tests/test_03_gravity.json",
    "scenarios/flip_tests/test_04_conservation.json",
    "scenarios/flip_tests/test_05_nan_check.json",
    "scenarios/flip_tests/test_06_settling.json",
];

//=============================================================================
// Test Config Format
//=============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlipTestConfig {
    pub name: String,
    pub description: String,
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,
    pub gravity: f32,
    pub open_top: bool,
    pub cpu_boundary_clamp: bool,
    pub vorticity_epsilon: f32,
    pub pressure_iterations: u32,
    pub particles: Vec<ParticleInit>,
    #[serde(default)]
    pub fill_region: Option<FillRegion>,
    pub expected: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticleInit {
    pub position: [f32; 3],
    #[serde(default)]
    pub velocity: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FillRegion {
    pub min_cell: [usize; 3],
    pub max_cell: [usize; 3],
}

impl FlipTestConfig {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }
}

//=============================================================================
// Shaders
//=============================================================================

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

//=============================================================================
// Main
//=============================================================================

fn main() {
    println!("=== FLIP Visual Test Harness ===");
    println!();
    println!("Loading test configs...");

    // Load all test configs
    let mut configs = Vec::new();
    for (i, path) in TEST_CONFIGS.iter().enumerate() {
        match FlipTestConfig::load(Path::new(path)) {
            Ok(c) => {
                println!("  [{}] {} - {}", i + 1, c.name, c.description);
                configs.push(c);
            }
            Err(e) => {
                eprintln!("  [{}] FAILED to load '{}': {}", i + 1, path, e);
            }
        }
    }

    if configs.is_empty() {
        eprintln!("No test configs loaded!");
        std::process::exit(1);
    }

    println!();
    println!("Controls:");
    println!("  1-6   = Switch test");
    println!("  SPACE = Pause/Resume");
    println!("  R     = Reset");
    println!("  ESC   = Exit");
    println!("  Mouse = Rotate");
    println!("  Scroll = Zoom");
    println!();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(configs);
    event_loop.run_app(&mut app).unwrap();
}

//=============================================================================
// App
//=============================================================================

struct App {
    configs: Vec<FlipTestConfig>,
    current_test: usize,

    // Window/GPU
    window: Option<Arc<Window>>,
    ctx: Option<WgpuContext>,
    depth_view: Option<wgpu::TextureView>,

    // Pipelines (shared across tests)
    point_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,
    vertex_buffer: Option<wgpu::Buffer>,
    line_buffer: Option<wgpu::Buffer>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    pipeline_layout: Option<wgpu::PipelineLayout>,

    // Simulation (recreated per test)
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
    initial_y_max: f32,
    dragging: bool,
    last_mouse: (f32, f32),
}

impl App {
    fn new(configs: Vec<FlipTestConfig>) -> Self {
        Self {
            configs,
            current_test: 0,
            window: None,
            ctx: None,
            depth_view: None,
            point_pipeline: None,
            line_pipeline: None,
            uniform_buffer: None,
            uniform_bind_group: None,
            vertex_buffer: None,
            line_buffer: None,
            bind_group_layout: None,
            pipeline_layout: None,
            flip: None,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            camera: Camera::new(0.5, 0.3, 1.0, Vec3::ZERO),
            paused: false,
            frame: 0,
            initial_count: 0,
            initial_y_max: 0.0,
            dragging: false,
            last_mouse: (0.0, 0.0),
        }
    }

    fn config(&self) -> &FlipTestConfig {
        &self.configs[self.current_test]
    }

    fn switch_test(&mut self, test_idx: usize) {
        if test_idx >= self.configs.len() {
            return;
        }
        if test_idx == self.current_test && self.flip.is_some() {
            // Same test, just reset
            self.reset_particles();
            return;
        }

        self.current_test = test_idx;

        // Clone config values we need before mutable borrows
        let config = self.configs[self.current_test].clone();

        println!();
        println!("========================================");
        println!("TEST {}: {}", test_idx + 1, config.name);
        println!("----------------------------------------");
        println!("Description: {}", config.description);
        println!("Expected: {}", config.expected);
        println!("Grid: {}x{}x{}, cell={}", config.grid_width, config.grid_height, config.grid_depth, config.cell_size);
        println!("Gravity: {}, Open top: {}, CPU clamp: {}", config.gravity, config.open_top, config.cpu_boundary_clamp);
        println!("========================================");

        // Update window title
        if let Some(ref window) = self.window {
            window.set_title(&format!("[{}] {} - {}", test_idx + 1, config.name, config.description));
        }

        // Recreate FLIP solver with new grid dimensions
        if let Some(ref ctx) = self.ctx {
            let mut flip = GpuFlip3D::new(
                &ctx.device,
                config.grid_width as u32,
                config.grid_height as u32,
                config.grid_depth as u32,
                config.cell_size,
                MAX_PARTICLES,
            );
            flip.vorticity_epsilon = config.vorticity_epsilon;
            flip.open_boundaries = if config.open_top { 1 } else { 0 };
            flip.density_projection_enabled = false; // Disable for basic tests
            self.flip = Some(flip);
        }

        // Setup cell types for new grid
        self.setup_cell_types();

        // Update camera for new grid
        let grid_center = Vec3::new(
            (config.grid_width as f32 * config.cell_size) / 2.0,
            (config.grid_height as f32 * config.cell_size) / 2.0,
            (config.grid_depth as f32 * config.cell_size) / 2.0,
        );
        let distance = (config.grid_width.max(config.grid_depth) as f32 * config.cell_size) * 2.0;
        self.camera = Camera::new(0.5, 0.3, distance, grid_center);

        // Reset particles
        self.reset_particles();
    }

    fn reset_particles(&mut self) {
        let config = &self.configs[self.current_test];

        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();

        let cell_size = config.cell_size;

        // Add explicit particles
        for p in &config.particles {
            self.positions.push(Vec3::from_array(p.position));
            self.velocities.push(Vec3::from_array(p.velocity));
            self.c_matrices.push(Mat3::ZERO);
            self.densities.push(1.0);
        }

        // Add fill region particles
        if let Some(ref fill) = config.fill_region {
            for x in fill.min_cell[0]..fill.max_cell[0] {
                for y in fill.min_cell[1]..fill.max_cell[1] {
                    for z in fill.min_cell[2]..fill.max_cell[2] {
                        let p = Vec3::new(
                            (x as f32 + 0.5) * cell_size,
                            (y as f32 + 0.5) * cell_size,
                            (z as f32 + 0.5) * cell_size,
                        );
                        self.positions.push(p);
                        self.velocities.push(Vec3::ZERO);
                        self.c_matrices.push(Mat3::ZERO);
                        self.densities.push(1.0);
                    }
                }
            }
        }

        self.initial_count = self.positions.len();
        self.initial_y_max = self.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
        self.frame = 0;
        self.paused = false;

        println!("RESET: {} particles, Y_max={:.4}", self.initial_count, self.initial_y_max);
    }

    fn setup_cell_types(&mut self) {
        let config = &self.configs[self.current_test];
        let w = config.grid_width;
        let h = config.grid_height;
        let d = config.grid_depth;
        let open_top = config.open_top;

        self.cell_types = vec![0u32; w * h * d];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let idx = z * w * h + y * w + x;
                    let is_floor = y == 0;
                    let is_ceiling = y == h - 1;
                    let is_x_wall = x == 0 || x == w - 1;
                    let is_z_wall = z == 0 || z == d - 1;

                    if is_floor || is_x_wall || is_z_wall {
                        self.cell_types[idx] = 2; // SOLID
                    }
                    if is_ceiling && !open_top {
                        self.cell_types[idx] = 2; // SOLID
                    }
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("FLIP Test Harness - Press 1-6 to switch tests")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

        // Create shared pipelines
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
        self.bind_group_layout = Some(bind_group_layout);
        self.pipeline_layout = Some(pipeline_layout);
        self.ctx = Some(ctx);

        // Initialize first test
        self.switch_test(0);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state: ElementState::Pressed, .. },
                ..
            } => match code {
                KeyCode::Digit1 => self.switch_test(0),
                KeyCode::Digit2 => self.switch_test(1),
                KeyCode::Digit3 => self.switch_test(2),
                KeyCode::Digit4 => self.switch_test(3),
                KeyCode::Digit5 => self.switch_test(4),
                KeyCode::Digit6 => self.switch_test(5),
                KeyCode::Space => {
                    self.paused = !self.paused;
                    println!("{}", if self.paused { "PAUSED" } else { "RUNNING" });
                }
                KeyCode::KeyR => self.reset_particles(),
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
                let config = self.config().clone();

                if let Some(ref mut flip) = self.flip {
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
                            config.gravity,
                            0.0,
                            config.pressure_iterations,
                        );

                        // No CPU clamping - core FLIP should handle boundaries

                        self.frame += 1;

                        // Status output
                        if self.frame % 30 == 0 {
                            let max_vel = self.velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                            let min_y = self.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
                            let max_y = self.positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
                            let floor_y = config.cell_size;
                            let dist_from_floor = min_y - floor_y;

                            println!(
                                "[Test {}] Frame {:4}: n={:4} vel={:.4} y=[{:.4},{:.4}] floor={:.2} dist={:.4}",
                                self.current_test + 1, self.frame, self.positions.len(), max_vel, min_y, max_y, floor_y, dist_from_floor
                            );

                            // Detailed output for single-particle tests (1-3)
                            if self.positions.len() == 1 && self.frame <= 120 {
                                let p = self.positions[0];
                                let v = self.velocities[0];
                                println!(
                                    "  [DETAIL] pos=({:.4},{:.4},{:.4}) vel=({:.4},{:.4},{:.4})",
                                    p.x, p.y, p.z, v.x, v.y, v.z
                                );
                            }

                            // Automated PASS/FAIL check at frame 600
                            if self.frame == 600 {
                                let test_passed = match self.current_test {
                                    0 => {
                                        // Test 1: Stationary - particle should not have moved much
                                        let delta = (self.positions[0] - Vec3::new(0.4, 0.4, 0.4)).length();
                                        let pass = delta < 0.02 && max_vel < 0.01;
                                        println!("  [CHECK] Test 1: delta={:.4} vel={:.4} -> {}", delta, max_vel, if pass { "PASS" } else { "FAIL" });
                                        pass
                                    }
                                    1 => {
                                        // Test 2: P2G/G2P Velocity Transfer - particle should move in +X
                                        let pos = self.positions[0];
                                        let moved_x = pos.x > 0.5; // Started at 0.4, should have moved right
                                        let vel_preserved = self.velocities[0].x > 0.3; // Started at 0.5, FLIP preserves most
                                        let pass = moved_x && vel_preserved;
                                        println!("  [CHECK] Test 2: x={:.4} vel_x={:.4} -> {}", pos.x, self.velocities[0].x, if pass { "PASS" } else { "FAIL" });
                                        pass
                                    }
                                    2 => {
                                        // Test 3: Gravity Accumulation - velocity should be increasingly negative
                                        let vel_y = self.velocities[0].y;
                                        let pass = vel_y < -1.0; // After 600 frames, should have significant downward vel
                                        println!("  [CHECK] Test 3: vel_y={:.4} (should be < -1.0) -> {}", vel_y, if pass { "PASS" } else { "FAIL" });
                                        pass
                                    }
                                    3 => {
                                        // Test 4: Conservation - particle count should be 64
                                        let pass = self.positions.len() == 64;
                                        println!("  [CHECK] Test 4: count={} expected=64 -> {}", self.positions.len(), if pass { "PASS" } else { "FAIL" });
                                        pass
                                    }
                                    4 => {
                                        // Test 5: No NaN - all positions should be finite
                                        let all_finite = self.positions.iter().all(|p| p.is_finite()) && self.velocities.iter().all(|v| v.is_finite());
                                        println!("  [CHECK] Test 5: all_finite={} -> {}", all_finite, if all_finite { "PASS" } else { "FAIL" });
                                        all_finite
                                    }
                                    5 => {
                                        // Test 6: Settling - Y should have decreased, velocity low
                                        let settled = max_y < self.initial_y_max && max_vel < 0.5;
                                        println!("  [CHECK] Test 6: y_max={:.4} (was {:.4}) vel={:.4} -> {}", max_y, self.initial_y_max, max_vel, if settled { "PASS" } else { "FAIL" });
                                        settled
                                    }
                                    _ => true
                                };
                                if !test_passed {
                                    println!("  *** TEST {} FAILED ***", self.current_test + 1);
                                }
                            }

                            // Check for NaN
                            for (i, p) in self.positions.iter().enumerate() {
                                if !p.is_finite() {
                                    println!("ERROR: NaN at particle {} position {:?}", i, p);
                                }
                            }
                            for (i, v) in self.velocities.iter().enumerate() {
                                if !v.is_finite() {
                                    println!("ERROR: NaN at particle {} velocity {:?}", i, v);
                                }
                            }
                        }
                    }
                }

                // Color particles by velocity magnitude
                let mut vertices: Vec<Pos3Color4Vertex> = Vec::with_capacity(self.positions.len() * 6);
                for (pos, vel) in self.positions.iter().zip(self.velocities.iter()) {
                    let speed = vel.length();
                    let t = (speed * 5.0).min(1.0);
                    let color = [0.2 + t * 0.3, 0.5 + t * 0.5, 1.0 - t * 0.5, 0.9];

                    for _ in 0..6 {
                        vertices.push(Pos3Color4Vertex {
                            position: [pos.x, pos.y, pos.z],
                            color,
                        });
                    }
                }

                // Bounding box
                let cell = config.cell_size;
                let x_min = cell;
                let x_max = (config.grid_width as f32 - 1.0) * cell;
                let y_min = cell;
                let y_max = (config.grid_height as f32 - 1.0) * cell;
                let z_min = cell;
                let z_max = (config.grid_depth as f32 - 1.0) * cell;

                let lc = if config.open_top {
                    [0.3f32, 0.6, 0.3, 1.0] // Green for open top
                } else {
                    [0.5f32, 0.5, 0.5, 1.0] // Gray for closed
                };

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
                    point_size: 12.0,
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
