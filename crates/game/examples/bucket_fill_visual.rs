//! Bucket Fill Volume Conservation Test
//!
//! Tests that FLIP simulation conserves volume:
//! 1. Emit a known number of particles (enough for half-grid fill)
//! 2. Stop emitter and wait for settling
//! 3. Verify final volume matches expected ±5%
//!
//! Controls: SPACE=pause, R=reset, E=toggle emitter, Mouse=rotate, Scroll=zoom

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

// Grid configuration
const GRID_WIDTH: usize = 16;   // X cells
const GRID_HEIGHT: usize = 24;  // Y cells (tall for filling)
const GRID_DEPTH: usize = 16;   // Z cells
const CELL_SIZE: f32 = 0.05;    // 5cm cells

// Emitter configuration
const EMITTER_CENTER_X: f32 = GRID_WIDTH as f32 * CELL_SIZE * 0.5;
const EMITTER_CENTER_Z: f32 = GRID_DEPTH as f32 * CELL_SIZE * 0.5;
const EMITTER_Y: f32 = (GRID_HEIGHT - 2) as f32 * CELL_SIZE; // Near top
const EMITTER_RADIUS: f32 = CELL_SIZE * 2.0; // 2 cell radius stream
const PARTICLES_PER_FRAME: usize = 50; // Emission rate
const EMITTER_VELOCITY: f32 = -0.5; // Downward initial velocity

// Volume target calculation:
// Interior cells (excluding 1-cell walls): 14 x 14 in XZ plane
// Target fill: 12 cells high (half of 24)
// Particles per cell: 8 (standard 2x2x2 sampling)
const INTERIOR_WIDTH: usize = GRID_WIDTH - 2;   // 14 cells
const INTERIOR_DEPTH: usize = GRID_DEPTH - 2;   // 14 cells
const TARGET_FILL_CELLS: usize = GRID_HEIGHT / 2; // 12 cells high
const PARTICLES_PER_CELL: usize = 8;

const TARGET_PARTICLE_COUNT: usize = INTERIOR_WIDTH * TARGET_FILL_CELLS * INTERIOR_DEPTH * PARTICLES_PER_CELL;
// = 14 * 12 * 14 * 8 = 18,816 particles

const TARGET_FILL_HEIGHT: f32 = TARGET_FILL_CELLS as f32 * CELL_SIZE; // 0.6m
const SETTLING_OBSERVATION_TIME: f32 = 30.0; // Observe settling for 30 seconds
const VOLUME_TOLERANCE: f32 = 0.05; // ±5% volume tolerance

const MAX_PARTICLES: usize = 200_000; // Room for lots of particles
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

/// Setup solid boundary cells - walls on all sides except top
fn setup_solid_boundaries(cell_types: &mut [u32], w: usize, h: usize, d: usize) {
    const CELL_SOLID: u32 = 2;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * w * h + y * w + x;
                let at_x_edge = x == 0 || x == w - 1;
                let at_y_floor = y == 0;
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

    camera: Camera,
    paused: bool,
    emitter_active: bool,
    frame: u32,
    sim_time: f32,
    dragging: bool,
    last_mouse: (f32, f32),

    // Simple RNG state
    rng_state: u32,

    // Fill tracking
    emitter_stopped_at: Option<f32>, // sim_time when emitter auto-stopped
    settling_complete: bool,
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
            camera: Camera::new(0.5, 0.3, 1.5, domain_center),
            paused: false,
            emitter_active: true,
            frame: 0,
            sim_time: 0.0,
            dragging: false,
            last_mouse: (0.0, 0.0),
            rng_state: 12345,
            emitter_stopped_at: None,
            settling_complete: false,
        }
    }

    fn reset(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();
        self.frame = 0;
        self.sim_time = 0.0;
        self.emitter_active = true;
        self.rng_state = 12345;
        self.emitter_stopped_at = None;
        self.settling_complete = false;

        println!("\n=== RESET ===");
        println!("Empty bucket - emitter active");
        println!("Target: {} particles (expect {:.2}m fill height)", TARGET_PARTICLE_COUNT, TARGET_FILL_HEIGHT);
        println!("Will observe settling for {:.0}s, then check volume ±{:.0}%", SETTLING_OBSERVATION_TIME, VOLUME_TOLERANCE * 100.0);
        println!("Press E to toggle emitter, SPACE to pause");
    }

    /// Simple xorshift random number generator
    fn random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        (self.rng_state as f32) / (u32::MAX as f32)
    }

    /// Get the current water surface height (95th percentile Y to ignore splashes)
    fn water_surface_height(&self) -> f32 {
        if self.positions.is_empty() {
            return 0.0;
        }

        // Use 95th percentile to get stable surface level (ignores spray/splashes)
        let mut y_values: Vec<f32> = self.positions.iter().map(|p| p.y).collect();
        y_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx_95 = (y_values.len() as f32 * 0.95) as usize;
        y_values[idx_95.min(y_values.len() - 1)]
    }

    /// Calculate expected surface height from particle count
    /// Assumes uniform distribution in the interior volume
    fn expected_surface_height(&self) -> f32 {
        let particle_count = self.positions.len();
        // particles = interior_width * interior_depth * height_cells * particles_per_cell
        // height_cells = particles / (interior_width * interior_depth * particles_per_cell)
        let height_cells = particle_count as f32 / (INTERIOR_WIDTH * INTERIOR_DEPTH * PARTICLES_PER_CELL) as f32;
        // Add 1 cell for floor offset (water starts at y = CELL_SIZE, not y = 0)
        (height_cells + 1.0) * CELL_SIZE
    }

    /// Check if we've emitted enough particles
    fn check_fill_complete(&mut self) {
        if self.emitter_active && self.emitter_stopped_at.is_none() {
            if self.positions.len() >= TARGET_PARTICLE_COUNT {
                self.emitter_active = false;
                self.emitter_stopped_at = Some(self.sim_time);
                let expected_height = self.expected_surface_height();
                println!("\n========================================");
                println!("FILL COMPLETE at t={:.1}s", self.sim_time);
                println!("Particles emitted: {} (target: {})", self.positions.len(), TARGET_PARTICLE_COUNT);
                println!("Expected fill height: {:.3}m", expected_height);
                println!("Emitter stopped - observing settling for {:.0}s...", SETTLING_OBSERVATION_TIME);
                println!("Volume tolerance: ±{:.0}%", VOLUME_TOLERANCE * 100.0);
                println!("========================================\n");
            }
        }
    }

    /// Emit new particles from the stream
    fn emit_particles(&mut self) {
        if !self.emitter_active || self.positions.len() >= MAX_PARTICLES - PARTICLES_PER_FRAME {
            return;
        }

        for _ in 0..PARTICLES_PER_FRAME {
            // Random position within emitter radius (circular distribution)
            let angle = self.random() * std::f32::consts::TAU;
            let r = self.random().sqrt() * EMITTER_RADIUS;

            let pos = Vec3::new(
                EMITTER_CENTER_X + r * angle.cos(),
                EMITTER_Y,
                EMITTER_CENTER_Z + r * angle.sin(),
            );

            // Initial downward velocity with small random spread
            let vel = Vec3::new(
                (self.random() - 0.5) * 0.1,
                EMITTER_VELOCITY,
                (self.random() - 0.5) * 0.1,
            );

            self.positions.push(pos);
            self.velocities.push(vel);
            self.c_matrices.push(Mat3::ZERO);
            self.densities.push(1.0);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("BUCKET FILL - Stream of water filling container")
                .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

        // Setup cell_types: solid walls on all sides
        self.cell_types = vec![0u32; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
        setup_solid_boundaries(&mut self.cell_types, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH);

        // FLIP solver with dynamic settings for splashing
        let mut flip = GpuFlip3D::new(
            &ctx.device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        // Normal FLIP settings - post-pressure BC removed from flip_3d.rs
        flip.flip_ratio = 0.95;      // Standard FLIP ratio for splashy behavior
        flip.slip_factor = 0.5;      // Floor friction (only applied before pressure solve now)
        flip.open_boundaries = 0;    // Closed boundaries
        flip.use_sorted_p2g = true;  // Use optimized sorting
        flip.density_projection_enabled = false;  // OFF - implementation needs more work
        flip.water_rest_density = 8.0;  // 8 particles per cell (standard 2x2x2 seeding)
        flip.vorticity_epsilon = 0.0;  // OFF for settling test

        self.flip = Some(flip);

        // Start empty
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

        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.point_pipeline = Some(point_pipeline);
        self.line_pipeline = Some(line_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.line_buffer = Some(line_buffer);
        self.ctx = Some(ctx);

        println!("\n=== BUCKET FILL VISUAL TEST ===");
        println!("Grid: {}x{}x{} cells @ {:.3}m", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("Domain: {:.2}m x {:.2}m x {:.2}m",
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE);
        println!("Emitter: center=({:.2}, {:.2}, {:.2}), radius={:.3}m",
            EMITTER_CENTER_X, EMITTER_Y, EMITTER_CENTER_Z, EMITTER_RADIUS);
        println!("Emission rate: {} particles/frame @ 60fps", PARTICLES_PER_FRAME);
        println!("");
        println!("TEST PARAMETERS:");
        println!("  Target particles: {} (for {:.2}m fill)", TARGET_PARTICLE_COUNT, TARGET_FILL_HEIGHT);
        println!("  Settling time:    {:.0}s", SETTLING_OBSERVATION_TIME);
        println!("  Volume tolerance: ±{:.0}%", VOLUME_TOLERANCE * 100.0);
        println!("  FLIP ratio:       0.95 (splashy behavior)");
        println!("");
        println!("Controls: SPACE=pause, R=reset, E=toggle emitter");
        println!("          Mouse drag=rotate, Scroll=zoom");
        println!("==========================================\n");
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
                KeyCode::KeyE => {
                    self.emitter_active = !self.emitter_active;
                    println!("Emitter: {}", if self.emitter_active { "ON" } else { "OFF" });
                }
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
                let dt = 1.0 / 60.0;

                // Physics step (emit particles first, before borrowing ctx)
                if !self.paused {
                    // Emit new particles
                    self.emit_particles();

                    // Check if we've reached target fill height
                    self.check_fill_complete();
                }

                let ctx = self.ctx.as_ref().unwrap();

                // Run physics simulation
                if !self.paused && !self.positions.is_empty() {
                    self.flip.as_mut().unwrap().step(
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
                        200, // pressure iterations (increased to test convergence)
                    );

                    self.frame += 1;
                    self.sim_time += dt;

                    // Check if settling observation is complete
                    if let Some(stopped_at) = self.emitter_stopped_at {
                        let settling_time = self.sim_time - stopped_at;
                        if settling_time >= SETTLING_OBSERVATION_TIME && !self.settling_complete {
                            self.settling_complete = true;
                            let max_vel = self.velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                            let avg_vel = self.velocities.iter().map(|v| v.length()).sum::<f32>() / self.velocities.len() as f32;

                            // Volume conservation check
                            let actual_surface = self.water_surface_height();
                            let expected_surface = self.expected_surface_height();
                            let volume_error = (actual_surface - expected_surface).abs() / expected_surface;
                            let volume_ok = volume_error <= VOLUME_TOLERANCE;

                            println!("\n========================================");
                            println!("VOLUME CONSERVATION TEST COMPLETE");
                            println!("========================================");
                            println!("Time: t={:.1}s (settled for {:.0}s)", self.sim_time, settling_time);
                            println!("Particles: {}", self.positions.len());
                            println!("");
                            println!("VOLUME CHECK:");
                            println!("  Expected surface: {:.3}m", expected_surface);
                            println!("  Actual surface:   {:.3}m", actual_surface);
                            println!("  Error: {:.1}% (tolerance: ±{:.0}%)", volume_error * 100.0, VOLUME_TOLERANCE * 100.0);
                            if volume_ok {
                                println!("  ✓ PASS - Volume conserved within tolerance");
                            } else {
                                println!("  ✗ FAIL - Volume NOT conserved!");
                                println!("  Water collapsed by {:.3}m ({:.1}%)", expected_surface - actual_surface, (1.0 - actual_surface/expected_surface) * 100.0);
                            }
                            println!("");
                            println!("VELOCITY CHECK:");
                            println!("  Max velocity: {:.4} m/s", max_vel);
                            println!("  Avg velocity: {:.4} m/s", avg_vel);
                            if max_vel < 0.01 {
                                println!("  ✓ EQUILIBRIUM (max_vel < 0.01 m/s)");
                            } else if max_vel < 0.1 {
                                println!("  ~ MOSTLY SETTLED (max_vel < 0.1 m/s)");
                            } else {
                                println!("  ✗ STILL MOVING (max_vel >= 0.1 m/s)");
                            }
                            println!("========================================\n");
                        }
                    }

                    // Print stats every second
                    if self.frame % 60 == 0 {
                        let max_vel = self.velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                        let avg_vel = self.velocities.iter().map(|v| v.length()).sum::<f32>() / self.velocities.len() as f32;
                        let actual_surface = self.water_surface_height();
                        let expected_surface = self.expected_surface_height();

                        // Status indicator
                        let status = if let Some(stopped_at) = self.emitter_stopped_at {
                            let settling_time = self.sim_time - stopped_at;
                            format!("SETTLING {:.0}s/{:.0}s", settling_time, SETTLING_OBSERVATION_TIME)
                        } else {
                            let pct = (self.positions.len() as f32 / TARGET_PARTICLE_COUNT as f32 * 100.0).min(100.0);
                            format!("FILLING {:.0}%", pct)
                        };

                        println!(
                            "t={:5.1}s | {} | n={:6} | vel: max={:.3} avg={:.3} | surface={:.3}m (expect {:.3}m)",
                            self.sim_time, status, self.positions.len(), max_vel, avg_vel, actual_surface, expected_surface
                        );
                    }
                }

                // Color particles by velocity magnitude
                let mut vertices: Vec<Pos3Color4Vertex> = Vec::with_capacity(self.positions.len() * 6);
                for (i, pos) in self.positions.iter().enumerate() {
                    let vel_mag = self.velocities[i].length();
                    // Color: blue = slow, cyan = medium, white = fast
                    let t = (vel_mag / 2.0).min(1.0);
                    let color = [
                        0.3 + 0.7 * t,      // R
                        0.5 + 0.5 * t,      // G
                        1.0,                // B
                        0.85,
                    ];
                    for _ in 0..6 {
                        vertices.push(Pos3Color4Vertex {
                            position: [pos.x, pos.y, pos.z],
                            color,
                        });
                    }
                }

                // Build bounding box lines
                let x_max = GRID_WIDTH as f32 * CELL_SIZE;
                let y_max = GRID_HEIGHT as f32 * CELL_SIZE;
                let z_max = GRID_DEPTH as f32 * CELL_SIZE;
                let lc = [0.4f32, 0.4, 0.4, 1.0];
                let ec = [1.0f32, 0.8, 0.2, 1.0]; // Emitter indicator (gold)

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
                    // Emitter indicator (cross at emitter position)
                    Pos3Color4Vertex { position: [EMITTER_CENTER_X - EMITTER_RADIUS, EMITTER_Y, EMITTER_CENTER_Z], color: ec },
                    Pos3Color4Vertex { position: [EMITTER_CENTER_X + EMITTER_RADIUS, EMITTER_Y, EMITTER_CENTER_Z], color: ec },
                    Pos3Color4Vertex { position: [EMITTER_CENTER_X, EMITTER_Y, EMITTER_CENTER_Z - EMITTER_RADIUS], color: ec },
                    Pos3Color4Vertex { position: [EMITTER_CENTER_X, EMITTER_Y, EMITTER_CENTER_Z + EMITTER_RADIUS], color: ec },
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
                    point_size: 4.0, // Smaller particles for dense fill
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

                    // Draw bounding box and emitter
                    pass.set_pipeline(self.line_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, self.uniform_bind_group.as_ref().unwrap(), &[]);
                    pass.set_vertex_buffer(0, self.line_buffer.as_ref().unwrap().slice(..));
                    pass.draw(0..28, 0..1);

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
