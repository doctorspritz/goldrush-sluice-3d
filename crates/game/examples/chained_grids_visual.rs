//! Visual Chained Grids Demo
//!
//! Two small FLIP grids connected in series with real-time rendering.
//! Demonstrates domain decomposition for fluid simulation.
//!
//! Controls:
//! - Mouse drag: rotate camera
//! - Scroll: zoom
//! - Space: pause/resume
//! - R: reset
//!
//! Run with: cargo run --example chained_grids_visual --release

use bytemuck::{Pod, Zeroable};
use game::example_utils::{Camera, WgpuContext, create_depth_view, Pos3Color4Vertex};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

// Custom shader that renders particles as screenspace quads
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

    // Project center to clip space
    let clip_center = uniforms.view_proj * vec4<f32>(in.position, 1.0);

    // Get quad corner offset based on vertex index (0-5 for 2 triangles)
    // Triangle 1: 0,1,2 = bottom-left, top-left, top-right
    // Triangle 2: 3,4,5 = bottom-left, top-right, bottom-right
    let corner_idx = in.vertex_index % 6u;
    var offset: vec2<f32>;
    switch corner_idx {
        case 0u: { offset = vec2<f32>(-1.0, -1.0); }  // bottom-left
        case 1u: { offset = vec2<f32>(-1.0, 1.0); }   // top-left
        case 2u: { offset = vec2<f32>(1.0, 1.0); }    // top-right
        case 3u: { offset = vec2<f32>(-1.0, -1.0); }  // bottom-left
        case 4u: { offset = vec2<f32>(1.0, 1.0); }    // top-right
        case 5u: { offset = vec2<f32>(1.0, -1.0); }   // bottom-right
        default: { offset = vec2<f32>(0.0, 0.0); }
    }

    // Scale offset by point size and apply in clip space (screenspace pixels)
    let size = uniforms.point_size / 500.0;  // Normalize to reasonable scale
    out.clip_position = clip_center + vec4<f32>(offset * size, 0.0, 0.0);
    out.color = in.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

// Simple line shader (reuse from BASIC_SHADER style)
const LINE_SHADER: &str = r#"
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
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Grid dimensions (smaller grids to see volume building)
const GRID_WIDTH: usize = 12;
const GRID_HEIGHT: usize = 10;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.04;
const MAX_PARTICLES: usize = 50000;

// Outflow only allowed above this height (70% up - higher weir to see more fill)
const OUTFLOW_MIN_Y: f32 = (GRID_HEIGHT as f32 * 0.7) * CELL_SIZE;

// Simulation
const DT: f32 = 1.0 / 120.0;
const SUBSTEPS: u32 = 2;
const PRESSURE_ITERS: u32 = 50;
const GRAVITY: f32 = -9.8;
const FLOW_ACCEL: f32 = 2.0;  // Slower flow
const INLET_RATE: usize = 15; // Slower inlet to see fill
const INLET_VELOCITY: f32 = 0.2; // Slower injection

// Particle sprite size (in pixels)
const POINT_SIZE: f32 = 4.0;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    point_size: f32,
}

fn rand_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32) / (u32::MAX as f32)
}

fn grid_boundary_lines(offset: Vec3, color: [f32; 4]) -> Vec<Pos3Color4Vertex> {
    // Draw where particles can ACTUALLY go (must match CPU collision bounds!)
    // CPU collision uses: min=1.0*CELL, max=(N-1.5)*CELL
    let x_min = CELL_SIZE * 1.0;
    let x_max = (GRID_WIDTH as f32 - 1.5) * CELL_SIZE;   // Matches exit_x
    let y_min = CELL_SIZE * 1.0;
    let y_max = (GRID_HEIGHT as f32 - 1.5) * CELL_SIZE;  // Matches ceiling_y
    let z_min = CELL_SIZE * 1.0;
    let z_max = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;   // Matches max_z

    // 8 corners of the fluid region box
    let corners = [
        offset + Vec3::new(x_min, y_min, z_min),
        offset + Vec3::new(x_max, y_min, z_min),
        offset + Vec3::new(x_max, y_max, z_min),
        offset + Vec3::new(x_min, y_max, z_min),
        offset + Vec3::new(x_min, y_min, z_max),
        offset + Vec3::new(x_max, y_min, z_max),
        offset + Vec3::new(x_max, y_max, z_max),
        offset + Vec3::new(x_min, y_max, z_max),
    ];
    // 12 edges as line pairs
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // front face
        (4, 5), (5, 6), (6, 7), (7, 4), // back face
        (0, 4), (1, 5), (2, 6), (3, 7), // connecting edges
    ];
    let mut lines = Vec::with_capacity(28); // 24 + 4 for weir line
    for (a, b) in edges {
        lines.push(Pos3Color4Vertex { position: corners[a].to_array(), color });
        lines.push(Pos3Color4Vertex { position: corners[b].to_array(), color });
    }

    // Add weir line (outflow threshold) on exit face - yellow/orange color
    let weir_color = [1.0, 0.8, 0.2, 1.0];
    lines.push(Pos3Color4Vertex { position: (offset + Vec3::new(x_max, OUTFLOW_MIN_Y, z_min)).to_array(), color: weir_color });
    lines.push(Pos3Color4Vertex { position: (offset + Vec3::new(x_max, OUTFLOW_MIN_Y, z_max)).to_array(), color: weir_color });

    lines
}

struct GridSegment {
    gpu: GpuFlip3D,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,
    world_offset: Vec3,
    input_buffer: Vec<(Vec3, Vec3)>,
}

impl GridSegment {
    fn new(device: &wgpu::Device, world_offset: Vec3) -> Self {
        let mut gpu = GpuFlip3D::new(
            device, GRID_WIDTH as u32, GRID_HEIGHT as u32, GRID_DEPTH as u32, CELL_SIZE, MAX_PARTICLES,
        );
        gpu.vorticity_epsilon = 0.0;
        gpu.open_boundaries = 2;

        let mut cell_types = vec![0u32; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
        for z in 0..GRID_DEPTH {
            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    let idx = z * GRID_WIDTH * GRID_HEIGHT + y * GRID_WIDTH + x;
                    // Solid boundaries:
                    // - Floor: y=0
                    // - Ceiling: y=GRID_HEIGHT-1 (closed top)
                    // - Side walls: z=0 and z=GRID_DEPTH-1
                    // - Inlet wall: x=0
                    // - Exit wall below weir: x=GRID_WIDTH-1 && y < outflow threshold
                    let outflow_cell_y = (OUTFLOW_MIN_Y / CELL_SIZE) as usize;
                    let is_floor = y == 0;
                    let is_ceiling = y == GRID_HEIGHT - 1;
                    let is_z_wall = z == 0 || z == GRID_DEPTH - 1;
                    let is_inlet = x == 0;
                    let is_exit_wall = x == GRID_WIDTH - 1 && y < outflow_cell_y;

                    if is_floor || is_ceiling || is_z_wall || is_inlet || is_exit_wall {
                        cell_types[idx] = 2; // SOLID
                    }
                }
            }
        }

        Self {
            gpu, positions: Vec::new(), velocities: Vec::new(), c_matrices: Vec::new(),
            densities: Vec::new(), cell_types, world_offset, input_buffer: Vec::new(),
        }
    }

    fn inject(&mut self, pos: Vec3, vel: Vec3) {
        self.positions.push(pos);
        self.velocities.push(vel);
        self.c_matrices.push(Mat3::ZERO);
        self.densities.push(1.0);
    }

    fn process_input(&mut self) {
        let buf: Vec<_> = self.input_buffer.drain(..).collect();
        for (p, v) in buf { self.inject(p, v); }
    }

    fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.positions.is_empty() { return; }

        for idx in 0..self.cell_types.len() {
            if self.cell_types[idx] != 2 { self.cell_types[idx] = 0; }
        }
        for pos in &self.positions {
            let i = (pos.x / CELL_SIZE).floor() as i32;
            let j = (pos.y / CELL_SIZE).floor() as i32;
            let k = (pos.z / CELL_SIZE).floor() as i32;
            if i >= 0 && i < GRID_WIDTH as i32 && j >= 0 && j < GRID_HEIGHT as i32 && k >= 0 && k < GRID_DEPTH as i32 {
                let idx = k as usize * GRID_WIDTH * GRID_HEIGHT + j as usize * GRID_WIDTH + i as usize;
                if self.cell_types[idx] != 2 { self.cell_types[idx] = 1; }
            }
        }

        self.gpu.step(device, queue, &mut self.positions, &mut self.velocities, &mut self.c_matrices,
            &self.densities, &self.cell_types, None, None, DT, GRAVITY, FLOW_ACCEL, PRESSURE_ITERS);

        // Collision boundaries must match cell_types solid cells:
        // - Solid at y=0 → floor_y = 0.5 * CELL_SIZE (center of cell 0)
        // - Solid at y=HEIGHT-1 → ceiling_y = (HEIGHT-1.5) * CELL_SIZE
        // - Solid at z=0,DEPTH-1 → z in [0.5*CELL, (DEPTH-1.5)*CELL]
        // - Solid at x=0 → min_x = 0.5 * CELL_SIZE
        // - Solid at x=WIDTH-1 for y<weir → exit wall
        let min_x = CELL_SIZE * 1.0;  // Inside cell 1 (cell 0 is solid inlet)
        let floor_y = CELL_SIZE * 1.0;  // Inside cell 1 (cell 0 is solid floor)
        let ceiling_y = (GRID_HEIGHT as f32 - 1.5) * CELL_SIZE;  // Below solid ceiling cell
        let min_z = CELL_SIZE * 1.0;  // Inside cell 1 (cell 0 is solid)
        let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;  // Inside cell DEPTH-2
        let exit_x = (GRID_WIDTH as f32 - 1.5) * CELL_SIZE;  // Before exit wall cells

        for i in 0..self.positions.len() {
            // Floor collision
            if self.positions[i].y < floor_y {
                self.positions[i].y = floor_y;
                self.velocities[i].y = self.velocities[i].y.abs() * 0.1;
            }
            // Ceiling collision
            if self.positions[i].y > ceiling_y {
                self.positions[i].y = ceiling_y;
                self.velocities[i].y = -self.velocities[i].y.abs() * 0.1;
            }
            // Z walls
            if self.positions[i].z < min_z {
                self.positions[i].z = min_z;
                self.velocities[i].z = self.velocities[i].z.abs() * 0.1;
            }
            if self.positions[i].z > max_z {
                self.positions[i].z = max_z;
                self.velocities[i].z = -self.velocities[i].z.abs() * 0.1;
            }
            // Inlet wall
            if self.positions[i].x < min_x {
                self.positions[i].x = min_x;
                self.velocities[i].x = self.velocities[i].x.abs() * 0.1;
            }
            // Exit wall below outflow threshold
            if self.positions[i].x >= exit_x && self.positions[i].y < OUTFLOW_MIN_Y {
                self.positions[i].x = exit_x - CELL_SIZE * 0.1;
                self.velocities[i].x = -self.velocities[i].x.abs() * 0.1;
            }
        }
    }

    fn extract_exit(&mut self) -> Vec<(Vec3, Vec3)> {
        let mut out = Vec::new();
        let exit_x = (GRID_WIDTH as f32 - 1.5) * CELL_SIZE;  // Match collision boundary
        let mut i = 0;
        while i < self.positions.len() {
            // Only allow exit if particle is at the exit edge AND above the outflow threshold
            if self.positions[i].x >= exit_x && self.positions[i].y >= OUTFLOW_MIN_Y {
                out.push((self.positions[i] + self.world_offset, self.velocities[i]));
                self.positions.swap_remove(i); self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i); self.densities.swap_remove(i);
            } else { i += 1; }
        }
        out
    }

    fn remove_oob(&mut self) {
        let max_y = (GRID_HEIGHT as f32 - 0.5) * CELL_SIZE;
        let mut i = 0;
        while i < self.positions.len() {
            let p = self.positions[i];
            if p.y < 0.0 || p.y > max_y || !p.is_finite() || !self.velocities[i].is_finite() {
                self.positions.swap_remove(i); self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i); self.densities.swap_remove(i);
            } else { i += 1; }
        }
    }
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

    grid: Option<GridSegment>,

    camera: Camera,
    paused: bool,
    frame: u32,
    seed: u64,
    total_injected: usize,
    total_exited: usize,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    last_fps_time: Instant,
    fps_count: u32,
    fps: f32,
}

impl App {
    fn new() -> Self {
        // Center camera on single grid
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.5,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5
        );
        Self {
            window: None, ctx: None, depth_view: None, point_pipeline: None, line_pipeline: None,
            uniform_buffer: None, uniform_bind_group: None, vertex_buffer: None, line_buffer: None,
            grid: None,
            camera: Camera::new(0.0, 0.3, 1.5, center),
            paused: false, frame: 0, seed: 12345,
            total_injected: 0, total_exited: 0,
            mouse_pressed: false, last_mouse: None,
            last_fps_time: Instant::now(), fps_count: 0, fps: 0.0,
        }
    }

    fn spawn_inlet(&mut self) {
        if let Some(grid) = &mut self.grid {
            // Spawn inside valid fluid region (avoiding solid boundary cells)
            // Valid region: x in [1,WIDTH-2], y in [1,HEIGHT-2], z in [1,DEPTH-2]
            let floor_y = CELL_SIZE * 1.5;  // Just above floor
            let max_y = CELL_SIZE * 3.5;    // Low in the tank
            let min_z = CELL_SIZE * 1.5;    // Inside z walls
            let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;
            for _ in 0..INLET_RATE {
                let x = CELL_SIZE * 1.5;  // Just inside inlet wall
                let y = floor_y + rand_f32(&mut self.seed) * (max_y - floor_y);
                let z = min_z + rand_f32(&mut self.seed) * (max_z - min_z);
                grid.inject(Vec3::new(x, y, z), Vec3::new(INLET_VELOCITY, 0.0, 0.0));
                self.total_injected += 1;
            }
        }
    }

    fn step(&mut self) {
        self.spawn_inlet();
        if let Some(g) = &mut self.grid { g.process_input(); }

        let ctx = self.ctx.as_ref().unwrap();
        for _ in 0..SUBSTEPS {
            if let Some(g) = &mut self.grid { g.step(&ctx.device, &ctx.queue); }
        }

        // Count particles that exit over the weir
        if let Some(g) = &mut self.grid {
            self.total_exited += g.extract_exit().len();
            g.remove_oob();
        }
        self.frame += 1;
    }

    fn render(&mut self) {
        let ctx = self.ctx.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();

        let surface_texture = match ctx.surface.get_current_texture() {
            Ok(t) => t, Err(_) => return,
        };
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Collect particles - 6 vertices per particle for quad rendering
        let mut vertices: Vec<Pos3Color4Vertex> = Vec::new();
        if let Some(g) = &self.grid {
            for p in &g.positions {
                let wp = *p + g.world_offset;
                let color = [0.2, 0.5, 1.0, 1.0];
                // 6 vertices per particle (2 triangles = 1 quad)
                for _ in 0..6 {
                    vertices.push(Pos3Color4Vertex { position: wp.to_array(), color });
                }
            }
        }

        // Update vertex buffer (clamp to buffer size - 6 verts per particle)
        let max_verts = MAX_PARTICLES * 6;
        if vertices.len() > max_verts { vertices.truncate(max_verts); }
        if !vertices.is_empty() {
            let data = bytemuck::cast_slice(&vertices);
            if let Some(vb) = &self.vertex_buffer {
                ctx.queue.write_buffer(vb, 0, data);
            }
        }

        // Grid boundary lines (single grid)
        let line_verts: Vec<Pos3Color4Vertex> = grid_boundary_lines(Vec3::ZERO, [0.3, 0.6, 1.0, 1.0]);
        if let Some(lb) = &self.line_buffer {
            ctx.queue.write_buffer(lb, 0, bytemuck::cast_slice(&line_verts));
        }

        // Update uniforms
        let view_proj = self.camera.view_proj_matrix(size.width as f32 / size.height as f32);
        let cam_pos = self.camera.position();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [cam_pos.x, cam_pos.y, cam_pos.z],
            point_size: POINT_SIZE,
        };
        if let Some(ub) = &self.uniform_buffer {
            ctx.queue.write_buffer(ub, 0, bytemuck::bytes_of(&uniforms));
        }

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let (Some(pipeline), Some(bg), Some(vb)) = (&self.point_pipeline, &self.uniform_bind_group, &self.vertex_buffer) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..vertices.len() as u32, 0..1);
            }

            // Draw grid boundary lines
            if let (Some(pipeline), Some(bg), Some(lb)) = (&self.line_pipeline, &self.uniform_bind_group, &self.line_buffer) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.set_vertex_buffer(0, lb.slice(..));
                pass.draw(0..line_verts.len() as u32, 0..1);
            }
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        // FPS
        self.fps_count += 1;
        if Instant::now().duration_since(self.last_fps_time).as_secs_f32() >= 1.0 {
            self.fps = self.fps_count as f32;
            self.fps_count = 0;
            self.last_fps_time = Instant::now();
            let count = self.grid.as_ref().map(|g| g.positions.len()).unwrap_or(0);
            println!("FPS:{:.0} particles:{} injected:{} exited:{}", self.fps, count, self.total_injected, self.total_exited);
        }
    }

    fn reset(&mut self) {
        if let Some(g) = &mut self.grid {
            g.positions.clear();
            g.velocities.clear();
            g.c_matrices.clear();
            g.densities.clear();
        }
        self.total_injected = 0;
        self.total_exited = 0;
        self.frame = 0;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes().with_title("Chained Grids").with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        let size = window.inner_size();

        // Depth
        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

        // Uniform buffer
        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniforms"), size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        let uniform_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
        });

        // Pipelines - separate shaders for quads and lines
        let quad_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Quad Shader"), source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()) });
        let line_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Line Shader"), source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()) });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
        });

        // Point pipeline renders particles as screen-space quads (TriangleList)
        let point_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Quad Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &quad_shader, entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Pos3Color4Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &quad_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format: ctx.config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Vertex buffer (6 vertices per particle for quads)
        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertices"),
            size: (MAX_PARTICLES * 6 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Line pipeline (for grid boundaries)
        let line_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader, entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Pos3Color4Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format: ctx.config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Line buffer (for single grid boundary + weir line = 28 vertices)
        let line_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lines"),
            size: (28 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Single grid
        self.grid = Some(GridSegment::new(&ctx.device, Vec3::ZERO));

        self.ctx = Some(ctx);
        self.point_pipeline = Some(point_pipeline);
        self.line_pipeline = Some(line_pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.vertex_buffer = Some(vertex_buffer);
        self.line_buffer = Some(line_buffer);

        println!("=== Single Grid Volume Fill ===");
        println!("Blue=water, Yellow=outflow weir (halfway up)");
        println!("Water fills up below weir before overflowing");
        println!("Mouse drag=rotate, Scroll=zoom, Space=pause, R=reset");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(ctx) = &mut self.ctx {
                        ctx.config.width = size.width; ctx.config.height = size.height;
                        ctx.surface.configure(&ctx.device, &ctx.config);
                        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Space) => { self.paused = !self.paused; println!("{}", if self.paused { "PAUSED" } else { "RESUMED" }); }
                    PhysicalKey::Code(KeyCode::KeyR) => { self.reset(); println!("RESET"); }
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
                if !self.mouse_pressed { self.last_mouse = None; }
            }
            WindowEvent::CursorMoved { position, .. } if self.mouse_pressed => {
                if let Some((lx, ly)) = self.last_mouse {
                    self.camera.handle_mouse_move((position.x - lx) as f32, (position.y - ly) as f32);
                }
                self.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let s = match delta { MouseScrollDelta::LineDelta(_, y) => y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01 };
                self.camera.handle_zoom(s);
            }
            WindowEvent::RedrawRequested => {
                if !self.paused { self.step(); }
                self.render();
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()).unwrap();
}
