//! Visual Chained Grids Demo
//!
//! Three FLIP grids connected in series with real-time rendering.
//! Grid A → Grid B → Grid C → exit
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

// Grid dimensions
const GRID_WIDTH: usize = 16;
const GRID_HEIGHT: usize = 10;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.04;
const MAX_PARTICLES: usize = 50000;

// Weir at 50% height
const OUTFLOW_MIN_Y: f32 = (GRID_HEIGHT as f32 * 0.5) * CELL_SIZE;

// Simulation
const DT: f32 = 1.0 / 120.0;
const SUBSTEPS: u32 = 2;
const PRESSURE_ITERS: u32 = 50;
const GRAVITY: f32 = -9.8;
const FLOW_ACCEL: f32 = 2.0; // Small acceleration for visual demo (headless uses 0.0)
const INLET_RATE: usize = 30;
const INLET_VELOCITY: f32 = 0.6;

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

fn grid_boundary_lines(offset: Vec3, color: [f32; 4], show_weir: bool) -> Vec<Pos3Color4Vertex> {
    let x_min = CELL_SIZE * 1.0;
    let x_max = (GRID_WIDTH as f32 - 1.5) * CELL_SIZE;
    let y_min = CELL_SIZE * 1.0;
    let y_max = (GRID_HEIGHT as f32 - 1.5) * CELL_SIZE;
    let z_min = CELL_SIZE * 1.0;
    let z_max = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;

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
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    let mut lines = Vec::with_capacity(32);
    for (a, b) in edges {
        lines.push(Pos3Color4Vertex { position: corners[a].to_array(), color });
        lines.push(Pos3Color4Vertex { position: corners[b].to_array(), color });
    }

    // Weir line (yellow) on exit face
    if show_weir {
        let weir_color = [1.0, 0.8, 0.2, 1.0];
        lines.push(Pos3Color4Vertex { position: (offset + Vec3::new(x_max, OUTFLOW_MIN_Y, z_min)).to_array(), color: weir_color });
        lines.push(Pos3Color4Vertex { position: (offset + Vec3::new(x_max, OUTFLOW_MIN_Y, z_max)).to_array(), color: weir_color });
    }

    lines
}

struct GridSegment {
    gpu: GpuFlip3D,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    is_ghost: Vec<bool>, // Ghost particles are removed after one step
    cell_types: Vec<u32>,
    world_offset: Vec3,
    input_buffer: Vec<(Vec3, Vec3, Mat3, f32, bool)>, // pos, vel, c_mat, density, is_ghost
}

impl GridSegment {
    fn new(device: &wgpu::Device, world_offset: Vec3) -> Self {
        let mut gpu = GpuFlip3D::new(
            device, GRID_WIDTH as u32, GRID_HEIGHT as u32, GRID_DEPTH as u32, CELL_SIZE, MAX_PARTICLES,
        );
        gpu.vorticity_epsilon = 0.0;
        gpu.open_boundaries = 2; // +X open

        let mut cell_types = vec![0u32; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
        for z in 0..GRID_DEPTH {
            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    let idx = z * GRID_WIDTH * GRID_HEIGHT + y * GRID_WIDTH + x;
                    let is_floor = y == 0;
                    let is_ceiling = y == GRID_HEIGHT - 1;
                    let is_z_wall = z == 0 || z == GRID_DEPTH - 1;
                    let is_inlet = x == 0;
                    // No solid exit wall - open boundary
                    if is_floor || is_ceiling || is_z_wall || is_inlet {
                        cell_types[idx] = 2; // SOLID
                    }
                }
            }
        }

        Self {
            gpu, positions: Vec::new(), velocities: Vec::new(), c_matrices: Vec::new(),
            densities: Vec::new(), is_ghost: Vec::new(), cell_types, world_offset, input_buffer: Vec::new(),
        }
    }

    fn inject(&mut self, pos: Vec3, vel: Vec3) {
        self.positions.push(pos);
        self.velocities.push(vel);
        self.c_matrices.push(Mat3::ZERO);
        self.densities.push(1.0);
        self.is_ghost.push(false);
    }

    fn process_input(&mut self) {
        let buf: Vec<_> = self.input_buffer.drain(..).collect();
        for (p, v, c, d, ghost) in buf { self.inject_full(p, v, c, d, ghost); }
    }

    fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.positions.is_empty() { return; }

        // Update fluid cells
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

        // CPU collision
        let min_x = CELL_SIZE * 1.0;
        let floor_y = CELL_SIZE * 1.0;
        let ceiling_y = (GRID_HEIGHT as f32 - 1.5) * CELL_SIZE;
        let min_z = CELL_SIZE * 1.0;
        let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;

        for i in 0..self.positions.len() {
            if self.positions[i].y < floor_y {
                self.positions[i].y = floor_y;
                self.velocities[i].y = self.velocities[i].y.abs() * 0.1;
            }
            if self.positions[i].y > ceiling_y {
                self.positions[i].y = ceiling_y;
                self.velocities[i].y = -self.velocities[i].y.abs() * 0.1;
            }
            if self.positions[i].z < min_z {
                self.positions[i].z = min_z;
                self.velocities[i].z = self.velocities[i].z.abs() * 0.1;
            }
            if self.positions[i].z > max_z {
                self.positions[i].z = max_z;
                self.velocities[i].z = -self.velocities[i].z.abs() * 0.1;
            }
            if self.positions[i].x < min_x {
                self.positions[i].x = min_x;
                self.velocities[i].x = self.velocities[i].x.abs() * 0.1;
            }
            // No +X collision - particles can exit freely
        }
    }

    /// Extract particles that exit through +X boundary above the weir
    /// Returns: (world_pos, velocity, c_matrix, density) - only real particles, not ghosts
    fn extract_exit(&mut self) -> Vec<(Vec3, Vec3, Mat3, f32)> {
        let mut out = Vec::new();
        let exit_x = (GRID_WIDTH as f32 - 2.0) * CELL_SIZE; // Near the exit edge
        let mut i = 0;
        while i < self.positions.len() {
            if self.positions[i].x >= exit_x && self.positions[i].y >= OUTFLOW_MIN_Y && !self.is_ghost[i] {
                out.push((
                    self.positions[i] + self.world_offset,
                    self.velocities[i],
                    self.c_matrices[i],
                    self.densities[i],
                ));
                self.positions.swap_remove(i); self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i); self.densities.swap_remove(i);
                self.is_ghost.swap_remove(i);
            } else { i += 1; }
        }
        out
    }

    /// Inject particle with full state (for handoff)
    fn inject_full(&mut self, pos: Vec3, vel: Vec3, c_mat: Mat3, density: f32, is_ghost: bool) {
        self.positions.push(pos);
        self.velocities.push(vel);
        self.c_matrices.push(c_mat);
        self.densities.push(density);
        self.is_ghost.push(is_ghost);
    }

    /// Remove ghost particles (call after step - they've done their P2G job)
    fn remove_ghosts(&mut self) {
        let mut i = 0;
        while i < self.positions.len() {
            if self.is_ghost[i] {
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
                self.is_ghost.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn remove_oob(&mut self) {
        let max_y = (GRID_HEIGHT as f32 - 0.5) * CELL_SIZE;
        let max_x = GRID_WIDTH as f32 * CELL_SIZE;
        let mut i = 0;
        while i < self.positions.len() {
            let p = self.positions[i];
            if p.y < 0.0 || p.y > max_y || p.x > max_x || p.x < 0.0 || !p.is_finite() || !self.velocities[i].is_finite() {
                self.positions.swap_remove(i); self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i); self.densities.swap_remove(i);
                self.is_ghost.swap_remove(i);
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

    grid_a: Option<GridSegment>,
    grid_b: Option<GridSegment>,
    grid_c: Option<GridSegment>,

    camera: Camera,
    paused: bool,
    seed: u64,
    total_injected: usize,
    total_handoffs: usize,
    total_exited: usize,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    last_fps_time: Instant,
    fps_count: u32,
}

impl App {
    fn new() -> Self {
        // Center camera on all three grids
        let grid_length = GRID_WIDTH as f32 * CELL_SIZE;
        let center = Vec3::new(
            grid_length * 1.5, // Center of three grids
            GRID_HEIGHT as f32 * CELL_SIZE * 0.5,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5
        );
        Self {
            window: None, ctx: None, depth_view: None, point_pipeline: None, line_pipeline: None,
            uniform_buffer: None, uniform_bind_group: None, vertex_buffer: None, line_buffer: None,
            grid_a: None, grid_b: None, grid_c: None,
            camera: Camera::new(0.0, 0.3, 3.5, center),
            paused: false, seed: 12345,
            total_injected: 0, total_handoffs: 0, total_exited: 0,
            mouse_pressed: false, last_mouse: None,
            last_fps_time: Instant::now(), fps_count: 0,
        }
    }

    fn spawn_inlet(&mut self) {
        if let Some(grid) = &mut self.grid_a {
            let floor_y = CELL_SIZE * 1.5;
            let max_y = (GRID_HEIGHT as f32 - 2.0) * CELL_SIZE;
            let min_z = CELL_SIZE * 1.5;
            let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;
            for _ in 0..INLET_RATE {
                let x = CELL_SIZE * 1.5;
                let y = floor_y + rand_f32(&mut self.seed) * (max_y - floor_y);
                let z = min_z + rand_f32(&mut self.seed) * (max_z - min_z);
                grid.inject(Vec3::new(x, y, z), Vec3::new(INLET_VELOCITY, 0.0, 0.0));
                self.total_injected += 1;
            }
        }
    }

    fn step(&mut self) {
        self.spawn_inlet();

        if let Some(g) = &mut self.grid_a { g.process_input(); }
        if let Some(g) = &mut self.grid_b { g.process_input(); }
        if let Some(g) = &mut self.grid_c { g.process_input(); }

        let ctx = self.ctx.as_ref().unwrap();
        for _ in 0..SUBSTEPS {
            if let Some(g) = &mut self.grid_a { g.step(&ctx.device, &ctx.queue); }
            if let Some(g) = &mut self.grid_b { g.step(&ctx.device, &ctx.queue); }
            if let Some(g) = &mut self.grid_c { g.step(&ctx.device, &ctx.queue); }
        }

        // Handoff: A exits -> B enters (preserve full particle state)
        // Also inject ghost particles to establish velocity field at inlet
        if let Some(ga) = &mut self.grid_a {
            let exited = ga.extract_exit();
            if let Some(gb) = &mut self.grid_b {
                for (world_pos, vel, c_mat, density) in exited {
                    // Convert to Grid B local coords
                    let local_pos = world_pos - gb.world_offset;
                    // Place at inlet of Grid B, preserving Y and Z
                    let local_pos = Vec3::new(CELL_SIZE * 1.5, local_pos.y, local_pos.z);

                    // Inject the actual particle (not a ghost)
                    gb.input_buffer.push((local_pos, vel, c_mat, density, false));

                    // Inject ghost particles ahead to establish velocity field
                    // These are removed after one step (after P2G)
                    for dx in 1..4 {
                        let ghost_pos = Vec3::new(
                            local_pos.x + dx as f32 * CELL_SIZE,
                            local_pos.y,
                            local_pos.z,
                        );
                        gb.input_buffer.push((ghost_pos, vel, c_mat, density, true)); // ghost=true
                    }

                    self.total_handoffs += 1;
                }
            }
        }

        // Handoff: B exits -> C enters
        if let Some(gb) = &mut self.grid_b {
            let exited = gb.extract_exit();
            if let Some(gc) = &mut self.grid_c {
                for (world_pos, vel, c_mat, density) in exited {
                    let local_pos = world_pos - gc.world_offset;
                    let local_pos = Vec3::new(CELL_SIZE * 1.5, local_pos.y, local_pos.z);

                    gc.input_buffer.push((local_pos, vel, c_mat, density, false));

                    // Ghost particles
                    for dx in 1..4 {
                        let ghost_pos = Vec3::new(
                            local_pos.x + dx as f32 * CELL_SIZE,
                            local_pos.y,
                            local_pos.z,
                        );
                        gc.input_buffer.push((ghost_pos, vel, c_mat, density, true));
                    }

                    self.total_handoffs += 1;
                }
            }
        }

        // Remove ghost particles after they've done their P2G job
        if let Some(gb) = &mut self.grid_b {
            gb.remove_ghosts();
        }
        if let Some(gc) = &mut self.grid_c {
            gc.remove_ghosts();
        }

        // C exits -> leave system
        if let Some(gc) = &mut self.grid_c {
            self.total_exited += gc.extract_exit().len();
            gc.remove_oob();
        }
        if let Some(gb) = &mut self.grid_b {
            gb.remove_oob();
        }
        if let Some(ga) = &mut self.grid_a {
            ga.remove_oob();
        }
    }

    fn render(&mut self) {
        let ctx = self.ctx.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();

        let surface_texture = match ctx.surface.get_current_texture() {
            Ok(t) => t, Err(_) => return,
        };
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Collect particles - 6 vertices per particle
        let mut vertices: Vec<Pos3Color4Vertex> = Vec::new();

        // Grid A particles - blue
        if let Some(g) = &self.grid_a {
            for p in &g.positions {
                let wp = *p + g.world_offset;
                let color = [0.2, 0.5, 1.0, 1.0];
                for _ in 0..6 {
                    vertices.push(Pos3Color4Vertex { position: wp.to_array(), color });
                }
            }
        }

        // Grid B particles - cyan
        if let Some(g) = &self.grid_b {
            for p in &g.positions {
                let wp = *p + g.world_offset;
                let color = [0.2, 0.9, 0.8, 1.0];
                for _ in 0..6 {
                    vertices.push(Pos3Color4Vertex { position: wp.to_array(), color });
                }
            }
        }

        // Grid C particles - green
        if let Some(g) = &self.grid_c {
            for p in &g.positions {
                let wp = *p + g.world_offset;
                let color = [0.4, 1.0, 0.4, 1.0];
                for _ in 0..6 {
                    vertices.push(Pos3Color4Vertex { position: wp.to_array(), color });
                }
            }
        }

        let max_verts = MAX_PARTICLES * 3 * 6;
        if vertices.len() > max_verts { vertices.truncate(max_verts); }
        if !vertices.is_empty() {
            if let Some(vb) = &self.vertex_buffer {
                ctx.queue.write_buffer(vb, 0, bytemuck::cast_slice(&vertices));
            }
        }

        // Grid boundary lines
        let grid_length = GRID_WIDTH as f32 * CELL_SIZE;
        let mut line_verts: Vec<Pos3Color4Vertex> = Vec::new();
        line_verts.extend(grid_boundary_lines(Vec3::ZERO, [0.3, 0.6, 1.0, 1.0], true)); // Grid A - blue
        line_verts.extend(grid_boundary_lines(Vec3::new(grid_length, 0.0, 0.0), [0.2, 0.8, 0.7, 1.0], true)); // Grid B - cyan
        line_verts.extend(grid_boundary_lines(Vec3::new(grid_length * 2.0, 0.0, 0.0), [0.4, 0.9, 0.4, 1.0], true)); // Grid C - green

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

            if let (Some(pipeline), Some(bg), Some(lb)) = (&self.line_pipeline, &self.uniform_bind_group, &self.line_buffer) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.set_vertex_buffer(0, lb.slice(..));
                pass.draw(0..line_verts.len() as u32, 0..1);
            }
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        // FPS and stats
        self.fps_count += 1;
        if Instant::now().duration_since(self.last_fps_time).as_secs_f32() >= 1.0 {
            let count_a = self.grid_a.as_ref().map(|g| g.positions.len()).unwrap_or(0);
            let count_b = self.grid_b.as_ref().map(|g| g.positions.len()).unwrap_or(0);
            let count_c = self.grid_c.as_ref().map(|g| g.positions.len()).unwrap_or(0);
            println!("FPS:{} A:{} B:{} C:{} handoffs:{} exited:{}",
                self.fps_count, count_a, count_b, count_c, self.total_handoffs, self.total_exited);
            self.fps_count = 0;
            self.last_fps_time = Instant::now();
        }
    }

    fn reset(&mut self) {
        if let Some(g) = &mut self.grid_a {
            g.positions.clear(); g.velocities.clear();
            g.c_matrices.clear(); g.densities.clear();
            g.is_ghost.clear(); g.input_buffer.clear();
        }
        if let Some(g) = &mut self.grid_b {
            g.positions.clear(); g.velocities.clear();
            g.c_matrices.clear(); g.densities.clear();
            g.is_ghost.clear(); g.input_buffer.clear();
        }
        if let Some(g) = &mut self.grid_c {
            g.positions.clear(); g.velocities.clear();
            g.c_matrices.clear(); g.densities.clear();
            g.is_ghost.clear(); g.input_buffer.clear();
        }
        self.total_injected = 0;
        self.total_handoffs = 0;
        self.total_exited = 0;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes().with_title("Chained Grids").with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        ).unwrap());
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window.clone()));

        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));

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

        let quad_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Quad Shader"), source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()) });
        let line_shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Line Shader"), source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()) });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
        });

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

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertices"),
            size: (MAX_PARTICLES * 3 * 6 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        // Three grids worth of boundary lines (28 verts each)
        let line_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lines"),
            size: (96 * std::mem::size_of::<Pos3Color4Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create three grids
        let grid_length = GRID_WIDTH as f32 * CELL_SIZE;
        self.grid_a = Some(GridSegment::new(&ctx.device, Vec3::ZERO));
        self.grid_b = Some(GridSegment::new(&ctx.device, Vec3::new(grid_length, 0.0, 0.0)));
        self.grid_c = Some(GridSegment::new(&ctx.device, Vec3::new(grid_length * 2.0, 0.0, 0.0)));

        self.ctx = Some(ctx);
        self.point_pipeline = Some(point_pipeline);
        self.line_pipeline = Some(line_pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.vertex_buffer = Some(vertex_buffer);
        self.line_buffer = Some(line_buffer);

        println!("=== CHAINED GRIDS ===");
        println!("Blue = Grid A, Cyan = Grid B, Green = Grid C");
        println!("Particles flow: A -> B -> C -> exit");
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
