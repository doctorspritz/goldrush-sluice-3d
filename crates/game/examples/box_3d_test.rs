//! 3D Closed Box Test - Simple GPU Water Stacking Test
//!
//! Tests whether GPU FLIP simulation can stack water properly in a closed box.
//! No inlet, no outlet, no slope - just drop water and see if it fills up.
//! Run with: cargo run --example box_3d_test --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Mat4, Vec3};
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

// Simple grid - smaller for testing
const GRID_WIDTH: usize = 32;
const GRID_HEIGHT: usize = 32;
const GRID_DEPTH: usize = 16;
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
    gpu_flip: Option<GpuFlip3D>,
    sim: FlipSimulation3D,
    paused: bool,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    frame: u32,
    solid_instances: Vec<ParticleInstance>,
    // Particle data for GPU
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    bed_height: Vec<f32>,
    cell_types: Vec<u32>,
    use_gpu_sim: bool,
    // Mouse drag state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // FPS tracking
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
    // Emitter control
    emitter_enabled: bool,
    // Exit tracking
    particles_exited: u32,
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

/// Create a sluice-like box with sloped floor, riffles, and exit
/// Floor slopes from left (high) to right (low)
/// Riffles are obstacles on the floor
/// Exit is an opening at the right side
fn create_closed_box(sim: &mut FlipSimulation3D) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    // Slope parameters: floor height varies from left to right
    let floor_height_left = 8;   // 8 cells high on left side
    let floor_height_right = 2;  // 2 cells high on right side

    // Riffle parameters
    let riffle_spacing = 6;      // Riffles every 6 cells
    let riffle_height = 2;       // Riffles are 2 cells tall
    let riffle_start_x = 8;      // Start riffles after this X position
    let riffle_end_x = width - 4; // Stop riffles before exit

    // Exit parameters (opening at right wall)
    let exit_start_z = depth / 4;
    let exit_end_z = 3 * depth / 4;
    let exit_height = 6;  // Exit opening is 6 cells tall

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                // Calculate floor height at this x position (linear interpolation)
                let t = i as f32 / (width - 1) as f32;
                let floor_height = floor_height_left as f32 * (1.0 - t) + floor_height_right as f32 * t;
                let floor_j = floor_height as usize;

                // Check if this is a riffle position
                let is_riffle = i >= riffle_start_x && i < riffle_end_x &&
                    (i - riffle_start_x) % riffle_spacing < 2 &&  // Riffle is 2 cells wide
                    j <= floor_j + riffle_height &&
                    j > floor_j;

                // Check if this is the exit opening (right wall, middle section, lower part)
                let is_exit = i == width - 1 &&
                    k >= exit_start_z && k < exit_end_z &&
                    j > floor_j && j <= floor_j + exit_height;

                let is_boundary =
                    (i == 0) ||                      // Left wall
                    (i == width - 1 && !is_exit) ||  // Right wall (except exit)
                    j <= floor_j ||                   // Sloped floor
                    j == height - 1 ||                // Ceiling
                    k == 0 || k == depth - 1 ||       // Z walls
                    is_riffle;                        // Riffles on floor

                if is_boundary {
                    sim.grid.set_solid(i, j, k);
                }
            }
        }
    }

    sim.grid.compute_sdf();

    // Count riffles
    let num_riffles = ((riffle_end_x - riffle_start_x) / riffle_spacing) as usize;
    println!("Created sluice: slope {}→{} cells, {} riffles, exit at right wall",
             floor_height_left, floor_height_right, num_riffles);
}

/// Spawn water block in the center, raised above floor
fn spawn_water_block(sim: &mut FlipSimulation3D, count: usize) {
    let cell_size = sim.grid.cell_size;

    // Spawn in center of box, starting from floor and going up
    // Leave margin from walls (2 cells)
    let margin = 2.0;
    let spawn_width = (GRID_WIDTH as f32) - 2.0 * margin;
    let spawn_depth = (GRID_DEPTH as f32) - 2.0 * margin;
    let spawn_height = 10.0;  // Water block height in cells

    let start_x = margin * cell_size;
    let start_y = 1.5 * cell_size;  // Just above floor
    let start_z = margin * cell_size;

    // Calculate spacing from physical volume
    let physical_width = spawn_width * cell_size;
    let physical_height = spawn_height * cell_size;
    let physical_depth = spawn_depth * cell_size;
    let physical_volume = physical_width * physical_height * physical_depth;
    let spacing = (physical_volume / count as f32).powf(1.0 / 3.0);

    let nx = (physical_width / spacing).ceil() as usize;
    let ny = (physical_height / spacing).ceil() as usize;
    let nz = (physical_depth / spacing).ceil() as usize;

    let mut spawned = 0;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if spawned >= count {
                    break;
                }

                let x = start_x + (i as f32 + 0.5) * spacing;
                let y = start_y + (j as f32 + 0.5) * spacing;
                let z = start_z + (k as f32 + 0.5) * spacing;

                // Add small jitter
                let jx = (rand_float() - 0.5) * 0.1 * spacing;
                let jy = (rand_float() - 0.5) * 0.1 * spacing;
                let jz = (rand_float() - 0.5) * 0.1 * spacing;

                sim.spawn_particle(Vec3::new(x + jx, y + jy, z + jz));
                spawned += 1;
            }
        }
    }

    println!("Spawned {} particles in block", spawned);
}

fn rand_float() -> f32 {
    // Simple LCG for deterministic "random" jitter
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED >> 16) as f32 / 65535.0
    }
}

impl App {
    fn new() -> Self {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 100;  // Smaller grid, fewer iterations needed

        // Create closed box geometry
        create_closed_box(&mut sim);

        // Collect solid cell positions for rendering
        let solid_instances = Self::collect_solids(&sim);

        // Don't spawn initial water - use emitter instead
        println!("Solid cells: {}", solid_instances.len());
        println!("Controls: SPACE=pause, R=reset, G=toggle GPU/CPU, E=toggle emitter, ESC=quit");
        println!("Watching for DEFLATION - water should flow through and maintain height!");

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            paused: false,
            camera_angle: 0.5,
            camera_pitch: 0.4,
            camera_distance: 3.0,
            frame: 0,
            solid_instances,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            bed_height: vec![0.0; GRID_WIDTH * GRID_DEPTH],
            cell_types: Vec::new(),
            use_gpu_sim: true,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            emitter_enabled: true,
            particles_exited: 0,
        }
    }

    /// Spawn particles from emitter on left side (above the higher part of slope)
    fn emit_particles(&mut self, count: usize) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return; // Don't exceed max
        }

        let cell_size = CELL_SIZE;
        let max_to_spawn = (MAX_PARTICLES - self.sim.particles.len()).min(count);

        // Emit from left side, above the sloped floor (floor is 8 cells high on left)
        let emit_x = 3.0 * cell_size;  // Near left wall
        let center_z = GRID_DEPTH as f32 * cell_size * 0.5;
        let emit_y = 12.0 * cell_size; // Above the 8-cell high left floor

        // Emit in a small region
        let spread_x = 2.0 * cell_size;
        let spread_z = 3.0 * cell_size;

        for _ in 0..max_to_spawn {
            let x = emit_x + rand_float() * spread_x;
            let z = center_z + (rand_float() - 0.5) * spread_z;
            let y = emit_y + rand_float() * cell_size;

            self.sim.spawn_particle(Vec3::new(x, y, z));
        }
    }

    fn collect_solids(sim: &FlipSimulation3D) -> Vec<ParticleInstance> {
        let mut solids = Vec::new();
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;
        let cell_size = sim.grid.cell_size;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if sim.grid.is_solid(i, j, k) {
                        // Only render exposed faces (optimization)
                        let exposed =
                            (i == 0 || !sim.grid.is_solid(i-1, j, k)) ||
                            (i == width-1 || !sim.grid.is_solid(i+1, j, k)) ||
                            (j == 0 || !sim.grid.is_solid(i, j-1, k)) ||
                            (j == height-1 || !sim.grid.is_solid(i, j+1, k)) ||
                            (k == 0 || !sim.grid.is_solid(i, j, k-1)) ||
                            (k == depth-1 || !sim.grid.is_solid(i, j, k+1));

                        if exposed {
                            solids.push(ParticleInstance {
                                position: [
                                    (i as f32 + 0.5) * cell_size,
                                    (j as f32 + 0.5) * cell_size,
                                    (k as f32 + 0.5) * cell_size,
                                ],
                                color: [0.4, 0.35, 0.3, 0.3],
                            });
                        }
                    }
                }
            }
        }
        solids
    }

    fn reset_sim(&mut self) {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 100;

        create_closed_box(&mut sim);
        // Don't spawn water initially - use emitter

        self.sim = sim;
        self.frame = 0;
        self.emitter_enabled = true;
        self.particles_exited = 0;
        println!("Reset - emitter enabled");
    }

    fn particle_stats(&self) -> (Vec3, f32, f32, f32) {
        let mut sum_vel = Vec3::ZERO;
        let mut max_vel = 0.0f32;
        let mut max_y = 0.0f32;
        let mut max_x = 0.0f32;

        for p in &self.sim.particles.list {
            sum_vel += p.velocity;
            max_vel = max_vel.max(p.velocity.length());
            max_y = max_y.max(p.position.y);
            max_x = max_x.max(p.position.x);
        }

        let count = self.sim.particles.len() as f32;
        let avg_vel = if count > 0.0 { sum_vel / count } else { Vec3::ZERO };

        (avg_vel, max_vel, max_y, max_x)
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        // Update simulation (before borrowing gpu)
        if !self.paused {
            let dt = 1.0 / 60.0;

            // Emit particles from top of box
            if self.emitter_enabled && self.frame % 2 == 0 {
                self.emit_particles(50); // 50 particles every other frame
            }

            if self.use_gpu_sim {
                // Sync from CPU
                self.positions.clear();
                self.velocities.clear();
                self.c_matrices.clear();
                self.densities.clear();

                for p in &self.sim.particles.list {
                    self.positions.push(p.position);
                    self.velocities.push(p.velocity);
                    self.c_matrices.push(p.affine_velocity);
                    self.densities.push(p.density);
                }

                // Build cell types
                let w = self.sim.grid.width;
                let h = self.sim.grid.height;
                let d = self.sim.grid.depth;
                self.cell_types.clear();
                self.cell_types.resize(w * h * d, 0);

                for k in 0..d {
                    for j in 0..h {
                        for i in 0..w {
                            let idx = k * w * h + j * w + i;
                            if self.sim.grid.is_solid(i, j, k) {
                                self.cell_types[idx] = 2;
                            }
                        }
                    }
                }

                for p in &self.sim.particles.list {
                    let i = (p.position.x / CELL_SIZE).floor() as i32;
                    let j = (p.position.y / CELL_SIZE).floor() as i32;
                    let k = (p.position.z / CELL_SIZE).floor() as i32;
                    if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                        let idx = k as usize * w * h + j as usize * w + i as usize;
                        if self.cell_types[idx] != 2 {
                            self.cell_types[idx] = 1;
                        }
                    }
                }

                // Run GPU step (need to borrow gpu here)
                let pressure_iters = self.sim.pressure_iterations as u32;
                if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                    let sdf = self.sim.grid.sdf.as_slice();
                    let positions = &mut self.positions;
                    let velocities = &mut self.velocities;
                    let c_matrices = &mut self.c_matrices;
                    let densities = &self.densities;
                    let cell_types = &self.cell_types;
                    let bed_height = &self.bed_height;
                    gpu_flip.step(
                        &gpu.device,
                        &gpu.queue,
                        positions,
                        velocities,
                        c_matrices,
                        densities,
                        cell_types,
                        Some(sdf),
                        Some(bed_height),
                        dt,
                        -9.8,
                        0.0,  // No flow acceleration for closed box
                        pressure_iters,
                    );
                }

                // Sync back to CPU and advect
                for (idx, p) in self.sim.particles.list.iter_mut().enumerate() {
                    if idx < self.velocities.len() {
                        p.velocity = self.velocities[idx];
                        p.affine_velocity = self.c_matrices[idx];
                    }
                    // Position was advanced and collision-resolved on the GPU.
                    p.position = self.positions[idx];

                    // Exit zone (right wall, middle Z section, lower part)
                    let cell_size = CELL_SIZE;
                    let t = (p.position.x / cell_size) / (GRID_WIDTH as f32 - 1.0);
                    let t = t.clamp(0.0, 1.0);
                    let floor_height = 8.0 * (1.0 - t) + 2.0 * t; // 8 cells left, 2 cells right
                    let exit_start_z = GRID_DEPTH as f32 * cell_size / 4.0;
                    let exit_end_z = GRID_DEPTH as f32 * cell_size * 3.0 / 4.0;
                    let exit_max_y = (floor_height + 6.0) * cell_size;  // Exit is 6 cells tall
                    let is_in_exit_zone = p.position.z >= exit_start_z && p.position.z < exit_end_z
                        && p.position.y < exit_max_y;

                    if p.position.x >= (GRID_WIDTH as f32 - 0.5) * cell_size && is_in_exit_zone {
                        // Mark for removal by moving far outside
                        p.position.x = 1000.0;
                    }
                }

                // Remove particles that exited and count them
                let before = self.sim.particles.len();
                self.sim.particles.list.retain(|p| p.position.x < 100.0);
                let exited_this_frame = before - self.sim.particles.len();
                self.particles_exited += exited_this_frame as u32;
            } else {
                // CPU simulation
                self.sim.update(dt);
            }

            self.frame += 1;
        }

        // Update FPS
        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.last_fps_time = now;
        }

        // Print stats
        if self.frame % 10 == 0 {
            let (avg_vel, _max_vel, max_y, max_x) = self.particle_stats();
            let mode = if self.use_gpu_sim { "GPU" } else { "CPU" };
            println!(
                "[{}] Frame {:5} | FPS: {:5.1} | Particles: {:5} | Exited: {:5} | AvgVel: ({:6.2}, {:5.2}, {:5.2}) | MaxY: {:.3} | MaxX: {:.3}",
                mode,
                self.frame,
                self.current_fps,
                self.sim.particles.len(),
                self.particles_exited,
                avg_vel.x, avg_vel.y, avg_vel.z,
                max_y,
                max_x,
            );
        }

        // Create particle instances
        let instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .map(|p| {
                let speed = p.velocity.length();
                let t = (speed / 3.0).min(1.0);
                ParticleInstance {
                    position: [p.position.x, p.position.y, p.position.z],
                    color: [0.2 + t * 0.6, 0.5 - t * 0.3, 0.9 - t * 0.4, 0.8],
                }
            })
            .collect();

        // Now borrow gpu for rendering
        let gpu = self.gpu.as_ref().unwrap();
        let window = self.window.as_ref().unwrap();

        gpu.queue.write_buffer(&gpu.instance_buffer, 0, bytemuck::cast_slice(&instances));

        // Update camera
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.5,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        let eye = center + Vec3::new(
            self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
        );

        let view = Mat4::look_at_rh(eye, center, Vec3::Y);
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Render
        let frame = gpu.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));

            // Draw solids
            pass.set_vertex_buffer(1, gpu.solid_buffer.slice(..));
            pass.draw(0..4, 0..self.solid_instances.len() as u32);

            // Draw particles
            pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
            pass.draw(0..4, 0..instances.len() as u32);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("3D Sluice Test - Slope + Riffles + Exit")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1200, 800)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find adapter");

        // Request higher limits for GPU FLIP solver (needs 11+ storage buffers)
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create GPU FLIP solver
        let gpu_flip = GpuFlip3D::new(
            &device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );
        self.gpu_flip = Some(gpu_flip);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let vertices = [
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [1.0, -1.0] },
            Vertex { position: [-1.0, 1.0] },
            Vertex { position: [1.0, 1.0] },
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

        let solid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Solid Buffer"),
            contents: bytemuck::cast_slice(&self.solid_instances),
            usage: wgpu::BufferUsages::VERTEX,
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
            label: Some("Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
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

        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                            println!("Paused: {}", self.paused);
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.reset_sim();
                            println!("Reset simulation");
                        }
                        PhysicalKey::Code(KeyCode::KeyG) => {
                            self.use_gpu_sim = !self.use_gpu_sim;
                            println!("Simulation mode: {}", if self.use_gpu_sim { "GPU" } else { "CPU" });
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            self.emitter_enabled = !self.emitter_enabled;
                            println!("Emitter: {}", if self.emitter_enabled { "ON" } else { "OFF" });
                        }
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
                        self.camera_pitch = (self.camera_pitch + dy * 0.01).clamp(-1.5, 1.5);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.3).clamp(0.5, 20.0);
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
    let size = 0.015;

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
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
