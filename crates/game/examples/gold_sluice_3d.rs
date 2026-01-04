//! Gold Sluice 3D - Realistic vortex formation and sediment separation
//!
//! A proper gold sluice simulation that creates vortices behind riffles
//! and separates gold from sand based on density differences.
//!
//! Physics:
//! - FLIP water flow creates natural vortices from riffle geometry
//! - Density-based settling: gold (19.3) sinks faster than sand (2.65)
//! - Vortex pockets trap heavy gold while light sand is carried downstream
//!
//! Run with: cargo run --example gold_sluice_3d --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Mat4, Vec3};
use sim3d::{create_sluice, spawn_inlet_sediment, spawn_inlet_water, FlipSimulation3D, SluiceConfig};
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

// Grid dimensions - wider and longer for better vortex formation
const GRID_WIDTH: usize = 48;   // Flow direction (X)
const GRID_HEIGHT: usize = 20;  // Vertical (Y)
const GRID_DEPTH: usize = 12;   // Width of channel (Z)
const CELL_SIZE: f32 = 0.04;    // Smaller cells for better vortex resolution
const MAX_PARTICLES: usize = 80000;

// Material densities (relative to water)
const WATER_DENSITY: f32 = 1.0;
const SAND_DENSITY: f32 = 2.65;
const GOLD_DENSITY: f32 = 19.3;

// Real sluice physics:
// - Typical sluice velocity: 1.0 - 1.5 m/s (3-5 ft/s)
// - Too slow: sediment clogs riffles
// - Too fast: gold washes out with sand
//
// Settling is NATURAL - just buoyancy-reduced gravity, not an artificial force
// Gold settles ~7x faster than sand due to density difference, not a magic number

// Flow acceleration for sluice
// For a 20% slope: g * sin(arctan(0.20)) ≈ 1.9 m/s²
// We use higher to overcome pressure resistance
const FLOW_ACCEL: f32 = 3.0; // m/s² - downstream acceleration applied ON GRID before pressure solve

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
    sluice_config: SluiceConfig,
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
    cell_types: Vec<u32>,
    use_gpu_sim: bool,
    // Mouse drag state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // FPS tracking
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
    // Simulation control
    emitter_enabled: bool,
    sediment_enabled: bool,
    // Tracking stats
    gold_exited: u32,
    sand_exited: u32,
    gold_trapped: u32,  // Gold behind riffles
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

impl App {
    fn new() -> Self {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.95;  // Slightly more PIC for stability
        sim.pressure_iterations = 80;

        // Configure sluice for vortex formation
        let sluice_config = SluiceConfig {
            slope: 0.20,           // ~11 degree slope - steeper for faster flow
            slick_plate_len: 6,    // Flat inlet section
            riffle_spacing: 8,     // More space between riffles for vortex development
            riffle_height: 3,      // Taller riffles for stronger vortices
            riffle_width: 1,       // Thin riffles
        };

        create_sluice(&mut sim, &sluice_config);

        // Collect solid cell positions for rendering
        let solid_instances = Self::collect_solids(&sim);

        println!("=== Gold Sluice 3D ===");
        println!("Grid: {}x{}x{} cells, cell size: {}m", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("Sluice: slope={:.0}%, riffle_height={}, riffle_spacing={}",
                 sluice_config.slope * 100.0, sluice_config.riffle_height, sluice_config.riffle_spacing);
        println!("Solid cells: {}", solid_instances.len());
        println!();
        println!("Controls:");
        println!("  SPACE = pause");
        println!("  E = toggle water emitter");
        println!("  S = toggle sediment spawning");
        println!("  G = toggle GPU/CPU simulation");
        println!("  R = reset");
        println!("  Drag = rotate camera");
        println!("  Scroll = zoom");
        println!();
        println!("Physics: Gold (yellow) should settle in vortex pockets behind riffles.");
        println!("         Sand (tan) should be carried downstream by the water flow.");

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            sluice_config,
            paused: false,
            camera_angle: 0.8,
            camera_pitch: 0.5,
            camera_distance: 2.5,
            frame: 0,
            solid_instances,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            cell_types: Vec::new(),
            use_gpu_sim: true,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            emitter_enabled: true,
            sediment_enabled: true,
            gold_exited: 0,
            sand_exited: 0,
            gold_trapped: 0,
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
                                color: [0.45, 0.35, 0.25, 0.25], // Brown, more transparent
                            });
                        }
                    }
                }
            }
        }
        solids
    }

    /// Emit water at the inlet
    fn emit_water(&mut self, count: usize) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        // Lower inlet velocity prevents pooling at inlet
        let flow_velocity = Vec3::new(0.5, 0.0, 0.0);
        spawn_inlet_water(&mut self.sim, &self.sluice_config, count, flow_velocity);
    }

    /// Emit mixed sediment (sand + gold)
    fn emit_sediment(&mut self, sand_count: usize, gold_count: usize) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        let flow_velocity = Vec3::new(0.5, 0.0, 0.0); // Same as water inlet

        // Spawn sand
        spawn_inlet_sediment(&mut self.sim, &self.sluice_config, sand_count, flow_velocity, SAND_DENSITY);

        // Spawn gold
        spawn_inlet_sediment(&mut self.sim, &self.sluice_config, gold_count, flow_velocity, GOLD_DENSITY);
    }

    /// Apply density-based settling to sediment particles
    ///
    /// Physics: FLIP applies same gravity to all particles via grid.
    /// Heavy particles need EXTRA downward force from buoyancy difference:
    ///   extra_gravity = g * (ρ - 1) / ρ
    /// Gold (19.3): extra = 9.8 * 18.3/19.3 = 9.3 m/s² (almost full extra g!)
    /// Sand (2.65): extra = 9.8 * 1.65/2.65 = 6.1 m/s² (~62% extra g)
    fn apply_settling(&mut self, dt: f32) {
        let base_gravity = 9.8;

        for p in &mut self.sim.particles.list {
            if p.density > WATER_DENSITY {
                // Buoyancy-reduced settling - correct physics
                let buoyancy_factor = (p.density - WATER_DENSITY) / p.density;
                let extra_gravity = base_gravity * buoyancy_factor;

                // Apply as velocity delta (not an artificial multiplier)
                p.velocity.y -= extra_gravity * dt;
            }
        }
    }

    /// Count particles behind riffles (trapped in vortices)
    fn count_trapped(&self) -> (u32, u32) {
        let dx = CELL_SIZE;
        let riffle_start = self.sluice_config.slick_plate_len;
        let riffle_spacing = self.sluice_config.riffle_spacing;
        let riffle_width = self.sluice_config.riffle_width;

        let mut gold_trapped = 0u32;
        let mut sand_trapped = 0u32;

        for p in &self.sim.particles.list {
            if p.density <= WATER_DENSITY {
                continue; // Skip water
            }

            let cell_x = (p.position.x / dx) as usize;

            // Check if particle is in the "wake" region behind a riffle
            // Wake is the 2-3 cells after each riffle
            if cell_x >= riffle_start {
                let rel_x = cell_x - riffle_start;
                let pos_in_cycle = rel_x % riffle_spacing;

                // Behind riffle = just after riffle ends (riffle_width .. riffle_width+3)
                if pos_in_cycle >= riffle_width && pos_in_cycle < riffle_width + 3 {
                    if p.density >= 10.0 {
                        gold_trapped += 1;
                    } else {
                        sand_trapped += 1;
                    }
                }
            }
        }

        (gold_trapped, sand_trapped)
    }

    fn reset_sim(&mut self) {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.95;
        sim.pressure_iterations = 80;

        create_sluice(&mut sim, &self.sluice_config);

        self.sim = sim;
        self.frame = 0;
        self.emitter_enabled = true;
        self.sediment_enabled = true;
        self.gold_exited = 0;
        self.sand_exited = 0;
        self.gold_trapped = 0;
        println!("Reset simulation");
    }

    fn particle_stats(&self) -> (usize, usize, usize, f32, f32, f32, f32, f32) {
        let mut water = 0;
        let mut sand = 0;
        let mut gold = 0;
        let mut max_vel = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_vx = 0.0f32;
        let mut min_x = f32::MAX;
        let mut max_x = 0.0f32;

        for p in &self.sim.particles.list {
            if p.density <= WATER_DENSITY {
                water += 1;
            } else if p.density >= 10.0 {
                gold += 1;
            } else {
                sand += 1;
            }
            max_vel = max_vel.max(p.velocity.length());
            sum_x += p.position.x;
            sum_vx += p.velocity.x;
            min_x = min_x.min(p.position.x);
            max_x = max_x.max(p.position.x);
        }

        let count = self.sim.particles.len() as f32;
        let avg_x = if count > 0.0 { sum_x / count } else { 0.0 };
        let avg_vx = if count > 0.0 { sum_vx / count } else { 0.0 };

        (water, sand, gold, max_vel, avg_x, avg_vx, min_x, max_x)
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        // Update simulation
        if !self.paused {
            let dt = 1.0 / 60.0;

            // Emit water continuously but at lower rate
            if self.emitter_enabled && self.frame % 4 == 0 {
                self.emit_water(20);
            }

            // Emit sediment periodically
            if self.sediment_enabled && self.frame % 30 == 0 {
                // 5:1 sand to gold ratio
                self.emit_sediment(8, 2);
            }

            if self.use_gpu_sim {
                // Sync from CPU
                self.positions.clear();
                self.velocities.clear();
                self.c_matrices.clear();

                for p in &self.sim.particles.list {
                    self.positions.push(p.position);
                    self.velocities.push(p.velocity);
                    self.c_matrices.push(p.affine_velocity);
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

                // Run GPU step
                let pressure_iters = self.sim.pressure_iterations as u32;
                if let (Some(gpu_flip), Some(gpu)) = (&self.gpu_flip, &self.gpu) {
                    gpu_flip.step(
                        &gpu.device,
                        &gpu.queue,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.c_matrices,
                        &self.cell_types,
                        dt,
                        -9.8,
                        FLOW_ACCEL,  // Downstream flow acceleration - applied on grid BEFORE pressure solve
                        pressure_iters,
                    );
                }

                // Sync back to CPU
                for (idx, p) in self.sim.particles.list.iter_mut().enumerate() {
                    if idx < self.velocities.len() {
                        p.velocity = self.velocities[idx];
                        p.affine_velocity = self.c_matrices[idx];
                    }
                }

                // Apply settling force (density-based, not handled on GPU)
                // Flow acceleration is now handled on GPU BEFORE pressure solve
                self.apply_settling(dt);

                // Advect from density-corrected position (positions[] was modified by step())
                for (idx, p) in self.sim.particles.list.iter_mut().enumerate() {
                    p.position = self.positions[idx] + p.velocity * dt;
                }

                // Enforce boundaries using SDF
                for p in &mut self.sim.particles.list {
                    let sdf = self.sim.grid.sample_sdf(p.position);
                    if sdf < 0.0 {
                        // Push out of solid
                        let grad = self.sim.grid.sdf_gradient(p.position);
                        p.position -= grad * sdf * 1.1;

                        // Kill velocity into solid
                        let vel_into = p.velocity.dot(grad);
                        if vel_into < 0.0 {
                            p.velocity -= grad * vel_into;
                        }
                    }

                    // Track exits at the right edge
                    let max_x = GRID_WIDTH as f32 * CELL_SIZE;
                    if p.position.x >= max_x {
                        if p.density >= 10.0 {
                            self.gold_exited += 1;
                        } else if p.density > WATER_DENSITY {
                            self.sand_exited += 1;
                        }
                        // Mark for removal
                        p.position.x = 1000.0;
                    }
                }

                // Remove exited particles
                self.sim.particles.list.retain(|p| p.position.x < 100.0);
            } else {
                // CPU simulation (note: flow acceleration not implemented in CPU path)
                self.sim.update(dt);

                // Apply settling force
                self.apply_settling(dt);

                // Track exits
                let max_x = GRID_WIDTH as f32 * CELL_SIZE;
                for p in &self.sim.particles.list {
                    if p.position.x >= max_x {
                        if p.density >= 10.0 {
                            self.gold_exited += 1;
                        } else if p.density > WATER_DENSITY {
                            self.sand_exited += 1;
                        }
                    }
                }
                self.sim.particles.list.retain(|p| p.position.x < max_x);
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
        if self.frame % 60 == 0 {
            let (water, sand, gold, _max_vel, avg_x, avg_vx, _min_x, _max_x_pos) = self.particle_stats();
            let (gold_trap, sand_trap) = self.count_trapped();
            let mode = if self.use_gpu_sim { "GPU" } else { "CPU" };
            let world_max_x = GRID_WIDTH as f32 * CELL_SIZE;

            // Key diagnostic: are particles flowing or pooling?
            let flow_status = if avg_vx > 0.3 {
                "FLOW"
            } else if avg_vx < -0.1 {
                "BACK!"
            } else {
                "SLOW"
            };

            println!(
                "[{}] F{:4} | W:{:4} S:{:3} G:{:2} | avgX={:.2}/{:.2} Vx={:.2} {} | Trap G:{:2} S:{:2} | Exit G:{:2} S:{:3}",
                mode,
                self.frame,
                water, sand, gold,
                avg_x, world_max_x,
                avg_vx, flow_status,
                gold_trap, sand_trap,
                self.gold_exited, self.sand_exited,
            );
        }

        // Create particle instances with material-based colors
        let instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .map(|p| {
                let speed = p.velocity.length();
                let t = (speed / 3.0).min(1.0);

                // Color based on material
                let color = if p.density >= 10.0 {
                    // Gold: bright yellow
                    [1.0, 0.85, 0.1, 0.95]
                } else if p.density > WATER_DENSITY {
                    // Sand: tan/brown
                    [0.76, 0.70, 0.50, 0.9]
                } else {
                    // Water: blue with velocity-based brightness
                    [0.2 + t * 0.3, 0.4 + t * 0.3, 0.8 - t * 0.2, 0.7]
                };

                ParticleInstance {
                    position: [p.position.x, p.position.y, p.position.z],
                    color,
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
            GRID_HEIGHT as f32 * CELL_SIZE * 0.4,
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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.08, b: 0.12, a: 1.0 }),
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

            // Draw particles FIRST (so they're visible through transparent solids)
            pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
            pass.draw(0..4, 0..instances.len() as u32);

            // Draw solids (transparent, on top)
            pass.set_vertex_buffer(1, gpu.solid_buffer.slice(..));
            pass.draw(0..4, 0..self.solid_instances.len() as u32);
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
                        .with_title("Gold Sluice 3D - Vortex Sediment Separation")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1400, 900)),
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

        // Request higher limits for GPU FLIP solver
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
                        }
                        PhysicalKey::Code(KeyCode::KeyG) => {
                            self.use_gpu_sim = !self.use_gpu_sim;
                            println!("Simulation mode: {}", if self.use_gpu_sim { "GPU" } else { "CPU" });
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            self.emitter_enabled = !self.emitter_enabled;
                            println!("Water emitter: {}", if self.emitter_enabled { "ON" } else { "OFF" });
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            self.sediment_enabled = !self.sediment_enabled;
                            println!("Sediment spawning: {}", if self.sediment_enabled { "ON" } else { "OFF" });
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
                self.camera_distance = (self.camera_distance - scroll * 0.2).clamp(0.3, 10.0);
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
    // Size based on alpha (sediment slightly larger)
    let size = 0.012 + in.color.a * 0.003;

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
