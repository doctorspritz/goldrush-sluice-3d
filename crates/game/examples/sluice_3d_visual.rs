//! 3D Sluice Box - Visual Demo (GPU Accelerated)
//!
//! Renders a 3D sluice box simulation with continuous water inlet.
//! Uses GPU compute shaders for the FLIP simulation.
//! Run with: cargo run --example sluice_3d_visual --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Mat4, Vec3};
use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig};
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

// Fine grid for vortex resolution - need 5+ cells per riffle height
const GRID_WIDTH: usize = 200;  // Flow direction (X)
const GRID_HEIGHT: usize = 80;  // Vertical (Y)
const GRID_DEPTH: usize = 40;   // Width (Z)
const CELL_SIZE: f32 = 0.016;   // Fine enough for vortices (~5 cells per riffle)
const MAX_PARTICLES: usize = 800000;  // More particles for finer grid
const MAX_SOLIDS: usize = 100000;  // More solid cells with finer grid

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
    max_frames: u32,
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
        // Tilted gravity: sluice has 10% slope, sin(atan(0.1)) ≈ 0.10
        // gravity_x = 9.8 * 0.10 ≈ 1.0 m/s² (pushes water downstream)
        sim.gravity = Vec3::new(1.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        // CRITICAL: For a 200-cell-wide grid, pressure information propagates ~1 cell/iteration.
        // Need at least width iterations for full convergence. Using 500 for headroom.
        sim.pressure_iterations = 500;

        // Create sluice geometry
        // CRITICAL: riffle_height MUST be < water depth above floor!
        // - Water spawns at floor + 2 cells (see sluice.rs spawn_inlet_water)
        // - If riffle_height >= 2, water can't naturally flow over riffles
        // - Water would need to BUILD UP behind each riffle, losing energy
        // - Setting riffle_height = 1 ensures water depth (2) > riffle height (1)
        // Test with very short riffles to verify water CAN flow through the channel
        // If this works, the issue is with riffle height blocking flow
        let sluice_config = SluiceConfig {
            slope: 0.10,            // 10% grade
            slick_plate_len: 20,    // Flat inlet section
            riffle_spacing: 16,     // Space between riffles
            riffle_height: 4,       // TALLER - test if water stacks to overflow
            riffle_width: 2,        // Thicker riffles
        };
        create_sluice(&mut sim, &sluice_config);

        // Collect solid cell positions for rendering
        let solid_instances = Self::collect_solids(&sim);

        // Spawn initial water at inlet
        spawn_inlet_water(&mut sim, &sluice_config, 2000, Vec3::new(1.5, 0.0, 0.0));

        println!("Spawned {} particles", sim.particle_count());
        println!("Solid cells: {}", solid_instances.len());
        println!("Controls: SPACE=pause, R=reset, G=toggle GPU/CPU, ESC=quit, Click+Drag=rotate, Scroll=zoom");
        println!("Running GPU simulation for 30000 frames...");

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            sluice_config,
            paused: false,
            camera_angle: 0.8,  // Angled view to see flow along X and depth along Z
            camera_pitch: 0.5,  // Higher elevation to see the slope
            camera_distance: 4.5,  // Farther out to see the whole sluice
            frame: 0,
            solid_instances,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            cell_types: Vec::new(),
            use_gpu_sim: true,  // GPU mode - testing with fixes
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            max_frames: 30000,
        }
    }

    fn collect_solids(sim: &FlipSimulation3D) -> Vec<ParticleInstance> {
        let mut solids = Vec::new();
        let dx = sim.grid.cell_size;
        let w = sim.grid.width;
        let h = sim.grid.height;
        let d = sim.grid.depth;

        // Only render surface solid cells (cells with at least one non-solid neighbor)
        // This dramatically reduces the number of rendered solids
        for k in 0..d {
            for j in 0..h {
                for i in 0..w {
                    let idx = sim.grid.cell_index(i, j, k);
                    if !sim.grid.solid[idx] {
                        continue;
                    }

                    // Check if this is a surface cell (has at least one air/fluid neighbor)
                    let is_surface =
                        (i == 0 || !sim.grid.is_solid(i - 1, j, k)) ||
                        (i == w - 1 || !sim.grid.is_solid(i + 1, j, k)) ||
                        (j == 0 || !sim.grid.is_solid(i, j - 1, k)) ||
                        (j == h - 1 || !sim.grid.is_solid(i, j + 1, k)) ||
                        (k == 0 || !sim.grid.is_solid(i, j, k - 1)) ||
                        (k == d - 1 || !sim.grid.is_solid(i, j, k + 1));

                    if !is_surface {
                        continue;
                    }

                    // Cell center position
                    let pos = [
                        (i as f32 + 0.5) * dx,
                        (j as f32 + 0.5) * dx,
                        (k as f32 + 0.5) * dx,
                    ];
                    // Brown/gray color for solid cells
                    solids.push(ParticleInstance {
                        position: pos,
                        color: [0.45, 0.38, 0.32, 1.0],
                    });
                }
            }
        }
        solids
    }

    fn reset_sim(&mut self) {
        self.sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        // Tilted gravity: sluice has 10% slope, sin(atan(0.1)) ≈ 0.10
        self.sim.gravity = Vec3::new(1.0, -9.8, 0.0);
        self.sim.flip_ratio = 0.97;
        self.sim.pressure_iterations = 500;  // Must match new() - need 500 for 200-wide grid

        create_sluice(&mut self.sim, &self.sluice_config);
        self.solid_instances = Self::collect_solids(&self.sim);

        spawn_inlet_water(&mut self.sim, &self.sluice_config, 500, Vec3::new(1.5, 0.0, 0.0));
        self.frame = 0;

        // Clear GPU particle data
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.cell_types.clear();
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

        // Request higher limits for GPU compute (need 11+ storage buffers per stage for P2G 3D)
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024; // 256 MB

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
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

        // Instance buffer for water particles
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Solid buffer for terrain
        let solid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Solid Buffer"),
            size: (MAX_SOLIDS * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload solid instances
        if !self.solid_instances.is_empty() {
            queue.write_buffer(
                &solid_buffer,
                0,
                bytemuck::cast_slice(&self.solid_instances),
            );
        }

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

        // Create GPU FLIP simulation
        let gpu_flip = GpuFlip3D::new(
            &device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );
        self.gpu_flip = Some(gpu_flip);

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
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(window) = &self.window else { return };

        // Update simulation
        if !self.paused {
            let dt = 1.0 / 120.0;
            let substeps = 2;

            if self.use_gpu_sim {
                if let Some(gpu_flip) = &self.gpu_flip {
                    for _ in 0..substeps {
                        // Sync particle data from CPU sim to GPU format
                        let particle_count = self.sim.particles.list.len();
                        self.positions.clear();
                        self.velocities.clear();
                        self.c_matrices.clear();

                        self.positions.reserve(particle_count);
                        self.velocities.reserve(particle_count);
                        self.c_matrices.reserve(particle_count);

                        for p in &self.sim.particles.list {
                            self.positions.push(p.position);
                            self.velocities.push(p.velocity);
                            self.c_matrices.push(p.affine_velocity);
                        }

                        // Build cell types from grid
                        let w = self.sim.grid.width;
                        let h = self.sim.grid.height;
                        let d = self.sim.grid.depth;
                        self.cell_types.clear();
                        self.cell_types.resize(w * h * d, 0); // 0 = air

                        // Mark solid and fluid cells
                        for k in 0..d {
                            for j in 0..h {
                                for i in 0..w {
                                    let idx = k * w * h + j * w + i;
                                    if self.sim.grid.is_solid(i, j, k) {
                                        self.cell_types[idx] = 2; // solid
                                    }
                                }
                            }
                        }

                        // Mark fluid cells based on particle presence
                        for p in &self.sim.particles.list {
                            let i = (p.position.x / CELL_SIZE).floor() as i32;
                            let j = (p.position.y / CELL_SIZE).floor() as i32;
                            let k = (p.position.z / CELL_SIZE).floor() as i32;
                            if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                                let idx = k as usize * w * h + j as usize * w + i as usize;
                                if self.cell_types[idx] != 2 {
                                    self.cell_types[idx] = 1; // fluid
                                }
                            }
                        }

                        // Run GPU FLIP step
                        // Use the sim's gravity (tilted for sluice slope) and pressure_iterations
                        gpu_flip.step(
                            &gpu.device,
                            &gpu.queue,
                            &self.positions,
                            &mut self.velocities,
                            &mut self.c_matrices,
                            &self.cell_types,
                            dt,
                            self.sim.gravity,  // Tilted gravity: (1.0, -9.8, 0.0)
                            self.sim.pressure_iterations as u32,
                        );

                        // Copy velocities back to particles, apply settling, and advect
                        let gravity = -9.8_f32;
                        let settling_factor = 0.05; // Drag-limited settling rate

                        for (idx, p) in self.sim.particles.list.iter_mut().enumerate() {
                            if idx < self.velocities.len() {
                                p.velocity = self.velocities[idx];
                                p.affine_velocity = self.c_matrices[idx];
                            }

                            // Density-based settling: heavier particles sink faster
                            // Settling velocity = (ρ_p - ρ_water) / ρ_water * g * factor
                            if p.density > 1.0 {
                                let buoyancy_ratio = (p.density - 1.0) / p.density;
                                let settling_vel = buoyancy_ratio * gravity * settling_factor;
                                p.velocity.y += settling_vel * dt;
                            }

                            // Advect particle
                            p.position += p.velocity * dt;

                            // Domain boundary handling (same as CPU path)
                            let min = CELL_SIZE * 0.5;
                            let max_z = (d as f32 - 0.5) * CELL_SIZE;
                            // Note: x and y max are OPEN boundaries (outlet, top) - handled by retain below

                            // Inlet (x=0): CLOSED - bounce back
                            if p.position.x < min {
                                p.position.x = min;
                                p.velocity.x = p.velocity.x.abs() * 0.1;
                            }
                            // Outlet (x=max): OPEN - let particles exit

                            // Floor (y=0): CLOSED - bounce
                            if p.position.y < min {
                                p.position.y = min;
                                p.velocity.y = p.velocity.y.abs() * 0.1;
                            }
                            // Top (y=max): OPEN - let particles exit

                            // Side walls (z): CLOSED - bounce
                            if p.position.z < min {
                                p.position.z = min;
                                p.velocity.z = p.velocity.z.abs() * 0.1;
                            }
                            if p.position.z > max_z {
                                p.position.z = max_z;
                                p.velocity.z = -p.velocity.z.abs() * 0.1;
                            }

                            // SDF-based collision with solid geometry (same as CPU path)
                            let sdf = self.sim.grid.sample_sdf(p.position);
                            if sdf < 0.0 {
                                // Particle is inside solid - push out along gradient
                                let normal = self.sim.grid.sdf_gradient(p.position);
                                let penetration = -sdf + CELL_SIZE * 0.1; // Small margin
                                p.position += normal * penetration;

                                // Reflect ONLY velocity component into solid
                                let vel_into_solid = p.velocity.dot(normal);
                                if vel_into_solid < 0.0 {
                                    // Remove velocity into solid, add small bounce
                                    p.velocity -= normal * vel_into_solid * 1.1;
                                }
                            }
                        }

                        // Remove particles that went out of bounds or have invalid velocities
                        self.sim.particles.list.retain(|p| {
                            p.position.x > 0.0
                                && p.position.x < (w as f32 - 1.0) * CELL_SIZE
                                && p.position.y > 0.0
                                && p.position.y < (h as f32 - 1.0) * CELL_SIZE
                                && p.position.z > 0.0
                                && p.position.z < (d as f32 - 1.0) * CELL_SIZE
                                && p.velocity.is_finite()
                                && p.position.is_finite()
                        });
                    }
                }
            } else {
                // CPU simulation fallback
                for _ in 0..substeps {
                    self.sim.update(dt);
                }
            }

            // Continuously spawn water at inlet (water-only mode)
            if self.frame % 2 == 0 && self.sim.particle_count() < MAX_PARTICLES - 500 {
                let inlet_vel = Vec3::new(1.5, 0.0, 0.0);
                spawn_inlet_water(&mut self.sim, &self.sluice_config, 50, inlet_vel);
            }

            self.frame += 1;
            self.fps_frame_count += 1;

            // FPS tracking - print every second
            let now = Instant::now();
            let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
            if elapsed >= 1.0 {
                self.current_fps = self.fps_frame_count as f32 / elapsed;
                let avg_vel: Vec3 = if !self.sim.particles.list.is_empty() {
                    self.sim.particles.list.iter()
                        .map(|p| p.velocity)
                        .fold(Vec3::ZERO, |a, b| a + b) / self.sim.particles.list.len() as f32
                } else {
                    Vec3::ZERO
                };
                let max_vel = self.sim.particles.list.iter()
                    .map(|p| p.velocity.length())
                    .fold(0.0f32, f32::max);
                // Track water level (max Y) to see if water is building up
                let max_y = self.sim.particles.list.iter()
                    .map(|p| p.position.y)
                    .fold(0.0f32, f32::max);
                let max_x = self.sim.particles.list.iter()
                    .map(|p| p.position.x)
                    .fold(0.0f32, f32::max);
                println!(
                    "Frame {:5}/{} | FPS: {:5.1} | Particles: {:6} | AvgVel: ({:6.2},{:6.2},{:6.2}) | MaxVel: {:5.2} | MaxY: {:5.3} | MaxX: {:5.3}",
                    self.frame, self.max_frames, self.current_fps,
                    self.sim.particle_count(),
                    avg_vel.x, avg_vel.y, avg_vel.z, max_vel, max_y, max_x
                );
                self.fps_frame_count = 0;
                self.last_fps_time = now;
            }

            // Stop at max frames
            if self.frame >= self.max_frames {
                println!("\n=== SIMULATION COMPLETE: {} frames ===", self.frame);
                println!("Final particle count: {}", self.sim.particle_count());
                self.paused = true;
            }
        }

        // Camera - look at center of sluice channel (where water flows)
        // Floor at inlet (x=0) is at y ≈ (width-1)*slope = 19.9 cells
        // Floor at outlet (x=width) is at y = 0
        // Look at middle of the slope
        let mid_x = GRID_WIDTH as f32 * CELL_SIZE * 0.5;
        let floor_at_mid = ((GRID_WIDTH as f32 - 1.0) * 0.5 * self.sluice_config.slope) * CELL_SIZE;
        let center = Vec3::new(
            mid_x,
            floor_at_mid + CELL_SIZE * 3.0,  // A few cells above floor
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        // Spherical: angle = azimuth (horizontal), pitch = elevation
        let cos_pitch = self.camera_pitch.cos();
        let camera_pos = center
            + Vec3::new(
                self.camera_angle.cos() * cos_pitch * self.camera_distance,
                self.camera_pitch.sin() * self.camera_distance,
                self.camera_angle.sin() * cos_pitch * self.camera_distance,
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

        // Update particle instances with colors based on density
        let water_instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .take(MAX_PARTICLES)
            .map(|p| {
                // Color based on particle density
                let color = if p.density > 10.0 {
                    // Gold (density ~19.3) - rich gold color
                    [0.85, 0.65, 0.1, 1.0]
                } else if p.density > 4.0 {
                    // Magnetite (density ~5.2) - dark gray
                    [0.2, 0.2, 0.22, 1.0]
                } else if p.density > 2.0 {
                    // Sand (density ~2.65) - pale tan
                    [0.95, 0.9, 0.8, 1.0]
                } else if p.density > 1.5 {
                    // Mud (density ~2.0) - brown
                    [0.6, 0.4, 0.2, 1.0]
                } else {
                    // Water (density 1.0) - blue with speed variation
                    let speed = p.velocity.length();
                    let t = (speed / 3.0).min(1.0);
                    [0.2 + t * 0.2, 0.4 + t * 0.3, 0.85, 0.75]
                };
                ParticleInstance {
                    position: p.position.to_array(),
                    color,
                }
            })
            .collect();

        if !water_instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.instance_buffer,
                0,
                bytemuck::cast_slice(&water_instances),
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

            // Draw solids first (terrain)
            let solid_count = self.solid_instances.len().min(MAX_SOLIDS);
            if solid_count > 0 {
                pass.set_vertex_buffer(1, gpu.solid_buffer.slice(..));
                pass.draw(0..6, 0..solid_count as u32);
            }

            // Draw water particles
            if !water_instances.is_empty() {
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                pass.draw(0..6, 0..water_instances.len() as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("3D Sluice Box - FLIP Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

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
                        PhysicalKey::Code(KeyCode::KeyG) => {
                            self.use_gpu_sim = !self.use_gpu_sim;
                            println!(
                                "Simulation mode: {}",
                                if self.use_gpu_sim { "GPU" } else { "CPU" }
                            );
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

                        // Horizontal drag rotates camera angle
                        self.camera_angle += dx * 0.01;

                        // Vertical drag adjusts pitch (clamped to avoid gimbal lock)
                        self.camera_pitch = (self.camera_pitch - dy * 0.01)
                            .clamp(-1.4, 1.4); // ~80 degrees up/down
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.3).clamp(1.0, 20.0);
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
    let size = 0.015;

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
