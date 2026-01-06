//! 15-Degree Sluice Box Demo
//!
//! A realistic sluice with 15-degree slope, evenly spaced riffles,
//! inlet at top, and outlet at bottom. Riffles designed to
//! show water pooling and vortex formation.
//!
//! Built from scratch - no old sluice module dependencies.
//!
//! Run with: cargo run --example sluice_15deg --release

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

// ============================================================================
// SLUICE CONFIGURATION
// ============================================================================

/// 15 degrees in radians
const SLOPE_ANGLE_DEG: f32 = 15.0;
/// tan(15°) ≈ 0.268 - rise per horizontal unit
const SLOPE_TAN: f32 = 0.268;

/// Grid dimensions
const GRID_WIDTH: usize = 150;   // Flow direction (X) - ~2.4m sluice length
const GRID_HEIGHT: usize = 90;   // Vertical (Y) - enough for slope + water + headroom
const GRID_DEPTH: usize = 25;    // Width (Z) - channel width
const CELL_SIZE: f32 = 0.016;    // 1.6cm cells - fine enough for vortices

const MAX_PARTICLES: usize = 500_000;
const MAX_SOLIDS: usize = 100_000;

/// Flow acceleration: g * sin(15°) ≈ 9.8 * 0.259 ≈ 2.54 m/s²
const FLOW_ACCEL: f32 = 2.54;

/// Riffle configuration
const RIFFLE_HEIGHT_CELLS: usize = 6;     // 6 cells = ~9.6cm - tall for vortex
const RIFFLE_THICKNESS_CELLS: usize = 2;  // 2 cells thick
const RIFFLE_SPACING_CELLS: usize = 20;   // 20 cells between riffles
const NUM_RIFFLES: usize = 5;             // 5 evenly spaced riffles
const SLICK_PLATE_CELLS: usize = 15;      // Smooth section before first riffle

/// Wall configuration
const WALL_HEIGHT_CELLS: usize = 10;      // Side wall height above floor

// ============================================================================
// GPU TYPES
// ============================================================================

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

// ============================================================================
// APPLICATION STATE
// ============================================================================

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
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    bed_height: Vec<f32>,
    cell_types: Vec<u32>,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
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
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 350;

        // Build sluice geometry from scratch
        build_sluice(&mut sim);
        let solid_instances = collect_solids(&sim);

        // Spawn initial water
        spawn_water(&mut sim, 2500, Vec3::new(0.8, 0.0, 0.0));

        println!("=== 15-DEGREE SLUICE BOX ===");
        println!("Grid: {}x{}x{} cells at {:.3}m", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("Physical size: {:.2}m x {:.2}m x {:.2}m",
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE
        );
        println!("Slope: {:.1}° ({:.1}% grade)", SLOPE_ANGLE_DEG, SLOPE_TAN * 100.0);
        println!("Riffle height: {} cells = {:.1}cm", RIFFLE_HEIGHT_CELLS,
            RIFFLE_HEIGHT_CELLS as f32 * CELL_SIZE * 100.0);
        println!("Riffle spacing: {} cells = {:.1}cm", RIFFLE_SPACING_CELLS,
            RIFFLE_SPACING_CELLS as f32 * CELL_SIZE * 100.0);
        println!("{} riffles", NUM_RIFFLES);
        println!("Initial particles: {}", sim.particle_count());
        println!("Solid cells: {}", solid_instances.len());
        println!("\nControls: SPACE=pause, R=reset, ESC=quit");
        println!("Mouse: Click+Drag=rotate, Scroll=zoom\n");

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            paused: false,
            camera_angle: 0.2,
            camera_pitch: 0.35,
            camera_distance: 3.0,
            frame: 0,
            solid_instances,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            bed_height: vec![0.0; GRID_WIDTH * GRID_DEPTH],
            cell_types: Vec::new(),
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
        }
    }

    fn reset_sim(&mut self) {
        self.sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        self.sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        self.sim.flip_ratio = 0.97;
        self.sim.pressure_iterations = 350;

        build_sluice(&mut self.sim);
        self.solid_instances = collect_solids(&self.sim);
        spawn_water(&mut self.sim, 2500, Vec3::new(0.8, 0.0, 0.0));
        self.frame = 0;

        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.cell_types.clear();

        println!("Simulation reset");
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

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

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

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Solid Buffer"),
            size: (MAX_SOLIDS * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if !self.solid_instances.is_empty() {
            queue.write_buffer(&solid_buffer, 0, bytemuck::cast_slice(&self.solid_instances));
        }

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

        if !self.paused {
            let dt = 1.0 / 120.0;
            let substeps = 2;

            if let Some(gpu_flip) = &mut self.gpu_flip {
                for _ in 0..substeps {
                    let particle_count = self.sim.particles.list.len();
                    self.positions.clear();
                    self.velocities.clear();
                    self.c_matrices.clear();
                    self.densities.clear();

                    self.positions.reserve(particle_count);
                    self.velocities.reserve(particle_count);
                    self.c_matrices.reserve(particle_count);
                    self.densities.reserve(particle_count);

                    for p in &self.sim.particles.list {
                        self.positions.push(p.position);
                        self.velocities.push(p.velocity);
                        self.c_matrices.push(p.affine_velocity);
                        self.densities.push(p.density);
                    }

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

                    gpu_flip.step(
                        &gpu.device,
                        &gpu.queue,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.c_matrices,
                        &self.densities,
                        &self.cell_types,
                        None,
                        Some(&self.bed_height),
                        dt,
                        -9.8,
                        FLOW_ACCEL,
                        self.sim.pressure_iterations as u32,
                    );

                    for (idx, p) in self.sim.particles.list.iter_mut().enumerate() {
                        if idx < self.velocities.len() {
                            p.velocity = self.velocities[idx];
                            p.affine_velocity = self.c_matrices[idx];
                        }

                        p.position = self.positions[idx] + p.velocity * dt;

                        let min = CELL_SIZE * 0.5;
                        let max_z = (d as f32 - 0.5) * CELL_SIZE;

                        if p.position.x < min {
                            p.position.x = min;
                            p.velocity.x = p.velocity.x.abs() * 0.1;
                        }
                        if p.position.y < min {
                            p.position.y = min;
                            p.velocity.y = p.velocity.y.abs() * 0.1;
                        }
                        if p.position.z < min {
                            p.position.z = min;
                            p.velocity.z = p.velocity.z.abs() * 0.1;
                        }
                        if p.position.z > max_z {
                            p.position.z = max_z;
                            p.velocity.z = -p.velocity.z.abs() * 0.1;
                        }

                        let sdf = self.sim.grid.sample_sdf(p.position);
                        if sdf < 0.0 {
                            let normal = self.sim.grid.sdf_gradient(p.position);
                            let penetration = -sdf + CELL_SIZE * 0.1;
                            p.position += normal * penetration;

                            let vel_into_solid = p.velocity.dot(normal);
                            if vel_into_solid < 0.0 {
                                p.velocity -= normal * vel_into_solid * 1.1;
                            }
                        }
                    }

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

            // Continuous water inlet
            if self.frame % 3 == 0 && self.sim.particle_count() < MAX_PARTICLES - 100 {
                spawn_water(&mut self.sim, 30, Vec3::new(0.8, 0.0, 0.0));
            }

            self.frame += 1;
            self.fps_frame_count += 1;

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
                let max_y = self.sim.particles.list.iter()
                    .map(|p| p.position.y)
                    .fold(0.0f32, f32::max);
                let max_x = self.sim.particles.list.iter()
                    .map(|p| p.position.x)
                    .fold(0.0f32, f32::max);
                println!(
                    "Frame {:5} | FPS: {:5.1} | Particles: {:6} | AvgVel: ({:5.2},{:5.2},{:5.2}) | MaxX: {:.2}m MaxY: {:.2}m",
                    self.frame, self.current_fps,
                    self.sim.particle_count(),
                    avg_vel.x, avg_vel.y, avg_vel.z, max_x, max_y
                );
                self.fps_frame_count = 0;
                self.last_fps_time = now;
            }
        }

        // Camera: look at center of sluice, positioned to see the slope
        let center_x = GRID_WIDTH as f32 * CELL_SIZE * 0.5;
        let inlet_floor_y = get_floor_y(0);
        let outlet_floor_y = get_floor_y(GRID_WIDTH - 1);
        let center_y = ((inlet_floor_y + outlet_floor_y) as f32 / 2.0 + 5.0) * CELL_SIZE;
        let center_z = GRID_DEPTH as f32 * CELL_SIZE * 0.5;
        let center = Vec3::new(center_x, center_y, center_z);

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

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Color particles: blue base, lighter for fast flow, cyan tint for vortex motion
        let water_instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .take(MAX_PARTICLES)
            .map(|p| {
                let speed = p.velocity.length();
                // Detect vortex motion: high vertical or lateral velocity relative to forward
                let vorticity = (p.velocity.y.abs() + p.velocity.z.abs()) / (speed + 0.1);

                let t = (speed / 2.0).min(1.0);
                let v = vorticity.min(1.0);

                // Blue water, greener for vortex, lighter for speed
                let color = [
                    0.15 + v * 0.25,         // R: more for vortex
                    0.35 + t * 0.35 + v * 0.15, // G: speed + vortex
                    0.75 + t * 0.15,         // B: base water
                    0.85,
                ];
                ParticleInstance {
                    position: p.position.to_array(),
                    color,
                }
            })
            .collect();

        if !water_instances.is_empty() {
            gpu.queue.write_buffer(&gpu.instance_buffer, 0, bytemuck::cast_slice(&water_instances));
        }

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
                            b: 0.04,
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

            let solid_count = self.solid_instances.len().min(MAX_SOLIDS);
            if solid_count > 0 {
                pass.set_vertex_buffer(1, gpu.solid_buffer.slice(..));
                pass.draw(0..6, 0..solid_count as u32);
            }

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
            .with_title("15-Degree Sluice Box")
            .with_inner_size(winit::dpi::LogicalSize::new(1400, 800));

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
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                            println!("{}", if self.paused { "PAUSED" } else { "RUNNING" });
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => self.reset_sim(),
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
                        self.camera_pitch = (self.camera_pitch - dy * 0.01).clamp(-1.4, 1.4);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.3).clamp(0.5, 12.0);
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

// ============================================================================
// SLUICE GEOMETRY (BUILT FROM SCRATCH)
// ============================================================================

/// Calculate floor Y at given X position (inlet is high, outlet is low)
fn get_floor_y(x: usize) -> usize {
    // Floor drops as X increases
    // At x=0 (inlet): floor is highest
    // At x=width-1 (outlet): floor is lowest (y=0)
    let drop = (GRID_WIDTH - 1 - x) as f32 * SLOPE_TAN;
    (drop as usize).min(GRID_HEIGHT - 15)
}

/// Build the sluice geometry from scratch
fn build_sluice(sim: &mut FlipSimulation3D) {
    sim.grid.clear_solids();

    // 1. SLOPED FLOOR
    // Floor is highest at inlet (x=0), lowest at outlet (x=width-1)
    for i in 0..GRID_WIDTH {
        let floor_y = get_floor_y(i);
        for k in 0..GRID_DEPTH {
            for j in 0..=floor_y {
                sim.grid.set_solid(i, j, k);
            }
        }
    }

    // 2. SIDE WALLS (contain water laterally)
    for i in 0..GRID_WIDTH {
        let floor_y = get_floor_y(i);
        for dy in 1..=WALL_HEIGHT_CELLS {
            let y = floor_y + dy;
            if y < GRID_HEIGHT {
                sim.grid.set_solid(i, y, 0);                  // Front wall (z=0)
                sim.grid.set_solid(i, y, GRID_DEPTH - 1);     // Back wall (z=max)
            }
        }
    }

    // 3. INLET WALL (prevent water escaping backwards at x=0)
    let inlet_floor_y = get_floor_y(0);
    for dy in 1..=WALL_HEIGHT_CELLS {
        let y = inlet_floor_y + dy;
        if y < GRID_HEIGHT {
            for k in 0..GRID_DEPTH {
                sim.grid.set_solid(0, y, k);
            }
        }
    }

    // 4. OUTLET IS OPEN (x=width-1) - water exits freely

    // 5. RIFFLES - evenly spaced bars across the channel
    for riffle_idx in 0..NUM_RIFFLES {
        // Position riffles evenly after slick plate
        let riffle_x = SLICK_PLATE_CELLS + riffle_idx * RIFFLE_SPACING_CELLS;

        if riffle_x + RIFFLE_THICKNESS_CELLS >= GRID_WIDTH - 3 {
            break; // Don't place riffles too close to outlet
        }

        let floor_y = get_floor_y(riffle_x);

        // Build riffle across width (excluding side walls)
        for dx in 0..RIFFLE_THICKNESS_CELLS {
            let x = riffle_x + dx;
            if x >= GRID_WIDTH {
                break;
            }
            // Span interior (not side walls)
            for k in 1..GRID_DEPTH - 1 {
                for dy in 1..=RIFFLE_HEIGHT_CELLS {
                    let y = floor_y + dy;
                    if y < GRID_HEIGHT {
                        sim.grid.set_solid(x, y, k);
                    }
                }
            }
        }
    }

    println!("Built {} riffles, {} cells tall, {} cells apart",
        NUM_RIFFLES, RIFFLE_HEIGHT_CELLS, RIFFLE_SPACING_CELLS);

    // Compute SDF for particle collision
    sim.grid.compute_sdf();
}

/// Spawn water at the inlet
fn spawn_water(sim: &mut FlipSimulation3D, count: usize, velocity: Vec3) {
    let dx = CELL_SIZE;

    // Spawn just inside inlet, above the first riffle
    let first_riffle_x = SLICK_PLATE_CELLS;
    let first_riffle_floor_y = get_floor_y(first_riffle_x);
    let first_riffle_top_y = first_riffle_floor_y + RIFFLE_HEIGHT_CELLS;

    // Water spawns 2-3 cells above riffle top so it can overflow
    let spawn_x_start = 2.0 * dx;
    let spawn_y_base = (first_riffle_top_y as f32 + 3.0) * dx;

    // Span across channel width (inside walls)
    let z_start = 1.5 * dx;
    let z_end = (GRID_DEPTH as f32 - 1.5) * dx;

    let num_z = ((GRID_DEPTH - 3) as f32).sqrt().ceil() as usize;
    let num_per_z = (count / num_z.max(1)).max(1);
    let cluster_dim = (num_per_z as f32).sqrt().ceil() as usize;
    let spacing = dx * 0.35;

    let mut spawned = 0;

    for zi in 0..num_z {
        let t = if num_z > 1 { zi as f32 / (num_z - 1) as f32 } else { 0.5 };
        let z = z_start + t * (z_end - z_start);

        for xi in 0..cluster_dim {
            for yi in 0..cluster_dim {
                if spawned >= count {
                    return;
                }

                let pos = Vec3::new(
                    spawn_x_start + xi as f32 * spacing,
                    spawn_y_base + yi as f32 * spacing,
                    z,
                );

                // Check not inside solid
                if sim.grid.sample_sdf(pos) > 0.0 {
                    sim.spawn_particle_with_velocity(pos, velocity);
                    spawned += 1;
                }
            }
        }
    }
}

/// Collect surface solid cells for rendering
fn collect_solids(sim: &FlipSimulation3D) -> Vec<ParticleInstance> {
    let mut solids = Vec::new();
    let dx = sim.grid.cell_size;
    let w = sim.grid.width;
    let h = sim.grid.height;
    let d = sim.grid.depth;

    for k in 0..d {
        for j in 0..h {
            for i in 0..w {
                let idx = sim.grid.cell_index(i, j, k);
                if !sim.grid.solid[idx] {
                    continue;
                }

                // Only render surface cells
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

                let pos = [
                    (i as f32 + 0.5) * dx,
                    (j as f32 + 0.5) * dx,
                    (k as f32 + 0.5) * dx,
                ];
                // Warm wood brown for sluice
                solids.push(ParticleInstance {
                    position: pos,
                    color: [0.55, 0.42, 0.28, 1.0],
                });
            }
        }
    }
    solids
}

// ============================================================================
// SHADER
// ============================================================================

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
    let size = 0.011;

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

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
