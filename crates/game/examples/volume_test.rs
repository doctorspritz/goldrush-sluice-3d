//! Static Water Volume Test
//!
//! Simple open-top box with water. Monitors volume conservation.
//! No inlet, no outlet, no slope - just spawn water and watch.
//!
//! Run with: cargo run --example volume_test --release

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

// Simple open-top box - sized for ~8 particles per cell
// 100k particles / 8 = 12.5k cells target
// 20x30x20 = 12k cells (close to target)
const GRID_WIDTH: usize = 20;
const GRID_HEIGHT: usize = 30;  // Taller to see water rise/fall
const GRID_DEPTH: usize = 20;
const CELL_SIZE: f32 = 0.05;  // Larger cells for fewer total
const MAX_PARTICLES: usize = 150000;
const DT: f32 = 1.0 / 60.0;

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
    cell_types: Vec<u32>,
    // Mouse drag state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // FPS tracking
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
    // Volume tracking
    initial_particle_count: usize,
    avg_height_history: Vec<f32>,  // Track avg Y over time
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

/// Create a simple open-top box - 4 walls and floor, no ceiling
fn create_open_box(sim: &mut FlipSimulation3D) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                let is_floor = j == 0;
                let is_wall = i == 0 || i == width - 1 || k == 0 || k == depth - 1;

                if is_floor || is_wall {
                    sim.grid.set_solid(i, j, k);
                }
            }
        }
    }

    println!("Created open-top box: {}x{}x{}", width, height, depth);
}

/// Spawn water block at rest
fn spawn_water_block(sim: &mut FlipSimulation3D, count: usize) {
    let cell_size = sim.grid.cell_size;

    // Fill lower portion of box with particles
    let margin = 2.0;
    let spawn_width = (GRID_WIDTH as f32) - 2.0 * margin;
    let spawn_depth = (GRID_DEPTH as f32) - 2.0 * margin;
    let spawn_height = 15.0;  // Water fills lower 15 cells (proportional to smaller grid)

    let start_x = margin * cell_size;
    let start_y = 1.5 * cell_size;  // Just above floor
    let start_z = margin * cell_size;

    // Calculate spacing
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

                // Small jitter
                let jx = (rand_float() - 0.5) * 0.1 * spacing;
                let jy = (rand_float() - 0.5) * 0.1 * spacing;
                let jz = (rand_float() - 0.5) * 0.1 * spacing;

                sim.spawn_particle(Vec3::new(x + jx, y + jy, z + jz));
                spawned += 1;
            }
        }
    }

    println!("Spawned {} particles in static block", spawned);
}

fn rand_float() -> f32 {
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
        sim.pressure_iterations = 100;

        create_open_box(&mut sim);

        // Spawn initial water block
        let initial_count = 10000;  // Start small to verify shaders work
        spawn_water_block(&mut sim, initial_count);

        let solid_instances = Self::collect_solids(&sim);

        println!("\n=== STATIC WATER VOLUME TEST ===");
        println!("Watching for DEFLATION - water should maintain its height!");
        println!("Initial particles: {}", sim.particles.len());
        println!("Controls: SPACE=pause, R=reset, ESC=quit\n");

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
            cell_types: Vec::new(),
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            initial_particle_count: initial_count,
            avg_height_history: Vec::new(),
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
                        // Only render exposed faces
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
        // Reset random seed for reproducibility
        unsafe {
            static mut SEED: u32 = 12345;
            SEED = 12345;
        }

        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 100;

        create_open_box(&mut sim);
        spawn_water_block(&mut sim, self.initial_particle_count);

        self.sim = sim;
        self.frame = 0;
        self.avg_height_history.clear();
        println!("\n=== RESET ===");
    }

    fn compute_volume_stats(&self) -> (f32, f32, f32, f32) {
        // Compute avg height, max height, min height, and standard deviation
        let mut sum_y = 0.0f32;
        let mut max_y = 0.0f32;
        let mut min_y = f32::MAX;

        let count = self.sim.particles.len();
        if count == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        for p in &self.sim.particles.list {
            sum_y += p.position.y;
            max_y = max_y.max(p.position.y);
            min_y = min_y.min(p.position.y);
        }

        let avg_y = sum_y / count as f32;

        // Standard deviation
        let mut var_sum = 0.0f32;
        for p in &self.sim.particles.list {
            let diff = p.position.y - avg_y;
            var_sum += diff * diff;
        }
        let std_y = (var_sum / count as f32).sqrt();

        (avg_y, max_y, min_y, std_y)
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        // Update simulation
        if !self.paused {
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
                    DT,
                    -9.8,
                    0.0,  // No flow acceleration
                    pressure_iters,
                );
            }

            // Sync back and advect
            for (idx, p) in self.sim.particles.list.iter_mut().enumerate() {
                if idx < self.velocities.len() {
                    p.velocity = self.velocities[idx];
                    p.affine_velocity = self.c_matrices[idx];
                }
                // Advect from density-corrected position
                p.position = self.positions[idx] + p.velocity * DT;

                // Enforce boundaries (simple box)
                let cell_size = CELL_SIZE;
                let min_x = cell_size * 1.01;
                let max_x = (GRID_WIDTH as f32 - 1.01) * cell_size;
                let min_y = cell_size * 1.01;
                let max_y = (GRID_HEIGHT as f32 - 0.01) * cell_size;  // Open top
                let min_z = cell_size * 1.01;
                let max_z = (GRID_DEPTH as f32 - 1.01) * cell_size;

                if p.position.x < min_x { p.position.x = min_x; p.velocity.x = 0.0; }
                if p.position.x > max_x { p.position.x = max_x; p.velocity.x = 0.0; }
                if p.position.y < min_y { p.position.y = min_y; p.velocity.y = 0.0; }
                if p.position.y > max_y { p.position.y = max_y; p.velocity.y = 0.0; }
                if p.position.z < min_z { p.position.z = min_z; p.velocity.z = 0.0; }
                if p.position.z > max_z { p.position.z = max_z; p.velocity.z = 0.0; }
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

        // Print volume stats every 30 frames
        if self.frame % 30 == 0 {
            let (avg_y, max_y, min_y, std_y) = self.compute_volume_stats();
            self.avg_height_history.push(avg_y);

            // Compute drift from initial
            let initial_avg = self.avg_height_history.first().copied().unwrap_or(avg_y);
            let drift = avg_y - initial_avg;
            let drift_pct = if initial_avg > 0.0 { drift / initial_avg * 100.0 } else { 0.0 };

            println!(
                "Frame {:5} | Particles: {:5} | AvgY: {:.4} | MaxY: {:.4} | MinY: {:.4} | StdY: {:.4} | Drift: {:+.2}%",
                self.frame,
                self.sim.particles.len(),
                avg_y,
                max_y,
                min_y,
                std_y,
                drift_pct
            );

            // Alert if significant deflation
            if drift_pct < -5.0 {
                println!("  WARNING: Significant deflation detected! Water level dropping.");
            } else if drift_pct > 5.0 {
                println!("  INFO: Water level rising (density projection pushing up?)");
            }
        }

        // Create particle instances
        let instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .map(|p| {
                let speed = p.velocity.length();
                let t = (speed / 2.0).min(1.0);
                ParticleInstance {
                    position: [p.position.x, p.position.y, p.position.z],
                    color: [0.2 + t * 0.6, 0.5 - t * 0.3, 0.9 - t * 0.4, 0.8],
                }
            })
            .collect();

        // Render
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
                        .with_title("Static Water Volume Test")
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
