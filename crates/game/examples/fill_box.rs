//! Simple box filling test - GPU simulation
//!
//! A closed box with water inlet at top. Water should fill the box.
//! Target: 30 FPS
//!
//! Run with: cargo run --example fill_box --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Mat4, Vec3};
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

// Simple box parameters - narrow and tall for stacking test
const GRID_WIDTH: usize = 8;
const GRID_HEIGHT: usize = 64;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.05;
const MAX_PARTICLES: usize = 500_000;

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
    grid: sim3d::Grid3D,
    paused: bool,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    frame: u32,
    // Particle data
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    cell_types: Vec<u32>,
    // Mouse state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // FPS
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
    // RNG seed
    rand_seed: u32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn simple_rand(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) & 0x7FFF) as f32 / 32767.0
}

impl App {
    fn new() -> Self {
        let mut grid = sim3d::Grid3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);

        // Create closed box - all boundaries are solid
        for k in 0..GRID_DEPTH {
            for j in 0..GRID_HEIGHT {
                for i in 0..GRID_WIDTH {
                    let is_boundary = i == 0 || i == GRID_WIDTH - 1 ||
                                      j == 0 ||  // floor
                                      k == 0 || k == GRID_DEPTH - 1;
                    // Leave top open for inlet (j == GRID_HEIGHT - 1 is NOT solid)
                    if is_boundary {
                        grid.set_solid(i, j, k);
                    }
                }
            }
        }
        grid.compute_sdf();

        println!("Fill Box Test - GPU Simulation");
        println!("Grid: {}x{}x{}, cell_size={}", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        println!("Controls: SPACE=pause, R=reset, ESC=quit");

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            grid,
            paused: false,
            camera_angle: 0.5,
            camera_pitch: 0.3,
            camera_distance: 3.0,
            frame: 0,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            cell_types: Vec::new(),
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            rand_seed: 12345,
        }
    }

    fn emit_particles(&mut self, count: usize) {
        let center_x = GRID_WIDTH as f32 * CELL_SIZE * 0.5;
        let center_z = GRID_DEPTH as f32 * CELL_SIZE * 0.5;
        let emit_y = (GRID_HEIGHT - 2) as f32 * CELL_SIZE; // Near top

        for _ in 0..count {
            if self.positions.len() >= MAX_PARTICLES {
                break;
            }
            let r1 = simple_rand(&mut self.rand_seed);
            let r2 = simple_rand(&mut self.rand_seed);
            let r3 = simple_rand(&mut self.rand_seed);

            // Spawn in center area, spread across a few cells
            let x = center_x + (r1 - 0.5) * 4.0 * CELL_SIZE;
            let z = center_z + (r2 - 0.5) * 4.0 * CELL_SIZE;
            let y = emit_y + r3 * CELL_SIZE;

            self.positions.push(Vec3::new(x, y, z));
            self.velocities.push(Vec3::ZERO); // Just drop with gravity
            self.c_matrices.push(Mat3::ZERO);
        }
    }

    fn reset(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.frame = 0;
        self.rand_seed = 12345;
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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

        // Initialize GPU FLIP solver
        let gpu_flip = GpuFlip3D::new(
            &device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            instance_buffer,
            uniform_buffer,
            bind_group,
        });
        self.gpu_flip = Some(gpu_flip);
    }

    fn update_cell_types(&mut self) {
        self.cell_types.clear();
        for k in 0..GRID_DEPTH {
            for j in 0..GRID_HEIGHT {
                for i in 0..GRID_WIDTH {
                    let idx = self.grid.cell_index(i, j, k);
                    if self.grid.solid[idx] {
                        self.cell_types.push(2); // SOLID
                    } else {
                        self.cell_types.push(0); // AIR (GPU will mark FLUID based on particles)
                    }
                }
            }
        }
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() || self.gpu_flip.is_none() {
            return;
        }

        if !self.paused {
            let dt = 1.0 / 60.0;

            // Emit particles from top (stop at 100k to observe settling)
            const EMIT_LIMIT: usize = 100_000;
            if self.positions.len() < EMIT_LIMIT {
                let center_x = GRID_WIDTH as f32 * CELL_SIZE * 0.5;
                let center_z = GRID_DEPTH as f32 * CELL_SIZE * 0.5;
                let emit_y = (GRID_HEIGHT - 2) as f32 * CELL_SIZE;

                // Emit 50 particles per frame to fill faster
                for _ in 0..50 {
                    if self.positions.len() >= EMIT_LIMIT {
                        break;
                    }
                    let r1 = simple_rand(&mut self.rand_seed);
                    let r2 = simple_rand(&mut self.rand_seed);
                    let r3 = simple_rand(&mut self.rand_seed);

                    // Narrow emission area (2 cells wide)
                    let x = center_x + (r1 - 0.5) * 2.0 * CELL_SIZE;
                    let z = center_z + (r2 - 0.5) * 2.0 * CELL_SIZE;
                    let y = emit_y + r3 * CELL_SIZE;

                    self.positions.push(Vec3::new(x, y, z));
                    self.velocities.push(Vec3::ZERO);
                    self.c_matrices.push(Mat3::ZERO);
                }
            }

            // Update cell types
            self.cell_types.clear();
            self.cell_types.resize(GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH, 0); // AIR

            // Mark solid cells
            for k in 0..GRID_DEPTH {
                for j in 0..GRID_HEIGHT {
                    for i in 0..GRID_WIDTH {
                        let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                        if self.grid.solid[self.grid.cell_index(i, j, k)] {
                            self.cell_types[idx] = 2; // SOLID
                        }
                    }
                }
            }

            // Mark fluid cells based on particle presence
            for pos in &self.positions {
                let i = (pos.x / CELL_SIZE).floor() as i32;
                let j = (pos.y / CELL_SIZE).floor() as i32;
                let k = (pos.z / CELL_SIZE).floor() as i32;
                if i >= 0 && i < GRID_WIDTH as i32 &&
                   j >= 0 && j < GRID_HEIGHT as i32 &&
                   k >= 0 && k < GRID_DEPTH as i32 {
                    let idx = k as usize * GRID_WIDTH * GRID_HEIGHT + j as usize * GRID_WIDTH + i as usize;
                    if self.cell_types[idx] != 2 {
                        self.cell_types[idx] = 1; // FLUID
                    }
                }
            }

            // Run GPU simulation
            let gravity = Vec3::new(0.0, -9.8, 0.0);
            let gpu = self.gpu.as_ref().unwrap();
            let gpu_flip = self.gpu_flip.as_ref().unwrap();
            gpu_flip.step(
                &gpu.device,
                &gpu.queue,
                &self.positions,
                &mut self.velocities,
                &mut self.c_matrices,
                &self.cell_types,
                dt,
                gravity,
                50,
            );

            // Advect positions on CPU
            // NOTE: j=0 is SOLID (floor), so particles must stay in j>=1
            // Similarly, i=0 and k=0,k=depth-1 are solid walls
            let min_x = CELL_SIZE * 1.5;  // Center of cell i=1 (above solid i=0)
            let min_y = CELL_SIZE * 1.5;  // Center of cell j=1 (above solid floor j=0)
            let min_z = CELL_SIZE * 1.5;  // Center of cell k=1 (above solid k=0)
            let max_x = (GRID_WIDTH as f32 - 1.5) * CELL_SIZE;   // Inside solid i=width-1
            let max_y = (GRID_HEIGHT as f32 - 0.5) * CELL_SIZE;  // Top is open
            let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;   // Inside solid k=depth-1

            for (pos, vel) in self.positions.iter_mut().zip(self.velocities.iter()) {
                *pos += *vel * dt;
                if pos.x < min_x { pos.x = min_x; }
                if pos.x > max_x { pos.x = max_x; }
                if pos.y < min_y { pos.y = min_y; }
                if pos.y > max_y { pos.y = max_y; }
                if pos.z < min_z { pos.z = min_z; }
                if pos.z > max_z { pos.z = max_z; }
            }

            // ========== GPU PARTICLE SEPARATION (Houdini-style) ==========
            // Push particles apart when too close using GPU compute
            const MIN_DIST: f32 = CELL_SIZE * 0.5 * 1.8;  // Push apart if closer than this
            const PUSH_STRENGTH: f32 = 0.3;                // Relaxation strength
            const SEPARATION_ITERS: u32 = 2;

            gpu_flip.separate_particles(
                &gpu.device,
                &gpu.queue,
                &mut self.positions,
                MIN_DIST,
                PUSH_STRENGTH,
                SEPARATION_ITERS,
            );

            // Re-clamp positions to bounds after separation
            for pos in self.positions.iter_mut() {
                if pos.x < min_x { pos.x = min_x; }
                if pos.x > max_x { pos.x = max_x; }
                if pos.y < min_y { pos.y = min_y; }
                if pos.y > max_y { pos.y = max_y; }
                if pos.z < min_z { pos.z = min_z; }
                if pos.z > max_z { pos.z = max_z; }
            }

            self.frame += 1;
            self.fps_frame_count += 1;

            let now = Instant::now();
            let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
            if elapsed >= 1.0 {
                self.current_fps = self.fps_frame_count as f32 / elapsed;
                let max_y = self.positions.iter().map(|p| p.y).fold(0.0f32, f32::max);
                let min_y = self.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);

                // Compute pile height (10th percentile from TOP - where the settled water surface is)
                let mut y_sorted: Vec<f32> = self.positions.iter().map(|p| p.y).collect();
                y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let pile_height = if y_sorted.len() > 10 {
                    y_sorted[y_sorted.len() * 9 / 10] // 90th percentile - top of settled pile
                } else { max_y };

                // Count fluid cells
                let fluid_count = self.cell_types.iter().filter(|&&t| t == 1).count();

                // Velocity stats
                let avg_vel_y = if !self.velocities.is_empty() {
                    self.velocities.iter().map(|v| v.y).sum::<f32>() / self.velocities.len() as f32
                } else { 0.0 };
                let max_vel_y = self.velocities.iter().map(|v| v.y).fold(f32::MIN, f32::max);
                let min_vel_y = self.velocities.iter().map(|v| v.y).fold(f32::MAX, f32::min);

                // Debug: read pressure and divergence
                let gpu_flip2 = self.gpu_flip.as_ref().unwrap();
                let (pressure, divergence) = gpu_flip2.debug_read_pressure(&gpu.device, &gpu.queue);
                let nonzero_p: Vec<_> = pressure.iter().filter(|&&p| p.abs() > 1e-10).collect();
                let nonzero_d: Vec<_> = divergence.iter().filter(|&&d| d.abs() > 1e-10).collect();
                let max_p = pressure.iter().fold(0.0f32, |a, &b| a.max(b));
                let min_p = pressure.iter().fold(0.0f32, |a, &b| a.min(b));
                let max_d = divergence.iter().fold(0.0f32, |a, &b| a.max(b));
                let min_d = divergence.iter().fold(0.0f32, |a, &b| a.min(b));

                println!(
                    "Frame {:5} | FPS: {:5.1} | Particles: {:5} | PileHeight: {:.2} | Y: {:.2}-{:.2} | Fluids: {:4}",
                    self.frame, self.current_fps, self.positions.len(),
                    pile_height / CELL_SIZE,
                    min_y / CELL_SIZE, max_y / CELL_SIZE,
                    fluid_count
                );
                println!(
                    "           | P: {} nonzero ({:.4} to {:.4}) | D: {} nonzero ({:.4} to {:.4})",
                    nonzero_p.len(), min_p, max_p, nonzero_d.len(), min_d, max_d
                );
                self.fps_frame_count = 0;
                self.last_fps_time = now;
            }
        }

        // Camera
        let window = self.window.as_ref().unwrap();
        let gpu = self.gpu.as_ref().unwrap();

        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.5,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
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
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 50.0);
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create particle instances for rendering
        let water_instances: Vec<ParticleInstance> = self.positions.iter()
            .zip(self.velocities.iter())
            .take(MAX_PARTICLES)
            .map(|(pos, vel)| {
                let speed = vel.length();
                let t = (speed / 2.0).min(1.0);
                ParticleInstance {
                    position: pos.to_array(),
                    color: [0.2 + t * 0.3, 0.5 + t * 0.3, 0.9, 0.8],
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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));

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
            .with_title("Fill Box - GPU Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

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
                        PhysicalKey::Code(KeyCode::KeyR) => self.reset(),
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
                self.camera_distance = (self.camera_distance - scroll * 0.2).clamp(0.5, 10.0);
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
