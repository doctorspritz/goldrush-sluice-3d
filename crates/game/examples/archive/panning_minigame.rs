//! Gold panning minigame example (Phase 1 - basic CPU sim).

use std::time::Instant;
use std::{f32, sync::Arc};

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use game::panning::{PanInput, PanSample, PanSim};
use glam::{Mat3, Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.08,
    g: 0.12,
    b: 0.18,
    a: 1.0,
};

const MAX_INSTANCES: usize = 12000;
const PAN_RING_POINTS: usize = 64;
const PAN_RING_LAYERS: usize = 10;
const FLIP_GRID: u32 = 80;
const FLIP_CELL_SIZE: f32 = 0.01;
const MAX_FLIP_PARTICLES: usize = 20000;
const PRESSURE_ITERS: u32 = 30;

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
    size: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
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

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    gpu_flip: Option<GpuFlip3D>,
    sim: PanSim,
    input: PanInput,
    last_frame: Instant,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    key_left: bool,
    key_right: bool,
    key_up: bool,
    key_down: bool,
    mouse_pressed: bool,
    last_cursor_pos: Option<(f64, f64)>,
    cursor_pos: (f64, f64),
    instances: Vec<ParticleInstance>,
    grid_origin: Vec3,
    cell_types_base: Vec<u32>,
    cell_types: Vec<u32>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim: PanSim::new(PanSample::tutorial()),
            input: PanInput::default(),
            last_frame: Instant::now(),
            camera_angle: 0.6,
            camera_pitch: 0.35,
            camera_distance: 1.2,
            key_left: false,
            key_right: false,
            key_up: false,
            key_down: false,
            mouse_pressed: false,
            last_cursor_pos: None,
            cursor_pos: (0.0, 0.0),
            instances: Vec::with_capacity(MAX_INSTANCES),
            grid_origin: Vec3::ZERO,
            cell_types_base: Vec::new(),
            cell_types: Vec::new(),
            positions: Vec::with_capacity(MAX_FLIP_PARTICLES),
            velocities: Vec::with_capacity(MAX_FLIP_PARTICLES),
            c_matrices: Vec::with_capacity(MAX_FLIP_PARTICLES),
            densities: Vec::with_capacity(MAX_FLIP_PARTICLES),
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

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
                label: Some("Panning Device"),
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
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Panning Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let vertices = [
            Vertex {
                position: [-1.0, -1.0],
            },
            Vertex {
                position: [1.0, -1.0],
            },
            Vertex {
                position: [-1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Panning Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Panning Instance Buffer"),
            size: (MAX_INSTANCES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Panning Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Panning Bind Group Layout"),
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
            label: Some("Panning Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Panning Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Panning Pipeline"),
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
                            wgpu::VertexAttribute {
                                offset: 28,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
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

        let domain = FLIP_GRID as f32 * FLIP_CELL_SIZE;
        self.grid_origin = Vec3::new(
            self.sim.pan_center.x - domain * 0.5,
            self.sim.pan_center.y - 0.05,
            self.sim.pan_center.z - domain * 0.5,
        );
        self.cell_types_base = Self::build_cell_types_base(
            self.grid_origin,
            self.sim.pan_center,
            self.sim.pan_radius,
            self.sim.pan_depth,
        );
        self.cell_types = self.cell_types_base.clone();

        self.gpu_flip = Some(GpuFlip3D::new(
            &device,
            FLIP_GRID,
            FLIP_GRID,
            FLIP_GRID,
            FLIP_CELL_SIZE,
            MAX_FLIP_PARTICLES,
        ));

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
        self.window = Some(window);
        self.last_frame = Instant::now();
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if let Some(gpu) = &mut self.gpu {
            gpu.config.width = size.width.max(1);
            gpu.config.height = size.height.max(1);
            gpu.surface.configure(&gpu.device, &gpu.config);
        }
    }

    fn update_time_and_camera(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.033);
        self.last_frame = now;

        let rotate_speed = 1.2;
        if self.key_left {
            self.camera_angle += rotate_speed * dt;
        }
        if self.key_right {
            self.camera_angle -= rotate_speed * dt;
        }
        if self.key_up {
            self.camera_pitch = (self.camera_pitch + rotate_speed * 0.6 * dt).min(1.1);
        }
        if self.key_down {
            self.camera_pitch = (self.camera_pitch - rotate_speed * 0.6 * dt).max(-0.2);
        }

        dt
    }

    fn step_simulation(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.sim.update_controls_only(&self.input, dt);

        if self.input.dump {
            self.sim.particles.clear();
        }

        if self.sim.particles.len() > MAX_FLIP_PARTICLES {
            self.sim.particles.truncate(MAX_FLIP_PARTICLES);
        }

        self.positions.clear();
        self.velocities.clear();
        self.c_matrices.clear();
        self.densities.clear();

        let tilt_accel = Vec3::new(
            9.81 * self.sim.current_tilt.x.sin(),
            0.0,
            9.81 * self.sim.current_tilt.y.sin(),
        );
        let swirl = self.sim.current_swirl;

        for particle in &self.sim.particles {
            if self.positions.len() >= MAX_FLIP_PARTICLES {
                break;
            }
            let mut velocity = particle.velocity;
            velocity.x += tilt_accel.x * dt;
            velocity.z += tilt_accel.z * dt;

            if swirl > 0.1 {
                let to_center = particle.position - self.sim.pan_center;
                let r = Vec2::new(to_center.x, to_center.z).length();
                if r > 0.0001 {
                    let omega = swirl * std::f32::consts::TAU / 60.0;
                    let tangent = Vec3::new(-to_center.z / r, 0.0, to_center.x / r);
                    let target = tangent * omega * r;
                    let blend = 1.0 - (-6.0 * dt).exp();
                    velocity.x += (target.x - velocity.x) * blend;
                    velocity.z += (target.z - velocity.z) * blend;
                }
            }

            self.positions.push(particle.position);
            self.velocities.push(velocity);
            self.c_matrices.push(Mat3::ZERO);
            self.densities.push(particle.specific_gravity());
        }

        if self.positions.is_empty() {
            self.input.add_water = false;
            self.input.shake = false;
            self.input.dump = false;
            return;
        }

        if self.cell_types.len() != self.cell_types_base.len() {
            self.cell_types.resize(self.cell_types_base.len(), 0);
        }
        self.cell_types.copy_from_slice(&self.cell_types_base);

        let width = FLIP_GRID as i32;
        let height = FLIP_GRID as i32;
        let depth = FLIP_GRID as i32;

        for pos in &self.positions {
            let local = (*pos - self.grid_origin) / FLIP_CELL_SIZE;
            let i = local.x.floor() as i32;
            let j = local.y.floor() as i32;
            let k = local.z.floor() as i32;
            if i < 0 || j < 0 || k < 0 || i >= width || j >= height || k >= depth {
                continue;
            }
            let idx = (k as usize) * (width as usize) * (height as usize)
                + (j as usize) * (width as usize)
                + (i as usize);
            if self.cell_types[idx] != 2 {
                self.cell_types[idx] = 1;
            }
        }

        if let Some(flip) = &mut self.gpu_flip {
            flip.step(
                device,
                queue,
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
                PRESSURE_ITERS,
            );
        }

        for (pos, vel) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *pos += *vel * dt;
        }

        let center = self.sim.pan_center;
        let radius = self.sim.pan_radius;
        let depth = self.sim.pan_depth;
        Self::apply_bowl_collision(
            center,
            radius,
            depth,
            &mut self.positions,
            &mut self.velocities,
        );

        for (pos, vel) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
            let to_center = *pos - center;
            let r = Vec2::new(to_center.x, to_center.z).length();
            if r > radius * 1.2 {
                vel.x *= 0.6;
                vel.z *= 0.6;
            }
        }

        for (i, particle) in self.sim.particles.iter_mut().enumerate() {
            if i >= self.positions.len() {
                break;
            }
            particle.position = self.positions[i];
            particle.velocity = self.velocities[i];
        }

        self.sim.cull_overflow();
        self.input.add_water = false;
        self.input.shake = false;
        self.input.dump = false;
    }

    fn apply_bowl_collision(
        center: Vec3,
        radius: f32,
        depth: f32,
        positions: &mut [Vec3],
        velocities: &mut [Vec3],
    ) {
        for (pos, vel) in positions.iter_mut().zip(velocities.iter_mut()) {
            let to_center = *pos - center;
            let r = Vec2::new(to_center.x, to_center.z).length();
            if r > radius {
                continue;
            }
            let t = (r / radius).clamp(0.0, 1.0);
            let bowl_y = center.y + depth * t * t;
            if pos.y < bowl_y {
                pos.y = bowl_y;
                if vel.y < 0.0 {
                    vel.y = 0.0;
                }
                vel.x *= 0.85;
                vel.z *= 0.85;
            }
        }
    }

    fn build_cell_types_base(
        grid_origin: Vec3,
        pan_center: Vec3,
        pan_radius: f32,
        pan_depth: f32,
    ) -> Vec<u32> {
        let cell_count = (FLIP_GRID * FLIP_GRID * FLIP_GRID) as usize;
        let mut cells = vec![0u32; cell_count];

        let width = FLIP_GRID as usize;
        let height = FLIP_GRID as usize;
        let depth = FLIP_GRID as usize;

        for k in 0..depth {
            let z = grid_origin.z + (k as f32 + 0.5) * FLIP_CELL_SIZE;
            for j in 0..height {
                let y = grid_origin.y + (j as f32 + 0.5) * FLIP_CELL_SIZE;
                for i in 0..width {
                    let x = grid_origin.x + (i as f32 + 0.5) * FLIP_CELL_SIZE;
                    let dx = x - pan_center.x;
                    let dz = z - pan_center.z;
                    let r = Vec2::new(dx, dz).length();
                    if r <= pan_radius {
                        let t = (r / pan_radius).clamp(0.0, 1.0);
                        let bowl_y = pan_center.y + pan_depth * t * t;
                        if y < bowl_y {
                            let idx = k * width * height + j * width + i;
                            cells[idx] = 2;
                        }
                    }
                }
            }
        }

        cells
    }

    fn build_instances(&mut self) -> usize {
        self.instances.clear();

        let rotation = Mat4::from_rotation_z(self.sim.current_tilt.x)
            * Mat4::from_rotation_x(-self.sim.current_tilt.y);
        let pan_center = self.sim.pan_center;
        let pivot = self.pan_pivot();

        let mut push_pan_point = |pos: Vec3, size: f32, color: [f32; 4]| {
            if self.instances.len() >= MAX_INSTANCES {
                return;
            }
            let local = pos - pivot;
            let rotated = rotation.transform_point3(local) + pivot;
            self.instances.push(ParticleInstance {
                position: [rotated.x, rotated.y, rotated.z],
                color,
                size,
            });
        };

        let ring_color = [0.45, 0.45, 0.45, 0.9];
        for layer in 0..=PAN_RING_LAYERS {
            let t = layer as f32 / PAN_RING_LAYERS as f32;
            let radius = self.sim.pan_radius * t;
            let height = self.sim.pan_center.y + self.sim.pan_depth * t * t;
            let size = 0.006 + (1.0 - t) * 0.002;

            if layer == 0 {
                push_pan_point(
                    Vec3::new(self.sim.pan_center.x, height, self.sim.pan_center.z),
                    size,
                    ring_color,
                );
                continue;
            }

            let point_count = ((PAN_RING_POINTS as f32) * t).round().max(12.0) as usize;
            for i in 0..point_count {
                let angle = (i as f32 / point_count as f32) * f32::consts::TAU;
                let pos = Vec3::new(
                    pan_center.x + angle.cos() * radius,
                    height,
                    pan_center.z + angle.sin() * radius,
                );
                push_pan_point(pos, size, ring_color);
            }
        }

        for particle in &self.sim.particles {
            if self.instances.len() >= MAX_INSTANCES {
                break;
            }
            let rgb = particle.color();
            let size = (particle.diameter * 1.6).max(0.003);
            let local = particle.position - pivot;
            let rotated = rotation.transform_point3(local) + pivot;
            self.instances.push(ParticleInstance {
                position: [rotated.x, rotated.y, rotated.z],
                color: [rgb[0], rgb[1], rgb[2], 1.0],
                size,
            });
        }

        self.instances.len()
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let window = match &self.window {
            Some(window) => Arc::clone(window),
            None => return,
        };

        let dt = self.update_time_and_camera();
        let mut gpu = match self.gpu.take() {
            Some(gpu) => gpu,
            None => return,
        };
        self.step_simulation(&gpu.device, &gpu.queue, dt);
        let instance_count = self.build_instances();

        let frame = match gpu.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                self.resize(window.inner_size());
                self.gpu = Some(gpu);
                return;
            }
            Err(SurfaceError::OutOfMemory) => {
                event_loop.exit();
                self.gpu = Some(gpu);
                return;
            }
            Err(SurfaceError::Timeout) => {
                self.gpu = Some(gpu);
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let target = self.pan_pivot();
        let offset = Vec3::new(
            self.camera_distance * self.camera_pitch.cos() * self.camera_angle.cos(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * self.camera_pitch.cos() * self.camera_angle.sin(),
        );
        let camera_pos = target + offset + Vec3::new(0.0, 0.18, 0.0);
        let view_mat = Mat4::look_at_rh(camera_pos, target, Vec3::Y);
        let proj_mat = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 10.0);
        let view_proj = proj_mat * view_mat;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z],
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        gpu.queue.write_buffer(
            &gpu.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instances[..instance_count]),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Panning Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Panning Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
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
            pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
            pass.draw(0..4, 0..instance_count as u32);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        window.request_redraw();
        self.gpu = Some(gpu);
    }

    fn pan_pivot(&self) -> Vec3 {
        let pan_center = self.sim.pan_center;
        let rim_height = pan_center.y + self.sim.pan_depth;
        let tilt_dir = Vec2::new(self.sim.current_tilt.x, self.sim.current_tilt.y);
        let pivot_offset = if tilt_dir.length_squared() > 1.0e-6 {
            -tilt_dir.normalize() * self.sim.pan_radius
        } else {
            Vec2::ZERO
        };
        Vec3::new(
            pan_center.x + pivot_offset.x,
            rim_height,
            pan_center.z + pivot_offset.y,
        )
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
                        .with_title("Gold Panning - Phase 1")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1100, 720)),
                )
                .expect("Failed to create window"),
        );

        self.init_gpu(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size),
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.input.swirl_rpm = (self.input.swirl_rpm + scroll * 8.0).clamp(0.0, 120.0);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if self.mouse_pressed {
                        self.last_cursor_pos = Some(self.cursor_pos);
                    } else {
                        self.last_cursor_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x, position.y);
                if self.mouse_pressed {
                    if let Some((last_x, last_y)) = self.last_cursor_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.input.tilt.x += dx * 0.005;
                        self.input.tilt.y -= dy * 0.005;
                        self.input.clamp_tilt();
                    }
                    self.last_cursor_pos = Some(self.cursor_pos);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        if pressed {
                            event_loop.exit();
                        }
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.key_left = pressed,
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.key_right = pressed,
                    PhysicalKey::Code(KeyCode::ArrowUp) => self.key_up = pressed,
                    PhysicalKey::Code(KeyCode::ArrowDown) => self.key_down = pressed,
                    PhysicalKey::Code(KeyCode::Space) => {
                        if pressed {
                            self.input.add_water = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        if pressed {
                            self.input.shake = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        if pressed {
                            self.input.dump = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if pressed {
                            self.sim = PanSim::new(PanSample::tutorial());
                            self.input = PanInput::default();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => self.render(event_loop),
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
    @location(3) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let to_camera = normalize(uniforms.camera_pos - in.position);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, to_camera));
    let up = cross(to_camera, right);

    let world_pos = in.position + right * in.vertex.x * in.size + up * in.vertex.y * in.size;

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
