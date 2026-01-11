//! Tailings Pond Hybrid Simulation
//!
//! Demonstrates 3D particles (Wash Plant Tailings) interacting with a 2.5D Heightfield (Pond).
//!
//! Controls:
//! - WASD: Move
//! - SPACE/SHIFT: Up/Down
//! - Left Mouse: Spawn Tailings

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use game::gpu::bridge_3d::GpuBridge3D;
use game::gpu::flip_3d::GpuFlip3D;
use game::gpu::g2p_3d::DruckerPragerParams;
use game::gpu::heightfield::GpuHeightfield;
use sim3d::World;

const WORLD_WIDTH: usize = 256;
const WORLD_DEPTH: usize = 256;
const CELL_SIZE: f32 = 0.5;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    cell_size: f32,
    grid_width: u32,
    grid_depth: u32,
    time: f32,
    _pad: f32,
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.cos() * cos_pitch,
            self.pitch.sin(),
            self.yaw.sin() * cos_pitch,
        )
        .normalize()
    }

    fn forward_flat(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    fn right_flat(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }
}

struct InputState {
    keys: HashSet<KeyCode>,
    mouse_look: bool,
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pos: (f32, f32),
}

struct App {
    window: Option<Arc<Window>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,

    heightfield: Option<GpuHeightfield>,
    flip: Option<GpuFlip3D>,
    bridge: Option<GpuBridge3D>,

    world: World,
    camera: Camera,
    input: InputState,

    uniform_buffer: Option<wgpu::Buffer>,
    particle_render_pipeline: Option<wgpu::RenderPipeline>,
    particle_bind_group: Option<wgpu::BindGroup>,
    uniform_bind_group: Option<wgpu::BindGroup>,

    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,

    last_frame: Instant,
    start_time: Instant,
    active_particles: u32,
    cell_types: Vec<u32>,
    bed_heights: Vec<f32>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            config: None,
            heightfield: None,
            flip: None,
            bridge: None,
            world: build_world(),
            camera: Camera {
                position: Vec3::new(64.0, 50.0, 64.0),
                yaw: -1.57,
                pitch: -0.4,
                speed: 30.0,
                sensitivity: 0.003,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                last_mouse_pos: None,
                mouse_pos: (0.0, 0.0),
            },
            uniform_buffer: None,
            particle_render_pipeline: None,
            particle_bind_group: None,
            uniform_bind_group: None,
            depth_texture: None,
            depth_view: None,
            last_frame: Instant::now(),
            start_time: Instant::now(),
            active_particles: 0,
            cell_types: vec![0u32; WORLD_WIDTH * 64 * WORLD_DEPTH],
            bed_heights: vec![0.0f32; WORLD_WIDTH * WORLD_DEPTH],
        }
    }

    fn update_camera(&mut self, dt: f32) {
        let mut direction = Vec3::ZERO;

        if self.input.keys.contains(&KeyCode::KeyW) {
            direction += self.camera.forward_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyS) {
            direction -= self.camera.forward_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyA) {
            direction -= self.camera.right_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyD) {
            direction += self.camera.right_flat();
        }
        if self.input.keys.contains(&KeyCode::Space) {
            direction.y += 1.0;
        }
        if self.input.keys.contains(&KeyCode::ShiftLeft)
            || self.input.keys.contains(&KeyCode::ShiftRight)
        {
            direction.y -= 1.0;
        }

        if direction.length_squared() > 0.0 {
            self.camera.position += direction.normalize() * self.camera.speed * dt;
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if let (Some(device), Some(surface), Some(config)) =
            (&self.device, &self.surface, &mut self.config)
        {
            config.width = new_size.width.max(1);
            config.height = new_size.height.max(1);
            surface.configure(device, config);

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view =
                Some(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.depth_texture = Some(depth_texture);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.window = Some(window.clone());

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .unwrap();

            // Request high performance discrete GPU or integrated with enough features
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Simulation Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_storage_buffers_per_shader_stage: 16, // Need more for P2G
                            ..wgpu::Limits::default()
                        }
                        .using_resolution(adapter.limits()),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .unwrap();

            let size = window.inner_size();
            let config = surface
                .get_default_config(&adapter, size.width, size.height)
                .unwrap();
            surface.configure(&device, &config);

            // Initialize Simulation
            let mut flip = GpuFlip3D::new(
                &device,
                WORLD_WIDTH as u32,
                64,
                WORLD_DEPTH as u32,
                CELL_SIZE,
                200000,
            );
            flip.sediment_rest_particles = 8.0;

            let bridge = GpuBridge3D::new(&device, &flip, WORLD_WIDTH as u32, WORLD_DEPTH as u32);

            // Link bridge to heightfield
            let mut hf = GpuHeightfield::new(
                &device,
                WORLD_WIDTH as u32,
                WORLD_DEPTH as u32,
                CELL_SIZE,
                10.0,
                config.format,
            );

            // Link bridge to heightfield
            hf.set_bridge_buffers(
                &device,
                &bridge.transfer_sediment_buffer,
                &bridge.transfer_water_buffer,
            );

            // Setup Uniforms
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_bg_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Uniform Layout"),
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
            let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Uniform BG"),
                layout: &uniform_bg_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            // Setup Particle Rendering
            let particle_bg_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Particle Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let particle_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particle BG"),
                layout: &particle_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: flip.positions_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: flip.densities_buffer.as_entire_binding(),
                    },
                ],
            });

            let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Render"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../src/gpu/shaders/particle_3d.wgsl").into(),
                ),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&uniform_bg_layout, &particle_bg_layout],
                ..Default::default()
            });

            let particle_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Particle Render"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &particle_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &particle_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(config.format.into())],
                        compilation_options: Default::default(),
                    }),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

            // Upload initial world data!
            hf.upload_from_world(&queue, &self.world);

            // Set reasonable defaults for sediment physics
            flip.set_drucker_prager_params(&queue, DruckerPragerParams::default());

            self.device = Some(device);
            self.queue = Some(queue);
            self.surface = Some(surface);
            self.config = Some(config);
            self.heightfield = Some(hf);
            self.flip = Some(flip);
            self.bridge = Some(bridge);
            self.uniform_buffer = Some(uniform_buffer);
            self.uniform_bind_group = Some(uniform_bg);
            self.particle_bind_group = Some(particle_bg);
            self.particle_render_pipeline = Some(particle_pipeline);

            self.resize(size);
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size),
            WindowEvent::KeyboardInput {
                event: kb_input, ..
            } => {
                if let PhysicalKey::Code(code) = kb_input.physical_key {
                    if kb_input.state == ElementState::Pressed {
                        self.input.keys.insert(code);
                        if code == KeyCode::Escape {
                            event_loop.exit();
                        }
                    } else {
                        self.input.keys.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.input.mouse_look = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = (position.x, position.y);
                if self.input.mouse_look {
                    if let Some(last_pos) = self.input.last_mouse_pos {
                        let dx = (current_pos.0 - last_pos.0) as f32;
                        let dy = (current_pos.1 - last_pos.1) as f32;
                        self.camera.yaw += dx * self.camera.sensitivity;
                        self.camera.pitch -= dy * self.camera.sensitivity;
                        self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
                    }
                }
                self.input.last_mouse_pos = Some(current_pos);
                self.input.mouse_pos = (position.x as f32, position.y as f32);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.update_camera(dt);

                let device = self.device.as_ref().unwrap();
                let queue = self.queue.as_ref().unwrap();
                let surface = self.surface.as_ref().unwrap();
                let hf = self.heightfield.as_mut().unwrap();
                let flip = self.flip.as_mut().unwrap();
                let bridge = self.bridge.as_ref().unwrap();

                let sim_dt = 0.016;

                // 1. Simulation Steps
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                // Calculate view/proj matrices for raycasting
                let view_matrix = self.camera.view_matrix();
                let config = self.config.as_ref().unwrap();
                let proj_matrix = Mat4::perspective_rh(
                    0.8,
                    config.width as f32 / config.height as f32,
                    0.1,
                    2000.0,
                );

                // Raycast from mouse
                let (mouse_x, mouse_y) = self.input.mouse_pos;
                let size = self.window.as_ref().unwrap().inner_size();
                let ndc_x = (mouse_x / size.width as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (mouse_y / size.height as f32) * 2.0;

                let view_proj = proj_matrix * view_matrix;
                let inv_view_proj = view_proj.inverse();

                // Ray start (near plane) and end (far plane)
                let ray_start_clip = glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
                let ray_end_clip = glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

                let mut ray_start = inv_view_proj * ray_start_clip;
                ray_start /= ray_start.w;
                let mut ray_end = inv_view_proj * ray_end_clip;
                ray_end /= ray_end.w;

                let ray_origin = ray_start.truncate();
                let ray_dir = (ray_end.truncate() - ray_origin).normalize();

                // Raycast against terrain (simple steps)
                let mut hit_pos = ray_origin + ray_dir * 10.0; // Default
                let mut t = 0.0;
                let max_dist = 200.0;
                let step = 0.5;

                while t < max_dist {
                    let p = ray_origin + ray_dir * t;
                    if p.x >= 0.0
                        && p.x < WORLD_WIDTH as f32
                        && p.z >= 0.0
                        && p.z < WORLD_DEPTH as f32
                    {
                        let idx = (p.z as usize) * WORLD_WIDTH + (p.x as usize);
                        let terrain_h = self.world.bedrock_elevation[idx]
                            + self.world.overburden_thickness[idx]
                            + self.world.paydirt_thickness[idx];
                        if p.y < terrain_h {
                            hit_pos = p;
                            break;
                        }
                    } else if p.y < 0.0 {
                        // Hit base plane
                        hit_pos = p;
                        break;
                    }
                    t += step;
                }

                let mut spawned_this_frame = 0;

                if self.input.keys.contains(&KeyCode::Digit1) {
                    // Spawn Tailings (Brown) - continuous stream
                    let mut pos = hit_pos;
                    pos.y += 20.0; // Spawn 20m above hit
                    let vel = glam::Vec3::new(0.0, -5.0, 0.0);
                    bridge.dispatch_emitter(
                        queue,
                        &mut encoder,
                        pos.into(),
                        vel.into(),
                        0.1,
                        0.05,
                        5,
                        2.5, // Smaller radius, fewer particles per frame
                        self.start_time.elapsed().as_secs_f32(),
                    );
                    spawned_this_frame = 5;
                } else if self.input.keys.contains(&KeyCode::Digit2) {
                    // Spawn Water (Blue) - continuous stream
                    let mut pos = hit_pos;
                    pos.y += 20.0; // Spawn 20m above hit
                    let vel = glam::Vec3::new(0.0, -5.0, 0.0);
                    bridge.dispatch_emitter(
                        queue,
                        &mut encoder,
                        pos.into(),
                        vel.into(),
                        0.1,
                        0.05,
                        5,
                        1.0, // Smaller radius, fewer particles per frame
                        self.start_time.elapsed().as_secs_f32(),
                    );
                    spawned_this_frame = 5;
                }

                if spawned_this_frame > 0 {
                    self.active_particles = self
                        .active_particles
                        .saturating_add(spawned_this_frame)
                        .min(200000);
                }

                // Sync full terrain surface to flip (so particles bounce off ground)
                for i in 0..self.bed_heights.len() {
                    self.bed_heights[i] = self.world.bedrock_elevation[i]
                        + self.world.overburden_thickness[i]
                        + self.world.paydirt_thickness[i];
                }

                flip.step_in_place(
                    device,
                    queue,
                    self.active_particles,
                    &self.cell_types[..],
                    None,
                    Some(&self.bed_heights[..]),
                    sim_dt,
                    -9.8, // Gravity
                    0.0,  // Flow
                    40,   // Iterations
                );

                // Absorb onto Heightfield (smaller radius to prevent premature pooling)
                // let absorption_radius = 0.05;
                // bridge.dispatch_absorption(queue, &mut encoder, self.active_particles, WORLD_WIDTH as u32, WORLD_DEPTH as u32, CELL_SIZE, sim_dt, absorption_radius);

                // Merge Bridge into Heightfield
                hf.dispatch_bridge_merge(&mut encoder);
                bridge.clear_transfers(&mut encoder);

                // Step Heightfield (Erosion, Water)
                hf.dispatch_tile(&mut encoder, WORLD_WIDTH as u32, WORLD_DEPTH as u32);

                let frame = surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let depth_view = self.depth_view.as_ref().unwrap();

                // Update Uniforms (Camera)
                let view_matrix = self.camera.view_matrix();
                let config = self.config.as_ref().unwrap();
                let proj_matrix = Mat4::perspective_rh(
                    0.8,
                    config.width as f32 / config.height as f32,
                    0.1,
                    2000.0,
                );
                let view_proj = proj_matrix * view_matrix;

                queue.write_buffer(
                    self.uniform_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        view: view_matrix.to_cols_array_2d(),
                        proj: proj_matrix.to_cols_array_2d(),
                        camera_pos: self.camera.position.into(),
                        cell_size: CELL_SIZE * 2.0, // Double visual size for better observability
                        grid_width: WORLD_WIDTH as u32,
                        grid_depth: WORLD_DEPTH as u32,
                        time: self.start_time.elapsed().as_secs_f32(),
                        _pad: 0.0,
                    }),
                );

                // Render Heightfield
                hf.render(
                    &mut encoder,
                    &view,
                    &depth_view,
                    queue,
                    view_proj.to_cols_array_2d(),
                    self.camera.position.into(),
                    self.start_time.elapsed().as_secs_f32(),
                );

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Particles"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    // Render Particles
                    pass.set_pipeline(self.particle_render_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, self.uniform_bind_group.as_ref().unwrap(), &[]);
                    pass.set_bind_group(1, self.particle_bind_group.as_ref().unwrap(), &[]);
                    pass.draw(0..4, 0..self.active_particles);
                }

                queue.submit(std::iter::once(encoder.finish()));
                frame.present();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

fn build_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 10.0);

    let center_x = 64.0; // Center of FLIP volume

    for z in 0..WORLD_DEPTH {
        for x in 0..WORLD_WIDTH {
            let idx = world.idx(x, z);

            // General Slope South (increasing Z)
            let slope_drop = (z as f32 / WORLD_DEPTH as f32) * 20.0;
            let base_h = 20.0 - slope_drop;

            world.bedrock_elevation[idx] = base_h * 0.5;
            world.overburden_thickness[idx] = base_h * 0.2;
            world.paydirt_thickness[idx] = base_h * 0.3;

            // Cascading Ponds (centered in FLIP volume)
            let mut basin_depth = 0.0;

            // Pond 1 (Top)
            if z > 15 && z < 35 && x > 44 && x < 84 {
                let dx = ((x as f32 - center_x) / 20.0).powi(2);
                let dz = ((z as f32 - 25.0) / 10.0).powi(2);
                let d = dx + dz;
                if d < 1.0 {
                    basin_depth = 5.0 * (1.0 - d);
                }
            }

            // Pond 2 (Middle)
            if z > 45 && z < 65 && x > 44 && x < 84 {
                let dx = ((x as f32 - center_x) / 20.0).powi(2);
                let dz = ((z as f32 - 55.0) / 10.0).powi(2);
                let d = dx + dz;
                if d < 1.0 {
                    basin_depth = 5.0 * (1.0 - d);
                }
            }

            // Pond 3 (Bottom)
            if z > 75 && z < 95 && x > 44 && x < 84 {
                let dx = ((x as f32 - center_x) / 20.0).powi(2);
                let dz = ((z as f32 - 85.0) / 10.0).powi(2);
                let d = dx + dz;
                if d < 1.0 {
                    basin_depth = 5.0 * (1.0 - d);
                }
            }

            if basin_depth > 0.0 {
                let ob = world.overburden_thickness[idx];
                let dug_ob = basin_depth.min(ob);
                world.overburden_thickness[idx] -= dug_ob;
                let remaining = basin_depth - dug_ob;
                if remaining > 0.0 {
                    world.paydirt_thickness[idx] =
                        (world.paydirt_thickness[idx] - remaining).max(0.0);
                }
            }
        }
    }
    world
}
