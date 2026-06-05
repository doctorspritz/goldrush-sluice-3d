//! World Simulation - Heightfield terrain with water and erosion
//!
//! A clean heightfield-based simulation without FLIP particle grids.
//! Uses GPU-accelerated shallow water equations for fluid flow.
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look around
//! - Left Click: Excavate terrain
//! - Ctrl + Left Click: Add material
//! - G/O/T: Select material (Gravel/Overburden/sediment)
//! - H: Toggle water visibility
//! - V: Toggle velocity coloring (TODO: not yet implemented in heightfield shader)
//! - E: Toggle emitter on/off
//! - 3: Place and toggle emitter at cursor
//! - Up/Down arrows: Increase/decrease emitter rate
//! - 1: Add water at cursor
//! - 2: Add muddy water at cursor
//! - R: Reset world
//! - ESC: Quit
//!
//! Run: cargo run --example world_sim --release

use bytemuck::{Pod, Zeroable};
use game::gpu::heightfield::GpuHeightfield;
use game::water_emitter::WaterEmitter;
use glam::{Mat4, Vec3};
use sim3d::World;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// World dimensions
const WORLD_WIDTH: usize = 512;
const WORLD_DEPTH: usize = 512;
const CELL_SIZE: f32 = 1.0;
const INITIAL_HEIGHT: f32 = 10.0;

// Interaction
const DIG_RADIUS: f32 = 3.0;
const DIG_DEPTH: f32 = 0.5;
const ADD_RADIUS: f32 = 3.0;
const ADD_HEIGHT: f32 = 0.5;
const WATER_ADD_VOLUME: f32 = 5.0;

// Camera
const MOVE_SPEED: f32 = 20.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

// Simulation
const STEPS_PER_FRAME: usize = 10;
const DT: f32 = 0.02;
const DEBUG_HEIGHTFIELD_STATS: bool = true;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct WorldVertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad0: f32,
}

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl Mesh {
    fn new(device: &wgpu::Device, vertices: &[WorldVertex], indices: &[u32], label: &str) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertices", label)),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Indices", label)),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }

    fn update(&self, queue: &wgpu::Queue, vertices: &[WorldVertex], indices: &[u32]) {
        if !vertices.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        }
        if !indices.is_empty() {
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
        }
    }
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
    left_mouse: bool,
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pos: (f32, f32),
    scroll_delta: f32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    heightfield: GpuHeightfield,
}

/// On-screen egui HUD layer: owns the egui context, the winit input bridge,
/// and the wgpu renderer (pipeline + texture atlas). Created in `init_gpu`
/// once device/queue/config/window exist.
struct EguiLayer {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

/// Plain data snapshot filled from `App` fields each frame, then handed to
/// the HUD builder. Keeps the egui closure free of borrows on `App`.
struct HudState {
    mode: String,
    material: String,
    tool: String,
    water_on: bool,
    emitter_on: bool,
    fps: f32,
}

/// Build the on-screen HUD: a top-left status panel (mode / material / tool /
/// water / emitter / FPS) plus a top-right help panel. Each status line and the
/// help entries carry hover tooltips describing the relevant controls.
fn build_hud_ui(ctx: &egui::Context, hud: &HudState) {
    egui::Area::new(egui::Id::new("hud_top_left"))
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(
                    egui::RichText::new(format!("Mode: {}", hud.mode))
                        .strong()
                        .size(16.0),
                )
                .on_hover_text(
                    "World = free-fly navigation.\nEditing = left mouse held (terrain tool active).",
                );
                ui.label(format!("Material: {}", hud.material)).on_hover_text(
                    "Active material for Add/Excavate.\nG = Gravel, O = Overburden, T = Sediment",
                );
                ui.label(format!("Tool: {}", hud.tool)).on_hover_text(
                    "Excavate by default.\nHold Ctrl + Left-click to Add material instead.",
                );
                ui.label(format!(
                    "Water: {}   Emitter: {}",
                    if hud.water_on { "ON" } else { "OFF" },
                    if hud.emitter_on { "ON" } else { "OFF" },
                ))
                .on_hover_text("H toggles water visibility.\nE toggles the emitter on/off.");
                ui.label(
                    egui::RichText::new(format!("FPS: {:.0}", hud.fps))
                        .monospace(),
                )
                .on_hover_text("Frames per second (smoothed).");
            });
        });

    egui::Area::new(egui::Id::new("hud_top_right"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                let help = ui.button("? Controls");
                help.on_hover_text(
                    "WASD: move camera\n\
                     Space / Shift: up / down\n\
                     Right-drag: look around\n\
                     Left-click: excavate terrain\n\
                     Ctrl + Left-click: add material\n\
                     G / O / T: select material (Gravel / Overburden / Sediment)\n\
                     H: toggle water    V: toggle velocity coloring\n\
                     E: toggle emitter    3: place emitter at cursor\n\
                     1: add water    2: add muddy water\n\
                     Up / Down: emitter rate\n\
                     R: reset world    Esc: quit",
                );
                ui.small("hover for controls");
            });
        });
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    world: World,
    emitter: WaterEmitter,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    last_stats: Instant,
    start_time: Instant,
    window_size: (u32, u32),
    terrain_dirty: bool,
    show_water: bool,
    show_velocity: bool,
    selected_material: u32, // 0=sediment, 1=overburden, 2=gravel
    emitter_mesh: Option<Mesh>,
    egui: Option<EguiLayer>,
    fps: f32,
}

impl App {
    fn new() -> Self {
        let world = build_world();

        let center_x = WORLD_WIDTH / 2;
        // Emitter at upstream end (z=10, high elevation)
        let emitter_z = 10;
        let emitter_y = world.ground_height(center_x, emitter_z);

        Self {
            window: None,
            gpu: None,
            world,
            emitter: {
                let mut e = WaterEmitter::new(
                    Vec3::new(center_x as f32, emitter_y + 5.0, emitter_z as f32),
                    100.0, // rate
                    3.0,   // radius
                );
                // Clean water - the river picks up sediment as it erodes
                e.sediment_conc = 0.02;
                e.overburden_conc = 0.01;
                e.gravel_conc = 0.005;
                e.paydirt_conc = 0.001;
                e.enabled = true;
                e
            },
            camera: Camera {
                // Camera looking downstream from above the emitter
                position: Vec3::new(center_x as f32 - 30.0, emitter_y + 40.0, emitter_z as f32 + 20.0),
                yaw: 0.3,   // Looking slightly right and down the valley
                pitch: -0.5,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                left_mouse: false,
                last_mouse_pos: None,
                mouse_pos: (0.0, 0.0),
                scroll_delta: 0.0,
            },
            last_frame: Instant::now(),
            last_stats: Instant::now(),
            start_time: Instant::now(),
            window_size: (1280, 720),
            terrain_dirty: true,
            show_water: true,
            show_velocity: false,
            selected_material: 2, // gravel by default
            emitter_mesh: None,
            egui: None,
            fps: 0.0,
        }
    }

    fn reset_world(&mut self) {
        self.world = build_world();
        self.terrain_dirty = true;
    }

    fn projection_matrix(&self) -> Mat4 {
        let aspect = self.window_size.0 as f32 / self.window_size.1 as f32;
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0)
    }

    fn raycast_terrain(&self) -> Option<Vec3> {
        let (mx, my) = self.input.mouse_pos;
        let (w, h) = self.window_size;
        let ndc_x = (2.0 * mx / w as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * my / h as f32);

        let proj = self.projection_matrix();
        let view = self.camera.view_matrix();
        let inv_vp = (proj * view).inverse();

        let near = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, -1.0));
        let far = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize();

        // March along ray
        let mut t = 0.0;
        let max_t = 500.0;
        let step = 0.5;

        while t < max_t {
            let p = self.camera.position + dir * t;
            if p.x >= 0.0
                && p.x < WORLD_WIDTH as f32
                && p.z >= 0.0
                && p.z < WORLD_DEPTH as f32
            {
                let x = p.x as usize;
                let z = p.z as usize;
                let ground = self.world.ground_height(x, z);
                let idx = self.world.idx(x, z);
                let water = self.world.water_surface[idx];
                let surface = ground.max(water);
                if p.y <= surface {
                    return Some(p);
                }
            }
            t += step;
        }
        None
    }

    fn update(&mut self, dt: f32) {
        self.update_camera(dt);

        // Handle terrain modification (left click)
        if self.input.left_mouse {
            let is_adding = self.input.keys.contains(&KeyCode::ControlLeft)
                || self.input.keys.contains(&KeyCode::ControlRight);
            self.modify_terrain_at_cursor(is_adding, dt);
        }

        // Run simulation steps
        if let Some(gpu) = &mut self.gpu {
            for _ in 0..STEPS_PER_FRAME {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sim Encoder"),
                        });

                // Update emitter and dispatch
                self.emitter.update_gpu(&gpu.heightfield, &gpu.queue, DT);
                gpu.heightfield.dispatch_emitter(&mut encoder);

                // Update water sim params and dispatch
                gpu.heightfield.update_params(&gpu.queue, DT);
                gpu.heightfield.dispatch(&mut encoder, DT);

                gpu.queue.submit(Some(encoder.finish()));
            }

            // Sync back to CPU
            pollster::block_on(gpu.heightfield.download_to_world(
                &gpu.device,
                &gpu.queue,
                &mut self.world,
            ));
        }

        // Print stats periodically
        if self.last_stats.elapsed() > Duration::from_secs(2) {
            if let Some(gpu) = &self.gpu {
                if DEBUG_HEIGHTFIELD_STATS {
                    println!("Heightfield debug stats are unavailable in the current build.");
                }
            }
            self.last_stats = Instant::now();
        }

        // Update emitter mesh
        if self.emitter.enabled {
            if let Some(gpu) = self.gpu.as_ref() {
                let (positions, indices) = self.emitter.visualize(16);
                let vertices: Vec<WorldVertex> = positions
                    .iter()
                    .map(|p| WorldVertex {
                        position: p.to_array(),
                        color: [0.0, 1.0, 1.0, 1.0],
                    })
                    .collect();

                if let Some(mesh) = self.emitter_mesh.as_ref() {
                    mesh.update(&gpu.queue, &vertices, &indices);
                } else {
                    self.emitter_mesh = Some(Mesh::new(
                        &gpu.device,
                        &vertices,
                        &indices,
                        "Emitter Mesh",
                    ));
                }
            }
        } else {
            self.emitter_mesh = None;
        }
    }

    fn add_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world.add_water(hit, WATER_ADD_VOLUME);
            if let Some(gpu) = &self.gpu {
                gpu.heightfield.upload_from_world(&gpu.queue, &self.world);
            }
        }
    }

    fn add_muddy_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world
                .add_sediment_water(hit, WATER_ADD_VOLUME, WATER_ADD_VOLUME * 0.1);
            if let Some(gpu) = &self.gpu {
                gpu.heightfield.upload_from_world(&gpu.queue, &self.world);
            }
        }
    }

    fn modify_terrain_at_cursor(&mut self, is_adding: bool, dt: f32) {
        if let Some(hit) = self.raycast_terrain() {
            let radius = if is_adding { ADD_RADIUS } else { DIG_RADIUS };
            let amount = if is_adding { ADD_HEIGHT * 50.0 } else { -DIG_DEPTH * 50.0 };

            if let Some(gpu) = &self.gpu {
                gpu.heightfield.update_material_tool(
                    &gpu.queue,
                    hit.x,
                    hit.z,
                    radius,
                    amount,
                    self.selected_material,
                    dt,
                    true,
                );

                let mut encoder = gpu.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Terrain Modify") }
                );

                if is_adding {
                    gpu.heightfield.dispatch_material_tool(&mut encoder);
                } else {
                    gpu.heightfield.dispatch_excavate(&mut encoder);
                }

                gpu.queue.submit(Some(encoder.finish()));
            }
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

        // Scroll zoom
        if self.input.scroll_delta != 0.0 {
            let forward = self.camera.forward();
            self.camera.position += forward * self.input.scroll_delta * 2.0;
            self.input.scroll_delta = 0.0;
        }

        // Clamp to world bounds
        let world_size = self.world.world_size();
        self.camera.position.x = self.camera.position.x.clamp(0.0, world_size.x);
        self.camera.position.z = self.camera.position.z.clamp(0.0, world_size.z);
        self.camera.position.y = self.camera.position.y.clamp(2.0, world_size.y + 100.0);
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // GpuHeightfield needs higher storage buffer limits
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
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<WorldVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // GPU Heightfield
        let mut heightfield = GpuHeightfield::new(
            &device,
            self.world.width as u32,
            self.world.depth as u32,
            self.world.cell_size,
            INITIAL_HEIGHT,
            config.format,
        );
        if DEBUG_HEIGHTFIELD_STATS {
            println!("Heightfield debug stats are unavailable in the current build.");
        }
        heightfield.upload_from_world(&queue, &self.world);

        // --- egui HUD layer ---
        // Created AFTER device/queue/config exist. Uses the exact surface
        // `config.format` (caps.formats[0]) so the HUD pipeline matches the
        // swapchain. No depth (HUD is a 2D overlay), 1 msaa sample, no dither.
        if let Some(win) = self.window.as_ref() {
            let ctx = egui::Context::default();
            let viewport_id = ctx.viewport_id();
            let egui_state = egui_winit::State::new(
                ctx.clone(),
                viewport_id,
                win.as_ref(),
                Some(win.scale_factor() as f32),
                None,
                None,
            );
            let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);
            self.egui = Some(EguiLayer {
                ctx,
                state: egui_state,
                renderer: egui_renderer,
            });
        }

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            bind_group,
            depth_texture,
            depth_view,
            heightfield,
        });
    }

    fn render(&mut self) {
        let window = match self.window.as_ref() {
            Some(w) => w.clone(),
            None => return,
        };

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.update(dt);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            _pad0: 0.0,
        };

        // Smoothed FPS (exponential moving average over instantaneous frame dt).
        if dt > 0.0 {
            let inst = 1.0 / dt;
            self.fps = if self.fps <= 0.0 {
                inst
            } else {
                self.fps * 0.9 + inst * 0.1
            };
        }

        // Snapshot HUD values BEFORE borrowing gpu/egui (avoids overlapping
        // borrows of `self` inside the egui closure).
        let hud = HudState {
            mode: if self.input.left_mouse { "Sluice/Edit" } else { "World" }.into(),
            material: match self.selected_material {
                2 => "Gravel",
                1 => "Overburden",
                _ => "Sediment",
            }
            .into(),
            tool: if self.input.keys.contains(&KeyCode::ControlLeft)
                || self.input.keys.contains(&KeyCode::ControlRight)
            {
                "Add"
            } else {
                "Excavate"
            }
            .into(),
            water_on: self.show_water,
            emitter_on: self.emitter.enabled,
            fps: self.fps,
        };

        // Split the borrows: take egui out by reference disjointly from gpu.
        let egui = self.egui.as_mut();
        let Some(gpu) = self.gpu.as_mut() else { return };

        // Update uniforms
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let frame_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render heightfield (terrain + water)
        gpu.heightfield.render(
            &mut encoder,
            &frame_view,
            &gpu.depth_view,
            &gpu.queue,
            view_proj.to_cols_array_2d(),
            self.camera.position.to_array(),
            self.start_time.elapsed().as_secs_f32(),
            self.show_water,
            self.show_velocity,
        );

        // Render emitter sphere
        if let Some(mesh) = self.emitter_mesh.as_ref() {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Emitter Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&gpu.pipeline);
            rpass.set_bind_group(0, &gpu.bind_group, &[]);
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);
        }

        // --- egui HUD pass: drawn AFTER the scene, into the SAME frame_view ---
        // LoadOp::Load preserves the terrain/water/emitter already rendered.
        // No depth attachment: the HUD is a 2D overlay.
        let mut egui_cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
        if let Some(egui) = egui {
            let raw_input = egui.state.take_egui_input(window.as_ref());
            let full_output = egui.ctx.run(raw_input, |ctx| build_hud_ui(ctx, &hud));
            egui.state
                .handle_platform_output(window.as_ref(), full_output.platform_output);

            let tris = egui
                .ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);

            for (id, delta) in &full_output.textures_delta.set {
                egui.renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
            }

            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [gpu.config.width, gpu.config.height],
                pixels_per_point: full_output.pixels_per_point,
            };

            egui_cmd_bufs =
                egui.renderer
                    .update_buffers(&gpu.device, &gpu.queue, &mut encoder, &tris, &screen);

            {
                let mut rpass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui HUD"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    })
                    .forget_lifetime();
                egui.renderer.render(&mut rpass, &tris, &screen);
            }

            for id in &full_output.textures_delta.free {
                egui.renderer.free_texture(id);
            }
        }

        // Submit any egui-generated command buffers (texture uploads) BEFORE
        // the main encoder, then the main encoder (terrain + HUD render pass).
        gpu.queue.submit(
            egui_cmd_bufs
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        output.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("World Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.window_size = (size.width.max(1), size.height.max(1));
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // Feed the event to egui first so the HUD gets hover/click state and
        // tooltips work. If egui consumed it (pointer/keyboard over a widget),
        // skip the game's own input handling so digging/look don't fire through
        // the HUD. RedrawRequested must still reach render(), so never swallow it.
        if !matches!(event, WindowEvent::RedrawRequested) {
            if let (Some(egui), Some(win)) = (self.egui.as_mut(), self.window.as_ref()) {
                let resp = egui.state.on_window_event(win.as_ref(), &event);
                if resp.repaint {
                    win.request_redraw();
                }
                if resp.consumed {
                    return;
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.window_size = (size.width.max(1), size.height.max(1));
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);

                    let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Depth Texture"),
                        size: wgpu::Extent3d {
                            width: gpu.config.width,
                            height: gpu.config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    gpu.depth_view =
                        depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    gpu.depth_texture = depth_texture;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.input.keys.insert(key);
                            match key {
                                KeyCode::Escape => event_loop.exit(),
                                KeyCode::KeyR => self.reset_world(),
                                KeyCode::Digit1 => self.add_water_at_cursor(),
                                KeyCode::Digit2 => self.add_muddy_water_at_cursor(),
                                KeyCode::Digit3 => {
                                    if let Some(hit) = self.raycast_terrain() {
                                        let hf_height =
                                            self.world.ground_height(hit.x as usize, hit.z as usize);
                                        self.emitter.place_at_cursor(hit, hf_height);
                                        self.emitter.enabled = !self.emitter.enabled;
                                        println!(
                                            "Emitter toggle: {} at {:?}",
                                            self.emitter.enabled, self.emitter.position
                                        );
                                    }
                                }
                                KeyCode::KeyE => {
                                    self.emitter.enabled = !self.emitter.enabled;
                                    println!(
                                        "Emitter: {}",
                                        if self.emitter.enabled { "ON" } else { "OFF" }
                                    );
                                }
                                // Material selection
                                KeyCode::KeyG => {
                                    self.selected_material = 2;
                                    println!("Selected: Gravel");
                                }
                                KeyCode::KeyO => {
                                    self.selected_material = 1;
                                    println!("Selected: Overburden");
                                }
                                KeyCode::KeyT => {
                                    self.selected_material = 0;
                                    println!("Selected: Sediment");
                                }
                                // Visibility toggles
                                KeyCode::KeyH => {
                                    self.show_water = !self.show_water;
                                    println!("Water: {}", if self.show_water { "ON" } else { "OFF" });
                                }
                                KeyCode::KeyV => {
                                    self.show_velocity = !self.show_velocity;
                                    println!("Velocity coloring: {}", if self.show_velocity { "ON" } else { "OFF" });
                                }
                                // Emitter rate adjustment
                                KeyCode::ArrowUp => {
                                    self.emitter.rate *= 1.5;
                                    println!("Emitter rate: {:.0}", self.emitter.rate);
                                }
                                KeyCode::ArrowDown => {
                                    self.emitter.rate = (self.emitter.rate / 1.5).max(1.0);
                                    println!("Emitter rate: {:.0}", self.emitter.rate);
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.input.keys.remove(&key);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Right => {
                    self.input.mouse_look = state == ElementState::Pressed;
                    if !self.input.mouse_look {
                        self.input.last_mouse_pos = None;
                    }
                }
                MouseButton::Left => {
                    self.input.left_mouse = state == ElementState::Pressed;
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_pos = (position.x as f32, position.y as f32);

                if self.input.mouse_look {
                    if let Some((last_x, last_y)) = self.input.last_mouse_pos {
                        let dx = (position.x - last_x) as f32;
                        let dy = (position.y - last_y) as f32;
                        self.camera.yaw += dx * self.camera.sensitivity;
                        self.camera.pitch =
                            (self.camera.pitch - dy * self.camera.sensitivity).clamp(-1.4, 1.4);
                    }
                    self.input.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y * 2.0,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.input.scroll_delta += y;
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

fn build_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 50.0);

    // Noise functions
    fn noise2d(x: f32, y: f32) -> f32 {
        let n = (x * 127.1 + y * 311.7).sin() * 43758.5453;
        n.fract()
    }

    fn smooth_noise(x: f32, y: f32, scale: f32) -> f32 {
        let sx = x / scale;
        let sy = y / scale;
        let x0 = sx.floor();
        let y0 = sy.floor();
        let fx = sx - x0;
        let fy = sy - y0;

        let u = fx * fx * (3.0 - 2.0 * fx);
        let v = fy * fy * (3.0 - 2.0 * fy);

        let n00 = noise2d(x0, y0);
        let n10 = noise2d(x0 + 1.0, y0);
        let n01 = noise2d(x0, y0 + 1.0);
        let n11 = noise2d(x0 + 1.0, y0 + 1.0);

        let nx0 = n00 + (n10 - n00) * u;
        let nx1 = n01 + (n11 - n01) * u;
        nx0 + (nx1 - nx0) * v
    }

    fn fbm(x: f32, y: f32, octaves: u32) -> f32 {
        let mut value = 0.0;
        let mut amp = 0.5;
        let mut freq = 1.0;
        for _ in 0..octaves {
            value += amp * smooth_noise(x * freq, y * freq, 1.0);
            amp *= 0.5;
            freq *= 2.0;
        }
        value
    }

    // =========================================================================
    // GEOLOGICAL PARAMETERS
    // =========================================================================
    // River valley carved into mountainous terrain
    // Z=0 is upstream (high), Z=WORLD_DEPTH is downstream (low)
    // Gradient: ~15m drop over the length of the map

    let center_x = WORLD_WIDTH as f32 / 2.0;
    let river_width = WORLD_WIDTH as f32 * 0.12;      // Active channel width
    let floodplain_width = WORLD_WIDTH as f32 * 0.35; // Valley floor width

    // Elevation parameters
    let upstream_bedrock = 35.0;   // Bedrock elevation at z=0
    let downstream_bedrock = 8.0;  // Bedrock elevation at z=max
    let ridge_height = 45.0;       // Maximum ridge height above valley floor

    for z in 0..WORLD_DEPTH {
        let zf = z as f32;
        let z_progress = zf / WORLD_DEPTH as f32; // 0.0 upstream, 1.0 downstream

        // Base bedrock elevation follows river gradient
        let base_bedrock = upstream_bedrock + (downstream_bedrock - upstream_bedrock) * z_progress;

        // River meanders slightly
        let meander = ((zf * 0.012).sin() * 0.6 + (zf * 0.007).cos() * 0.4)
            * floodplain_width * 0.15;
        let river_center = center_x + meander;

        // Pool-riffle sequence in bedrock (natural river feature)
        let pool_riffle = (zf * 0.035).sin() * 0.8
            + (zf * 0.09).sin() * 0.4
            + (zf * 0.19).cos() * 0.2;

        for x in 0..WORLD_WIDTH {
            let xf = x as f32;
            let idx = world.idx(x, z);

            let dist_from_river = xf - river_center;
            let abs_dist = dist_from_river.abs();

            // Noise layers for terrain variation
            let large_noise = fbm(xf * 0.015, zf * 0.015, 4);
            let medium_noise = fbm(xf * 0.04, zf * 0.04, 3);
            let detail_noise = fbm(xf * 0.1, zf * 0.1, 2);
            let crack_noise = fbm(xf * 0.25 + 100.0, zf * 0.25, 2);

            let bedrock: f32;
            let mut paydirt = 0.0_f32;
            let mut gravel = 0.0_f32;
            let mut overburden = 0.0_f32;
            let mut sediment = 0.0_f32;

            // Determine zone based on distance from river
            let in_channel = abs_dist < river_width / 2.0;
            let in_floodplain = abs_dist < floodplain_width / 2.0;

            if in_channel {
                // =====================================================
                // ACTIVE RIVER CHANNEL - Where the gold concentrates
                // =====================================================
                let channel_t = abs_dist / (river_width / 2.0);

                // V-shaped channel carved into bedrock
                let channel_depth = 2.5 * (1.0 - channel_t.powi(2));

                // Bedrock outcrops and pools
                let outcrop = if detail_noise > 0.72 { (detail_noise - 0.72) * 4.0 } else { 0.0 };
                let pool = if detail_noise < 0.22 { (0.22 - detail_noise) * 3.0 } else { 0.0 };

                // Cracks in bedrock where gold settles
                let crack_depth = if crack_noise > 0.65 { (crack_noise - 0.65) * 1.5 } else { 0.0 };

                bedrock = (base_bedrock - channel_depth + pool_riffle + outcrop - pool - crack_depth).max(1.0);

                // PAYDIRT: Gold-bearing layer sits directly on bedrock
                // Concentrates in cracks, pools, and behind outcrops
                let gold_trap = crack_depth > 0.3 || pool > 0.5 || (outcrop > 0.0 && detail_noise < 0.5);
                if gold_trap {
                    paydirt = 0.15 + rand_float() * 0.25 + crack_depth * 0.3;
                } else {
                    paydirt = 0.02 + rand_float() * 0.05;
                }

                // GRAVEL: Coarse material on top of paydirt
                gravel = 0.1 + rand_float() * 0.15 + medium_noise * 0.2;
                if detail_noise < 0.35 { gravel += 0.2; } // Gravel bars

            } else if in_floodplain {
                // =====================================================
                // FLOODPLAIN / VALLEY FLOOR
                // =====================================================
                let flood_t = (abs_dist - river_width / 2.0)
                    / (floodplain_width / 2.0 - river_width / 2.0);
                let flood_t = flood_t.clamp(0.0, 1.0);

                // Gentle rise from channel edge to valley wall
                let bank_rise = flood_t.powf(1.5) * 4.0;
                bedrock = base_bedrock + bank_rise + large_noise * 2.0 + detail_noise * 0.5;

                // Inside of meander bends: point bars with paydirt
                let bend_inside = dist_from_river * meander < 0.0;
                if bend_inside && flood_t < 0.25 {
                    // Point bar deposit - ancient river channel
                    paydirt = 0.08 + rand_float() * 0.15;
                    gravel = 0.4 + rand_float() * 0.5 + medium_noise * 0.3;
                } else if flood_t < 0.4 {
                    // Near-channel floodplain
                    paydirt = rand_float() * 0.03;
                    gravel = (0.3 - flood_t) + rand_float() * 0.15;
                } else {
                    // Distal floodplain
                    gravel = rand_float() * 0.1;
                }

                // OVERBURDEN: Clay and soil, thicker away from channel
                overburden = flood_t * 1.5 + medium_noise * 0.4;

                // SEDIMENT: Topsoil layer
                sediment = 0.1 + detail_noise * 0.15;

            } else {
                // =====================================================
                // VALLEY WALLS / RIDGES - High terrain
                // =====================================================
                let wall_start = floodplain_width / 2.0;
                let wall_end = WORLD_WIDTH as f32 / 2.0;
                let wall_t = ((abs_dist - wall_start) / (wall_end - wall_start)).clamp(0.0, 1.0);

                // Steep valley walls rising to ridges
                let wall_height = wall_t.sqrt() * ridge_height;

                // Add mountain character with noise
                let mountain_noise = large_noise * 8.0 + medium_noise * 3.0;

                bedrock = base_bedrock + wall_height + mountain_noise;

                // Thin soil on steep slopes, thicker in hollows
                let slope_factor = wall_t.sqrt();
                overburden = (1.0 - slope_factor) * 1.5 + detail_noise * 0.3;
                sediment = (1.0 - slope_factor) * 0.3 + detail_noise * 0.1;

                // Occasional gravel on lower slopes (colluvium)
                if wall_t < 0.3 {
                    gravel = (0.3 - wall_t) * 0.5 * detail_noise;
                }
            }

            world.bedrock_elevation[idx] = bedrock;
            world.paydirt_thickness[idx] = paydirt;
            world.gravel_thickness[idx] = gravel;
            world.overburden_thickness[idx] = overburden;
            world.terrain_sediment[idx] = sediment;
        }
    }

    // Pre-fill river with water (thin layer following gradient)
    let river_width_i = (river_width / 2.0) as i32;
    for z in 0..WORLD_DEPTH {
        let zf = z as f32;
        let meander = ((zf * 0.012).sin() * 0.6 + (zf * 0.007).cos() * 0.4)
            * floodplain_width * 0.15;
        let river_center = center_x + meander;

        for x in 0..WORLD_WIDTH {
            let dist = (x as f32 - river_center).abs();
            if dist < river_width_i as f32 {
                let idx = world.idx(x, z);
                world.water_surface[idx] = world.ground_height(x, z) + 0.3;
            }
        }
    }

    world
}

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ambient = 0.6;
    let final_color = in.color.rgb * ambient;
    return vec4<f32>(final_color, in.color.a);
}
"#;

// =============================================================================
// HEADLESS SCREENSHOT SUPPORT
// =============================================================================

const SHOT_WIDTH: u32 = 1024;
const SHOT_HEIGHT: u32 = 768;

/// Save raw RGBA8 pixels to a PNG file. Copied from
/// `crates/game/tests/visual_regression_tests.rs`.
fn save_png(path: &std::path::Path, pixels: &[u8], width: u32, height: u32) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;

    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("Failed to write PNG header: {}", e))?;

    writer
        .write_image_data(pixels)
        .map_err(|e| format!("Failed to write PNG data: {}", e))?;

    Ok(())
}

/// Run the world simulation for `frames` steps fully offscreen and write a PNG
/// screenshot to `path`. Reuses the same heightfield render + sim update used by
/// the live `render()` path, rendering to an offscreen 1024x768 texture.
fn run_headless_screenshot(path: &std::path::Path, frames: u32) {
    let width = SHOT_WIDTH;
    let height = SHOT_HEIGHT;

    // --- Build CPU world + emitter, mirroring App::new() ---
    let mut app = App::new();

    // --- Create a surface-less device/queue ---
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No GPU adapter available for headless screenshot");

    // GpuHeightfield needs higher storage buffer limits (matches init_gpu()).
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;
    limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Headless Screenshot Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create headless device");

    // Offscreen render format (sRGB, matching the visual regression harness).
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    // --- Color target texture (RENDER_ATTACHMENT + COPY_SRC for readback) ---
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Screenshot Color Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // --- Depth texture (matches live render's Depth32Float) ---
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Screenshot Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // --- Emitter sphere pipeline (mirrors init_gpu()'s SHADER pipeline) ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Screenshot Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Screenshot Uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Screenshot Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Screenshot Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Screenshot Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let emitter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Screenshot Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<WorldVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
            }],
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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // --- GPU heightfield, mirroring init_gpu() ---
    let heightfield = GpuHeightfield::new(
        &device,
        app.world.width as u32,
        app.world.depth as u32,
        app.world.cell_size,
        INITIAL_HEIGHT,
        format,
    );
    heightfield.upload_from_world(&queue, &app.world);

    // Drive the simulation offscreen for `frames` frames. Each frame mirrors the
    // sim-update portion of App::update(): STEPS_PER_FRAME emitter+water dispatches,
    // then a readback into the CPU world.
    println!("Running {} simulation frames offscreen...", frames);
    for frame in 0..frames {
        for _ in 0..STEPS_PER_FRAME {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Headless Sim Encoder"),
            });

            app.emitter.update_gpu(&heightfield, &queue, DT);
            heightfield.dispatch_emitter(&mut encoder);

            heightfield.update_params(&queue, DT);
            heightfield.dispatch(&mut encoder, DT);

            queue.submit(Some(encoder.finish()));
        }

        // Sync back to CPU world (matches live path).
        pollster::block_on(heightfield.download_to_world(&device, &queue, &mut app.world));

        if frame % 30 == 0 || frame == frames - 1 {
            println!("  frame {}/{}", frame + 1, frames);
        }
    }

    // Build emitter visualization mesh for the final render (mirrors App::update()).
    let emitter_mesh = if app.emitter.enabled {
        let (positions, indices) = app.emitter.visualize(16);
        let vertices: Vec<WorldVertex> = positions
            .iter()
            .map(|p| WorldVertex {
                position: p.to_array(),
                color: [0.0, 1.0, 1.0, 1.0],
            })
            .collect();
        Some(Mesh::new(&device, &vertices, &indices, "Screenshot Emitter Mesh"))
    } else {
        None
    };

    // --- Render one frame to the offscreen texture (mirrors App::render()) ---
    let aspect = width as f32 / height as f32;
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
    let view = app.camera.view_matrix();
    let view_proj = proj * view;

    let uniforms = Uniforms {
        view_proj: view_proj.to_cols_array_2d(),
        camera_pos: app.camera.position.to_array(),
        _pad0: 0.0,
    };
    queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Screenshot Render Encoder"),
    });

    // Heightfield (terrain + water) — this clears the color + depth attachments.
    heightfield.render(
        &mut encoder,
        &color_view,
        &depth_view,
        &queue,
        view_proj.to_cols_array_2d(),
        app.camera.position.to_array(),
        app.start_time.elapsed().as_secs_f32(),
        app.show_water,
        app.show_velocity,
    );

    // Emitter sphere overlay.
    if let Some(mesh) = emitter_mesh.as_ref() {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Screenshot Emitter Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
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
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&emitter_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }

    // --- egui HUD overlay (headless) ---
    // No winit window here, so drive egui directly with a synthetic RawInput
    // (full-screen rect, 1.0 pixels-per-point). Renders the SAME build_hud_ui
    // as the live path into the SAME color_view with LoadOp::Load.
    let egui_cmd_bufs: Vec<wgpu::CommandBuffer> = {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let mut egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        let hud = HudState {
            mode: "World".into(),
            material: match app.selected_material {
                2 => "Gravel",
                1 => "Overburden",
                _ => "Sediment",
            }
            .into(),
            tool: "Excavate".into(),
            water_on: app.show_water,
            emitter_on: app.emitter.enabled,
            fps: 0.0,
        };

        let make_input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };
        // First pass warms up egui layout/state (a brand-new Context can emit
        // zero shapes on frame 0); the second pass produces the real geometry.
        // The font-atlas texture is emitted on the FIRST pass, so its delta must
        // be applied too (otherwise glyphs render blank).
        let warmup = ctx.run(make_input(), |ctx| build_hud_ui(ctx, &hud));
        for (id, delta) in &warmup.textures_delta.set {
            egui_renderer.update_texture(&device, &queue, *id, delta);
        }
        let full_output = ctx.run(make_input(), |ctx| build_hud_ui(ctx, &hud));
        let tris = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(&device, &queue, *id, delta);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: full_output.pixels_per_point,
        };
        let cmd_bufs = egui_renderer.update_buffers(&device, &queue, &mut encoder, &tris, &screen);

        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Screenshot egui HUD"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            egui_renderer.render(&mut rpass, &tris, &screen);
        }

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        cmd_bufs
    };

    // --- Copy color texture to a staging buffer (copied from visual harness) ---
    let bytes_per_pixel = 4u32; // RGBA8
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Screenshot Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    // egui texture-upload command buffers must run before the encoder's egui
    // render pass, which itself must run before the texture->buffer copy.
    queue.submit(
        egui_cmd_bufs
            .into_iter()
            .chain(std::iter::once(encoder.finish())),
    );

    // --- Map and read back pixels (copied from visual harness) ---
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();

    // Unpad rows.
    let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        let row_end = row_start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&data[row_start..row_end]);
    }
    drop(data);
    staging_buffer.unmap();

    // --- Write PNG ---
    save_png(path, &pixels, width, height).expect("Failed to save screenshot PNG");
    println!(
        "Screenshot written: {} ({}x{})",
        path.display(),
        width,
        height
    );
}

/// Parse `--screenshot <PATH>` and `--frames <N>` from argv.
/// Returns `Some((path, frames))` when `--screenshot` is present.
fn parse_screenshot_args() -> Option<(std::path::PathBuf, u32)> {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<std::path::PathBuf> = None;
    let mut frames: u32 = 120;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--screenshot" => {
                let v = args.get(i + 1).expect("--screenshot requires a PATH argument");
                path = Some(std::path::PathBuf::from(v));
                i += 2;
            }
            "--frames" => {
                let v = args.get(i + 1).expect("--frames requires an N argument");
                frames = v.parse().expect("--frames must be a positive integer");
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    path.map(|p| (p, frames))
}

fn main() {
    env_logger::init();

    // Headless screenshot mode: run offscreen, write PNG, then exit.
    if let Some((path, frames)) = parse_screenshot_args() {
        run_headless_screenshot(&path, frames);
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
