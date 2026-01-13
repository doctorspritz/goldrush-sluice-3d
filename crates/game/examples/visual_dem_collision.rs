//! Visual DEM Collision Tests
//!
//! Visual validation for DEM collision physics tests:
//!   1 = Floor collision (drop, bounce, settle)
//!   2 = Wall collision (hit wall, bounce back)
//!   3 = Clump collision (two clumps collide)
//!   4 = No penetration (100 clumps settle)
//!   5 = Gutter SDF collision (clumps in trough)
//!
//! Run: cargo run --example visual_dem_collision --release
//!
//! Controls:
//!   1-5       = Switch test scenario
//!   SPACE     = Pause/unpause
//!   R         = Reset current test
//!   Arrows    = Orbit camera

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const MAX_INSTANCES: usize = 1000;
const DT: f32 = 1.0 / 120.0;
const SUBSTEPS: usize = 2;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const PARTICLE_RADIUS: f32 = 0.02; // 2cm for visibility
const PARTICLE_MASS: f32 = 0.1;
const GRAVITY: f32 = -9.81;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TestScenario {
    FloorCollision, // Test 1: Drop clump onto floor
    WallCollision,  // Test 2: Launch clump at wall
    ClumpCollision, // Test 3: Two clumps head-on
    NoPenetration,  // Test 4: 100 clumps settle
    GutterSdf,      // Test 5: Clumps in trough shape
}

impl TestScenario {
    fn name(&self) -> &'static str {
        match self {
            TestScenario::FloorCollision => "1: Floor Collision (drop & bounce)",
            TestScenario::WallCollision => "2: Wall Collision (hit & reflect)",
            TestScenario::ClumpCollision => "3: Clump Collision (head-on)",
            TestScenario::NoPenetration => "4: No Penetration (100 clumps)",
            TestScenario::GutterSdf => "5: Gutter SDF (trough shape)",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    position: [f32; 3],
    scale: f32,
    rotation: [f32; 4],
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
    sim: ClusterSimulation3D,
    template_idx: usize,
    scenario: TestScenario,
    instances: Vec<Instance>,
    paused: bool,
    camera_angle: f32,
    camera_distance: f32,
    camera_height: f32,
    frame_count: u32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    floor_pipeline: wgpu::RenderPipeline,
    mesh_vertex_buffer: wgpu::Buffer,
    mesh_vertex_count: u32,
    floor_vertex_buffer: wgpu::Buffer,
    floor_vertex_count: u32,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl App {
    fn new() -> Self {
        let scenario = TestScenario::FloorCollision;
        let (sim, template_idx) = create_scenario(scenario);

        Self {
            window: None,
            gpu: None,
            sim,
            template_idx,
            scenario,
            instances: Vec::new(),
            paused: false,
            camera_angle: 0.8,
            camera_distance: 3.0,
            camera_height: 1.5,
            frame_count: 0,
        }
    }

    fn switch_scenario(&mut self, scenario: TestScenario) {
        self.scenario = scenario;
        let (sim, template_idx) = create_scenario(scenario);
        self.sim = sim;
        self.template_idx = template_idx;
        self.frame_count = 0;
        println!("\n=== {} ===", scenario.name());
    }

    fn reset(&mut self) {
        self.switch_scenario(self.scenario);
    }

    fn build_instances(&mut self) {
        self.instances.clear();

        for clump in &self.sim.clumps {
            let color = match clump.template_idx {
                0 => [0.8, 0.6, 0.2, 1.0], // Gold
                1 => [0.5, 0.5, 0.6, 1.0], // Gray
                _ => [0.6, 0.3, 0.3, 1.0], // Red
            };

            self.instances.push(Instance {
                position: clump.position.to_array(),
                scale: self.sim.templates[clump.template_idx].bounding_radius,
                rotation: clump.rotation.to_array(),
                color,
            });
        }
    }

    fn print_status(&self) {
        if self.frame_count % 60 == 0 {
            let clump_count = self.sim.clumps.len();
            if clump_count > 0 {
                let min_y = self
                    .sim
                    .clumps
                    .iter()
                    .map(|c| c.position.y)
                    .fold(f32::INFINITY, f32::min);
                let max_vel = self
                    .sim
                    .clumps
                    .iter()
                    .map(|c| c.velocity.length())
                    .fold(0.0f32, f32::max);
                println!(
                    "Frame {}: {} clumps, min_y={:.4}, max_vel={:.3} m/s",
                    self.frame_count, clump_count, min_y, max_vel
                );
            }
        }
    }
}

fn create_scenario(scenario: TestScenario) -> (ClusterSimulation3D, usize) {
    let bounds = 2.0;
    let mut sim =
        ClusterSimulation3D::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(bounds, bounds, bounds));
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, PARTICLE_RADIUS, PARTICLE_MASS);
    let template_idx = sim.add_template(template);

    match scenario {
        TestScenario::FloorCollision => {
            // Drop single clump from 1m
            let drop_height = 1.0;
            let spawn_pos = Vec3::new(bounds / 2.0, drop_height + PARTICLE_RADIUS, bounds / 2.0);
            sim.spawn(template_idx, spawn_pos, Vec3::ZERO);
            println!("TEST 1: Floor Collision");
            println!("  Expected: Clump drops, bounces to ~4cm (e²=0.04), settles");
            println!("  Watch: Does it bounce? Does it settle above floor?");
        }

        TestScenario::WallCollision => {
            // Launch clump horizontally toward wall
            sim.gravity = Vec3::ZERO; // No gravity for clean wall bounce
            let spawn_pos = Vec3::new(0.5, bounds / 2.0, bounds / 2.0);
            let velocity = Vec3::new(2.0, 0.0, 0.0); // Toward +X wall
            sim.spawn(template_idx, spawn_pos, velocity);
            println!("TEST 2: Wall Collision");
            println!("  Expected: Clump hits wall at x={}, bounces back", bounds);
            println!("  Watch: Does it reverse direction? v_x should flip sign");
        }

        TestScenario::ClumpCollision => {
            // Two clumps approach head-on
            sim.gravity = Vec3::ZERO;
            let center = bounds / 2.0;
            let speed = 1.0;
            let separation = 1.0;

            // Clump A from left
            sim.spawn(
                template_idx,
                Vec3::new(center - separation / 2.0, center, center),
                Vec3::new(speed, 0.0, 0.0),
            );
            // Clump B from right
            sim.spawn(
                template_idx,
                Vec3::new(center + separation / 2.0, center, center),
                Vec3::new(-speed, 0.0, 0.0),
            );
            println!("TEST 3: Clump Collision");
            println!("  Expected: Clumps collide at center, both bounce back");
            println!("  Watch: Symmetric bounce, momentum conserved");
        }

        TestScenario::NoPenetration => {
            // Spawn 100 clumps in grid
            let grid_size = 10;
            let spacing = 0.12;
            let start_height = 1.5;

            for iz in 0..grid_size {
                for ix in 0..grid_size {
                    let pos = Vec3::new(
                        0.3 + ix as f32 * spacing,
                        start_height,
                        0.3 + iz as f32 * spacing,
                    );
                    // Small random velocity
                    let vel = Vec3::new(
                        (ix as f32 * 0.01) % 0.1 - 0.05,
                        0.0,
                        (iz as f32 * 0.01) % 0.1 - 0.05,
                    );
                    sim.spawn(template_idx, pos, vel);
                }
            }
            println!("TEST 4: No Penetration");
            println!("  Expected: 100 clumps settle, NONE go below floor");
            println!(
                "  Watch: min_y should stay >= {} (particle radius)",
                PARTICLE_RADIUS
            );
        }

        TestScenario::GutterSdf => {
            // TODO: This requires SDF collision which needs piece geometry
            // For now, just show clumps in a box
            let spawn_count = 20;
            for i in 0..spawn_count {
                let angle = (i as f32 / spawn_count as f32) * std::f32::consts::TAU;
                let r = 0.3;
                let pos = Vec3::new(
                    bounds / 2.0 + r * angle.cos(),
                    1.0 + (i as f32 * 0.05),
                    bounds / 2.0 + r * angle.sin(),
                );
                sim.spawn(template_idx, pos, Vec3::ZERO);
            }
            println!("TEST 5: Gutter SDF Collision");
            println!("  NOTE: Requires piece SDF integration - showing box bounds only");
            println!("  TODO: Add gutter geometry from washplant_editor");
        }
    }

    (sim, template_idx)
}

fn create_sphere_mesh(subdivisions: u32) -> Vec<Vertex> {
    let mut vertices = Vec::new();

    // Simple icosphere approximation
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let base_verts = [
        Vec3::new(-1.0, phi, 0.0).normalize(),
        Vec3::new(1.0, phi, 0.0).normalize(),
        Vec3::new(-1.0, -phi, 0.0).normalize(),
        Vec3::new(1.0, -phi, 0.0).normalize(),
        Vec3::new(0.0, -1.0, phi).normalize(),
        Vec3::new(0.0, 1.0, phi).normalize(),
        Vec3::new(0.0, -1.0, -phi).normalize(),
        Vec3::new(0.0, 1.0, -phi).normalize(),
        Vec3::new(phi, 0.0, -1.0).normalize(),
        Vec3::new(phi, 0.0, 1.0).normalize(),
        Vec3::new(-phi, 0.0, -1.0).normalize(),
        Vec3::new(-phi, 0.0, 1.0).normalize(),
    ];

    let faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ];

    fn subdivide(v1: Vec3, v2: Vec3, v3: Vec3, depth: u32, verts: &mut Vec<Vertex>) {
        if depth == 0 {
            let n1 = v1.normalize();
            let n2 = v2.normalize();
            let n3 = v3.normalize();
            verts.push(Vertex {
                position: n1.to_array(),
                normal: n1.to_array(),
            });
            verts.push(Vertex {
                position: n2.to_array(),
                normal: n2.to_array(),
            });
            verts.push(Vertex {
                position: n3.to_array(),
                normal: n3.to_array(),
            });
        } else {
            let m12 = ((v1 + v2) / 2.0).normalize();
            let m23 = ((v2 + v3) / 2.0).normalize();
            let m31 = ((v3 + v1) / 2.0).normalize();
            subdivide(v1, m12, m31, depth - 1, verts);
            subdivide(v2, m23, m12, depth - 1, verts);
            subdivide(v3, m31, m23, depth - 1, verts);
            subdivide(m12, m23, m31, depth - 1, verts);
        }
    }

    for (i1, i2, i3) in faces {
        subdivide(
            base_verts[i1],
            base_verts[i2],
            base_verts[i3],
            subdivisions,
            &mut vertices,
        );
    }

    vertices
}

fn create_floor_mesh(size: f32) -> Vec<Vertex> {
    let y = 0.0;
    let normal = [0.0, 1.0, 0.0];

    vec![
        Vertex {
            position: [-size, y, -size],
            normal,
        },
        Vertex {
            position: [size, y, -size],
            normal,
        },
        Vertex {
            position: [size, y, size],
            normal,
        },
        Vertex {
            position: [-size, y, -size],
            normal,
        },
        Vertex {
            position: [size, y, size],
            normal,
        },
        Vertex {
            position: [-size, y, size],
            normal,
        },
    ]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Visual DEM Collision Tests - Press 1-5 to switch")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
        println!("\nControls: 1-5=switch test, SPACE=pause, R=reset, arrows=orbit");
        println!("\n=== {} ===", self.scenario.name());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Digit1 => self.switch_scenario(TestScenario::FloorCollision),
                        KeyCode::Digit2 => self.switch_scenario(TestScenario::WallCollision),
                        KeyCode::Digit3 => self.switch_scenario(TestScenario::ClumpCollision),
                        KeyCode::Digit4 => self.switch_scenario(TestScenario::NoPenetration),
                        KeyCode::Digit5 => self.switch_scenario(TestScenario::GutterSdf),
                        KeyCode::Space => {
                            self.paused = !self.paused;
                            println!("{}", if self.paused { "PAUSED" } else { "RUNNING" });
                        }
                        KeyCode::KeyR => self.reset(),
                        KeyCode::ArrowLeft => self.camera_angle -= 0.1,
                        KeyCode::ArrowRight => self.camera_angle += 0.1,
                        KeyCode::ArrowUp => {
                            self.camera_distance = (self.camera_distance - 0.2).max(1.0)
                        }
                        KeyCode::ArrowDown => self.camera_distance += 0.2,
                        KeyCode::Escape => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Clump pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clump Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 5,
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
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Floor pipeline
        let floor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Floor Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_floor"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_floor"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sphere_mesh = create_sphere_mesh(2);
        let floor_mesh = create_floor_mesh(5.0);

        let mesh_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&sphere_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let floor_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Floor Vertex Buffer"),
            contents: bytemuck::cast_slice(&floor_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (MAX_INSTANCES * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let (depth_texture, depth_view) = create_depth_texture(&device, &config);

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            floor_pipeline,
            mesh_vertex_buffer,
            mesh_vertex_count: sphere_mesh.len() as u32,
            floor_vertex_buffer,
            floor_vertex_count: floor_mesh.len() as u32,
            instance_buffer,
            uniform_buffer,
            bind_group,
            depth_texture,
            depth_view,
        });
    }

    fn render(&mut self) {
        let window = match &self.window {
            Some(w) => Arc::clone(w),
            None => return,
        };
        let mut gpu = match self.gpu.take() {
            Some(g) => g,
            None => return,
        };

        if !self.paused {
            for _ in 0..SUBSTEPS {
                self.sim.step(DT);
            }
            self.frame_count += 1;
            self.print_status();
        }

        let bounds_center = (self.sim.bounds_min + self.sim.bounds_max) * 0.5;
        let camera_pos = bounds_center
            + Vec3::new(
                self.camera_angle.cos() * self.camera_distance,
                self.camera_height,
                self.camera_angle.sin() * self.camera_distance,
            );

        let view = Mat4::look_at_rh(camera_pos, bounds_center, Vec3::Y);
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.build_instances();
        let instance_count = self.instances.len().min(MAX_INSTANCES);
        if instance_count > 0 {
            gpu.queue.write_buffer(
                &gpu.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances[..instance_count]),
            );
        }

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = window.inner_size();
                gpu.config.width = size.width.max(1);
                gpu.config.height = size.height.max(1);
                gpu.surface.configure(&gpu.device, &gpu.config);
                let (dt, dv) = create_depth_texture(&gpu.device, &gpu.config);
                gpu.depth_texture = dt;
                gpu.depth_view = dv;
                self.gpu = Some(gpu);
                return;
            }
            Err(_) => {
                self.gpu = Some(gpu);
                return;
            }
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
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw floor
            pass.set_pipeline(&gpu.floor_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.floor_vertex_buffer.slice(..));
            pass.draw(0..gpu.floor_vertex_count, 0..1);

            // Draw clumps
            if instance_count > 0 {
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.mesh_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                pass.draw(0..gpu.mesh_vertex_count, 0..instance_count as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.gpu = Some(gpu);
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
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
    @location(1) normal: vec3<f32>,
}

struct InstanceInput {
    @location(2) instance_pos: vec3<f32>,
    @location(3) scale: f32,
    @location(4) rotation: vec4<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
}

fn rotate_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let scaled_pos = vertex.position * instance.scale;
    let rotated_pos = rotate_by_quat(scaled_pos, instance.rotation);
    let world_pos = rotated_pos + instance.instance_pos;

    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.color;
    out.normal = rotate_by_quat(vertex.normal, instance.rotation);
    out.world_pos = world_pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ambient = 0.3;
    let diffuse = max(dot(normalize(in.normal), light_dir), 0.0) * 0.7;
    let intensity = ambient + diffuse;

    return vec4<f32>(in.color.rgb * intensity, in.color.a);
}

@vertex
fn vs_floor(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(vertex.position, 1.0);
    out.color = vec4<f32>(0.2, 0.25, 0.2, 1.0);
    out.normal = vertex.normal;
    out.world_pos = vertex.position;
    return out;
}

@fragment
fn fs_floor(in: VertexOutput) -> @location(0) vec4<f32> {
    // Grid pattern
    let grid = abs(fract(in.world_pos.x * 2.0) - 0.5) + abs(fract(in.world_pos.z * 2.0) - 0.5);
    let grid_line = smoothstep(0.45, 0.5, grid);
    let base_color = mix(vec3<f32>(0.15, 0.18, 0.15), vec3<f32>(0.2, 0.25, 0.2), grid_line);
    return vec4<f32>(base_color, 1.0);
}
"#;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
