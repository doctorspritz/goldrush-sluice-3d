//! Riffled Chute Density Separation Visual Demo
//!
//! Shows a sloped chute with riffles that trap heavy gold particles
//! while lighter sand washes downstream.
//!
//! Run with: cargo run --example riffle_chute_visual --release

use bytemuck::{Pod, Zeroable};
use game::equipment_geometry::{ChuteConfig, ChuteGeometryBuilder};
use game::sluice_geometry::SluiceVertex;
use glam::{Mat4, Vec3};
use sim3d::{
    ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FlipSimulation3D, IrregularStyle3D,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Grid configuration
const CELL_SIZE: f32 = 0.01; // 1cm cells
const GRID_WIDTH: usize = 80;
const GRID_HEIGHT: usize = 24;
const GRID_DEPTH: usize = 20;

// Chute configuration
const FLOOR_START: usize = 16; // Upstream height
const FLOOR_END: usize = 4; // Downstream height
const WALL_HEIGHT: usize = 6;
const WALL_THICKNESS: usize = 1;
const RIFFLE_SPACING: usize = 10;
const RIFFLE_HEIGHT: usize = 2;
const RIFFLE_THICKNESS: usize = 1;

// Particle configuration - relative density (water=1.0)
const GOLD_DENSITY: f32 = 19.3;
const SAND_DENSITY: f32 = 2.65;
const GOLD_RADIUS: f32 = 0.006;
const SAND_RADIUS: f32 = 0.008;
const NUM_GOLD: usize = 8;
const NUM_SAND: usize = 16;

// Water configuration
const WATER_VELOCITY: f32 = 0.15;
const WATER_EMIT_RATE: usize = 80;
const MAX_WATER_PARTICLES: usize = 80000;

// Rendering
const MAX_SEDIMENT: usize = 100;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 3],
    color: [f32; 4],
    scale: f32,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sim: FlipSimulation3D,
    dem: ClusterSimulation3D,
    chute_config: ChuteConfig,
    gold_template_idx: usize,
    sand_template_idx: usize,
    gold_indices: Vec<usize>,
    sand_indices: Vec<usize>,
    paused: bool,
    camera_angle: f32,
    camera_distance: f32,
    camera_height: f32,
    frame: u32,
    water_emitter_active: bool,
    sediment_spawned: bool,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    // Geometry pipeline
    geometry_pipeline: wgpu::RenderPipeline,
    geometry_vertex_buffer: wgpu::Buffer,
    geometry_index_buffer: wgpu::Buffer,
    geometry_index_count: u32,

    // Particle pipeline
    particle_pipeline: wgpu::RenderPipeline,
    water_instance_buffer: wgpu::Buffer,
    sediment_instance_buffer: wgpu::Buffer,
    quad_vertex_buffer: wgpu::Buffer,

    // Uniforms
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl App {
    fn new() -> Self {
        // Configure chute with riffles
        let chute_config = ChuteConfig {
            grid_width: GRID_WIDTH,
            grid_height: GRID_HEIGHT,
            grid_depth: GRID_DEPTH,
            cell_size: CELL_SIZE,
            floor_height_start: FLOOR_START,
            floor_height_end: FLOOR_END,
            side_wall_height: WALL_HEIGHT,
            wall_thickness: WALL_THICKNESS,
            riffle_spacing: RIFFLE_SPACING,
            riffle_height: RIFFLE_HEIGHT,
            riffle_thickness: RIFFLE_THICKNESS,
            ..Default::default()
        };

        let bounds_min = Vec3::ZERO;
        let bounds_max = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        );

        // Setup DEM
        let mut dem = ClusterSimulation3D::new(bounds_min, bounds_max);
        dem.gravity = Vec3::new(0.0, -9.8, 0.0);
        dem.restitution = 0.1;
        dem.friction = 0.6;
        dem.floor_friction = 1.0;
        dem.normal_stiffness = 5000.0;
        dem.tangential_stiffness = 2500.0;
        dem.rolling_friction = 0.15;
        dem.use_dem = true;

        // Gold template
        let gold_mass = GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * GOLD_RADIUS.powi(3);
        let gold_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 42,
                style: IrregularStyle3D::Round,
            },
            GOLD_RADIUS,
            gold_mass,
        );
        let gold_template_idx = dem.add_template(gold_template);

        // Sand template
        let sand_mass = SAND_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * SAND_RADIUS.powi(3);
        let sand_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 123,
                style: IrregularStyle3D::Round,
            },
            SAND_RADIUS,
            sand_mass,
        );
        let sand_template_idx = dem.add_template(sand_template);

        // Setup FLIP simulation
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);

        // Set chute geometry as solid cells
        for k in 0..GRID_DEPTH {
            for j in 0..GRID_HEIGHT {
                for i in 0..GRID_WIDTH {
                    if chute_config.is_solid(i, j, k) {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
        sim.grid.compute_sdf();

        let num_riffles = (GRID_WIDTH - RIFFLE_SPACING) / (RIFFLE_SPACING + RIFFLE_THICKNESS);
        println!("=== Riffled Chute Density Separation Demo ===");
        println!();
        println!(
            "Chute: {}x{}x{} cells ({:.2}m x {:.2}m x {:.2}m)",
            GRID_WIDTH,
            GRID_HEIGHT,
            GRID_DEPTH,
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        );
        println!("Slope: {} -> {} cells", FLOOR_START, FLOOR_END);
        println!(
            "Riffles: {} (height={}, spacing={})",
            num_riffles, RIFFLE_HEIGHT, RIFFLE_SPACING
        );
        println!();
        println!("Particles:");
        println!(
            "  Gold: {} particles, density={} kg/m³ (YELLOW)",
            NUM_GOLD, GOLD_DENSITY
        );
        println!(
            "  Sand: {} particles, density={} kg/m³ (BROWN)",
            NUM_SAND, SAND_DENSITY
        );
        println!();
        println!("Controls:");
        println!("  SPACE     - Pause/Resume");
        println!("  LEFT/RIGHT - Rotate camera");
        println!("  UP/DOWN   - Zoom in/out");
        println!("  R         - Reset simulation");
        println!("  S         - Spawn more sediment");

        Self {
            window: None,
            gpu: None,
            sim,
            dem,
            chute_config,
            gold_template_idx,
            sand_template_idx,
            gold_indices: Vec::new(),
            sand_indices: Vec::new(),
            paused: false,
            camera_angle: 0.8,
            camera_distance: 1.5,
            camera_height: 0.4,
            frame: 0,
            water_emitter_active: true,
            sediment_spawned: false,
        }
    }

    fn reset(&mut self) {
        self.sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        self.sim.gravity = Vec3::new(0.0, -9.8, 0.0);

        for k in 0..GRID_DEPTH {
            for j in 0..GRID_HEIGHT {
                for i in 0..GRID_WIDTH {
                    if self.chute_config.is_solid(i, j, k) {
                        self.sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
        self.sim.grid.compute_sdf();

        let bounds_min = Vec3::ZERO;
        let bounds_max = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        );
        self.dem = ClusterSimulation3D::new(bounds_min, bounds_max);
        self.dem.gravity = Vec3::new(0.0, -9.8, 0.0);
        self.dem.restitution = 0.1;
        self.dem.friction = 0.6;
        self.dem.floor_friction = 1.0;
        self.dem.normal_stiffness = 5000.0;
        self.dem.tangential_stiffness = 2500.0;
        self.dem.rolling_friction = 0.15;
        self.dem.use_dem = true;

        let gold_mass = GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * GOLD_RADIUS.powi(3);
        let gold_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 42,
                style: IrregularStyle3D::Round,
            },
            GOLD_RADIUS,
            gold_mass,
        );
        self.gold_template_idx = self.dem.add_template(gold_template);

        let sand_mass = SAND_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * SAND_RADIUS.powi(3);
        let sand_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 123,
                style: IrregularStyle3D::Round,
            },
            SAND_RADIUS,
            sand_mass,
        );
        self.sand_template_idx = self.dem.add_template(sand_template);

        self.gold_indices.clear();
        self.sand_indices.clear();
        self.frame = 0;
        self.water_emitter_active = true;
        self.sediment_spawned = false;
        println!("Simulation reset");
    }

    fn emit_water(&mut self) {
        if !self.water_emitter_active || self.sim.particles.len() > MAX_WATER_PARTICLES {
            return;
        }

        let inlet_x = CELL_SIZE * 3.0;
        let floor_y = (self.chute_config.floor_height_at(3) as f32 + 1.5) * CELL_SIZE;
        let center_z = GRID_DEPTH as f32 * CELL_SIZE / 2.0;
        let spread_z = (GRID_DEPTH - 2 * WALL_THICKNESS) as f32 * CELL_SIZE * 0.4;

        let flow_vel = Vec3::new(WATER_VELOCITY, -0.02, 0.0);

        for i in 0..WATER_EMIT_RATE {
            let seed = self.frame as f32 + i as f32 * 0.1;
            let z_offset = (rand_simple(seed) - 0.5) * spread_z;
            let y_offset = rand_simple(seed * 1.3) * CELL_SIZE;
            let pos = Vec3::new(inlet_x, floor_y + y_offset, center_z + z_offset);
            self.sim.spawn_particle_with_velocity(pos, flow_vel);
        }
    }

    fn spawn_sediment(&mut self) {
        let inlet_x = CELL_SIZE * 12.0;
        let floor_y = (self.chute_config.floor_height_at(12) as f32 + 3.0) * CELL_SIZE;
        let center_z = GRID_DEPTH as f32 * CELL_SIZE / 2.0;

        // Spawn gold
        for i in 0..NUM_GOLD {
            let seed = i as f32 * 7.3 + self.frame as f32;
            let offset_z = (rand_simple(seed) - 0.5) * CELL_SIZE * 8.0;
            let offset_x = (rand_simple(seed * 1.5) - 0.5) * CELL_SIZE * 4.0;
            let idx = self.dem.clumps.len();
            self.dem.spawn(
                self.gold_template_idx,
                Vec3::new(inlet_x + offset_x, floor_y, center_z + offset_z),
                Vec3::ZERO,
            );
            self.gold_indices.push(idx);
        }

        // Spawn sand
        for i in 0..NUM_SAND {
            let seed = i as f32 * 11.7 + self.frame as f32;
            let offset_z = (rand_simple(seed) - 0.5) * CELL_SIZE * 8.0;
            let offset_x = (rand_simple(seed * 1.5) - 0.5) * CELL_SIZE * 4.0;
            let idx = self.dem.clumps.len();
            self.dem.spawn(
                self.sand_template_idx,
                Vec3::new(inlet_x + offset_x, floor_y + 0.01, center_z + offset_z),
                Vec3::ZERO,
            );
            self.sand_indices.push(idx);
        }

        self.sediment_spawned = true;
        println!(
            "Spawned {} gold + {} sand at frame {}",
            NUM_GOLD, NUM_SAND, self.frame
        );
    }

    fn step(&mut self) {
        if self.paused {
            return;
        }

        let dt = 1.0 / 60.0;

        // Emit water
        self.emit_water();

        // Spawn sediment after water flow establishes
        if self.frame == 150 && !self.sediment_spawned {
            self.spawn_sediment();
        }

        // Step FLIP
        self.sim.update(dt);

        // Apply simple fluid forces to DEM particles
        if !self.dem.clumps.is_empty() {
            // Manual buoyancy and drag calculation
            let water_density = 1000.0;
            let gravity_mag = 9.8;
            let drag_coeff = 1.0;

            for i in 0..self.dem.clumps.len() {
                let template_idx = self.dem.clumps[i].template_idx;
                let template = &self.dem.templates[template_idx];
                let radius = template.bounding_radius;
                let mass = template.mass;

                let volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);

                // Buoyancy force
                let buoyancy = Vec3::new(0.0, water_density * volume * gravity_mag, 0.0);
                let buoyancy_accel = buoyancy / mass;
                self.dem.clumps[i].velocity += buoyancy_accel * dt;

                // Simple drag based on velocity difference from water (assume flowing +X)
                let water_vel = Vec3::new(WATER_VELOCITY * 0.5, 0.0, 0.0);
                let rel_vel = water_vel - self.dem.clumps[i].velocity;
                let area = std::f32::consts::PI * radius * radius;
                let drag_force =
                    0.5 * water_density * drag_coeff * area * rel_vel.length() * rel_vel;
                let drag_accel = drag_force / mass;
                self.dem.clumps[i].velocity += drag_accel * dt;
            }

            self.dem.step(dt);
        }

        self.frame += 1;

        // Print stats periodically
        if self.frame % 120 == 0 {
            let gold_mean_x: f32 = self
                .gold_indices
                .iter()
                .filter_map(|&idx| self.dem.clumps.get(idx).map(|c| c.position.x))
                .sum::<f32>()
                / self.gold_indices.len().max(1) as f32;
            let sand_mean_x: f32 = self
                .sand_indices
                .iter()
                .filter_map(|&idx| self.dem.clumps.get(idx).map(|c| c.position.x))
                .sum::<f32>()
                / self.sand_indices.len().max(1) as f32;

            if self.sediment_spawned {
                let separation =
                    (sand_mean_x - gold_mean_x) / (GRID_WIDTH as f32 * CELL_SIZE) * 100.0;
                println!(
                    "Frame {}: water={}, gold_x={:.3}m, sand_x={:.3}m, separation={:.1}%",
                    self.frame,
                    self.sim.particles.len(),
                    gold_mean_x,
                    sand_mean_x,
                    separation
                );
            } else {
                println!(
                    "Frame {}: water={} (waiting for sediment spawn at frame 150)",
                    self.frame,
                    self.sim.particles.len()
                );
            }
        }
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create depth texture
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create uniform buffer and bind group
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Build chute geometry
        let mut builder = ChuteGeometryBuilder::new(self.chute_config.clone());
        builder.build_mesh(|i, j, k| self.chute_config.is_solid(i, j, k));

        let geometry_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Geometry Vertex Buffer"),
            contents: bytemuck::cast_slice(builder.vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let geometry_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Geometry Index Buffer"),
            contents: bytemuck::cast_slice(builder.indices()),
            usage: wgpu::BufferUsages::INDEX,
        });
        let geometry_index_count = builder.indices().len() as u32;

        // Create geometry pipeline
        let geometry_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Geometry Shader"),
            source: wgpu::ShaderSource::Wgsl(GEOMETRY_SHADER.into()),
        });

        let geometry_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let geometry_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Geometry Pipeline"),
            layout: Some(&geometry_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &geometry_shader,
                entry_point: Some("vs_main"),
                buffers: &[SluiceVertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &geometry_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Create particle pipeline
        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
        });

        let particle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&particle_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Quad vertices
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Instance data
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
                module: &particle_shader,
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Create quad vertex buffer
        let quad_vertices: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create instance buffers
        let water_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Instance Buffer"),
            size: (MAX_WATER_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Instance Buffer"),
            size: (MAX_SEDIMENT * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            depth_view,
            geometry_pipeline,
            geometry_vertex_buffer,
            geometry_index_buffer,
            geometry_index_count,
            particle_pipeline,
            water_instance_buffer,
            sediment_instance_buffer,
            quad_vertex_buffer,
            uniform_buffer,
            uniform_bind_group,
        });
    }

    fn render(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE / 2.0,
            GRID_HEIGHT as f32 * CELL_SIZE / 2.0,
            GRID_DEPTH as f32 * CELL_SIZE / 2.0,
        );
        let camera_pos = center
            + Vec3::new(
                self.camera_distance * self.camera_angle.cos(),
                self.camera_height,
                self.camera_distance * self.camera_angle.sin(),
            );

        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let view_matrix = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
        let proj_matrix = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let view_proj = proj_matrix * view_matrix;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Update water instances (sample every N particles for performance)
        let stride = (self.sim.particles.list.len() / 20000).max(1);
        let water_instances: Vec<ParticleInstance> = self
            .sim
            .particles
            .list
            .iter()
            .step_by(stride)
            .take(MAX_WATER_PARTICLES)
            .map(|p| ParticleInstance {
                position: p.position.to_array(),
                color: [0.2, 0.5, 0.9, 0.6],
                scale: 0.003,
            })
            .collect();

        if !water_instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.water_instance_buffer,
                0,
                bytemuck::cast_slice(&water_instances),
            );
        }

        // Update sediment instances
        let mut sediment_instances: Vec<ParticleInstance> = Vec::new();

        for &idx in &self.gold_indices {
            if let Some(clump) = self.dem.clumps.get(idx) {
                sediment_instances.push(ParticleInstance {
                    position: clump.position.to_array(),
                    color: [1.0, 0.85, 0.0, 1.0], // Gold color
                    scale: GOLD_RADIUS * 2.0,
                });
            }
        }

        for &idx in &self.sand_indices {
            if let Some(clump) = self.dem.clumps.get(idx) {
                sediment_instances.push(ParticleInstance {
                    position: clump.position.to_array(),
                    color: [0.7, 0.5, 0.3, 1.0], // Sand/brown color
                    scale: SAND_RADIUS * 2.0,
                });
            }
        }

        if !sediment_instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.sediment_instance_buffer,
                0,
                bytemuck::cast_slice(&sediment_instances),
            );
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.12,
                            b: 0.15,
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

            // Draw geometry
            render_pass.set_pipeline(&gpu.geometry_pipeline);
            render_pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, gpu.geometry_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                gpu.geometry_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..gpu.geometry_index_count, 0, 0..1);

            // Draw water particles
            if !water_instances.is_empty() {
                render_pass.set_pipeline(&gpu.particle_pipeline);
                render_pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, gpu.quad_vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, gpu.water_instance_buffer.slice(..));
                render_pass.draw(0..4, 0..water_instances.len() as u32);
            }

            // Draw sediment particles
            if !sediment_instances.is_empty() {
                render_pass.set_pipeline(&gpu.particle_pipeline);
                render_pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, gpu.quad_vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, gpu.sediment_instance_buffer.slice(..));
                render_pass.draw(0..4, 0..sediment_instances.len() as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

fn rand_simple(seed: f32) -> f32 {
    let x = (seed * 12.9898).sin() * 43758.5453;
    x - x.floor()
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Riffled Chute - Density Separation Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());
            pollster::block_on(self.init_gpu(window));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                            println!("{}", if self.paused { "Paused" } else { "Running" });
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => self.camera_angle -= 0.1,
                        PhysicalKey::Code(KeyCode::ArrowRight) => self.camera_angle += 0.1,
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.camera_distance = (self.camera_distance - 0.1).max(0.5)
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => self.camera_distance += 0.1,
                        PhysicalKey::Code(KeyCode::KeyR) => self.reset(),
                        PhysicalKey::Code(KeyCode::KeyS) => self.spawn_sediment(),
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.step();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

const GEOMETRY_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
}

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
    let ambient = 0.4;
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let brightness = ambient + (1.0 - ambient) * max(0.0, in.world_pos.y * 5.0);
    return vec4<f32>(in.color.rgb * brightness, in.color.a);
}
"#;

const PARTICLE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) quad_pos: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) scale: f32,
) -> VertexOutput {
    var out: VertexOutput;

    // Billboard: offset in camera space
    let to_camera = normalize(uniforms.camera_pos - position);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    let up = cross(to_camera, right);

    let world_pos = position + right * quad_pos.x * scale + up * quad_pos.y * scale;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = color;
    out.uv = quad_pos * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circular particle
    let dist = length(in.uv - vec2<f32>(0.5, 0.5));
    if dist > 0.5 {
        discard;
    }

    // Soft edge
    let alpha = in.color.a * smoothstep(0.5, 0.3, dist);
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
