//! Sediment Test - Tests gold/sand separation using washplant_editor framework
//!
//! Uses the same MultiGridSim infrastructure as washplant_editor to test
//! density-based settling in a sluice.
//!
//! Run with: cargo run --example sediment_test --release
//!
//! Expected behavior:
//! - Gold (19.3x density) should settle upstream behind riffles
//! - Sand (2.65x density) should wash further downstream
//! - Positive separation percentage = correct physics

use bytemuck::{Pod, Zeroable};
use game::editor::{Rotation, SluicePiece};
use game::gpu::flip_3d::GpuFlip3D;
use game::gpu::fluid_renderer::ScreenSpaceFluidRenderer;
use game::sluice_geometry::SluiceVertex;
use glam::{Mat3, Mat4, Vec3};
use sim3d::FlipSimulation3D;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Simulation constants (matching washplant_editor)
const SIM_CELL_SIZE: f32 = 0.025;
const SIM_MAX_PARTICLES: usize = 50_000;
const SIM_PRESSURE_ITERS: u32 = 30;
const SIM_GRAVITY: f32 = -9.8;

// Densities
const WATER_DENSITY: f32 = 1.0;
const GOLD_DENSITY: f32 = 19.3;
const SAND_DENSITY: f32 = 2.65;

// Colors
const GOLD_COLOR: [f32; 4] = [1.0, 0.85, 0.1, 1.0];
const SAND_COLOR: [f32; 4] = [0.76, 0.7, 0.5, 1.0];

const NUM_GOLD: usize = 30;
const NUM_SAND: usize = 60;

fn rand_float() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);
    (x as f32) / (u64::MAX as f32)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SedimentInstance {
    position: [f32; 3],
    scale: f32,
    color: [f32; 4],
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    sluice_pipeline: wgpu::RenderPipeline,
    sediment_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    sluice_vertex_buffer: wgpu::Buffer,
    sluice_index_buffer: wgpu::Buffer,
    sluice_index_count: u32,
    sediment_instance_buffer: wgpu::Buffer,
    depth_texture: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,

    // Sluice config
    sluice: SluicePiece,
    grid_offset: Vec3,
    grid_dims: (usize, usize, usize),

    // Simulation (matching washplant_editor pattern)
    cpu_sim: FlipSimulation3D,
    gpu_flip: Option<GpuFlip3D>,
    fluid_renderer: Option<ScreenSpaceFluidRenderer>,

    // Particle data
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,

    // Per-particle tracking (same length as positions)
    is_sediment: Vec<bool>,
    is_gold: Vec<bool>,

    // State
    paused: bool,
    frame: u32,
    camera_angle: f32,
    camera_distance: f32,
    camera_height: f32,
    sediment_spawned: bool,
}

impl App {
    fn new() -> Self {
        let sluice = SluicePiece {
            id: 0,
            position: Vec3::new(0.6, 0.2, 0.15),
            rotation: Rotation::R0,
            length: 1.2,
            width: 0.3,
            slope_deg: 8.0,
            riffle_spacing: 0.12,
            riffle_height: 0.02,
        };

        let margin = SIM_CELL_SIZE * 4.0;
        let gw = ((sluice.length + margin * 2.0) / SIM_CELL_SIZE).ceil() as usize;
        let gh = ((0.5 + margin) / SIM_CELL_SIZE).ceil() as usize;
        let gd = ((sluice.width + margin * 2.0) / SIM_CELL_SIZE).ceil() as usize;

        let gw = gw.clamp(20, 80);
        let gh = gh.clamp(10, 40);
        let gd = gd.clamp(10, 40);

        let grid_offset = Vec3::new(
            sluice.position.x - sluice.length / 2.0 - margin,
            sluice.position.y - margin,
            sluice.position.z - sluice.width / 2.0 - margin,
        );

        let mut cpu_sim = FlipSimulation3D::new(gw, gh, gd, SIM_CELL_SIZE);
        cpu_sim.pressure_iterations = SIM_PRESSURE_ITERS as usize;

        Self::mark_sluice_solid(&mut cpu_sim, &sluice, SIM_CELL_SIZE, margin);
        cpu_sim.grid.compute_sdf();

        println!("=== Sediment Test (Washplant Framework) ===\n");
        println!("Grid: {}x{}x{} cells", gw, gh, gd);
        println!(
            "Sluice: {:.1}m @ {:.0}° slope",
            sluice.length, sluice.slope_deg
        );
        println!(
            "\nDensities: Gold={:.1}x, Sand={:.1}x",
            GOLD_DENSITY, SAND_DENSITY
        );
        println!("\nControls: SPACE=pause, Arrows=camera, ESC=quit\n");

        Self {
            window: None,
            gpu: None,
            sluice,
            grid_offset,
            grid_dims: (gw, gh, gd),
            cpu_sim,
            gpu_flip: None,
            fluid_renderer: None,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
            is_sediment: Vec::new(),
            is_gold: Vec::new(),
            paused: false,
            frame: 0,
            camera_angle: 0.3,
            camera_distance: 1.5,
            camera_height: 0.6,
            sediment_spawned: false,
        }
    }

    fn mark_sluice_solid(
        sim: &mut FlipSimulation3D,
        sluice: &SluicePiece,
        cell_size: f32,
        margin: f32,
    ) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        let center_i = (margin + sluice.length / 2.0) / cell_size;
        let center_k = (margin + sluice.width / 2.0) / cell_size;
        let half_len = ((sluice.length / 2.0) / cell_size).ceil() as i32;
        let half_wid = ((sluice.width / 2.0) / cell_size).ceil() as i32;

        let slope_rad = sluice.slope_deg.to_radians();
        let total_drop = sluice.length * slope_rad.tan();
        let riffle_cells = (sluice.riffle_spacing / cell_size).round() as i32;
        let riffle_h = (sluice.riffle_height / cell_size).ceil() as i32;
        let wall_h = 12_i32;

        let ci = center_i.round() as i32;
        let ck = center_k.round() as i32;

        for i in 0..width {
            let ii = i as i32;
            let t = if ii <= ci - half_len {
                0.0
            } else if ii >= ci + half_len {
                1.0
            } else {
                ((ii - (ci - half_len)) as f32) / ((half_len * 2) as f32).max(1.0)
            };

            let base_y = margin + sluice.position.y;
            let floor_y = base_y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (floor_y / cell_size).floor() as i32;

            let t_next =
                ((t * (half_len * 2) as f32 + 1.0) / (half_len * 2) as f32).clamp(0.0, 1.0);
            let floor_j_next =
                ((base_y + (total_drop / 2.0) - t_next * total_drop) / cell_size).floor() as i32;
            let eff_floor = floor_j.max(floor_j_next);

            let wall_top = eff_floor + riffle_h + wall_h;
            let dist = ii - (ci - half_len);
            let is_riffle_x = riffle_cells > 0 && dist > 4 && (dist % riffle_cells) < 2;

            let i_start = (ci - half_len).max(0) as usize;
            let i_end = ((ci + half_len) as usize).min(width);
            let k_start = (ck - half_wid).max(0) as usize;
            let k_end = ((ck + half_wid) as usize).min(depth);

            for k in 0..depth {
                let kk = k as i32;
                let in_width = k >= k_start && k < k_end;
                let in_len = i >= i_start && i < i_end;

                for j in 0..height {
                    let jj = j as i32;
                    let is_floor = jj <= eff_floor && in_len && in_width;
                    let is_riffle = is_riffle_x
                        && in_width
                        && in_len
                        && jj > eff_floor
                        && jj <= eff_floor + riffle_h;
                    let at_wall = kk < (ck - half_wid) || kk >= (ck + half_wid);
                    let is_wall = at_wall && in_len && jj <= wall_top;

                    if is_floor || is_riffle || is_wall {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }

    fn emit_water(&mut self) {
        if self.positions.len() >= SIM_MAX_PARTICLES {
            return;
        }

        let margin = SIM_CELL_SIZE * 4.0;
        let emit_x = margin + SIM_CELL_SIZE * 3.0;
        let slope_rad = self.sluice.slope_deg.to_radians();
        let total_drop = self.sluice.length * slope_rad.tan();
        let inlet_y = margin + total_drop / 2.0 + SIM_CELL_SIZE * 3.0;
        let center_z = margin + self.sluice.width / 2.0;
        let spread_z = self.sluice.width * 0.4;
        let init_vel = Vec3::new(0.4, -0.1, 0.0);

        for _ in 0..30 {
            if self.positions.len() >= SIM_MAX_PARTICLES {
                break;
            }
            let x = emit_x + rand_float() * SIM_CELL_SIZE * 2.0;
            let y = inlet_y + rand_float() * SIM_CELL_SIZE * 2.0;
            let z = center_z + (rand_float() - 0.5) * spread_z;

            self.positions.push(Vec3::new(x, y, z));
            self.velocities.push(init_vel);
            self.affine_vels.push(Mat3::ZERO);
            self.densities.push(WATER_DENSITY);
            self.is_sediment.push(false);
            self.is_gold.push(false);
        }
    }

    fn spawn_sediment(&mut self) {
        let margin = SIM_CELL_SIZE * 4.0;
        let emit_x = margin + SIM_CELL_SIZE * 8.0;
        let slope_rad = self.sluice.slope_deg.to_radians();
        let total_drop = self.sluice.length * slope_rad.tan();
        let emit_y = margin + total_drop / 2.0 + SIM_CELL_SIZE * 5.0;
        let center_z = margin + self.sluice.width / 2.0;
        let init_vel = Vec3::new(0.3, -0.05, 0.0);

        println!("Spawning sediment at x={:.3}m, y={:.3}m", emit_x, emit_y);

        // Gold
        for i in 0..NUM_GOLD {
            let z_off = (i as f32 - NUM_GOLD as f32 / 2.0) * SIM_CELL_SIZE * 1.2;
            let pos = Vec3::new(
                emit_x + rand_float() * SIM_CELL_SIZE,
                emit_y + rand_float() * SIM_CELL_SIZE,
                center_z + z_off,
            );
            self.positions.push(pos);
            self.velocities.push(init_vel);
            self.affine_vels.push(Mat3::ZERO);
            self.densities.push(GOLD_DENSITY);
            self.is_sediment.push(true);
            self.is_gold.push(true);
        }

        // Sand
        for i in 0..NUM_SAND {
            let z_off = (i as f32 - NUM_SAND as f32 / 2.0) * SIM_CELL_SIZE * 0.8;
            let pos = Vec3::new(
                emit_x + SIM_CELL_SIZE + rand_float() * SIM_CELL_SIZE,
                emit_y + SIM_CELL_SIZE + rand_float() * SIM_CELL_SIZE,
                center_z + z_off,
            );
            self.positions.push(pos);
            self.velocities.push(init_vel);
            self.affine_vels.push(Mat3::ZERO);
            self.densities.push(SAND_DENSITY);
            self.is_sediment.push(true);
            self.is_gold.push(false);
        }

        self.sediment_spawned = true;
        println!(
            "Spawned {} gold + {} sand at frame {}",
            NUM_GOLD, NUM_SAND, self.frame
        );
    }

    fn report(&self) {
        if !self.sediment_spawned {
            return;
        }

        let mut gold_x = 0.0f32;
        let mut gold_n = 0;
        let mut sand_x = 0.0f32;
        let mut sand_n = 0;

        for i in 0..self.positions.len() {
            if self.is_sediment[i] {
                let x = self.positions[i].x;
                if self.is_gold[i] {
                    gold_x += x;
                    gold_n += 1;
                } else {
                    sand_x += x;
                    sand_n += 1;
                }
            }
        }

        let gold_avg = if gold_n > 0 {
            gold_x / gold_n as f32
        } else {
            0.0
        };
        let sand_avg = if sand_n > 0 {
            sand_x / sand_n as f32
        } else {
            0.0
        };
        let sluice_len = self.grid_dims.0 as f32 * SIM_CELL_SIZE;
        let sep = (sand_avg - gold_avg) / sluice_len * 100.0;

        println!(
            "Frame {}: n={}, gold={}/{} x={:.3}m, sand={}/{} x={:.3}m, sep={:.1}%",
            self.frame,
            self.positions.len(),
            gold_n,
            NUM_GOLD,
            gold_avg,
            sand_n,
            NUM_SAND,
            sand_avg,
            sep
        );
    }

    fn step(&mut self) {
        if self.paused || self.gpu.is_none() {
            return;
        }

        self.emit_water();

        if self.frame == 180 && !self.sediment_spawned {
            self.spawn_sediment();
        }

        let (gw, gh, gd) = self.grid_dims;
        let cell_count = gw * gh * gd;
        let mut cell_types = vec![0u32; cell_count];

        for pos in &self.positions {
            let i = (pos.x / SIM_CELL_SIZE).floor() as isize;
            let j = (pos.y / SIM_CELL_SIZE).floor() as isize;
            let k = (pos.z / SIM_CELL_SIZE).floor() as isize;
            if i >= 0
                && (i as usize) < gw
                && j >= 0
                && (j as usize) < gh
                && k >= 0
                && (k as usize) < gd
            {
                let idx = (k as usize) * gw * gh + (j as usize) * gw + i as usize;
                cell_types[idx] = 1;
            }
        }

        let sdf = Some(self.cpu_sim.grid.sdf.clone());
        let dt = 1.0 / 60.0;

        let gpu = self.gpu.as_ref().unwrap();
        if let Some(gpu_flip) = &mut self.gpu_flip {
            gpu_flip.step(
                &gpu.device,
                &gpu.queue,
                &mut self.positions,
                &mut self.velocities,
                &mut self.affine_vels,
                &self.densities,
                &cell_types,
                sdf.as_deref(),
                None,
                dt,
                SIM_GRAVITY,
                0.0,
                SIM_PRESSURE_ITERS,
            );
        }

        // Remove out-of-bounds particles
        let exit_margin = SIM_CELL_SIZE * 10.0;
        let max_x = (gw as f32) * SIM_CELL_SIZE + exit_margin;
        let max_y = (gh as f32) * SIM_CELL_SIZE + exit_margin;
        let max_z = (gd as f32) * SIM_CELL_SIZE + exit_margin;
        let min_b = -exit_margin;

        let mut i = 0;
        while i < self.positions.len() {
            let p = self.positions[i];
            if p.x < min_b
                || p.x > max_x
                || p.y < min_b
                || p.y > max_y
                || p.z < min_b
                || p.z > max_z
            {
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.affine_vels.swap_remove(i);
                self.densities.swap_remove(i);
                self.is_sediment.swap_remove(i);
                self.is_gold.swap_remove(i);
            } else {
                i += 1;
            }
        }

        self.frame += 1;
        if self.frame % 120 == 0 {
            self.report();
        }
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..Default::default()
                    },
                    memory_hints: wgpu::MemoryHints::default(),
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
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_format = wgpu::TextureFormat::Depth32Float;
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
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (vertices, indices) = self.build_sluice_mesh();
        let sluice_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sluice_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let sluice_index_count = indices.len() as u32;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let sluice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sluice Shader"),
            source: wgpu::ShaderSource::Wgsl(SLUICE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });

        let sluice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sluice Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sluice_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SluiceVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sluice_shader,
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sediment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Shader"),
            source: wgpu::ShaderSource::Wgsl(SEDIMENT_SHADER.into()),
        });

        let sediment_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sediment Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sediment_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SedimentInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sediment_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sediment_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Instances"),
            size: ((NUM_GOLD + NUM_SAND) * 10 * std::mem::size_of::<SedimentInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (gw, gh, gd) = self.grid_dims;
        let mut gpu_flip = GpuFlip3D::new(
            &device,
            gw as u32,
            gh as u32,
            gd as u32,
            SIM_CELL_SIZE,
            SIM_MAX_PARTICLES,
        );
        gpu_flip.open_boundaries = 2;
        gpu_flip.sediment_drag_coefficient = 8.0;
        gpu_flip.gold_density_threshold = 5.0;

        let mut fluid_renderer = ScreenSpaceFluidRenderer::new(&device, format);
        fluid_renderer.particle_radius = SIM_CELL_SIZE * 0.5;
        fluid_renderer.resize(&device, config.width, config.height);

        self.gpu_flip = Some(gpu_flip);
        self.fluid_renderer = Some(fluid_renderer);

        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            config,
            sluice_pipeline,
            sediment_pipeline,
            uniform_buffer,
            uniform_bind_group,
            sluice_vertex_buffer,
            sluice_index_buffer,
            sluice_index_count,
            sediment_instance_buffer,
            depth_texture: depth_view,
        });
    }

    fn build_sluice_mesh(&self) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let margin = SIM_CELL_SIZE * 4.0;
        let cx = margin + self.sluice.length / 2.0;
        let cz = margin + self.sluice.width / 2.0;
        let hl = self.sluice.length / 2.0;
        let hw = self.sluice.width / 2.0;

        let slope_rad = self.sluice.slope_deg.to_radians();
        let drop = self.sluice.length * slope_rad.tan();
        let base_y = margin;
        let color = [0.35, 0.3, 0.25, 1.0];

        let inlet_y = base_y + drop / 2.0;
        let outlet_y = base_y - drop / 2.0;

        let v0 = vertices.len() as u32;
        vertices.push(SluiceVertex {
            position: [cx - hl, inlet_y, cz - hw],
            color,
        });
        vertices.push(SluiceVertex {
            position: [cx + hl, outlet_y, cz - hw],
            color,
        });
        vertices.push(SluiceVertex {
            position: [cx + hl, outlet_y, cz + hw],
            color,
        });
        vertices.push(SluiceVertex {
            position: [cx - hl, inlet_y, cz + hw],
            color,
        });
        indices.extend_from_slice(&[v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3]);

        let riffle_color = [0.25, 0.2, 0.15, 1.0];
        let mut x = cx - hl + self.sluice.riffle_spacing;
        while x < cx + hl - 0.05 {
            let t = (x - (cx - hl)) / self.sluice.length;
            let floor_y = inlet_y - t * drop;
            let top = floor_y + self.sluice.riffle_height;
            let thick = 0.015;

            let v = vertices.len() as u32;
            vertices.push(SluiceVertex {
                position: [x - thick, floor_y, cz - hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, floor_y, cz - hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, top, cz - hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x - thick, top, cz - hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x - thick, floor_y, cz + hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, floor_y, cz + hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, top, cz + hw],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x - thick, top, cz + hw],
                color: riffle_color,
            });

            indices.extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
            indices.extend_from_slice(&[v + 4, v + 6, v + 5, v + 4, v + 7, v + 6]);
            indices.extend_from_slice(&[v + 3, v + 2, v + 6, v + 3, v + 6, v + 7]);

            x += self.sluice.riffle_spacing;
        }

        (vertices, indices)
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(window) = &self.window else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let margin = SIM_CELL_SIZE * 4.0;
        let center = Vec3::new(
            margin + self.sluice.length / 2.0,
            0.15,
            margin + self.sluice.width / 2.0,
        );
        let cam_pos = center
            + Vec3::new(
                self.camera_distance * self.camera_angle.cos(),
                self.camera_height,
                self.camera_distance * self.camera_angle.sin(),
            );

        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let view_mat = Mat4::look_at_rh(cam_pos, center, Vec3::Y);
        let proj_mat = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let vp = proj_mat * view_mat;

        let uniforms = Uniforms {
            view_proj: vp.to_cols_array_2d(),
            camera_pos: cam_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut instances: Vec<SedimentInstance> = Vec::new();
        for i in 0..self.positions.len() {
            if self.is_sediment[i] {
                let pos = self.positions[i];
                let (color, r) = if self.is_gold[i] {
                    (GOLD_COLOR, 0.008)
                } else {
                    (SAND_COLOR, 0.006)
                };
                instances.push(SedimentInstance {
                    position: pos.to_array(),
                    scale: r * 2.0,
                    color,
                });
            }
        }

        if !instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.sediment_instance_buffer,
                0,
                bytemuck::cast_slice(&instances),
            );
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
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
                    view: &gpu.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&gpu.sluice_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.sluice_vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.sluice_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.sluice_index_count, 0, 0..1);

            if !instances.is_empty() {
                pass.set_pipeline(&gpu.sediment_pipeline);
                pass.set_vertex_buffer(0, gpu.sediment_instance_buffer.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }
        }

        if let (Some(fluid_renderer), Some(gpu_flip)) = (&self.fluid_renderer, &self.gpu_flip) {
            fluid_renderer.render(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &view,
                gpu_flip,
                self.positions.len() as u32,
                view_mat.to_cols_array_2d(),
                proj_mat.to_cols_array_2d(),
                cam_pos.to_array(),
                gpu.config.width,
                gpu.config.height,
            );
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Sediment Test (Washplant Framework)")
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
                            self.camera_distance = (self.camera_distance - 0.1).max(0.3)
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => self.camera_distance += 0.1,
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.step();
                self.render();
            }
            _ => {}
        }
    }
}

const SLUICE_SHADER: &str = r#"
struct Uniforms { view_proj: mat4x4<f32>, camera_pos: vec3<f32>, _pad: f32 }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }
@vertex fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> { return in.color; }
"#;

const SEDIMENT_SHADER: &str = r#"
struct Uniforms { view_proj: mat4x4<f32>, camera_pos: vec3<f32>, _pad: f32 }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32>, @location(1) uv: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) idx: u32, @location(0) center: vec3<f32>, @location(1) scale: f32, @location(2) color: vec4<f32>) -> VertexOutput {
    let quad = array<vec2<f32>, 4>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    let uv = quad[idx];
    let view_dir = normalize(uniforms.camera_pos - center);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), view_dir));
    let up = cross(view_dir, right);
    let world_pos = center + right * uv.x * scale + up * uv.y * scale;
    var out: VertexOutput;
    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = color;
    out.uv = uv;
    return out;
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if dist > 1.0 { discard; }
    return vec4<f32>(in.color.rgb * (1.0 - dist * 0.3), in.color.a);
}
"#;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
