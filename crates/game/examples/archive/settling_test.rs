//! Settling Test - Validates sediment separation using GpuFlip3D
//!
//! Tests that gold (heavy) settles upstream behind riffles while sand
//! (light) washes downstream.
//!
//! Uses the same framework as washplant_editor:
//! - GpuFlip3D for GPU-accelerated FLIP simulation
//! - ScreenSpaceFluidRenderer for water rendering
//! - SluicePiece geometry from editor module
//!
//! Run with: cargo run --example settling_test --release

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
const SIM_CELL_SIZE: f32 = 0.02; // 2cm cells
const SIM_MAX_PARTICLES: usize = 100_000;
const SIM_PRESSURE_ITERS: u32 = 40;
const SIM_GRAVITY: f32 = -9.8;

// Sluice configuration
const SLUICE_LENGTH: f32 = 1.2; // 1.2m long
const SLUICE_WIDTH: f32 = 0.3; // 30cm wide
const SLUICE_SLOPE: f32 = 8.0; // 8 degrees
const RIFFLE_SPACING: f32 = 0.12; // 12cm between riffles
const RIFFLE_HEIGHT: f32 = 0.02; // 2cm tall riffles

// Particle densities (relative to water = 1.0)
const WATER_DENSITY: f32 = 1.0;
const GOLD_DENSITY: f32 = 19.3; // Gold is 19.3x denser than water
const SAND_DENSITY: f32 = 2.65; // Sand is 2.65x denser than water

// Colors
const GOLD_COLOR: [f32; 4] = [1.0, 0.85, 0.1, 1.0];
const SAND_COLOR: [f32; 4] = [0.76, 0.7, 0.5, 1.0];

// Spawn counts
const NUM_GOLD: usize = 20;
const NUM_SAND: usize = 40;

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

    // Sluice definition
    sluice: SluicePiece,
    grid_offset: Vec3,
    grid_dims: (usize, usize, usize),

    // Simulation
    cpu_sim: FlipSimulation3D, // For SDF and solid cells
    gpu_flip: Option<GpuFlip3D>,
    fluid_renderer: Option<ScreenSpaceFluidRenderer>,

    // Particle data (GPU format)
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,

    // Track which particles are sediment (per-particle, same length as positions)
    is_sediment: Vec<bool>,
    is_gold: Vec<bool>, // Only meaningful if is_sediment[i] is true

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
        // Create sluice piece
        let sluice = SluicePiece {
            id: 0,
            position: Vec3::new(SLUICE_LENGTH / 2.0, 0.2, SLUICE_WIDTH / 2.0),
            rotation: Rotation::R0,
            length: SLUICE_LENGTH,
            width: SLUICE_WIDTH,
            slope_deg: SLUICE_SLOPE,
            riffle_spacing: RIFFLE_SPACING,
            riffle_height: RIFFLE_HEIGHT,
        };

        // Calculate grid dimensions
        let margin = SIM_CELL_SIZE * 4.0;
        let grid_width = ((sluice.length + margin * 2.0) / SIM_CELL_SIZE).ceil() as usize;
        let grid_height = ((0.5 + margin) / SIM_CELL_SIZE).ceil() as usize;
        let grid_depth = ((sluice.width + margin * 2.0) / SIM_CELL_SIZE).ceil() as usize;

        let grid_width = grid_width.clamp(20, 100);
        let grid_height = grid_height.clamp(10, 40);
        let grid_depth = grid_depth.clamp(10, 40);

        // Grid offset
        let grid_offset = Vec3::new(
            sluice.position.x - sluice.length / 2.0 - margin,
            sluice.position.y - margin,
            sluice.position.z - sluice.width / 2.0 - margin,
        );

        // Create CPU sim for SDF
        let mut cpu_sim = FlipSimulation3D::new(grid_width, grid_height, grid_depth, SIM_CELL_SIZE);
        cpu_sim.pressure_iterations = SIM_PRESSURE_ITERS as usize;

        // Mark sluice solid cells (same logic as washplant_editor)
        Self::mark_sluice_solid_cells(&mut cpu_sim, &sluice, SIM_CELL_SIZE, margin);
        cpu_sim.grid.compute_sdf();

        println!("=== Settling Test (GpuFlip3D) ===\n");
        println!(
            "Grid: {}x{}x{} cells ({:.2}m x {:.2}m x {:.2}m)",
            grid_width,
            grid_height,
            grid_depth,
            grid_width as f32 * SIM_CELL_SIZE,
            grid_height as f32 * SIM_CELL_SIZE,
            grid_depth as f32 * SIM_CELL_SIZE
        );
        println!(
            "Sluice: {:.1}m long, {:.1}° slope, {:.0}cm riffle spacing",
            SLUICE_LENGTH, SLUICE_SLOPE, RIFFLE_SPACING * 100.0
        );
        println!("\nDensities (relative to water):");
        println!("  Gold: {:.1}x", GOLD_DENSITY);
        println!("  Sand: {:.1}x", SAND_DENSITY);
        println!("\nControls: SPACE=pause, Arrows=camera, ESC=quit\n");

        Self {
            window: None,
            gpu: None,
            sluice,
            grid_offset,
            grid_dims: (grid_width, grid_height, grid_depth),
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

    fn mark_sluice_solid_cells(
        sim: &mut FlipSimulation3D,
        sluice: &SluicePiece,
        cell_size: f32,
        margin: f32,
    ) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Sluice is centered in local grid space
        let center_i = (margin + sluice.length / 2.0) / cell_size;
        let center_k = (margin + sluice.width / 2.0) / cell_size;

        let half_len_cells = ((sluice.length / 2.0) / cell_size).ceil() as i32;
        let half_wid_cells = ((sluice.width / 2.0) / cell_size).ceil() as i32;

        let slope_rad = sluice.slope_deg.to_radians();
        let total_drop = sluice.length * slope_rad.tan();

        let riffle_spacing_cells = (sluice.riffle_spacing / cell_size).round() as i32;
        let riffle_height_cells = (sluice.riffle_height / cell_size).ceil() as i32;
        let riffle_thick_cells = 2_i32;
        let wall_height_cells = 12_i32;

        let center_i = center_i.round() as i32;
        let center_k = center_k.round() as i32;

        for i in 0..width {
            let i_i = i as i32;

            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Floor height at this position
            let base_y = margin + sluice.position.y - sim.grid.depth as f32 * cell_size * 0.0;
            let mesh_floor_y = base_y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;

            // Staircase fix
            let t_next = ((t * (half_len_cells * 2) as f32 + 1.0) / (half_len_cells * 2) as f32)
                .clamp(0.0, 1.0);
            let floor_j_next =
                ((base_y + (total_drop / 2.0) - t_next * total_drop) / cell_size).floor() as i32;
            let effective_floor_j = floor_j.max(floor_j_next);

            let wall_top_j = effective_floor_j + riffle_height_cells + wall_height_cells;

            // Check if on riffle
            let dist_from_start = i_i - (center_i - half_len_cells);
            let is_riffle_x = if riffle_spacing_cells > 0 && dist_from_start > 4 {
                (dist_from_start % riffle_spacing_cells) < riffle_thick_cells
            } else {
                false
            };

            let i_start = (center_i - half_len_cells).max(0) as usize;
            let i_end = ((center_i + half_len_cells) as usize).min(width);
            let k_start = (center_k - half_wid_cells).max(0) as usize;
            let k_end = ((center_k + half_wid_cells) as usize).min(depth);

            for k in 0..depth {
                let k_i = k as i32;
                let in_channel_width = k >= k_start && k < k_end;
                let in_channel_length = i >= i_start && i < i_end;

                for j in 0..height {
                    let j_i = j as i32;

                    // Floor
                    let is_floor = j_i <= effective_floor_j && in_channel_length && in_channel_width;

                    // Riffles
                    let is_riffle = is_riffle_x
                        && in_channel_width
                        && in_channel_length
                        && j_i > effective_floor_j
                        && j_i <= effective_floor_j + riffle_height_cells;

                    // Side walls
                    let at_left_wall = k_i < (center_k - half_wid_cells);
                    let at_right_wall = k_i >= (center_k + half_wid_cells);
                    let is_side_wall =
                        (at_left_wall || at_right_wall) && in_channel_length && j_i <= wall_top_j;

                    if is_floor || is_riffle || is_side_wall {
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

        // Emit at inlet (upstream end of sluice)
        let margin = SIM_CELL_SIZE * 4.0;
        let emit_x = margin + SIM_CELL_SIZE * 3.0; // Near inlet
        let slope_rad = self.sluice.slope_deg.to_radians();
        let total_drop = self.sluice.length * slope_rad.tan();
        let inlet_floor_y = margin + total_drop / 2.0;
        let emit_y = inlet_floor_y + SIM_CELL_SIZE * 3.0;
        let center_z = margin + self.sluice.width / 2.0;
        let spread_z = self.sluice.width * 0.4;

        let init_vel = Vec3::new(0.4, -0.1, 0.0);

        // Emit 30 water particles per frame
        for _ in 0..30 {
            if self.positions.len() >= SIM_MAX_PARTICLES {
                break;
            }
            let x = emit_x + rand_float() * SIM_CELL_SIZE * 2.0;
            let y = emit_y + rand_float() * SIM_CELL_SIZE * 2.0;
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
        let emit_x = margin + SIM_CELL_SIZE * 8.0; // Slightly downstream of inlet
        let slope_rad = self.sluice.slope_deg.to_radians();
        let total_drop = self.sluice.length * slope_rad.tan();
        let inlet_floor_y = margin + total_drop / 2.0;
        let emit_y = inlet_floor_y + SIM_CELL_SIZE * 4.0;
        let center_z = margin + self.sluice.width / 2.0;

        let init_vel = Vec3::new(0.3, -0.05, 0.0);

        println!(
            "Spawning sediment at sim-local x={:.3}m, y={:.3}m",
            emit_x, emit_y
        );

        // Spawn gold particles
        for i in 0..NUM_GOLD {
            let z_offset = (i as f32 - NUM_GOLD as f32 / 2.0) * SIM_CELL_SIZE * 1.2;
            let pos = Vec3::new(
                emit_x + rand_float() * SIM_CELL_SIZE,
                emit_y + rand_float() * SIM_CELL_SIZE,
                center_z + z_offset,
            );

            self.positions.push(pos);
            self.velocities.push(init_vel);
            self.affine_vels.push(Mat3::ZERO);
            self.densities.push(GOLD_DENSITY);
            self.is_sediment.push(true);
            self.is_gold.push(true);
        }

        // Spawn sand particles
        for i in 0..NUM_SAND {
            let z_offset = (i as f32 - NUM_SAND as f32 / 2.0) * SIM_CELL_SIZE * 0.8;
            let pos = Vec3::new(
                emit_x + SIM_CELL_SIZE + rand_float() * SIM_CELL_SIZE,
                emit_y + SIM_CELL_SIZE + rand_float() * SIM_CELL_SIZE,
                center_z + z_offset,
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
            "Spawned {} gold (density {}) + {} sand (density {}) at frame {}",
            NUM_GOLD, GOLD_DENSITY, NUM_SAND, SAND_DENSITY, self.frame
        );
    }

    fn report_separation(&self) {
        if !self.sediment_spawned {
            return;
        }

        let mut gold_x_sum = 0.0f32;
        let mut gold_count = 0;
        let mut sand_x_sum = 0.0f32;
        let mut sand_count = 0;

        for i in 0..self.positions.len() {
            if self.is_sediment[i] {
                let x = self.positions[i].x;
                if self.is_gold[i] {
                    gold_x_sum += x;
                    gold_count += 1;
                } else {
                    sand_x_sum += x;
                    sand_count += 1;
                }
            }
        }

        let gold_mean_x = if gold_count > 0 {
            gold_x_sum / gold_count as f32
        } else {
            0.0
        };
        let sand_mean_x = if sand_count > 0 {
            sand_x_sum / sand_count as f32
        } else {
            0.0
        };

        let sluice_length = self.grid_dims.0 as f32 * SIM_CELL_SIZE;
        let separation = (sand_mean_x - gold_mean_x) / sluice_length * 100.0;

        println!(
            "Frame {}: particles={}, gold={}/{}, sand={}/{}, gold_x={:.3}m, sand_x={:.3}m, sep={:.1}%",
            self.frame,
            self.positions.len(),
            gold_count, NUM_GOLD,
            sand_count, NUM_SAND,
            gold_mean_x,
            sand_mean_x,
            separation
        );
    }

    fn step(&mut self) {
        if self.paused {
            return;
        }

        if self.gpu.is_none() {
            return;
        }

        // Emit water (mutable operations first)
        self.emit_water();

        // Spawn sediment after water flow establishes
        if self.frame == 180 && !self.sediment_spawned {
            self.spawn_sediment();
        }

        // Create cell types
        let (gw, gh, gd) = self.grid_dims;
        let cell_count = gw * gh * gd;
        let mut cell_types = vec![0u32; cell_count];

        // Mark particle cells as fluid
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
                cell_types[idx] = 1; // FLUID
            }
        }

        let sdf = Some(self.cpu_sim.grid.sdf.clone());
        let dt = 1.0 / 60.0;

        // Now borrow gpu for the simulation step
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

        // Remove particles that exit the grid
        let exit_margin = SIM_CELL_SIZE * 10.0;
        let max_x = (gw as f32) * SIM_CELL_SIZE + exit_margin;
        let max_y = (gh as f32) * SIM_CELL_SIZE + exit_margin;
        let max_z = (gd as f32) * SIM_CELL_SIZE + exit_margin;
        let min_bound = -exit_margin;

        let mut i = 0;
        while i < self.positions.len() {
            let pos = self.positions[i];
            let out_of_bounds = pos.x < min_bound
                || pos.x > max_x
                || pos.y < min_bound
                || pos.y > max_y
                || pos.z < min_bound
                || pos.z > max_z;

            if out_of_bounds {
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

        // Report every 2 seconds
        if self.frame % 120 == 0 {
            self.report_separation();
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

        // Depth texture
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

        // Build sluice mesh
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

        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Sluice pipeline
        let sluice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sluice Shader"),
            source: wgpu::ShaderSource::Wgsl(SLUICE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
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

        // Sediment billboard pipeline
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

        // Sediment instance buffer
        let sediment_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Instances"),
            size: ((NUM_GOLD + NUM_SAND) * 10 * std::mem::size_of::<SedimentInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create GpuFlip3D
        let (gw, gh, gd) = self.grid_dims;
        let mut gpu_flip = GpuFlip3D::new(
            &device,
            gw as u32,
            gh as u32,
            gd as u32,
            SIM_CELL_SIZE,
            SIM_MAX_PARTICLES,
        );
        // Open outlet boundary
        gpu_flip.open_boundaries = 2; // +X open
        // Enable sediment drag based on density
        gpu_flip.sediment_drag_coefficient = 8.0; // Moderate drag
        gpu_flip.gold_density_threshold = 5.0; // Particles > 5x water density are "heavy"

        // Create fluid renderer
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
        let center_x = margin + self.sluice.length / 2.0;
        let center_z = margin + self.sluice.width / 2.0;
        let half_len = self.sluice.length / 2.0;
        let half_wid = self.sluice.width / 2.0;

        let slope_rad = self.sluice.slope_deg.to_radians();
        let total_drop = self.sluice.length * slope_rad.tan();
        let base_y = margin;

        let color = [0.35, 0.3, 0.25, 1.0]; // Brown sluice

        // Floor quad
        let inlet_y = base_y + total_drop / 2.0;
        let outlet_y = base_y - total_drop / 2.0;

        let v_base = vertices.len() as u32;
        vertices.push(SluiceVertex {
            position: [center_x - half_len, inlet_y, center_z - half_wid],
            color,
        });
        vertices.push(SluiceVertex {
            position: [center_x + half_len, outlet_y, center_z - half_wid],
            color,
        });
        vertices.push(SluiceVertex {
            position: [center_x + half_len, outlet_y, center_z + half_wid],
            color,
        });
        vertices.push(SluiceVertex {
            position: [center_x - half_len, inlet_y, center_z + half_wid],
            color,
        });
        indices.extend_from_slice(&[v_base, v_base + 1, v_base + 2, v_base, v_base + 2, v_base + 3]);

        // Riffles
        let riffle_color = [0.25, 0.2, 0.15, 1.0];
        let mut x = center_x - half_len + self.sluice.riffle_spacing;
        while x < center_x + half_len - 0.05 {
            let t = (x - (center_x - half_len)) / self.sluice.length;
            let floor_y = inlet_y - t * total_drop;
            let riffle_top = floor_y + self.sluice.riffle_height;

            let v_base = vertices.len() as u32;
            let thick = 0.015;
            vertices.push(SluiceVertex {
                position: [x - thick, floor_y, center_z - half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, floor_y, center_z - half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, riffle_top, center_z - half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x - thick, riffle_top, center_z - half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x - thick, floor_y, center_z + half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, floor_y, center_z + half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x + thick, riffle_top, center_z + half_wid],
                color: riffle_color,
            });
            vertices.push(SluiceVertex {
                position: [x - thick, riffle_top, center_z + half_wid],
                color: riffle_color,
            });

            // Front face
            indices.extend_from_slice(&[v_base, v_base + 1, v_base + 2, v_base, v_base + 2, v_base + 3]);
            // Back face
            indices.extend_from_slice(&[v_base + 4, v_base + 6, v_base + 5, v_base + 4, v_base + 7, v_base + 6]);
            // Top face
            indices.extend_from_slice(&[v_base + 3, v_base + 2, v_base + 6, v_base + 3, v_base + 6, v_base + 7]);

            x += self.sluice.riffle_spacing;
        }

        // Side walls
        let wall_height = 0.08;
        let wall_color = [0.4, 0.35, 0.3, 0.8];

        // Left wall
        let v_base = vertices.len() as u32;
        vertices.push(SluiceVertex {
            position: [center_x - half_len, inlet_y, center_z - half_wid],
            color: wall_color,
        });
        vertices.push(SluiceVertex {
            position: [center_x + half_len, outlet_y, center_z - half_wid],
            color: wall_color,
        });
        vertices.push(SluiceVertex {
            position: [center_x + half_len, outlet_y + wall_height, center_z - half_wid],
            color: wall_color,
        });
        vertices.push(SluiceVertex {
            position: [center_x - half_len, inlet_y + wall_height, center_z - half_wid],
            color: wall_color,
        });
        indices.extend_from_slice(&[v_base, v_base + 1, v_base + 2, v_base, v_base + 2, v_base + 3]);

        // Right wall
        let v_base = vertices.len() as u32;
        vertices.push(SluiceVertex {
            position: [center_x - half_len, inlet_y, center_z + half_wid],
            color: wall_color,
        });
        vertices.push(SluiceVertex {
            position: [center_x + half_len, outlet_y, center_z + half_wid],
            color: wall_color,
        });
        vertices.push(SluiceVertex {
            position: [center_x + half_len, outlet_y + wall_height, center_z + half_wid],
            color: wall_color,
        });
        vertices.push(SluiceVertex {
            position: [center_x - half_len, inlet_y + wall_height, center_z + half_wid],
            color: wall_color,
        });
        indices.extend_from_slice(&[v_base, v_base + 2, v_base + 1, v_base, v_base + 3, v_base + 2]);

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

        // Camera
        let margin = SIM_CELL_SIZE * 4.0;
        let center = Vec3::new(
            margin + self.sluice.length / 2.0,
            0.15,
            margin + self.sluice.width / 2.0,
        );
        let camera_pos = center
            + Vec3::new(
                self.camera_distance * self.camera_angle.cos(),
                self.camera_height,
                self.camera_distance * self.camera_angle.sin(),
            );

        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let view_mat = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
        let proj_mat = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let view_proj = proj_mat * view_mat;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Build sediment instances
        let mut instances: Vec<SedimentInstance> = Vec::new();
        for i in 0..self.positions.len() {
            if self.is_sediment[i] {
                let pos = self.positions[i];
                let (color, radius) = if self.is_gold[i] {
                    (GOLD_COLOR, 0.008) // 8mm gold
                } else {
                    (SAND_COLOR, 0.006) // 6mm sand
                };
                instances.push(SedimentInstance {
                    position: pos.to_array(),
                    scale: radius * 2.0,
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

            // Draw sluice
            pass.set_pipeline(&gpu.sluice_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.sluice_vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.sluice_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.sluice_index_count, 0, 0..1);

            // Draw sediment
            if !instances.is_empty() {
                pass.set_pipeline(&gpu.sediment_pipeline);
                pass.set_vertex_buffer(0, gpu.sediment_instance_buffer.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }
        }

        // Render water using ScreenSpaceFluidRenderer
        if let (Some(fluid_renderer), Some(gpu_flip)) = (&self.fluid_renderer, &self.gpu_flip) {
            let view_mat = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
            fluid_renderer.render(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &view,
                gpu_flip,
                self.positions.len() as u32,
                view_mat.to_cols_array_2d(),
                proj_mat.to_cols_array_2d(),
                camera_pos.to_array(),
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
                .with_title("Settling Test (GpuFlip3D)")
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
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const SEDIMENT_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) idx: u32,
    @location(0) center: vec3<f32>,
    @location(1) scale: f32,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    let quad = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
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
