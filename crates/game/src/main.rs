//! Goldrush 3D Simulation
//!
//! Entry point for the 3D FLIP+DEM simulation.
//!
//! Run with: cargo run --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder, SluiceVertex};
use game::water_heightfield::{WaterHeightfieldRenderer, WaterRenderConfig, WaterVertex};
use glam::{Mat3, Mat4, Vec3};
use sim3d::{constants, ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FlipSimulation3D, SdfParams};
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

// Grid configuration
const CELL_SIZE: f32 = 0.01;
const SLUICE_WIDTH: usize = 150; // 1.50 m
const EXIT_BUFFER: usize = 12;
const GRID_WIDTH: usize = SLUICE_WIDTH + EXIT_BUFFER;
const GRID_DEPTH: usize = 40; // 0.40 m
const GRID_HEIGHT: usize = 52; // 0.52 m
const MAX_PARTICLES: usize = 300_000;

// Simulation
const GRAVITY: f32 = constants::GRAVITY;
const PRESSURE_ITERS: u32 = 120;
const SUBSTEPS: u32 = 2;
const TRACER_INTERVAL_FRAMES: u32 = 300; // 5s at 60 FPS
const TRACER_COUNT: u32 = 3;

// Emission rates (10-20% solids by mass)
const WATER_EMIT_RATE: usize = 500; // Normal rate
const SEDIMENT_EMIT_RATE: usize = 2;
const GPU_SYNC_STRIDE: u32 = 4; // GPU readback cadence (frames)

// Grain sizing (relative to cell size)
const GANGUE_RADIUS_CELLS: f32 = 0.12; // Coarse gangue grains
const GOLD_RADIUS_CELLS: f32 = 0.02; // Fine gold grains
const GANGUE_DENSITY: f32 = constants::GANGUE_DENSITY;
const GOLD_DENSITY: f32 = constants::GOLD_DENSITY;
const GOLD_FRACTION: f32 = 0.05; // 5% of sediment spawns as gold

// Sediment colors
const GANGUE_COLOR: [f32; 4] = constants::GANGUE_COLOR;
const GOLD_COLOR: [f32; 4] = constants::GOLD_COLOR;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

use game::example_utils::{Camera, MeshVertex, Pos3Color4Vertex, WgpuContext, SEDIMENT_SHADER, BASIC_SHADER, build_rock_mesh, create_depth_view};

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
    rotation: [f32; 4],
    color: [f32; 4],
}

struct GpuState {
    ctx: WgpuContext,

    // Pipelines
    sluice_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    sediment_pipeline: wgpu::RenderPipeline,

    // Buffers
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    sluice_vertex_buffer: wgpu::Buffer,
    sluice_index_buffer: wgpu::Buffer,
    water_vertex_buffer: wgpu::Buffer,
    rock_mesh_vertex_buffer: wgpu::Buffer,
    rock_mesh_vertex_count: u32,
    sediment_instance_buffer: wgpu::Buffer,

    // Depth
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    gpu_flip: Option<GpuFlip3D>,
    sim: FlipSimulation3D,
    sluice_builder: SluiceGeometryBuilder,
    water_renderer: WaterHeightfieldRenderer,
    dem: ClusterSimulation3D,
    gangue_template_idx: usize,
    gold_template_idx: usize,

    // Persistent FLIP<->DEM mapping
    // sediment_clump_idx[i] = index into dem.clumps for sediment particle i
    // We track sediment particles separately from water
    sediment_flip_indices: Vec<usize>, // FLIP particle indices that are sediment
    tracer_particles: Vec<TracerInfo>,

    // Particle data for GPU transfer
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,

    // State
    paused: bool,
    frame: u32,
    camera: Camera,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    water_emit_rate: usize,
    sediment_emit_rate: usize,
    use_dem: bool, // Toggle DEM on/off
    use_async_readback: bool,
    gpu_readback_pending: bool,
    gpu_sync_substep: u32,
    gpu_needs_upload: bool,
    pending_water_emits: usize,
    pending_sediment_emits: usize,

    // Timing
    start_time: Instant,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
}

struct TracerInfo {
    index: usize,
    spawn_frame: u32,
}
struct FlowMetrics {
    sample_count: usize,
    vel_mean: f32,
    depth_p50: f32,
    depth_p90: f32,
    flow_width: f32,
    flow_rate_m3s: f32,
    flow_rate_m3min: f32,
    sample_x_min: f32,
    sample_x_max: f32,
}

impl App {
    fn new() -> Self {
        // Configure sluice geometry - smooth ramp feed, then riffles
        // Sluice ends before the grid boundary, leaving buffer zone for clean outflow
        let sluice_config = SluiceConfig {
            grid_width: SLUICE_WIDTH, // Sluice width, not full grid width
            grid_height: GRID_HEIGHT,
            grid_depth: GRID_DEPTH,
            cell_size: CELL_SIZE,
            // slope: 10 deg (drop 26 cells over 150)
            floor_height_left: 30,
            floor_height_right: 4,
            // riffles: spacing 0.32 m, height 0.03 m
            riffle_spacing: 32,
            riffle_height: 3,
            riffle_thickness: 2,
            riffle_start_x: 40, // 0.40 m slick plate
            riffle_end_pad: 12, // 0.12 m tail clearance
            // wall height above floor+riffle: (4 + 8) * 0.01 = 0.12 m (H ~ 0.3W)
            wall_margin: 8,
            exit_width_fraction: 1.0,
            exit_height: 12,
            ..Default::default()
        };

        let sluice_builder = SluiceGeometryBuilder::new(sluice_config.clone());

        // Create simulation
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.pressure_iterations = PRESSURE_ITERS as usize;

        // Mark solid cells from sluice geometry
        let mut solid_count = 0;
        for (i, j, k) in sluice_builder.solid_cells() {
            sim.grid.set_solid(i, j, k);
            solid_count += 1;
        }
        println!("Sluice: Marked {} solid cells", solid_count);

        // Compute SDF from solid cells
        sim.grid.compute_sdf();

        // Debug: Check SDF values
        let sdf_min = sim.grid.sdf.iter().cloned().fold(f32::INFINITY, f32::min);
        let sdf_max = sim
            .grid
            .sdf
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sdf_neg_count = sim.grid.sdf.iter().filter(|&&v| v < 0.0).count();
        println!(
            "SDF: min={:.3}, max={:.3}, negative_count={}",
            sdf_min, sdf_max, sdf_neg_count
        );
        let sample_k = GRID_DEPTH / 2;
        for &(label, sample_i) in &[("emit", 2usize), ("riffle", sluice_config.riffle_start_x)] {
            let floor_j = sluice_config.floor_height_at(sample_i);
            let idx =
                |i: usize, j: usize, k: usize| k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
            let sdf_floor = sim.grid.sdf[idx(sample_i, floor_j, sample_k)];
            let sdf_above = sim.grid.sdf[idx(sample_i, floor_j + 1, sample_k)];
            let sdf_below = if floor_j > 0 {
                sim.grid.sdf[idx(sample_i, floor_j - 1, sample_k)]
            } else {
                0.0
            };
            println!(
                "SDF sample {}: i={} j={} k={} | sdf[j-1]={:.3} sdf[j]={:.3} sdf[j+1]={:.3}",
                label, sample_i, floor_j, sample_k, sdf_below, sdf_floor, sdf_above
            );
        }

        // Water renderer
        let water_config = WaterRenderConfig {
            subdivisions: 2,
            ..Default::default()
        };
        let water_renderer =
            WaterHeightfieldRenderer::new(GRID_WIDTH, GRID_DEPTH, CELL_SIZE, water_config);

        // DEM for sediment - using existing ClusterSimulation3D with single-sphere clumps
        let bounds_min = Vec3::ZERO;
        let bounds_max = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE,
            GRID_HEIGHT as f32 * CELL_SIZE,
            GRID_DEPTH as f32 * CELL_SIZE,
        );
        let mut dem = ClusterSimulation3D::new(bounds_min, bounds_max);
        // Zero gravity - FLIP already applies gravity to sediment particles.
        // DEM is only for collision detection and friction.
        dem.gravity = Vec3::ZERO;
        dem.restitution = 0.0; // No bounce - gravel should not bounce out of water
        dem.friction = 0.5; // Sand friction
        dem.floor_friction = 0.6; // Higher on floor
        dem.normal_stiffness = 5000.0;
        dem.tangential_stiffness = 2500.0;
        dem.rolling_friction = 0.05;
        dem.use_dem = true;

        let gangue_radius = CELL_SIZE * GANGUE_RADIUS_CELLS;
        let gangue_mass =
            GANGUE_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * gangue_radius.powi(3);
        let gangue_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 42,
                style: sim3d::IrregularStyle3D::Round,
            },
            gangue_radius,
            gangue_mass,
        );
        let gangue_template_idx = dem.add_template(gangue_template);

        let gold_radius = CELL_SIZE * GOLD_RADIUS_CELLS;
        let gold_mass = GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * gold_radius.powi(3);
        let gold_template = ClumpTemplate3D::generate(ClumpShape3D::Flat4, gold_radius, gold_mass);
        let gold_template_idx = dem.add_template(gold_template);

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            sluice_builder,
            water_renderer,
            dem,
            gangue_template_idx,
            gold_template_idx,
            sediment_flip_indices: Vec::new(),
            tracer_particles: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            paused: false,
            frame: 0,
            camera: Camera::new(0.5, 0.4, 4.0, Vec3::new(
                (GRID_WIDTH as f32 * 0.5) * CELL_SIZE,
                (GRID_HEIGHT as f32 * 0.2) * CELL_SIZE,
                (GRID_DEPTH as f32 * 0.5) * CELL_SIZE,
            )),
            mouse_pressed: false,
            last_mouse_pos: None,
            water_emit_rate: WATER_EMIT_RATE,
            sediment_emit_rate: SEDIMENT_EMIT_RATE,
            use_dem: true, // Enable DEM by default
            use_async_readback: true,
            gpu_readback_pending: false,
            gpu_sync_substep: 0,
            gpu_needs_upload: true,
            pending_water_emits: 0,
            pending_sediment_emits: 0,
            start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
        }
    }

    /// Get the floor surface height (top of solid floor) at a given X position.
    /// Note: floor_height_* is the cell index of the top solid cell,
    /// so the floor surface is at (floor_height + 1) * cell_size.
    fn floor_height_at(&self, x: f32) -> f32 {
        let config = self.sluice_builder.config();
        let t = x / (config.grid_width as f32 * config.cell_size);
        let left = (config.floor_height_left + 1) as f32 * config.cell_size;
        let right = (config.floor_height_right + 1) as f32 * config.cell_size;
        left * (1.0 - t) + right * t
    }

    fn flow_accel(&self) -> f32 {
        let config = self.sluice_builder.config();
        let rise =
            (config.floor_height_left as f32 - config.floor_height_right as f32) * config.cell_size;
        let run = config.grid_width as f32 * config.cell_size;
        let slope = rise / run;
        9.81 * slope
    }

    fn compute_flow_metrics(&self) -> FlowMetrics {
        let config = self.sluice_builder.config();
        let cell_size = config.cell_size;
        let sample_x_min = 6.0 * cell_size;
        let mut sample_x_max = (config.riffle_start_x.saturating_sub(4) as f32) * cell_size;
        if sample_x_max <= sample_x_min {
            sample_x_max = (config.riffle_start_x as f32 * 0.5) * cell_size;
        }
        let sample_x_max = sample_x_max.min(config.grid_width as f32 * cell_size);
        let i_min = (sample_x_min / cell_size).floor() as i32;
        let i_max = (sample_x_max / cell_size).floor() as i32;
        let i_min = i_min.clamp(0, (config.grid_width as i32 - 1).max(0));
        let i_max = i_max.clamp(0, (config.grid_width as i32 - 1).max(0));
        let i_count = (i_max - i_min + 1).max(0) as usize;
        let depth_count = config.grid_depth.max(1);
        let mut max_depth_cells = vec![-1i32; i_count * depth_count];

        let mut vel_sum = 0.0f32;
        let mut vel_count = 0usize;
        let mut min_k = config.grid_depth as i32;
        let mut max_k = -1i32;

        for p in &self.sim.particles.list {
            if p.density > 1.0 {
                continue;
            }
            if p.position.x < sample_x_min || p.position.x > sample_x_max {
                continue;
            }
            let i = (p.position.x / cell_size).floor() as i32;
            let j = (p.position.y / cell_size).floor() as i32;
            let k = (p.position.z / cell_size).floor() as i32;
            if i < i_min || i > i_max || k < 0 || k >= config.grid_depth as i32 {
                continue;
            }
            let floor_j = config.floor_height_at(i as usize) as i32;
            let depth_cells = j - floor_j;
            if depth_cells > 0 {
                let idx = (i - i_min) as usize * depth_count + k as usize;
                if depth_cells > max_depth_cells[idx] {
                    max_depth_cells[idx] = depth_cells;
                }
                min_k = min_k.min(k);
                max_k = max_k.max(k);
            }
            if p.velocity.x > 0.0 {
                vel_sum += p.velocity.x;
                vel_count += 1;
            }
        }

        let mut depths: Vec<f32> = Vec::new();
        for depth_cells in max_depth_cells {
            if depth_cells > 0 {
                depths.push(depth_cells as f32 * cell_size);
            }
        }

        let (depth_p50, depth_p90) = if depths.is_empty() {
            (0.0, 0.0)
        } else {
            depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid_idx = depths.len() / 2;
            let p90_idx = ((depths.len() as f32 - 1.0) * 0.9).round() as usize;
            (depths[mid_idx], depths[p90_idx.min(depths.len() - 1)])
        };
        let vel_mean = if vel_count > 0 {
            vel_sum / vel_count as f32
        } else {
            0.0
        };

        let flow_width = if max_k >= min_k {
            (max_k - min_k + 1) as f32 * cell_size
        } else {
            0.0
        };
        let flow_rate_m3s = vel_mean * flow_width * depth_p90;
        let flow_rate_m3min = flow_rate_m3s * 60.0;

        FlowMetrics {
            sample_count: depths.len(),
            vel_mean,
            depth_p50,
            depth_p90,
            flow_width,
            flow_rate_m3s,
            flow_rate_m3min,
            sample_x_min,
            sample_x_max,
        }
    }

    fn queue_emissions(&mut self) {
        if self.frame % 2 == 0 {
            self.pending_water_emits = self
                .pending_water_emits
                .saturating_add(self.water_emit_rate);
            self.pending_sediment_emits = self
                .pending_sediment_emits
                .saturating_add(self.sediment_emit_rate);
        }
    }

    fn emit_pending_particles(&mut self) {
        let water_count = self.pending_water_emits;
        let sediment_count = self.pending_sediment_emits;
        if water_count == 0 && sediment_count == 0 {
            return;
        }
        self.pending_water_emits = 0;
        self.pending_sediment_emits = 0;
        self.emit_particles(water_count, sediment_count);
    }

    fn emit_particles(&mut self, water_count: usize, sediment_count: usize) {
        if self.paused || self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        let config = self.sluice_builder.config();
        let cell_size = config.cell_size;
        let grid_depth = config.grid_depth;
        // Emit at upstream (high) end, before the riffles
        let emit_x = 2.0 * cell_size; // Near left wall (upstream)
        let center_z = grid_depth as f32 * cell_size * 0.5;
        let floor_y = self.floor_height_at(emit_x);
        // Keep the feed sheet thin (<= 2 cells ~ 2 cm at 0.01m)
        let drop_height = 2.5 * cell_size;
        let sheet_height = 4.0 * cell_size;

        let water_spread_z = (grid_depth as f32 - 4.0) * cell_size * 0.3;
        let sediment_spread_z = (grid_depth as f32 - 4.0) * cell_size * 0.2;

        // Initial velocity tuned for ~0.3-1.0 m/s sheet flow and CFL < 1.
        let init_vel = Vec3::new(0.5, -0.05, 0.0); // Downstream + slight down

        // Emit water
        for _ in 0..water_count {
            if self.sim.particles.len() >= MAX_PARTICLES {
                break;
            }
            let x = emit_x + (rand_float() - 0.5) * 2.0 * cell_size;
            let z = center_z + (rand_float() - 0.5) * water_spread_z;
            let y = floor_y + drop_height + rand_float() * sheet_height;
            self.sim
                .spawn_particle_with_velocity(Vec3::new(x, y, z), init_vel);
        }

        // Emit sediment - create both FLIP particle and DEM clump
        for _ in 0..sediment_count {
            if self.sim.particles.len() >= MAX_PARTICLES {
                break;
            }
            let x = emit_x + (rand_float() - 0.5) * 2.0 * cell_size;
            let z = center_z + (rand_float() - 0.5) * sediment_spread_z;
            let band = rand_float();
            let (band_base, band_jitter, band_vel_scale, band_down) = if band < 0.7 {
                (0.5 * cell_size, 0.7 * cell_size, 1.0, 1.0)
            } else {
                (0.1 * cell_size, 0.3 * cell_size, 0.8, 1.2)
            };
            let y = floor_y + band_base + rand_float() * band_jitter;
            let sediment_vel = Vec3::new(
                init_vel.x * band_vel_scale,
                init_vel.y * band_down,
                init_vel.z,
            );
            let pos = Vec3::new(x, y, z);

            let is_gold = rand_float() < GOLD_FRACTION;
            let density = if is_gold {
                GOLD_DENSITY
            } else {
                GANGUE_DENSITY
            };
            let template_idx = if is_gold {
                self.gold_template_idx
            } else {
                self.gangue_template_idx
            };

            // Track FLIP particle index before spawning
            let flip_idx = self.sim.particles.len();
            self.sim.spawn_sediment(pos, sediment_vel, density);

            // Create corresponding DEM clump
            self.dem.spawn(template_idx, pos, sediment_vel);

            // Record the mapping
            self.sediment_flip_indices.push(flip_idx);
        }

        if water_count > 0 && self.frame % TRACER_INTERVAL_FRAMES == 0 {
            for _ in 0..TRACER_COUNT {
                self.spawn_tracer(
                    emit_x,
                    center_z,
                    floor_y,
                    drop_height,
                    sheet_height,
                    water_spread_z,
                    init_vel,
                    cell_size,
                );
            }
        }
    }

    fn spawn_tracer(
        &mut self,
        emit_x: f32,
        center_z: f32,
        floor_y: f32,
        drop_height: f32,
        sheet_height: f32,
        water_spread_z: f32,
        init_vel: Vec3,
        cell_size: f32,
    ) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }
        let x = emit_x + (rand_float() - 0.5) * 2.0 * cell_size;
        let z = center_z + (rand_float() - 0.5) * water_spread_z;
        let y = floor_y + drop_height + 0.5 * sheet_height;
        let idx = self.sim.particles.len();
        self.sim
            .spawn_particle_with_velocity(Vec3::new(x, y, z), init_vel);
        self.tracer_particles.push(TracerInfo {
            index: idx,
            spawn_frame: self.frame,
        });
        println!("Tracer spawned at frame {} (idx {})", self.frame, idx);
    }

    fn prepare_gpu_inputs(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.affine_vels.clear();
        self.densities.clear();

        for p in &self.sim.particles.list {
            self.positions.push(p.position);
            self.velocities.push(p.velocity);
            self.affine_vels.push(p.affine_velocity);
            self.densities.push(p.density);
        }

        self.cell_types.clear();
        self.cell_types
            .resize(GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH, 0);

        // Mark solids from SDF
        for (idx, &sdf_val) in self.sim.grid.sdf.iter().enumerate() {
            if sdf_val < 0.0 {
                self.cell_types[idx] = 2; // Solid
            }
        }

        // Mark fluid cells from particles
        for pos in &self.positions {
            let i = (pos.x / CELL_SIZE) as i32;
            let j = (pos.y / CELL_SIZE) as i32;
            let k = (pos.z / CELL_SIZE) as i32;
            if i >= 0
                && i < GRID_WIDTH as i32
                && j >= 0
                && j < GRID_HEIGHT as i32
                && k >= 0
                && k < GRID_DEPTH as i32
            {
                let idx =
                    k as usize * GRID_WIDTH * GRID_HEIGHT + j as usize * GRID_WIDTH + i as usize;
                if self.cell_types[idx] != 2 {
                    self.cell_types[idx] = 1; // Fluid
                }
            }
        }
    }

    fn ensure_readback_buffers_len(&mut self, particle_count: usize) {
        if self.positions.len() < particle_count {
            self.positions.resize(particle_count, Vec3::ZERO);
        }
        if self.velocities.len() < particle_count {
            self.velocities.resize(particle_count, Vec3::ZERO);
        }
        if self.affine_vels.len() < particle_count {
            self.affine_vels.resize(particle_count, Mat3::ZERO);
        }
    }

    fn apply_gpu_results(&mut self, count: usize) {
        let limit = count.min(self.sim.particles.list.len());
        for (i, p) in self.sim.particles.list.iter_mut().enumerate().take(limit) {
            if i < self.positions.len() {
                p.position = self.positions[i];
            }
            if i < self.velocities.len() {
                p.velocity = self.velocities[i];
            }
            if i < self.affine_vels.len() {
                p.affine_velocity = self.affine_vels[i];
            }
        }
    }

    fn run_dem_and_cleanup(&mut self, dt: f32) {
        // Run DEM collision response on sediment particles (after GPU FLIP)
        // DEM clumps are persistent - we sync state, not rebuild
        // IMPORTANT: Use collision_response_only, NOT step_with_sdf
        // step_with_sdf does full physics integration (double-moves particles)
        // collision_response_only only handles collision detection + velocity correction
        if self.use_dem && !self.sediment_flip_indices.is_empty() {
            // Debug: track velocities
            let mut flip_vels: Vec<Vec3> = Vec::new();

            // NOTE: Buoyancy-reduced gravity is now applied in the GPU shader (g2p_3d.wgsl)
            // using the drag-based entrainment model. No extra settling needed here.

            // Sync FLIP -> DEM: update clump positions/velocities from FLIP results
            // but PRESERVE rotation, angular_velocity, and contact history
            for (clump_idx, &flip_idx) in self.sediment_flip_indices.iter().enumerate() {
                if clump_idx < self.dem.clumps.len() && flip_idx < self.sim.particles.list.len() {
                    let p = &self.sim.particles.list[flip_idx];
                    let clump = &mut self.dem.clumps[clump_idx];
                    flip_vels.push(p.velocity);
                    // Take position/velocity from FLIP (which includes pressure, advection, drag, gravity)
                    clump.position = p.position;
                    clump.velocity = p.velocity;
                    // rotation and angular_velocity are PRESERVED from previous frame
                }
            }

            // Run DEM collision response only - detects collisions, pushes out of solids,
            // corrects velocity (bounce/friction), but does NOT integrate position
            // wet=true because gravel is in water - uses low friction so it slides
            let sdf_params = SdfParams {
                sdf: &self.sim.grid.sdf,
                grid_width: GRID_WIDTH,
                grid_height: GRID_HEIGHT,
                grid_depth: GRID_DEPTH,
                cell_size: CELL_SIZE,
                grid_offset: Vec3::ZERO, // Grid is at world origin
            };
            self.dem.collision_response_only(dt, &sdf_params, true); // wet=true

            // Sync DEM -> FLIP: copy results back + enforce back wall
            let back_wall_x = CELL_SIZE * 1.5; // Back wall boundary
            for (clump_idx, &flip_idx) in self.sediment_flip_indices.iter().enumerate() {
                if clump_idx < self.dem.clumps.len() && flip_idx < self.sim.particles.list.len() {
                    let clump = &mut self.dem.clumps[clump_idx];
                    let p = &mut self.sim.particles.list[flip_idx];

                    // Enforce back wall - gravel can't go behind x=back_wall_x
                    if clump.position.x < back_wall_x {
                        clump.position.x = back_wall_x;
                        if clump.velocity.x < 0.0 {
                            clump.velocity.x = 0.0;
                        }
                    }

                    // Debug: compare FLIP vs DEM velocities
                    if self.frame % 60 == 0 && clump_idx == 0 {
                        let flip_vel = flip_vels.get(clump_idx).copied().unwrap_or(Vec3::ZERO);
                        let dem_vel = clump.velocity;
                        println!(
                            "Gravel[0] FLIP vel: {:5.2} m/s | DEM vel: {:5.2} m/s",
                            flip_vel.length(),
                            dem_vel.length()
                        );
                    }

                    p.position = clump.position;
                    p.velocity = clump.velocity;
                }
            }
        }

        // Remove particles that exited the sluice (not the grid - buffer zone is part of grid)
        // Delete at sluice boundary, not grid boundary, for clean outflow
        // Must also delete corresponding DEM clumps and update mappings
        let sluice_exit_x = SLUICE_WIDTH as f32 * CELL_SIZE;
        let bounds_check = |p: &sim3d::Particle3D| {
            p.position.x > 0.0
                && p.position.x < sluice_exit_x
                && p.position.y > -CELL_SIZE
                && p.position.y < GRID_HEIGHT as f32 * CELL_SIZE
                && p.position.z > 0.0
                && p.position.z < GRID_DEPTH as f32 * CELL_SIZE
        };

        // Find which particles to delete
        let mut delete_indices: Vec<usize> = Vec::new();
        for (i, p) in self.sim.particles.list.iter().enumerate() {
            if !bounds_check(p) {
                delete_indices.push(i);
            }
        }

        // Delete from back to front to preserve indices
        for &del_idx in delete_indices.iter().rev() {
            // Check if this is a sediment particle (has corresponding DEM clump)
            if let Some(sediment_pos) = self
                .sediment_flip_indices
                .iter()
                .position(|&idx| idx == del_idx)
            {
                // Delete the DEM clump
                if sediment_pos < self.dem.clumps.len() {
                    self.dem.clumps.swap_remove(sediment_pos);
                }
                self.sediment_flip_indices.swap_remove(sediment_pos);
            }

            if let Some(tracer_pos) = self
                .tracer_particles
                .iter()
                .position(|t| t.index == del_idx)
            {
                let spawn_frame = self.tracer_particles[tracer_pos].spawn_frame;
                let travel_frames = self.frame.saturating_sub(spawn_frame);
                let travel_seconds = travel_frames as f32 / 60.0;
                println!(
                    "Tracer exited in {:.2} s ({} frames)",
                    travel_seconds, travel_frames
                );
                self.tracer_particles.swap_remove(tracer_pos);
            }

            // Delete the FLIP particle
            self.sim.particles.list.swap_remove(del_idx);

            // Update sediment_flip_indices: any index > del_idx needs to be decremented
            // Also, if we swap_removed, the last particle moved to del_idx
            let last_idx = self.sim.particles.list.len(); // This is the OLD last index (before swap_remove)
            for flip_idx in &mut self.sediment_flip_indices {
                if *flip_idx == last_idx {
                    // This particle was swapped into del_idx's position
                    *flip_idx = del_idx;
                }
            }
            for tracer in &mut self.tracer_particles {
                if tracer.index == last_idx {
                    tracer.index = del_idx;
                }
            }
        }
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }

        // Check for GPU device lost
        if game::gpu::is_device_lost() {
            log::error!("GPU device lost - pausing simulation");
            self.paused = true;
            return;
        }

        let dt = 1.0 / 60.0;
        let dt_sub = dt / SUBSTEPS as f32;
        let flow_accel = self.flow_accel();

        if self.use_async_readback {
            if self.gpu_readback_pending {
                let readback = if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu)
                {
                    gpu_flip.try_readback(
                        &gpu.ctx.device,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.affine_vels,
                    )
                } else {
                    None
                };

                if let Some(count) = readback {
                    self.gpu_readback_pending = false;
                    self.apply_gpu_results(count);
                    self.gpu_needs_upload = true;
                } else {
                    return;
                }
            }

            self.queue_emissions();

            if self.gpu_needs_upload {
                self.emit_pending_particles();
                self.run_dem_and_cleanup(dt_sub); // Use substep dt for DEM physics sync
                self.prepare_gpu_inputs();
            }

            let particle_count = self.sim.particles.list.len();
            let next_substep = if particle_count > 0 {
                self.gpu_sync_substep.saturating_add(1)
            } else {
                self.gpu_sync_substep
            };
            let schedule_readback = particle_count > 0 && next_substep >= GPU_SYNC_STRIDE;
            if schedule_readback {
                self.ensure_readback_buffers_len(particle_count);
            }

            if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                let sdf = self.sim.grid.sdf.as_slice();

                if self.gpu_needs_upload {
                    gpu_flip.step_no_readback(
                        &gpu.ctx.device,
                        &gpu.ctx.queue,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.affine_vels,
                        &self.densities,
                        &self.cell_types,
                        Some(sdf),
                        None,
                        dt_sub,
                        GRAVITY,
                        flow_accel,
                        PRESSURE_ITERS,
                    );
                    self.gpu_needs_upload = false;
                    for _ in 1..SUBSTEPS {
                        gpu_flip.step_in_place(
                            &gpu.ctx.device,
                            &gpu.ctx.queue,
                            particle_count as u32,
                            &self.cell_types,
                            Some(sdf),
                            None,
                            dt_sub,
                            GRAVITY,
                            flow_accel,
                            PRESSURE_ITERS,
                        );
                    }
                } else {
                    for _ in 0..SUBSTEPS {
                        gpu_flip.step_in_place(
                            &gpu.ctx.device,
                            &gpu.ctx.queue,
                            particle_count as u32,
                            &self.cell_types,
                            Some(sdf),
                            None,
                            dt_sub,
                            GRAVITY,
                            flow_accel,
                            PRESSURE_ITERS,
                        );
                    }
                }

                if schedule_readback {
                    if gpu_flip.request_readback(&gpu.ctx.device, &gpu.ctx.queue, particle_count) {
                        self.gpu_readback_pending = true;
                        self.gpu_sync_substep = 0;
                    } else {
                        self.gpu_sync_substep = next_substep;
                    }
                } else {
                    self.gpu_sync_substep = next_substep;
                }
            }
        } else {
            self.queue_emissions();
            self.emit_pending_particles();
            self.prepare_gpu_inputs();

            if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                let sdf = self.sim.grid.sdf.as_slice();

                for _ in 0..SUBSTEPS {
                    gpu_flip.step(
                        &gpu.ctx.device,
                        &gpu.ctx.queue,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.affine_vels,
                        &self.densities,
                        &self.cell_types,
                        Some(sdf),
                        None,
                        dt_sub,
                        GRAVITY,
                        flow_accel,
                        PRESSURE_ITERS,
                    );
                }
                self.apply_gpu_results(self.positions.len());
            }

            self.run_dem_and_cleanup(dt_sub); // Use substep dt for DEM physics sync
        }

        self.frame += 1;

        // FPS
        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.last_fps_time = now;

            let water_count = self
                .sim
                .particles
                .list
                .iter()
                .filter(|p| p.density <= 1.0)
                .count();
            let sediment_count = self.sim.particles.list.len() - water_count;
            let sort_mode = self
                .gpu_flip
                .as_ref()
                .map(|f| {
                    if f.use_sorted_p2g {
                        "SORTED"
                    } else {
                        "unsorted"
                    }
                })
                .unwrap_or("N/A");
            println!(
                "Frame {} | FPS: {:.1} | Particles: {} (water: {}, sediment: {}) [P2G: {}]",
                self.frame,
                self.current_fps,
                self.sim.particles.list.len(),
                water_count,
                sediment_count,
                sort_mode
            );
            let flow = self.compute_flow_metrics();
            if flow.sample_count == 0 {
                println!(
                    "Flow: n/a (no water samples in [{:.2}m, {:.2}m])",
                    flow.sample_x_min, flow.sample_x_max,
                );
            } else {
                println!(
                    "Flow: v={:.2} m/s | depth p50={:.3} m p90={:.3} m | width={:.3} m | Q={:.3} m3/s ({:.2} m3/min) | samples={} | window=[{:.2}m, {:.2}m]",
                    flow.vel_mean,
                    flow.depth_p50,
                    flow.depth_p90,
                    flow.flow_width,
                    flow.flow_rate_m3s,
                    flow.flow_rate_m3min,
                    flow.sample_count,
                    flow.sample_x_min,
                    flow.sample_x_max,
                );
            }
        }
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };

        let output = match gpu.ctx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        // Update uniforms
        let aspect = gpu.ctx.config.width as f32 / gpu.ctx.config.height as f32;
        let view_proj = self.camera.view_proj_matrix(aspect);
        let eye = self.camera.position();

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.ctx.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Build water mesh
        let time = self.start_time.elapsed().as_secs_f32();

        // Update sediment heights first (for water surface bridging over gravel)
        let sediment_particles = self
            .sim
            .particles
            .list
            .iter()
            .filter(|p| p.density > 1.0)
            .map(|p| p.position.to_array());
        self.water_renderer.update_sediment(sediment_particles);

        // Now build water mesh with sediment-aware bridging
        let water_particles = self
            .sim
            .particles
            .list
            .iter()
            .filter(|p| p.density <= 1.0)
            .map(|p| (p.position.to_array(), p.velocity.to_array()));

        // Capture floor height params to avoid borrowing self in closure
        // Note: floor_height_* is the cell index of the top solid cell.
        // The TOP of that cell (visible floor surface) is at (floor_height + 1) * cell_size
        let (floor_surface_left, floor_surface_right, total_width) = {
            let config = self.sluice_builder.config();
            (
                (config.floor_height_left + 1) as f32 * config.cell_size, // Top of floor cell
                (config.floor_height_right + 1) as f32 * config.cell_size, // Top of floor cell
                config.grid_width as f32 * config.cell_size,
            )
        };

        self.water_renderer.build_mesh(water_particles, time, |x| {
            let t = x / total_width;
            floor_surface_left * (1.0 - t) + floor_surface_right * t
        });

        // Upload water vertices
        let water_vertices = self.water_renderer.vertices();
        if !water_vertices.is_empty() {
            gpu.ctx.queue.write_buffer(
                &gpu.water_vertex_buffer,
                0,
                bytemuck::cast_slice(water_vertices),
            );
        }

        // Build sediment instances from DEM clumps
        let sediment_instances: Vec<SedimentInstance> = self
            .dem
            .clumps
            .iter()
            .map(|clump| {
                let template = &self.dem.templates[clump.template_idx];
                let color = if clump.template_idx == self.gold_template_idx {
                    GOLD_COLOR
                } else {
                    GANGUE_COLOR
                };
                SedimentInstance {
                    position: clump.position.to_array(),
                    scale: template.particle_radius,
                    rotation: clump.rotation.to_array(),
                    color,
                }
            })
            .collect();

        if !sediment_instances.is_empty() {
            gpu.ctx.queue.write_buffer(
                &gpu.sediment_instance_buffer,
                0,
                bytemuck::cast_slice(&sediment_instances),
            );
        }

        let mut encoder = gpu.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
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

            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);

            // Draw sluice
            pass.set_pipeline(&gpu.sluice_pipeline);
            pass.set_vertex_buffer(0, gpu.sluice_vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.sluice_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.sluice_builder.indices().len() as u32, 0, 0..1);

            // Draw water
            if !water_vertices.is_empty() {
                pass.set_pipeline(&gpu.water_pipeline);
                pass.set_vertex_buffer(0, gpu.water_vertex_buffer.slice(..));
                pass.draw(0..water_vertices.len() as u32, 0..1);
            }

            // Draw sediment as 3D rocks
            if !sediment_instances.is_empty() {
                pass.set_pipeline(&gpu.sediment_pipeline);
                pass.set_vertex_buffer(0, gpu.rock_mesh_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.sediment_instance_buffer.slice(..));
                pass.draw(
                    0..gpu.rock_mesh_vertex_count,
                    0..sediment_instances.len() as u32,
                );
            }
        }

        gpu.ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let ctx = pollster::block_on(WgpuContext::init(window.clone()));
        let device = &ctx.device;
        let queue = &ctx.queue;
        let format = ctx.config.format;

        // Build sluice mesh
        self.sluice_builder
            .build_mesh(|i, j, k| self.sim.grid.is_solid(i, j, k));
        self.sluice_builder.upload(&device);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(BASIC_SHADER.into()),
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: &[0u8; std::mem::size_of::<Uniforms>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Sluice pipeline
        let sluice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sluice Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[SluiceVertex::buffer_layout()],
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

        // Water pipeline (same vertex format as sluice)
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<WaterVertex>() as u64,
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
                cull_mode: None, // Water visible from both sides
                ..Default::default()
            },
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

        // Sediment shader (3D rock mesh with lighting)
        let sediment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Shader"),
            source: wgpu::ShaderSource::Wgsl(SEDIMENT_SHADER.into()),
        });

        // Sediment pipeline (instanced 3D rocks)
        let sediment_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sediment Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sediment_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Rock mesh vertices
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
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
                                format: wgpu::VertexFormat::Float32x3,
                            },
                        ],
                    },
                    // Instance data
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SedimentInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x3,
                            }, // position
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            }, // scale
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32x4,
                            }, // rotation
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32x4,
                            }, // color
                        ],
                    },
                ],
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
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

        // Buffers - create from builder data
        let sluice_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Vertices"),
            contents: bytemuck::cast_slice(self.sluice_builder.vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sluice_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Indices"),
            contents: bytemuck::cast_slice(self.sluice_builder.indices()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let water_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Vertices"),
            size: (GRID_WIDTH * GRID_DEPTH * 6 * 4 * std::mem::size_of::<WaterVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Rock mesh for sediment (icosahedron with jitter)
        let rock_mesh_vertices = build_rock_mesh();
        let rock_mesh_vertex_count = rock_mesh_vertices.len() as u32;
        let rock_mesh_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Rock Mesh"),
                contents: bytemuck::cast_slice(&rock_mesh_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let sediment_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Instances"),
            size: (MAX_PARTICLES * std::mem::size_of::<SedimentInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = create_depth_view(device, &ctx.config);
        // GPU FLIP
        let mut gpu_flip = GpuFlip3D::new(
            device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        // Friction-only sediment model:
        // - Sediment flows like water but settles and has friction when slow
        // - Density correction now applies to sediment, so it won't bunch unnaturally
        // - When sediment accumulates (friction > flow), cells marked SOLID so water flows OVER
        // Drag-based entrainment model:
        // - Gravel is pulled toward water velocity (drag)
        // - Buoyancy-reduced gravity pulls it down
        // - When water is fast, drag wins -> entrainment
        // - When water is slow, gravity wins -> settling
        gpu_flip.sediment_rest_particles = 0.0; // Disabled - don't mark sediment as SOLID
        gpu_flip.sediment_porosity_drag = 0.0; // Disable porosity drag
        gpu_flip.sediment_drag_coefficient = 3.0; // Drag rate (1/s) - how fast gravel approaches water velocity
        gpu_flip.sediment_settling_velocity = 0.5; // For vorticity lift calculation
        gpu_flip.sediment_friction_threshold = 0.05; // Low threshold - let gravel move freely
        gpu_flip.sediment_friction_strength = 0.1; // Light friction when slow
        gpu_flip.gold_density_threshold = 10.0;
        gpu_flip.gold_drag_multiplier = 3.0; // Fine gold entrains more than gangue
        gpu_flip.gold_settling_velocity = 0.1;
        gpu_flip.gold_flake_lift = 0.6;

        self.gpu = Some(GpuState {
            ctx,
            sluice_pipeline,
            water_pipeline,
            sediment_pipeline,
            uniform_buffer,
            uniform_bind_group,
            sluice_vertex_buffer,
            sluice_index_buffer,
            water_vertex_buffer,
            rock_mesh_vertex_buffer,
            rock_mesh_vertex_count,
            sediment_instance_buffer,
            depth_view,
        });
        self.gpu_flip = Some(gpu_flip);
        self.window = Some(window);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Friction Sluice - Sediment Test")
                .with_inner_size(winit::dpi::LogicalSize::new(1200, 800));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.init_gpu(window.clone());
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.ctx.resize(size.width.max(1), size.height.max(1));
                    gpu.depth_view = create_depth_view(&gpu.ctx.device, &gpu.ctx.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        PhysicalKey::Code(KeyCode::Space) => self.paused = !self.paused,
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.sim.particles.list.clear();
                            self.dem.clumps.clear();
                            self.sediment_flip_indices.clear();
                            self.frame = 0;
                            self.gpu_readback_pending = false;
                            self.gpu_sync_substep = 0;
                            self.gpu_needs_upload = true;
                            self.pending_water_emits = 0;
                            self.pending_sediment_emits = 0;
                            self.positions.clear();
                            self.velocities.clear();
                            self.affine_vels.clear();
                            self.densities.clear();
                            self.cell_types.clear();
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.water_emit_rate = (self.water_emit_rate + 25).min(500);
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.water_emit_rate = self.water_emit_rate.saturating_sub(25);
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.sediment_emit_rate = (self.sediment_emit_rate + 10).min(200);
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.sediment_emit_rate = self.sediment_emit_rate.saturating_sub(10);
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            self.use_dem = !self.use_dem;
                            println!("DEM: {}", if self.use_dem { "ON" } else { "OFF" });
                        }
                        PhysicalKey::Code(KeyCode::Digit5) => {
                            if let Some(flip) = self.gpu_flip.as_mut() {
                                flip.use_sorted_p2g = !flip.use_sorted_p2g;
                                println!(
                                    "Sorted P2G: {}",
                                    if flip.use_sorted_p2g { "ON" } else { "OFF" }
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        self.camera.handle_mouse_move(dx as f32, dy as f32);
                    }
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.camera.handle_zoom(scroll);
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}




/// Thread-safe random float generator using thread-local storage.
/// Returns a value in [0.0, 1.0).
fn rand_float() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u32> = const { Cell::new(12345) };
    }
    SEED.with(|seed| {
        let s = seed.get().wrapping_mul(1103515245).wrapping_add(12345);
        seed.set(s);
        (s as f32) / (u32::MAX as f32)
    })
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
