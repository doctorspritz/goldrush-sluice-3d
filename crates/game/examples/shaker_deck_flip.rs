//! Shaker Deck FLIP Demo
//!
//! Minimal FLIP setup for a perforated shaker deck with a gutter below.
//!
//! Run with: cargo run --example shaker_deck_flip --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::{GpuFlip3D, GravelObstacle};
use game::gpu::fluid_renderer::ScreenSpaceFluidRenderer;
use game::sluice_geometry::SluiceVertex;
use glam::{Mat3, Mat4, Quat, Vec3};
use sim3d::{
    ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FlipSimulation3D, Grid3D, SdfParams,
};
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
const CELL_SIZE: f32 = 0.05;
const GRID_WIDTH: usize = 140;
const GRID_DEPTH: usize = 64;
const GRID_HEIGHT: usize = 60;
const MAX_PARTICLES: usize = 80_000;

// Shaker deck geometry (meters)
const DECK_LENGTH: f32 = 6.0;
const DECK_WIDTH: f32 = 2.2;
const DECK_ORIGIN_X: f32 = 0.4;
const DECK_ORIGIN_Y: f32 = 1.6;
const DECK_ORIGIN_Z: f32 = 0.4;
const DECK_ANGLE_DEG: f32 = 12.0;
const DECK_THICKNESS: f32 = 0.14;
const DECK_WALL_HEIGHT: f32 = 0.25;
const DECK_WALL_THICKNESS: f32 = 0.08;
const DECK_SAFETY_WALL_HEIGHT: f32 = 0.6;
const GRAVEL_DECK_PAD: f32 = CELL_SIZE * 2.0;
const HOLE_SPACING: f32 = 0.18;
const HOLE_RADIUS: f32 = 0.02;

const GUTTER_OFFSET_CELLS: f32 = 2.0;
const GUTTER_WALL_HEIGHT: f32 = 0.3;
const GUTTER_WALL_THICKNESS: f32 = 0.1;

// Simulation
const GRAVITY: f32 = -9.8;
const PRESSURE_ITERS: u32 = 50;
const SUBSTEPS: u32 = 1;

const SPRAY_BAR_COUNT: usize = 4;
const SPRAY_BAR_HEIGHT: f32 = 0.5;
const GRAVEL_ONLY: bool = false;

// Emission rates (10-20% solids by mass)
const WATER_EMIT_RATE: usize = 20;
const SEDIMENT_EMIT_RATE: usize = 0;
const GPU_SYNC_STRIDE: u32 = 4; // GPU readback cadence (frames)

// Grain sizing (relative to cell size)
const GANGUE_RADIUS_CELLS: f32 = 0.12; // Coarse gangue grains
const GOLD_RADIUS_CELLS: f32 = 0.02; // Fine gold grains
const GANGUE_DENSITY: f32 = 2.7;
const GOLD_DENSITY: f32 = 19.3;
const GOLD_FRACTION: f32 = 0.05; // 5% of sediment spawns as gold

const GRAVEL_RADII: [f32; 5] = [0.03, 0.003, 0.0025, 0.002, 0.0015];
const LARGE_GRAVEL_CHANCE: f32 = 0.02;
const GRAVEL_OBSTACLE_RADIUS_CUTOFF: f32 = 0.003;
const GRAVEL_COUNTS: [usize; 4] = [0, 0, 0, 0];
const NUGGET_RADII: [f32; 2] = [0.02, 0.012];
const NUGGET_COUNTS: [usize; 2] = [0, 0];

const GRAVEL_STREAM_ENABLED: bool = true;
const GRAVEL_STREAM_INTERVAL_SECONDS: f32 = 0.01;
const GRAVEL_STREAM_BATCH: usize = 50;
const GRAVEL_STREAM_GAP_MULTIPLIER: f32 = 2.2;
const GRAVEL_WATER_DRAG: f32 = 2.0;

// Sediment colors
const GANGUE_COLOR: [f32; 4] = [0.6, 0.4, 0.2, 1.0];
const GOLD_COLOR: [f32; 4] = [0.95, 0.85, 0.2, 1.0];
const GRAVEL_COLOR: [f32; 4] = [0.38, 0.35, 0.32, 1.0];
const GRAVEL_VISUAL_SCALE: f32 = 1.0;
const NUGGET_VISUAL_SCALE: f32 = 1.0;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    // Pipelines
    sluice_pipeline: wgpu::RenderPipeline,
    sediment_pipeline: wgpu::RenderPipeline,

    // Buffers
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    deck_vertex_buffer: wgpu::Buffer,
    deck_index_buffer: wgpu::Buffer,
    deck_index_count: u32,
    rock_mesh_vertex_buffer: wgpu::Buffer,
    rock_mesh_vertex_count: u32,
    sediment_instance_buffer: wgpu::Buffer,

    // Depth
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    gpu_flip: Option<GpuFlip3D>,
    sim: FlipSimulation3D,
    deck_mesh: DeckMesh,
    fluid_renderer: Option<ScreenSpaceFluidRenderer>,
    dem: ClusterSimulation3D,
    dem_gravel: ClusterSimulation3D,
    gravel_sdf: Vec<f32>,
    gangue_template_idx: usize,
    gold_template_idx: usize,
    gravel_template_indices: Vec<usize>,
    nugget_template_indices: Vec<usize>,

    // Persistent FLIP<->DEM mapping
    // sediment_clump_idx[i] = index into dem.clumps for sediment particle i
    // We track sediment particles separately from water
    sediment_flip_indices: Vec<usize>, // FLIP particle indices that are sediment
    sediment_clump_indices: Vec<usize>,

    // Particle data for GPU transfer
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,

    // State
    paused: bool,
    frame: u32,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
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
    gravel_stream_timer: f32,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
}

struct DeckMesh {
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
}

impl App {
    fn new() -> Self {
        let deck_slope = -DECK_ANGLE_DEG.to_radians().tan();
        let gutter_floor_y = gutter_floor_y(deck_slope);
        let deck_mesh = build_deck_mesh(deck_slope, gutter_floor_y);
        let gravel_sdf = build_gravel_sdf(deck_slope, gutter_floor_y);

        // Create simulation
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.pressure_iterations = PRESSURE_ITERS as usize;

        // Mark solid cells for deck plate + gutter channel
        let mut solid_count = 0;
        for k in 0..GRID_DEPTH {
            for j in 0..GRID_HEIGHT {
                for i in 0..GRID_WIDTH {
                    let x = (i as f32 + 0.5) * CELL_SIZE;
                    let y = (j as f32 + 0.5) * CELL_SIZE;
                    let z = (k as f32 + 0.5) * CELL_SIZE;

                    if x < DECK_ORIGIN_X
                        || x > DECK_ORIGIN_X + DECK_LENGTH
                        || z < DECK_ORIGIN_Z
                        || z > DECK_ORIGIN_Z + DECK_WIDTH
                    {
                        continue;
                    }

                    let deck_height = DECK_ORIGIN_Y + deck_slope * (x - DECK_ORIGIN_X);
                    let deck_bottom = deck_height - DECK_THICKNESS;
                    let in_plate = y <= deck_height && y >= deck_bottom;
                    let mut solid = false;

                    if in_plate && !in_hole(x, z) {
                        solid = true;
                    }

                    if y <= gutter_floor_y {
                        solid = true;
                    }

                    let wall_bottom = gutter_floor_y;
                    let wall_top = deck_height + DECK_WALL_HEIGHT;
                    let near_side_wall = z <= DECK_ORIGIN_Z + DECK_WALL_THICKNESS
                        || z >= DECK_ORIGIN_Z + DECK_WIDTH - DECK_WALL_THICKNESS;
                    let near_back_wall = x <= DECK_ORIGIN_X + GUTTER_WALL_THICKNESS;
                    if (near_side_wall || near_back_wall) && y >= wall_bottom && y <= wall_top {
                        solid = true;
                    }

                    if solid {
                        sim.grid.set_solid(i, j, k);
                        solid_count += 1;
                    }
                }
            }
        }
        println!("Deck solids: {}", solid_count);

        // Compute SDF from solid cells
        sim.grid.compute_sdf();

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

        let mut dem_gravel = ClusterSimulation3D::new(bounds_min, bounds_max);
        dem_gravel.gravity = Vec3::new(0.0, -9.8, 0.0);
        dem_gravel.restitution = 0.2;
        dem_gravel.friction = 0.3;
        dem_gravel.floor_friction = 0.3;
        dem_gravel.normal_stiffness = 6000.0;
        dem_gravel.tangential_stiffness = 3500.0;
        dem_gravel.rolling_friction = 0.08;
        // Use simple integration + SDF push-out (avoid stiff DEM spring explosions).
        dem_gravel.use_dem = false;

        let mut gravel_template_indices = Vec::new();
        for (idx, radius) in GRAVEL_RADII.iter().enumerate() {
            let mass = GANGUE_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
            let template = ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 6,
                    seed: 100 + idx as u64,
                    style: sim3d::IrregularStyle3D::Sharp,
                },
                *radius,
                mass,
            );
            gravel_template_indices.push(dem_gravel.add_template(template));
        }

        let mut nugget_template_indices = Vec::new();
        for (idx, radius) in NUGGET_RADII.iter().enumerate() {
            let mass = GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
            let template = ClumpTemplate3D::generate(ClumpShape3D::Flat4, *radius, mass);
            nugget_template_indices.push(dem_gravel.add_template(template));
        }

        let mut app = Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            sim,
            deck_mesh,
            fluid_renderer: None,
            dem,
            dem_gravel,
            gravel_sdf,
            gangue_template_idx,
            gold_template_idx,
            gravel_template_indices,
            nugget_template_indices,
            sediment_flip_indices: Vec::new(),
            sediment_clump_indices: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
            cell_types: Vec::new(),
            paused: false,
            frame: 0,
            camera_angle: 0.5,
            camera_pitch: 0.4,
            camera_distance: 4.0,
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
            gravel_stream_timer: 0.0,
        };

        app.spawn_gravel_and_nuggets();
        app
    }

    fn deck_height_at(&self, x: f32) -> f32 {
        let slope = -DECK_ANGLE_DEG.to_radians().tan();
        DECK_ORIGIN_Y + slope * (x - DECK_ORIGIN_X)
    }

    fn flow_accel(&self) -> f32 {
        let slope = DECK_ANGLE_DEG.to_radians().tan();
        9.8 * slope
    }

    fn spawn_gravel_and_nuggets(&mut self) {
        // Spawn a tighter, single-stream band near the top of the deck.
        let x_min = DECK_ORIGIN_X + 0.4;
        let x_max = DECK_ORIGIN_X + DECK_LENGTH * 0.18;
        let z_center = DECK_ORIGIN_Z + DECK_WIDTH * 0.5;
        let z_span = 0.12;
        let z_min = z_center - z_span;
        let z_max = z_center + z_span;
        let mut spawn_positions: Vec<(Vec3, f32)> = Vec::new();
        let max_attempts = 32;

        for (template_idx, count) in self.gravel_template_indices.iter().zip(GRAVEL_COUNTS) {
            let radius = self.dem_gravel.templates[*template_idx].particle_radius;
            for _ in 0..count {
                let mut spawned = false;
                for _ in 0..max_attempts {
                    let x = x_min + rand_float() * (x_max - x_min);
                    let z = z_min + rand_float() * (z_max - z_min);
                    let y = self.deck_height_at(x) + radius + 0.28 + rand_float() * 0.18;
                    let pos = Vec3::new(x, y, z);
                    if spawn_positions.iter().all(|(p, r)| {
                        let min_sep = (radius + *r) * 2.2;
                        (pos - *p).length_squared() > min_sep * min_sep
                    }) {
                        self.dem_gravel.spawn(*template_idx, pos, Vec3::ZERO);
                        spawn_positions.push((pos, radius));
                        spawned = true;
                        break;
                    }
                }
                if !spawned {
                    break;
                }
            }
        }

        for (template_idx, count) in self.nugget_template_indices.iter().zip(NUGGET_COUNTS) {
            let radius = self.dem_gravel.templates[*template_idx].particle_radius;
            for _ in 0..count {
                let mut spawned = false;
                for _ in 0..max_attempts {
                    let x = x_min + rand_float() * (x_max - x_min);
                    let z = z_min + rand_float() * (z_max - z_min);
                    let y = self.deck_height_at(x) + radius + 0.20 + rand_float() * 0.12;
                    let pos = Vec3::new(x, y, z);
                    if spawn_positions.iter().all(|(p, r)| {
                        let min_sep = (radius + *r) * 2.2;
                        (pos - *p).length_squared() > min_sep * min_sep
                    }) {
                        self.dem_gravel.spawn(*template_idx, pos, Vec3::ZERO);
                        spawn_positions.push((pos, radius));
                        spawned = true;
                        break;
                    }
                }
                if !spawned {
                    break;
                }
            }
        }
    }

    fn emit_gravel_stream(&mut self) {
        if self.gravel_stream_timer < GRAVEL_STREAM_INTERVAL_SECONDS {
            return;
        }
        self.gravel_stream_timer -= GRAVEL_STREAM_INTERVAL_SECONDS;

        // Spawn a tighter, single-stream band near the top of the deck.
        let x_min = DECK_ORIGIN_X + 0.4;
        let x_max = DECK_ORIGIN_X + DECK_LENGTH * 0.18;
        let z_center = DECK_ORIGIN_Z + DECK_WIDTH * 0.5;
        let z_span = DECK_WIDTH * 0.38;
        let z_min = z_center - z_span;
        let z_max = z_center + z_span;

        let max_attempts = 32;
        let template_count = self.gravel_template_indices.len();
        if template_count == 0 {
            return;
        }

        for _ in 0..GRAVEL_STREAM_BATCH {
            let template_idx = if template_count > 1 && rand_float() < LARGE_GRAVEL_CHANCE {
                self.gravel_template_indices[0]
            } else {
                let small_count = template_count.saturating_sub(1).max(1);
                let emit_idx = 1 + ((rand_float() * small_count as f32) as usize % small_count);
                self.gravel_template_indices[emit_idx]
            };
            let radius = self.dem_gravel.templates[template_idx].particle_radius;
            for _ in 0..max_attempts {
                let x = x_min + rand_float() * (x_max - x_min);
                let z = z_min + rand_float() * (z_max - z_min);
                let y = self.deck_height_at(x) + radius + 0.28 + rand_float() * 0.18;
                let pos = Vec3::new(x, y, z);
                let clear = self.dem_gravel.clumps.iter().all(|clump| {
                    let other = &self.dem_gravel.templates[clump.template_idx];
                    let min_sep = (radius + other.particle_radius) * GRAVEL_STREAM_GAP_MULTIPLIER;
                    (pos - clump.position).length_squared() > min_sep * min_sep
                });
                if clear {
                    self.dem_gravel.spawn(template_idx, pos, Vec3::ZERO);
                    break;
                }
            }
        }
    }

    fn integrate_gravel(&mut self, dt: f32) {
        // Simple integration without clump-clump impulses to avoid energy injection.
        let bounds_min = self.dem_gravel.bounds_min;
        let bounds_max = self.dem_gravel.bounds_max;
        let downhill_accel = self.flow_accel() * 0.6;
        let deck_origin_y = DECK_ORIGIN_Y;
        let deck_origin_x = DECK_ORIGIN_X;
        let deck_slope = -DECK_ANGLE_DEG.to_radians().tan();
        let grid = &self.sim.grid;
        for clump in &mut self.dem_gravel.clumps {
            let template = &self.dem_gravel.templates[clump.template_idx];
            let deck_height = deck_origin_y + deck_slope * (clump.position.x - deck_origin_x);
            if clump.position.y <= deck_height + CELL_SIZE * 2.0 {
                clump.velocity.x += downhill_accel * dt;
                if in_hole(clump.position.x, clump.position.z) && clump.velocity.y < 0.0 {
                    // Grate impact: damp vertical velocity and add a small lateral scatter.
                    clump.velocity.y *= 0.4;
                    let jitter = 0.15 + rand_float() * 0.1;
                    let dir = if rand_float() > 0.5 { 1.0 } else { -1.0 };
                    clump.velocity.x += jitter * dir;
                    clump.velocity.z += jitter * (1.0 - rand_float() * 2.0);
                }
            }
            if !GRAVEL_ONLY {
                let water_vel = sample_grid_velocity(grid, clump.position);
                let drag = (water_vel - clump.velocity) * (GRAVEL_WATER_DRAG * dt);
                clump.velocity += drag;
            }
            clump.velocity += self.dem_gravel.gravity * dt;
            clump.position += clump.velocity * dt;
            if clump.angular_velocity.length_squared() > 1.0e-10 {
                let delta = Quat::from_scaled_axis(clump.angular_velocity * dt);
                clump.rotation = (delta * clump.rotation).normalize();
            }

            // Soft bounds clamp to keep gravel within the sim box.
            let r = template.particle_radius;
            if clump.position.x < bounds_min.x + r {
                clump.position.x = bounds_min.x + r;
                clump.velocity.x = clump.velocity.x.max(0.4);
            } else if clump.position.x < bounds_min.x + r + CELL_SIZE * 0.5 {
                clump.velocity.x = clump.velocity.x.max(0.2);
            }
            if clump.position.y < bounds_min.y + r {
                clump.position.y = bounds_min.y + r;
                clump.velocity.y = 0.0;
            } else if clump.position.y > bounds_max.y - r {
                clump.position.y = bounds_max.y - r;
                clump.velocity.y = 0.0;
            }
            let z_min = bounds_min.z + r;
            let z_max = bounds_max.z - r;
            if clump.position.z < z_min {
                clump.position.z = bounds_min.z + r;
                clump.velocity.z = 0.0;
            } else if clump.position.z > z_max {
                clump.position.z = bounds_max.z - r;
                clump.velocity.z = 0.0;
            }
            clump.position.z = clump.position.z.clamp(z_min, z_max);
        }
    }

    fn clamp_gravel_bounds(&mut self) {
        let bounds_min = self.dem_gravel.bounds_min;
        let bounds_max = self.dem_gravel.bounds_max;
        for clump in &mut self.dem_gravel.clumps {
            let r = self.dem_gravel.templates[clump.template_idx].particle_radius;
            if clump.position.x < bounds_min.x + r {
                clump.position.x = bounds_min.x + r;
                clump.velocity.x = clump.velocity.x.max(0.4);
            }
            if clump.position.y < bounds_min.y + r {
                clump.position.y = bounds_min.y + r;
                clump.velocity.y = 0.0;
            } else if clump.position.y > bounds_max.y - r {
                clump.position.y = bounds_max.y - r;
                clump.velocity.y = 0.0;
            }
            if clump.position.z < bounds_min.z + r {
                clump.position.z = bounds_min.z + r;
                clump.velocity.z = 0.0;
            } else if clump.position.z > bounds_max.z - r {
                clump.position.z = bounds_max.z - r;
                clump.velocity.z = 0.0;
            }
        }
    }

    fn cull_gravel_outflow(&mut self) {
        let max_x = self.dem_gravel.bounds_max.x + CELL_SIZE * 2.0;
        let templates = &self.dem_gravel.templates;
        self.dem_gravel.clumps.retain(|clump| {
            let r = templates[clump.template_idx].particle_radius;
            clump.position.x <= max_x + r
        });
    }

    fn build_gravel_obstacles(&self) -> Vec<GravelObstacle> {
        let mut obstacles: Vec<GravelObstacle> = Vec::new();
        for clump in &self.dem_gravel.clumps {
            let radius = self.dem_gravel.templates[clump.template_idx].particle_radius;
            if radius > GRAVEL_OBSTACLE_RADIUS_CUTOFF {
                obstacles.push(GravelObstacle {
                    position: clump.position.to_array(),
                    radius,
                });
            }
        }
        obstacles
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
        let sediment_count = if self.sediment_emit_rate == 0 {
            0
        } else {
            self.pending_sediment_emits
        };
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

        let cell_size = CELL_SIZE;
        let grid_depth = GRID_DEPTH;
        // Spray bars above the grate so sediment falls through the holes
        let span = DECK_LENGTH * 0.7;
        let start_x = DECK_ORIGIN_X + DECK_LENGTH * 0.15;
        let bar_spacing = span / SPRAY_BAR_COUNT.max(1) as f32;
        let center_z = DECK_ORIGIN_Z + DECK_WIDTH * 0.5;

        let water_spread_z = DECK_WIDTH;
        let sediment_spread_z = DECK_WIDTH * 0.6;

        // Initial velocity tuned for spray + slight downstream push.
        let init_vel = Vec3::new(0.35, -0.45, 0.0);

        // Emit water from a single transverse emitter above the grate near the start.
        if water_count > 0 {
            let water_x = DECK_ORIGIN_X + DECK_LENGTH * 0.12;
            let deck_y = self.deck_height_at(water_x);
            let water_y = deck_y + SPRAY_BAR_HEIGHT + rand_float() * 0.1;
            for i in 0..water_count {
                if self.sim.particles.len() >= MAX_PARTICLES {
                    break;
                }
                let t = (i as f32 + 0.5) / water_count.max(1) as f32;
                let x = water_x + (rand_float() - 0.5) * 2.0 * cell_size;
                let z = DECK_ORIGIN_Z + t * water_spread_z + (rand_float() - 0.5) * 0.12;
                let y = water_y + rand_float() * 0.1;
                self.sim
                    .spawn_particle_with_velocity(Vec3::new(x, y, z), init_vel);
            }
        }

        let mut sediment_remaining = sediment_count;

        for bar in 0..SPRAY_BAR_COUNT {
            if sediment_remaining == 0 {
                break;
            }
            let bar_x = start_x + (bar as f32 + 0.5) * bar_spacing;
            let deck_y = self.deck_height_at(bar_x);
            let sediment_emit =
                (sediment_remaining + (SPRAY_BAR_COUNT - bar - 1)) / (SPRAY_BAR_COUNT - bar);
            sediment_remaining = sediment_remaining.saturating_sub(sediment_emit);

            // Emit sediment - create both FLIP particle and DEM clump
            for i in 0..sediment_emit {
                if self.sim.particles.len() >= MAX_PARTICLES {
                    break;
                }
                let t = (i as f32 + 0.5) / sediment_emit.max(1) as f32;
                let x = bar_x + (rand_float() - 0.5) * 2.0 * cell_size;
                let z = center_z + (t - 0.5) * sediment_spread_z + (rand_float() - 0.5) * 0.06;
                let band = rand_float();
                let (band_base, band_jitter, band_vel_scale, band_down) = if band < 0.7 {
                    (0.5 * cell_size, 0.7 * cell_size, 1.0, 1.0)
                } else {
                    (0.1 * cell_size, 0.3 * cell_size, 0.8, 1.2)
                };
                let y = deck_y + SPRAY_BAR_HEIGHT + band_base + rand_float() * band_jitter;
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
                let clump_idx = self.dem.spawn(template_idx, pos, sediment_vel);
                // Record the mapping
                self.sediment_flip_indices.push(flip_idx);
                self.sediment_clump_indices.push(clump_idx);
            }
        }
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

    fn strip_sediment_particles(&mut self) {
        if self.sediment_emit_rate != 0 {
            return;
        }
        if !self.sediment_flip_indices.is_empty() || !self.dem.clumps.is_empty() {
            self.sediment_flip_indices.clear();
            self.sediment_clump_indices.clear();
            self.dem.clumps.clear();
        }
        self.sim.particles.list.retain(|p| p.density <= 1.0);
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
            for (map_idx, &flip_idx) in self.sediment_flip_indices.iter().enumerate() {
                let clump_idx = *self
                    .sediment_clump_indices
                    .get(map_idx)
                    .unwrap_or(&usize::MAX);
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
            };
            self.dem.collision_response_only(dt, &sdf_params, true); // wet=true

            // Sync DEM -> FLIP: copy results back + enforce back wall
            let back_wall_x = CELL_SIZE * 1.5; // Back wall boundary
            for (map_idx, &flip_idx) in self.sediment_flip_indices.iter().enumerate() {
                let clump_idx = *self
                    .sediment_clump_indices
                    .get(map_idx)
                    .unwrap_or(&usize::MAX);
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

        // Remove particles that exited the grid bounds.
        // Must also delete corresponding DEM clumps and update mappings.
        let x_max = GRID_WIDTH as f32 * CELL_SIZE;
        let y_max = GRID_HEIGHT as f32 * CELL_SIZE;
        let z_max = GRID_DEPTH as f32 * CELL_SIZE;
        let deck_x_min = DECK_ORIGIN_X - CELL_SIZE;
        let deck_x_max = DECK_ORIGIN_X + DECK_LENGTH + CELL_SIZE * 2.0;
        let deck_z_min = DECK_ORIGIN_Z - CELL_SIZE;
        let deck_z_max = DECK_ORIGIN_Z + DECK_WIDTH + CELL_SIZE;
        let gutter_y = gutter_floor_y(-DECK_ANGLE_DEG.to_radians().tan());
        let deck_y_min = gutter_y - CELL_SIZE * 2.0;
        let deck_y_max =
            DECK_ORIGIN_Y + DECK_WALL_HEIGHT + DECK_SAFETY_WALL_HEIGHT + CELL_SIZE * 2.0;

        let bounds_check = |p: &sim3d::Particle3D| {
            let in_grid = p.position.x > 0.0
                && p.position.x < x_max
                && p.position.y > -CELL_SIZE
                && p.position.y < y_max
                && p.position.z > 0.0
                && p.position.z < z_max;
            if !in_grid {
                return false;
            }
            if p.density <= 1.0 {
                return p.position.x >= deck_x_min
                    && p.position.x <= deck_x_max
                    && p.position.z >= deck_z_min
                    && p.position.z <= deck_z_max
                    && p.position.y >= deck_y_min
                    && p.position.y <= deck_y_max;
            }
            true
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
                if let Some(&clump_idx) = self.sediment_clump_indices.get(sediment_pos) {
                    if clump_idx < self.dem.clumps.len() {
                        let last_idx = self.dem.clumps.len() - 1;
                        self.dem.clumps.swap_remove(clump_idx);
                        for idx in &mut self.sediment_clump_indices {
                            if *idx == last_idx {
                                *idx = clump_idx;
                            }
                        }
                    }
                }
                self.sediment_flip_indices.swap_remove(sediment_pos);
                self.sediment_clump_indices.swap_remove(sediment_pos);
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
        }
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }

        let dt = 1.0 / 60.0;
        let dt_sub = dt / SUBSTEPS as f32;
        let flow_accel = self.flow_accel();
        if GRAVEL_STREAM_ENABLED {
            self.gravel_stream_timer += dt;
            self.emit_gravel_stream();
        }
        let gravel_substeps = 2;
        let gravel_dt = dt / gravel_substeps as f32;
        for _ in 0..gravel_substeps {
            self.integrate_gravel(gravel_dt);
            let sdf_params = SdfParams {
                sdf: &self.gravel_sdf,
                grid_width: GRID_WIDTH,
                grid_height: GRID_HEIGHT,
                grid_depth: GRID_DEPTH,
                cell_size: CELL_SIZE,
            };
            for _ in 0..2 {
                self.dem_gravel
                    .collision_response_only(gravel_dt, &sdf_params, true);
            }
            self.clamp_gravel_bounds();
            self.cull_gravel_outflow();
            for clump in &mut self.dem_gravel.clumps {
                clump.velocity *= 0.998;
            }
        }
        if self.frame % 60 == 0 {
            if let Some(clump) = self.dem_gravel.clumps.first() {
                let p = clump.position;
                let v = clump.velocity;
                println!(
                    "Gravel: count={} pos=({:.2},{:.2},{:.2}) vel=({:.2},{:.2},{:.2})",
                    self.dem_gravel.clumps.len(),
                    p.x,
                    p.y,
                    p.z,
                    v.x,
                    v.y,
                    v.z
                );
            } else {
                println!("Gravel: count=0");
            }
        }
        let reset_x = DECK_ORIGIN_X + 0.6;
        let reset_z = DECK_ORIGIN_Z + 0.5 * DECK_WIDTH;
        let reset_y = self.deck_height_at(reset_x) + 0.2;
        for clump in &mut self.dem_gravel.clumps {
            let p = clump.position;
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                clump.position = Vec3::new(reset_x, reset_y, reset_z);
                clump.velocity = Vec3::ZERO;
                clump.angular_velocity = Vec3::ZERO;
            }
        }

        if GRAVEL_ONLY {
            return;
        }

        if self.use_async_readback {
            if self.gpu_readback_pending {
                let readback = if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu)
                {
                    gpu_flip.try_readback(
                        &gpu.device,
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
                self.strip_sediment_particles();
                self.run_dem_and_cleanup(dt);
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

            let gravel_obstacles = if !GRAVEL_ONLY {
                self.build_gravel_obstacles()
            } else {
                Vec::new()
            };
            if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                let sdf = self.sim.grid.sdf.as_slice();
                if !gravel_obstacles.is_empty() {
                    gpu_flip.upload_gravel_obstacles(&gpu.queue, &gravel_obstacles);
                }

                if self.gpu_needs_upload {
                    gpu_flip.step_no_readback(
                        &gpu.device,
                        &gpu.queue,
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
                            &gpu.device,
                            &gpu.queue,
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
                            &gpu.device,
                            &gpu.queue,
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
                    if gpu_flip.request_readback(&gpu.device, &gpu.queue, particle_count) {
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
            self.strip_sediment_particles();
            self.prepare_gpu_inputs();

            let gravel_obstacles = if !GRAVEL_ONLY {
                self.build_gravel_obstacles()
            } else {
                Vec::new()
            };
            if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                let sdf = self.sim.grid.sdf.as_slice();
                if !gravel_obstacles.is_empty() {
                    gpu_flip.upload_gravel_obstacles(&gpu.queue, &gravel_obstacles);
                }

                for _ in 0..SUBSTEPS {
                    gpu_flip.step(
                        &gpu.device,
                        &gpu.queue,
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

            self.run_dem_and_cleanup(dt);
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
            println!(
                "Frame {} | FPS: {:.1} | Particles: {} (water: {}, sediment: {})",
                self.frame,
                self.current_fps,
                self.sim.particles.list.len(),
                water_count,
                sediment_count
            );
        }
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        // Update uniforms
        let center = Vec3::new(
            DECK_ORIGIN_X + DECK_LENGTH * 0.5,
            DECK_ORIGIN_Y + 0.2,
            DECK_ORIGIN_Z + DECK_WIDTH * 0.5,
        );
        let eye = center
            + Vec3::new(
                self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
                self.camera_distance * self.camera_pitch.sin(),
                self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
            );
        let view_matrix = Mat4::look_at_rh(eye, center, Vec3::Y);
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let proj_matrix = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let view_proj = proj_matrix * view_matrix;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Build gravel-only instances from DEM clumps
        let sediment_instances: Vec<SedimentInstance> = self
            .dem_gravel
            .clumps
            .iter()
            .map(|clump| {
                let template = &self.dem_gravel.templates[clump.template_idx];
                let is_nugget = self.nugget_template_indices.contains(&clump.template_idx);
                let color = if is_nugget { GOLD_COLOR } else { GRAVEL_COLOR };
                let scale = if is_nugget {
                    template.particle_radius * NUGGET_VISUAL_SCALE
                } else {
                    template.particle_radius * GRAVEL_VISUAL_SCALE
                };
                SedimentInstance {
                    position: clump.position.to_array(),
                    scale,
                    rotation: clump.rotation.to_array(),
                    color,
                }
            })
            .collect();

        if !sediment_instances.is_empty() {
            gpu.queue.write_buffer(
                &gpu.sediment_instance_buffer,
                0,
                bytemuck::cast_slice(&sediment_instances),
            );
        }

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
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

            // Draw deck + gutter
            pass.set_pipeline(&gpu.sluice_pipeline);
            pass.set_vertex_buffer(0, gpu.deck_vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.deck_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.deck_index_count, 0, 0..1);
        }

        if !GRAVEL_ONLY {
            if let (Some(gpu_flip), Some(_)) = (&self.gpu_flip, &self.gpu) {
                let active_count = self.sim.particles.list.len() as u32;
                if let Some(fluid_renderer) = &self.fluid_renderer {
                    fluid_renderer.render(
                        &gpu.device,
                        &gpu.queue,
                        &mut encoder,
                        &view,
                        gpu_flip,
                        active_count,
                        view_matrix.to_cols_array_2d(),
                        proj_matrix.to_cols_array_2d(),
                        eye.to_array(),
                        gpu.config.width,
                        gpu.config.height,
                    );
                }
            }
        }

        if !sediment_instances.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sediment Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                ..Default::default()
            });
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_pipeline(&gpu.sediment_pipeline);
            pass.set_vertex_buffer(0, gpu.rock_mesh_vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.sediment_instance_buffer.slice(..));
            pass.draw(
                0..gpu.rock_mesh_vertex_count,
                0..sediment_instances.len() as u32,
            );
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..wgpu::Limits::default()
                    }
                    .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ),
        )
        .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

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

        let mut fluid_renderer = ScreenSpaceFluidRenderer::new(&device, config.format);
        fluid_renderer.particle_radius = CELL_SIZE * 0.5;
        fluid_renderer.resize(&device, config.width, config.height);
        self.fluid_renderer = Some(fluid_renderer);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
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

        // Buffers - create from deck mesh data
        let deck_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deck Vertices"),
            contents: bytemuck::cast_slice(&self.deck_mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let deck_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deck Indices"),
            contents: bytemuck::cast_slice(&self.deck_mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let deck_index_count = self.deck_mesh.indices.len() as u32;

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

        // Depth texture
        let (depth_texture, depth_view) = create_depth_texture(&device, &config);

        // GPU FLIP
        let mut gpu_flip = GpuFlip3D::new(
            &device,
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
            device,
            queue,
            surface,
            config,
            sluice_pipeline,
            sediment_pipeline,
            uniform_buffer,
            uniform_bind_group,
            deck_vertex_buffer,
            deck_index_buffer,
            deck_index_count,
            rock_mesh_vertex_buffer,
            rock_mesh_vertex_count,
            sediment_instance_buffer,
            depth_texture,
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
                .with_title("Shaker Deck - FLIP Variant")
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
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    // Recreate depth texture for new size
                    let (depth_texture, depth_view) =
                        create_depth_texture(&gpu.device, &gpu.config);
                    gpu.depth_texture = depth_texture;
                    gpu.depth_view = depth_view;
                    if let Some(fluid_renderer) = &mut self.fluid_renderer {
                        fluid_renderer.resize(&gpu.device, gpu.config.width, gpu.config.height);
                    }
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
                            self.dem_gravel.clumps.clear();
                            self.sediment_flip_indices.clear();
                            self.sediment_clump_indices.clear();
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
                            self.spawn_gravel_and_nuggets();
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
                        self.camera_angle -= dx as f32 * 0.01;
                        self.camera_pitch = (self.camera_pitch + dy as f32 * 0.01).clamp(-1.4, 1.4);
                    }
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.3).clamp(1.0, 15.0);
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

fn in_hole(x: f32, z: f32) -> bool {
    let local_x = x - DECK_ORIGIN_X;
    let local_z = z - DECK_ORIGIN_Z;
    let gx = (local_x / HOLE_SPACING).round();
    let gz = (local_z / HOLE_SPACING).round();
    let cx = gx * HOLE_SPACING;
    let cz = gz * HOLE_SPACING;
    let dx = local_x - cx;
    let dz = local_z - cz;
    dx * dx + dz * dz <= HOLE_RADIUS * HOLE_RADIUS
}

fn sample_grid_velocity(grid: &Grid3D, pos: Vec3) -> Vec3 {
    let cell_size = grid.cell_size;
    let mut i = (pos.x / cell_size).floor() as i32;
    let mut j = (pos.y / cell_size).floor() as i32;
    let mut k = (pos.z / cell_size).floor() as i32;

    i = i.clamp(0, grid.width as i32 - 1);
    j = j.clamp(0, grid.height as i32 - 1);
    k = k.clamp(0, grid.depth as i32 - 1);

    let i0 = i as usize;
    let j0 = j as usize;
    let k0 = k as usize;

    let i1 = (i0 + 1).min(grid.width);
    let j1 = (j0 + 1).min(grid.height);
    let k1 = (k0 + 1).min(grid.depth);

    let u0 = grid.u[grid.u_index(i0, j0, k0)];
    let u1 = grid.u[grid.u_index(i1, j0, k0)];
    let v0 = grid.v[grid.v_index(i0, j0, k0)];
    let v1 = grid.v[grid.v_index(i0, j1, k0)];
    let w0 = grid.w[grid.w_index(i0, j0, k0)];
    let w1 = grid.w[grid.w_index(i0, j0, k1)];

    Vec3::new(0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (w0 + w1))
}

fn push_quad(
    vertices: &mut Vec<SluiceVertex>,
    indices: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    color: [f32; 4],
) {
    let base = vertices.len() as u32;
    vertices.push(SluiceVertex::new(a, color));
    vertices.push(SluiceVertex::new(b, color));
    vertices.push(SluiceVertex::new(c, color));
    vertices.push(SluiceVertex::new(d, color));
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn build_deck_mesh(deck_slope: f32, gutter_floor_y: f32) -> DeckMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let x0 = DECK_ORIGIN_X;
    let x1 = DECK_ORIGIN_X + DECK_LENGTH;
    let z0 = DECK_ORIGIN_Z;
    let z1 = DECK_ORIGIN_Z + DECK_WIDTH;
    let y0 = DECK_ORIGIN_Y + deck_slope * (x0 - DECK_ORIGIN_X);
    let y1 = DECK_ORIGIN_Y + deck_slope * (x1 - DECK_ORIGIN_X);
    let y0b = y0 - DECK_THICKNESS;
    let y1b = y1 - DECK_THICKNESS;

    let deck_color = [0.26, 0.28, 0.3, 1.0];
    let gutter_color = [0.15, 0.16, 0.17, 1.0];
    let wall_color = [0.2, 0.21, 0.22, 1.0];
    let safety_wall_color = [0.22, 0.25, 0.28, 0.2];
    let y0_high = y0 + DECK_WALL_HEIGHT;
    let y1_high = y1 + DECK_WALL_HEIGHT;
    let y0_safety = y0_high + DECK_SAFETY_WALL_HEIGHT;
    let y1_safety = y1_high + DECK_SAFETY_WALL_HEIGHT;

    // Deck top with literal holes (skip quads near hole centers)
    let step = HOLE_SPACING * 0.2;
    let mut x = x0;
    while x < x1 {
        let x_next = (x + step).min(x1);
        let y_a = DECK_ORIGIN_Y + deck_slope * (x - DECK_ORIGIN_X);
        let y_b = DECK_ORIGIN_Y + deck_slope * (x_next - DECK_ORIGIN_X);
        let mut z = z0;
        while z < z1 {
            let z_next = (z + step).min(z1);
            let cx = 0.5 * (x + x_next);
            let cz = 0.5 * (z + z_next);
            let in_hole_quad = in_hole(cx, cz)
                || in_hole(x, z)
                || in_hole(x_next, z)
                || in_hole(x, z_next)
                || in_hole(x_next, z_next);
            if !in_hole_quad {
                push_quad(
                    &mut vertices,
                    &mut indices,
                    [x, y_a, z],
                    [x_next, y_b, z],
                    [x_next, y_b, z_next],
                    [x, y_a, z_next],
                    deck_color,
                );
                // Double-sided so the grate renders from above and below.
                push_quad(
                    &mut vertices,
                    &mut indices,
                    [x, y_a, z],
                    [x, y_a, z_next],
                    [x_next, y_b, z_next],
                    [x_next, y_b, z],
                    deck_color,
                );
            }
            z = z_next;
        }
        x = x_next;
    }

    // Deck bottom with matching holes
    let mut xb = x0;
    while xb < x1 {
        let xb_next = (xb + step).min(x1);
        let y_b0 = DECK_ORIGIN_Y + deck_slope * (xb - DECK_ORIGIN_X) - DECK_THICKNESS;
        let y_b1 = DECK_ORIGIN_Y + deck_slope * (xb_next - DECK_ORIGIN_X) - DECK_THICKNESS;
        let mut zb = z0;
        while zb < z1 {
            let zb_next = (zb + step).min(z1);
            let cx = 0.5 * (xb + xb_next);
            let cz = 0.5 * (zb + zb_next);
            let in_hole_quad = in_hole(cx, cz)
                || in_hole(xb, zb)
                || in_hole(xb_next, zb)
                || in_hole(xb, zb_next)
                || in_hole(xb_next, zb_next);
            if !in_hole_quad {
                push_quad(
                    &mut vertices,
                    &mut indices,
                    [xb, y_b0, zb],
                    [xb, y_b0, zb_next],
                    [xb_next, y_b1, zb_next],
                    [xb_next, y_b1, zb],
                    deck_color,
                );
            }
            zb = zb_next;
        }
        xb = xb_next;
    }

    // Deck side skirts
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, y0b, z0],
        [x1, y1b, z0],
        [x1, y1, z0],
        [x0, y0, z0],
        deck_color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, y0b, z1],
        [x0, y0, z1],
        [x1, y1, z1],
        [x1, y1b, z1],
        deck_color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, y0b, z0],
        [x0, y0, z0],
        [x0, y0, z1],
        [x0, y0b, z1],
        deck_color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [x1, y1b, z0],
        [x1, y1b, z1],
        [x1, y1, z1],
        [x1, y1, z0],
        deck_color,
    );

    // Deck side walls (left/right) - extend down to gutter floor to avoid gaps.
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, gutter_floor_y, z0],
        [x1, gutter_floor_y, z0],
        [x1, y1_high, z0],
        [x0, y0_high, z0],
        wall_color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, gutter_floor_y, z1],
        [x0, y0_high, z1],
        [x1, y1_high, z1],
        [x1, gutter_floor_y, z1],
        wall_color,
    );

    // Safety wall above deck (low alpha).
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, y0_high, z0],
        [x1, y1_high, z0],
        [x1, y1_safety, z0],
        [x0, y0_safety, z0],
        safety_wall_color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, y0_high, z1],
        [x0, y0_safety, z1],
        [x1, y1_safety, z1],
        [x1, y1_high, z1],
        safety_wall_color,
    );

    // Gutter floor
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, gutter_floor_y, z0],
        [x1, gutter_floor_y, z0],
        [x1, gutter_floor_y, z1],
        [x0, gutter_floor_y, z1],
        gutter_color,
    );

    // Back wall (upstream) - connect gutter floor to deck wall top.
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, gutter_floor_y, z0],
        [x0, gutter_floor_y, z1],
        [x0, y0_high, z1],
        [x0, y0_high, z0],
        wall_color,
    );
    push_quad(
        &mut vertices,
        &mut indices,
        [x0, y0_high, z0],
        [x0, y0_high, z1],
        [x0, y0_safety, z1],
        [x0, y0_safety, z0],
        safety_wall_color,
    );

    DeckMesh { vertices, indices }
}

fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

fn gutter_floor_y(deck_slope: f32) -> f32 {
    let x_end = DECK_ORIGIN_X + DECK_LENGTH;
    let y_low = DECK_ORIGIN_Y + deck_slope * (x_end - DECK_ORIGIN_X);
    y_low - DECK_THICKNESS - GUTTER_OFFSET_CELLS * CELL_SIZE
}

fn build_gravel_sdf(deck_slope: f32, gutter_floor_y: f32) -> Vec<f32> {
    let mut grid = Grid3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);

    for k in 0..GRID_DEPTH {
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                let x = (i as f32 + 0.5) * CELL_SIZE;
                let y = (j as f32 + 0.5) * CELL_SIZE;
                let z = (k as f32 + 0.5) * CELL_SIZE;

                if x < DECK_ORIGIN_X
                    || x > DECK_ORIGIN_X + DECK_LENGTH
                    || z < DECK_ORIGIN_Z
                    || z > DECK_ORIGIN_Z + DECK_WIDTH
                {
                    continue;
                }

                let deck_height = DECK_ORIGIN_Y + deck_slope * (x - DECK_ORIGIN_X);
                let deck_bottom = deck_height - DECK_THICKNESS - GRAVEL_DECK_PAD;
                let in_plate = y <= deck_height && y >= deck_bottom;
                let mut solid = false;

                if in_plate {
                    solid = true;
                }

                if y <= gutter_floor_y {
                    solid = true;
                }

                let wall_bottom = gutter_floor_y;
                let wall_top = deck_height + DECK_WALL_HEIGHT;
                let near_side_wall = z <= DECK_ORIGIN_Z + DECK_WALL_THICKNESS
                    || z >= DECK_ORIGIN_Z + DECK_WIDTH - DECK_WALL_THICKNESS;
                let near_back_wall = x <= DECK_ORIGIN_X + GUTTER_WALL_THICKNESS;
                if (near_side_wall || near_back_wall) && y >= wall_bottom && y <= wall_top {
                    solid = true;
                }

                if solid {
                    grid.set_solid(i, j, k);
                }
            }
        }
    }

    grid.compute_sdf();
    grid.sdf
}

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

struct ParticleInput {
    @location(0) quad_pos: vec2<f32>,
    @location(2) position: vec3<f32>,
    @location(3) color: vec4<f32>,
}

@vertex
fn vs_particle(in: ParticleInput) -> VertexOutput {
    var out: VertexOutput;
    let size = 0.008;
    let world_pos = in.position + vec3<f32>(in.quad_pos * size, 0.0);
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
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
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_pos: vec3<f32>,
    @location(3) instance_scale: f32,
    @location(4) instance_rot: vec4<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let scaled = in.position * in.instance_scale;
    let world_pos = in.instance_pos + quat_rotate(in.instance_rot, scaled);
    let normal = normalize(quat_rotate(in.instance_rot, in.normal));
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.2));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let view_dir = normalize(uniforms.camera_pos - world_pos);
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
    let shade = 0.35 + 0.65 * diffuse;
    let tint = in.color.rgb * shade + vec3<f32>(0.08) * rim;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(tint, in.color.a);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
"#;

/// Build an icosahedron mesh with slight jitter for rock-like appearance
fn build_rock_mesh() -> Vec<MeshVertex> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let inv_len = 1.0 / (1.0 + phi * phi).sqrt();
    let a = inv_len;
    let b = phi * inv_len;

    // Icosahedron vertices with jitter
    let mut verts = [
        Vec3::new(-a, b, 0.0),
        Vec3::new(a, b, 0.0),
        Vec3::new(-a, -b, 0.0),
        Vec3::new(a, -b, 0.0),
        Vec3::new(0.0, -a, b),
        Vec3::new(0.0, a, b),
        Vec3::new(0.0, -a, -b),
        Vec3::new(0.0, a, -b),
        Vec3::new(b, 0.0, -a),
        Vec3::new(b, 0.0, a),
        Vec3::new(-b, 0.0, -a),
        Vec3::new(-b, 0.0, a),
    ];

    // Apply slight jitter for rock-like appearance
    let seed = 0xB2D4_09A7_u32;
    for (idx, pos) in verts.iter_mut().enumerate() {
        let idx_u = idx as u32;
        let radial = 1.0 + 0.08 * hash_to_unit(seed ^ idx_u.wrapping_mul(11));
        let lateral = Vec3::new(
            hash_to_unit(seed ^ idx_u.wrapping_mul(13)),
            hash_to_unit(seed ^ idx_u.wrapping_mul(17)),
            hash_to_unit(seed ^ idx_u.wrapping_mul(19)),
        ) * 0.04;
        *pos = (*pos * radial) + lateral;
    }

    // Normalize to unit sphere
    let mut max_len = 0.0_f32;
    for pos in &verts {
        max_len = max_len.max(pos.length());
    }
    if max_len > 0.0 {
        for pos in &mut verts {
            *pos /= max_len;
        }
    }

    // Icosahedron faces
    let indices: [[usize; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    let mut vertices = Vec::with_capacity(indices.len() * 3);
    for tri in indices {
        let va = verts[tri[0]];
        let vb = verts[tri[1]];
        let vc = verts[tri[2]];
        let normal = (vb - va).cross(vc - va).normalize();
        for pos in [va, vb, vc] {
            vertices.push(MeshVertex {
                position: pos.to_array(),
                normal: normal.to_array(),
            });
        }
    }

    vertices
}

fn hash_to_unit(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    let unit = x as f32 / u32::MAX as f32;
    unit * 2.0 - 1.0
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
