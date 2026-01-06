//! Industrial Scale Sluice - Large 3D GPU FLIP Test
//!
//! Much larger grid, longer sluice, shallower gradient, more water.
//! Target: 50k-100k+ particles at playable framerates.
//!
//! Run with: cargo run --example industrial_sluice --release

use bytemuck::{Pod, Zeroable};
use game::gpu::bed_3d::{self, GpuBed3D, GpuBedParams};
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Mat4, Vec3};
use sim3d::FlipSimulation3D;
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

// INDUSTRIAL SCALE - much larger than box_3d_test
const GRID_WIDTH: usize = 160;   // 5x wider (was 32)
const GRID_HEIGHT: usize = 40;   // Taller for water depth
const GRID_DEPTH: usize = 24;    // Thinner sluice (was 32)
const CELL_SIZE: f32 = 0.03;     // Smaller cells for detail
const MAX_PARTICLES: usize = 500_000;  // Increased cap to avoid emitter saturation
const FLOW_PARTICLE_STRIDE: usize = 8; // Render every Nth particle for flow viz
const MAX_FLOW_PARTICLES: usize = MAX_PARTICLES / FLOW_PARTICLE_STRIDE;
const TARGET_FPS: f32 = 60.0;
const PRESSURE_ITERS_MIN: u32 = 30;
const PRESSURE_ITERS_MAX: u32 = 120;
const PRESSURE_ITERS_STEP: u32 = 5;
const MAX_SURFACE_VERTICES: usize = GRID_WIDTH * GRID_DEPTH * 6;
const VORTICITY_EPSILON_DEFAULT: f32 = 0.05;
const VORTICITY_EPSILON_STEP: f32 = 0.01;
const VORTICITY_EPSILON_MAX: f32 = 0.25;
const GPU_SYNC_STRIDE: u32 = 4;
const FLOOR_HEIGHT_LEFT: usize = 10;
const FLOOR_HEIGHT_RIGHT: usize = 3;
const RIFFLE_SPACING: usize = 12;
const RIFFLE_HEIGHT: usize = 2;
const RIFFLE_START_X: usize = 12;
const RIFFLE_END_PAD: usize = 8;
const SEDIMENT_EMIT_FRACTION: f32 = 0.25;
const RIFFLE_THICKNESS_CELLS: i32 = 2;
const SEDIMENT_REL_DENSITY: f32 = 2.65;
const SEDIMENT_REST_PARTICLES: f32 = 8.0;
const SEDIMENT_SETTLING_VELOCITY: f32 = 0.35;
const BED_POROSITY: f32 = 0.4;
const BED_SAMPLE_HEIGHT_CELLS: f32 = 2.0;
const BED_FRICTION: f32 = 0.004;
const SEDIMENT_GRAIN_DIAMETER: f32 = 0.0015;
const WATER_DENSITY: f32 = 1000.0;
const SHIELDS_CRITICAL: f32 = 0.045;
const SHIELDS_SMOOTH: f32 = 0.02;
const BEDLOAD_COEFF: f32 = 0.25;
const ENTRAINMENT_COEFF: f32 = 0.2;
const RIFFLE_PROBE_PAD: i32 = 2;
const BED_AIR_MARGIN_CELLS: f32 = 1.5;

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
struct SurfaceVertex {
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
    gpu_bed: Option<GpuBed3D>,
    sim: FlipSimulation3D,
    paused: bool,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    frame: u32,
    solid_instances: Vec<ParticleInstance>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    bed_height: Vec<f32>,
    bed_base_height: Vec<f32>,
    bed_water_vel_sum: Vec<Vec3>,
    bed_water_count: Vec<u32>,
    bed_sediment_count: Vec<u32>,
    bed_flux_x: Vec<f32>,
    bed_flux_z: Vec<f32>,
    cell_types: Vec<u32>,
    use_gpu_sim: bool,
    pressure_iters_gpu: u32,
    vorticity_epsilon: f32,
    use_async_readback: bool,
    gpu_readback_pending: bool,
    render_heightfield: bool,
    render_flow_particles: bool,
    debug_riffle_probe: bool,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
    emitter_enabled: bool,
    particles_exited: u32,
    pending_emit: usize,
    gpu_particle_count: u32,
    gpu_probe_stats: Vec<i32>,
    heightfield: Vec<f32>,
    surface_vertices: Vec<SurfaceVertex>,
    flow_particles: Vec<ParticleInstance>,
}

struct RiffleProbeStats {
    water_count: u32,
    sediment_count: u32,
    water_avg_y: f32,
    sediment_avg_y: f32,
    water_max_y: f32,
    sediment_max_y: f32,
    water_avg_vy: f32,
    sediment_avg_vy: f32,
    water_sdf_neg: u32,
    sediment_sdf_neg: u32,
    water_below_bed: u32,
    sediment_below_bed: u32,
    water_above_bed: u32,
    sediment_above_bed: u32,
    water_up: u32,
    sediment_up: u32,
    water_avg_offset: f32,
    sediment_avg_offset: f32,
    water_max_offset: f32,
    sediment_max_offset: f32,
    bed_min: f32,
    bed_max: f32,
}

struct MaterialProbeStats {
    count: u32,
    avg_y: f32,
    max_y: f32,
    avg_vy: f32,
    sdf_neg: u32,
    below_bed: u32,
    above_bed: u32,
    avg_offset: f32,
    max_offset: f32,
    up: u32,
}

struct SedimentThroughputStats {
    total: u32,
    upstream: u32,
    at_riffle: u32,
    downstream: u32,
    max_x: f32,
    max_y: f32,
    lofted: u32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    solid_buffer: wgpu::Buffer,
    surface_vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    surface_pipeline: wgpu::RenderPipeline,
}

/// Create an industrial-scale sluice:
/// - 2° slope (very shallow, like real sluices)
/// - Many riffles spaced along the length
/// - Wide exit at the end
fn create_industrial_sluice(sim: &mut FlipSimulation3D) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    // 2° slope: over 160 cells, drop = 160 * tan(2°) ≈ 5.6 cells
    let floor_height_left = FLOOR_HEIGHT_LEFT;  // Start 10 cells high
    let floor_height_right = FLOOR_HEIGHT_RIGHT;  // End 4 cells high (6 cell drop over 160 cells ≈ 2.1°)

    // Riffle parameters - more riffles for industrial scale
    let riffle_spacing = RIFFLE_SPACING;     // Riffles every 12 cells
    let riffle_height = RIFFLE_HEIGHT;       // Riffles are 4 cells tall (deeper pooling)
    let riffle_start_x = RIFFLE_START_X;     // Start riffles after inlet
    let riffle_end_x = width - RIFFLE_END_PAD; // Stop before exit

    // Exit parameters - wide exit
    let exit_start_z = depth / 6;
    let exit_end_z = 5 * depth / 6;
    let exit_height = 8;

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                // Calculate floor height at this x position (linear interpolation)
                let t = i as f32 / (width - 1) as f32;
                let floor_height = floor_height_left as f32 * (1.0 - t) + floor_height_right as f32 * t;
                let floor_j = floor_height as usize;

                // Check if this is a riffle position
                let is_riffle = i >= riffle_start_x && i < riffle_end_x &&
                    (i - riffle_start_x) % riffle_spacing < 2 &&
                    j <= floor_j + riffle_height &&
                    j > floor_j;

                // Check if this is the exit opening
                let is_exit = i == width - 1 &&
                    k >= exit_start_z && k < exit_end_z &&
                    j > floor_j && j <= floor_j + exit_height;

                let is_boundary =
                    (i == 0) ||                      // Left wall
                    (i == width - 1 && !is_exit) ||  // Right wall (except exit)
                    j <= floor_j ||                   // Sloped floor
                    j == height - 1 ||                // Ceiling
                    k == 0 || k == depth - 1 ||       // Z walls
                    is_riffle;                        // Riffles on floor

                if is_boundary {
                    sim.grid.set_solid(i, j, k);
                }
            }
        }
    }

    sim.grid.compute_sdf();

    let num_riffles = ((riffle_end_x - riffle_start_x) / riffle_spacing) as usize;
    let slope_deg = ((floor_height_left - floor_height_right) as f32 / width as f32).atan().to_degrees();
    println!("Industrial sluice: {}x{}x{} grid", width, height, depth);
    println!("  Slope: {:.1}° ({} → {} cells)", slope_deg, floor_height_left, floor_height_right);
    println!("  {} riffles, exit width: {} cells", num_riffles, exit_end_z - exit_start_z);
}

fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED >> 16) as f32 / 65535.0
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smooth_positive(x: f32, width: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if width <= 0.0 {
        return x;
    }
    x * smoothstep(0.0, width, x)
}

fn flow_accel_from_slope() -> f32 {
    let drop = (FLOOR_HEIGHT_LEFT as f32 - FLOOR_HEIGHT_RIGHT as f32).max(0.0);
    let slope = drop / (GRID_WIDTH as f32 - 1.0);
    9.8 * slope
}

fn bed_surface_height_at(i: usize) -> f32 {
    // Bed height excludes riffles; riffles remain solid via SDF only.
    let t = i as f32 / (GRID_WIDTH as f32 - 1.0);
    let floor_height = FLOOR_HEIGHT_LEFT as f32 * (1.0 - t) + FLOOR_HEIGHT_RIGHT as f32 * t;
    floor_height * CELL_SIZE
}

fn build_bed_base_height() -> Vec<f32> {
    let mut height = vec![0.0f32; GRID_WIDTH * GRID_DEPTH];
    for k in 0..GRID_DEPTH {
        for i in 0..GRID_WIDTH {
            height[k * GRID_WIDTH + i] = bed_surface_height_at(i);
        }
    }
    height
}

fn build_gpu_bed_params(dt: f32) -> GpuBedParams {
    let riffle_start = RIFFLE_START_X as i32;
    let riffle_end = riffle_start + RIFFLE_THICKNESS_CELLS - 1;
    let riffle_min_i = (riffle_start - RIFFLE_PROBE_PAD).max(0);
    let riffle_max_i = (riffle_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);

    let downstream_start = riffle_end + 1 + RIFFLE_PROBE_PAD;
    let downstream_end = downstream_start + RIFFLE_THICKNESS_CELLS - 1;
    let downstream_min_i = (downstream_start - RIFFLE_PROBE_PAD).max(0);
    let downstream_max_i = (downstream_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);

    let riffle_start_x = RIFFLE_START_X as f32 * CELL_SIZE;
    let riffle_end_x = (RIFFLE_START_X as i32 + RIFFLE_THICKNESS_CELLS) as f32 * CELL_SIZE;
    let downstream_x = riffle_end_x + CELL_SIZE;

    GpuBedParams {
        dt,
        sample_height: BED_SAMPLE_HEIGHT_CELLS * CELL_SIZE,
        bed_air_margin: BED_AIR_MARGIN_CELLS * CELL_SIZE,
        loft_height: CELL_SIZE * 2.0,
        riffle_min_i,
        riffle_max_i,
        downstream_min_i,
        downstream_max_i,
        riffle_start_x,
        riffle_end_x,
        downstream_x,
        bed_friction: BED_FRICTION,
        sediment_rel_density: SEDIMENT_REL_DENSITY,
        water_density: WATER_DENSITY,
        sediment_grain_diameter: SEDIMENT_GRAIN_DIAMETER,
        shields_critical: SHIELDS_CRITICAL,
        shields_smooth: SHIELDS_SMOOTH,
        bedload_coeff: BEDLOAD_COEFF,
        entrainment_coeff: ENTRAINMENT_COEFF,
        sediment_settling_velocity: SEDIMENT_SETTLING_VELOCITY,
        bed_porosity: BED_POROSITY,
        max_bed_height: (GRID_HEIGHT as f32 - 2.0) * CELL_SIZE,
    }
}

impl App {
    fn new() -> Self {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 80;  // Reduced for better FPS (was 150)

        create_industrial_sluice(&mut sim);

        let bed_base_height = build_bed_base_height();
        let bed_height = bed_base_height.clone();
        let bed_column_count = GRID_WIDTH * GRID_DEPTH;
        let bed_water_vel_sum = vec![Vec3::ZERO; bed_column_count];
        let bed_water_count = vec![0u32; bed_column_count];
        let bed_sediment_count = vec![0u32; bed_column_count];
        let bed_flux_x = vec![0.0f32; bed_column_count];
        let bed_flux_z = vec![0.0f32; bed_column_count];

        let solid_instances = Self::collect_solids(&sim);
        let pressure_iters_gpu = sim.pressure_iterations as u32;

        println!("Solid cells: {}", solid_instances.len());
        println!("Max particles: {}", MAX_PARTICLES);
        println!("Controls: SPACE=pause, R=reset, G=toggle GPU/CPU, E=toggle emitter, H=heightfield, F=flow particles, V/B=vorticity -/+, P=riffle probe, ESC=quit");

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            gpu_bed: None,
            sim,
            paused: false,
            camera_angle: 0.3,
            camera_pitch: 0.3,
            camera_distance: 8.0,  // Start further back for larger scene
            frame: 0,
            solid_instances,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            bed_height,
            bed_base_height,
            bed_water_vel_sum,
            bed_water_count,
            bed_sediment_count,
            bed_flux_x,
            bed_flux_z,
            cell_types: Vec::new(),
            use_gpu_sim: true,
            pressure_iters_gpu,
            vorticity_epsilon: VORTICITY_EPSILON_DEFAULT,
            use_async_readback: false,
            gpu_readback_pending: false,
            render_heightfield: true,
            render_flow_particles: true,
            debug_riffle_probe: true,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            emitter_enabled: true,
            particles_exited: 0,
            pending_emit: 0,
            gpu_particle_count: 0,
            gpu_probe_stats: vec![0; bed_3d::PROBE_STAT_BUFFER_LEN],
            heightfield: vec![f32::NEG_INFINITY; GRID_WIDTH * GRID_DEPTH],
            surface_vertices: Vec::with_capacity(MAX_SURFACE_VERTICES),
            flow_particles: Vec::with_capacity(MAX_FLOW_PARTICLES),
        }
    }

    /// Emit particles from inlet on left side - higher rate for industrial scale
    fn emit_particles(&mut self, count: usize) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        let cell_size = CELL_SIZE;
        let max_to_spawn = (MAX_PARTICLES - self.sim.particles.len()).min(count);
        let sediment_spawn = (max_to_spawn as f32 * SEDIMENT_EMIT_FRACTION) as usize;
        let water_spawn = max_to_spawn.saturating_sub(sediment_spawn);

        // Emit above the first riffle so particles drop under gravity.
        let emit_x = (RIFFLE_START_X as f32 + 0.5) * cell_size;
        let center_z = GRID_DEPTH as f32 * cell_size * 0.5;
        let drop_height = 8.0 * cell_size;

        // Wider spread for industrial inlet
        let spread_x = 4.0 * cell_size;
        let spread_z = (GRID_DEPTH as f32 - 4.0) * cell_size * 0.6;

        for _ in 0..water_spawn {
            let x = emit_x + (rand_float() - 0.5) * spread_x;
            let z = center_z + (rand_float() - 0.5) * spread_z;
            let i = (x / cell_size).floor() as i32;
            let k = (z / cell_size).floor() as i32;
            let bed_y = if i >= 0 && i < GRID_WIDTH as i32 && k >= 0 && k < GRID_DEPTH as i32 {
                let idx = k as usize * GRID_WIDTH + i as usize;
                self.bed_height[idx]
            } else {
                0.0
            };
            let y = bed_y + drop_height + rand_float() * 2.0 * cell_size;

            self.sim.spawn_particle(Vec3::new(x, y, z));
        }

        for _ in 0..sediment_spawn {
            if self.sim.particles.len() >= MAX_PARTICLES {
                break;
            }

            let x = emit_x + (rand_float() - 0.5) * spread_x;
            let z = center_z + (rand_float() - 0.5) * spread_z;
            let i = (x / cell_size).floor() as i32;
            let k = (z / cell_size).floor() as i32;
            let bed_y = if i >= 0 && i < GRID_WIDTH as i32 && k >= 0 && k < GRID_DEPTH as i32 {
                let idx = k as usize * GRID_WIDTH + i as usize;
                self.bed_height[idx]
            } else {
                0.0
            };

            let y = bed_y + drop_height + rand_float() * 2.0 * cell_size;
            let vel = Vec3::ZERO;
            self.sim.spawn_sediment(Vec3::new(x, y, z), vel, SEDIMENT_REL_DENSITY);
        }
    }

    fn update_sediment_bed(&mut self, dt: f32) {
        let w = GRID_WIDTH;
        let d = GRID_DEPTH;
        let cell_size = CELL_SIZE;
        let column_count = w * d;
        let sample_height = cell_size * BED_SAMPLE_HEIGHT_CELLS;

        self.bed_water_vel_sum.fill(Vec3::ZERO);
        self.bed_water_count.fill(0);
        self.bed_sediment_count.fill(0);

        for p in &self.sim.particles.list {
            let i = (p.position.x / cell_size).floor() as i32;
            let k = (p.position.z / cell_size).floor() as i32;
            if i < 0 || i >= w as i32 || k < 0 || k >= d as i32 {
                continue;
            }
            let idx = k as usize * w + i as usize;
            let bed_y = self.bed_height[idx];
            if p.position.y >= bed_y && p.position.y <= bed_y + sample_height {
                if p.density <= 1.0 {
                    self.bed_water_vel_sum[idx] += p.velocity;
                    self.bed_water_count[idx] += 1;
                } else {
                    self.bed_sediment_count[idx] += 1;
                }
            }
        }

        let particle_height = cell_size / (SEDIMENT_REST_PARTICLES * (1.0 - BED_POROSITY));
        let mut deposit_quota = vec![0u32; column_count];
        let mut erode_quota = vec![0u32; column_count];
        let density_diff = (SEDIMENT_REL_DENSITY - 1.0) * WATER_DENSITY;

        for idx in 0..column_count {
            let count = self.bed_water_count[idx];
            let avg_vel = if count > 0 {
                self.bed_water_vel_sum[idx] / count as f32
            } else {
                Vec3::ZERO
            };
            self.bed_water_vel_sum[idx] = avg_vel;

            let speed = avg_vel.length();
            let tau = WATER_DENSITY * BED_FRICTION * speed * speed;
            let theta = if speed > 0.0 {
                tau / (density_diff * 9.81 * SEDIMENT_GRAIN_DIAMETER)
            } else {
                0.0
            };
            let excess = smooth_positive(theta - SHIELDS_CRITICAL, SHIELDS_SMOOTH);
            let available_height = (self.bed_height[idx] - self.bed_base_height[idx]).max(0.0);
            let availability = (available_height / (cell_size * 2.0)).clamp(0.0, 1.0);
            let bedload_mag = BEDLOAD_COEFF * excess.powf(1.5) * availability;
            let flow_dir = if speed > 1e-3 {
                Vec3::new(avg_vel.x, 0.0, avg_vel.z).normalize()
            } else {
                Vec3::ZERO
            };
            self.bed_flux_x[idx] = flow_dir.x * bedload_mag;
            self.bed_flux_z[idx] = flow_dir.z * bedload_mag;

            let sediment_count = self.bed_sediment_count[idx];
            let water_count = self.bed_water_count[idx];
            let total_count = sediment_count + water_count;
            let sediment_conc = if total_count > 0 {
                sediment_count as f32 / total_count as f32
            } else {
                0.0
            };
            let shear_factor = 1.0 - smoothstep(SHIELDS_CRITICAL * 0.7, SHIELDS_CRITICAL * 1.3, theta);
            let deposit_rate = SEDIMENT_SETTLING_VELOCITY * sediment_conc * shear_factor;
            let entrain_rate = ENTRAINMENT_COEFF * excess;
            let desired_delta = (deposit_rate - entrain_rate) * dt;

            if desired_delta > 0.0 {
                deposit_quota[idx] = (desired_delta / particle_height).floor() as u32;
            } else if desired_delta < 0.0 {
                let available_height = (self.bed_height[idx] - self.bed_base_height[idx]).max(0.0);
                let max_erode = (available_height / particle_height).floor() as u32;
                let want = ((-desired_delta) / particle_height).floor() as u32;
                erode_quota[idx] = want.min(max_erode);
            }
        }

        let mut bedload_delta = vec![0.0f32; column_count];
        for k in 0..d {
            for i in 0..w {
                let idx = k * w + i;
                let fx_p = if i + 1 < w { self.bed_flux_x[idx + 1] } else { 0.0 };
                let fx_m = if i > 0 { self.bed_flux_x[idx - 1] } else { 0.0 };
                let fz_p = if k + 1 < d { self.bed_flux_z[idx + w] } else { 0.0 };
                let fz_m = if k > 0 { self.bed_flux_z[idx - w] } else { 0.0 };
                let div = (fx_p - fx_m + fz_p - fz_m) / (2.0 * cell_size);
                bedload_delta[idx] = -div * dt / (1.0 - BED_POROSITY);
            }
        }

        let mut removed = vec![0u32; column_count];
        let mut deposit_quota = deposit_quota;
        let bed_height = &self.bed_height;
        self.sim.particles.list.retain(|p| {
            if p.is_sediment() {
                let i = (p.position.x / cell_size).floor() as i32;
                let k = (p.position.z / cell_size).floor() as i32;
                if i >= 0 && i < w as i32 && k >= 0 && k < d as i32 {
                    let idx = k as usize * w + i as usize;
                    if deposit_quota[idx] > 0 && p.position.y <= bed_height[idx] + sample_height {
                        deposit_quota[idx] -= 1;
                        removed[idx] += 1;
                        return false;
                    }
                }
            }
            true
        });

        let max_bed_height = (GRID_HEIGHT as f32 - 2.0) * cell_size;
        for idx in 0..column_count {
            let mut spawned = 0u32;
            let spawn_count = erode_quota[idx];
            if spawn_count > 0 {
                let avg_vel = self.bed_water_vel_sum[idx];
                let k = idx / w;
                let i = idx % w;
                for _ in 0..spawn_count {
                    if self.sim.particles.len() >= MAX_PARTICLES {
                        break;
                    }
                    let x = (i as f32 + rand_float()) * cell_size;
                    let z = (k as f32 + rand_float()) * cell_size;
                    let y = self.bed_height[idx] + 0.25 * cell_size + rand_float() * 0.5 * cell_size;
                    let mut vel = avg_vel;
                    vel.y = vel.y.max(0.0) + 0.05;
                    self.sim.spawn_sediment(Vec3::new(x, y, z), vel, SEDIMENT_REL_DENSITY);
                    spawned += 1;
                }
            }

            let delta_particles = removed[idx] as i32 - spawned as i32;
            self.bed_height[idx] += delta_particles as f32 * particle_height;
            self.bed_height[idx] += bedload_delta[idx];
            self.bed_height[idx] = self.bed_height[idx].clamp(self.bed_base_height[idx], max_bed_height);
        }
    }

    fn apply_gpu_results(&mut self, count: usize) {
        let limit = count.min(self.sim.particles.len());

        for (idx, p) in self.sim.particles.list.iter_mut().enumerate().take(limit) {
            if idx < self.velocities.len() {
                p.velocity = self.velocities[idx];
                p.affine_velocity = self.c_matrices[idx];
            }
            if idx < self.positions.len() {
                p.position = self.positions[idx];
            }

            // Exit zone detection
            let cell_size = CELL_SIZE;
            let t = (p.position.x / cell_size) / (GRID_WIDTH as f32 - 1.0);
            let t = t.clamp(0.0, 1.0);
            let floor_height = FLOOR_HEIGHT_LEFT as f32 * (1.0 - t) + FLOOR_HEIGHT_RIGHT as f32 * t;
            let exit_start_z = GRID_DEPTH as f32 * cell_size / 6.0;
            let exit_end_z = GRID_DEPTH as f32 * cell_size * 5.0 / 6.0;
            let exit_max_y = (floor_height + 8.0) * cell_size;
            let is_in_exit_zone = p.position.z >= exit_start_z && p.position.z < exit_end_z
                && p.position.y < exit_max_y;

            if p.position.x >= (GRID_WIDTH as f32 - 0.5) * cell_size && is_in_exit_zone {
                p.position.x = 1000.0;
            }
        }

        let before = self.sim.particles.len();
        self.sim.particles.list.retain(|p| p.position.x < 100.0);
        let exited_this_frame = before - self.sim.particles.len();
        self.particles_exited += exited_this_frame as u32;
    }

    fn collect_solids(sim: &FlipSimulation3D) -> Vec<ParticleInstance> {
        let mut solids = Vec::new();
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;
        let cell_size = sim.grid.cell_size;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if sim.grid.is_solid(i, j, k) {
                        let exposed =
                            (i == 0 || !sim.grid.is_solid(i-1, j, k)) ||
                            (i == width-1 || !sim.grid.is_solid(i+1, j, k)) ||
                            (j == 0 || !sim.grid.is_solid(i, j-1, k)) ||
                            (j == height-1 || !sim.grid.is_solid(i, j+1, k)) ||
                            (k == 0 || !sim.grid.is_solid(i, j, k-1)) ||
                            (k == depth-1 || !sim.grid.is_solid(i, j, k+1));

                        if exposed {
                            solids.push(ParticleInstance {
                                position: [
                                    (i as f32 + 0.5) * cell_size,
                                    (j as f32 + 0.5) * cell_size,
                                    (k as f32 + 0.5) * cell_size,
                                ],
                                color: [0.4, 0.35, 0.3, 0.3],
                            });
                        }
                    }
                }
            }
        }
        solids
    }

    fn reset_sim(&mut self) {
        let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
        sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        sim.flip_ratio = 0.97;
        sim.pressure_iterations = 150;

        create_industrial_sluice(&mut sim);

        let bed_base_height = build_bed_base_height();
        self.bed_base_height = bed_base_height.clone();
        self.bed_height = bed_base_height;
        self.bed_water_vel_sum.fill(Vec3::ZERO);
        self.bed_water_count.fill(0);
        self.bed_sediment_count.fill(0);
        self.bed_flux_x.fill(0.0);
        self.bed_flux_z.fill(0.0);
        self.densities.clear();

        if let (Some(gpu_bed), Some(gpu)) = (&self.gpu_bed, &self.gpu) {
            gpu_bed.reset_bed(&gpu.queue, &self.bed_base_height);
        }

        self.sim = sim;
        self.pressure_iters_gpu = self.sim.pressure_iterations as u32;
        self.vorticity_epsilon = VORTICITY_EPSILON_DEFAULT;
        self.gpu_readback_pending = false;
        self.frame = 0;
        self.emitter_enabled = true;
        self.particles_exited = 0;
        self.pending_emit = 0;
        self.gpu_particle_count = 0;
        println!("Reset - emitter enabled");
    }

    fn particle_stats(&self) -> (Vec3, f32, f32, f32) {
        let mut sum_vel = Vec3::ZERO;
        let mut max_vel = 0.0f32;
        let mut max_y = 0.0f32;
        let mut max_x = 0.0f32;

        for p in &self.sim.particles.list {
            sum_vel += p.velocity;
            max_vel = max_vel.max(p.velocity.length());
            max_y = max_y.max(p.position.y);
            max_x = max_x.max(p.position.x);
        }

        let count = self.sim.particles.len() as f32;
        let avg_vel = if count > 0.0 { sum_vel / count } else { Vec3::ZERO };

        (avg_vel, max_vel, max_y, max_x)
    }

    fn sample_sdf_cell(&self, pos: Vec3) -> f32 {
        self.sim.grid.sample_sdf(pos)
    }

    fn bed_min_max(&self, min_i: i32, max_i: i32) -> (f32, f32) {
        if min_i > max_i {
            return (0.0, 0.0);
        }
        let depth = GRID_DEPTH as i32;
        let mut bed_min = f32::INFINITY;
        let mut bed_max = f32::NEG_INFINITY;
        for k in 0..depth {
            for i in min_i..=max_i {
                let idx = k as usize * GRID_WIDTH + i as usize;
                let bed = self.bed_height[idx];
                bed_min = bed_min.min(bed);
                bed_max = bed_max.max(bed);
            }
        }
        if !bed_min.is_finite() {
            bed_min = 0.0;
            bed_max = 0.0;
        }
        (bed_min, bed_max)
    }

    fn probe_riffle_band(&self, min_i: i32, max_i: i32) -> Option<RiffleProbeStats> {
        if min_i > max_i {
            return None;
        }
        let depth = GRID_DEPTH as i32;
        let bed_air_margin = CELL_SIZE * BED_AIR_MARGIN_CELLS;

        let (bed_min, bed_max) = self.bed_min_max(min_i, max_i);

        let mut water_count = 0u32;
        let mut sediment_count = 0u32;
        let mut water_sum_y = 0.0f32;
        let mut sediment_sum_y = 0.0f32;
        let mut water_max_y = 0.0f32;
        let mut sediment_max_y = 0.0f32;
        let mut water_sum_vy = 0.0f32;
        let mut sediment_sum_vy = 0.0f32;
        let mut water_sdf_neg = 0u32;
        let mut sediment_sdf_neg = 0u32;
        let mut water_below_bed = 0u32;
        let mut sediment_below_bed = 0u32;
        let mut water_above_bed = 0u32;
        let mut sediment_above_bed = 0u32;
        let mut water_up = 0u32;
        let mut sediment_up = 0u32;
        let mut water_sum_offset = 0.0f32;
        let mut sediment_sum_offset = 0.0f32;
        let mut water_max_offset = f32::NEG_INFINITY;
        let mut sediment_max_offset = f32::NEG_INFINITY;

        let cell_size = CELL_SIZE;
        for p in &self.sim.particles.list {
            let i = (p.position.x / cell_size).floor() as i32;
            if i < min_i || i > max_i {
                continue;
            }
            let k = (p.position.z / cell_size).floor() as i32;
            if k < 0 || k >= depth {
                continue;
            }
            let idx = k as usize * GRID_WIDTH + i as usize;
            let bed = self.bed_height[idx];
            let sdf = self.sample_sdf_cell(p.position);
            let below_bed = p.position.y < bed;
            let offset = p.position.y - bed;
            let moving_up = p.velocity.y > 0.0;

            if p.is_sediment() {
                sediment_count += 1;
                sediment_sum_y += p.position.y;
                sediment_max_y = sediment_max_y.max(p.position.y);
                sediment_sum_vy += p.velocity.y;
                sediment_sum_offset += offset;
                sediment_max_offset = sediment_max_offset.max(offset);
                if sdf < 0.0 {
                    sediment_sdf_neg += 1;
                }
                if below_bed {
                    sediment_below_bed += 1;
                }
                if offset > bed_air_margin {
                    sediment_above_bed += 1;
                }
                if moving_up {
                    sediment_up += 1;
                }
            } else {
                water_count += 1;
                water_sum_y += p.position.y;
                water_max_y = water_max_y.max(p.position.y);
                water_sum_vy += p.velocity.y;
                water_sum_offset += offset;
                water_max_offset = water_max_offset.max(offset);
                if sdf < 0.0 {
                    water_sdf_neg += 1;
                }
                if below_bed {
                    water_below_bed += 1;
                }
                if offset > bed_air_margin {
                    water_above_bed += 1;
                }
                if moving_up {
                    water_up += 1;
                }
            }
        }

        if water_count == 0 && sediment_count == 0 {
            return None;
        }

        let water_avg_y = if water_count > 0 { water_sum_y / water_count as f32 } else { 0.0 };
        let sediment_avg_y = if sediment_count > 0 { sediment_sum_y / sediment_count as f32 } else { 0.0 };
        let water_avg_vy = if water_count > 0 { water_sum_vy / water_count as f32 } else { 0.0 };
        let sediment_avg_vy = if sediment_count > 0 { sediment_sum_vy / sediment_count as f32 } else { 0.0 };
        let water_avg_offset = if water_count > 0 { water_sum_offset / water_count as f32 } else { 0.0 };
        let sediment_avg_offset = if sediment_count > 0 { sediment_sum_offset / sediment_count as f32 } else { 0.0 };
        if water_max_offset.is_infinite() {
            water_max_offset = 0.0;
        }
        if sediment_max_offset.is_infinite() {
            sediment_max_offset = 0.0;
        }

        Some(RiffleProbeStats {
            water_count,
            sediment_count,
            water_avg_y,
            sediment_avg_y,
            water_max_y,
            sediment_max_y,
            water_avg_vy,
            sediment_avg_vy,
            water_sdf_neg,
            sediment_sdf_neg,
            water_below_bed,
            sediment_below_bed,
            water_above_bed,
            sediment_above_bed,
            water_up,
            sediment_up,
            water_avg_offset,
            sediment_avg_offset,
            water_max_offset,
            sediment_max_offset,
            bed_min,
            bed_max,
        })
    }

    fn probe_first_riffle(&self) -> Option<RiffleProbeStats> {
        let riffle_start = RIFFLE_START_X as i32;
        let riffle_end = riffle_start + RIFFLE_THICKNESS_CELLS - 1;
        let min_i = (riffle_start - RIFFLE_PROBE_PAD).max(0);
        let max_i = (riffle_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);
        self.probe_riffle_band(min_i, max_i)
    }

    fn probe_downstream_riffle(&self) -> Option<RiffleProbeStats> {
        let riffle_start = RIFFLE_START_X as i32;
        let riffle_end = riffle_start + RIFFLE_THICKNESS_CELLS - 1;
        let downstream_start = riffle_end + 1 + RIFFLE_PROBE_PAD;
        let downstream_end = downstream_start + RIFFLE_THICKNESS_CELLS - 1;
        let min_i = (downstream_start - RIFFLE_PROBE_PAD).max(0);
        let max_i = (downstream_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);
        self.probe_riffle_band(min_i, max_i)
    }

    fn sediment_throughput_stats(&self) -> SedimentThroughputStats {
        let riffle_start_x = RIFFLE_START_X as f32 * CELL_SIZE;
        let riffle_end_x = (RIFFLE_START_X + 2) as f32 * CELL_SIZE;
        let downstream_x = riffle_end_x + CELL_SIZE;
        let loft_height = CELL_SIZE * 2.0;
        let cell_size = CELL_SIZE;
        let mut total = 0u32;
        let mut upstream = 0u32;
        let mut at_riffle = 0u32;
        let mut downstream = 0u32;
        let mut max_x = 0.0f32;
        let mut max_y = 0.0f32;
        let mut lofted = 0u32;

        for p in &self.sim.particles.list {
            if !p.is_sediment() {
                continue;
            }
            total += 1;
            max_x = max_x.max(p.position.x);
            max_y = max_y.max(p.position.y);

            if p.position.x < riffle_start_x {
                upstream += 1;
            } else if p.position.x <= riffle_end_x {
                at_riffle += 1;
            } else if p.position.x > downstream_x {
                downstream += 1;
            }

            if p.velocity.y > 0.0 {
                let i = (p.position.x / cell_size).floor() as i32;
                let k = (p.position.z / cell_size).floor() as i32;
                if i >= 0 && i < GRID_WIDTH as i32 && k >= 0 && k < GRID_DEPTH as i32 {
                    let idx = k as usize * GRID_WIDTH + i as usize;
                    let bed = self.bed_height[idx];
                    if p.position.y > bed + loft_height {
                        lofted += 1;
                    }
                }
            }
        }

        SedimentThroughputStats {
            total,
            upstream,
            at_riffle,
            downstream,
            max_x,
            max_y,
            lofted,
        }
    }

    fn material_stats_from_gpu(&self, stats: &[i32], zone: usize, material: usize) -> MaterialProbeStats {
        let base = zone * bed_3d::PROBE_ZONE_STRIDE + material * bed_3d::PROBE_MATERIAL_STRIDE;
        let count = stats[base + bed_3d::PROBE_STAT_COUNT_IDX].max(0) as u32;
        let sum_y = stats[base + bed_3d::PROBE_STAT_SUM_Y_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;
        let max_y = stats[base + bed_3d::PROBE_STAT_MAX_Y_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;
        let sum_vy = stats[base + bed_3d::PROBE_STAT_SUM_VY_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;
        let sum_offset = stats[base + bed_3d::PROBE_STAT_SUM_OFFSET_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;
        let max_offset = stats[base + bed_3d::PROBE_STAT_MAX_OFFSET_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;

        let avg_y = if count > 0 { sum_y / count as f32 } else { 0.0 };
        let avg_vy = if count > 0 { sum_vy / count as f32 } else { 0.0 };
        let avg_offset = if count > 0 { sum_offset / count as f32 } else { 0.0 };

        MaterialProbeStats {
            count,
            avg_y,
            max_y,
            avg_vy,
            sdf_neg: stats[base + bed_3d::PROBE_STAT_SDF_NEG_IDX].max(0) as u32,
            below_bed: stats[base + bed_3d::PROBE_STAT_BELOW_BED_IDX].max(0) as u32,
            above_bed: stats[base + bed_3d::PROBE_STAT_ABOVE_BED_IDX].max(0) as u32,
            avg_offset,
            max_offset,
            up: stats[base + bed_3d::PROBE_STAT_UP_IDX].max(0) as u32,
        }
    }

    fn riffle_stats_from_gpu(
        &self,
        stats: &[i32],
        zone: usize,
        bed_min: f32,
        bed_max: f32,
    ) -> Option<RiffleProbeStats> {
        let water = self.material_stats_from_gpu(stats, zone, 0);
        let sediment = self.material_stats_from_gpu(stats, zone, 1);
        if water.count == 0 && sediment.count == 0 {
            return None;
        }

        Some(RiffleProbeStats {
            water_count: water.count,
            sediment_count: sediment.count,
            water_avg_y: water.avg_y,
            sediment_avg_y: sediment.avg_y,
            water_max_y: water.max_y,
            sediment_max_y: sediment.max_y,
            water_avg_vy: water.avg_vy,
            sediment_avg_vy: sediment.avg_vy,
            water_sdf_neg: water.sdf_neg,
            sediment_sdf_neg: sediment.sdf_neg,
            water_below_bed: water.below_bed,
            sediment_below_bed: sediment.below_bed,
            water_above_bed: water.above_bed,
            sediment_above_bed: sediment.above_bed,
            water_up: water.up,
            sediment_up: sediment.up,
            water_avg_offset: water.avg_offset,
            sediment_avg_offset: sediment.avg_offset,
            water_max_offset: water.max_offset,
            sediment_max_offset: sediment.max_offset,
            bed_min,
            bed_max,
        })
    }

    fn throughput_stats_from_gpu(&self, stats: &[i32]) -> SedimentThroughputStats {
        let base = bed_3d::PROBE_THROUGHPUT_OFFSET;
        let total = stats[base + bed_3d::PROBE_THROUGHPUT_TOTAL_IDX].max(0) as u32;
        let upstream = stats[base + bed_3d::PROBE_THROUGHPUT_UPSTREAM_IDX].max(0) as u32;
        let at_riffle = stats[base + bed_3d::PROBE_THROUGHPUT_AT_RIFFLE_IDX].max(0) as u32;
        let downstream = stats[base + bed_3d::PROBE_THROUGHPUT_DOWNSTREAM_IDX].max(0) as u32;
        let max_x = stats[base + bed_3d::PROBE_THROUGHPUT_MAX_X_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;
        let max_y = stats[base + bed_3d::PROBE_THROUGHPUT_MAX_Y_IDX] as f32 / bed_3d::PROBE_STAT_SCALE;
        let lofted = stats[base + bed_3d::PROBE_THROUGHPUT_LOFTED_IDX].max(0) as u32;

        SedimentThroughputStats {
            total,
            upstream,
            at_riffle,
            downstream,
            max_x,
            max_y,
            lofted,
        }
    }

    fn build_heightfield_vertices(&mut self) -> usize {
        self.heightfield.fill(f32::NEG_INFINITY);

        let width = GRID_WIDTH;
        let depth = GRID_DEPTH;
        let cell_size = CELL_SIZE;

        for p in &self.sim.particles.list {
            let i = (p.position.x / cell_size).floor() as i32;
            let k = (p.position.z / cell_size).floor() as i32;
            if i >= 0 && i < width as i32 && k >= 0 && k < depth as i32 {
                let idx = k as usize * width + i as usize;
                let y = p.position.y;
                if y > self.heightfield[idx] {
                    self.heightfield[idx] = y;
                }
            }
        }

        self.surface_vertices.clear();

        let color = [0.12, 0.5, 0.86, 0.75];
        let y_offset = cell_size * 0.1;

        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let y = self.heightfield[idx];
                if !y.is_finite() {
                    continue;
                }

                let x0 = i as f32 * cell_size;
                let x1 = (i + 1) as f32 * cell_size;
                let z0 = k as f32 * cell_size;
                let z1 = (k + 1) as f32 * cell_size;
                let y = y + y_offset;

                self.surface_vertices.extend_from_slice(&[
                    SurfaceVertex { position: [x0, y, z0], color },
                    SurfaceVertex { position: [x1, y, z0], color },
                    SurfaceVertex { position: [x1, y, z1], color },
                    SurfaceVertex { position: [x0, y, z0], color },
                    SurfaceVertex { position: [x1, y, z1], color },
                    SurfaceVertex { position: [x0, y, z1], color },
                ]);
            }
        }

        self.surface_vertices.len()
    }

    /// Build sparse flow particles colored by velocity
    fn build_flow_particles(&mut self) -> usize {
        self.flow_particles.clear();

        let particles = &self.sim.particles.list;
        let total = particles.len();
        if total == 0 {
            return 0;
        }

        // Sample every Nth particle
        for (i, p) in particles.iter().enumerate() {
            if i % FLOW_PARTICLE_STRIDE != 0 {
                continue;
            }

            let speed = p.velocity.length();
            // Color by speed: water=blue/cyan, sediment=golden/brown
            let t = (speed / 2.5).min(1.0);
            let color = if p.is_sediment() {
                [
                    0.45 + t * 0.4,
                    0.35 + t * 0.3,
                    0.15 + t * 0.2,
                    0.7,
                ]
            } else {
                [
                    0.1 + t * 0.9,      // R: dark -> bright
                    0.3 + t * 0.7,      // G: blue -> cyan
                    0.7 + t * 0.3,      // B: stays high
                    0.6 + t * 0.3,      // A: more visible when fast
                ]
            };

            self.flow_particles.push(ParticleInstance {
                position: [p.position.x, p.position.y, p.position.z],
                color,
            });
        }

        self.flow_particles.len()
    }

    /// Build ALL particle instances for direct rendering (when heightfield is OFF)
    fn build_all_particle_instances(&self) -> Vec<ParticleInstance> {
        self.sim.particles.list.iter().map(|p| {
            let speed = p.velocity.length();
            let t = (speed / 2.5).min(1.0);
            let color = if p.is_sediment() {
                [
                    0.6 + t * 0.2,
                    0.5 + t * 0.2,
                    0.2 + t * 0.1,
                    0.85,
                ]
            } else {
                [
                    0.2 + t * 0.6,
                    0.4 + t * 0.4,
                    0.8,
                    0.8,
                ]
            };
            ParticleInstance {
                position: [p.position.x, p.position.y, p.position.z],
                color,
            }
        }).collect()
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        if !self.paused {
            let dt = 1.0 / 60.0;

            if self.use_gpu_sim {
                let bed_params = build_gpu_bed_params(dt);
                if self.use_async_readback {
                    if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                        if self.gpu_readback_pending {
                            if let Some(count) = gpu_flip.try_readback(
                                &gpu.device,
                                &mut self.positions,
                                &mut self.velocities,
                                &mut self.c_matrices,
                            ) {
                                self.gpu_readback_pending = false;
                                self.apply_gpu_results(count);
                            }
                        }
                    }
                }

                let should_schedule = if self.use_async_readback {
                    !self.gpu_readback_pending
                } else {
                    true
                };

                let do_sync = self.frame % GPU_SYNC_STRIDE == 0;
                if self.emitter_enabled && self.frame % 2 == 0 {
                    self.pending_emit = self.pending_emit.saturating_add(200);
                }

                if should_schedule {
                    if do_sync {
                        if self.pending_emit > 0 {
                            self.emit_particles(self.pending_emit);
                            self.pending_emit = 0;
                        }

                        if self.gpu_bed.is_none() {
                            self.update_sediment_bed(dt);
                        }

                        self.positions.clear();
                        self.velocities.clear();
                        self.c_matrices.clear();
                        self.densities.clear();

                        for p in &self.sim.particles.list {
                            self.positions.push(p.position);
                            self.velocities.push(p.velocity);
                            self.c_matrices.push(p.affine_velocity);
                            self.densities.push(p.density);
                        }

                        let w = self.sim.grid.width;
                        let h = self.sim.grid.height;
                        let d = self.sim.grid.depth;
                        self.cell_types.clear();
                        self.cell_types.resize(w * h * d, 0);

                        for k in 0..d {
                            for i in 0..w {
                                for j in 0..h {
                                    let idx = k * w * h + j * w + i;
                                    if self.sim.grid.is_solid(i, j, k) {
                                        self.cell_types[idx] = 2;
                                    }
                                }
                            }
                        }

                        for p in &self.sim.particles.list {
                            if p.is_sediment() {
                                continue;
                            }
                            let i = (p.position.x / CELL_SIZE).floor() as i32;
                            let j = (p.position.y / CELL_SIZE).floor() as i32;
                            let k = (p.position.z / CELL_SIZE).floor() as i32;
                            if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                                let idx = k as usize * w * h + j as usize * w + i as usize;
                                if self.cell_types[idx] != 2 {
                                    self.cell_types[idx] = 1;
                                }
                            }
                        }

                        let pressure_iters = self.pressure_iters_gpu;
                        let flow_accel = flow_accel_from_slope();
                        if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                            gpu_flip.vorticity_epsilon = self.vorticity_epsilon;
                            gpu_flip.sediment_rest_particles = SEDIMENT_REST_PARTICLES;
                            gpu_flip.sediment_settling_velocity = SEDIMENT_SETTLING_VELOCITY;
                            let sdf = self.sim.grid.sdf.as_slice();
                            let positions = &mut self.positions;
                            let velocities = &mut self.velocities;
                            let c_matrices = &mut self.c_matrices;
                            let densities = &self.densities;
                            let cell_types = &self.cell_types;
                            let bed_height = if self.gpu_bed.is_some() {
                                None
                            } else {
                                Some(self.bed_height.as_slice())
                            };
                            if self.use_async_readback {
                                if gpu_flip.step_async(
                                    &gpu.device,
                                    &gpu.queue,
                                    positions,
                                    velocities,
                                    c_matrices,
                                    densities,
                                    cell_types,
                                    Some(sdf),
                                    bed_height,
                                    dt,
                                    -9.8,
                                    flow_accel,
                                    pressure_iters,
                                ) {
                                    self.gpu_readback_pending = true;
                                } else {
                                    gpu_flip.step(
                                        &gpu.device,
                                        &gpu.queue,
                                        positions,
                                        velocities,
                                        c_matrices,
                                        densities,
                                        cell_types,
                                        Some(sdf),
                                        bed_height,
                                        dt,
                                        -9.8,
                                        flow_accel,
                                        pressure_iters,
                                    );
                                    self.apply_gpu_results(self.positions.len());
                                }
                            } else {
                                gpu_flip.step(
                                    &gpu.device,
                                    &gpu.queue,
                                    positions,
                                    velocities,
                                    c_matrices,
                                    densities,
                                    cell_types,
                                    Some(sdf),
                                    bed_height,
                                    dt,
                                    -9.8,
                                    flow_accel,
                                    pressure_iters,
                                );
                                self.apply_gpu_results(self.positions.len());
                            }
                        }
                        self.gpu_particle_count = self.positions.len() as u32;
                        if let (Some(gpu_bed), Some(gpu)) = (&mut self.gpu_bed, &self.gpu) {
                            gpu_bed.update(&gpu.device, &gpu.queue, self.gpu_particle_count, &bed_params);
                            gpu_bed.read_bed_height(&gpu.device, &gpu.queue, &mut self.bed_height);
                        }
                    } else {
                        let pressure_iters = self.pressure_iters_gpu;
                        let flow_accel = flow_accel_from_slope();
                        if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                            let sdf = self.sim.grid.sdf.as_slice();
                            let cell_types = &self.cell_types;
                            let bed_height = if self.gpu_bed.is_some() {
                                None
                            } else {
                                Some(self.bed_height.as_slice())
                            };
                            gpu_flip.step_in_place(
                                &gpu.device,
                                &gpu.queue,
                                self.gpu_particle_count,
                                cell_types,
                                Some(sdf),
                                bed_height,
                                dt,
                                -9.8,
                                flow_accel,
                                pressure_iters,
                            );
                        }
                        if let (Some(gpu_bed), Some(gpu)) = (&mut self.gpu_bed, &self.gpu) {
                            gpu_bed.update(&gpu.device, &gpu.queue, self.gpu_particle_count, &bed_params);
                        }
                    }
                }
            } else {
                if self.emitter_enabled && self.frame % 2 == 0 {
                    self.emit_particles(200);
                }
                self.sim.update(dt);
            }

            self.frame += 1;
        }

        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.last_fps_time = now;
            if self.use_gpu_sim && !self.paused {
                let mut iters = self.pressure_iters_gpu;
                if self.current_fps < TARGET_FPS - 5.0 {
                    iters = iters.saturating_sub(PRESSURE_ITERS_STEP);
                } else if self.current_fps > TARGET_FPS + 5.0 {
                    iters = iters.saturating_add(PRESSURE_ITERS_STEP);
                }
                iters = iters.max(PRESSURE_ITERS_MIN).min(PRESSURE_ITERS_MAX);
                self.pressure_iters_gpu = iters;
            }
        }

        // Print stats every 30 frames (less spam for larger sim)
        if self.frame % 30 == 0 {
            let (avg_vel, _max_vel, max_y, max_x) = self.particle_stats();
            let mode = if self.use_gpu_sim { "GPU" } else { "CPU" };
            println!(
                "[{}] Frame {:5} | FPS: {:5.1} | Particles: {:6} | Exited: {:6} | PIters: {:3} | AvgVel: ({:6.2}, {:5.2}, {:5.2}) | MaxY: {:.3} | MaxX: {:.3}",
                mode,
                self.frame,
                self.current_fps,
                self.sim.particles.len(),
                self.particles_exited,
                self.pressure_iters_gpu,
                avg_vel.x, avg_vel.y, avg_vel.z,
                max_y,
                max_x,
            );

            if self.debug_riffle_probe {
                if self.use_gpu_sim {
                    if let (Some(gpu_bed), Some(gpu)) = (&self.gpu_bed, &self.gpu) {
                        gpu_bed.read_probe_stats(&gpu.device, &gpu.queue, &mut self.gpu_probe_stats);

                        let riffle_start = RIFFLE_START_X as i32;
                        let riffle_end = riffle_start + RIFFLE_THICKNESS_CELLS - 1;
                        let min_i = (riffle_start - RIFFLE_PROBE_PAD).max(0);
                        let max_i = (riffle_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);
                        let (bed_min, bed_max) = self.bed_min_max(min_i, max_i);

                        if let Some(stats) = self.riffle_stats_from_gpu(
                            &self.gpu_probe_stats,
                            bed_3d::PROBE_ZONE_RIFFLE,
                            bed_min,
                            bed_max,
                        ) {
                            println!(
                                "[Probe] Riffle x={}..{} bed[{:.3},{:.3}] | water n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={} | sediment n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={}",
                                min_i,
                                max_i,
                                stats.bed_min,
                                stats.bed_max,
                                stats.water_count,
                                stats.water_avg_y,
                                stats.water_max_y,
                                stats.water_avg_vy,
                                stats.water_sdf_neg,
                                stats.water_below_bed,
                                stats.water_above_bed,
                                stats.water_avg_offset,
                                stats.water_max_offset,
                                stats.water_up,
                                stats.sediment_count,
                                stats.sediment_avg_y,
                                stats.sediment_max_y,
                                stats.sediment_avg_vy,
                                stats.sediment_sdf_neg,
                                stats.sediment_below_bed,
                                stats.sediment_above_bed,
                                stats.sediment_avg_offset,
                                stats.sediment_max_offset,
                                stats.sediment_up,
                            );
                        } else {
                            println!("[Probe] Riffle x={}..{} no particles", RIFFLE_START_X, RIFFLE_START_X + 1);
                        }

                        let downstream_start = riffle_end + 1 + RIFFLE_PROBE_PAD;
                        let downstream_end = downstream_start + RIFFLE_THICKNESS_CELLS - 1;
                        let min_i = (downstream_start - RIFFLE_PROBE_PAD).max(0);
                        let max_i = (downstream_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);
                        let (bed_min, bed_max) = self.bed_min_max(min_i, max_i);

                        if let Some(stats) = self.riffle_stats_from_gpu(
                            &self.gpu_probe_stats,
                            bed_3d::PROBE_ZONE_DOWNSTREAM,
                            bed_min,
                            bed_max,
                        ) {
                            println!(
                                "[Probe] Downstream x={}..{} bed[{:.3},{:.3}] | water n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={} | sediment n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={}",
                                min_i,
                                max_i,
                                stats.bed_min,
                                stats.bed_max,
                                stats.water_count,
                                stats.water_avg_y,
                                stats.water_max_y,
                                stats.water_avg_vy,
                                stats.water_sdf_neg,
                                stats.water_below_bed,
                                stats.water_above_bed,
                                stats.water_avg_offset,
                                stats.water_max_offset,
                                stats.water_up,
                                stats.sediment_count,
                                stats.sediment_avg_y,
                                stats.sediment_max_y,
                                stats.sediment_avg_vy,
                                stats.sediment_sdf_neg,
                                stats.sediment_below_bed,
                                stats.sediment_above_bed,
                                stats.sediment_avg_offset,
                                stats.sediment_max_offset,
                                stats.sediment_up,
                            );
                        } else {
                            println!("[Probe] Downstream no particles");
                        }

                        let sed_stats = self.throughput_stats_from_gpu(&self.gpu_probe_stats);
                        println!(
                            "[Sediment] total={} upstream={} at_riffle={} downstream={} max_x={:.3} max_y={:.3} lofted={}",
                            sed_stats.total,
                            sed_stats.upstream,
                            sed_stats.at_riffle,
                            sed_stats.downstream,
                            sed_stats.max_x,
                            sed_stats.max_y,
                            sed_stats.lofted,
                        );
                    } else {
                        println!("[Probe] GPU bed not initialized");
                    }
                } else {
                    if let Some(stats) = self.probe_first_riffle() {
                        let riffle_start = RIFFLE_START_X as i32;
                        let riffle_end = riffle_start + RIFFLE_THICKNESS_CELLS - 1;
                        let min_i = (riffle_start - RIFFLE_PROBE_PAD).max(0);
                        let max_i = (riffle_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);
                        println!(
                            "[Probe] Riffle x={}..{} bed[{:.3},{:.3}] | water n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={} | sediment n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={}",
                            min_i,
                            max_i,
                            stats.bed_min,
                            stats.bed_max,
                            stats.water_count,
                            stats.water_avg_y,
                            stats.water_max_y,
                            stats.water_avg_vy,
                            stats.water_sdf_neg,
                            stats.water_below_bed,
                            stats.water_above_bed,
                            stats.water_avg_offset,
                            stats.water_max_offset,
                            stats.water_up,
                            stats.sediment_count,
                            stats.sediment_avg_y,
                            stats.sediment_max_y,
                            stats.sediment_avg_vy,
                            stats.sediment_sdf_neg,
                            stats.sediment_below_bed,
                            stats.sediment_above_bed,
                            stats.sediment_avg_offset,
                            stats.sediment_max_offset,
                            stats.sediment_up,
                        );
                    } else {
                        println!("[Probe] Riffle x={}..{} no particles", RIFFLE_START_X, RIFFLE_START_X + 1);
                    }
                    if let Some(stats) = self.probe_downstream_riffle() {
                        let riffle_start = RIFFLE_START_X as i32;
                        let riffle_end = riffle_start + RIFFLE_THICKNESS_CELLS - 1;
                        let downstream_start = riffle_end + 1 + RIFFLE_PROBE_PAD;
                        let downstream_end = downstream_start + RIFFLE_THICKNESS_CELLS - 1;
                        let min_i = (downstream_start - RIFFLE_PROBE_PAD).max(0);
                        let max_i = (downstream_end + RIFFLE_PROBE_PAD).min(GRID_WIDTH as i32 - 1);
                        println!(
                            "[Probe] Downstream x={}..{} bed[{:.3},{:.3}] | water n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={} | sediment n={} avg_y={:.3} max_y={:.3} avg_vy={:.3} sdf<0={} below_bed={} above_bed={} avg_off={:.3} max_off={:.3} up={}",
                            min_i,
                            max_i,
                            stats.bed_min,
                            stats.bed_max,
                            stats.water_count,
                            stats.water_avg_y,
                            stats.water_max_y,
                            stats.water_avg_vy,
                            stats.water_sdf_neg,
                            stats.water_below_bed,
                            stats.water_above_bed,
                            stats.water_avg_offset,
                            stats.water_max_offset,
                            stats.water_up,
                            stats.sediment_count,
                            stats.sediment_avg_y,
                            stats.sediment_max_y,
                            stats.sediment_avg_vy,
                            stats.sediment_sdf_neg,
                            stats.sediment_below_bed,
                            stats.sediment_above_bed,
                            stats.sediment_avg_offset,
                            stats.sediment_max_offset,
                            stats.sediment_up,
                        );
                    } else {
                        println!("[Probe] Downstream no particles");
                    }
                    let sed_stats = self.sediment_throughput_stats();
                    println!(
                        "[Sediment] total={} upstream={} at_riffle={} downstream={} max_x={:.3} max_y={:.3} lofted={}",
                        sed_stats.total,
                        sed_stats.upstream,
                        sed_stats.at_riffle,
                        sed_stats.downstream,
                        sed_stats.max_x,
                        sed_stats.max_y,
                        sed_stats.lofted,
                    );
                }
            }
        }

        // Build geometry based on render mode
        let (surface_vertex_count, flow_particle_count, all_instances) = if self.render_heightfield {
            let svc = self.build_heightfield_vertices();
            let fpc = if self.render_flow_particles {
                self.build_flow_particles()
            } else {
                0
            };
            (svc, fpc, Vec::new())
        } else {
            // Non-heightfield mode: draw all particles
            (0, 0, self.build_all_particle_instances())
        };
        let instance_count = all_instances.len();

        let window = self.window.as_ref().unwrap().clone();
        let gpu = self.gpu.as_mut().unwrap();

        // Upload buffers
        if surface_vertex_count > 0 {
            gpu.queue.write_buffer(
                &gpu.surface_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.surface_vertices),
            );
        }
        if flow_particle_count > 0 {
            gpu.queue.write_buffer(
                &gpu.instance_buffer,
                0,
                bytemuck::cast_slice(&self.flow_particles),
            );
        }
        if instance_count > 0 {
            gpu.queue.write_buffer(
                &gpu.instance_buffer,
                0,
                bytemuck::cast_slice(&all_instances),
            );
        }

        // Camera centered on the sluice
        let center = Vec3::new(
            GRID_WIDTH as f32 * CELL_SIZE * 0.5,
            GRID_HEIGHT as f32 * CELL_SIZE * 0.3,
            GRID_DEPTH as f32 * CELL_SIZE * 0.5,
        );
        let eye = center + Vec3::new(
            self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
        );

        let view = Mat4::look_at_rh(eye, center, Vec3::Y);
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0);
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let frame = gpu.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_bind_group(0, &gpu.bind_group, &[]);

            // Draw solid cells
            pass.set_pipeline(&gpu.pipeline);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.solid_buffer.slice(..));
            pass.draw(0..4, 0..self.solid_instances.len() as u32);

            if self.render_heightfield {
                // Draw water surface
                if surface_vertex_count > 0 {
                    pass.set_pipeline(&gpu.surface_pipeline);
                    pass.set_vertex_buffer(0, gpu.surface_vertex_buffer.slice(..));
                    pass.draw(0..surface_vertex_count as u32, 0..1);
                }
                // Draw sparse flow particles to show interior flow
                if self.render_flow_particles && flow_particle_count > 0 {
                    pass.set_pipeline(&gpu.pipeline);
                    pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                    pass.draw(0..4, 0..flow_particle_count as u32);
                }
            } else if instance_count > 0 {
                // Draw particles directly
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                pass.draw(0..4, 0..instance_count as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        window.request_redraw();
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
                        .with_title("Industrial Sluice - 160x40x32 Grid")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1400, 900)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find adapter");

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 512 * 1024 * 1024; // 512MB for larger buffers

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
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
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create GPU FLIP solver for industrial scale
        let gpu_flip = GpuFlip3D::new(
            &device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );
        let positions_buffer = gpu_flip.positions_buffer();
        let velocities_buffer = gpu_flip.velocities_buffer();
        let densities_buffer = gpu_flip.densities_buffer();
        let gpu_bed = GpuBed3D::new(
            &device,
            &queue,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            positions_buffer.as_ref(),
            velocities_buffer.as_ref(),
            densities_buffer.as_ref(),
            gpu_flip.sdf_buffer(),
            gpu_flip.bed_height_buffer(),
            &self.bed_base_height,
        );
        self.gpu_bed = Some(gpu_bed);
        self.gpu_flip = Some(gpu_flip);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let surface_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Shader"),
            source: wgpu::ShaderSource::Wgsl(HEIGHTFIELD_SHADER.into()),
        });

        let vertices = [
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [1.0, -1.0] },
            Vertex { position: [-1.0, 1.0] },
            Vertex { position: [1.0, 1.0] },
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
        let surface_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Vertex Buffer"),
            size: (MAX_SURFACE_VERTICES * std::mem::size_of::<SurfaceVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let solid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Solid Buffer"),
            contents: bytemuck::cast_slice(&self.solid_instances),
            usage: wgpu::BufferUsages::VERTEX,
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
            label: Some("Pipeline"),
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
        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &surface_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SurfaceVertex>() as u64,
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
                module: &surface_shader,
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
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            instance_buffer,
            solid_buffer,
            surface_vertex_buffer,
            uniform_buffer,
            bind_group,
            surface_pipeline,
        });

        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                            println!("Paused: {}", self.paused);
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.reset_sim();
                            println!("Reset simulation");
                        }
                        PhysicalKey::Code(KeyCode::KeyG) => {
                            self.use_gpu_sim = !self.use_gpu_sim;
                            self.gpu_readback_pending = false;
                            println!("Simulation mode: {}", if self.use_gpu_sim { "GPU" } else { "CPU" });
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            self.emitter_enabled = !self.emitter_enabled;
                            println!("Emitter: {}", if self.emitter_enabled { "ON" } else { "OFF" });
                        }
                        PhysicalKey::Code(KeyCode::KeyH) => {
                            self.render_heightfield = !self.render_heightfield;
                            println!("Heightfield: {}", if self.render_heightfield { "ON" } else { "OFF" });
                        }
                        PhysicalKey::Code(KeyCode::KeyF) => {
                            self.render_flow_particles = !self.render_flow_particles;
                            println!("Flow particles: {}", if self.render_flow_particles { "ON" } else { "OFF" });
                        }
                        PhysicalKey::Code(KeyCode::KeyV) => {
                            self.vorticity_epsilon = (self.vorticity_epsilon - VORTICITY_EPSILON_STEP).max(0.0);
                            println!("Vorticity epsilon: {:.3}", self.vorticity_epsilon);
                        }
                        PhysicalKey::Code(KeyCode::KeyB) => {
                            self.vorticity_epsilon =
                                (self.vorticity_epsilon + VORTICITY_EPSILON_STEP).min(VORTICITY_EPSILON_MAX);
                            println!("Vorticity epsilon: {:.3}", self.vorticity_epsilon);
                        }
                        PhysicalKey::Code(KeyCode::KeyP) => {
                            self.debug_riffle_probe = !self.debug_riffle_probe;
                            println!("Riffle probe: {}", if self.debug_riffle_probe { "ON" } else { "OFF" });
                        }
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
                        self.camera_pitch = (self.camera_pitch + dy * 0.01).clamp(-1.5, 1.5);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.5).clamp(1.0, 30.0);
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

// Smaller particle size for denser look at industrial scale
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
    let size = 0.008;  // Smaller particles for industrial scale

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

const HEIGHTFIELD_SHADER: &str = r#"
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
