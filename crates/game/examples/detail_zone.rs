//! Detail Zone Test - FLIP + Heightfield Integration
//!
//! Tests 3D FLIP fluid simulation over a terrain heightfield.
//! Uses FlipSimulation3D for particle/grid data and GpuFlip3D for GPU compute.
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look
//! - P: Toggle Pointer/Dig mode
//!   - Dig mode: click to add water/dig terrain
//!   - Pointer mode: hover highlights sluice, click to focus
//! - Left click (Pointer mode): Focus on sluice detail view
//! - F or Middle click: Toggle focus (alternative)
//! - 1: Add heightfield water at cursor
//! - 2: Add muddy water at cursor
//! - 3: Toggle heightfield emitter
//! - V: Toggle velocity coloring (blue=slow -> red=fast)
//! - X: Reposition detail emitter to cursor (in focus mode)
//! - R: Reset
//! - ESC: Quit
//!
//! Run: cargo run --example detail_zone --release

use bytemuck::{Pod, Zeroable};
use game::equipment_geometry::SluiceVertex;
use game::gpu::flip_3d::GpuFlip3D;
use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder};
use glam::{Mat3, Mat4, Vec3, Vec4};
use sim3d::{
    constants, ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FlipSimulation3D, Grid3D, SdfParams,
    TerrainMaterial, World,
};
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

const WORLD_WIDTH: usize = 512;
const WORLD_DEPTH: usize = 512;
const CELL_SIZE: f32 = 1.0;
const INITIAL_HEIGHT: f32 = 10.0;

const DIG_RADIUS: f32 = 3.0;
const DIG_DEPTH: f32 = 0.5;
const ADD_RADIUS: f32 = 3.0;
const ADD_HEIGHT: f32 = 0.5;
const WATER_ADD_VOLUME: f32 = 5.0;

const DETAIL_PAD_RADIUS: f32 = 6.0;
const DETAIL_PAD_FALLOFF: f32 = 4.0;
const DETAIL_PAD_HEIGHT_OFFSET: f32 = -1.5;
const DETAIL_SLUICE_CELL_SIZE: f32 = 0.02;
const DETAIL_SLUICE_WIDTH: usize = 150;
const DETAIL_SLUICE_EXIT_BUFFER: usize = 12;
const DETAIL_FLIP_GRID_X: usize = DETAIL_SLUICE_WIDTH + DETAIL_SLUICE_EXIT_BUFFER;
const DETAIL_SLUICE_HEIGHT: usize = 52;
const DETAIL_SLUICE_DEPTH: usize = 40;
const DETAIL_SLUICE_HEIGHT_OFFSET: f32 = 0.05;
const DETAIL_FLIP_GRID_Y: usize = DETAIL_SLUICE_HEIGHT;
const DETAIL_FLIP_GRID_Z: usize = DETAIL_SLUICE_DEPTH;
const DETAIL_FLIP_CELL_SIZE: f32 = DETAIL_SLUICE_CELL_SIZE;
const FLIP_PRESSURE_ITERS: usize = 120;
const MAX_FLIP_PARTICLES: usize = 300_000;
const WATER_EMIT_RATE: usize = 200;
const SEDIMENT_EMIT_RATE: usize = 2;
const PARTICLE_SIZE: f32 = DETAIL_SLUICE_CELL_SIZE * 0.6;
const GANGUE_DENSITY: f32 = constants::GANGUE_DENSITY;
const GOLD_DENSITY: f32 = constants::GOLD_DENSITY;
const GOLD_FRACTION: f32 = 0.05;
const GANGUE_RADIUS_CELLS: f32 = 0.12;
const GOLD_RADIUS_CELLS: f32 = 0.02;
const DETAIL_EMITTER_VISUAL_RADIUS: f32 = 0.05;

const MOVE_SPEED: f32 = 20.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

const STEPS_PER_FRAME: usize = 10; // More steps for faster filling
const DT: f32 = 0.02;
const DEBUG_HEIGHTFIELD_STATS: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionMode {
    Dig,     // Default: click to add water/dig terrain
    Pointer, // Hover highlights equipment, click to focus
}

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
    particle_size: f32,
    camera_right: [f32; 3],
    show_velocity: f32, // 1.0 = velocity coloring, 0.0 = normal water blue
    camera_up: [f32; 3],
    highlight_tint: f32, // 1.0 = normal, >1.0 = brighter
}

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    vertex_capacity: usize,
    index_capacity: usize,
}

impl Mesh {
    fn new(device: &wgpu::Device, vertices: &[WorldVertex], indices: &[u32], label: &str) -> Self {
        let (vertex_buffer, vertex_capacity) = Self::create_vertex_buffer(device, vertices, label);
        let (index_buffer, index_capacity) = Self::create_index_buffer(device, indices, label);

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            vertex_capacity,
            index_capacity,
        }
    }

    fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[WorldVertex],
        indices: &[u32],
        label: &str,
    ) {
        if !vertices.is_empty() {
            if vertices.len() > self.vertex_capacity {
                let (buffer, capacity) = Self::create_vertex_buffer(device, vertices, label);
                self.vertex_buffer = buffer;
                self.vertex_capacity = capacity;
            } else {
                queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
            }
        }

        if !indices.is_empty() {
            if indices.len() > self.index_capacity {
                let (buffer, capacity) = Self::create_index_buffer(device, indices, label);
                self.index_buffer = buffer;
                self.index_capacity = capacity;
            } else {
                queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
            }
        }

        self.num_indices = indices.len() as u32;
    }

    fn create_vertex_buffer(
        device: &wgpu::Device,
        vertices: &[WorldVertex],
        label: &str,
    ) -> (wgpu::Buffer, usize) {
        let capacity = vertices.len().max(1);
        let data = if vertices.is_empty() {
            [WorldVertex::default()]
        } else {
            [vertices[0]]
        };

        let contents = if vertices.is_empty() {
            bytemuck::cast_slice(&data)
        } else {
            bytemuck::cast_slice(vertices)
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", label)),
            contents,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, capacity)
    }

    fn create_index_buffer(
        device: &wgpu::Device,
        indices: &[u32],
        label: &str,
    ) -> (wgpu::Buffer, usize) {
        let capacity = indices.len().max(1);
        let data = if indices.is_empty() {
            [0u32]
        } else {
            [indices[0]]
        };

        let contents = if indices.is_empty() {
            bytemuck::cast_slice(&data)
        } else {
            bytemuck::cast_slice(indices)
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", label)),
            contents,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, capacity)
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

use game::water_emitter::WaterEmitter;

use game::gpu::heightfield::GpuHeightfield;

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
    // Particle rendering
    particle_pipeline: wgpu::RenderPipeline,
    particle_vertex_buffer: wgpu::Buffer,
    particle_instance_buffer: wgpu::Buffer,
    sluice_vertex_buffer: wgpu::Buffer,
    sluice_index_buffer: wgpu::Buffer,
    sluice_index_count: u32,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    world: World,
    emitter: WaterEmitter,
    sluice_config: SluiceConfig,
    pending_water_emits: usize,
    pending_sediment_emits: usize,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    last_stats: Instant,
    start_time: Instant,
    window_size: (u32, u32),
    selected_material: u32,
    terrain_dirty: bool,
    show_water: bool,
    show_velocity: bool, // Toggle velocity-based particle coloring

    // FLIP simulation
    flip_sim: FlipSimulation3D,
    flip_origin: Vec3,
    gpu_flip: Option<GpuFlip3D>,
    flip_positions: Vec<Vec3>,
    flip_velocities: Vec<Vec3>,
    flip_c_matrices: Vec<Mat3>,
    flip_densities: Vec<f32>,
    flip_cell_types: Vec<u32>,
    dem: ClusterSimulation3D,
    gangue_template_idx: usize,
    gold_template_idx: usize,
    sediment_flip_indices: Vec<usize>,
    sluice_vertices: Vec<SluiceVertex>,
    sluice_indices: Vec<u32>,
    focus_mode: bool,
    focus_bounds_min: Vec3,
    focus_bounds_max: Vec3,
    interaction_mode: InteractionMode,

    sluice_hovered: bool,
    emitter_mesh: Option<Mesh>,
    detail_emitter_mesh: Option<Mesh>,
    detail_emitter_pos: Vec3,
}

impl App {
    fn new() -> Self {
        let world = build_world();

        let sluice_config = detail_sluice_config();
        let sluice_offset = detail_sluice_offset(&world, &sluice_config);

        let mut flip_sim = FlipSimulation3D::new(
            DETAIL_FLIP_GRID_X,
            DETAIL_FLIP_GRID_Y,
            DETAIL_FLIP_GRID_Z,
            DETAIL_FLIP_CELL_SIZE,
        );
        flip_sim.gravity = Vec3::new(0.0, constants::GRAVITY, 0.0);
        flip_sim.flip_ratio = 0.95;
        flip_sim.pressure_iterations = FLIP_PRESSURE_ITERS;

        let (sluice_vertices, sluice_indices) =
            prepare_sluice_mesh(&sluice_config, &mut flip_sim.grid, sluice_offset);
        flip_sim.grid.compute_sdf();

        let dem_bounds_max = Vec3::new(
            DETAIL_FLIP_GRID_X as f32 * DETAIL_FLIP_CELL_SIZE,
            DETAIL_FLIP_GRID_Y as f32 * DETAIL_FLIP_CELL_SIZE,
            DETAIL_FLIP_GRID_Z as f32 * DETAIL_FLIP_CELL_SIZE,
        );
        let mut dem = ClusterSimulation3D::new(Vec3::ZERO, dem_bounds_max);
        dem.gravity = Vec3::ZERO;
        dem.restitution = 0.0;
        dem.friction = 0.5;
        dem.floor_friction = 0.6;
        dem.normal_stiffness = 5000.0;
        dem.tangential_stiffness = 2500.0;
        dem.rolling_friction = 0.05;
        dem.wet_friction = 0.08;
        dem.wet_rolling_friction = 0.005;
        dem.use_dem = true;

        let gangue_radius = DETAIL_SLUICE_CELL_SIZE * GANGUE_RADIUS_CELLS;
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

        let gold_radius = DETAIL_SLUICE_CELL_SIZE * GOLD_RADIUS_CELLS;
        let gold_mass = GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * gold_radius.powi(3);
        let gold_template = ClumpTemplate3D::generate(ClumpShape3D::Flat4, gold_radius, gold_mass);
        let gold_template_idx = dem.add_template(gold_template);

        println!("=== DETAIL ZONE: SLUICE FLIP ===");
        println!(
            "FLIP grid: {}x{}x{} at {:.2}m",
            DETAIL_FLIP_GRID_X, DETAIL_FLIP_GRID_Y, DETAIL_FLIP_GRID_Z, DETAIL_FLIP_CELL_SIZE
        );
        println!("Constant FLIP emitter filling the sluice");

        let center_x = (WORLD_WIDTH / 2) as usize;
        let center_z = (WORLD_DEPTH / 2) as usize;
        let terrain_y = world.ground_height(center_x, center_z);
        let flip_origin = sluice_offset; // Align FLIP grid with the sluice mesh

        let sluice_length = sluice_config.grid_width as f32 * sluice_config.cell_size;
        let sluice_height = sluice_config.grid_height as f32 * sluice_config.cell_size;
        let sluice_depth = sluice_config.grid_depth as f32 * sluice_config.cell_size;
        let focus_bounds_min = sluice_offset;
        let focus_bounds_max =
            sluice_offset + Vec3::new(sluice_length, sluice_height, sluice_depth);

        Self {
            window: None,
            gpu: None,
            world,
            emitter: {
                let mut e = WaterEmitter::new(
                    Vec3::new(center_x as f32, terrain_y + 10.0, center_z as f32),
                    7500.0, // rate
                    15.0,   // radius
                );
                e.sediment_conc = 0.15;
                e.overburden_conc = 0.05;
                e.gravel_conc = 0.03;
                e.paydirt_conc = 0.02;
                e.enabled = true;
                e
            },
            sluice_config: sluice_config.clone(),
            pending_water_emits: 0,
            pending_sediment_emits: 0,
            camera: Camera {
                position: Vec3::new(center_x as f32, terrain_y + 50.0, center_z as f32 + 50.0),
                yaw: -1.57,  // Looking -Z
                pitch: -0.7, // Tilted down
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
            selected_material: 2,
            terrain_dirty: true,
            show_water: true,
            show_velocity: false,

            // FLIP
            flip_sim,
            flip_origin,
            gpu_flip: None, // Initialized when GPU is ready
            flip_positions: Vec::new(),
            flip_velocities: Vec::new(),
            flip_c_matrices: Vec::new(),
            flip_densities: Vec::new(),
            flip_cell_types: Vec::new(),
            dem,
            gangue_template_idx,
            gold_template_idx,
            sediment_flip_indices: Vec::new(),
            sluice_vertices,
            sluice_indices,
            focus_mode: false,
            focus_bounds_min,
            focus_bounds_max,
            interaction_mode: InteractionMode::Dig,

            sluice_hovered: false,
            emitter_mesh: None,
            detail_emitter_mesh: None,
            detail_emitter_pos: Vec3::new(
                2.0 * DETAIL_FLIP_CELL_SIZE,                               // emit_x
                0.0, // Will be computed from sluice floor
                (DETAIL_SLUICE_DEPTH as f32 * DETAIL_FLIP_CELL_SIZE) * 0.5, // center_z
            ),
        }
    }

    fn reset_world(&mut self) {
        self.world = build_world();
        let sluice_offset = detail_sluice_offset(&self.world, &self.sluice_config);
        let (sluice_vertices, sluice_indices) =
            prepare_sluice_mesh(&self.sluice_config, &mut self.flip_sim.grid, sluice_offset);
        self.sluice_vertices = sluice_vertices;
        self.sluice_indices = sluice_indices;
        self.flip_origin = sluice_offset;
        let sluice_length = self.sluice_config.grid_width as f32 * self.sluice_config.cell_size;
        let sluice_height = self.sluice_config.grid_height as f32 * self.sluice_config.cell_size;
        let sluice_depth = self.sluice_config.grid_depth as f32 * self.sluice_config.cell_size;
        self.focus_bounds_min = sluice_offset;
        self.focus_bounds_max =
            sluice_offset + Vec3::new(sluice_length, sluice_height, sluice_depth);
        self.flip_sim.particles.clear();
        self.pending_water_emits = 0;
        self.pending_sediment_emits = 0;
        self.flip_sim.frame = 0;
        self.dem.clumps.clear();
        self.sediment_flip_indices.clear();
        self.focus_mode = false;
        if let Some(gpu) = &mut self.gpu {
            let (vertex_buffer, index_buffer, index_count) =
                create_sluice_buffers(&gpu.device, &self.sluice_vertices, &self.sluice_indices);
            gpu.sluice_vertex_buffer = vertex_buffer;
            gpu.sluice_index_buffer = index_buffer;
            gpu.sluice_index_count = index_count;
        }
    }

    fn update(&mut self, dt: f32) {
        self.update_camera(dt);

        // Manage adaptive fine region based on camera zoom
        self.update_fine_region_for_zoom();

        let mut steps = 0;
        let mut total_flip_dt = 0.0;

        if let Some(gpu) = self.gpu.as_mut() {
            // Run GPU Sim with fixed timestep substepping
            let sim_dt = DT; // Use fixed 0.02s timestep
            steps = ((dt / sim_dt).ceil() as usize).min(STEPS_PER_FRAME);
            total_flip_dt = (steps as f32) * (1.0 / 60.0);

            for _ in 0..steps {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sim Encoder"),
                        });

                // Update GPU emitter and dispatch (before water sim)
                // Update GPU emitter and dispatch (before water sim)
                self.emitter.update_gpu(
                    &gpu.heightfield,
                    &gpu.queue,
                    sim_dt,
                );
                gpu.heightfield.dispatch_emitter(&mut encoder);

                // Update water sim params and dispatch
                gpu.heightfield.update_params(&gpu.queue, sim_dt);
                gpu.heightfield.dispatch(&mut encoder, sim_dt);

                gpu.queue.submit(Some(encoder.finish()));
            }

            // Sync back to CPU for rendering
            pollster::block_on(gpu.heightfield.download_to_world(
                &gpu.device,
                &gpu.queue,
                &mut self.world,
            ));

            // Update fine region (if active) with boundary conditions from coarse grid
            for _ in 0..steps {
                let do_erosion = self.world.next_erosion_step();
                self.world.update_fine_region(DT, do_erosion);
            }
        } else {
            // Fallback if no GPU? (Shouldn't happen in this example)
            // self.world.update(dt);
        }

        // ===== FLIP Simulation Step (substepped to match heightfield) =====
        if self.focus_mode {
            self.queue_emissions();
            self.emit_pending_particles();

            if self.flip_sim.particle_count() > 0 && steps > 0 {
                if let (Some(gpu), Some(gpu_flip)) = (&self.gpu, &mut self.gpu_flip) {
                    // Extract from FlipSimulation3D
                    self.flip_positions.clear();
                    self.flip_velocities.clear();
                    self.flip_c_matrices.clear();
                    self.flip_densities.clear();

                    for p in &self.flip_sim.particles.list {
                        self.flip_positions.push(p.position);
                        self.flip_velocities.push(p.velocity);
                        self.flip_c_matrices.push(p.affine_velocity);
                        self.flip_densities.push(p.density);
                    }

                    let w = self.flip_sim.grid.width;
                    let h = self.flip_sim.grid.height;
                    let d = self.flip_sim.grid.depth;

                    // Substep loop - match heightfield substepping
                    for _ in 0..steps {
                        // Build cell types: SOLID where grid says solid, FLUID where particles are
                        self.flip_cell_types.clear();
                        self.flip_cell_types.resize(w * h * d, 0); // AIR by default

                        for k in 0..d {
                            for j in 0..h {
                                for i in 0..w {
                                    let idx = k * w * h + j * w + i;
                                    if self.flip_sim.grid.is_solid(i, j, k) {
                                        self.flip_cell_types[idx] = 2; // SOLID
                                    }
                                }
                            }
                        }

                        // Mark FLUID cells from particles (using current positions)
                        let cell_size = self.flip_sim.grid.cell_size;
                        for pos in &self.flip_positions {
                            let i = (pos.x / cell_size).floor() as i32;
                            let j = (pos.y / cell_size).floor() as i32;
                            let k = (pos.z / cell_size).floor() as i32;
                            if i >= 0
                                && i < w as i32
                                && j >= 0
                                && j < h as i32
                                && k >= 0
                                && k < d as i32
                            {
                                let idx = k as usize * w * h + j as usize * w + i as usize;
                                if self.flip_cell_types[idx] != 2 {
                                    self.flip_cell_types[idx] = 1; // FLUID
                                }
                            }
                        }

                        gpu_flip.step(
                            &gpu.device,
                            &gpu.queue,
                            &mut self.flip_positions,
                            &mut self.flip_velocities,
                            &mut self.flip_c_matrices,
                            &self.flip_densities,
                            &self.flip_cell_types,
                            Some(&self.flip_sim.grid.sdf),
                            None, // bed_height
                            DT,   // Use same timestep as heightfield
                            constants::GRAVITY,
                            0.0,
                            self.flip_sim.pressure_iterations as u32,
                        );
                    }

                    // Sync results back to FlipSimulation3D (once after all substeps)
                    let mut avg_y = 0.0;
                    let mut max_y: f32 = f32::NEG_INFINITY;
                    let mut min_y: f32 = f32::INFINITY;
                    let mut max_vy: f32 = 0.0;
                    for (i, p) in self.flip_sim.particles.list.iter_mut().enumerate() {
                        let pos = self.flip_positions[i];
                        p.position = pos;
                        p.velocity = self.flip_velocities[i];
                        p.affine_velocity = self.flip_c_matrices[i];

                        let vy = p.velocity.y;
                        if vy.abs() > 0.1 {
                            max_vy = if vy < 0.0 {
                                max_vy.min(vy)
                            } else {
                                max_vy.max(vy)
                            };
                        }

                        avg_y += pos.y;
                        max_y = max_y.max(pos.y);
                        min_y = min_y.min(pos.y);
                    }
                    if self.flip_sim.particle_count() > 0 {
                        avg_y /= self.flip_sim.particle_count() as f32;
                        if self.flip_sim.frame % 30 == 0 {
                            println!("FLIP Count: {} | Height: avg={:.2}, min={:.2}, max={:.2} | Max |Vy|: {:.2}",
                                self.flip_sim.particle_count(), avg_y, min_y, max_y, max_vy.abs());
                        }
                    }
                }
            }

            if steps > 0 {
                self.run_dem_collision_response(total_flip_dt);
            }

            self.flip_sim.frame = self.flip_sim.frame.wrapping_add(1);
        } else {
            self.flip_sim.frame = 0;
        }

        // Handle mouse click - material tool
        if self.input.left_mouse {
            if let Some(hit) = self.raycast_terrain() {
                if let Some(gpu) = self.gpu.as_ref() {
                    // Ctrl = add material, else = excavate
                    let is_adding = self.input.keys.contains(&KeyCode::ControlLeft)
                        || self.input.keys.contains(&KeyCode::ControlRight);

                    let amount = if is_adding {
                        ADD_HEIGHT * 50.0
                    } else {
                        -(DIG_DEPTH * 50.0)
                    };
                    let radius = if is_adding { ADD_RADIUS } else { DIG_RADIUS };

                    // Update material tool params
                    gpu.heightfield.update_material_tool(
                        &gpu.queue,
                        hit.x,
                        hit.z,
                        radius,
                        amount,
                        self.selected_material,
                        dt,
                        true, // enabled
                    );

                    // Dispatch appropriate tool
                    let mut encoder =
                        gpu.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Material Tool Encoder"),
                            });

                    if is_adding {
                        gpu.heightfield.dispatch_material_tool(&mut encoder);
                    } else {
                        gpu.heightfield.dispatch_excavate(&mut encoder);
                    }

                    gpu.queue.submit(Some(encoder.finish()));
                    self.terrain_dirty = true; // Rebuild terrain mesh on next render
                }
            }
        }

        if self.last_stats.elapsed() > Duration::from_secs(1) {
            let water = self.world.total_water_volume();
            let sediment = self.world.total_sediment_volume();
            let flip_count = self.flip_sim.particle_count();

            println!(
                "Heightfield water: {:.2}, sediment: {:.2} | FLIP particles: {}",
                water, sediment, flip_count
            );
            if DEBUG_HEIGHTFIELD_STATS {
                if let Some(gpu) = &self.gpu {
                    let debug = pollster::block_on(
                        gpu.heightfield.read_debug_stats(&gpu.device, &gpu.queue),
                    );
                    println!(
                        "Erosion debug: erode_cells={} deposit_cells={} max_erode={:.6}m max_deposit={:.6}m | erode(s/o/g/p)={}/{}/{}/{} deposit(s/o/g/p)={}/{}/{}/{}",
                        debug.erosion_cells,
                        debug.deposition_cells,
                        debug.erosion_max_height,
                        debug.deposition_max_height,
                        debug.erosion_layers[0],
                        debug.erosion_layers[1],
                        debug.erosion_layers[2],
                        debug.erosion_layers[3],
                        debug.deposition_layers[0],
                        debug.deposition_layers[1],
                        debug.deposition_layers[2],
                        debug.deposition_layers[3],
                    );
                    gpu.heightfield.reset_debug_stats(&gpu.queue);
                }
            }
            self.last_stats = Instant::now();
        }

        // Update Emitter Mesh if enabled (heightfield emitter)
        if self.emitter.enabled && !self.focus_mode {
            if let Some(gpu) = self.gpu.as_ref() {
                let (positions, indices) = self.emitter.visualize(16);
                let vertices: Vec<WorldVertex> = positions
                    .iter()
                    .map(|p| WorldVertex {
                        position: p.to_array(),
                        color: [0.0, 1.0, 1.0, 1.0], // Cyan
                    })
                    .collect();

                if let Some(mesh) = self.emitter_mesh.as_mut() {
                    mesh.update(&gpu.device, &gpu.queue, &vertices, &indices, "Emitter Mesh");
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

        // Update Detail Zone Emitter Mesh (small sphere at emission point)
        if self.focus_mode {
            if let Some(gpu) = self.gpu.as_ref() {
                // Emission location matches emit_particles() logic
                let cell_size = self.sluice_config.cell_size;
                let emit_x = self.detail_emitter_pos.x;
                let center_z = self.detail_emitter_pos.z;
                let floor_y = self.sluice_floor_height(emit_x);
                let drop_height = 2.5 * cell_size;
                let sheet_height = 4.0 * cell_size;

                // Position emitter visualization at center of emission region
                let emitter_pos = self.flip_origin
                    + Vec3::new(emit_x, floor_y + drop_height + sheet_height * 0.5, center_z);

                let (positions, indices) =
                    create_small_sphere(emitter_pos, DETAIL_EMITTER_VISUAL_RADIUS, 12);
                let vertices: Vec<WorldVertex> = positions
                    .iter()
                    .map(|p| WorldVertex {
                        position: p.to_array(),
                        color: [1.0, 0.5, 0.0, 1.0], // Orange for detail emitter
                    })
                    .collect();

                if let Some(mesh) = self.detail_emitter_mesh.as_mut() {
                    mesh.update(
                        &gpu.device,
                        &gpu.queue,
                        &vertices,
                        &indices,
                        "Detail Emitter Mesh",
                    );
                } else {
                    self.detail_emitter_mesh = Some(Mesh::new(
                        &gpu.device,
                        &vertices,
                        &indices,
                        "Detail Emitter Mesh",
                    ));
                }
            }
        } else {
            self.detail_emitter_mesh = None;
        }
    }

    fn add_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world.add_water(hit, WATER_ADD_VOLUME);
        }
    }

    fn add_muddy_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world
                .add_sediment_water(hit, WATER_ADD_VOLUME, WATER_ADD_VOLUME * 0.1);
        }
    }

    fn queue_emissions(&mut self) {
        if self.flip_sim.frame % 2 == 0 {
            self.pending_water_emits = self.pending_water_emits.saturating_add(WATER_EMIT_RATE);
            self.pending_sediment_emits = self
                .pending_sediment_emits
                .saturating_add(SEDIMENT_EMIT_RATE);
        }
    }

    fn emit_pending_particles(&mut self) {
        if self.pending_water_emits == 0 && self.pending_sediment_emits == 0 {
            return;
        }
        let available = MAX_FLIP_PARTICLES.saturating_sub(self.flip_sim.particle_count());
        if available == 0 {
            return;
        }

        let water_count = self.pending_water_emits.min(available);
        self.pending_water_emits -= water_count;
        let remaining = available - water_count;
        let sediment_count = self.pending_sediment_emits.min(remaining);
        self.pending_sediment_emits -= sediment_count;

        if water_count > 0 || sediment_count > 0 {
            self.emit_particles(water_count, sediment_count);
        }
    }

    fn emit_particles(&mut self, water_count: usize, sediment_count: usize) {
        if water_count == 0 && sediment_count == 0 {
            return;
        }

        let config = &self.sluice_config;
        let cell_size = config.cell_size;
        let grid_depth = config.grid_depth as f32;
        // Use detail_emitter_pos for emission location
        let emit_x = self.detail_emitter_pos.x;
        let center_z = self.detail_emitter_pos.z;
        let floor_y = self.sluice_floor_height(emit_x);
        let drop_height = 2.5 * cell_size;
        let sheet_height = 4.0 * cell_size;
        let water_spread_z = (grid_depth - 4.0) * cell_size * 0.3;
        let sediment_spread_z = (grid_depth - 4.0) * cell_size * 0.2;
        let init_vel = Vec3::new(0.5, -0.05, 0.0);

        for _ in 0..water_count {
            if self.flip_sim.particles.len() >= MAX_FLIP_PARTICLES {
                break;
            }
            let x = emit_x + (rand_float() - 0.5) * 2.0 * cell_size;
            let z = center_z + (rand_float() - 0.5) * water_spread_z;
            let y = floor_y + drop_height + rand_float() * sheet_height;
            self.flip_sim
                .spawn_particle_with_velocity(Vec3::new(x, y, z), init_vel);
        }

        for _ in 0..sediment_count {
            if self.flip_sim.particles.len() >= MAX_FLIP_PARTICLES {
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
            let pos = Vec3::new(x, y, z);
            let flip_idx = self.flip_sim.particles.len();
            self.flip_sim.spawn_sediment(pos, sediment_vel, density);
            self.sediment_flip_indices.push(flip_idx);
            self.dem.spawn(template_idx, pos, sediment_vel);
        }
    }

    fn sluice_floor_height(&self, x: f32) -> f32 {
        let config = &self.sluice_config;
        let length = config.grid_width as f32 * config.cell_size;
        let t = (x / length).clamp(0.0, 1.0);
        let left = (config.floor_height_left + 1) as f32 * config.cell_size;
        let right = (config.floor_height_right + 1) as f32 * config.cell_size;
        left * (1.0 - t) + right * t
    }

    fn run_dem_collision_response(&mut self, dt: f32) {
        if self.sediment_flip_indices.is_empty() || self.dem.clumps.is_empty() {
            return;
        }

        let sdf_params = SdfParams {
            sdf: &self.flip_sim.grid.sdf,
            grid_width: self.flip_sim.grid.width,
            grid_height: self.flip_sim.grid.height,
            grid_depth: self.flip_sim.grid.depth,
            cell_size: self.flip_sim.grid.cell_size,
            grid_offset: self.flip_origin,
        };

        // Sync FLIP -> DEM (positions + velocities)
        for (clump_idx, &flip_idx) in self.sediment_flip_indices.iter().enumerate() {
            if flip_idx >= self.flip_sim.particles.list.len() {
                continue;
            }
            if let Some(clump) = self.dem.clumps.get_mut(clump_idx) {
                let particle = &self.flip_sim.particles.list[flip_idx];
                clump.position = particle.position;
                clump.velocity = particle.velocity;
            }
        }

        self.dem.collision_response_only(dt, &sdf_params, true);

        // Sync DEM -> FLIP
        for (clump_idx, &flip_idx) in self.sediment_flip_indices.iter().enumerate() {
            if flip_idx >= self.flip_sim.particles.list.len() {
                continue;
            }
            if let Some(clump) = self.dem.clumps.get(clump_idx) {
                let particle = &mut self.flip_sim.particles.list[flip_idx];
                particle.position = clump.position;
                particle.velocity = clump.velocity;
            }
        }
    }

    fn focus_reset(&mut self) {
        self.flip_sim.particles.clear();
        self.pending_water_emits = 0;
        self.pending_sediment_emits = 0;
        self.flip_sim.frame = 0;
        self.dem.clumps.clear();
        self.sediment_flip_indices.clear();
    }

    fn enter_focus_zone(&mut self) {
        if self.focus_mode {
            return;
        }
        self.focus_mode = true;
        self.focus_reset();
        println!("Focus zone activated: sluice detail view");
    }

    fn exit_focus_zone(&mut self) {
        if !self.focus_mode {
            return;
        }
        self.focus_mode = false;
        self.focus_reset();
        println!("Focus zone exited");
    }

    fn point_in_focus_zone(&self, pos: Vec3) -> bool {
        pos.x >= self.focus_bounds_min.x
            && pos.x <= self.focus_bounds_max.x
            && pos.z >= self.focus_bounds_min.z
            && pos.z <= self.focus_bounds_max.z
    }

    fn toggle_focus_from_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            if self.point_in_focus_zone(hit) {
                if self.focus_mode {
                    self.exit_focus_zone();
                } else {
                    self.enter_focus_zone();
                }
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

        // Apply scroll zoom
        if self.input.scroll_delta != 0.0 {
            let forward = self.camera.forward();
            self.camera.position += forward * self.input.scroll_delta * 2.0;
            self.input.scroll_delta = 0.0;
        }

        let world_size = self.world.world_size();
        self.camera.position.x = self.camera.position.x.clamp(0.0, world_size.x);
        self.camera.position.z = self.camera.position.z.clamp(0.0, world_size.z);
        self.camera.position.y = self.camera.position.y.clamp(2.0, world_size.y + 100.0);
    }

    /// Manage fine region based on camera zoom level.
    /// Creates a fine region when zoomed in, removes when zoomed out.
    fn update_fine_region_for_zoom(&mut self) {
        const FINE_REGION_THRESHOLD: f32 = 15.0; // Camera height below which fine region activates
        const FINE_REGION_RADIUS: usize = 5; // Radius in coarse cells
        const FINE_REGION_SCALE: usize = 4; // 4x resolution

        let camera_height = self.camera.position.y;

        if camera_height < FINE_REGION_THRESHOLD {
            // Zoomed in - activate fine region centered on camera look-at point
            let look_at = self.camera.position + self.camera.forward() * 10.0;
            let center = Vec3::new(
                look_at.x.clamp(0.0, self.world.world_size().x),
                0.0,
                look_at.z.clamp(0.0, self.world.world_size().z),
            );

            // Only recreate if fine region doesn't exist or camera moved significantly
            let should_create = match &self.world.fine_region {
                None => true,
                Some(fine) => {
                    // Check if camera moved outside current fine region bounds
                    let fine_world_x_min = fine.coarse_x_min as f32 * self.world.cell_size;
                    let fine_world_z_min = fine.coarse_z_min as f32 * self.world.cell_size;
                    let fine_world_x_max = (fine.coarse_x_max + 1) as f32 * self.world.cell_size;
                    let fine_world_z_max = (fine.coarse_z_max + 1) as f32 * self.world.cell_size;

                    // Add some hysteresis to avoid constant recreation
                    let margin = self.world.cell_size * 2.0;
                    center.x < fine_world_x_min + margin
                        || center.x > fine_world_x_max - margin
                        || center.z < fine_world_z_min + margin
                        || center.z > fine_world_z_max - margin
                }
            };

            if should_create {
                self.world.create_fine_region(center, FINE_REGION_RADIUS, FINE_REGION_SCALE);
                if let Some(ref fine) = self.world.fine_region {
                    println!(
                        "Fine region created: {}x{} cells at {:.3}m resolution (coarse: {:.3}m)",
                        fine.width, fine.depth, fine.cell_size, self.world.cell_size
                    );
                }
            }
        } else {
            // Zoomed out - remove fine region
            if self.world.fine_region.is_some() {
                println!("Fine region removed (camera too high)");
                self.world.remove_fine_region();
            }
        }
    }

    fn screen_to_world_ray(&self, screen_x: f32, screen_y: f32) -> Vec3 {
        let ndc_x = (2.0 * screen_x / self.window_size.0 as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / self.window_size.1 as f32);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let inv_vp = (proj * view).inverse();

        let near = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;

        (far - near).normalize()
    }

    fn projection_matrix(&self) -> Mat4 {
        let aspect = self.window_size.0 as f32 / self.window_size.1 as f32;
        Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 500.0)
    }

    fn raycast_terrain(&self) -> Option<Vec3> {
        let ray_dir = self.screen_to_world_ray(self.input.mouse_pos.0, self.input.mouse_pos.1);
        let ray_origin = self.camera.position;

        let step = 0.5;
        let max_dist = 200.0;

        let mut t = 0.0;
        while t < max_dist {
            let p = ray_origin + ray_dir * t;

            if let Some((x, z)) = self.world.world_to_cell(p) {
                let ground = self.world.ground_height(x, z);
                if p.y <= ground {
                    return Some(Vec3::new(p.x, ground, p.z));
                }
            }

            t += step;
        }

        None
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        self.window_size = (size.width.max(1), size.height.max(1));

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

        // GpuFlip3D needs higher storage buffer limits
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
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, // Need both for highlight_tint
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
            label: Some("Equipment Pipeline"),
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

        let mut heightfield = GpuHeightfield::new(
            &device,
            self.world.width as u32,
            self.world.depth as u32,
            self.world.cell_size,
            INITIAL_HEIGHT,
            config.format,
        );
        if DEBUG_HEIGHTFIELD_STATS {
            heightfield.set_debug_flags(1);
            heightfield.reset_debug_stats(&queue);
        }
        heightfield.upload_from_world(&queue, &self.world);

        // ===== Particle Rendering Setup =====
        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
        });

        // Quad vertices (billboard)
        let quad_vertices: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let particle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Vertices"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Instance buffer for particle positions
        let particle_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instances"),
            size: 100_000 * 16, // 100k particles * 16 bytes (vec3 + padding)
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pipeline for particles
        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4], // xyz = position, w = velocity magnitude
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (sluice_vertex_buffer, sluice_index_buffer, sluice_index_count) =
            create_sluice_buffers(&device, &self.sluice_vertices, &self.sluice_indices);

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
            particle_pipeline,
            particle_vertex_buffer,
            particle_instance_buffer,
            sluice_vertex_buffer,
            sluice_index_buffer,
            sluice_index_count,
        });

        // Initialize GpuFlip3D for FLIP simulation
        if let Some(gpu) = &self.gpu {
            let flip_grid = &self.flip_sim.grid;
            let gpu_flip = GpuFlip3D::new(
                &gpu.device,
                flip_grid.width as u32,
                flip_grid.height as u32,
                flip_grid.depth as u32,
                flip_grid.cell_size,
                100_000, // max particles
            );
            self.gpu_flip = Some(gpu_flip);
            println!("GpuFlip3D initialized");
        }
    }

    fn render(&mut self) {
        let window = match self.window.as_ref() {
            Some(window) => window.clone(),
            None => return,
        };

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.update(dt);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;
        let camera_forward = self.camera.forward();
        let camera_right = camera_forward.cross(Vec3::Y).normalize();
        let camera_up = camera_right.cross(camera_forward).normalize();

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            particle_size: PARTICLE_SIZE,
            camera_right: camera_right.to_array(),
            show_velocity: if self.show_velocity { 1.0 } else { 0.0 },
            camera_up: camera_up.to_array(),
            highlight_tint: 1.0, // Default, updated per-object
        };

        let Some(gpu) = self.gpu.as_mut() else { return };

        // Only rebuild terrain mesh when it changes (expensive with side faces)
        if self.terrain_dirty {
            // Force update? Shader handles vertex displacement, so we don't need to rebuild geometry.
            // But we should ensure buffers are up to date?
            // GpuHeightfield buffers (bedrock etc) are updated explicitly by dispatch/download?
            // "Displacement" uses buffers directly. So if sim runs, rendering is up to date.
            self.terrain_dirty = false;
        }

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let frame_view = output.texture.create_view(&Default::default());

        let mut encoder = gpu.device.create_command_encoder(&Default::default());

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

        // Render Emitter (heightfield emitter - only when not in focus mode)
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

        // Render Detail Emitter (small sphere in focus mode)
        if let Some(mesh) = self.detail_emitter_mesh.as_ref() {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Detail Emitter Render Pass"),
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

        // ===== Render equipment geometry (sluice) =====
        // Always render sluice so it's visible on the 2.5D map for clicking
        if gpu.sluice_index_count > 0 {
            // Highlight sluice when hovered in Pointer mode
            let sluice_uniforms = Uniforms {
                highlight_tint: if self.sluice_hovered { 1.5 } else { 1.0 },
                ..uniforms
            };
            gpu.queue
                .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&sluice_uniforms));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Equipment Pass"),
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
                ..Default::default()
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.sluice_vertex_buffer.slice(..));
            pass.set_index_buffer(gpu.sluice_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.sluice_index_count, 0, 0..1);
        }

        // ===== Render FLIP particles =====
        if self.focus_mode {
            let particle_count = self.flip_sim.particle_count();
            if particle_count > 0 {
                // Upload particle positions (with flip_origin offset) and velocity magnitude
                let positions: Vec<[f32; 4]> = self
                    .flip_sim
                    .particles
                    .list
                    .iter()
                    .map(|p| {
                        let world_pos = p.position + self.flip_origin;
                        let vel_mag = p.velocity.length();
                        [world_pos.x, world_pos.y, world_pos.z, vel_mag]
                    })
                    .collect();
                gpu.queue.write_buffer(
                    &gpu.particle_instance_buffer,
                    0,
                    bytemuck::cast_slice(&positions),
                );

                // Update uniforms for particle rendering
                gpu.queue
                    .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                // Draw particles
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Particle Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load, // Keep heightfield
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

                    pass.set_pipeline(&gpu.particle_pipeline);
                    pass.set_bind_group(0, &gpu.bind_group, &[]);
                    pass.set_vertex_buffer(0, gpu.particle_vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, gpu.particle_instance_buffer.slice(..));
                    pass.draw(0..4, 0..particle_count as u32);
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("World Heightfield Test")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.window_size = (size.width.max(1), size.height.max(1));
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.window_size = (size.width.max(1), size.height.max(1));
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);

                    // Recreate depth texture
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
                                        let hf_height = self.world.ground_height(hit.x as usize, hit.z as usize);
                                        self.emitter.place_at_cursor(hit, hf_height);
                                        self.emitter.enabled = !self.emitter.enabled; // Toggle on/off
                                        println!("Emitter toggle: {} at {:?}", self.emitter.enabled, self.emitter.position);
                                    } else {
                                        println!("Raycast missed terrain!");

                                    }
                                }
                                // Material selection keys
                                KeyCode::KeyG => {
                                    self.selected_material = 2; // Gravel
                                    println!("Selected material: Gravel");
                                }
                                KeyCode::KeyO => {
                                    self.selected_material = 1; // Overburden
                                    println!("Selected material: Overburden");
                                }
                                KeyCode::KeyT => {
                                    self.selected_material = 0; // Sediment (T for terrain/sediment)
                                    println!("Selected material: Sediment");
                                }
                                KeyCode::KeyF => {
                                    self.toggle_focus_from_cursor();
                                }
                                KeyCode::KeyH => {
                                    self.show_water = !self.show_water;
                                    println!("Show water: {}", self.show_water);
                                }
                                KeyCode::KeyP => {
                                    self.interaction_mode = match self.interaction_mode {
                                        InteractionMode::Dig => {
                                            println!("Pointer mode: click sluice to focus");
                                            InteractionMode::Pointer
                                        }
                                        InteractionMode::Pointer => {
                                            println!("Dig mode: click to add water/dig");
                                            InteractionMode::Dig
                                        }
                                    };
                                }
                                KeyCode::KeyE => {
                                    self.emitter.enabled = !self.emitter.enabled;
                                    println!("Emitter: {}", if self.emitter.enabled { "ON" } else { "OFF" });
                                }
                                KeyCode::KeyV => {
                                    self.show_velocity = !self.show_velocity;
                                    println!("Velocity coloring: {}", if self.show_velocity { "ON" } else { "OFF" });
                                }
                                KeyCode::KeyQ => {
                                    // Reset emitter to river source
                                    self.emitter.position.x = (WORLD_WIDTH as f32 * 0.5) * CELL_SIZE;
                                    self.emitter.position.z = 5.0 * CELL_SIZE;
                                    self.emitter.position.y = 12.0; // Above river start
                                    self.emitter.radius = (WORLD_WIDTH as f32 * 0.05 * CELL_SIZE) * 0.8;
                                    self.emitter.rate = 2.0;
                                    self.emitter.enabled = true;
                                    println!("Emitter reset to river source");
                                }
                                KeyCode::KeyX => {
                                    // Reposition detail emitter at cursor (in focus mode)
                                    if self.focus_mode {
                                        if let Some(hit) = self.raycast_terrain() {
                                            // Convert world coords to grid-local coords
                                            let local_pos = hit - self.flip_origin;
                                            // Clamp to valid grid bounds
                                            let grid_max_x = DETAIL_FLIP_GRID_X as f32
                                                * DETAIL_FLIP_CELL_SIZE;
                                            let grid_max_z = DETAIL_FLIP_GRID_Z as f32
                                                * DETAIL_FLIP_CELL_SIZE;
                                            self.detail_emitter_pos.x =
                                                local_pos.x.clamp(0.0, grid_max_x);
                                            self.detail_emitter_pos.z =
                                                local_pos.z.clamp(0.0, grid_max_z);
                                            println!(
                                                "Detail emitter moved to ({:.3}, {:.3})",
                                                self.detail_emitter_pos.x, self.detail_emitter_pos.z
                                            );
                                        }
                                    } else {
                                        println!("X: Enter focus mode first to reposition detail emitter");
                                    }
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
                    if state == ElementState::Pressed
                        && self.interaction_mode == InteractionMode::Pointer
                        && self.sluice_hovered
                    {
                        self.toggle_focus_from_cursor();
                    } else {
                        self.input.left_mouse = state == ElementState::Pressed;
                    }
                }
                MouseButton::Middle => {
                    if state == ElementState::Pressed {
                        self.toggle_focus_from_cursor();
                    }
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

                // Update hover state in Pointer mode
                if self.interaction_mode == InteractionMode::Pointer {
                    self.sluice_hovered = self
                        .raycast_terrain()
                        .map(|hit| self.point_in_focus_zone(hit))
                        .unwrap_or(false);
                } else {
                    self.sluice_hovered = false;
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

fn build_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 20.0);
    
    // ========== NOISE FUNCTIONS ==========
    fn noise2d(x: f32, y: f32) -> f32 {
        // Simple hash-based noise
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
        
        // Smooth interpolation
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
    
    // ========== VALLEY PARAMETERS ==========
    let valley_width_cells = (WORLD_WIDTH as f32 * 0.35) as i32; // 35% of width is the valley
    let river_width_cells = (WORLD_WIDTH as f32 * 0.08) as i32;  // 8% is the actual river
    let center_x = WORLD_WIDTH as i32 / 2;
    
    // Ridge heights
    let ridge_base = 18.0;
    let valley_floor_base = 6.0;
    
    for z in 0..WORLD_DEPTH {
        let zf = z as f32;
        let z_progress = zf / WORLD_DEPTH as f32;
        
        // River gradient (drops 6m over length)
        let river_gradient = z_progress * 6.0;
        
        // Meander - river curves side to side
        let meander = ((zf * 0.015).sin() * 0.7 + (zf * 0.008).cos() * 0.3) 
                    * valley_width_cells as f32 * 0.2;
        let river_center = center_x as f32 + meander;
        
        // Pool-riffle sequence with varying amplitude
        let pool_riffle = (zf * 0.04).sin() * 0.5 
                        + (zf * 0.11).sin() * 0.25
                        + (zf * 0.23).cos() * 0.15;
        
        for x in 0..WORLD_WIDTH {
            let xf = x as f32;
            let idx = world.idx(x, z);
            
            // Distance from river center (signed)
            let dist_from_river = xf - river_center;
            let abs_dist = dist_from_river.abs();
            
            // Large-scale terrain noise
            let terrain_noise = fbm(xf * 0.02, zf * 0.02, 4);
            let detail_noise = fbm(xf * 0.08, zf * 0.08, 3);
            
            // ========== BEDROCK ELEVATION ==========
            let bedrock;
            let mut gravel = 0.0;
            let mut paydirt = 0.0;
            let mut overburden = 0.0;
            
            if abs_dist < river_width_cells as f32 / 2.0 {
                // === RIVER BED ===
                // U-shaped channel with irregular bottom
                let cross = (abs_dist / (river_width_cells as f32 / 2.0)).powi(2);
                let channel_depth = 1.8 * (1.0 - cross);
                
                // Add rock outcrops and pools
                let outcrop = if detail_noise > 0.7 { (detail_noise - 0.7) * 2.0 } else { 0.0 };
                let pool_depth = if detail_noise < 0.25 { (0.25 - detail_noise) * 1.5 } else { 0.0 };
                
                bedrock = (valley_floor_base - river_gradient - channel_depth + pool_riffle + outcrop - pool_depth).max(0.5);
                
                // Gravel deposits in pools and behind outcrops
                gravel = 0.15 + rand_float() * 0.2;
                if detail_noise < 0.3 { gravel += 0.3; } // More gravel in pools
                
                // Paydirt in certain spots (gold bearing)
                if detail_noise > 0.4 && detail_noise < 0.6 {
                    paydirt = 0.1 + rand_float() * 0.15;
                }
                
            } else if abs_dist < valley_width_cells as f32 / 2.0 {
                // === VALLEY FLOOR / BANKS ===
                let bank_t = (abs_dist - river_width_cells as f32 / 2.0) 
                           / (valley_width_cells as f32 / 2.0 - river_width_cells as f32 / 2.0);
                let bank_t = bank_t.clamp(0.0, 1.0);
                
                // Smooth transition from river to valley walls
                let floor_elev = valley_floor_base - river_gradient * (1.0 - bank_t * 0.5);
                let wall_rise = bank_t.powi(2) * 3.0; // Steeper near edges
                
                bedrock = floor_elev + wall_rise + terrain_noise * 1.5;
                
                // Point bars and gravel deposits on inside of bends
                let bend_inside = dist_from_river * meander < 0.0; // Inside of meander bend
                if bend_inside && bank_t < 0.3 {
                    gravel = 0.3 + rand_float() * 0.4;
                    paydirt = rand_float() * 0.2; // Gold accumulates on point bars
                } else {
                    gravel = (0.3 - bank_t * 0.3).max(0.0) + rand_float() * 0.1;
                }
                overburden = bank_t * 0.5 + detail_noise * 0.3;
                
            } else {
                // === VALLEY WALLS / RIDGES ===
                let wall_t = (abs_dist - valley_width_cells as f32 / 2.0) 
                           / (WORLD_WIDTH as f32 / 2.0 - valley_width_cells as f32 / 2.0);
                let wall_t = wall_t.clamp(0.0, 1.0);
                
                // Rising valley walls with rough texture
                let wall_height = wall_t.sqrt() * (ridge_base - valley_floor_base);
                bedrock = valley_floor_base + wall_height + terrain_noise * 3.0 + detail_noise * 1.0;
                
                // Overburden on slopes
                let slope = wall_t.sqrt();
                overburden = (1.0 - slope) * 2.0 + detail_noise * 0.5;
                gravel = detail_noise * 0.2;
            }
            
            world.bedrock_elevation[idx] = bedrock;
            world.gravel_thickness[idx] = gravel;
            world.paydirt_thickness[idx] = paydirt;
            world.overburden_thickness[idx] = overburden;
            world.terrain_sediment[idx] = 0.0;
        }
    }
    
    // Pre-fill river with water
    for z in 0..WORLD_DEPTH {
        let zf = z as f32;
        let meander = ((zf * 0.015).sin() * 0.7 + (zf * 0.008).cos() * 0.3) 
                    * valley_width_cells as f32 * 0.2;
        let river_center = center_x as f32 + meander;
        
        for x in 0..WORLD_WIDTH {
            let dist = (x as f32 - river_center).abs();
            if dist < river_width_cells as f32 / 2.0 {
                let idx = world.idx(x, z);
                world.water_surface[idx] = world.ground_height(x, z) + 0.4;
            }
        }
    }
    
    world
}




fn detail_center(world: &World) -> Vec3 {
    Vec3::new(
        world.width as f32 * world.cell_size * 0.5,
        0.0,
        world.depth as f32 * world.cell_size * 0.5,
    )
}

fn detail_sluice_config() -> SluiceConfig {
    let mut config = SluiceConfig::default();
    config.grid_width = DETAIL_SLUICE_WIDTH;
    config.grid_height = DETAIL_SLUICE_HEIGHT;
    config.grid_depth = DETAIL_SLUICE_DEPTH;
    config.cell_size = DETAIL_SLUICE_CELL_SIZE;
    config.floor_height_left = 30;
    config.floor_height_right = 4;
    config.riffle_spacing = 32;
    config.riffle_height = 3;
    config.riffle_thickness = 2;
    config.riffle_start_x = 40;
    config.riffle_end_pad = 12;
    config.wall_margin = 8;
    config.exit_width_fraction = 1.0;
    config.exit_height = 12;
    config
}

fn detail_sluice_offset(world: &World, config: &SluiceConfig) -> Vec3 {
    let center = detail_center(world);
    let (cx, cz) = world
        .world_to_cell(center)
        .unwrap_or((world.width / 2, world.depth / 2));
    let base_height = world.ground_height(cx, cz) + DETAIL_SLUICE_HEIGHT_OFFSET;
    let sluice_length = config.grid_width as f32 * config.cell_size;
    let sluice_depth = config.grid_depth as f32 * config.cell_size;
    Vec3::new(
        center.x - sluice_length * 0.5,
        base_height,
        center.z - sluice_depth * 0.5,
    )
}

fn prepare_sluice_mesh(
    config: &SluiceConfig,
    grid: &mut Grid3D,
    offset: Vec3,
) -> (Vec<SluiceVertex>, Vec<u32>) {
    grid.clear_solids();
    let mut builder = SluiceGeometryBuilder::new(config.clone());
    for (i, j, k) in builder.solid_cells() {
        grid.set_solid(i, j, k);
    }

    builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

    let vertices = builder
        .vertices()
        .iter()
        .map(|v| SluiceVertex {
            position: [
                v.position[0] + offset.x,
                v.position[1] + offset.y,
                v.position[2] + offset.z,
            ],
            color: v.color,
        })
        .collect();
    let indices = builder.indices().to_vec();
    (vertices, indices)
}

fn carve_flat_pad(world: &mut World, center: Vec3, radius: f32, falloff: f32, height_offset: f32) {
    let Some((cx, cz)) = world.world_to_cell(center) else {
        return;
    };
    let target_height = world.ground_height(cx, cz) + height_offset;
    let max_radius = radius + falloff;
    let max_radius_sq = max_radius * max_radius;
    let radius_sq = radius * radius;
    let falloff = falloff.max(1e-3);

    for z in 0..world.depth {
        for x in 0..world.width {
            let wx = (x as f32 + 0.5) * world.cell_size;
            let wz = (z as f32 + 0.5) * world.cell_size;
            let dx = wx - center.x;
            let dz = wz - center.z;
            let dist_sq = dx * dx + dz * dz;
            if dist_sq > max_radius_sq {
                continue;
            }

            let blend = if dist_sq <= radius_sq {
                1.0
            } else {
                let dist = dist_sq.sqrt();
                let t = (dist - radius) / falloff;
                (1.0 - t).clamp(0.0, 1.0)
            };

            if blend <= 0.0 {
                continue;
            }

            let current = world.ground_height(x, z);
            let desired = current + (target_height - current) * blend;
            set_ground_height(world, x, z, desired, true);
        }
    }
}

fn set_ground_height(world: &mut World, x: usize, z: usize, target_height: f32, dry: bool) {
    let idx = world.idx(x, z);
    let bedrock = world.bedrock_elevation[idx];
    let mut paydirt = world.paydirt_thickness[idx];
    let mut gravel = world.gravel_thickness[idx];
    let mut overburden = world.overburden_thickness[idx];
    let mut sediment = world.terrain_sediment[idx];

    let current = bedrock + paydirt + gravel + overburden + sediment;
    let delta = target_height - current;

    if delta > 0.0 {
        sediment += delta;
    } else if delta < 0.0 {
        let mut remove = -delta;
        let take = sediment.min(remove);
        sediment -= take;
        remove -= take;

        if remove > 0.0 {
            let take = overburden.min(remove);
            overburden -= take;
            remove -= take;
        }

        if remove > 0.0 {
            let take = paydirt.min(remove);
            paydirt -= take;
            remove -= take;
        }

        if remove > 0.0 {
            let take = gravel.min(remove);
            gravel -= take;
        }
    }

    world.paydirt_thickness[idx] = paydirt;
    world.gravel_thickness[idx] = gravel;
    world.overburden_thickness[idx] = overburden;
    world.terrain_sediment[idx] = sediment;

    let new_ground = bedrock + paydirt + gravel + overburden + sediment;
    if dry {
        world.water_surface[idx] = new_ground;
    } else if world.water_surface[idx] < new_ground {
        world.water_surface[idx] = new_ground;
    }
}

fn create_sluice_buffers(
    device: &wgpu::Device,
    vertices: &[SluiceVertex],
    indices: &[u32],
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let vertex_data = if vertices.is_empty() {
        vec![SluiceVertex::default()]
    } else {
        vertices.to_vec()
    };
    let index_data = if indices.is_empty() {
        vec![0u32]
    } else {
        indices.to_vec()
    };

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sluice Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sluice Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertex_buffer, index_buffer, indices.len() as u32)
}

fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

// Old build_world code kept as reference (commented out)
#[allow(dead_code)]
fn _build_world_old() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, INITIAL_HEIGHT);

    // Cascading Tailings Ponds Generator
    // Map: 512x512
    // Pond 1: Top (Z: 50-150)
    // Pond 2: Middle (Z: 200-300)
    // Pond 3: Bottom (Z: 350-450)
    // Slope: General slope North-South (Z direction)

    let center_x = (WORLD_WIDTH / 2) as f32; // 256.0

    for z in 0..WORLD_DEPTH {
        for x in 0..WORLD_WIDTH {
            let idx = world.idx(x, z);

            // 1. General Slope
            // Drop 40m over 512m
            let slope_drop = (z as f32 / WORLD_DEPTH as f32) * 40.0;
            let base_h = INITIAL_HEIGHT + 40.0 - slope_drop;

            world.bedrock_elevation[idx] = base_h * 0.5; // Deep bedrock
            world.overburden_thickness[idx] = base_h * 0.2;
            world.paydirt_thickness[idx] = base_h * 0.3;

            // 2. Sculpt Basins (Deeper areas)
            let mut basin_depth = 0.0;

            // Pond 1 (Top) - Reduced Scale
            if z > 50 && z < 100 && x > 200 && x < 312 {
                // Length 50, Width 112
                let dx = ((x as f32 - center_x) / 50.0).powi(2);
                let dz = ((z as f32 - 75.0) / 25.0).powi(2);
                let d = dx + dz;
                if d < 1.0 {
                    basin_depth = 5.0 * (1.0 - d);
                }
            }

            // Pond 2 (Middle)
            if z > 150 && z < 200 && x > 200 && x < 312 {
                let dx = ((x as f32 - center_x) / 50.0).powi(2);
                let dz = ((z as f32 - 175.0) / 25.0).powi(2);
                let d = dx + dz;
                if d < 1.0 {
                    basin_depth = 5.0 * (1.0 - d);
                }
            }

            // Pond 3 (Bottom)
            if z > 250 && z < 300 && x > 200 && x < 312 {
                let dx = ((x as f32 - center_x) / 50.0).powi(2);
                let dz = ((z as f32 - 275.0) / 25.0).powi(2);
                let d = dx + dz;
                if d < 1.0 {
                    basin_depth = 5.0 * (1.0 - d);
                }
            }

            // Apply Basin: Dig into layers
            if basin_depth > 0.0 {
                // Remove overburden first
                let ob = world.overburden_thickness[idx];
                let dug_ob = basin_depth.min(ob);
                world.overburden_thickness[idx] -= dug_ob;

                let remaining = basin_depth - dug_ob;
                if remaining > 0.0 {
                    world.paydirt_thickness[idx] =
                        (world.paydirt_thickness[idx] - remaining).max(0.0);
                }
            }

            // 3. Build Berms (Walls) below each pond
            // Berm 1 (Z~110)
            // Berm 2 (Z~210)
            // Berm 3 (Z~310)

            let mut berm_height = 0.0;

            // Berm 1
            if z > 110 && z < 115 {
                berm_height = 5.0; // Wall
                                   // Spillway (Center)
                if (x as f32 - center_x).abs() < 5.0 {
                    berm_height = 1.0;
                }
                // Pre-cut breach point (offset from spillway for testing)
                // This creates a weak point that will erode and widen
                if (x as f32 - (center_x + 15.0)).abs() < 2.0 {
                    berm_height = 3.0; // Lower than wall but higher than spillway
                }
            }

            // Berm 2
            if z > 210 && z < 215 {
                berm_height = 5.0; // Wall
                                   // Spillway (Offset Left)
                if (x as f32 - (center_x - 30.0)).abs() < 5.0 {
                    berm_height = 1.0;
                }
            }
            // Berm 3 (End Dam)
            if z > 310 && z < 315 {
                berm_height = 6.0; // Wall
                                   // Spillway (Offset Right)
                if (x as f32 - (center_x + 30.0)).abs() < 5.0 {
                    berm_height = 2.0;
                }
            }

            if berm_height > 0.0 {
                // Add Overburden pile (Berm)
                world.overburden_thickness[idx] += berm_height;
                // Compact it? No need.
            }
        }
    }

    // Pre-fill ponds with water to berm spillway level
    // Calculate reference heights for each pond's water level
    let pond1_water_level = {
        // Z~75, X~256
        let ref_idx = world.idx(256, 110); // Berm spillway location
        world.bedrock_elevation[ref_idx]
            + world.paydirt_thickness[ref_idx]
            + world.overburden_thickness[ref_idx]
            - 1.0 // Below spillway
    };
    let pond2_water_level = {
        let ref_idx = world.idx(226, 210); // Berm 2 spillway (center_x - 30)
        world.bedrock_elevation[ref_idx]
            + world.paydirt_thickness[ref_idx]
            + world.overburden_thickness[ref_idx]
            - 1.0
    };
    let pond3_water_level = {
        let ref_idx = world.idx(286, 310); // Berm 3 spillway (center_x + 30)
        world.bedrock_elevation[ref_idx]
            + world.paydirt_thickness[ref_idx]
            + world.overburden_thickness[ref_idx]
            - 1.0
    };

    for z in 0..WORLD_DEPTH {
        for x in 0..WORLD_WIDTH {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);

            // Pond 1 fill
            if z > 50 && z < 110 && x > 200 && x < 312 {
                if ground < pond1_water_level {
                    world.water_surface[idx] = pond1_water_level;
                }
            }
            // Pond 2 fill
            if z > 150 && z < 210 && x > 200 && x < 312 {
                if ground < pond2_water_level {
                    world.water_surface[idx] = pond2_water_level;
                }
            }
            // Pond 3 fill
            if z > 250 && z < 310 && x > 200 && x < 312 {
                if ground < pond3_water_level {
                    world.water_surface[idx] = pond3_water_level;
                }
            }
        }
    }

    world
}

fn build_terrain_mesh(world: &World) -> (Vec<WorldVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(world.width * world.depth * 4);
    let mut indices = Vec::with_capacity(world.width * world.depth * 6);

    for z in 0..world.depth {
        for x in 0..world.width {
            let idx = world.idx(x, z);
            let height = world.ground_height(x, z);
            let sediment = world.terrain_sediment[idx];
            let sediment_ratio = (sediment / 2.0).min(1.0);
            let base_color = match world.surface_material(x, z) {
                TerrainMaterial::Dirt => [0.4, 0.3, 0.2], // Overburden (Brown)
                TerrainMaterial::Gravel => [0.6, 0.5, 0.2], // Paydirt (Gold-ish gravel)
                TerrainMaterial::Sand => [0.8, 0.7, 0.5], // Sediment (Tan)
                TerrainMaterial::Clay => [0.6, 0.4, 0.3],
                TerrainMaterial::Bedrock => [0.2, 0.2, 0.25], // Bedrock (Dark Grey)
            };

            let sediment_color = [0.6, 0.5, 0.4];

            let color = [
                base_color[0] * (1.0 - sediment_ratio) + sediment_color[0] * sediment_ratio,
                base_color[1] * (1.0 - sediment_ratio) + sediment_color[1] * sediment_ratio,
                base_color[2] * (1.0 - sediment_ratio) + sediment_color[2] * sediment_ratio,
                1.0,
            ];

            let x0 = x as f32 * world.cell_size;
            let x1 = (x + 1) as f32 * world.cell_size;
            let z0 = z as f32 * world.cell_size;
            let z1 = (z + 1) as f32 * world.cell_size;

            let base = vertices.len() as u32;
            vertices.push(WorldVertex {
                position: [x0, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z1],
                color,
            });
            vertices.push(WorldVertex {
                position: [x0, height, z1],
                color,
            });

            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

            // Add side faces where neighbor is lower
            let side_color = [
                (color[0] * 0.7).max(0.0), // Darken sides for depth
                (color[1] * 0.7).max(0.0),
                (color[2] * 0.7).max(0.0),
                1.0,
            ];

            // Check each neighbor and add side quad if lower (0.5m threshold for perf)
            const SIDE_THRESHOLD: f32 = 0.5;

            // Right neighbor (X+1)
            if x + 1 < world.width {
                let neighbor_height = world.ground_height(x + 1, z);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x1, height, z0],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, height, z1],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, neighbor_height, z1],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, neighbor_height, z0],
                        color: side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }

            // Front neighbor (Z+1)
            if z + 1 < world.depth {
                let neighbor_height = world.ground_height(x, z + 1);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x0, height, z1],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, height, z1],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, neighbor_height, z1],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, neighbor_height, z1],
                        color: side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }

            // Left neighbor (X-1)
            if x > 0 {
                let neighbor_height = world.ground_height(x - 1, z);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x0, height, z1],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, height, z0],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, neighbor_height, z0],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, neighbor_height, z1],
                        color: side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }

            // Back neighbor (Z-1)
            if z > 0 {
                let neighbor_height = world.ground_height(x, z - 1);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x1, height, z0],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, height, z0],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, neighbor_height, z0],
                        color: side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, neighbor_height, z0],
                        color: side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }
        }
    }

    (vertices, indices)
}

fn build_water_mesh(world: &World) -> (Vec<WorldVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for z in 0..world.depth {
        for x in 0..world.width {
            let depth = world.water_depth(x, z);
            if depth < 0.001 {
                continue;
            }

            let idx = world.idx(x, z);
            let height = world.water_surface[idx];
            let turbidity = world.suspended_sediment[idx];

            let alpha = 0.5 + (depth).min(0.3); // Visible even when shallow
            let brown = turbidity.min(0.5) * 2.0;
            let color = [
                0.2 + brown * 0.4,
                0.4 + brown * 0.2,
                0.8 - brown * 0.4,
                alpha,
            ];

            let x0 = x as f32 * world.cell_size;
            let x1 = (x + 1) as f32 * world.cell_size;
            let z0 = z as f32 * world.cell_size;
            let z1 = (z + 1) as f32 * world.cell_size;

            let base = vertices.len() as u32;
            vertices.push(WorldVertex {
                position: [x0, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z0],
                color,
            });
            vertices.push(WorldVertex {
                position: [x1, height, z1],
                color,
            });
            vertices.push(WorldVertex {
                position: [x0, height, z1],
                color,
            });

            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

            // Add side faces for water (down to ground level)
            let ground_height = world.ground_height(x, z);
            let water_side_color = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.9, alpha * 0.8];

            // Right side (X+1)
            if x + 1 < world.width {
                let neighbor_water = world.water_surface[world.idx(x + 1, z)];
                let neighbor_ground = world.ground_height(x + 1, z);
                let neighbor_depth = neighbor_water - neighbor_ground;
                if neighbor_depth < 0.001 {
                    // No water in neighbor, draw side down to ground
                    let bottom = ground_height.max(neighbor_ground);
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x1, height, z0],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, height, z1],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, bottom, z1],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, bottom, z0],
                        color: water_side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }

            // Front side (Z+1)
            if z + 1 < world.depth {
                let neighbor_water = world.water_surface[world.idx(x, z + 1)];
                let neighbor_ground = world.ground_height(x, z + 1);
                let neighbor_depth = neighbor_water - neighbor_ground;
                if neighbor_depth < 0.001 {
                    let bottom = ground_height.max(neighbor_ground);
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x0, height, z1],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, height, z1],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, bottom, z1],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, bottom, z1],
                        color: water_side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }

            // Left side (X-1)
            if x > 0 {
                let neighbor_water = world.water_surface[world.idx(x - 1, z)];
                let neighbor_ground = world.ground_height(x - 1, z);
                let neighbor_depth = neighbor_water - neighbor_ground;
                if neighbor_depth < 0.001 {
                    let bottom = ground_height.max(neighbor_ground);
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x0, height, z1],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, height, z0],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, bottom, z0],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, bottom, z1],
                        color: water_side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }

            // Back side (Z-1)
            if z > 0 {
                let neighbor_water = world.water_surface[world.idx(x, z - 1)];
                let neighbor_ground = world.ground_height(x, z - 1);
                let neighbor_depth = neighbor_water - neighbor_ground;
                if neighbor_depth < 0.001 {
                    let bottom = ground_height.max(neighbor_ground);
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex {
                        position: [x1, height, z0],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, height, z0],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x0, bottom, z0],
                        color: water_side_color,
                    });
                    vertices.push(WorldVertex {
                        position: [x1, bottom, z0],
                        color: water_side_color,
                    });
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }
        }
    }

    (vertices, indices)
}

// Particle shader - blue billboards with optional velocity coloring
const PARTICLE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_size: f32,
    camera_right: vec3<f32>,
    show_velocity: f32,
    camera_up: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) velocity_mag: f32,
}

@vertex
fn vs_main(
    @location(0) vertex: vec2<f32>,
    @location(1) pos_vel: vec4<f32>, // xyz = position, w = velocity magnitude
) -> VertexOutput {
    let pos = pos_vel.xyz;
    let vel_mag = pos_vel.w;
    let size = uniforms.particle_size;
    let offset_world = (uniforms.camera_right * vertex.x + uniforms.camera_up * vertex.y) * size;
    let clip_pos = uniforms.view_proj * vec4<f32>(pos + offset_world, 1.0);

    var out: VertexOutput;
    out.position = clip_pos;
    out.uv = vertex;
    out.velocity_mag = vel_mag;
    return out;
}

// Velocity to color mapping (blue = slow, green = medium, red = fast)
fn velocity_color(vel_mag: f32) -> vec3<f32> {
    // Normalize velocity to 0-1 range (assuming max ~2.0 m/s in sluice)
    let t = clamp(vel_mag / 1.5, 0.0, 1.0);

    // Blue -> Cyan -> Green -> Yellow -> Red
    if (t < 0.25) {
        let s = t / 0.25;
        return mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), s);
    } else if (t < 0.5) {
        let s = (t - 0.25) / 0.25;
        return mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), s);
    } else if (t < 0.75) {
        let s = (t - 0.5) / 0.25;
        return mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), s);
    } else {
        let s = (t - 0.75) / 0.25;
        return mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), s);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }
    let shading = 1.0 - dist * 0.4;

    var color: vec3<f32>;
    if (uniforms.show_velocity > 0.5) {
        // Velocity-based coloring
        color = velocity_color(in.velocity_mag) * shading;
    } else {
        // Default blue/cyan water color
        color = vec3<f32>(0.0, 0.5 * shading, 1.0 * shading);
    }
    return vec4<f32>(color, 1.0);
}
"#;

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_size: f32,
    camera_right: vec3<f32>,
    _pad0: f32,
    camera_up: vec3<f32>,
    highlight_tint: f32,
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
    // Simple ambient lighting for equipment
    let ambient = 0.6;

    // Distance-based fog
    let dist_to_cam = length(in.world_pos - uniforms.camera_pos);
    let fog_factor = clamp(dist_to_cam / 100.0, 0.0, 0.3);
    let fog_color = vec3<f32>(0.6, 0.7, 0.8);

    var final_color = in.color.rgb * ambient;
    final_color = mix(final_color, fog_color, fog_factor);

    // Apply highlight tint
    final_color = final_color * uniforms.highlight_tint;

    return vec4<f32>(final_color, in.color.a);
}
"#;

/// Create a small sphere mesh for visualization
fn create_small_sphere(center: Vec3, radius: f32, resolution: u32) -> (Vec<Vec3>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    let segments = resolution;
    let rings = resolution / 2;

    for r in 0..=rings {
        let theta = std::f32::consts::PI * r as f32 / rings as f32;
        let y = radius * theta.cos();
        let ring_radius = radius * theta.sin();

        for s in 0..=segments {
            let phi = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;
            let x = ring_radius * phi.cos();
            let z = ring_radius * phi.sin();

            positions.push(center + Vec3::new(x, y, z));
        }
    }

    for r in 0..rings {
        for s in 0..segments {
            let cur = r * (segments + 1) + s;
            let next = cur + segments + 1;

            indices.push(cur as u32);
            indices.push((cur + 1) as u32);
            indices.push(next as u32);

            indices.push((cur + 1) as u32);
            indices.push((next + 1) as u32);
            indices.push(next as u32);
        }
    }

    (positions, indices)
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
