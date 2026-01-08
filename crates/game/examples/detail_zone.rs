//! Detail Zone Test - FLIP + Heightfield Integration
//!
//! Tests 3D FLIP fluid simulation over a terrain heightfield.
//! Uses FlipSimulation3D for particle/grid data and GpuFlip3D for GPU compute.
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look
//! - 1: Spawn water at cursor
//! - R: Reset
//! - ESC: Quit
//!
//! Run: cargo run --example detail_zone --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use sim3d::{FlipSimulation3D, TerrainMaterial, World};
use game::gpu::flip_3d::GpuFlip3D;
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

const MOVE_SPEED: f32 = 20.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

const STEPS_PER_FRAME: usize = 10; // More steps for faster filling
const DT: f32 = 0.02;

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
    _pad: f32,
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
        let data = if indices.is_empty() { [0u32] } else { [indices[0]] };

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

struct WaterEmitter {
    position: Vec3,
    rate: f32, // Volume per second
    radius: f32,
    enabled: bool,
}

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
    selected_material: u32,
    terrain_dirty: bool,
    
    // FLIP simulation
    flip_sim: FlipSimulation3D,
    flip_origin: Vec3,
    gpu_flip: Option<GpuFlip3D>,
    flip_positions: Vec<Vec3>,
    flip_velocities: Vec<Vec3>,
    flip_c_matrices: Vec<Mat3>,
    flip_densities: Vec<f32>,
    flip_cell_types: Vec<u32>,
}

impl App {
    fn new() -> Self {
        let world = build_world();
        
        // Create a small FLIP zone: 64x32x64 cells at 1.0m = 64x32x64m
        // This overlays a portion of the heightfield for testing
        const FLIP_GRID_X: usize = 64;
        const FLIP_GRID_Y: usize = 32;
        const FLIP_GRID_Z: usize = 64;
        const FLIP_CELL_SIZE: f32 = 1.0;
        
        let mut flip_sim = FlipSimulation3D::new(FLIP_GRID_X, FLIP_GRID_Y, FLIP_GRID_Z, FLIP_CELL_SIZE);
        flip_sim.gravity = Vec3::new(0.0, -9.8, 0.0);
        flip_sim.flip_ratio = 0.95;
        flip_sim.pressure_iterations = 80;
        
        // Build bowl-shaped floor in center of FLIP grid
        for k in 0..FLIP_GRID_Z {
            for i in 0..FLIP_GRID_X {
                let cx = FLIP_GRID_X as f32 * 0.5;
                let cz = FLIP_GRID_Z as f32 * 0.5;
                let dx = (i as f32 - cx) / (FLIP_GRID_X as f32 * 0.4);
                let dz = (k as f32 - cz) / (FLIP_GRID_Z as f32 * 0.4);
                let dist_sq = dx * dx + dz * dz;
                
                let floor_height = if dist_sq < 1.0 {
                    2.0 + 6.0 * dist_sq  // Bowl: 2m center, rising to 8m at edges
                } else {
                    8.0  // Rim height
                };
                
                let floor_cells = (floor_height / FLIP_CELL_SIZE).ceil() as usize;
                for j in 0..floor_cells.min(FLIP_GRID_Y) {
                    flip_sim.grid.set_solid(i, j, k);
                }
            }
        }
        
        println!("=== DETAIL ZONE: FLIP + Heightfield ===");
        println!("FLIP grid: {}x{}x{} at {:.1}m", FLIP_GRID_X, FLIP_GRID_Y, FLIP_GRID_Z, FLIP_CELL_SIZE);
        println!("Press 1 to spawn water");

        // Initialize flip_origin to center of world at terrain height
        let center_x = (WORLD_WIDTH / 2) as usize;
        let center_z = (WORLD_DEPTH / 2) as usize;
        let terrain_y = world.ground_height(center_x, center_z);
        // We set the origin so the terrain is roughly at Y=5 in our 32m high grid
        let flip_origin = Vec3::new(
            (center_x as f32) - (FLIP_GRID_X as f32 * 0.5),
            terrain_y - 5.0,
            (center_z as f32) - (FLIP_GRID_Z as f32 * 0.5),
        );

        Self {
            window: None,
            gpu: None,
            world,
            emitter: WaterEmitter {
                position: Vec3::new(center_x as f32, terrain_y + 10.0, center_z as f32),
                rate: 50.0,
                radius: 5.0,
                enabled: false,
            },
            camera: Camera {
                position: Vec3::new(center_x as f32, terrain_y + 50.0, center_z as f32 + 50.0), 
                yaw: -1.57, // Looking -Z
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
            
            // FLIP
            flip_sim,
            flip_origin,
            gpu_flip: None, // Initialized when GPU is ready
            flip_positions: Vec::new(),
            flip_velocities: Vec::new(),
            flip_c_matrices: Vec::new(),
            flip_densities: Vec::new(),
            flip_cell_types: Vec::new(),
        }
    }

    fn reset_world(&mut self) {
        self.world = build_world();
    }

    fn update(&mut self, dt: f32) {
        self.update_camera(dt);

        if let Some(gpu) = self.gpu.as_mut() {
            // Run GPU Sim with fixed timestep substepping
            let sim_dt = DT; // Use fixed 0.02s timestep
            let steps = ((dt / sim_dt).ceil() as usize).min(STEPS_PER_FRAME);
            
            for _ in 0..steps {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Sim Encoder"),
                });
                
                // Update GPU emitter and dispatch (before water sim)
                gpu.heightfield.update_emitter(
                    &gpu.queue,
                    self.emitter.position.x,
                    self.emitter.position.z,
                    self.emitter.radius,
                    self.emitter.rate,
                    sim_dt,
                    self.emitter.enabled,
                );
                gpu.heightfield.dispatch_emitter(&mut encoder);
                
                // Update water sim params and dispatch
                gpu.heightfield.update_params(&gpu.queue, sim_dt);
                gpu.heightfield.dispatch(&mut encoder, sim_dt);
                
                gpu.queue.submit(Some(encoder.finish()));
            }
            
            // Sync back to CPU for rendering
            pollster::block_on(gpu.heightfield.download_to_world(&gpu.device, &gpu.queue, &mut self.world));
            
        } else {
            // Fallback if no GPU? (Shouldn't happen in this example)
            // self.world.update(dt);
        }
        
        // ===== FLIP Simulation Step =====
        // Auto-spawn water for verification (or when pressing 1)
        let auto_spawn = self.flip_sim.frame < 60 && self.flip_sim.particle_count() < 5000;
        if (auto_spawn || self.input.keys.contains(&KeyCode::Digit1)) && self.flip_sim.particle_count() < 100_000 {
            let grid = &self.flip_sim.grid;
            let spawn_x = grid.width as f32 * grid.cell_size * 0.5;
            let spawn_y = grid.height as f32 * grid.cell_size * 0.5; // Lowered to 50% height
            let spawn_z = grid.depth as f32 * grid.cell_size * 0.5;
            let pos = Vec3::new(spawn_x, spawn_y, spawn_z);
            
            let d = 4; // Fewer particles per frame (4x4x4=64)
            for i in 0..d {
                for j in 0..d {
                    for k in 0..d {
                        let offset = (Vec3::new(i as f32, j as f32 * 2.0, k as f32) / d as f32 - 0.5) * 2.0;
                        let jitter = Vec3::new(
                            ((i * 123 + j * 456 + k * 789) as f32 * 0.123).fract() - 0.5,
                            ((i * 789 + j * 123 + k * 456) as f32 * 0.456).fract() - 0.5,
                            ((i * 456 + j * 789 + k * 123) as f32 * 0.789).fract() - 0.5,
                        ) * 0.5;
                        self.flip_sim.spawn_particle(pos + offset + jitter);
                    }
                }
            }
        }
        
        // Run FLIP GPU step
        if self.flip_sim.particle_count() > 0 {
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
                
                // Update FLIP grid solids based on World heightfield terrain
                let w = self.flip_sim.grid.width;
                let h = self.flip_sim.grid.height;
                let d = self.flip_sim.grid.depth;
                let cell_size = self.flip_sim.grid.cell_size;

                for k in 0..d {
                    for i in 0..w {
                        // Sample terrain at center of cell in WORLD coordinates
                        let wx = (i as f32 + 0.5) * cell_size + self.flip_origin.x;
                        let wz = (k as f32 + 0.5) * cell_size + self.flip_origin.z;
                        
                        // Get terrain height at this coordinate
                        if let Some((ux, uz)) = self.world.world_to_cell(Vec3::new(wx, 0.0, wz)) {
                            let terrain_height = self.world.ground_height(ux, uz);
                            
                            // Mark cells below terrain as solid
                            for j in 0..h {
                                let cell_y = (j as f32 + 0.5) * cell_size + self.flip_origin.y;
                                if cell_y <= terrain_height {
                                    self.flip_sim.grid.set_solid(i, j, k);
                                } else {
                                    // Important: clear solids above terrain if user dug a hole
                                    self.flip_sim.grid.clear_solid(i, j, k);
                                }
                            }
                        }
                    }
                }

                // Recalculate SDF if solids changed
                self.flip_sim.grid.compute_sdf();

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
                
                // Mark FLUID cells from particles
                for p in &self.flip_sim.particles.list {
                    let cell_size = self.flip_sim.grid.cell_size;
                    let i = (p.position.x / cell_size).floor() as i32;
                    let j = (p.position.y / cell_size).floor() as i32;
                    let k = (p.position.z / cell_size).floor() as i32;
                    if i >= 0 && i < w as i32 && j >= 0 && j < h as i32 && k >= 0 && k < d as i32 {
                        let idx = k as usize * w * h + j as usize * w + i as usize;
                        if self.flip_cell_types[idx] != 2 {
                            self.flip_cell_types[idx] = 1; // FLUID
                        }
                    }
                }
                
                let sdf = self.flip_sim.grid.sdf.as_slice();
                
                // Force SDF re-upload as solids might have changed
                gpu_flip.upload_sdf(&gpu.queue, &self.flip_sim.grid.sdf);

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
                    1.0 / 60.0,
                    -9.8,
                    0.0,
                    self.flip_sim.pressure_iterations as u32,
                );
                
                // Sync results back to FlipSimulation3D
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
                        max_vy = if vy < 0.0 { max_vy.min(vy) } else { max_vy.max(vy) };
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

        // Handle mouse click - material tool
        if self.input.left_mouse {
            if let Some(hit) = self.raycast_terrain() {
                if let Some(gpu) = self.gpu.as_ref() {
                    // Ctrl = add material, else = excavate
                    let is_adding = self.input.keys.contains(&KeyCode::ControlLeft)
                        || self.input.keys.contains(&KeyCode::ControlRight);
                    
                    let amount = if is_adding { ADD_HEIGHT * 50.0 } else { -(DIG_DEPTH * 50.0) };
                    let radius = if is_adding { ADD_RADIUS } else { DIG_RADIUS };
                    
                    // Update material tool params
                    gpu.heightfield.update_material_tool(
                        &gpu.queue,
                        hit.x, hit.z,
                        radius,
                        amount,
                        self.selected_material,
                        dt,
                        true, // enabled
                    );
                    
                    // Dispatch appropriate tool
                    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            
            println!("Heightfield water: {:.2}, sediment: {:.2} | FLIP particles: {}", water, sediment, flip_count);
            self.last_stats = Instant::now();
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
        if self.input.keys.contains(&KeyCode::ShiftLeft) || self.input.keys.contains(&KeyCode::ShiftRight) {
            direction.y -= 1.0;
        }

        if direction.length_squared() > 0.0 {
            self.camera.position += direction.normalize() * self.camera.speed * dt;
        }

        if let Some(gpu) = &mut self.gpu {
            // ... (Wait, this is update_camera, not update?)
            // update_camera has no gpu argument.
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
            label: Some("World Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<WorldVertex>() as u64,
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

        let heightfield = GpuHeightfield::new(
            &device, 
            self.world.width as u32, 
            self.world.depth as u32, 
            self.world.cell_size,
            INITIAL_HEIGHT,
            config.format,
        );
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
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3],
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

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            _pad: 0.0,
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
        );
        
        // ===== Render FLIP particles =====
        let particle_count = self.flip_sim.particle_count();
        if particle_count > 0 {
            // Upload particle positions (with flip_origin offset)
            let positions: Vec<[f32; 4]> = self.flip_sim.particles.list.iter()
                .map(|p| {
                    let world_pos = p.position + self.flip_origin;
                    [world_pos.x, world_pos.y, world_pos.z, 0.0]
                })
                .collect();
            gpu.queue.write_buffer(&gpu.particle_instance_buffer, 0, bytemuck::cast_slice(&positions));
            
            // Update uniforms for particle rendering
            gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            
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
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    gpu.depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
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
                                        self.emitter.position = hit;
                                        self.emitter.enabled = true;
                                        println!("Emitter placed at {:?}", hit);
                                    } else {
                                        self.emitter.enabled = !self.emitter.enabled;
                                        println!("Emitter enabled: {}", self.emitter.enabled);
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
                        self.camera.pitch = (self.camera.pitch - dy * self.camera.sensitivity)
                            .clamp(-1.4, 1.4);
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

fn build_world() -> World {
    use sim3d::{generate_klondike_terrain, TerrainConfig};
    
    let config = TerrainConfig::default();
    generate_klondike_terrain(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, &config)
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
            if z > 50 && z < 100 && x > 200 && x < 312 { // Length 50, Width 112
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
                     world.paydirt_thickness[idx] = (world.paydirt_thickness[idx] - remaining).max(0.0);
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
    let pond1_water_level = { // Z~75, X~256
        let ref_idx = world.idx(256, 110); // Berm spillway location
        world.bedrock_elevation[ref_idx] + world.paydirt_thickness[ref_idx] + world.overburden_thickness[ref_idx] - 1.0 // Below spillway
    };
    let pond2_water_level = {
        let ref_idx = world.idx(226, 210); // Berm 2 spillway (center_x - 30)
        world.bedrock_elevation[ref_idx] + world.paydirt_thickness[ref_idx] + world.overburden_thickness[ref_idx] - 1.0
    };
    let pond3_water_level = {
        let ref_idx = world.idx(286, 310); // Berm 3 spillway (center_x + 30)
        world.bedrock_elevation[ref_idx] + world.paydirt_thickness[ref_idx] + world.overburden_thickness[ref_idx] - 1.0
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
                    vertices.push(WorldVertex { position: [x1, height, z0], color: side_color });
                    vertices.push(WorldVertex { position: [x1, height, z1], color: side_color });
                    vertices.push(WorldVertex { position: [x1, neighbor_height, z1], color: side_color });
                    vertices.push(WorldVertex { position: [x1, neighbor_height, z0], color: side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            
            // Front neighbor (Z+1)
            if z + 1 < world.depth {
                let neighbor_height = world.ground_height(x, z + 1);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex { position: [x0, height, z1], color: side_color });
                    vertices.push(WorldVertex { position: [x1, height, z1], color: side_color });
                    vertices.push(WorldVertex { position: [x1, neighbor_height, z1], color: side_color });
                    vertices.push(WorldVertex { position: [x0, neighbor_height, z1], color: side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            
            // Left neighbor (X-1)
            if x > 0 {
                let neighbor_height = world.ground_height(x - 1, z);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex { position: [x0, height, z1], color: side_color });
                    vertices.push(WorldVertex { position: [x0, height, z0], color: side_color });
                    vertices.push(WorldVertex { position: [x0, neighbor_height, z0], color: side_color });
                    vertices.push(WorldVertex { position: [x0, neighbor_height, z1], color: side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            
            // Back neighbor (Z-1)
            if z > 0 {
                let neighbor_height = world.ground_height(x, z - 1);
                if neighbor_height < height - SIDE_THRESHOLD {
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex { position: [x1, height, z0], color: side_color });
                    vertices.push(WorldVertex { position: [x0, height, z0], color: side_color });
                    vertices.push(WorldVertex { position: [x0, neighbor_height, z0], color: side_color });
                    vertices.push(WorldVertex { position: [x1, neighbor_height, z0], color: side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
            let water_side_color = [
                color[0] * 0.8,
                color[1] * 0.8,
                color[2] * 0.9,
                alpha * 0.8,
            ];
            
            // Right side (X+1)
            if x + 1 < world.width {
                let neighbor_water = world.water_surface[world.idx(x + 1, z)];
                let neighbor_ground = world.ground_height(x + 1, z);
                let neighbor_depth = neighbor_water - neighbor_ground;
                if neighbor_depth < 0.001 {
                    // No water in neighbor, draw side down to ground
                    let bottom = ground_height.max(neighbor_ground);
                    let base = vertices.len() as u32;
                    vertices.push(WorldVertex { position: [x1, height, z0], color: water_side_color });
                    vertices.push(WorldVertex { position: [x1, height, z1], color: water_side_color });
                    vertices.push(WorldVertex { position: [x1, bottom, z1], color: water_side_color });
                    vertices.push(WorldVertex { position: [x1, bottom, z0], color: water_side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
                    vertices.push(WorldVertex { position: [x0, height, z1], color: water_side_color });
                    vertices.push(WorldVertex { position: [x1, height, z1], color: water_side_color });
                    vertices.push(WorldVertex { position: [x1, bottom, z1], color: water_side_color });
                    vertices.push(WorldVertex { position: [x0, bottom, z1], color: water_side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
                    vertices.push(WorldVertex { position: [x0, height, z1], color: water_side_color });
                    vertices.push(WorldVertex { position: [x0, height, z0], color: water_side_color });
                    vertices.push(WorldVertex { position: [x0, bottom, z0], color: water_side_color });
                    vertices.push(WorldVertex { position: [x0, bottom, z1], color: water_side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
                    vertices.push(WorldVertex { position: [x1, height, z0], color: water_side_color });
                    vertices.push(WorldVertex { position: [x0, height, z0], color: water_side_color });
                    vertices.push(WorldVertex { position: [x0, bottom, z0], color: water_side_color });
                    vertices.push(WorldVertex { position: [x1, bottom, z0], color: water_side_color });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }
    }

    (vertices, indices)
}

// Particle shader - blue billboards
const PARTICLE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) vertex: vec2<f32>,
    @location(1) pos: vec3<f32>,
) -> VertexOutput {
    let clip_pos = uniforms.view_proj * vec4<f32>(pos, 1.0);
    
    // Normalized device coordinates (screen space)
    let size = 0.01;
    let offset = vertex * size;
    
    var out: VertexOutput;
    out.position = vec4<f32>(
        clip_pos.x + offset.x * clip_pos.w, 
        clip_pos.y + offset.y * clip_pos.w, 
        clip_pos.z, 
        clip_pos.w
    );
    out.uv = vertex;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }
    // Blue/Cyan water color
    let shading = 1.0 - dist * 0.4;
    return vec4<f32>(0.0, 0.5 * shading, 1.0 * shading, 1.0); 
}
"#;

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

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
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
