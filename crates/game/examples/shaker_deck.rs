//! Shaker Deck Demo
//!
//! Demonstrates a vibrating shaker deck using DEM clumps for coarse gravel
//! and gold nuggets, with fine sediment intended for the FLIP system.
//!
//! Run with: cargo run --example shaker_deck --release

use bytemuck::{Pod, Zeroable};
use game::gpu::flip_3d::GpuFlip3D;
use game::gpu::fluid_renderer::ScreenSpaceFluidRenderer;
use glam::{Mat3, Mat4, Vec3};
use sim3d::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FlipSimulation3D, IrregularStyle3D};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const MAX_INSTANCES: usize = 80_000;
const DT: f32 = 1.0 / 240.0;
const FLIP_DT: f32 = 1.0 / 60.0;
const SUBSTEPS: usize = 4;
const CLUSTER_VISUAL_SCALE: f32 = 0.9;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const DECK_LENGTH: f32 = 6.0;
const DECK_WIDTH: f32 = 2.2;
const DECK_HEIGHT: f32 = 2.0;
const DECK_ANGLE_DEG: f32 = 12.0;
const HOLE_SPACING: f32 = 0.18;
const APERTURE_DIAMETER: f32 = 0.06;

const SHAKE_FREQ: f32 = 18.0;
const SHAKE_AMP: f32 = 0.012;
const SHAKE_TANGENT: f32 = 0.55;
const WATER_DRAG: f32 = 1.6;
const WATER_FLOW: f32 = 2.8;

const FLIP_CELL_SIZE: f32 = 0.05;
const FLIP_MAX_PARTICLES: usize = 12_000;
const FLIP_PRESSURE_ITERS: u32 = 40;
const FLIP_SUBSTEPS: u32 = 1;
const FLIP_GRAVITY: f32 = -9.8;
const FLIP_FLOW_ACCEL: f32 = 1.2;
const FLIP_GPU_SYNC_STRIDE: u32 = 6;

const SPRAY_BAR_COUNT: usize = 3;
const SPRAY_BAR_HEIGHT: f32 = 0.35;
const SPRAY_INTERVAL_FRAMES: u32 = 4;
const SPRAY_PARTICLES_PER_BAR: usize = 8;
const SPRAY_SPREAD_Z: f32 = 0.12;
const SPRAY_DOWN_SPEED: f32 = -1.5;
const SPRAY_TANGENT_SPEED: f32 = 0.8;
const GRAVEL_DENSITY: f32 = 2.7;
const HEAVY_MINERAL_DENSITY: f32 = 4.6;
const GOLD_DENSITY: f32 = 19.3;

// Bulk mix (by mass). Sand/heavy/silt/gold flour are intended for FLIP sediment.
const BULK_GRAVEL_FRACTION: f32 = 0.70;
const BULK_SAND_FRACTION: f32 = 0.24;
const BULK_HEAVY_FRACTION: f32 = 0.002;
const BULK_GOLD_FRACTION: f32 = 0.0000003;
const BULK_SILT_FRACTION: f32 = 1.0
    - (BULK_GRAVEL_FRACTION + BULK_SAND_FRACTION + BULK_HEAVY_FRACTION + BULK_GOLD_FRACTION);
const GOLD_NUGGET_SHARE: f32 = 0.25; // remainder is gold flour -> FLIP
const SEDIMENT_TOTAL_FRACTION: f32 = BULK_SAND_FRACTION
    + BULK_SILT_FRACTION
    + BULK_HEAVY_FRACTION
    + BULK_GOLD_FRACTION * (1.0 - GOLD_NUGGET_SHARE);
const SEDIMENT_SAND_SHARE: f32 = BULK_SAND_FRACTION / SEDIMENT_TOTAL_FRACTION;
const SEDIMENT_SILT_SHARE: f32 = BULK_SILT_FRACTION / SEDIMENT_TOTAL_FRACTION;
const SEDIMENT_HEAVY_SHARE: f32 = BULK_HEAVY_FRACTION / SEDIMENT_TOTAL_FRACTION;
const SEDIMENT_GOLD_SHARE: f32 =
    (BULK_GOLD_FRACTION * (1.0 - GOLD_NUGGET_SHARE)) / SEDIMENT_TOTAL_FRACTION;

// DEM gravel bins (meters). Keeps counts reasonable while matching coarse mix.
const GRAVEL_RADII: [f32; 4] = [0.04, 0.03, 0.02, 0.012];
const GRAVEL_BIN_MASS: [f32; 4] = [0.35, 0.30, 0.20, 0.15];
const TARGET_GRAVEL_COUNT: usize = 48;

const GOLD_NUGGET_RADII: [f32; 2] = [0.006, 0.003];
const GOLD_NUGGET_MASS: [f32; 2] = [0.7, 0.3];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneVertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterInstance {
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
    base_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaterialTag {
    Rock,
    Gold,
}

struct DeckSpec {
    origin: Vec3,
    length: f32,
    width: f32,
    slope: f32,
    hole_spacing: f32,
    hole_radius: f32,
}

impl DeckSpec {
    fn height_at(&self, x: f32) -> f32 {
        self.origin.y + self.slope * (x - self.origin.x)
    }

    fn inside_bounds(&self, pos: Vec3) -> bool {
        pos.x >= self.origin.x
            && pos.x <= self.origin.x + self.length
            && pos.z >= self.origin.z
            && pos.z <= self.origin.z + self.width
    }

    fn in_hole(&self, pos: Vec3) -> bool {
        let local_x = pos.x - self.origin.x;
        let local_z = pos.z - self.origin.z;
        let gx = (local_x / self.hole_spacing).round();
        let gz = (local_z / self.hole_spacing).round();
        let cx = gx * self.hole_spacing;
        let cz = gz * self.hole_spacing;
        let dx = local_x - cx;
        let dz = local_z - cz;
        dx * dx + dz * dz <= self.hole_radius * self.hole_radius
    }

    fn normal(&self) -> Vec3 {
        Vec3::new(-self.slope, 1.0, 0.0).normalize()
    }

    fn tangent(&self) -> Vec3 {
        Vec3::new(1.0, self.slope, 0.0).normalize()
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sim: ClusterSimulation3D,
    flip: FlipSimulation3D,
    gpu_flip: Option<GpuFlip3D>,
    fluid_renderer: Option<ScreenSpaceFluidRenderer>,
    template_colors: Vec<[f32; 4]>,
    instances: Vec<ClusterInstance>,
    round_instances: Vec<ClusterInstance>,
    sharp_instances: Vec<ClusterInstance>,
    materials: Vec<MaterialTag>,
    flip_positions: Vec<Vec3>,
    flip_velocities: Vec<Vec3>,
    flip_affine: Vec<Mat3>,
    flip_densities: Vec<f32>,
    flip_cell_types: Vec<u32>,
    flip_gpu_readback_pending: bool,
    flip_gpu_sync_substep: u32,
    flip_gpu_needs_upload: bool,
    frame: u32,
    time: f32,
    paused: bool,
    camera_angle: f32,
    camera_distance: f32,
    camera_height: f32,
    deck: DeckSpec,
    gold_underflow: usize,
    gold_overflow: usize,
    counted: Vec<bool>,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    plane_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    plane_vertex_buffer: wgpu::Buffer,
    plane_vertex_count: u32,
    round_mesh_vertex_buffer: wgpu::Buffer,
    round_mesh_vertex_count: u32,
    sharp_mesh_vertex_buffer: wgpu::Buffer,
    sharp_mesh_vertex_count: u32,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl App {
    fn new() -> Self {
        let (sim, template_colors, materials, deck) = build_sim();
        let flip = build_flip(&deck);
        let (camera_distance, camera_height) = camera_defaults(&sim);
        let counted = vec![false; sim.clumps.len()];

        println!(
            "Shaker deck: rocks={} gold nuggets={}",
            materials.iter().filter(|m| **m == MaterialTag::Rock).count(),
            materials.iter().filter(|m| **m == MaterialTag::Gold).count()
        );
        println!("Controls: SPACE=pause, R=reset, arrows=orbit/zoom");

        Self {
            window: None,
            gpu: None,
            sim,
            flip,
            gpu_flip: None,
            fluid_renderer: None,
            template_colors,
            instances: Vec::new(),
            round_instances: Vec::new(),
            sharp_instances: Vec::new(),
            materials,
            flip_positions: Vec::new(),
            flip_velocities: Vec::new(),
            flip_affine: Vec::new(),
            flip_densities: Vec::new(),
            flip_cell_types: Vec::new(),
            flip_gpu_readback_pending: false,
            flip_gpu_sync_substep: 0,
            flip_gpu_needs_upload: true,
            frame: 0,
            time: 0.0,
            paused: false,
            camera_angle: 0.6,
            camera_distance,
            camera_height,
            deck,
            gold_underflow: 0,
            gold_overflow: 0,
            counted,
        }
    }

    fn reset_sim(&mut self) {
        let (sim, template_colors, materials, deck) = build_sim();
        let (camera_distance, camera_height) = camera_defaults(&sim);
        self.sim = sim;
        self.flip = build_flip(&deck);
        self.flip_positions.clear();
        self.flip_velocities.clear();
        self.flip_affine.clear();
        self.flip_densities.clear();
        self.flip_cell_types.clear();
        self.flip_gpu_readback_pending = false;
        self.flip_gpu_sync_substep = 0;
        self.flip_gpu_needs_upload = true;
        self.frame = 0;
        self.template_colors = template_colors;
        self.materials = materials;
        self.deck = deck;
        self.time = 0.0;
        self.gold_underflow = 0;
        self.gold_overflow = 0;
        self.counted = vec![false; self.sim.clumps.len()];
        self.camera_distance = camera_distance;
        self.camera_height = camera_height;
    }

    fn update_sim(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        let sub_dt = dt / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.time += sub_dt;
            let shake_accel = self.shake_accel();
            self.apply_shake(shake_accel, sub_dt);
            self.sim.step(sub_dt);
            self.apply_deck_constraints(sub_dt);
            self.apply_water_drag(sub_dt);
            self.update_counts();
        }
    }

    fn update_flip(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        let mut readback_completed = false;
        if self.flip_gpu_readback_pending {
            let count = {
                let gpu = match &self.gpu {
                    Some(gpu) => gpu,
                    None => return,
                };
                let gpu_flip = match &mut self.gpu_flip {
                    Some(gpu_flip) => gpu_flip,
                    None => return,
                };
                gpu_flip.try_readback(
                    &gpu.device,
                    &mut self.flip_positions,
                    &mut self.flip_velocities,
                    &mut self.flip_affine,
                )
            };
            if let Some(count) = count {
                self.flip_gpu_readback_pending = false;
                self.apply_flip_readback(count);
                readback_completed = true;
                self.flip_gpu_needs_upload = true;
            } else {
                return;
            }
        }

        if (self.flip.particles.is_empty() || readback_completed)
            && self.frame % SPRAY_INTERVAL_FRAMES == 0
        {
            self.emit_spray();
        }

        if self.flip_gpu_needs_upload {
            self.prepare_flip_inputs();
        }

        let particle_count = self.flip.particles.list.len();
        if particle_count == 0 {
            self.frame = self.frame.wrapping_add(1);
            return;
        }

        let sdf = self.flip.grid.sdf.as_slice();
        let dt_sub = dt / FLIP_SUBSTEPS as f32;

        if self.flip_gpu_needs_upload {
            let (gpu_flip, gpu) = match (&mut self.gpu_flip, &self.gpu) {
                (Some(gpu_flip), Some(gpu)) => (gpu_flip, gpu),
                _ => return,
            };
            gpu_flip.step_no_readback(
                &gpu.device,
                &gpu.queue,
                &mut self.flip_positions,
                &mut self.flip_velocities,
                &mut self.flip_affine,
                &self.flip_densities,
                &self.flip_cell_types,
                Some(sdf),
                None,
                dt_sub,
                FLIP_GRAVITY,
                FLIP_FLOW_ACCEL,
                FLIP_PRESSURE_ITERS,
            );
            self.flip_gpu_needs_upload = false;
            for _ in 1..FLIP_SUBSTEPS {
                gpu_flip.step_in_place(
                    &gpu.device,
                    &gpu.queue,
                    particle_count as u32,
                    &self.flip_cell_types,
                    Some(sdf),
                    None,
                    dt_sub,
                    FLIP_GRAVITY,
                    FLIP_FLOW_ACCEL,
                    FLIP_PRESSURE_ITERS,
                );
            }
        } else {
            let (gpu_flip, gpu) = match (&mut self.gpu_flip, &self.gpu) {
                (Some(gpu_flip), Some(gpu)) => (gpu_flip, gpu),
                _ => return,
            };
            for _ in 0..FLIP_SUBSTEPS {
                gpu_flip.step_in_place(
                    &gpu.device,
                    &gpu.queue,
                    particle_count as u32,
                    &self.flip_cell_types,
                    Some(sdf),
                    None,
                    dt_sub,
                    FLIP_GRAVITY,
                    FLIP_FLOW_ACCEL,
                    FLIP_PRESSURE_ITERS,
                );
            }
        }

        let next_substep = self.flip_gpu_sync_substep.saturating_add(1);
        let schedule_readback = next_substep >= FLIP_GPU_SYNC_STRIDE;
        if schedule_readback {
            self.ensure_flip_readback_buffers_len(particle_count);
            let requested = {
                let (gpu_flip, gpu) = match (&mut self.gpu_flip, &self.gpu) {
                    (Some(gpu_flip), Some(gpu)) => (gpu_flip, gpu),
                    _ => return,
                };
                gpu_flip.request_readback(&gpu.device, &gpu.queue, particle_count)
            };
            if requested {
                self.flip_gpu_readback_pending = true;
                self.flip_gpu_sync_substep = 0;
            } else {
                self.flip_gpu_sync_substep = next_substep;
            }
        } else {
            self.flip_gpu_sync_substep = next_substep;
        }

        self.frame = self.frame.wrapping_add(1);
    }

    fn shake_accel(&self) -> Vec3 {
        let omega = std::f32::consts::TAU * SHAKE_FREQ;
        let phase = (omega * self.time).sin();
        let deck_normal = self.deck.normal();
        let deck_tangent = self.deck.tangent();
        let normal_accel = deck_normal * (SHAKE_AMP * omega * omega * phase);
        let tangent_accel = deck_tangent * (SHAKE_TANGENT * phase.cos());
        normal_accel + tangent_accel
    }

    fn apply_shake(&mut self, accel: Vec3, dt: f32) {
        for clump in &mut self.sim.clumps {
            clump.velocity += accel * dt;
        }
    }

    fn apply_water_drag(&mut self, dt: f32) {
        let deck_tangent = self.deck.tangent();
        for (idx, clump) in self.sim.clumps.iter_mut().enumerate() {
            if matches!(self.materials[idx], MaterialTag::Rock) {
                continue;
            }
            let drag = (1.0 - WATER_DRAG * dt).max(0.0);
            clump.velocity *= drag;
            clump.velocity += deck_tangent * WATER_FLOW * dt;
        }
    }

    fn apply_deck_constraints(&mut self, dt: f32) {
        let deck = &self.deck;
        let normal = deck.normal();
        let friction = 0.55;
        for (idx, clump) in self.sim.clumps.iter_mut().enumerate() {
            let template = &self.sim.templates[clump.template_idx];
            let mut max_penetration = 0.0f32;
            let mut blocked = false;

            for offset in &template.local_offsets {
                let local = clump.rotation * *offset;
                let pos = clump.position + local;
                if !deck.inside_bounds(pos) {
                    continue;
                }
                let plane_y = deck.height_at(pos.x);
                let penetration = plane_y + template.particle_radius - pos.y;
                if penetration > 0.0 {
                    let in_hole = deck.in_hole(pos);
                    let fine = matches!(self.materials[idx], MaterialTag::Gold);
                    if in_hole && fine && template.particle_radius * 2.0 <= APERTURE_DIAMETER {
                        continue;
                    }
                    blocked = true;
                    if penetration > max_penetration {
                        max_penetration = penetration;
                    }
                }
            }

            if blocked && max_penetration > 0.0 {
                clump.position += normal * max_penetration;
                let v_n = clump.velocity.dot(normal);
                if v_n < 0.0 {
                    let v_t = clump.velocity - normal * v_n;
                    let v_t = v_t * (1.0 - friction * dt * 8.0).max(0.0);
                    clump.velocity = v_t;
                }
            }
        }
    }

    fn update_counts(&mut self) {
        let deck = &self.deck;
        let underflow_y = deck.origin.y - 0.25;
        let overflow_x = deck.origin.x + deck.length + 0.4;

        for (idx, clump) in self.sim.clumps.iter().enumerate() {
            if self.counted[idx] {
                continue;
            }
            let pos = clump.position;
            if pos.y < underflow_y {
                match self.materials[idx] {
                    MaterialTag::Gold => self.gold_underflow += 1,
                    MaterialTag::Rock => {}
                }
                self.counted[idx] = true;
            } else if pos.x > overflow_x {
                match self.materials[idx] {
                    MaterialTag::Gold => self.gold_overflow += 1,
                    MaterialTag::Rock => {}
                }
                self.counted[idx] = true;
            }
        }
    }

    fn emit_spray(&mut self) {
        if self.flip.particles.len() >= FLIP_MAX_PARTICLES {
            return;
        }

        let deck = &self.deck;
        let tangent = deck.tangent();
        let base_speed = tangent * SPRAY_TANGENT_SPEED + Vec3::Y * SPRAY_DOWN_SPEED;

        let span = deck.length * 0.7;
        let start_x = deck.origin.x + deck.length * 0.15;
        let spacing = if SPRAY_BAR_COUNT > 0 {
            span / SPRAY_BAR_COUNT as f32
        } else {
            0.0
        };

        for bar in 0..SPRAY_BAR_COUNT {
            let bar_x = start_x + (bar as f32 + 0.5) * spacing;
            let base_y = deck.height_at(bar_x) + SPRAY_BAR_HEIGHT;
            for i in 0..SPRAY_PARTICLES_PER_BAR {
                if self.flip.particles.len() >= FLIP_MAX_PARTICLES {
                    return;
                }
                let t = (i as f32 + 0.5) / SPRAY_PARTICLES_PER_BAR as f32;
                let mut z = deck.origin.z + t * deck.width;
                z += (rand_float() - 0.5) * SPRAY_SPREAD_Z;
                let y = base_y + (rand_float() - 0.5) * 0.05;
                let x = bar_x + (rand_float() - 0.5) * 0.06;

                let density = sediment_density(rand_float());
                self.flip
                    .spawn_sediment(Vec3::new(x, y, z), base_speed, density);
                self.flip_gpu_needs_upload = true;
            }
        }
    }

    fn prepare_flip_inputs(&mut self) {
        self.flip_positions.clear();
        self.flip_velocities.clear();
        self.flip_affine.clear();
        self.flip_densities.clear();

        for p in &self.flip.particles.list {
            self.flip_positions.push(p.position);
            self.flip_velocities.push(p.velocity);
            self.flip_affine.push(p.affine_velocity);
            self.flip_densities.push(p.density);
        }

        let grid_len = self.flip.grid.width * self.flip.grid.height * self.flip.grid.depth;
        self.flip_cell_types.clear();
        self.flip_cell_types.resize(grid_len, 0);

        for (idx, &sdf_val) in self.flip.grid.sdf.iter().enumerate() {
            if sdf_val < 0.0 {
                self.flip_cell_types[idx] = 2;
            }
        }

        for pos in &self.flip_positions {
            let i = (pos.x / FLIP_CELL_SIZE) as i32;
            let j = (pos.y / FLIP_CELL_SIZE) as i32;
            let k = (pos.z / FLIP_CELL_SIZE) as i32;
            if i >= 0
                && i < self.flip.grid.width as i32
                && j >= 0
                && j < self.flip.grid.height as i32
                && k >= 0
                && k < self.flip.grid.depth as i32
            {
                let idx = k as usize * self.flip.grid.width * self.flip.grid.height
                    + j as usize * self.flip.grid.width
                    + i as usize;
                if self.flip_cell_types[idx] != 2 {
                    self.flip_cell_types[idx] = 1;
                }
            }
        }
    }

    fn ensure_flip_readback_buffers_len(&mut self, particle_count: usize) {
        if self.flip_positions.len() < particle_count {
            self.flip_positions.resize(particle_count, Vec3::ZERO);
        }
        if self.flip_velocities.len() < particle_count {
            self.flip_velocities.resize(particle_count, Vec3::ZERO);
        }
        if self.flip_affine.len() < particle_count {
            self.flip_affine.resize(particle_count, Mat3::ZERO);
        }
    }

    fn apply_flip_readback(&mut self, count: usize) {
        let limit = count.min(self.flip.particles.list.len());
        for (i, p) in self.flip.particles.list.iter_mut().enumerate().take(limit) {
            if i < self.flip_positions.len() {
                p.position = self.flip_positions[i];
            }
            if i < self.flip_velocities.len() {
                p.velocity = self.flip_velocities[i];
            }
            if i < self.flip_affine.len() {
                p.affine_velocity = self.flip_affine[i];
            }
        }
    }

    fn rebuild_instances(&mut self) {
        self.instances.clear();
        self.round_instances.clear();
        self.sharp_instances.clear();

        for (_idx, clump) in self.sim.clumps.iter().enumerate() {
            let template = &self.sim.templates[clump.template_idx];
            let color = self.template_colors[clump.template_idx];
            let instance = ClusterInstance {
                position: clump.position.to_array(),
                scale: template.bounding_radius * CLUSTER_VISUAL_SCALE,
                rotation: clump.rotation.to_array(),
                color,
            };

            match template.shape {
                ClumpShape3D::Irregular { style, .. } => match style {
                    IrregularStyle3D::Round => self.round_instances.push(instance),
                    IrregularStyle3D::Sharp => self.sharp_instances.push(instance),
                },
                _ => self.round_instances.push(instance),
            }

        }

        self.instances.clear();
        self.instances.extend_from_slice(&self.round_instances);
        self.instances.extend_from_slice(&self.sharp_instances);
    }

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
            .request_device(
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
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut fluid_renderer = ScreenSpaceFluidRenderer::new(&device, config.format);
        fluid_renderer.particle_radius = FLIP_CELL_SIZE * 0.6;
        fluid_renderer.resize(&device, config.width, config.height);
        self.fluid_renderer = Some(fluid_renderer);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cluster Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let plane_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plane Shader"),
            source: wgpu::ShaderSource::Wgsl(PLANE_SHADER.into()),
        });

        let plane_color = [0.16, 0.18, 0.2, 0.85];
        let deck = &self.deck;
        let x0 = deck.origin.x;
        let x1 = deck.origin.x + deck.length;
        let z0 = deck.origin.z;
        let z1 = deck.origin.z + deck.width;
        let y0 = deck.height_at(x0);
        let y1 = deck.height_at(x1);

        let plane_vertices = [
            PlaneVertex {
                position: [x0, y0, z0],
                color: plane_color,
            },
            PlaneVertex {
                position: [x1, y1, z0],
                color: plane_color,
            },
            PlaneVertex {
                position: [x1, y1, z1],
                color: plane_color,
            },
            PlaneVertex {
                position: [x0, y0, z0],
                color: plane_color,
            },
            PlaneVertex {
                position: [x1, y1, z1],
                color: plane_color,
            },
            PlaneVertex {
                position: [x0, y0, z1],
                color: plane_color,
            },
        ];
        let plane_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Vertex Buffer"),
            contents: bytemuck::cast_slice(&plane_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
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

        let plane_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Plane Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &plane_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PlaneVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &plane_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            cache: None,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cluster Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
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
                        array_stride: std::mem::size_of::<ClusterInstance>() as u64,
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
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            cache: None,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let round_mesh = build_round_mesh();
        let sharp_mesh = build_sharp_mesh();

        let round_mesh_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Round Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&round_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sharp_mesh_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sharp Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&sharp_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<ClusterInstance>() * MAX_INSTANCES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (depth_texture, depth_view) = create_depth_texture(&device, config.width, config.height);

        let mut gpu_flip = GpuFlip3D::new(
            &device,
            self.flip.grid.width as u32,
            self.flip.grid.height as u32,
            self.flip.grid.depth as u32,
            FLIP_CELL_SIZE,
            FLIP_MAX_PARTICLES,
        );
        gpu_flip.vorticity_epsilon = 0.0;
        gpu_flip.sediment_rest_particles = 0.0;
        gpu_flip.sediment_porosity_drag = 0.0;
        gpu_flip.sediment_drag_coefficient = 3.0;
        gpu_flip.sediment_settling_velocity = 0.3;
        gpu_flip.sediment_friction_threshold = 0.04;
        gpu_flip.sediment_friction_strength = 0.1;
        gpu_flip.gold_density_threshold = 10.0;
        gpu_flip.gold_drag_multiplier = 3.0;
        gpu_flip.gold_settling_velocity = 0.1;
        gpu_flip.gold_flake_lift = 0.4;
        self.gpu_flip = Some(gpu_flip);

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            plane_pipeline,
            pipeline,
            plane_vertex_buffer,
            plane_vertex_count: plane_vertices.len() as u32,
            round_mesh_vertex_buffer,
            round_mesh_vertex_count: round_mesh.len() as u32,
            sharp_mesh_vertex_buffer,
            sharp_mesh_vertex_count: sharp_mesh.len() as u32,
            instance_buffer,
            uniform_buffer,
            bind_group,
            depth_texture,
            depth_view,
        });
    }

    fn update_gpu(&mut self) {
        let view_proj = self.camera_matrix();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [
                self.camera_distance * self.camera_angle.cos(),
                self.camera_height,
                self.camera_distance * self.camera_angle.sin(),
            ],
            base_scale: 1.0,
        };
        let gpu = self.gpu.as_mut().unwrap();
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let total_instances = self.instances.len().min(MAX_INSTANCES);
        let instance_data = &self.instances[..total_instances];
        gpu.queue.write_buffer(
            &gpu.instance_buffer,
            0,
            bytemuck::cast_slice(instance_data),
        );
    }

    fn camera_matrices(&self) -> (Mat4, Mat4) {
        let eye = Vec3::new(
            self.camera_distance * self.camera_angle.cos(),
            self.camera_height,
            self.camera_distance * self.camera_angle.sin(),
        );
        let target = Vec3::new(
            self.deck.origin.x + self.deck.length * 0.4,
            self.deck.origin.y + 0.2,
            self.deck.origin.z + self.deck.width * 0.5,
        );
        let up = Vec3::Y;
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh_gl(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        (view, proj)
    }

    fn camera_matrix(&self) -> Mat4 {
        let (view, proj) = self.camera_matrices();
        proj * view
    }

    fn render(&mut self) {
        let (view_matrix, proj_matrix) = self.camera_matrices();
        let eye = Vec3::new(
            self.camera_distance * self.camera_angle.cos(),
            self.camera_height,
            self.camera_distance * self.camera_angle.sin(),
        );
        let gpu = self.gpu.as_mut().unwrap();
        let frame = match gpu.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                gpu.surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let clear_color = wgpu::Color {
            r: 0.02,
            g: 0.02,
            b: 0.03,
            a: 1.0,
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
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
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            drop(pass);
        }

        if let (Some(gpu_flip), Some(fluid_renderer)) = (&self.gpu_flip, &self.fluid_renderer) {
            let active_count = self.flip.particles.list.len() as u32;
            if active_count > 0 {
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

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
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
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&gpu.plane_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.plane_vertex_buffer.slice(..));
            pass.draw(0..gpu.plane_vertex_count, 0..1);

            let total_round = self.round_instances.len().min(MAX_INSTANCES);
            if total_round > 0 {
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.round_mesh_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                pass.draw(0..gpu.round_mesh_vertex_count, 0..total_round as u32);
            }

            let total_sharp = self.sharp_instances.len().min(MAX_INSTANCES - total_round);
            if total_sharp > 0 {
                let instance_offset = (total_round * std::mem::size_of::<ClusterInstance>()) as u64;
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.sharp_mesh_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(instance_offset..));
                pass.draw(0..gpu.sharp_mesh_vertex_count, 0..total_sharp as u32);
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.window = Some(window.clone());
        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    let (depth_texture, depth_view) =
                        create_depth_texture(&gpu.device, size.width, size.height);
                    gpu.depth_texture = depth_texture;
                    gpu.depth_view = depth_view;
                }
                if let (Some(gpu), Some(fluid_renderer)) = (&self.gpu, &mut self.fluid_renderer) {
                    fluid_renderer.resize(&gpu.device, gpu.config.width, gpu.config.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == winit::event::ElementState::Pressed {
                        match code {
                            KeyCode::Space => self.paused = !self.paused,
                            KeyCode::KeyR => self.reset_sim(),
                            KeyCode::ArrowLeft => self.camera_angle -= 0.1,
                            KeyCode::ArrowRight => self.camera_angle += 0.1,
                            KeyCode::ArrowUp => self.camera_distance = (self.camera_distance - 0.4).max(2.0),
                            KeyCode::ArrowDown => self.camera_distance += 0.4,
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_flip(FLIP_DT);
                self.update_sim(DT);
                self.rebuild_instances();
                self.update_gpu();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn camera_defaults(sim: &ClusterSimulation3D) -> (f32, f32) {
    let extent = sim.bounds_max - sim.bounds_min;
    let span = extent.x.max(extent.z);
    let distance = (span * 0.9 + 2.0).max(6.0);
    let height = (extent.y * 0.7 + 1.0).max(2.5);
    (distance, height)
}

fn build_sim() -> (
    ClusterSimulation3D,
    Vec<[f32; 4]>,
    Vec<MaterialTag>,
    DeckSpec,
) {
    let slope = -DECK_ANGLE_DEG.to_radians().tan();
    let deck_origin = Vec3::new(0.4, DECK_HEIGHT, 0.4);
    let deck = DeckSpec {
        origin: deck_origin,
        length: DECK_LENGTH,
        width: DECK_WIDTH,
        slope,
        hole_spacing: HOLE_SPACING,
        hole_radius: APERTURE_DIAMETER * 0.5,
    };

    let bounds_min = Vec3::new(0.0, 0.0, 0.0);
    let bounds_max = Vec3::new(
        deck_origin.x + DECK_LENGTH + 2.0,
        DECK_HEIGHT + 3.0,
        deck_origin.z + DECK_WIDTH + 1.0,
    );

    let mut sim = ClusterSimulation3D::new(bounds_min, bounds_max);
    sim.use_dem = true;
    sim.restitution = 0.15;
    sim.friction = 0.7;
    sim.floor_friction = 0.9;
    sim.normal_stiffness = 40_000.0;
    sim.tangential_stiffness = 18_000.0;
    sim.rolling_friction = 0.05;

    let mut template_colors = Vec::new();
    let mut materials = Vec::new();

    let mut gravel_templates = Vec::new();
    for (idx, radius) in GRAVEL_RADII.iter().enumerate() {
        let mass = density_mass(GRAVEL_DENSITY, *radius);
        let template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 7,
                seed: 10u64 + idx as u64,
                style: IrregularStyle3D::Sharp,
            },
            *radius,
            mass,
        );
        let template_idx = sim.add_template(template);
        gravel_templates.push((template_idx, mass));
        template_colors.push([0.35, 0.34, 0.32, 0.95]);
    }

    let mut gold_templates = Vec::new();
    for (_idx, radius) in GOLD_NUGGET_RADII.iter().enumerate() {
        let mass = density_mass(GOLD_DENSITY, *radius);
        let template = ClumpTemplate3D::generate(ClumpShape3D::Flat4, *radius, mass);
        let template_idx = sim.add_template(template);
        gold_templates.push((template_idx, mass));
        template_colors.push([0.95, 0.78, 0.2, 0.95]);
    }

    let gravel_mass_denom: f32 = GRAVEL_BIN_MASS
        .iter()
        .zip(gravel_templates.iter())
        .map(|(mass_frac, (_, mass))| mass_frac / mass)
        .sum();
    let gravel_total_mass = TARGET_GRAVEL_COUNT as f32 / gravel_mass_denom;
    let mut gravel_counts: Vec<usize> = GRAVEL_BIN_MASS
        .iter()
        .zip(gravel_templates.iter())
        .map(|(mass_frac, (_, mass))| (gravel_total_mass * mass_frac / mass).round() as usize)
        .collect();

    let count_sum: usize = gravel_counts.iter().sum();
    if count_sum < TARGET_GRAVEL_COUNT {
        let mut remaining = TARGET_GRAVEL_COUNT - count_sum;
        let mut i = 0;
        let bin_count = gravel_counts.len();
        while remaining > 0 {
            gravel_counts[i % bin_count] += 1;
            remaining -= 1;
            i += 1;
        }
    } else if count_sum > TARGET_GRAVEL_COUNT {
        let mut excess = count_sum - TARGET_GRAVEL_COUNT;
        let mut i = 0;
        let bin_count = gravel_counts.len();
        while excess > 0 {
            let idx = i % bin_count;
            if gravel_counts[idx] > 0 {
                gravel_counts[idx] -= 1;
                excess -= 1;
            }
            i += 1;
        }
    }

    let bulk_mass = gravel_total_mass / BULK_GRAVEL_FRACTION;
    let _flip_mass = bulk_mass
        * (BULK_SAND_FRACTION
            + BULK_SILT_FRACTION
            + BULK_HEAVY_FRACTION
            + BULK_GOLD_FRACTION * (1.0 - GOLD_NUGGET_SHARE));
    let _flip_heavy_density = HEAVY_MINERAL_DENSITY;
    let gold_nugget_mass = bulk_mass * BULK_GOLD_FRACTION * GOLD_NUGGET_SHARE;
    let gold_counts: Vec<usize> = GOLD_NUGGET_MASS
        .iter()
        .zip(gold_templates.iter())
        .map(|(mass_frac, (_, mass))| (gold_nugget_mass * mass_frac / mass).round() as usize)
        .collect();

    let mut spawn_id = 0u32;
    let mut spawn_clump = |template_idx: usize, material: MaterialTag| {
        let seed = spawn_id;
        spawn_id += 1;
        let x = deck_origin.x
            + 0.6
            + hash_range(0xA53A_91C3 ^ seed.wrapping_mul(31), 0.0, DECK_LENGTH * 0.4);
        let z = deck_origin.z
            + 0.4
            + hash_range(0xC1B2_9ED1 ^ seed.wrapping_mul(27), 0.0, DECK_WIDTH * 0.8);
        let y = deck.height_at(x)
            + 0.3
            + hash_range(0x9C5B_D781 ^ seed.wrapping_mul(19), 0.0, 0.2);
        sim.spawn(template_idx, Vec3::new(x, y, z), Vec3::ZERO);
        materials.push(material);
    };

    for ((template_idx, _), count) in gravel_templates.iter().zip(gravel_counts.iter()) {
        for _ in 0..*count {
            spawn_clump(*template_idx, MaterialTag::Rock);
        }
    }

    for ((template_idx, _), count) in gold_templates.iter().zip(gold_counts.iter()) {
        for _ in 0..*count {
            spawn_clump(*template_idx, MaterialTag::Gold);
        }
    }

    (sim, template_colors, materials, deck)
}

fn build_flip(deck: &DeckSpec) -> FlipSimulation3D {
    let x_max = deck.origin.x + deck.length + 0.8;
    let z_max = deck.origin.z + deck.width + 0.6;
    let y_max = deck.origin.y + 1.2;
    let width = (x_max / FLIP_CELL_SIZE).ceil() as usize;
    let height = (y_max / FLIP_CELL_SIZE).ceil() as usize;
    let depth = (z_max / FLIP_CELL_SIZE).ceil() as usize;

    let mut sim = FlipSimulation3D::new(width, height, depth, FLIP_CELL_SIZE);
    sim.pressure_iterations = FLIP_PRESSURE_ITERS as usize;

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                let x = (i as f32 + 0.5) * FLIP_CELL_SIZE;
                let y = (j as f32 + 0.5) * FLIP_CELL_SIZE;
                let z = (k as f32 + 0.5) * FLIP_CELL_SIZE;
                if x < deck.origin.x
                    || x > deck.origin.x + deck.length
                    || z < deck.origin.z
                    || z > deck.origin.z + deck.width
                {
                    continue;
                }
                if y < deck.height_at(x) {
                    let idx = sim.grid.cell_index(i, j, k);
                    sim.grid.solid[idx] = true;
                }
            }
        }
    }
    sim.grid.compute_sdf();
    sim
}

fn density_mass(density: f32, radius: f32) -> f32 {
    density * (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3)
}

fn hash_to_unit(seed: u32) -> f32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

fn hash_range(seed: u32, min: f32, max: f32) -> f32 {
    min + (max - min) * hash_to_unit(seed)
}

fn sediment_density(roll: f32) -> f32 {
    if roll < SEDIMENT_GOLD_SHARE {
        GOLD_DENSITY
    } else if roll < SEDIMENT_GOLD_SHARE + SEDIMENT_HEAVY_SHARE {
        HEAVY_MINERAL_DENSITY
    } else if roll < SEDIMENT_GOLD_SHARE + SEDIMENT_HEAVY_SHARE + SEDIMENT_SAND_SHARE {
        GRAVEL_DENSITY
    } else if roll
        < SEDIMENT_GOLD_SHARE + SEDIMENT_HEAVY_SHARE + SEDIMENT_SAND_SHARE + SEDIMENT_SILT_SHARE
    {
        GRAVEL_DENSITY
    } else {
        GRAVEL_DENSITY
    }
}

fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

fn build_round_mesh() -> Vec<MeshVertex> {
    let mut vertices = Vec::new();
    let steps = 16;
    for i in 0..steps {
        let theta0 = (i as f32) * std::f32::consts::TAU / steps as f32;
        let theta1 = ((i + 1) as f32) * std::f32::consts::TAU / steps as f32;
        for j in 0..steps {
            let phi0 = (j as f32) * std::f32::consts::PI / steps as f32;
            let phi1 = ((j + 1) as f32) * std::f32::consts::PI / steps as f32;

            let p00 = spherical_point(theta0, phi0);
            let p10 = spherical_point(theta1, phi0);
            let p01 = spherical_point(theta0, phi1);
            let p11 = spherical_point(theta1, phi1);

            vertices.push(MeshVertex {
                position: p00.to_array(),
                normal: p00.to_array(),
            });
            vertices.push(MeshVertex {
                position: p10.to_array(),
                normal: p10.to_array(),
            });
            vertices.push(MeshVertex {
                position: p11.to_array(),
                normal: p11.to_array(),
            });

            vertices.push(MeshVertex {
                position: p00.to_array(),
                normal: p00.to_array(),
            });
            vertices.push(MeshVertex {
                position: p11.to_array(),
                normal: p11.to_array(),
            });
            vertices.push(MeshVertex {
                position: p01.to_array(),
                normal: p01.to_array(),
            });
        }
    }
    vertices
}

fn build_sharp_mesh() -> Vec<MeshVertex> {
    let points = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.9428, -0.3333),
        Vec3::new(-0.8165, -0.4714, -0.3333),
        Vec3::new(0.8165, -0.4714, -0.3333),
    ];
    let faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)];
    let mut vertices = Vec::new();
    for (a, b, c) in faces {
        let pa = points[a];
        let pb = points[b];
        let pc = points[c];
        let normal = (pb - pa).cross(pc - pa).normalize();
        vertices.push(MeshVertex {
            position: pa.to_array(),
            normal: normal.to_array(),
        });
        vertices.push(MeshVertex {
            position: pb.to_array(),
            normal: normal.to_array(),
        });
        vertices.push(MeshVertex {
            position: pc.to_array(),
            normal: normal.to_array(),
        });
    }
    vertices
}

fn spherical_point(theta: f32, phi: f32) -> Vec3 {
    let sin_phi = phi.sin();
    Vec3::new(theta.cos() * sin_phi, phi.cos(), theta.sin() * sin_phi)
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width,
        height,
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

const PLANE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    base_scale: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.color = input.color;
    out.world_pos = input.position;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let hole_spacing = 0.18;
    let hole_radius = 0.03;
    let grid = round(input.world_pos.xz / hole_spacing) * hole_spacing;
    let delta = input.world_pos.xz - grid;
    if (dot(delta, delta) < hole_radius * hole_radius) {
        discard;
    }
    return input.color;
}
"#;

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    base_scale: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_pos: vec3<f32>,
    @location(3) instance_scale: f32,
    @location(4) instance_rot: vec4<f32>,
    @location(5) instance_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let scaled = input.position * input.instance_scale * uniforms.base_scale;
    let rotated = quat_rotate(input.instance_rot, scaled);
    let world_pos = rotated + input.instance_pos;
    out.world_pos = world_pos;
    out.clip_pos = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.normal = normalize(quat_rotate(input.instance_rot, input.normal));
    out.color = input.instance_color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let ndotl = max(dot(input.normal, light_dir), 0.1);
    return vec4<f32>(input.color.rgb * ndotl, input.color.a);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
