//! Industrial Scale Sluice - Large 3D GPU FLIP Test
//!
//! Much larger grid, longer sluice, shallower gradient, more water.
//! Target: 50k-100k+ particles at playable framerates.
//!
//! Run with: cargo run --example industrial_sluice --release

use bytemuck::{Pod, Zeroable};
// Disabled: bed_3d was the Drucker-Prager + heightfield system with threshold issues
// use game::gpu::bed_3d::{self, GpuBed3D, GpuBedParams};
use game::gpu::flip_3d::GpuFlip3D;
// Disabled: DruckerPragerParams replaced by simple friction in g2p_3d
// use game::gpu::g2p_3d::DruckerPragerParams;
use glam::{Mat3, Mat4, Vec3, Vec4};
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
const RIFFLE_THICKNESS_CELLS: i32 = 2;
const SEDIMENT_REL_DENSITY: f32 = 2.65;
const DEFAULT_WATER_EMIT_RATE: usize = 150;
const DEFAULT_SEDIMENT_EMIT_RATE: usize = 50;
const EMIT_RATE_STEP: usize = 25;
const MAX_EMIT_RATE: usize = 1000;
const CLICK_SPAWN_WATER: usize = 120;
const CLICK_SPAWN_SEDIMENT: usize = 150;
const CLICK_SPAWN_RADIUS: f32 = 2.5 * CELL_SIZE;
const CLICK_WATER_LIFT: f32 = 3.0 * CELL_SIZE;
const CLICK_SEDIMENT_LIFT: f32 = 1.5 * CELL_SIZE;
const DP_FRICTION_ANGLE_DEFAULT_DEG: f32 = 32.0;
const DP_COHESION_DEFAULT: f32 = 0.0;
const DP_VISCOSITY_DEFAULT: f32 = 1.0;
const DP_JAMMED_DRAG_DEFAULT: f32 = 50.0;
const DP_BUOYANCY_FACTOR: f32 = 1.0 - 1.0 / SEDIMENT_REL_DENSITY;
const DP_MIN_PRESSURE: f32 = 0.1;
const DP_YIELD_SMOOTHING: f32 = 0.1;
const DP_FRICTION_STEP_DEG: f32 = 1.0;
const DP_COHESION_STEP: f32 = 5.0;
const DP_VISCOSITY_STEP: f32 = 0.1;
const DP_JAMMED_DRAG_STEP: f32 = 5.0;
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
const WALL_MARGIN: usize = 4;  // Wall height above floor+riffle
const BED_AIR_MARGIN_CELLS: f32 = 1.5;
const BED_MAX_SLOPE: f32 = 0.7;
const BED_RELAX_ITERS: usize = 2;
const BED_MAX_DELTA_LAYERS: f32 = 1.0;

// Screen-space fluid rendering constants
const WATER_PARTICLE_RADIUS: f32 = 0.025;  // Larger for better depth overlap
const SSFR_BLUR_RADIUS: i32 = 15;          // Bigger blur for smoother surface
const SSFR_BLUR_DEPTH_FALLOFF: f32 = 0.2;  // More permissive depth falloff
const SSFR_NEAR: f32 = 0.01;
const SSFR_FAR: f32 = 100.0;

// Hybrid water rendering: heightfield mesh + splats for overtopping
const WATER_MESH_SUBDIVISIONS: usize = 2;  // 2x2 = 4 sub-cells per grid cell (faster)
const MAX_WATER_SURFACE_VERTICES: usize = GRID_WIDTH * GRID_DEPTH * 6 * (WATER_MESH_SUBDIVISIONS * WATER_MESH_SUBDIVISIONS * 2);  // Surface + sides only
const WATER_OVERTOP_THRESHOLD: f32 = 0.02;  // Particles must be this much above heightfield to be splats

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
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SurfaceVertex {
    position: [f32; 3],
    color: [f32; 4],
}

/// Indexed mesh for efficient 3D rendering
struct BedMesh {
    vertices: Vec<SurfaceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl BedMesh {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bed Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));

        self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bed Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        }));
    }

    fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        // Recreate buffers if size changed (simple approach)
        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        // Check if buffers are large enough
        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            // Recreate larger buffers
            self.upload(device);
        }
    }

    fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }
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
struct WaterUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    particle_radius: f32,
    screen_size: [f32; 2],
    near: f32,
    far: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurUniforms {
    texel_size: [f32; 2],
    blur_radius: i32,
    depth_falloff: f32,
    direction: [f32; 2],
    _pad: [f32; 2],
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    gpu_flip: Option<GpuFlip3D>,
    // Disabled: bed_3d system removed
    // gpu_bed: Option<GpuBed3D>,
    sim: FlipSimulation3D,
    paused: bool,
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    frame: u32,
    solid_instances: Vec<ParticleInstance>,
    bed_mesh: BedMesh,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    bed_height: Vec<f32>,
    bed_base_height: Vec<f32>,
    bed_height_prev: Vec<f32>,
    bed_height_residual: Vec<f32>,
    bed_water_vel_sum: Vec<Vec3>,
    bed_water_count: Vec<u32>,
    bed_sediment_count: Vec<u32>,
    bed_flux_x: Vec<f32>,
    bed_flux_z: Vec<f32>,
    cell_types: Vec<u32>,
    use_gpu_sim: bool,
    pressure_iters_gpu: u32,
    vorticity_epsilon: f32,
    dp_friction_angle_deg: f32,
    dp_cohesion: f32,
    dp_viscosity: f32,
    dp_jammed_drag: f32,
    use_async_readback: bool,
    gpu_readback_pending: bool,
    render_heightfield: bool,
    render_flow_particles: bool,
    debug_riffle_probe: bool,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    cursor_pos: Option<(f32, f32)>,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,
    emitter_enabled: bool,
    water_emitter_enabled: bool,
    sediment_emitter_enabled: bool,
    water_emit_rate: usize,
    sediment_emit_rate: usize,
    particles_exited: u32,
    pending_emit_water: usize,
    pending_emit_sediment: usize,
    click_add_sediment: bool,
    gpu_particle_count: u32,
    // Disabled: bed_3d probe stats removed
    // gpu_probe_stats: Vec<i32>,
    heightfield: Vec<f32>,
    surface_vertices: Vec<SurfaceVertex>,
    flow_particles: Vec<ParticleInstance>,
    // Water heightfield for hybrid rendering
    water_heightfield: Vec<f32>,
    water_heightfield_smoothed: Vec<f32>,  // Temporally smoothed heightfield
    water_presence: Vec<f32>,  // Smooth presence [0,1] for edge fading
    water_surface_vertices: Vec<SurfaceVertex>,
    // Time for wave animation
    simulation_time: f32,
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

/// Screen-Space Fluid Rendering resources
struct WaterSSFR {
    // Textures
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    smoothed_depth_texture: wgpu::Texture,
    smoothed_depth_view: wgpu::TextureView,
    temp_blur_texture: wgpu::Texture,  // For separable blur
    temp_blur_view: wgpu::TextureView,
    thickness_texture: wgpu::Texture,
    thickness_view: wgpu::TextureView,
    // Scene depth for compositing
    scene_depth_texture: wgpu::Texture,
    scene_depth_view: wgpu::TextureView,
    // Samplers
    sampler: wgpu::Sampler,
    // Uniforms
    water_uniform_buffer: wgpu::Buffer,
    blur_uniform_buffer: wgpu::Buffer,
    // Depth pass
    depth_pipeline: wgpu::RenderPipeline,
    depth_bind_group: wgpu::BindGroup,
    // Blur passes
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    blur_h_bind_group: wgpu::BindGroup,
    blur_v_bind_group: wgpu::BindGroup,
    // Composite pass
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group: wgpu::BindGroup,
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
    water_surface_buffer: wgpu::Buffer,  // For hybrid water heightfield mesh
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    surface_pipeline: wgpu::RenderPipeline,
    // Screen-space fluid rendering
    water_ssfr: Option<WaterSSFR>,
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

                // Wall height: just above floor + riffle + margin (open-top trough)
                let wall_top = floor_j + riffle_height + WALL_MARGIN;

                let is_boundary =
                    j <= floor_j ||                                       // Sloped floor
                    is_riffle ||                                          // Riffles on floor
                    (i == 0 && j <= wall_top) ||                          // Left wall (short)
                    (i == width - 1 && !is_exit && j <= wall_top) ||      // Right wall (short, except exit)
                    ((k == 0 || k == depth - 1) && j <= wall_top);        // Z walls (short)

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

/// Create Screen-Space Fluid Rendering resources
fn create_water_ssfr(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    vertex_buffer: &wgpu::Buffer,
    instance_buffer: &wgpu::Buffer,
) -> WaterSSFR {
    let width = config.width;
    let height = config.height;

    // Create depth texture for water particles
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Smoothed depth (after blur)
    let smoothed_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Smoothed Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let smoothed_depth_view = smoothed_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Temp texture for separable blur
    let temp_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Temp Blur Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let temp_blur_view = temp_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Thickness texture
    let thickness_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Thickness Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let thickness_view = thickness_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Scene depth texture (for depth testing against scene geometry)
    let scene_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Scene Depth Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let scene_depth_view = scene_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Sampler
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Water Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // Uniform buffers
    let water_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Water Uniform Buffer"),
        size: std::mem::size_of::<WaterUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let blur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Blur Uniform Buffer"),
        size: std::mem::size_of::<BlurUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shaders
    let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Depth Shader"),
        source: wgpu::ShaderSource::Wgsl(WATER_DEPTH_SHADER.into()),
    });
    let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Blur Shader"),
        source: wgpu::ShaderSource::Wgsl(WATER_BLUR_SHADER.into()),
    });
    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Water Composite Shader"),
        source: wgpu::ShaderSource::Wgsl(WATER_COMPOSITE_SHADER.into()),
    });

    // Depth pass bind group layout and pipeline
    let depth_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Depth Bind Group Layout"),
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

    let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Depth Bind Group"),
        layout: &depth_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: water_uniform_buffer.as_entire_binding(),
        }],
    });

    let depth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Depth Pipeline Layout"),
        bind_group_layouts: &[&depth_bind_group_layout],
        push_constant_ranges: &[],
    });

    let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Depth Pipeline"),
        layout: Some(&depth_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &depth_shader,
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
            module: &depth_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
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

    // Blur pass bind group layout and pipelines
    let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Blur Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Horizontal blur: depth -> temp
    let blur_h_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Blur H Bind Group"),
        layout: &blur_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: blur_uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&temp_blur_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    // Vertical blur: temp -> smoothed
    let blur_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Blur V Bind Group"),
        layout: &blur_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: blur_uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&temp_blur_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&smoothed_depth_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Blur Pipeline Layout"),
        bind_group_layouts: &[&blur_bind_group_layout],
        push_constant_ranges: &[],
    });

    let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Water Blur H Pipeline"),
        layout: Some(&blur_pipeline_layout),
        module: &blur_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Water Blur V Pipeline"),
        layout: Some(&blur_pipeline_layout),
        module: &blur_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Composite pass bind group layout and pipeline
    let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water Composite Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Composite Bind Group"),
        layout: &composite_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: water_uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&smoothed_depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&thickness_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&scene_depth_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Composite Pipeline Layout"),
        bind_group_layouts: &[&composite_bind_group_layout],
        push_constant_ranges: &[],
    });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Water Composite Pipeline"),
        layout: Some(&composite_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &composite_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_shader,
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

    WaterSSFR {
        depth_texture,
        depth_view,
        smoothed_depth_texture,
        smoothed_depth_view,
        temp_blur_texture,
        temp_blur_view,
        thickness_texture,
        thickness_view,
        scene_depth_texture,
        scene_depth_view,
        sampler,
        water_uniform_buffer,
        blur_uniform_buffer,
        depth_pipeline,
        depth_bind_group,
        blur_h_pipeline,
        blur_v_pipeline,
        blur_h_bind_group,
        blur_v_bind_group,
        composite_pipeline,
        composite_bind_group,
    }
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

// Disabled: bed_3d system removed - using friction-only sediment model
// fn build_gpu_bed_params(dt: f32) -> GpuBedParams { ... }

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
        let bed_height_prev = bed_height.clone();
        let bed_height_residual = vec![0.0f32; bed_column_count];

        let solid_instances = Self::collect_solids(&sim);
        let bed_mesh = Self::build_bed_mesh(&sim);
        let pressure_iters_gpu = sim.pressure_iterations as u32;

        println!("Solid cells: {}", solid_instances.len());
        println!("Max particles: {}", MAX_PARTICLES);
        println!(
            "Controls: SPACE=pause, R=reset, G=toggle GPU/CPU, E=toggle emitter, W/S=water/sediment emit, Up/Down=water rate, Left/Right=sediment rate, M=click material, LMB=add burst, RMB=drag, C=clear water, X=clear sediment, Z=clear all, H=heightfield, F=flow particles, V/B=vorticity -/+, 1/2=friction, 3/4=cohesion, 5/6=viscosity, 7/8=jammed drag, P=riffle probe, ESC=quit"
        );

        Self {
            window: None,
            gpu: None,
            gpu_flip: None,
            // Disabled: gpu_bed removed
            // gpu_bed: None,
            sim,
            paused: false,
            camera_angle: 0.3,
            camera_pitch: 0.3,
            camera_distance: 8.0,  // Start further back for larger scene
            frame: 0,
            solid_instances,
            bed_mesh,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            bed_height,
            bed_base_height,
            bed_height_prev,
            bed_height_residual,
            bed_water_vel_sum,
            bed_water_count,
            bed_sediment_count,
            bed_flux_x,
            bed_flux_z,
            cell_types: Vec::new(),
            use_gpu_sim: true,
            pressure_iters_gpu,
            vorticity_epsilon: VORTICITY_EPSILON_DEFAULT,
            dp_friction_angle_deg: DP_FRICTION_ANGLE_DEFAULT_DEG,
            dp_cohesion: DP_COHESION_DEFAULT,
            dp_viscosity: DP_VISCOSITY_DEFAULT,
            dp_jammed_drag: DP_JAMMED_DRAG_DEFAULT,
            use_async_readback: false,
            gpu_readback_pending: false,
            render_heightfield: true,
            render_flow_particles: true,
            debug_riffle_probe: true,
            mouse_pressed: false,
            last_mouse_pos: None,
            cursor_pos: None,
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            emitter_enabled: true,
            water_emitter_enabled: true,
            sediment_emitter_enabled: true,
            water_emit_rate: DEFAULT_WATER_EMIT_RATE,
            sediment_emit_rate: DEFAULT_SEDIMENT_EMIT_RATE,
            particles_exited: 0,
            pending_emit_water: 0,
            pending_emit_sediment: 0,
            click_add_sediment: false,
            gpu_particle_count: 0,
            // Disabled: bed_3d probe stats removed
            // gpu_probe_stats: vec![0; bed_3d::PROBE_STAT_BUFFER_LEN],
            heightfield: vec![f32::NEG_INFINITY; GRID_WIDTH * GRID_DEPTH],
            surface_vertices: Vec::with_capacity(MAX_SURFACE_VERTICES),
            flow_particles: Vec::with_capacity(MAX_FLOW_PARTICLES),
            water_heightfield: vec![f32::NEG_INFINITY; GRID_WIDTH * GRID_DEPTH],
            water_heightfield_smoothed: vec![f32::NEG_INFINITY; GRID_WIDTH * GRID_DEPTH],
            water_presence: vec![0.0; GRID_WIDTH * GRID_DEPTH],
            water_surface_vertices: Vec::with_capacity(MAX_WATER_SURFACE_VERTICES),
            simulation_time: 0.0,
        }
    }

    /// Emit particles from inlet on left side - higher rate for industrial scale
    fn emit_particles(&mut self, water_count: usize, sediment_count: usize) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        let cell_size = CELL_SIZE;
        let mut remaining = MAX_PARTICLES - self.sim.particles.len();
        if remaining == 0 {
            return;
        }
        let water_spawn = water_count.min(remaining);
        remaining = remaining.saturating_sub(water_spawn);
        let sediment_spawn = sediment_count.min(remaining);

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

    fn bed_height_at(&self, x: f32, z: f32) -> f32 {
        let i = (x / CELL_SIZE).floor() as i32;
        let k = (z / CELL_SIZE).floor() as i32;
        if i >= 0 && i < GRID_WIDTH as i32 && k >= 0 && k < GRID_DEPTH as i32 {
            let idx = k as usize * GRID_WIDTH + i as usize;
            self.bed_height[idx]
        } else {
            0.0
        }
    }

    fn emit_particles_at(&mut self, center_x: f32, center_z: f32, water_count: usize, sediment_count: usize) {
        if self.sim.particles.len() >= MAX_PARTICLES {
            return;
        }

        let mut remaining = MAX_PARTICLES - self.sim.particles.len();
        if remaining == 0 {
            return;
        }
        let water_spawn = water_count.min(remaining);
        remaining = remaining.saturating_sub(water_spawn);
        let sediment_spawn = sediment_count.min(remaining);

        let cell_size = CELL_SIZE;
        let max_x = GRID_WIDTH as f32 * cell_size;
        let max_z = GRID_DEPTH as f32 * cell_size;
        let spread = CLICK_SPAWN_RADIUS;

        for _ in 0..water_spawn {
            let x = (center_x + (rand_float() - 0.5) * spread).clamp(0.0, max_x - 0.5 * cell_size);
            let z = (center_z + (rand_float() - 0.5) * spread).clamp(0.0, max_z - 0.5 * cell_size);
            let bed_y = self.bed_height_at(x, z);
            let y = bed_y + CLICK_WATER_LIFT + rand_float() * cell_size;
            self.sim.spawn_particle(Vec3::new(x, y, z));
        }

        for _ in 0..sediment_spawn {
            if self.sim.particles.len() >= MAX_PARTICLES {
                break;
            }
            let x = (center_x + (rand_float() - 0.5) * spread).clamp(0.0, max_x - 0.5 * cell_size);
            let z = (center_z + (rand_float() - 0.5) * spread).clamp(0.0, max_z - 0.5 * cell_size);
            let bed_y = self.bed_height_at(x, z);
            let y = bed_y + CLICK_SEDIMENT_LIFT + rand_float() * cell_size;
            self.sim.spawn_sediment(Vec3::new(x, y, z), Vec3::ZERO, SEDIMENT_REL_DENSITY);
        }
    }

    fn screen_to_world_ray(&self, screen_x: f32, screen_y: f32, width: f32, height: f32) -> (Vec3, Vec3) {
        let ndc_x = (2.0 * screen_x / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / height);

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
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, width / height, 0.01, 100.0);
        let inv_vp = (proj * view).inverse();

        let near = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;

        let dir = (far - near).normalize();
        (eye, dir)
    }

    fn handle_click(&mut self, screen_x: f32, screen_y: f32) {
        let (width, height) = if let Some(gpu) = &self.gpu {
            (gpu.config.width as f32, gpu.config.height as f32)
        } else {
            return;
        };

        if width <= 1.0 || height <= 1.0 {
            return;
        }

        let (origin, dir) = self.screen_to_world_ray(screen_x, screen_y, width, height);
        if dir.y.abs() < 1.0e-4 {
            return;
        }

        let t = (0.0 - origin.y) / dir.y;
        if t <= 0.0 {
            return;
        }

        let hit = origin + dir * t;
        let max_x = GRID_WIDTH as f32 * CELL_SIZE;
        let max_z = GRID_DEPTH as f32 * CELL_SIZE;
        if hit.x < 0.0 || hit.x > max_x || hit.z < 0.0 || hit.z > max_z {
            return;
        }

        if self.click_add_sediment {
            self.emit_particles_at(hit.x, hit.z, 0, CLICK_SPAWN_SEDIMENT);
        } else {
            self.emit_particles_at(hit.x, hit.z, CLICK_SPAWN_WATER, 0);
        }
    }

    fn request_gpu_sync(&mut self) {
        self.gpu_readback_pending = false;
        self.frame = 0;
    }

    fn clear_water(&mut self) {
        let before = self.sim.particles.len();
        self.sim.particles.list.retain(|p| p.is_sediment());
        let removed = before - self.sim.particles.len();
        self.pending_emit_water = 0;
        self.request_gpu_sync();
        println!("Cleared water particles: {}", removed);
    }

    fn clear_sediment(&mut self) {
        let before = self.sim.particles.len();
        self.sim.particles.list.retain(|p| !p.is_sediment());
        let removed = before - self.sim.particles.len();
        self.pending_emit_sediment = 0;
        self.request_gpu_sync();
        println!("Cleared sediment particles: {}", removed);
    }

    fn clear_all_particles(&mut self) {
        let removed = self.sim.particles.len();
        self.sim.particles.list.clear();
        self.pending_emit_water = 0;
        self.pending_emit_sediment = 0;
        self.request_gpu_sync();
        println!("Cleared all particles: {}", removed);
    }

    fn update_sediment_bed(&mut self, dt: f32) {
        let w = GRID_WIDTH;
        let d = GRID_DEPTH;
        let cell_size = CELL_SIZE;
        let column_count = w * d;
        let sample_height = cell_size * BED_SAMPLE_HEIGHT_CELLS;
        let sediment_band = cell_size * BED_AIR_MARGIN_CELLS;

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
            if p.density <= 1.0 {
                if p.position.y >= bed_y && p.position.y <= bed_y + sample_height {
                    self.bed_water_vel_sum[idx] += p.velocity;
                    self.bed_water_count[idx] += 1;
                }
            } else if p.position.y >= bed_y && p.position.y <= bed_y + sediment_band {
                self.bed_sediment_count[idx] += 1;
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
                    if deposit_quota[idx] > 0 && p.position.y <= bed_height[idx] + sediment_band {
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

    fn apply_bed_height_residual(&mut self) {
        let w = GRID_WIDTH;
        let d = GRID_DEPTH;
        let column_count = w * d;
        if self.bed_height_residual.len() != column_count {
            return;
        }

        let cell_size = CELL_SIZE;
        let sample_height = cell_size * BED_SAMPLE_HEIGHT_CELLS;
        let sediment_band = cell_size * BED_AIR_MARGIN_CELLS;
        let particle_height = cell_size / (SEDIMENT_REST_PARTICLES * (1.0 - BED_POROSITY));
        if particle_height <= 0.0 {
            return;
        }

        self.bed_water_vel_sum.fill(Vec3::ZERO);
        self.bed_water_count.fill(0);

        for p in &self.sim.particles.list {
            if p.is_sediment() {
                continue;
            }

            let i = (p.position.x / cell_size).floor() as i32;
            let k = (p.position.z / cell_size).floor() as i32;
            if i < 0 || i >= w as i32 || k < 0 || k >= d as i32 {
                continue;
            }
            let idx = k as usize * w + i as usize;
            let bed_y = self.bed_height[idx];
            if p.position.y >= bed_y && p.position.y <= bed_y + sample_height {
                self.bed_water_vel_sum[idx] += p.velocity;
                self.bed_water_count[idx] += 1;
            }
        }

        for idx in 0..column_count {
            let count = self.bed_water_count[idx];
            self.bed_water_vel_sum[idx] = if count > 0 {
                self.bed_water_vel_sum[idx] / count as f32
            } else {
                Vec3::ZERO
            };
        }

        let mut deposit_quota = vec![0u32; column_count];
        let mut erode_quota = vec![0u32; column_count];

        for idx in 0..column_count {
            let residual = self.bed_height_residual[idx];
            if residual >= particle_height {
                deposit_quota[idx] = (residual / particle_height).floor() as u32;
            } else if residual <= -particle_height {
                erode_quota[idx] = ((-residual) / particle_height).floor() as u32;
            }
        }

        let mut removed = vec![0u32; column_count];
        if deposit_quota.iter().any(|&quota| quota > 0) {
            let bed_height = &self.bed_height;
            self.sim.particles.list.retain(|p| {
                if p.is_sediment() {
                    let i = (p.position.x / cell_size).floor() as i32;
                    let k = (p.position.z / cell_size).floor() as i32;
                    if i >= 0 && i < w as i32 && k >= 0 && k < d as i32 {
                        let idx = k as usize * w + i as usize;
                        if deposit_quota[idx] > 0 && p.position.y <= bed_height[idx] + sediment_band {
                            deposit_quota[idx] -= 1;
                            removed[idx] += 1;
                            return false;
                        }
                    }
                }
                true
            });
        }

        let mut spawned = vec![0u32; column_count];
        for idx in 0..column_count {
            let spawn_count = erode_quota[idx];
            if spawn_count == 0 {
                continue;
            }
            let k = idx / w;
            let i = idx % w;
            let avg_vel = self.bed_water_vel_sum[idx];
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
                spawned[idx] += 1;
            }
        }

        let max_bed_height = (GRID_HEIGHT as f32 - 2.0) * cell_size;
        for idx in 0..column_count {
            let desired_delta = self.bed_height_residual[idx];
            let actual_delta = (removed[idx] as i32 - spawned[idx] as i32) as f32 * particle_height;
            let leftover = desired_delta - actual_delta;
            if leftover != 0.0 {
                self.bed_height[idx] -= leftover;
                self.bed_height[idx] = self.bed_height[idx].clamp(self.bed_base_height[idx], max_bed_height);
            }
            self.bed_height_residual[idx] = 0.0;
        }
    }

    fn accumulate_bed_height_delta(&mut self) {
        let column_count = GRID_WIDTH * GRID_DEPTH;
        if self.bed_height_prev.len() != column_count
            || self.bed_height.len() != column_count
            || self.bed_base_height.len() != column_count
        {
            return;
        }

        let particle_height = CELL_SIZE / (SEDIMENT_REST_PARTICLES * (1.0 - BED_POROSITY));
        if particle_height <= 0.0 {
            return;
        }
        let max_delta = particle_height * BED_MAX_DELTA_LAYERS;
        let max_bed_height = (GRID_HEIGHT as f32 - 2.0) * CELL_SIZE;

        for idx in 0..column_count {
            let mut delta = self.bed_height[idx] - self.bed_height_prev[idx];
            if delta > max_delta {
                delta = max_delta;
            } else if delta < -max_delta {
                delta = -max_delta;
            }
            let updated = (self.bed_height_prev[idx] + delta)
                .clamp(self.bed_base_height[idx], max_bed_height);
            self.bed_height[idx] = updated;
            self.bed_height_residual[idx] = updated - self.bed_height_prev[idx];
        }
    }

    fn relax_bed_height(&mut self) {
        let w = GRID_WIDTH;
        let d = GRID_DEPTH;
        let column_count = w * d;
        if self.bed_height.len() != column_count || self.bed_base_height.len() != column_count {
            return;
        }

        let max_diff = BED_MAX_SLOPE * CELL_SIZE;
        let max_bed_height = (GRID_HEIGHT as f32 - 2.0) * CELL_SIZE;
        let mut delta = vec![0.0f32; column_count];

        for _ in 0..BED_RELAX_ITERS {
            delta.fill(0.0);
            for k in 0..d {
                for i in 0..w {
                    let idx = k * w + i;
                    let h = self.bed_height[idx];
                    let base_h = self.bed_base_height[idx];

                    if i + 1 < w {
                        let n = idx + 1;
                        let diff = h - self.bed_height[n];
                        if diff.abs() > max_diff {
                            let excess = diff.abs() - max_diff;
                            let mut move_amt = 0.5 * excess;
                            if diff > 0.0 {
                                move_amt = move_amt.min(h - base_h);
                                delta[idx] -= move_amt;
                                delta[n] += move_amt;
                            } else {
                                let n_base = self.bed_base_height[n];
                                move_amt = move_amt.min(self.bed_height[n] - n_base);
                                delta[idx] += move_amt;
                                delta[n] -= move_amt;
                            }
                        }
                    }

                    if k + 1 < d {
                        let n = idx + w;
                        let diff = h - self.bed_height[n];
                        if diff.abs() > max_diff {
                            let excess = diff.abs() - max_diff;
                            let mut move_amt = 0.5 * excess;
                            if diff > 0.0 {
                                move_amt = move_amt.min(h - base_h);
                                delta[idx] -= move_amt;
                                delta[n] += move_amt;
                            } else {
                                let n_base = self.bed_base_height[n];
                                move_amt = move_amt.min(self.bed_height[n] - n_base);
                                delta[idx] += move_amt;
                                delta[n] -= move_amt;
                            }
                        }
                    }
                }
            }

            for idx in 0..column_count {
                let updated = self.bed_height[idx] + delta[idx];
                self.bed_height[idx] = updated.clamp(self.bed_base_height[idx], max_bed_height);
            }
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

    /// Build a 3D indexed mesh for the sluice bed geometry.
    /// Only exposed faces are included (faces adjacent to non-solid cells).
    fn build_bed_mesh(sim: &FlipSimulation3D) -> BedMesh {
        let mut mesh = BedMesh::new();
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;
        let cs = sim.grid.cell_size;

        // Colors for different face orientations (subtle shading)
        let color_top = [0.55, 0.50, 0.45, 1.0];    // Lighter for top faces
        let color_side = [0.45, 0.40, 0.35, 1.0];   // Medium for side faces
        let color_bottom = [0.35, 0.30, 0.25, 1.0]; // Darker for bottom faces

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !sim.grid.is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    // -X face (left)
                    if i == 0 || !sim.grid.is_solid(i - 1, j, k) {
                        let base = mesh.vertices.len() as u32;
                        mesh.vertices.extend_from_slice(&[
                            SurfaceVertex { position: [x0, y0, z0], color: color_side },
                            SurfaceVertex { position: [x0, y1, z0], color: color_side },
                            SurfaceVertex { position: [x0, y1, z1], color: color_side },
                            SurfaceVertex { position: [x0, y0, z1], color: color_side },
                        ]);
                        mesh.indices.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
                    }

                    // +X face (right)
                    if i == width - 1 || !sim.grid.is_solid(i + 1, j, k) {
                        let base = mesh.vertices.len() as u32;
                        mesh.vertices.extend_from_slice(&[
                            SurfaceVertex { position: [x1, y0, z0], color: color_side },
                            SurfaceVertex { position: [x1, y1, z0], color: color_side },
                            SurfaceVertex { position: [x1, y1, z1], color: color_side },
                            SurfaceVertex { position: [x1, y0, z1], color: color_side },
                        ]);
                        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
                    }

                    // -Y face (bottom)
                    if j == 0 || !sim.grid.is_solid(i, j - 1, k) {
                        let base = mesh.vertices.len() as u32;
                        mesh.vertices.extend_from_slice(&[
                            SurfaceVertex { position: [x0, y0, z0], color: color_bottom },
                            SurfaceVertex { position: [x1, y0, z0], color: color_bottom },
                            SurfaceVertex { position: [x1, y0, z1], color: color_bottom },
                            SurfaceVertex { position: [x0, y0, z1], color: color_bottom },
                        ]);
                        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
                    }

                    // +Y face (top) - most visible
                    if j == height - 1 || !sim.grid.is_solid(i, j + 1, k) {
                        let base = mesh.vertices.len() as u32;
                        mesh.vertices.extend_from_slice(&[
                            SurfaceVertex { position: [x0, y1, z0], color: color_top },
                            SurfaceVertex { position: [x1, y1, z0], color: color_top },
                            SurfaceVertex { position: [x1, y1, z1], color: color_top },
                            SurfaceVertex { position: [x0, y1, z1], color: color_top },
                        ]);
                        mesh.indices.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
                    }

                    // -Z face (front)
                    if k == 0 || !sim.grid.is_solid(i, j, k - 1) {
                        let base = mesh.vertices.len() as u32;
                        mesh.vertices.extend_from_slice(&[
                            SurfaceVertex { position: [x0, y0, z0], color: color_side },
                            SurfaceVertex { position: [x1, y0, z0], color: color_side },
                            SurfaceVertex { position: [x1, y1, z0], color: color_side },
                            SurfaceVertex { position: [x0, y1, z0], color: color_side },
                        ]);
                        mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
                    }

                    // +Z face (back)
                    if k == depth - 1 || !sim.grid.is_solid(i, j, k + 1) {
                        let base = mesh.vertices.len() as u32;
                        mesh.vertices.extend_from_slice(&[
                            SurfaceVertex { position: [x0, y0, z1], color: color_side },
                            SurfaceVertex { position: [x1, y0, z1], color: color_side },
                            SurfaceVertex { position: [x1, y1, z1], color: color_side },
                            SurfaceVertex { position: [x0, y1, z1], color: color_side },
                        ]);
                        mesh.indices.extend_from_slice(&[base, base+2, base+1, base, base+3, base+2]);
                    }
                }
            }
        }

        println!("Bed mesh: {} vertices, {} indices ({} triangles)",
            mesh.vertices.len(), mesh.indices.len(), mesh.indices.len() / 3);
        mesh
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
        self.bed_height_prev = self.bed_height.clone();
        self.bed_height_residual.fill(0.0);
        self.bed_water_vel_sum.fill(Vec3::ZERO);
        self.bed_water_count.fill(0);
        self.bed_sediment_count.fill(0);
        self.bed_flux_x.fill(0.0);
        self.bed_flux_z.fill(0.0);
        self.densities.clear();

        // Disabled: gpu_bed removed
        // if let (Some(gpu_bed), Some(gpu)) = (&self.gpu_bed, &self.gpu) {
        //     gpu_bed.reset_bed(&gpu.queue, &self.bed_base_height);
        // }

        self.solid_instances = Self::collect_solids(&sim);
        self.bed_mesh = Self::build_bed_mesh(&sim);
        // Upload bed mesh to GPU if initialized
        if let Some(gpu) = &self.gpu {
            self.bed_mesh.upload(&gpu.device);
        }

        self.sim = sim;
        self.pressure_iters_gpu = self.sim.pressure_iterations as u32;
        self.vorticity_epsilon = VORTICITY_EPSILON_DEFAULT;
        self.dp_friction_angle_deg = DP_FRICTION_ANGLE_DEFAULT_DEG;
        self.dp_cohesion = DP_COHESION_DEFAULT;
        self.dp_viscosity = DP_VISCOSITY_DEFAULT;
        self.dp_jammed_drag = DP_JAMMED_DRAG_DEFAULT;
        self.gpu_readback_pending = false;
        self.frame = 0;
        self.emitter_enabled = true;
        self.water_emitter_enabled = true;
        self.sediment_emitter_enabled = true;
        self.water_emit_rate = DEFAULT_WATER_EMIT_RATE;
        self.sediment_emit_rate = DEFAULT_SEDIMENT_EMIT_RATE;
        self.particles_exited = 0;
        self.pending_emit_water = 0;
        self.pending_emit_sediment = 0;
        self.click_add_sediment = false;
        self.cursor_pos = None;
        self.last_mouse_pos = None;
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

    // Disabled: bed_3d GPU probe stats functions removed
    // fn material_stats_from_gpu(...) { ... }
    // fn riffle_stats_from_gpu(...) { ... }
    // fn throughput_stats_from_gpu(...) { ... }

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

    /// Build water surface heightfield mesh with marching squares + physics-based wave displacement
    /// Water particles get their max Y stored per cell, smoothed, then organic triangles are built
    fn build_water_heightfield_mesh(&mut self) -> usize {
        self.water_heightfield.fill(f32::NEG_INFINITY);
        self.water_surface_vertices.clear();

        let width = GRID_WIDTH;
        let depth = GRID_DEPTH;
        let cell_size = CELL_SIZE;
        let time = self.simulation_time;

        // Wave parameters - rushing water feel
        const BASE_WAVE_AMPLITUDE: f32 = 0.001;  // Base ripple
        const VELOCITY_WAVE_SCALE: f32 = 0.003;  // Amplitude per m/s for main wave
        const CHOP_SCALE: f32 = 0.002;           // High-frequency chop amplitude per m/s
        const TURBULENCE_SCALE: f32 = 0.002;     // Splash amplitude
        const WAVE_FREQ: f32 = 5.0;              // Main wave frequency
        const CHOP_FREQ: f32 = 25.0;             // High-frequency chop
        const WAVE_SPEED_MULT: f32 = 8.0;        // How fast waves animate with flow

        // Velocity field per cell (for physics-based waves)
        let mut vel_sum_x = vec![0.0f32; width * depth];
        let mut vel_sum_z = vec![0.0f32; width * depth];
        let mut vel_sum_y = vec![0.0f32; width * depth];  // Vertical velocity for splashing
        let mut vel_count = vec![0u32; width * depth];

        // Maximum reasonable water height (reject outlier particles)
        let max_water_y = (GRID_HEIGHT as f32 - 2.0) * cell_size;

        // First pass: compute water surface height AND velocity per cell
        for p in &self.sim.particles.list {
            if p.is_sediment() {
                continue;
            }
            let i = (p.position.x / cell_size).floor() as i32;
            let k = (p.position.z / cell_size).floor() as i32;
            if i >= 0 && i < width as i32 && k >= 0 && k < depth as i32 {
                let idx = k as usize * width + i as usize;
                let y = p.position.y;
                // Clamp to reasonable height to prevent outlier spikes
                if y > self.water_heightfield[idx] && y < max_water_y {
                    self.water_heightfield[idx] = y;
                }
                // Accumulate velocity (only from particles below max height)
                if y < max_water_y {
                    vel_sum_x[idx] += p.velocity.x;
                    vel_sum_z[idx] += p.velocity.z;
                    vel_sum_y[idx] += p.velocity.y;
                    vel_count[idx] += 1;
                }
            }
        }

        // Compute average velocity per cell (raw)
        let mut vel_x_raw = vec![0.0f32; width * depth];
        let mut vel_z_raw = vec![0.0f32; width * depth];
        let mut vel_y_raw = vec![0.0f32; width * depth];
        for idx in 0..(width * depth) {
            if vel_count[idx] > 0 {
                let n = vel_count[idx] as f32;
                vel_x_raw[idx] = vel_sum_x[idx] / n;
                vel_z_raw[idx] = vel_sum_z[idx] / n;
                vel_y_raw[idx] = vel_sum_y[idx] / n;
            }
        }

        // Smooth velocity field (3x3 average) for gentler waves
        let mut vel_x = vec![0.0f32; width * depth];
        let mut vel_z = vec![0.0f32; width * depth];
        let mut vel_y = vec![0.0f32; width * depth];
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                if vel_count[idx] == 0 {
                    continue;
                }
                let mut sum_x = 0.0;
                let mut sum_z = 0.0;
                let mut sum_y = 0.0;
                let mut count = 0;
                for dk in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = i as i32 + di;
                        let nk = k as i32 + dk;
                        if ni >= 0 && ni < width as i32 && nk >= 0 && nk < depth as i32 {
                            let nidx = nk as usize * width + ni as usize;
                            if vel_count[nidx] > 0 {
                                sum_x += vel_x_raw[nidx];
                                sum_z += vel_z_raw[nidx];
                                sum_y += vel_y_raw[nidx];
                                count += 1;
                            }
                        }
                    }
                }
                if count > 0 {
                    vel_x[idx] = sum_x / count as f32;
                    vel_z[idx] = sum_z / count as f32;
                    vel_y[idx] = sum_y / count as f32;
                }
            }
        }

        // Second pass: smooth the heightfield (3x3 spatial average with neighbors)
        let mut smoothed = vec![f32::NEG_INFINITY; width * depth];
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let center = self.water_heightfield[idx];
                if !center.is_finite() {
                    continue;
                }

                let mut sum = 0.0;
                let mut count = 0;
                for dk in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = i as i32 + di;
                        let nk = k as i32 + dk;
                        if ni >= 0 && ni < width as i32 && nk >= 0 && nk < depth as i32 {
                            let nidx = nk as usize * width + ni as usize;
                            let h = self.water_heightfield[nidx];
                            if h.is_finite() {
                                sum += h;
                                count += 1;
                            }
                        }
                    }
                }
                if count > 0 {
                    smoothed[idx] = sum / count as f32;
                }
            }
        }

        // Third pass: temporal smoothing (blend with previous frame)
        const TEMPORAL_BLEND: f32 = 0.3;  // 0.3 = 30% new, 70% old (smoother)
        const PRESENCE_RISE_RATE: f32 = 0.15;  // How fast presence increases (slower = smoother edge appearance)
        const PRESENCE_FALL_RATE: f32 = 0.08;  // How fast presence decreases (slower = smoother edge disappearance)

        for idx in 0..(width * depth) {
            let new_h = smoothed[idx];
            let old_h = self.water_heightfield_smoothed[idx];

            if new_h.is_finite() && old_h.is_finite() {
                // Blend between old and new
                smoothed[idx] = old_h * (1.0 - TEMPORAL_BLEND) + new_h * TEMPORAL_BLEND;
            }
            // Update the stored smoothed heightfield for next frame
            self.water_heightfield_smoothed[idx] = smoothed[idx];

            // Update water presence with smooth transitions
            let has_water_now = new_h.is_finite();
            let old_presence = self.water_presence[idx];
            let target_presence = if has_water_now { 1.0 } else { 0.0 };

            let new_presence = if target_presence > old_presence {
                (old_presence + PRESENCE_RISE_RATE).min(1.0)
            } else {
                (old_presence - PRESENCE_FALL_RATE).max(0.0)
            };
            self.water_presence[idx] = new_presence;
        }

        // Copy presence for use in closures
        let presence = &self.water_presence;

        // Colors for depth-based shading (more transparent)
        let shallow_color = [0.15, 0.45, 0.85, 0.55];  // More transparent
        let deep_color = [0.06, 0.25, 0.60, 0.70];     // Slightly more opaque when deep

        // Physics-based wave displacement function - rushing water feel
        // Combines smooth undulation + high-frequency chop that scales with velocity
        let wave_offset = |x: f32, z: f32, local_vx: f32, local_vz: f32, local_vy: f32| -> f32 {
            let speed = (local_vx * local_vx + local_vz * local_vz).sqrt();
            let vert_speed = local_vy.abs();

            // Normalize flow direction
            let (dir_x, dir_z) = if speed > 0.05 {
                (local_vx / speed, local_vz / speed)
            } else {
                (1.0, 0.0)
            };

            // Distance along and across flow
            let flow_dist = dir_x * x + dir_z * z;
            let cross_dist = -dir_z * x + dir_x * z;

            // === MAIN SMOOTH WAVE ===
            // Amplitude grows with speed
            let main_amp = BASE_WAVE_AMPLITUDE + speed * VELOCITY_WAVE_SCALE;
            let main_phase = flow_dist * WAVE_FREQ - time * (2.0 + speed * WAVE_SPEED_MULT);
            let main_wave = main_phase.sin() * main_amp;

            // === HIGH-FREQUENCY CHOP ===
            // Fast ripples that only appear in moving water - gives "rushing" feel
            let chop_amp = speed * CHOP_SCALE;  // No chop when still
            let chop1 = (flow_dist * CHOP_FREQ - time * speed * 15.0).sin() * chop_amp;
            let chop2 = (cross_dist * CHOP_FREQ * 0.8 + time * 6.0).sin() * chop_amp * 0.5;
            // Diagonal chop for variety
            let chop3 = ((flow_dist + cross_dist) * CHOP_FREQ * 0.6 - time * 10.0).sin() * chop_amp * 0.3;

            // === CROSS WAVE ===
            let cross_amp = main_amp * 0.4;
            let cross_phase = cross_dist * WAVE_FREQ * 0.7 + time * 2.5;
            let cross_wave = cross_phase.sin() * cross_amp;

            // === TURBULENCE from vertical motion ===
            let splash = if vert_speed > 0.15 {
                let splash_amp = vert_speed * TURBULENCE_SCALE;
                (x * 30.0 + z * 35.0 + time * 12.0).sin() * splash_amp
            } else {
                0.0
            };

            main_wave + cross_wave + chop1 + chop2 + chop3 + splash
        };

        // Helper to check if cell has water (using presence for smooth edges)
        let has_water = |i: i32, k: i32| -> bool {
            if i < 0 || i >= width as i32 || k < 0 || k >= depth as i32 {
                return false;
            }
            // Use presence threshold for smoother edge transitions
            presence[k as usize * width + i as usize] > 0.1
        };

        // Get presence at cell
        let get_presence = |i: i32, k: i32| -> f32 {
            if i < 0 || i >= width as i32 || k < 0 || k >= depth as i32 {
                return 0.0;
            }
            presence[k as usize * width + i as usize]
        };

        // Helper to get velocity at a position (bilinear interpolation)
        let get_velocity_at = |x: f32, z: f32| -> (f32, f32, f32) {
            let fi = x / cell_size;
            let fk = z / cell_size;
            let i = fi.floor() as i32;
            let k = fk.floor() as i32;

            if i < 0 || i >= width as i32 - 1 || k < 0 || k >= depth as i32 - 1 {
                // Edge case: just return nearest cell velocity
                let ci = i.clamp(0, width as i32 - 1) as usize;
                let ck = k.clamp(0, depth as i32 - 1) as usize;
                let idx = ck * width + ci;
                return (vel_x[idx], vel_z[idx], vel_y[idx]);
            }

            // Bilinear interpolation weights
            let fx = fi - i as f32;
            let fz = fk - k as f32;
            let i = i as usize;
            let k = k as usize;

            let idx00 = k * width + i;
            let idx10 = k * width + i + 1;
            let idx01 = (k + 1) * width + i;
            let idx11 = (k + 1) * width + i + 1;

            let vx = vel_x[idx00] * (1.0 - fx) * (1.0 - fz)
                   + vel_x[idx10] * fx * (1.0 - fz)
                   + vel_x[idx01] * (1.0 - fx) * fz
                   + vel_x[idx11] * fx * fz;
            let vz = vel_z[idx00] * (1.0 - fx) * (1.0 - fz)
                   + vel_z[idx10] * fx * (1.0 - fz)
                   + vel_z[idx01] * (1.0 - fx) * fz
                   + vel_z[idx11] * fx * fz;
            let vy = vel_y[idx00] * (1.0 - fx) * (1.0 - fz)
                   + vel_y[idx10] * fx * (1.0 - fz)
                   + vel_y[idx01] * (1.0 - fx) * fz
                   + vel_y[idx11] * fx * fz;

            (vx, vz, vy)
        };

        // Helper to get height at position with physics-based wave displacement
        let get_height_at = |x: f32, z: f32, base_h: f32| -> f32 {
            let (vx, vz, vy) = get_velocity_at(x, z);
            base_h + wave_offset(x, z, vx, vz, vy)
        };

        // Get corner height from surrounding cells
        let get_corner_height = |ci: usize, ck: usize| -> Option<f32> {
            let mut sum = 0.0;
            let mut count = 0;
            for dk in 0..=1 {
                for di in 0..=1 {
                    if ci >= di && ck >= dk {
                        let cell_i = ci - di;
                        let cell_k = ck - dk;
                        if cell_i < width && cell_k < depth {
                            let idx = cell_k * width + cell_i;
                            let h = smoothed[idx];
                            if h.is_finite() {
                                sum += h;
                                count += 1;
                            }
                        }
                    }
                }
            }
            if count > 0 { Some(sum / count as f32) } else { None }
        };

        // Calculate color based on water depth, velocity, AND presence (for edge fading)
        let calc_color = |i: usize, k: usize, center_h: f32, speed: f32| -> [f32; 4] {
            let floor_t = i as f32 / (width - 1) as f32;
            let floor_height = (FLOOR_HEIGHT_LEFT as f32 * (1.0 - floor_t)
                + FLOOR_HEIGHT_RIGHT as f32 * floor_t) * cell_size;
            let water_depth = (center_h - floor_height).max(0.0);
            let depth_factor = (water_depth / (3.0 * cell_size)).min(1.0);

            // Get presence for edge fading
            let cell_presence = get_presence(i as i32, k as i32);

            // Base color from depth
            let base_r = shallow_color[0] * (1.0 - depth_factor) + deep_color[0] * depth_factor;
            let base_g = shallow_color[1] * (1.0 - depth_factor) + deep_color[1] * depth_factor;
            let base_b = shallow_color[2] * (1.0 - depth_factor) + deep_color[2] * depth_factor;
            let base_a = shallow_color[3] * (1.0 - depth_factor) + deep_color[3] * depth_factor;

            // Foam factor: fast water -> whiter (desaturated, brighter)
            let foam = (speed / 1.5).min(1.0);  // Full foam at 1.5 m/s
            let foam_color = [0.80, 0.88, 0.95, 0.65];  // Pale blue-white foam, semi-transparent

            // Apply presence to alpha for smooth edge fading
            let alpha = (base_a * (1.0 - foam * 0.5) + foam_color[3] * foam * 0.5) * cell_presence;

            [
                base_r * (1.0 - foam) + foam_color[0] * foam,
                base_g * (1.0 - foam) + foam_color[1] * foam,
                base_b * (1.0 - foam) + foam_color[2] * foam,
                alpha,
            ]
        };

        // Third pass: marching squares for organic water boundary
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let center_h = smoothed[idx];

                // Build marching squares case from 4 corners
                // Corners: 0=bottom-left(i,k), 1=bottom-right(i+1,k), 2=top-left(i,k+1), 3=top-right(i+1,k+1)
                let c0 = has_water(i as i32, k as i32);
                let c1 = has_water(i as i32 + 1, k as i32);
                let c2 = has_water(i as i32, k as i32 + 1);
                let c3 = has_water(i as i32 + 1, k as i32 + 1);

                let case = (c0 as u8) | ((c1 as u8) << 1) | ((c2 as u8) << 2) | ((c3 as u8) << 3);

                if case == 0 {
                    continue; // No water
                }

                // Get base heights at corners (with fallback to center)
                let h00 = get_corner_height(i, k).unwrap_or(center_h);
                let h10 = get_corner_height(i + 1, k).unwrap_or(center_h);
                let h01 = get_corner_height(i, k + 1).unwrap_or(center_h);
                let h11 = get_corner_height(i + 1, k + 1).unwrap_or(center_h);
                let h_center = (h00 + h10 + h01 + h11) / 4.0;

                // World positions
                let x0 = i as f32 * cell_size;
                let x1 = (i + 1) as f32 * cell_size;
                let xm = (x0 + x1) * 0.5;
                let z0 = k as f32 * cell_size;
                let z1 = (k + 1) as f32 * cell_size;
                let zm = (z0 + z1) * 0.5;

                // Heights with wave displacement at each position
                let y00 = get_height_at(x0, z0, h00);
                let y10 = get_height_at(x1, z0, h10);
                let y01 = get_height_at(x0, z1, h01);
                let y11 = get_height_at(x1, z1, h11);
                let ym0 = get_height_at(xm, z0, (h00 + h10) * 0.5);  // Mid bottom edge
                let ym1 = get_height_at(xm, z1, (h01 + h11) * 0.5);  // Mid top edge
                let y0m = get_height_at(x0, zm, (h00 + h01) * 0.5);  // Mid left edge
                let y1m = get_height_at(x1, zm, (h10 + h11) * 0.5);  // Mid right edge
                let ymm = get_height_at(xm, zm, h_center);           // Center

                // Get cell velocity for foam coloring
                let cell_speed = (vel_x[idx] * vel_x[idx] + vel_z[idx] * vel_z[idx]).sqrt();
                let color = calc_color(i, k, center_h, cell_speed);

                // Vertices at 9 positions (corners, edge midpoints, center)
                let v00 = SurfaceVertex { position: [x0, y00, z0], color };
                let v10 = SurfaceVertex { position: [x1, y10, z0], color };
                let v01 = SurfaceVertex { position: [x0, y01, z1], color };
                let v11 = SurfaceVertex { position: [x1, y11, z1], color };
                let vm0 = SurfaceVertex { position: [xm, ym0, z0], color };  // Bottom mid
                let vm1 = SurfaceVertex { position: [xm, ym1, z1], color };  // Top mid
                let v0m = SurfaceVertex { position: [x0, y0m, zm], color };  // Left mid
                let v1m = SurfaceVertex { position: [x1, y1m, zm], color };  // Right mid
                let vmm = SurfaceVertex { position: [xm, ymm, zm], color };  // Center

                // Generate triangles based on marching squares case
                // Using edge midpoints creates organic boundaries at water edges
                match case {
                    // Single corner cases - triangular wedges
                    1 => { // Only bottom-left
                        self.water_surface_vertices.extend_from_slice(&[v00, vm0, v0m]);
                    }
                    2 => { // Only bottom-right
                        self.water_surface_vertices.extend_from_slice(&[vm0, v10, v1m]);
                    }
                    4 => { // Only top-left
                        self.water_surface_vertices.extend_from_slice(&[v0m, vm1, v01]);
                    }
                    8 => { // Only top-right
                        self.water_surface_vertices.extend_from_slice(&[v1m, v11, vm1]);
                    }

                    // Two adjacent corners - half cells
                    3 => { // Bottom row
                        self.water_surface_vertices.extend_from_slice(&[
                            v00, v10, v1m,
                            v00, v1m, v0m,
                        ]);
                    }
                    5 => { // Left column
                        self.water_surface_vertices.extend_from_slice(&[
                            v00, vm0, vm1,
                            v00, vm1, v01,
                        ]);
                    }
                    10 => { // Right column
                        self.water_surface_vertices.extend_from_slice(&[
                            vm0, v10, v11,
                            vm0, v11, vm1,
                        ]);
                    }
                    12 => { // Top row
                        self.water_surface_vertices.extend_from_slice(&[
                            v0m, v1m, v11,
                            v0m, v11, v01,
                        ]);
                    }

                    // Diagonal cases - use center point
                    6 => { // Bottom-right and top-left (diagonal)
                        self.water_surface_vertices.extend_from_slice(&[
                            vm0, v10, v1m,
                            v1m, vmm, vm0,
                            v0m, vm1, v01,
                            v0m, vmm, vm1,
                        ]);
                    }
                    9 => { // Bottom-left and top-right (diagonal)
                        self.water_surface_vertices.extend_from_slice(&[
                            v00, vm0, v0m,
                            vm0, vmm, v0m,
                            v1m, v11, vm1,
                            vmm, v1m, vm1,
                        ]);
                    }

                    // Three corners - L shapes
                    7 => { // Missing top-right
                        self.water_surface_vertices.extend_from_slice(&[
                            v00, v10, v1m,
                            v00, v1m, vmm,
                            v00, vmm, vm1,
                            v00, vm1, v01,
                        ]);
                    }
                    11 => { // Missing top-left
                        self.water_surface_vertices.extend_from_slice(&[
                            v00, v10, v11,
                            v00, v11, vm1,
                            v00, vm1, vmm,
                            v00, vmm, v0m,
                        ]);
                    }
                    13 => { // Missing bottom-right
                        self.water_surface_vertices.extend_from_slice(&[
                            v00, vm0, vmm,
                            v00, vmm, v1m,
                            v00, v1m, v11,
                            v00, v11, v01,
                        ]);
                    }
                    14 => { // Missing bottom-left
                        self.water_surface_vertices.extend_from_slice(&[
                            vm0, v10, v11,
                            vm0, v11, v01,
                            vm0, v01, v0m,
                            vm0, v0m, vmm,
                        ]);
                    }

                    // Full cell - subdivided mesh for smooth high-resolution surface
                    15 => {
                        let n = WATER_MESH_SUBDIVISIONS;
                        let sub_size = cell_size / n as f32;

                        for sk in 0..n {
                            for si in 0..n {
                                // Normalized position within cell [0,1]
                                let u0 = si as f32 / n as f32;
                                let u1 = (si + 1) as f32 / n as f32;
                                let v0 = sk as f32 / n as f32;
                                let v1 = (sk + 1) as f32 / n as f32;
                                let um = (u0 + u1) * 0.5;
                                let vm = (v0 + v1) * 0.5;

                                // World positions for this sub-cell
                                let sx0 = x0 + si as f32 * sub_size;
                                let sx1 = x0 + (si + 1) as f32 * sub_size;
                                let sz0 = z0 + sk as f32 * sub_size;
                                let sz1 = z0 + (sk + 1) as f32 * sub_size;
                                let sxm = (sx0 + sx1) * 0.5;
                                let szm = (sz0 + sz1) * 0.5;

                                // Bilinear interpolation of base heights at sub-cell corners
                                let bh00 = h00 * (1.0 - u0) * (1.0 - v0) + h10 * u0 * (1.0 - v0)
                                         + h01 * (1.0 - u0) * v0 + h11 * u0 * v0;
                                let bh10 = h00 * (1.0 - u1) * (1.0 - v0) + h10 * u1 * (1.0 - v0)
                                         + h01 * (1.0 - u1) * v0 + h11 * u1 * v0;
                                let bh01 = h00 * (1.0 - u0) * (1.0 - v1) + h10 * u0 * (1.0 - v1)
                                         + h01 * (1.0 - u0) * v1 + h11 * u0 * v1;
                                let bh11 = h00 * (1.0 - u1) * (1.0 - v1) + h10 * u1 * (1.0 - v1)
                                         + h01 * (1.0 - u1) * v1 + h11 * u1 * v1;
                                let bhm = h00 * (1.0 - um) * (1.0 - vm) + h10 * um * (1.0 - vm)
                                        + h01 * (1.0 - um) * vm + h11 * um * vm;

                                // Heights with wave displacement
                                let sy00 = get_height_at(sx0, sz0, bh00);
                                let sy10 = get_height_at(sx1, sz0, bh10);
                                let sy01 = get_height_at(sx0, sz1, bh01);
                                let sy11 = get_height_at(sx1, sz1, bh11);

                                // Get velocity at sub-cell center for foam color
                                let (sub_vx, sub_vz, _) = get_velocity_at(sxm, szm);
                                let sub_speed = (sub_vx * sub_vx + sub_vz * sub_vz).sqrt();
                                let sub_color = calc_color(i, k, bhm, sub_speed);

                                // Two triangles per sub-cell
                                self.water_surface_vertices.extend_from_slice(&[
                                    SurfaceVertex { position: [sx0, sy00, sz0], color: sub_color },
                                    SurfaceVertex { position: [sx1, sy10, sz0], color: sub_color },
                                    SurfaceVertex { position: [sx1, sy11, sz1], color: sub_color },
                                    SurfaceVertex { position: [sx0, sy00, sz0], color: sub_color },
                                    SurfaceVertex { position: [sx1, sy11, sz1], color: sub_color },
                                    SurfaceVertex { position: [sx0, sy01, sz1], color: sub_color },
                                ]);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Unified water volume: bottom faces + side walls at same detail level as surface
        // This creates a solid water volume instead of just a surface

        // Second pass for volume: bottom faces and side walls (same detail as surface)
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let center_h = smoothed[idx];

                // Only process full water cells (case 15) for bottom - others are at boundary
                let c0 = has_water(i as i32, k as i32);
                let c1 = has_water(i as i32 + 1, k as i32);
                let c2 = has_water(i as i32, k as i32 + 1);
                let c3 = has_water(i as i32 + 1, k as i32 + 1);
                let case = (c0 as u8) | ((c1 as u8) << 1) | ((c2 as u8) << 2) | ((c3 as u8) << 3);

                if case == 0 || !center_h.is_finite() {
                    continue;
                }

                // Floor height interpolated across the cell
                let get_floor_y = |x: f32| -> f32 {
                    let t = x / ((width - 1) as f32 * cell_size);
                    (FLOOR_HEIGHT_LEFT as f32 * (1.0 - t) + FLOOR_HEIGHT_RIGHT as f32 * t) * cell_size
                };

                // Get base heights at corners
                let h00 = get_corner_height(i, k).unwrap_or(center_h);
                let h10 = get_corner_height(i + 1, k).unwrap_or(center_h);
                let h01 = get_corner_height(i, k + 1).unwrap_or(center_h);
                let h11 = get_corner_height(i + 1, k + 1).unwrap_or(center_h);

                let x0 = i as f32 * cell_size;
                let x1 = (i + 1) as f32 * cell_size;
                let z0 = k as f32 * cell_size;
                let z1 = (k + 1) as f32 * cell_size;

                // Simple side walls at water boundaries (single quad per edge for speed)
                let xm = (x0 + x1) * 0.5;
                let zm = (z0 + z1) * 0.5;

                // Get cell velocity for side color
                let (cell_vx, cell_vz, _) = get_velocity_at(xm, zm);
                let cell_speed = (cell_vx * cell_vx + cell_vz * cell_vz).sqrt();
                let mut side_color = calc_color(i, k, center_h, cell_speed);
                side_color[0] *= 0.8;
                side_color[1] *= 0.8;
                side_color[2] *= 0.9;

                let floor_y0 = get_floor_y(x0);
                let floor_y1 = get_floor_y(x1);
                let sy00 = get_height_at(x0, z0, h00);
                let sy10 = get_height_at(x1, z0, h10);
                let sy01 = get_height_at(x0, z1, h01);
                let sy11 = get_height_at(x1, z1, h11);

                // Front edge (z = z0)
                if k == 0 || !smoothed[(k - 1) * width + i].is_finite() {
                    self.water_surface_vertices.extend_from_slice(&[
                        SurfaceVertex { position: [x0, floor_y0, z0], color: side_color },
                        SurfaceVertex { position: [x1, floor_y1, z0], color: side_color },
                        SurfaceVertex { position: [x1, sy10, z0], color: side_color },
                        SurfaceVertex { position: [x0, floor_y0, z0], color: side_color },
                        SurfaceVertex { position: [x1, sy10, z0], color: side_color },
                        SurfaceVertex { position: [x0, sy00, z0], color: side_color },
                    ]);
                }

                // Back edge (z = z1)
                if k == depth - 1 || !smoothed[(k + 1) * width + i].is_finite() {
                    self.water_surface_vertices.extend_from_slice(&[
                        SurfaceVertex { position: [x0, floor_y0, z1], color: side_color },
                        SurfaceVertex { position: [x0, sy01, z1], color: side_color },
                        SurfaceVertex { position: [x1, sy11, z1], color: side_color },
                        SurfaceVertex { position: [x0, floor_y0, z1], color: side_color },
                        SurfaceVertex { position: [x1, sy11, z1], color: side_color },
                        SurfaceVertex { position: [x1, floor_y1, z1], color: side_color },
                    ]);
                }

                // Left edge (x = x0)
                if i == 0 || !smoothed[k * width + i - 1].is_finite() {
                    self.water_surface_vertices.extend_from_slice(&[
                        SurfaceVertex { position: [x0, floor_y0, z0], color: side_color },
                        SurfaceVertex { position: [x0, sy00, z0], color: side_color },
                        SurfaceVertex { position: [x0, sy01, z1], color: side_color },
                        SurfaceVertex { position: [x0, floor_y0, z0], color: side_color },
                        SurfaceVertex { position: [x0, sy01, z1], color: side_color },
                        SurfaceVertex { position: [x0, floor_y0, z1], color: side_color },
                    ]);
                }

                // Right edge (x = x1)
                if i == width - 1 || !smoothed[k * width + i + 1].is_finite() {
                    self.water_surface_vertices.extend_from_slice(&[
                        SurfaceVertex { position: [x1, floor_y1, z0], color: side_color },
                        SurfaceVertex { position: [x1, floor_y1, z1], color: side_color },
                        SurfaceVertex { position: [x1, sy11, z1], color: side_color },
                        SurfaceVertex { position: [x1, floor_y1, z0], color: side_color },
                        SurfaceVertex { position: [x1, sy11, z1], color: side_color },
                        SurfaceVertex { position: [x1, sy10, z0], color: side_color },
                    ]);
                }
            }
        }

        self.water_surface_vertices.len()
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

    /// Build particle instances sorted by type (water first, then sediment) for SSFR rendering
    /// In hybrid mode, only includes water particles ABOVE the water heightfield (overtopping)
    /// Returns (instances, water_count, sediment_count)
    fn build_particle_instances_sorted(&self) -> (Vec<ParticleInstance>, usize, usize) {
        let mut water_instances = Vec::new();
        let mut sediment_instances = Vec::new();

        let cell_size = CELL_SIZE;
        let width = GRID_WIDTH;
        let depth = GRID_DEPTH;

        for p in &self.sim.particles.list {
            let speed = p.velocity.length();
            let t = (speed / 2.5).min(1.0);

            if p.is_sediment() {
                sediment_instances.push(ParticleInstance {
                    position: [p.position.x, p.position.y, p.position.z],
                    color: [
                        0.6 + t * 0.2,
                        0.5 + t * 0.2,
                        0.2 + t * 0.1,
                        0.85,
                    ],
                });
            } else {
                // For water: only include if above heightfield (overtopping particles)
                let i = (p.position.x / cell_size).floor() as i32;
                let k = (p.position.z / cell_size).floor() as i32;

                let include = if i >= 0 && i < width as i32 && k >= 0 && k < depth as i32 {
                    let idx = k as usize * width + i as usize;
                    let surface_y = self.water_heightfield[idx];
                    // Include if no heightfield data OR particle is above surface + threshold
                    !surface_y.is_finite() || p.position.y > surface_y + WATER_OVERTOP_THRESHOLD
                } else {
                    true  // Outside grid, include as splat
                };

                if include {
                    water_instances.push(ParticleInstance {
                        position: [p.position.x, p.position.y, p.position.z],
                        color: [
                            0.3 + t * 0.5,  // Slightly brighter for splash visibility
                            0.5 + t * 0.4,
                            0.9,
                            0.9,
                        ],
                    });
                }
            }
        }

        let water_count = water_instances.len();
        let sediment_count = sediment_instances.len();

        // Water first, then sediment
        water_instances.extend(sediment_instances);
        (water_instances, water_count, sediment_count)
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        if !self.paused {
            let dt = 1.0 / 60.0;
            self.simulation_time += dt;

            if self.use_gpu_sim {
                let bed_dt = dt * GPU_SYNC_STRIDE as f32;
                // Disabled: build_gpu_bed_params removed
                let _ = bed_dt; // suppress unused warning
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
                    if self.water_emitter_enabled {
                        self.pending_emit_water = self.pending_emit_water.saturating_add(self.water_emit_rate);
                    }
                    if self.sediment_emitter_enabled {
                        self.pending_emit_sediment = self.pending_emit_sediment.saturating_add(self.sediment_emit_rate);
                    }
                }

                if should_schedule {
                    if do_sync {
                        if self.pending_emit_water > 0 || self.pending_emit_sediment > 0 {
                            self.emit_particles(self.pending_emit_water, self.pending_emit_sediment);
                            self.pending_emit_water = 0;
                            self.pending_emit_sediment = 0;
                        }

                        // Disabled: gpu_bed removed, always use CPU bed update
                        self.update_sediment_bed(dt);

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
                            // Disabled: DruckerPragerParams replaced by friction-only in g2p_3d
                            // gpu_flip.set_drucker_prager_params(&gpu.queue, dp_params);
                            let sdf = self.sim.grid.sdf.as_slice();
                            let positions = &mut self.positions;
                            let velocities = &mut self.velocities;
                            let c_matrices = &mut self.c_matrices;
                            let densities = &self.densities;
                            let cell_types = &self.cell_types;
                            // Disabled: gpu_bed removed, always use CPU bed_height
                            let bed_height = Some(self.bed_height.as_slice());
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
                        // Disabled: gpu_bed removed
                        // let mut bed_updated = false;
                        // if let (Some(gpu_bed), Some(gpu)) = (self.gpu_bed.as_mut(), self.gpu.as_ref()) {
                        //     gpu_bed.update(...);
                        //     gpu_bed.read_bed_height(...);
                        //     bed_updated = true;
                        // }

                        if false { // bed_updated always false now
                            self.accumulate_bed_height_delta();
                            self.apply_bed_height_residual();
                            self.relax_bed_height();
                            self.bed_height_prev.copy_from_slice(&self.bed_height);
                            // Disabled: gpu_bed removed
                            // if let (Some(gpu_bed), Some(gpu)) = (self.gpu_bed.as_ref(), self.gpu.as_ref()) {
                            //     gpu_bed.write_bed_height(&gpu.queue, &self.bed_height);
                            // }
                        }
                    } else {
                        let pressure_iters = self.pressure_iters_gpu;
                        let flow_accel = flow_accel_from_slope();
                        if let (Some(gpu_flip), Some(gpu)) = (&mut self.gpu_flip, &self.gpu) {
                            // Disabled: DruckerPragerParams replaced by friction-only in g2p_3d
                            // gpu_flip.set_drucker_prager_params(&gpu.queue, dp_params);
                            let sdf = self.sim.grid.sdf.as_slice();
                            let cell_types = &self.cell_types;
                            // Disabled: gpu_bed removed, always use CPU bed_height
                            let bed_height = Some(self.bed_height.as_slice());
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
                    }
                }
            } else {
                if self.emitter_enabled && self.frame % 2 == 0 {
                    let water = if self.water_emitter_enabled { self.water_emit_rate } else { 0 };
                    let sediment = if self.sediment_emitter_enabled { self.sediment_emit_rate } else { 0 };
                    self.emit_particles(water, sediment);
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

        // Print FPS every 10 frames for visibility
        if self.frame % 10 == 0 {
            println!(">>> FPS: {:.1} | Particles: {} <<<", self.current_fps, self.sim.particles.len());
        }

        // Print full stats every 30 frames
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

            // Print jamming diagnostics every 30 frames when using GPU
            if self.use_gpu_sim {
                if let (Some(gpu_flip), Some(gpu)) = (&self.gpu_flip, &self.gpu) {
                    gpu_flip.print_jamming_diagnostics(&gpu.device, &gpu.queue);
                }
            }

            if self.debug_riffle_probe {
                if self.use_gpu_sim {
                    // Disabled: gpu_bed probe stats removed - all GPU probe code deleted
                    println!("[Probe] GPU bed system disabled");
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
        let (surface_vertex_count, flow_particle_count, all_instances, water_particle_count, sediment_particle_count, water_mesh_vertex_count) = if self.render_heightfield {
            let svc = self.build_heightfield_vertices();
            let fpc = if self.render_flow_particles {
                self.build_flow_particles()
            } else {
                0
            };
            // Build water heightfield mesh FIRST (populates water_heightfield for filtering)
            let wmvc = self.build_water_heightfield_mesh();
            // For SSFR: build sorted particle instances (water first, then sediment)
            // Now uses water_heightfield to filter - only overtopping particles are splats
            let (instances, wpc, spc) = self.build_particle_instances_sorted();
            (svc, fpc, instances, wpc, spc, wmvc)
        } else {
            // Non-heightfield mode: draw all particles (unsorted)
            let instances = self.build_all_particle_instances();
            let count = instances.len();
            (0, 0, instances, count, 0, 0)  // Treat all as "water" for unsorted mode
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
        // Upload water heightfield mesh
        if water_mesh_vertex_count > 0 {
            gpu.queue.write_buffer(
                &gpu.water_surface_buffer,
                0,
                bytemuck::cast_slice(&self.water_surface_vertices),
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
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Use SSFR for water rendering when heightfield mode is on
        if self.render_heightfield && gpu.water_ssfr.is_some() {
            let ssfr = gpu.water_ssfr.as_ref().unwrap();

            // Update water uniforms
            let water_uniforms = WaterUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                view: view.to_cols_array_2d(),
                proj: proj.to_cols_array_2d(),
                inv_proj: proj.inverse().to_cols_array_2d(),
                camera_pos: eye.to_array(),
                particle_radius: WATER_PARTICLE_RADIUS,
                screen_size: [gpu.config.width as f32, gpu.config.height as f32],
                near: SSFR_NEAR,
                far: SSFR_FAR,
            };
            gpu.queue.write_buffer(&ssfr.water_uniform_buffer, 0, bytemuck::bytes_of(&water_uniforms));

            // Pass 1: Clear scene depth texture (bed mesh rendering skipped for now -
            // would require depth-only pipeline. Water still renders correctly without bed occlusion)
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Scene Depth Pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &ssfr.scene_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                // Note: Bed mesh rendering skipped - requires depth-only pipeline
            }

            // Pass 2: Render water particles to depth + thickness textures
            if water_particle_count > 0 {
                // Create a temporary depth texture view for the water depth pass
                let water_depth_stencil_view = ssfr.scene_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Water Depth Pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &ssfr.depth_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 0.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &water_depth_stencil_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(&ssfr.depth_pipeline);
                pass.set_bind_group(0, &ssfr.depth_bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                // Draw only water particles (first water_particle_count instances)
                pass.draw(0..4, 0..water_particle_count as u32);
            }

            // Pass 3: Bilateral blur (H then V)
            if water_particle_count > 0 {
                let workgroup_x = (gpu.config.width + 7) / 8;
                let workgroup_y = (gpu.config.height + 7) / 8;

                // Update blur uniforms for horizontal pass
                let blur_h_uniforms = BlurUniforms {
                    texel_size: [1.0 / gpu.config.width as f32, 1.0 / gpu.config.height as f32],
                    blur_radius: SSFR_BLUR_RADIUS,
                    depth_falloff: SSFR_BLUR_DEPTH_FALLOFF,
                    direction: [1.0, 0.0],
                    _pad: [0.0, 0.0],
                };
                gpu.queue.write_buffer(&ssfr.blur_uniform_buffer, 0, bytemuck::bytes_of(&blur_h_uniforms));

                // Horizontal blur: depth -> temp
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Blur H Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&ssfr.blur_h_pipeline);
                    pass.set_bind_group(0, &ssfr.blur_h_bind_group, &[]);
                    pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
                }

                // Update blur uniforms for vertical pass
                let blur_v_uniforms = BlurUniforms {
                    texel_size: [1.0 / gpu.config.width as f32, 1.0 / gpu.config.height as f32],
                    blur_radius: SSFR_BLUR_RADIUS,
                    depth_falloff: SSFR_BLUR_DEPTH_FALLOFF,
                    direction: [0.0, 1.0],
                    _pad: [0.0, 0.0],
                };
                gpu.queue.write_buffer(&ssfr.blur_uniform_buffer, 0, bytemuck::cast_slice(&[blur_v_uniforms]));

                // Vertical blur: temp -> smoothed
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Blur V Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&ssfr.blur_v_pipeline);
                    pass.set_bind_group(0, &ssfr.blur_v_bind_group, &[]);
                    pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
                }
            }

            // Pass 4: Final composite - render scene, then water overlay
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Final Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &frame_view,
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

                // Draw bed mesh
                if let (Some(vb), Some(ib)) = (&self.bed_mesh.vertex_buffer, &self.bed_mesh.index_buffer) {
                    pass.set_pipeline(&gpu.surface_pipeline);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.bed_mesh.num_indices(), 0, 0..1);
                }

                // Draw water heightfield mesh (bulk water surface)
                if water_mesh_vertex_count > 0 {
                    pass.set_pipeline(&gpu.surface_pipeline);
                    pass.set_vertex_buffer(0, gpu.water_surface_buffer.slice(..));
                    pass.draw(0..water_mesh_vertex_count as u32, 0..1);
                }

                // Composite water splats (overtopping particles only)
                if water_particle_count > 0 {
                    pass.set_pipeline(&ssfr.composite_pipeline);
                    pass.set_bind_group(0, &ssfr.composite_bind_group, &[]);
                    pass.draw(0..3, 0..1);  // Full-screen triangle
                }

                // Draw sediment particles on top
                if sediment_particle_count > 0 {
                    pass.set_pipeline(&gpu.pipeline);
                    pass.set_bind_group(0, &gpu.bind_group, &[]);
                    pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                    // Sediment particles start after water particles
                    pass.draw(0..4, water_particle_count as u32..instance_count as u32);
                }
            }
        } else {
            // Original rendering path (no SSFR)
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
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

            // Draw bed mesh (3D solid geometry)
            if let (Some(vb), Some(ib)) = (&self.bed_mesh.vertex_buffer, &self.bed_mesh.index_buffer) {
                pass.set_pipeline(&gpu.surface_pipeline);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.bed_mesh.num_indices(), 0, 0..1);
            }

            if instance_count > 0 {
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
        // Disabled: GpuBed3D removed - using friction-only sediment model
        // let positions_buffer = gpu_flip.positions_buffer();
        // let velocities_buffer = gpu_flip.velocities_buffer();
        // let densities_buffer = gpu_flip.densities_buffer();
        // let gpu_bed = GpuBed3D::new(...);
        // self.gpu_bed = Some(gpu_bed);
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
        let water_surface_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Surface Buffer"),
            size: (MAX_WATER_SURFACE_VERTICES * std::mem::size_of::<SurfaceVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let solid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Solid Buffer"),
            contents: bytemuck::cast_slice(&self.solid_instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Upload bed mesh to GPU
        self.bed_mesh.upload(&device);

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

        // Create SSFR water rendering
        let water_ssfr = create_water_ssfr(&device, &config, &vertex_buffer, &instance_buffer);

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
            water_surface_buffer,
            uniform_buffer,
            bind_group,
            surface_pipeline,
            water_ssfr: Some(water_ssfr),
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
                        PhysicalKey::Code(KeyCode::KeyW) => {
                            self.water_emitter_enabled = !self.water_emitter_enabled;
                            println!(
                                "Water emit: {} (rate {})",
                                if self.water_emitter_enabled { "ON" } else { "OFF" },
                                self.water_emit_rate
                            );
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            self.sediment_emitter_enabled = !self.sediment_emitter_enabled;
                            println!(
                                "Sediment emit: {} (rate {})",
                                if self.sediment_emitter_enabled { "ON" } else { "OFF" },
                                self.sediment_emit_rate
                            );
                        }
                        PhysicalKey::Code(KeyCode::KeyM) => {
                            self.click_add_sediment = !self.click_add_sediment;
                            println!(
                                "Click material: {}",
                                if self.click_add_sediment { "SEDIMENT" } else { "WATER" }
                            );
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.water_emit_rate = (self.water_emit_rate + EMIT_RATE_STEP).min(MAX_EMIT_RATE);
                            println!("Water emit rate: {}", self.water_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.water_emit_rate = self.water_emit_rate.saturating_sub(EMIT_RATE_STEP);
                            println!("Water emit rate: {}", self.water_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.sediment_emit_rate = (self.sediment_emit_rate + EMIT_RATE_STEP).min(MAX_EMIT_RATE);
                            println!("Sediment emit rate: {}", self.sediment_emit_rate);
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.sediment_emit_rate = self.sediment_emit_rate.saturating_sub(EMIT_RATE_STEP);
                            println!("Sediment emit rate: {}", self.sediment_emit_rate);
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
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            self.dp_friction_angle_deg =
                                (self.dp_friction_angle_deg - DP_FRICTION_STEP_DEG).max(20.0);
                            println!("DP friction angle: {:.1}°", self.dp_friction_angle_deg);
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            self.dp_friction_angle_deg =
                                (self.dp_friction_angle_deg + DP_FRICTION_STEP_DEG).min(45.0);
                            println!("DP friction angle: {:.1}°", self.dp_friction_angle_deg);
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            self.dp_cohesion = (self.dp_cohesion - DP_COHESION_STEP).max(0.0);
                            println!("DP cohesion: {:.1} Pa", self.dp_cohesion);
                        }
                        PhysicalKey::Code(KeyCode::Digit4) => {
                            self.dp_cohesion = (self.dp_cohesion + DP_COHESION_STEP).min(100.0);
                            println!("DP cohesion: {:.1} Pa", self.dp_cohesion);
                        }
                        PhysicalKey::Code(KeyCode::Digit5) => {
                            self.dp_viscosity = (self.dp_viscosity - DP_VISCOSITY_STEP).max(0.1);
                            println!("DP viscosity: {:.2} Pa*s", self.dp_viscosity);
                        }
                        PhysicalKey::Code(KeyCode::Digit6) => {
                            self.dp_viscosity = (self.dp_viscosity + DP_VISCOSITY_STEP).min(10.0);
                            println!("DP viscosity: {:.2} Pa*s", self.dp_viscosity);
                        }
                        PhysicalKey::Code(KeyCode::Digit7) => {
                            self.dp_jammed_drag = (self.dp_jammed_drag - DP_JAMMED_DRAG_STEP).max(10.0);
                            println!("DP jammed drag: {:.1}", self.dp_jammed_drag);
                        }
                        PhysicalKey::Code(KeyCode::Digit8) => {
                            self.dp_jammed_drag = (self.dp_jammed_drag + DP_JAMMED_DRAG_STEP).min(100.0);
                            println!("DP jammed drag: {:.1}", self.dp_jammed_drag);
                        }
                        PhysicalKey::Code(KeyCode::KeyC) => {
                            self.clear_water();
                        }
                        PhysicalKey::Code(KeyCode::KeyX) => {
                            self.clear_sediment();
                        }
                        PhysicalKey::Code(KeyCode::KeyZ) => {
                            self.clear_all_particles();
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
                if button == MouseButton::Right {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed {
                        self.last_mouse_pos = None;
                    }
                }
                if button == MouseButton::Left && state == ElementState::Pressed {
                    if let Some((x, y)) = self.cursor_pos {
                        self.handle_click(x, y);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = Some((position.x as f32, position.y as f32));
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

// Screen-Space Fluid Rendering Shaders

/// Water depth pass - render particles as spherical depth splats
const WATER_DEPTH_SHADER: &str = r#"
struct WaterUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_radius: f32,
    screen_size: vec2<f32>,
    near: f32,
    far: f32,
}

@group(0) @binding(0) var<uniform> uniforms: WaterUniforms;

struct VertexInput {
    @location(0) vertex: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Billboard facing camera
    let to_camera = normalize(uniforms.camera_pos - in.position);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, to_camera));
    let up = cross(to_camera, right);

    let world_pos = in.position + right * in.vertex.x * uniforms.particle_radius
                                + up * in.vertex.y * uniforms.particle_radius;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = in.vertex;  // -1 to 1 range
    out.view_pos = (uniforms.view * vec4<f32>(in.position, 1.0)).xyz;
    return out;
}

struct FragmentOutput {
    @builtin(frag_depth) frag_depth: f32,
    @location(0) depth_thickness: vec4<f32>,  // .r = depth, .g = thickness
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // Calculate sphere depth: uv is in -1..1 range
    let dist_sq = dot(in.uv, in.uv);
    if (dist_sq > 1.0) {
        discard;
    }

    // Sphere surface offset in view space (pointing towards camera = positive z in view)
    let z_offset = sqrt(1.0 - dist_sq) * uniforms.particle_radius;

    // Adjust view-space z (view space z is negative into screen)
    let adjusted_view_z = in.view_pos.z + z_offset;

    // Project to get proper depth
    let adjusted_view_pos = vec4<f32>(in.view_pos.xy, adjusted_view_z, 1.0);
    let clip_pos = uniforms.proj * adjusted_view_pos;
    let ndc_depth = clip_pos.z / clip_pos.w;

    // Convert from NDC (-1 to 1) to depth buffer (0 to 1)
    let depth = ndc_depth * 0.5 + 0.5;

    // Thickness for refraction color
    let thickness = sqrt(1.0 - dist_sq) * 2.0;  // 0 at edge, 2 at center

    var out: FragmentOutput;
    out.frag_depth = depth;
    out.depth_thickness = vec4<f32>(depth, thickness, 0.0, 1.0);  // Store depth in .r, thickness in .g
    return out;
}
"#;

/// Bilateral blur shader - smooths depth while preserving edges
const WATER_BLUR_SHADER: &str = r#"
struct BlurUniforms {
    texel_size: vec2<f32>,
    blur_radius: i32,
    depth_falloff: f32,
    direction: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: BlurUniforms;
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var tex_sampler: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(dims);
    let center_depth = textureSampleLevel(input_texture, tex_sampler, uv, 0.0).r;

    // Skip if no depth (background)
    if (center_depth >= 1.0 || center_depth <= 0.0) {
        textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(center_depth, 0.0, 0.0, 0.0));
        return;
    }

    var sum = 0.0;
    var weight_sum = 0.0;

    for (var i = -uniforms.blur_radius; i <= uniforms.blur_radius; i++) {
        let offset = vec2<f32>(f32(i)) * uniforms.direction * uniforms.texel_size;
        let sample_uv = uv + offset;
        let sample_depth = textureSampleLevel(input_texture, tex_sampler, sample_uv, 0.0).r;

        // Skip background samples
        if (sample_depth >= 1.0 || sample_depth <= 0.0) {
            continue;
        }

        // Spatial weight (Gaussian)
        let spatial = exp(-0.5 * f32(i * i) / f32(uniforms.blur_radius * uniforms.blur_radius));

        // Depth weight (bilateral term)
        let depth_diff = abs(sample_depth - center_depth);
        let depth_weight = exp(-depth_diff * depth_diff / (uniforms.depth_falloff * uniforms.depth_falloff));

        let weight = spatial * depth_weight;
        sum += sample_depth * weight;
        weight_sum += weight;
    }

    let result = select(center_depth, sum / weight_sum, weight_sum > 0.0);
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(result, 0.0, 0.0, 0.0));
}
"#;

/// Water composite shader - final water surface rendering
const WATER_COMPOSITE_SHADER: &str = r#"
struct WaterUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    particle_radius: f32,
    screen_size: vec2<f32>,
    near: f32,
    far: f32,
}

@group(0) @binding(0) var<uniform> uniforms: WaterUniforms;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var thickness_texture: texture_2d<f32>;
@group(0) @binding(3) var scene_depth_texture: texture_depth_2d;
@group(0) @binding(4) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = positions[vertex_index] * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;  // Flip Y for texture coordinates
    return out;
}

fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    var view_pos = uniforms.inv_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

fn calculate_normal(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let texel = 1.0 / uniforms.screen_size;

    // Sample neighboring depths
    let depth_l = textureSampleLevel(depth_texture, tex_sampler, uv - vec2<f32>(texel.x, 0.0), 0.0).r;
    let depth_r = textureSampleLevel(depth_texture, tex_sampler, uv + vec2<f32>(texel.x, 0.0), 0.0).r;
    let depth_t = textureSampleLevel(depth_texture, tex_sampler, uv - vec2<f32>(0.0, texel.y), 0.0).r;
    let depth_b = textureSampleLevel(depth_texture, tex_sampler, uv + vec2<f32>(0.0, texel.y), 0.0).r;

    // Reconstruct view-space positions
    let pos_c = reconstruct_position(uv, depth);
    let pos_l = reconstruct_position(uv - vec2<f32>(texel.x, 0.0), depth_l);
    let pos_r = reconstruct_position(uv + vec2<f32>(texel.x, 0.0), depth_r);
    let pos_t = reconstruct_position(uv - vec2<f32>(0.0, texel.y), depth_t);
    let pos_b = reconstruct_position(uv + vec2<f32>(0.0, texel.y), depth_b);

    // Calculate derivatives, preferring closer samples
    var ddx: vec3<f32>;
    if (abs(depth_l - depth) < abs(depth_r - depth)) {
        ddx = pos_c - pos_l;
    } else {
        ddx = pos_r - pos_c;
    }

    var ddy: vec3<f32>;
    if (abs(depth_t - depth) < abs(depth_b - depth)) {
        ddy = pos_c - pos_t;
    } else {
        ddy = pos_b - pos_c;
    }

    return normalize(cross(ddx, ddy));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let water_depth = textureSampleLevel(depth_texture, tex_sampler, in.uv, 0.0).r;
    let scene_depth = textureSampleLevel(scene_depth_texture, tex_sampler, in.uv, 0.0);

    // No water or water behind scene
    if (water_depth >= 1.0 || water_depth <= 0.0 || water_depth > scene_depth) {
        discard;
    }

    // Reconstruct normal from depth
    let normal = calculate_normal(in.uv, water_depth);

    // View direction (in view space, looking down -Z)
    let view_pos = reconstruct_position(in.uv, water_depth);
    let view_dir = normalize(-view_pos);

    // Fresnel effect
    let fresnel_base = 0.02;
    let fresnel = fresnel_base + (1.0 - fresnel_base) * pow(1.0 - max(dot(normal, view_dir), 0.0), 5.0);

    // Water color based on thickness - more saturated blue
    // Thickness is stored in .g of the depth texture (same texture, blurred together)
    let thickness = textureSampleLevel(depth_texture, tex_sampler, in.uv, 0.0).g;
    let shallow_color = vec3<f32>(0.2, 0.5, 0.85);  // More saturated blue
    let deep_color = vec3<f32>(0.05, 0.2, 0.5);     // Deeper blue
    let water_color = mix(shallow_color, deep_color, saturate(thickness * 0.3));

    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let light_view = (uniforms.view * vec4<f32>(light_dir, 0.0)).xyz;
    let diffuse = max(dot(normal, light_view), 0.0) * 0.4 + 0.6;

    // Specular highlight
    let half_vec = normalize(light_view + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), 32.0);

    // Combine - less fresnel to keep color more visible
    let lit_color = water_color * diffuse + vec3<f32>(1.0) * specular * 0.4;
    let final_color = mix(lit_color, vec3<f32>(0.7, 0.85, 1.0), fresnel * 0.2);

    return vec4<f32>(final_color, 0.75);  // Semi-transparent
}
"#;

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
