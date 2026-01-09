//! GPU-accelerated Particle-to-Grid (P2G) transfer for 3D FLIP simulation.
//!
//! This module implements P2G for APIC-FLIP using 3D compute shaders with atomic operations.
//! WebGPU/wgpu only supports atomicAdd for i32, so we encode floats as fixed-point:
//!   f32 * SCALE → i32 (scatter), then i32 / SCALE → f32 (divide)
//!
//! Two compute passes:
//! 1. Scatter: Each particle atomically adds its momentum contribution to grid nodes
//! 2. Divide: Each grid node divides accumulated momentum by weight to get velocity
//!
//! Grid layout (MAC staggered):
//! - U velocities: (width+1) x height x depth
//! - V velocities: width x (height+1) x depth
//! - W velocities: width x height x (depth+1)

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Fixed-point scale factor (must match shader)
#[allow(dead_code)]
const SCALE: f32 = 1_000_000.0;

/// Parameters for P2G 3D compute shaders
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct P2gParams3D {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    include_sediment: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-based Particle-to-Grid transfer for 3D simulation
pub struct GpuP2g3D {
    width: u32,
    height: u32,
    depth: u32,
    include_sediment: bool,

    // Particle buffers (uploaded each frame)
    positions_buffer: Arc<wgpu::Buffer>,
    velocities_buffer: Arc<wgpu::Buffer>,
    densities_buffer: Arc<wgpu::Buffer>,
    // C matrix stored as 3 separate vec3 columns
    c_col0_buffer: Arc<wgpu::Buffer>,
    c_col1_buffer: Arc<wgpu::Buffer>,
    c_col2_buffer: Arc<wgpu::Buffer>,

    // Grid accumulator buffers (atomic<i32>)
    u_sum_buffer: wgpu::Buffer,
    u_weight_buffer: wgpu::Buffer,
    v_sum_buffer: wgpu::Buffer,
    v_weight_buffer: wgpu::Buffer,
    w_sum_buffer: wgpu::Buffer,
    w_weight_buffer: wgpu::Buffer,

    // Output grid buffers (f32) - shared with pressure solver
    pub grid_u_buffer: wgpu::Buffer,
    pub grid_v_buffer: wgpu::Buffer,
    pub grid_w_buffer: wgpu::Buffer,

    // Particle count per cell (atomic i32) - for density projection
    pub particle_count_buffer: wgpu::Buffer,
    // Sediment count per cell (atomic i32) - for sediment fraction
    pub sediment_count_buffer: wgpu::Buffer,

    // Staging buffers for readback
    grid_u_staging: wgpu::Buffer,
    grid_v_staging: wgpu::Buffer,
    grid_w_staging: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,

    // Compute pipelines
    scatter_pipeline: wgpu::ComputePipeline,
    divide_u_pipeline: wgpu::ComputePipeline,
    divide_v_pipeline: wgpu::ComputePipeline,
    divide_w_pipeline: wgpu::ComputePipeline,

    // Bind groups
    scatter_bind_group: wgpu::BindGroup,
    divide_bind_group: wgpu::BindGroup,

    // Current capacity
    max_particles: usize,

    // Workgroup sizes
    scatter_workgroup_size: u32,
}

impl GpuP2g3D {
    /// Create a new GPU P2G 3D solver
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        max_particles: usize,
        include_sediment: bool,
        positions_buffer: Arc<wgpu::Buffer>,
        velocities_buffer: Arc<wgpu::Buffer>,
        densities_buffer: Arc<wgpu::Buffer>,
        c_col0_buffer: Arc<wgpu::Buffer>,
        c_col1_buffer: Arc<wgpu::Buffer>,
        c_col2_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;

        // Create shader modules
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G 3D Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_scatter_3d.wgsl").into()),
        });

        let divide_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G 3D Divide Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_divide_3d.wgsl").into()),
        });

        // Create grid accumulator buffers (atomic i32)
        let u_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D U Sum"),
            size: (u_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D U Weight"),
            size: (u_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let v_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D V Sum"),
            size: (v_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let v_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D V Weight"),
            size: (v_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let w_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D W Sum"),
            size: (w_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let w_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D W Weight"),
            size: (w_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output grid buffers (f32)
        let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Grid U"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Grid V"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_w_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Grid W"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create particle count per cell buffer (for density projection)
        let cell_count = (width * height * depth) as usize;
        let particle_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Particle Count"),
            size: (cell_count * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sediment_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Sediment Count"),
            size: (cell_count * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffers for readback
        let grid_u_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Grid U Staging"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Grid V Staging"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_w_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Grid W Staging"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G 3D Params"),
            size: std::mem::size_of::<P2gParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create scatter bind group layout
        let scatter_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("P2G 3D Scatter Bind Group Layout"),
            entries: &[
                // 0: params (uniform)
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
                // 1: positions (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: velocities (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: c_col0 (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: c_col1 (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: c_col2 (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: u_sum (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: u_weight (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: v_sum (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: v_weight (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 10: w_sum (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 11: w_weight (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 12: particle_count (read_write atomic) - for density projection
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 13: densities (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 14: sediment_count (read_write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create divide bind group layout
        let divide_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("P2G 3D Divide Bind Group Layout"),
            entries: &[
                // 0: params (uniform)
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
                // 1-6: sum/weight pairs (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7-9: output grids (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create scatter bind group
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G 3D Scatter Bind Group"),
            layout: &scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_col0_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: c_col1_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: c_col2_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: u_sum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: u_weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: v_sum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: v_weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: w_sum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: w_weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: particle_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: densities_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: sediment_count_buffer.as_entire_binding() },
            ],
        });

        // Create divide bind group
        let divide_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G 3D Divide Bind Group"),
            layout: &divide_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: u_sum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_sum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: v_weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: w_sum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: w_weight_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: grid_w_buffer.as_entire_binding() },
            ],
        });

        // Create pipelines
        let scatter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("P2G 3D Scatter Pipeline Layout"),
            bind_group_layouts: &[&scatter_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G 3D Scatter Pipeline"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_shader,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        let divide_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("P2G 3D Divide Pipeline Layout"),
            bind_group_layouts: &[&divide_bind_group_layout],
            push_constant_ranges: &[],
        });

        let divide_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G 3D Divide U Pipeline"),
            layout: Some(&divide_pipeline_layout),
            module: &divide_shader,
            entry_point: Some("divide_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let divide_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G 3D Divide V Pipeline"),
            layout: Some(&divide_pipeline_layout),
            module: &divide_shader,
            entry_point: Some("divide_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let divide_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G 3D Divide W Pipeline"),
            layout: Some(&divide_pipeline_layout),
            module: &divide_shader,
            entry_point: Some("divide_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            height,
            depth,
            include_sediment,
            positions_buffer,
            velocities_buffer,
            densities_buffer,
            c_col0_buffer,
            c_col1_buffer,
            c_col2_buffer,
            u_sum_buffer,
            u_weight_buffer,
            v_sum_buffer,
            v_weight_buffer,
            w_sum_buffer,
            w_weight_buffer,
            grid_u_buffer,
            grid_v_buffer,
            grid_w_buffer,
            particle_count_buffer,
            sediment_count_buffer,
            grid_u_staging,
            grid_v_staging,
            grid_w_staging,
            params_buffer,
            scatter_pipeline,
            divide_u_pipeline,
            divide_v_pipeline,
            divide_w_pipeline,
            scatter_bind_group,
            divide_bind_group,
            max_particles,
            scatter_workgroup_size: 256,
        }
    }

    /// Upload particle data to GPU
    ///
    /// Takes positions, velocities, and C matrices from sim3d particles.
    pub fn upload_particles(
        &self,
        queue: &wgpu::Queue,
        positions: &[glam::Vec3],
        velocities: &[glam::Vec3],
        densities: &[f32],
        c_matrices: &[glam::Mat3],
        cell_size: f32,
    ) -> u32 {
        let particle_count = positions.len().min(self.max_particles) as u32;
        let density_count = densities.len().min(self.max_particles);

        // Convert to padded vec4 format for GPU
        let positions_padded: Vec<[f32; 4]> = positions.iter()
            .take(particle_count as usize)
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();

        let velocities_padded: Vec<[f32; 4]> = velocities.iter()
            .take(particle_count as usize)
            .map(|v| [v.x, v.y, v.z, 0.0])
            .collect();

        // Extract C matrix columns
        let c_col0: Vec<[f32; 4]> = c_matrices.iter()
            .take(particle_count as usize)
            .map(|c| [c.x_axis.x, c.x_axis.y, c.x_axis.z, 0.0])
            .collect();

        let c_col1: Vec<[f32; 4]> = c_matrices.iter()
            .take(particle_count as usize)
            .map(|c| [c.y_axis.x, c.y_axis.y, c.y_axis.z, 0.0])
            .collect();

        let c_col2: Vec<[f32; 4]> = c_matrices.iter()
            .take(particle_count as usize)
            .map(|c| [c.z_axis.x, c.z_axis.y, c.z_axis.z, 0.0])
            .collect();

        // Upload to GPU
        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions_padded));
        queue.write_buffer(&self.velocities_buffer, 0, bytemuck::cast_slice(&velocities_padded));
        queue.write_buffer(&self.c_col0_buffer, 0, bytemuck::cast_slice(&c_col0));
        queue.write_buffer(&self.c_col1_buffer, 0, bytemuck::cast_slice(&c_col1));
        queue.write_buffer(&self.c_col2_buffer, 0, bytemuck::cast_slice(&c_col2));
        if density_count > 0 {
            queue.write_buffer(&self.densities_buffer, 0, bytemuck::cast_slice(&densities[..density_count]));
        }

        self.prepare(queue, particle_count, cell_size);

        particle_count
    }

    pub fn prepare(&self, queue: &wgpu::Queue, particle_count: u32, cell_size: f32) {
        let params = P2gParams3D {
            cell_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            include_sediment: if self.include_sediment { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        self.clear_accumulators(queue);
    }

    fn clear_accumulators(&self, queue: &wgpu::Queue) {
        let u_size = ((self.width + 1) * self.height * self.depth) as usize;
        let v_size = (self.width * (self.height + 1) * self.depth) as usize;
        let w_size = (self.width * self.height * (self.depth + 1)) as usize;
        let cell_count = (self.width * self.height * self.depth) as usize;

        queue.write_buffer(&self.u_sum_buffer, 0, &vec![0u8; u_size * 4]);
        queue.write_buffer(&self.u_weight_buffer, 0, &vec![0u8; u_size * 4]);
        queue.write_buffer(&self.v_sum_buffer, 0, &vec![0u8; v_size * 4]);
        queue.write_buffer(&self.v_weight_buffer, 0, &vec![0u8; v_size * 4]);
        queue.write_buffer(&self.w_sum_buffer, 0, &vec![0u8; w_size * 4]);
        queue.write_buffer(&self.w_weight_buffer, 0, &vec![0u8; w_size * 4]);
        queue.write_buffer(&self.particle_count_buffer, 0, &vec![0u8; cell_count * 4]);
        queue.write_buffer(&self.sediment_count_buffer, 0, &vec![0u8; cell_count * 4]);
    }

    /// Encode P2G compute passes into command encoder
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, particle_count: u32) {
        // Scatter pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G 3D Scatter Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.scatter_bind_group, &[]);
            let workgroups = (particle_count + self.scatter_workgroup_size - 1) / self.scatter_workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Divide U pass: (width+1) x height x depth with workgroup (8, 8, 4)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G 3D Divide U Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_u_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Divide V pass: width x (height+1) x depth
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G 3D Divide V Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_v_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Divide W pass: width x height x (depth+1)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G 3D Divide W Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_w_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 1 + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
    }

    /// Download grid velocities from GPU
    pub fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid_u: &mut [f32],
        grid_v: &mut [f32],
        grid_w: &mut [f32],
    ) {
        let u_size = ((self.width + 1) * self.height * self.depth) as usize;
        let v_size = (self.width * (self.height + 1) * self.depth) as usize;
        let w_size = (self.width * self.height * (self.depth + 1)) as usize;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("P2G 3D Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.grid_u_buffer, 0, &self.grid_u_staging, 0, (u_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.grid_v_buffer, 0, &self.grid_v_staging, 0, (v_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.grid_w_buffer, 0, &self.grid_w_staging, 0, (w_size * 4) as u64);

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read U
        Self::read_staging_buffer(device, &self.grid_u_staging, grid_u);
        Self::read_staging_buffer(device, &self.grid_v_staging, grid_v);
        Self::read_staging_buffer(device, &self.grid_w_staging, grid_w);
    }

    fn read_staging_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, output: &mut [f32]) {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            output.copy_from_slice(bytemuck::cast_slice(&data));
        }
        buffer.unmap();
    }

    /// Grid buffer sizes
    pub fn grid_sizes(&self) -> (usize, usize, usize) {
        let u_size = ((self.width + 1) * self.height * self.depth) as usize;
        let v_size = (self.width * (self.height + 1) * self.depth) as usize;
        let w_size = (self.width * self.height * (self.depth + 1)) as usize;
        (u_size, v_size, w_size)
    }
}
