//! GPU-accelerated Discrete Element Method (DEM) for sediment particles.
//!
//! Uses spatial hashing for O(n) collision detection:
//! 1. Clear bin counts
//! 2. Count particles per bin (GPU)
//! 3. Compute prefix sum for bin offsets (CPU - grid is small)
//! 4. Insert particles into sorted array (GPU)
//! 5. Compute DEM forces and integrate (GPU)

use bytemuck::{Pod, Zeroable};
use sim::particle::{ParticleMaterial, Particles};

use super::GpuContext;

/// Parameters for bin count/insert shaders
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BinParams {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
}

/// Parameters for bin clear shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ClearParams {
    grid_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Parameters for DEM forces shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DemParams {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
    dt: f32,
    gravity: f32,
    contact_stiffness: f32,
    damping_ratio: f32,
    friction_coeff: f32,
    velocity_damping: f32,
    sdf_width: u32,
    sdf_height: u32,
    water_level: f32,
    iteration: u32,  // 0 = first pass (gravity+integrate), 1+ = constraint-only
}

/// GPU-based DEM solver using spatial hashing
pub struct GpuDemSolver {
    grid_width: u32,
    grid_height: u32,
    grid_size: u32, // width * height
    max_particles: usize,

    // Particle buffers
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,
    radii_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    sleep_counters_buffer: wgpu::Buffer, // Frames of consecutive low velocity
    static_state_buffer: wgpu::Buffer,   // 0 = dynamic, 1 = static (frozen)

    // Spatial hash buffers
    bin_counts_buffer: wgpu::Buffer,
    bin_offsets_buffer: wgpu::Buffer,
    bin_counters_buffer: wgpu::Buffer, // Separate counter for insert phase
    sorted_indices_buffer: wgpu::Buffer,

    // SDF buffer
    sdf_buffer: wgpu::Buffer,

    // Staging buffers
    positions_staging: wgpu::Buffer,
    velocities_staging: wgpu::Buffer,
    bin_counts_staging: wgpu::Buffer,

    // Parameter buffers
    bin_params_buffer: wgpu::Buffer,
    clear_params_buffer: wgpu::Buffer,
    dem_params_buffer: wgpu::Buffer,

    // Pipelines
    bin_clear_pipeline: wgpu::ComputePipeline,
    bin_count_pipeline: wgpu::ComputePipeline,
    bin_insert_pipeline: wgpu::ComputePipeline,
    dem_forces_pipeline: wgpu::ComputePipeline,

    // Bind groups
    bin_clear_bind_group: wgpu::BindGroup,
    bin_count_bind_group: wgpu::BindGroup,
    bin_insert_bind_group: wgpu::BindGroup,
    dem_forces_bind_group: wgpu::BindGroup,

    // CPU-side bin offsets for prefix sum
    bin_offsets_cpu: Vec<u32>,
}

impl GpuDemSolver {
    pub fn new(gpu: &GpuContext, grid_width: u32, grid_height: u32, max_particles: usize) -> Self {
        Self::new_internal(&gpu.device, grid_width, grid_height, max_particles)
    }

    /// Create solver for headless testing (no GpuContext needed)
    pub fn new_headless(device: &wgpu::Device, _queue: &wgpu::Queue, grid_width: u32, grid_height: u32, max_particles: usize) -> Self {
        Self::new_internal(device, grid_width, grid_height, max_particles)
    }

    fn new_internal(device: &wgpu::Device, grid_width: u32, grid_height: u32, max_particles: usize) -> Self {
        let grid_size = grid_width * grid_height;

        // Create shaders
        let bin_clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DEM Bin Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dem_bin_clear.wgsl").into()),
        });

        let bin_count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DEM Bin Count Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dem_bin_count.wgsl").into()),
        });

        let bin_insert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DEM Bin Insert Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dem_bin_insert.wgsl").into()),
        });

        let dem_forces_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DEM Forces Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dem_forces.wgsl").into()),
        });

        // Create particle buffers
        let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Positions"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocities_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Velocities"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let radii_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Radii"),
            size: (max_particles * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Materials"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sleep_counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Sleep Counters"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let static_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Static State"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create spatial hash buffers (+1 for sentinel in offsets)
        let bin_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Bin Counts"),
            size: ((grid_size + 1) * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bin_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Bin Offsets"),
            size: ((grid_size + 1) * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bin_counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Bin Counters"),
            size: ((grid_size + 1) * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorted_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Sorted Indices"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create SDF buffer
        let sdf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM SDF"),
            size: (grid_size * std::mem::size_of::<f32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create staging buffers
        let positions_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Positions Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Velocities Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bin_counts_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Bin Counts Staging"),
            size: ((grid_size + 1) * std::mem::size_of::<u32>() as u32) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create parameter buffers
        let bin_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Bin Params"),
            size: std::mem::size_of::<BinParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clear_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Clear Params"),
            size: std::mem::size_of::<ClearParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dem_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEM Params"),
            size: std::mem::size_of::<DemParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layouts and pipelines
        // Bin Clear
        let bin_clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEM Bin Clear Layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bin_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEM Bin Clear Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DEM Bin Clear Pipeline Layout"),
                bind_group_layouts: &[&bin_clear_layout],
                push_constant_ranges: &[],
            })),
            module: &bin_clear_shader,
            entry_point: Some("bin_clear"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bin_clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEM Bin Clear Bind Group"),
            layout: &bin_clear_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clear_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bin_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Bin Count
        let bin_count_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEM Bin Count Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bin_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEM Bin Count Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DEM Bin Count Pipeline Layout"),
                bind_group_layouts: &[&bin_count_layout],
                push_constant_ranges: &[],
            })),
            module: &bin_count_shader,
            entry_point: Some("bin_count"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bin_count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEM Bin Count Bind Group"),
            layout: &bin_count_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bin_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Bin Insert
        let bin_insert_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEM Bin Insert Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let bin_insert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEM Bin Insert Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DEM Bin Insert Pipeline Layout"),
                bind_group_layouts: &[&bin_insert_layout],
                push_constant_ranges: &[],
            })),
            module: &bin_insert_shader,
            entry_point: Some("bin_insert"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bin_insert_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEM Bin Insert Bind Group"),
            layout: &bin_insert_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bin_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bin_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bin_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sorted_indices_buffer.as_entire_binding(),
                },
            ],
        });

        // DEM Forces
        let dem_forces_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEM Forces Layout"),
            entries: &[
                // 0: params
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
                // 1: positions (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: velocities (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: radii (read)
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
                // 4: materials (read)
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
                // 5: bin_offsets (read)
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
                // 6: sorted_indices (read)
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
                // 7: sdf_data (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: sleep_counters (read_write)
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
                // 9: static_state (read_write)
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

        let dem_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEM Forces Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DEM Forces Pipeline Layout"),
                bind_group_layouts: &[&dem_forces_layout],
                push_constant_ranges: &[],
            })),
            module: &dem_forces_shader,
            entry_point: Some("dem_forces"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dem_forces_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEM Forces Bind Group"),
            layout: &dem_forces_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dem_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bin_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sorted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: sleep_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: static_state_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            grid_width,
            grid_height,
            grid_size,
            max_particles,
            positions_buffer,
            velocities_buffer,
            radii_buffer,
            materials_buffer,
            sleep_counters_buffer,
            static_state_buffer,
            bin_counts_buffer,
            bin_offsets_buffer,
            bin_counters_buffer,
            sorted_indices_buffer,
            sdf_buffer,
            positions_staging,
            velocities_staging,
            bin_counts_staging,
            bin_params_buffer,
            clear_params_buffer,
            dem_params_buffer,
            bin_clear_pipeline,
            bin_count_pipeline,
            bin_insert_pipeline,
            dem_forces_pipeline,
            bin_clear_bind_group,
            bin_count_bind_group,
            bin_insert_bind_group,
            dem_forces_bind_group,
            bin_offsets_cpu: vec![0u32; (grid_size + 1) as usize],
        }
    }

    /// Upload SDF data (call when terrain changes)
    pub fn upload_sdf(&self, gpu: &GpuContext, sdf: &[f32]) {
        gpu.queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));
    }

    /// Upload SDF for headless testing
    pub fn upload_sdf_headless(&self, _device: &wgpu::Device, queue: &wgpu::Queue, sdf: &[f32]) {
        queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));
    }

    /// Execute GPU DEM for sediment particles
    pub fn execute(
        &mut self,
        gpu: &GpuContext,
        particles: &mut Particles,
        cell_size: f32,
        dt: f32,
        gravity: f32,
        water_level: f32, // -1.0 for no water
    ) {
        // Filter to sediment particles only
        let sediment_indices: Vec<usize> = particles
            .list
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_sediment())
            .map(|(i, _)| i)
            .collect();

        let particle_count = sediment_indices.len();
        if particle_count == 0 {
            return;
        }

        // Upload particle data
        self.upload_particles(gpu, particles, &sediment_indices, cell_size);

        // Upload parameters
        let bin_params = BinParams {
            cell_size,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            particle_count: particle_count as u32,
        };
        gpu.queue.write_buffer(&self.bin_params_buffer, 0, bytemuck::bytes_of(&bin_params));

        let clear_params = ClearParams {
            grid_size: self.grid_size + 1, // Include sentinel
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        gpu.queue.write_buffer(&self.clear_params_buffer, 0, bytemuck::bytes_of(&clear_params));

        let mut dem_params = DemParams {
            cell_size,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            particle_count: particle_count as u32,
            dt,
            gravity,
            contact_stiffness: 2000.0, // Reduced from 20000 for stability with dt=1/60
            damping_ratio: 0.7,
            friction_coeff: 0.6,
            velocity_damping: 0.98,
            sdf_width: self.grid_width,
            sdf_height: self.grid_height,
            water_level,
            iteration: 0, // First pass applies gravity
        };
        gpu.queue.write_buffer(&self.dem_params_buffer, 0, bytemuck::bytes_of(&dem_params));

        // Create command encoder
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Encoder"),
        });

        // Phase 1: Clear bin counts
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEM Bin Clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_clear_pipeline);
            pass.set_bind_group(0, &self.bin_clear_bind_group, &[]);
            let workgroups = (self.grid_size + 1 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Phase 2: Count particles per bin
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEM Bin Count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_count_pipeline);
            pass.set_bind_group(0, &self.bin_count_bind_group, &[]);
            let workgroups = (particle_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy bin counts to staging for CPU prefix sum
        encoder.copy_buffer_to_buffer(
            &self.bin_counts_buffer,
            0,
            &self.bin_counts_staging,
            0,
            ((self.grid_size + 1) * 4) as u64,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Read bin counts and compute prefix sum on CPU
        {
            let slice = self.bin_counts_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&data);

            // Exclusive prefix sum
            self.bin_offsets_cpu[0] = 0;
            for i in 0..self.grid_size as usize {
                self.bin_offsets_cpu[i + 1] = self.bin_offsets_cpu[i] + counts[i];
            }

            drop(data);
            self.bin_counts_staging.unmap();
        }

        // Upload bin offsets
        gpu.queue.write_buffer(&self.bin_offsets_buffer, 0, bytemuck::cast_slice(&self.bin_offsets_cpu));

        // Clear bin counters for insert phase
        gpu.queue.write_buffer(&self.bin_counters_buffer, 0, &vec![0u8; ((self.grid_size + 1) * 4) as usize]);

        // Create new encoder for remaining phases
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Encoder 2"),
        });

        // Phase 3: Insert particles into sorted array
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEM Bin Insert"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_insert_pipeline);
            pass.set_bind_group(0, &self.bin_insert_bind_group, &[]);
            let workgroups = (particle_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Submit bin insert phase before constraint loop
        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Phase 4: Compute DEM forces and integrate
        // Run multiple constraint iterations for pile stiffness:
        // - Iteration 0: Apply gravity + integrate + resolve collisions
        // - Iterations 1-3: Resolve collisions only (propagates pile resistance)
        const CONSTRAINT_ITERATIONS: u32 = 4;
        for iter in 0..CONSTRAINT_ITERATIONS {
            dem_params.iteration = iter;
            gpu.queue.write_buffer(&self.dem_params_buffer, 0, bytemuck::bytes_of(&dem_params));

            let mut iter_encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DEM Forces Iter"),
            });
            {
                let mut pass = iter_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("DEM Forces"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dem_forces_pipeline);
                pass.set_bind_group(0, &self.dem_forces_bind_group, &[]);
                let workgroups = (particle_count as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            gpu.queue.submit(std::iter::once(iter_encoder.finish()));
        }

        // Copy results to staging
        let pos_size = (particle_count * std::mem::size_of::<[f32; 2]>()) as u64;
        let vel_size = (particle_count * std::mem::size_of::<[f32; 2]>()) as u64;
        let mut copy_encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Copy"),
        });
        copy_encoder.copy_buffer_to_buffer(&self.positions_buffer, 0, &self.positions_staging, 0, pos_size);
        copy_encoder.copy_buffer_to_buffer(&self.velocities_buffer, 0, &self.velocities_staging, 0, vel_size);

        gpu.queue.submit(std::iter::once(copy_encoder.finish()));

        // Download results
        self.download_particles(gpu, particles, &sediment_indices);
    }

    /// Execute GPU DEM for headless testing
    pub fn execute_headless(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &mut Particles,
        cell_size: f32,
        dt: f32,
        gravity: f32,
        water_level: f32,
    ) {
        // Filter to sediment particles only
        let sediment_indices: Vec<usize> = particles
            .list
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_sediment())
            .map(|(i, _)| i)
            .collect();

        let particle_count = sediment_indices.len();
        if particle_count == 0 {
            return;
        }

        // Upload particle data
        self.upload_particles_headless(queue, particles, &sediment_indices, cell_size);

        // Upload parameters
        let bin_params = BinParams {
            cell_size,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            particle_count: particle_count as u32,
        };
        queue.write_buffer(&self.bin_params_buffer, 0, bytemuck::bytes_of(&bin_params));

        let clear_params = ClearParams {
            grid_size: self.grid_size + 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&self.clear_params_buffer, 0, bytemuck::bytes_of(&clear_params));

        let mut dem_params = DemParams {
            cell_size,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            particle_count: particle_count as u32,
            dt,
            gravity,
            contact_stiffness: 2000.0,
            damping_ratio: 0.7,
            friction_coeff: 0.6,
            velocity_damping: 0.98,
            sdf_width: self.grid_width,
            sdf_height: self.grid_height,
            water_level,
            iteration: 0,
        };
        queue.write_buffer(&self.dem_params_buffer, 0, bytemuck::bytes_of(&dem_params));

        // Phase 1: Clear bin counts
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Headless Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEM Bin Clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_clear_pipeline);
            pass.set_bind_group(0, &self.bin_clear_bind_group, &[]);
            let workgroups = (self.grid_size + 1 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Phase 2: Count particles per bin
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEM Bin Count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_count_pipeline);
            pass.set_bind_group(0, &self.bin_count_bind_group, &[]);
            let workgroups = (particle_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.bin_counts_buffer,
            0,
            &self.bin_counts_staging,
            0,
            ((self.grid_size + 1) * 4) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Read bin counts and compute prefix sum on CPU
        {
            let slice = self.bin_counts_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&data);

            self.bin_offsets_cpu[0] = 0;
            for i in 0..self.grid_size as usize {
                self.bin_offsets_cpu[i + 1] = self.bin_offsets_cpu[i] + counts[i];
            }

            drop(data);
            self.bin_counts_staging.unmap();
        }

        queue.write_buffer(&self.bin_offsets_buffer, 0, bytemuck::cast_slice(&self.bin_offsets_cpu));
        queue.write_buffer(&self.bin_counters_buffer, 0, &vec![0u8; ((self.grid_size + 1) * 4) as usize]);

        // Phase 3: Insert particles into sorted array
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Headless Encoder 2"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEM Bin Insert"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bin_insert_pipeline);
            pass.set_bind_group(0, &self.bin_insert_bind_group, &[]);
            let workgroups = (particle_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Phase 4: Compute DEM forces with multiple iterations
        const CONSTRAINT_ITERATIONS: u32 = 4;
        for iter in 0..CONSTRAINT_ITERATIONS {
            dem_params.iteration = iter;
            queue.write_buffer(&self.dem_params_buffer, 0, bytemuck::bytes_of(&dem_params));

            let mut iter_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DEM Forces Iter"),
            });
            {
                let mut pass = iter_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("DEM Forces"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dem_forces_pipeline);
                pass.set_bind_group(0, &self.dem_forces_bind_group, &[]);
                let workgroups = (particle_count as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(iter_encoder.finish()));
        }

        // Copy results to staging
        let pos_size = (particle_count * std::mem::size_of::<[f32; 2]>()) as u64;
        let vel_size = (particle_count * std::mem::size_of::<[f32; 2]>()) as u64;
        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEM Copy"),
        });
        copy_encoder.copy_buffer_to_buffer(&self.positions_buffer, 0, &self.positions_staging, 0, pos_size);
        copy_encoder.copy_buffer_to_buffer(&self.velocities_buffer, 0, &self.velocities_staging, 0, vel_size);
        queue.submit(std::iter::once(copy_encoder.finish()));

        // Download results
        self.download_particles_headless(device, particles, &sediment_indices);
    }

    fn upload_particles(&self, gpu: &GpuContext, particles: &Particles, indices: &[usize], cell_size: f32) {
        let mut positions: Vec<[f32; 2]> = Vec::with_capacity(indices.len());
        let mut velocities: Vec<[f32; 2]> = Vec::with_capacity(indices.len());
        let mut radii: Vec<f32> = Vec::with_capacity(indices.len());
        let mut materials: Vec<u32> = Vec::with_capacity(indices.len());

        for &i in indices {
            let p = &particles.list[i];
            positions.push([p.position.x, p.position.y]);
            velocities.push([p.velocity.x, p.velocity.y]);
            // Reduce collision radius to pack particles tighter visually
            radii.push(p.material.typical_diameter() * 0.35 * cell_size);
            materials.push(Self::material_to_u32(p.material));
        }

        gpu.queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions));
        gpu.queue.write_buffer(&self.velocities_buffer, 0, bytemuck::cast_slice(&velocities));
        gpu.queue.write_buffer(&self.radii_buffer, 0, bytemuck::cast_slice(&radii));
        gpu.queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(&materials));
    }

    fn upload_particles_headless(&self, queue: &wgpu::Queue, particles: &Particles, indices: &[usize], cell_size: f32) {
        let mut positions: Vec<[f32; 2]> = Vec::with_capacity(indices.len());
        let mut velocities: Vec<[f32; 2]> = Vec::with_capacity(indices.len());
        let mut radii: Vec<f32> = Vec::with_capacity(indices.len());
        let mut materials: Vec<u32> = Vec::with_capacity(indices.len());

        for &i in indices {
            let p = &particles.list[i];
            positions.push([p.position.x, p.position.y]);
            velocities.push([p.velocity.x, p.velocity.y]);
            radii.push(p.material.typical_diameter() * 0.35 * cell_size);
            materials.push(Self::material_to_u32(p.material));
        }

        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions));
        queue.write_buffer(&self.velocities_buffer, 0, bytemuck::cast_slice(&velocities));
        queue.write_buffer(&self.radii_buffer, 0, bytemuck::cast_slice(&radii));
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(&materials));
    }

    fn download_particles(&self, gpu: &GpuContext, particles: &mut Particles, indices: &[usize]) {
        let particle_count = indices.len();

        // Map and read positions
        {
            let slice = self.positions_staging.slice(..(particle_count * 8) as u64);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let positions: &[[f32; 2]] = bytemuck::cast_slice(&data);

            for (local_idx, &global_idx) in indices.iter().enumerate() {
                let p = &mut particles.list[global_idx];
                p.position.x = positions[local_idx][0];
                p.position.y = positions[local_idx][1];
            }

            drop(data);
            self.positions_staging.unmap();
        }

        // Map and read velocities
        {
            let slice = self.velocities_staging.slice(..(particle_count * 8) as u64);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let velocities: &[[f32; 2]] = bytemuck::cast_slice(&data);

            for (local_idx, &global_idx) in indices.iter().enumerate() {
                let p = &mut particles.list[global_idx];
                p.velocity.x = velocities[local_idx][0];
                p.velocity.y = velocities[local_idx][1];
            }

            drop(data);
            self.velocities_staging.unmap();
        }
    }

    fn download_particles_headless(&self, device: &wgpu::Device, particles: &mut Particles, indices: &[usize]) {
        let particle_count = indices.len();

        // Map and read positions
        {
            let slice = self.positions_staging.slice(..(particle_count * 8) as u64);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let positions: &[[f32; 2]] = bytemuck::cast_slice(&data);

            for (local_idx, &global_idx) in indices.iter().enumerate() {
                let p = &mut particles.list[global_idx];
                p.position.x = positions[local_idx][0];
                p.position.y = positions[local_idx][1];
            }

            drop(data);
            self.positions_staging.unmap();
        }

        // Map and read velocities
        {
            let slice = self.velocities_staging.slice(..(particle_count * 8) as u64);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let velocities: &[[f32; 2]] = bytemuck::cast_slice(&data);

            for (local_idx, &global_idx) in indices.iter().enumerate() {
                let p = &mut particles.list[global_idx];
                p.velocity.x = velocities[local_idx][0];
                p.velocity.y = velocities[local_idx][1];
            }

            drop(data);
            self.velocities_staging.unmap();
        }
    }

    fn material_to_u32(mat: ParticleMaterial) -> u32 {
        match mat {
            ParticleMaterial::Water => 0,
            ParticleMaterial::Mud => 1,
            ParticleMaterial::Sand => 2,
            ParticleMaterial::Magnetite => 3,
            ParticleMaterial::Gold => 4,
            ParticleMaterial::Gravel => 5,
        }
    }
}
