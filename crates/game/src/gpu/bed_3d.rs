//! GPU bed update + probe stats for 3D sediment scenes.
//!
//! This module keeps bed height updates and probe diagnostics on the GPU so
//! large particle readbacks are not required every frame.

use bytemuck::{Pod, Zeroable};
use std::sync::{mpsc, Arc};
use wgpu::util::DeviceExt;

pub const PROBE_STAT_SCALE: f32 = 1000.0;
pub const PROBE_MATERIAL_STRIDE: usize = 10;
pub const PROBE_ZONE_STRIDE: usize = PROBE_MATERIAL_STRIDE * 2;
pub const PROBE_ZONE_RIFFLE: usize = 0;
pub const PROBE_ZONE_DOWNSTREAM: usize = 1;
pub const PROBE_THROUGHPUT_OFFSET: usize = PROBE_ZONE_STRIDE * 2;
pub const PROBE_STAT_BUFFER_LEN: usize = 64;

pub const PROBE_STAT_COUNT_IDX: usize = 0;
pub const PROBE_STAT_SUM_Y_IDX: usize = 1;
pub const PROBE_STAT_MAX_Y_IDX: usize = 2;
pub const PROBE_STAT_SUM_VY_IDX: usize = 3;
pub const PROBE_STAT_SDF_NEG_IDX: usize = 4;
pub const PROBE_STAT_BELOW_BED_IDX: usize = 5;
pub const PROBE_STAT_ABOVE_BED_IDX: usize = 6;
pub const PROBE_STAT_UP_IDX: usize = 7;
pub const PROBE_STAT_SUM_OFFSET_IDX: usize = 8;
pub const PROBE_STAT_MAX_OFFSET_IDX: usize = 9;

pub const PROBE_THROUGHPUT_TOTAL_IDX: usize = 0;
pub const PROBE_THROUGHPUT_UPSTREAM_IDX: usize = 1;
pub const PROBE_THROUGHPUT_AT_RIFFLE_IDX: usize = 2;
pub const PROBE_THROUGHPUT_DOWNSTREAM_IDX: usize = 3;
pub const PROBE_THROUGHPUT_MAX_X_IDX: usize = 4;
pub const PROBE_THROUGHPUT_MAX_Y_IDX: usize = 5;
pub const PROBE_THROUGHPUT_LOFTED_IDX: usize = 6;

#[derive(Clone, Copy)]
pub struct GpuBedParams {
    pub dt: f32,
    pub sample_height: f32,
    pub bed_air_margin: f32,
    pub loft_height: f32,
    pub riffle_min_i: i32,
    pub riffle_max_i: i32,
    pub downstream_min_i: i32,
    pub downstream_max_i: i32,
    pub riffle_start_x: f32,
    pub riffle_end_x: f32,
    pub downstream_x: f32,
    pub bed_friction: f32,
    pub sediment_rel_density: f32,
    pub water_density: f32,
    pub sediment_grain_diameter: f32,
    pub shields_critical: f32,
    pub shields_smooth: f32,
    pub bedload_coeff: f32,
    pub entrainment_coeff: f32,
    pub sediment_settling_velocity: f32,
    pub bed_porosity: f32,
    pub max_bed_height: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BedStatsParams3D {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    sample_height: f32,
    bed_air_margin: f32,
    loft_height: f32,
    riffle_min_i: i32,
    riffle_max_i: i32,
    downstream_min_i: i32,
    downstream_max_i: i32,
    riffle_start_x: f32,
    riffle_end_x: f32,
    downstream_x: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BedFluxParams3D {
    width: u32,
    depth: u32,
    _pad0: [u32; 2],
    cell_size: f32,
    dt: f32,
    bed_friction: f32,
    sediment_rel_density: f32,
    water_density: f32,
    sediment_grain_diameter: f32,
    shields_critical: f32,
    shields_smooth: f32,
    bedload_coeff: f32,
    entrainment_coeff: f32,
    sediment_settling_velocity: f32,
    bed_porosity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BedUpdateParams3D {
    width: u32,
    depth: u32,
    _pad0: [u32; 2],
    cell_size: f32,
    dt: f32,
    bed_porosity: f32,
    max_bed_height: f32,
}

pub struct GpuBed3D {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
    column_count: usize,
    bed_water_count_buffer: wgpu::Buffer,
    bed_sediment_count_buffer: wgpu::Buffer,
    bed_water_vel_sum_buffer: wgpu::Buffer,
    bed_flux_x_buffer: wgpu::Buffer,
    bed_flux_z_buffer: wgpu::Buffer,
    bed_desired_delta_buffer: wgpu::Buffer,
    bed_base_height_buffer: wgpu::Buffer,
    bed_height_buffer: Arc<wgpu::Buffer>,
    probe_stats_buffer: wgpu::Buffer,
    probe_stats_staging: wgpu::Buffer,
    bed_height_staging: wgpu::Buffer,
    bed_stats_pipeline: wgpu::ComputePipeline,
    bed_stats_bind_group: wgpu::BindGroup,
    bed_stats_params_buffer: wgpu::Buffer,
    bed_flux_pipeline: wgpu::ComputePipeline,
    bed_flux_bind_group: wgpu::BindGroup,
    bed_flux_params_buffer: wgpu::Buffer,
    bed_update_pipeline: wgpu::ComputePipeline,
    bed_update_bind_group: wgpu::BindGroup,
    bed_update_params_buffer: wgpu::Buffer,
}

impl GpuBed3D {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        depth: u32,
        cell_size: f32,
        positions_buffer: &wgpu::Buffer,
        velocities_buffer: &wgpu::Buffer,
        densities_buffer: &wgpu::Buffer,
        sdf_buffer: &wgpu::Buffer,
        bed_height_buffer: Arc<wgpu::Buffer>,
        bed_base_height: &[f32],
    ) -> Self {
        let column_count = (width * depth) as usize;
        assert_eq!(
            bed_base_height.len(),
            column_count,
            "bed base height size mismatch: got {}, expected {}",
            bed_base_height.len(),
            column_count
        );

        let bed_stats_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bed Stats 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bed_stats_3d.wgsl").into()),
        });

        let bed_flux_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bed Flux 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bed_flux_3d.wgsl").into()),
        });

        let bed_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bed Update 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bed_update_3d.wgsl").into()),
        });

        let bed_stats_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Stats Params 3D"),
            size: std::mem::size_of::<BedStatsParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_flux_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Flux Params 3D"),
            size: std::mem::size_of::<BedFluxParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_update_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Update Params 3D"),
            size: std::mem::size_of::<BedUpdateParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_water_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Water Count"),
            size: (column_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_sediment_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Sediment Count"),
            size: (column_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_water_vel_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Water Vel Sum"),
            size: (column_count * 3 * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_flux_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Flux X"),
            size: (column_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bed_flux_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Flux Z"),
            size: (column_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bed_desired_delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Desired Delta"),
            size: (column_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bed_base_height_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bed Base Height 3D"),
            contents: bytemuck::cast_slice(bed_base_height),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let probe_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Probe Stats"),
            size: (PROBE_STAT_BUFFER_LEN * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let probe_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Probe Stats Staging"),
            size: (PROBE_STAT_BUFFER_LEN * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_height_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Height Staging"),
            size: (column_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_stats_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bed Stats 3D Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        let bed_stats_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bed Stats 3D Bind Group"),
            layout: &bed_stats_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bed_stats_params_buffer.as_entire_binding(),
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
                    resource: densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bed_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bed_water_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bed_sediment_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bed_water_vel_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: probe_stats_buffer.as_entire_binding(),
                },
            ],
        });

        let bed_stats_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bed Stats 3D Pipeline Layout"),
            bind_group_layouts: &[&bed_stats_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bed_stats_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bed Stats 3D Pipeline"),
            layout: Some(&bed_stats_pipeline_layout),
            module: &bed_stats_shader,
            entry_point: Some("bed_stats"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bed_flux_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bed Flux 3D Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        let bed_flux_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bed Flux 3D Bind Group"),
            layout: &bed_flux_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bed_flux_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bed_water_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bed_sediment_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bed_water_vel_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bed_base_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bed_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bed_flux_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bed_flux_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bed_desired_delta_buffer.as_entire_binding(),
                },
            ],
        });

        let bed_flux_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bed Flux 3D Pipeline Layout"),
            bind_group_layouts: &[&bed_flux_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bed_flux_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bed Flux 3D Pipeline"),
            layout: Some(&bed_flux_pipeline_layout),
            module: &bed_flux_shader,
            entry_point: Some("bed_flux"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bed_update_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bed Update 3D Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bed_update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bed Update 3D Bind Group"),
            layout: &bed_update_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bed_update_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bed_flux_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bed_flux_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bed_desired_delta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bed_base_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bed_height_buffer.as_entire_binding(),
                },
            ],
        });

        let bed_update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bed Update 3D Pipeline Layout"),
            bind_group_layouts: &[&bed_update_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bed_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bed Update 3D Pipeline"),
            layout: Some(&bed_update_pipeline_layout),
            module: &bed_update_shader,
            entry_point: Some("bed_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        queue.write_buffer(bed_height_buffer.as_ref(), 0, bytemuck::cast_slice(bed_base_height));

        Self {
            width,
            height,
            depth,
            cell_size,
            column_count,
            bed_water_count_buffer,
            bed_sediment_count_buffer,
            bed_water_vel_sum_buffer,
            bed_flux_x_buffer,
            bed_flux_z_buffer,
            bed_desired_delta_buffer,
            bed_base_height_buffer,
            bed_height_buffer,
            probe_stats_buffer,
            probe_stats_staging,
            bed_height_staging,
            bed_stats_pipeline,
            bed_stats_bind_group,
            bed_stats_params_buffer,
            bed_flux_pipeline,
            bed_flux_bind_group,
            bed_flux_params_buffer,
            bed_update_pipeline,
            bed_update_bind_group,
            bed_update_params_buffer,
        }
    }

    pub fn reset_bed(&self, queue: &wgpu::Queue, bed_base_height: &[f32]) {
        assert_eq!(
            bed_base_height.len(),
            self.column_count,
            "bed base height size mismatch: got {}, expected {}",
            bed_base_height.len(),
            self.column_count
        );
        queue.write_buffer(&self.bed_base_height_buffer, 0, bytemuck::cast_slice(bed_base_height));
        queue.write_buffer(&self.bed_height_buffer, 0, bytemuck::cast_slice(bed_base_height));
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        params: &GpuBedParams,
    ) {
        let bed_stats_params = BedStatsParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            cell_size: self.cell_size,
            sample_height: params.sample_height,
            bed_air_margin: params.bed_air_margin,
            loft_height: params.loft_height,
            riffle_min_i: params.riffle_min_i,
            riffle_max_i: params.riffle_max_i,
            downstream_min_i: params.downstream_min_i,
            downstream_max_i: params.downstream_max_i,
            riffle_start_x: params.riffle_start_x,
            riffle_end_x: params.riffle_end_x,
            downstream_x: params.downstream_x,
            _pad0: 0.0,
        };
        queue.write_buffer(&self.bed_stats_params_buffer, 0, bytemuck::bytes_of(&bed_stats_params));

        let bed_flux_params = BedFluxParams3D {
            width: self.width,
            depth: self.depth,
            _pad0: [0; 2],
            cell_size: self.cell_size,
            dt: params.dt,
            bed_friction: params.bed_friction,
            sediment_rel_density: params.sediment_rel_density,
            water_density: params.water_density,
            sediment_grain_diameter: params.sediment_grain_diameter,
            shields_critical: params.shields_critical,
            shields_smooth: params.shields_smooth,
            bedload_coeff: params.bedload_coeff,
            entrainment_coeff: params.entrainment_coeff,
            sediment_settling_velocity: params.sediment_settling_velocity,
            bed_porosity: params.bed_porosity,
        };
        queue.write_buffer(&self.bed_flux_params_buffer, 0, bytemuck::bytes_of(&bed_flux_params));

        let bed_update_params = BedUpdateParams3D {
            width: self.width,
            depth: self.depth,
            _pad0: [0; 2],
            cell_size: self.cell_size,
            dt: params.dt,
            bed_porosity: params.bed_porosity,
            max_bed_height: params.max_bed_height,
        };
        queue.write_buffer(&self.bed_update_params_buffer, 0, bytemuck::bytes_of(&bed_update_params));

        let column_count = (self.width * self.depth) as u32;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bed 3D Encoder"),
        });
        encoder.clear_buffer(&self.bed_water_count_buffer, 0, None);
        encoder.clear_buffer(&self.bed_sediment_count_buffer, 0, None);
        encoder.clear_buffer(&self.bed_water_vel_sum_buffer, 0, None);
        encoder.clear_buffer(&self.probe_stats_buffer, 0, None);

        if particle_count == 0 {
            queue.submit(std::iter::once(encoder.finish()));
            return;
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bed Stats 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bed_stats_pipeline);
            pass.set_bind_group(0, &self.bed_stats_bind_group, &[]);
            let workgroups = (particle_count + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bed Flux 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bed_flux_pipeline);
            pass.set_bind_group(0, &self.bed_flux_bind_group, &[]);
            let workgroups = (column_count + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bed Update 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bed_update_pipeline);
            pass.set_bind_group(0, &self.bed_update_bind_group, &[]);
            let workgroups = (column_count + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn read_bed_height(&self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.column_count,
            "bed height output size mismatch: got {}, expected {}",
            out.len(),
            self.column_count
        );
        let byte_size = (self.column_count * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bed Height Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.bed_height_buffer, 0, &self.bed_height_staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.bed_height_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = slice.get_mapped_range();
            let src: &[f32] = bytemuck::cast_slice(&data);
            out.copy_from_slice(&src[..self.column_count]);
        }
        self.bed_height_staging.unmap();
    }

    pub fn read_probe_stats(&self, device: &wgpu::Device, queue: &wgpu::Queue, out: &mut Vec<i32>) {
        let byte_size = (PROBE_STAT_BUFFER_LEN * std::mem::size_of::<i32>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bed Probe Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.probe_stats_buffer, 0, &self.probe_stats_staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.probe_stats_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = slice.get_mapped_range();
            let src: &[i32] = bytemuck::cast_slice(&data);
            out.clear();
            out.extend_from_slice(&src[..PROBE_STAT_BUFFER_LEN]);
        }
        self.probe_stats_staging.unmap();
    }
}
