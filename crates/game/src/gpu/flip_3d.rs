//! GPU-accelerated 3D FLIP/APIC simulation.
//!
//! This module provides a complete GPU-based 3D fluid simulation that combines:
//! - P2G (Particle-to-Grid) with atomic scatter
//! - Pressure solve (Red-Black Gauss-Seidel)
//! - G2P (Grid-to-Particle) with FLIP/PIC blend
//!
//! The simulation maintains particle data on CPU but does all heavy computation on GPU.

use super::g2p_3d::GpuG2p3D;
use super::p2g_3d::GpuP2g3D;
use super::pressure_3d::GpuPressure3D;

use bytemuck::{Pod, Zeroable};

/// Gravity application parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GravityParams3D {
    width: u32,
    height: u32,
    depth: u32,
    gravity_dt: f32,
}

/// Flow acceleration parameters (for sluice downstream flow)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FlowParams3D {
    width: u32,
    height: u32,
    depth: u32,
    flow_accel_dt: f32,  // flow_accel * dt
}

/// Boundary condition parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BcParams3D {
    width: u32,
    height: u32,
    depth: u32,
    _pad: u32,
}

/// Density error computation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DensityErrorParams3D {
    width: u32,
    height: u32,
    depth: u32,
    rest_density: f32,  // Target particles per cell (~8 for typical FLIP)
    dt: f32,            // Timestep for scaling
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Density position grid parameters (first pass - grid-based position changes)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DensityPositionGridParams3D {
    width: u32,
    height: u32,
    depth: u32,
    dt: f32,
}

/// Density position correction parameters (blub grid-based trilinear sampling)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DensityCorrectionParams3D {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// SDF collision parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SdfCollisionParams3D {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-accelerated 3D FLIP simulation
pub struct GpuFlip3D {
    // Grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,

    // Sub-solvers
    p2g: GpuP2g3D,
    g2p: GpuG2p3D,
    pressure: GpuPressure3D,

    // Gravity shader
    gravity_pipeline: wgpu::ComputePipeline,
    gravity_bind_group: wgpu::BindGroup,
    gravity_params_buffer: wgpu::Buffer,

    // Flow acceleration shader (for sluice downstream flow)
    flow_pipeline: wgpu::ComputePipeline,
    flow_bind_group: wgpu::BindGroup,
    flow_params_buffer: wgpu::Buffer,

    // Boundary condition enforcement shaders
    bc_u_pipeline: wgpu::ComputePipeline,
    bc_v_pipeline: wgpu::ComputePipeline,
    bc_w_pipeline: wgpu::ComputePipeline,
    bc_bind_group: wgpu::BindGroup,
    bc_params_buffer: wgpu::Buffer,

    // Grid velocity backup for FLIP delta
    grid_u_old_buffer: wgpu::Buffer,
    grid_v_old_buffer: wgpu::Buffer,
    grid_w_old_buffer: wgpu::Buffer,

    // Density projection (Implicit Density Projection for volume conservation)
    // Phase 1: Compute density error
    density_error_pipeline: wgpu::ComputePipeline,
    density_error_bind_group: wgpu::BindGroup,
    density_error_params_buffer: wgpu::Buffer,
    // Phase 2: Compute position changes on grid (blub approach)
    density_position_grid_pipeline: wgpu::ComputePipeline,
    density_position_grid_bind_group: wgpu::BindGroup,
    density_position_grid_params_buffer: wgpu::Buffer,
    position_delta_x_buffer: wgpu::Buffer,  // Grid-based delta X
    position_delta_y_buffer: wgpu::Buffer,  // Grid-based delta Y
    position_delta_z_buffer: wgpu::Buffer,  // Grid-based delta Z
    // Phase 3: Particles sample from grid with trilinear interpolation
    density_correct_pipeline: wgpu::ComputePipeline,
    density_correct_bind_group: wgpu::BindGroup,
    density_correct_params_buffer: wgpu::Buffer,
    position_delta_buffer: wgpu::Buffer,  // Per-particle delta (output)

    // SDF collision (advection + solid collision)
    sdf_collision_pipeline: wgpu::ComputePipeline,
    sdf_collision_bind_group: wgpu::BindGroup,
    sdf_collision_params_buffer: wgpu::Buffer,
    sdf_buffer: wgpu::Buffer,

    // Maximum particles supported
    max_particles: usize,
}

impl GpuFlip3D {
    /// Create a new GPU 3D FLIP simulation
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        cell_size: f32,
        max_particles: usize,
    ) -> Self {
        // Create P2G solver (owns the grid velocity buffers)
        let p2g = GpuP2g3D::new(device, width, height, depth, max_particles);

        // Create grid velocity backup buffers for FLIP delta
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;

        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid U Old 3D"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid V Old 3D"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_w_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid W Old 3D"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create pressure solver (references P2G's grid buffers)
        let pressure = GpuPressure3D::new(
            device,
            width,
            height,
            depth,
            &p2g.grid_u_buffer,
            &p2g.grid_v_buffer,
            &p2g.grid_w_buffer,
        );

        // Create G2P solver (binds to P2G and old grid buffers)
        let g2p = GpuG2p3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            &p2g.grid_u_buffer,
            &p2g.grid_v_buffer,
            &p2g.grid_w_buffer,
            &grid_u_old_buffer,
            &grid_v_old_buffer,
            &grid_w_old_buffer,
        );

        // Create gravity shader
        let gravity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gravity_3d.wgsl").into()),
        });

        let gravity_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gravity Params 3D"),
            size: std::mem::size_of::<GravityParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Note: We need a cell_type buffer for gravity - borrow from pressure solver
        // For now we'll create a simple gravity pipeline that just modifies grid_v
        let gravity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gravity 3D Bind Group Layout"),
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

        // Use the pressure solver's cell_type buffer for gravity
        let gravity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity 3D Bind Group"),
            layout: &gravity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gravity_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_v_buffer.as_entire_binding() },
            ],
        });

        let gravity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gravity 3D Pipeline Layout"),
            bind_group_layouts: &[&gravity_bind_group_layout],
            push_constant_ranges: &[],
        });

        let gravity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gravity 3D Pipeline"),
            layout: Some(&gravity_pipeline_layout),
            module: &gravity_shader,
            entry_point: Some("apply_gravity"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create flow acceleration shader (for sluice downstream flow)
        let flow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flow 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flow_3d.wgsl").into()),
        });

        let flow_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flow Params 3D"),
            size: std::mem::size_of::<FlowParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Flow shader bindings: params, cell_type, grid_u
        let flow_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flow 3D Bind Group Layout"),
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

        let flow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Flow 3D Bind Group"),
            layout: &flow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: flow_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_u_buffer.as_entire_binding() },
            ],
        });

        let flow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flow 3D Pipeline Layout"),
            bind_group_layouts: &[&flow_bind_group_layout],
            push_constant_ranges: &[],
        });

        let flow_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flow 3D Pipeline"),
            layout: Some(&flow_pipeline_layout),
            module: &flow_shader,
            entry_point: Some("apply_flow"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create boundary condition enforcement shader
        let bc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Enforce BC 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/enforce_bc_3d.wgsl").into()),
        });

        let bc_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BC Params 3D"),
            size: std::mem::size_of::<BcParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bc_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BC 3D Bind Group Layout"),
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

        let bc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BC 3D Bind Group"),
            layout: &bc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bc_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p2g.grid_w_buffer.as_entire_binding() },
            ],
        });

        let bc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BC 3D Pipeline Layout"),
            bind_group_layouts: &[&bc_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bc_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC U 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC V 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC W 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Density Projection (Implicit Density Projection) ==========
        // Creates pipelines for density error computation and position correction

        let cell_count = (width * height * depth) as usize;

        // Create density error shader
        let density_error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Error 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_error_3d.wgsl").into()),
        });

        let density_error_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Error Params 3D"),
            size: std::mem::size_of::<DensityErrorParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Density error bindings: params, particle_count, cell_type, density_error (uses divergence_buffer)
        let density_error_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Error 3D Bind Group Layout"),
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
            ],
        });

        let density_error_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Error 3D Bind Group"),
            layout: &density_error_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_error_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.particle_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pressure.divergence_buffer.as_entire_binding() },
            ],
        });

        let density_error_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Error 3D Pipeline Layout"),
            bind_group_layouts: &[&density_error_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_error_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Error 3D Pipeline"),
            layout: Some(&density_error_pipeline_layout),
            module: &density_error_shader,
            entry_point: Some("compute_density_error"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Phase 2: Density Position Grid (blub approach) ==========
        // Compute position changes on grid, then particles sample with trilinear

        // Create grid-based position delta buffers
        let position_delta_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta X Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let position_delta_y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Y Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let position_delta_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Z Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create density position grid shader
        let density_position_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Position Grid 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_position_grid_3d.wgsl").into()),
        });

        let density_position_grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Position Grid Params 3D"),
            size: std::mem::size_of::<DensityPositionGridParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bindings: params, pressure, cell_type, delta_x, delta_y, delta_z
        let density_position_grid_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Position Grid 3D Bind Group Layout"),
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

        let density_position_grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Position Grid 3D Bind Group"),
            layout: &density_position_grid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_position_grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.pressure_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: position_delta_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: position_delta_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: position_delta_z_buffer.as_entire_binding() },
            ],
        });

        let density_position_grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Position Grid 3D Pipeline Layout"),
            bind_group_layouts: &[&density_position_grid_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_position_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Position Grid 3D Pipeline"),
            layout: Some(&density_position_grid_pipeline_layout),
            module: &density_position_grid_shader,
            entry_point: Some("compute_position_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Phase 3: Particle Position Correction (trilinear sampling) ==========
        let density_correct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Correct 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_correct_3d.wgsl").into()),
        });

        let density_correct_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Correct Params 3D"),
            size: std::mem::size_of::<DensityCorrectionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-particle position delta buffer (output)
        let position_delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta 3D"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bindings: params, delta_x, delta_y, delta_z, cell_type, positions, particle_delta
        let density_correct_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Correct 3D Bind Group Layout"),
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
            ],
        });

        let density_correct_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Correct 3D Bind Group"),
            layout: &density_correct_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_correct_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: position_delta_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: position_delta_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: position_delta_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: g2p.positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: position_delta_buffer.as_entire_binding() },
            ],
        });

        let density_correct_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Correct 3D Pipeline Layout"),
            bind_group_layouts: &[&density_correct_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_correct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Correct 3D Pipeline"),
            layout: Some(&density_correct_pipeline_layout),
            module: &density_correct_shader,
            entry_point: Some("correct_positions"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== SDF Collision (Advection + Solid Collision) ==========
        let sdf_collision_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Collision 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf_collision_3d.wgsl").into()),
        });

        let sdf_collision_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Collision Params 3D"),
            size: std::mem::size_of::<SdfCollisionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sdf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Collision Grid 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sdf_collision_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Collision 3D Bind Group Layout"),
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
            ],
        });

        let sdf_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Collision 3D Bind Group"),
            layout: &sdf_collision_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sdf_collision_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: g2p.positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g2p.velocities_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sdf_buffer.as_entire_binding() },
            ],
        });

        let sdf_collision_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Collision 3D Pipeline Layout"),
            bind_group_layouts: &[&sdf_collision_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sdf_collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Collision 3D Pipeline"),
            layout: Some(&sdf_collision_pipeline_layout),
            module: &sdf_collision_shader,
            entry_point: Some("sdf_collision"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            height,
            depth,
            cell_size,
            p2g,
            g2p,
            pressure,
            gravity_pipeline,
            gravity_bind_group,
            gravity_params_buffer,
            flow_pipeline,
            flow_bind_group,
            flow_params_buffer,
            bc_u_pipeline,
            bc_v_pipeline,
            bc_w_pipeline,
            bc_bind_group,
            bc_params_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            grid_w_old_buffer,
            density_error_pipeline,
            density_error_bind_group,
            density_error_params_buffer,
            density_position_grid_pipeline,
            density_position_grid_bind_group,
            density_position_grid_params_buffer,
            position_delta_x_buffer,
            position_delta_y_buffer,
            position_delta_z_buffer,
            density_correct_pipeline,
            density_correct_bind_group,
            density_correct_params_buffer,
            position_delta_buffer,
            sdf_collision_pipeline,
            sdf_collision_bind_group,
            sdf_collision_params_buffer,
            sdf_buffer,
            max_particles,
        }
    }

    /// Run one simulation step
    ///
    /// This performs the full FLIP pipeline:
    /// 1. P2G: Transfer particle data to grid
    /// 2. Enforce boundary conditions (before storing old velocities!)
    /// 3. Save grid velocity (for FLIP delta)
    /// 4. Apply gravity (vertical)
    /// 5. Apply flow acceleration (horizontal, for sluice flow)
    /// 6. Pressure solve (includes divergence, iterations, gradient)
    /// 7. G2P: Transfer grid data back to particles
    /// 8. Optional: GPU advection + SDF collision (when `sdf` is provided)
    ///
    /// # Arguments
    /// * `flow_accel` - Downstream flow acceleration (m/s²). Set to 0.0 for closed box sims.
    ///                  For a sluice, use ~2-5 m/s² to drive water downstream.
    pub fn step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],  // Now mutable for density projection position correction
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        let particle_count = positions.len().min(self.max_particles);
        if particle_count == 0 {
            return;
        }

        // Upload cell types FIRST (needed for BC enforcement)
        self.pressure.upload_cell_types(queue, cell_types, self.cell_size);

        // Upload BC params
        let bc_params = BcParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            _pad: 0,
        };
        queue.write_buffer(&self.bc_params_buffer, 0, bytemuck::bytes_of(&bc_params));

        // 1. Upload particles and run P2G
        let count = self.p2g.upload_particles(
            queue,
            positions,
            velocities,
            c_matrices,
            self.cell_size,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Step Encoder"),
        });

        // Run P2G scatter and divide
        self.p2g.encode(&mut encoder, count);

        queue.submit(std::iter::once(encoder.finish()));

        // 2. Enforce boundary conditions BEFORE storing old velocities
        // This is critical for correct FLIP delta computation!
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D BC Encoder"),
        });

        // Enforce BC on U
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC U 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_u_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on V
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC V 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_v_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on W
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC W 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_w_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 1 + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 3. Save grid velocity for FLIP delta (now with proper BCs!)
        let (u_size, v_size, w_size) = self.p2g.grid_sizes();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Grid Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p2g.grid_u_buffer, 0, &self.grid_u_old_buffer, 0, (u_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.p2g.grid_v_buffer, 0, &self.grid_v_old_buffer, 0, (v_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.p2g.grid_w_buffer, 0, &self.grid_w_old_buffer, 0, (w_size * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // 4. Apply gravity

        let gravity_params = GravityParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            gravity_dt: gravity * dt,
        };
        queue.write_buffer(&self.gravity_params_buffer, 0, bytemuck::bytes_of(&gravity_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Gravity Encoder"),
        });

        // Apply gravity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravity 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravity_pipeline);
            pass.set_bind_group(0, &self.gravity_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 5. Apply flow acceleration (for sluice downstream flow)
        // This MUST happen before pressure solve so the solver can account for the flow!
        if flow_accel.abs() > 0.0001 {
            let flow_params = FlowParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                flow_accel_dt: flow_accel * dt,
            };
            queue.write_buffer(&self.flow_params_buffer, 0, bytemuck::bytes_of(&flow_params));

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flow 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.flow_pipeline);
            pass.set_bind_group(0, &self.flow_bind_group, &[]);
            // U grid: (width+1) x height x depth
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 6. Pressure solve (divergence → iterations → gradient)
        self.pressure.encode(&mut encoder, pressure_iterations);

        queue.submit(std::iter::once(encoder.finish()));

        // 6. Run G2P using grid buffers already on GPU
        let g2p_count = self.g2p.upload_particles(
            queue,
            positions,
            velocities,
            c_matrices,
            self.cell_size,
            dt,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D G2P Encoder"),
        });
        self.g2p.encode(&mut encoder, g2p_count);
        queue.submit(std::iter::once(encoder.finish()));

        // ========== Density Projection (Implicit Density Projection) ==========
        // Push particles from crowded regions to empty regions
        // This causes water level to "rise" when particles accumulate behind riffles

        // 1. Compute density error from particle counts (populated during P2G)
        let density_error_params = DensityErrorParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            rest_density: 8.0,  // Target ~8 particles per cell
            dt,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&self.density_error_params_buffer, 0, bytemuck::bytes_of(&density_error_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Density Projection Encoder"),
        });

        // Dispatch density error shader - writes to divergence_buffer
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Density Error 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_error_pipeline);
            pass.set_bind_group(0, &self.density_error_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 2. Clear pressure and run density pressure iterations
        self.pressure.clear_pressure(queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Density Pressure Encoder"),
        });

        // Run pressure solver iterations with density error as RHS
        // Uses same Jacobi solver, just different input
        let density_iterations = 40;  // More iterations for volume conservation
        self.pressure.encode_iterations_only(&mut encoder, density_iterations);

        queue.submit(std::iter::once(encoder.finish()));

        // 3. Compute position deltas on grid (blub approach)
        // Update grid shader params with dt
        let grid_params = DensityPositionGridParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            dt,
        };
        queue.write_buffer(&self.density_position_grid_params_buffer, 0, bytemuck::bytes_of(&grid_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Position Grid Encoder"),
        });

        // Dispatch grid position delta shader
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Density Position Grid 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_position_grid_pipeline);
            pass.set_bind_group(0, &self.density_position_grid_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 4. Apply position correction to particles (trilinear sampling from grid)
        let density_correct_params = DensityCorrectionParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count: particle_count as u32,
            cell_size: self.cell_size,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&self.density_correct_params_buffer, 0, bytemuck::bytes_of(&density_correct_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Position Correction Encoder"),
        });

        // Dispatch particle position correction shader
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Density Correct 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_correct_pipeline);
            pass.set_bind_group(0, &self.density_correct_bind_group, &[]);
            let workgroups = (particle_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 5. Download position deltas and apply to particle positions
        let mut position_deltas = vec![[0.0f32; 4]; particle_count];
        Self::read_buffer_vec4(device, queue, &self.position_delta_buffer, &mut position_deltas, particle_count);

        // Apply position corrections to particles
        for (i, delta) in position_deltas.iter().enumerate() {
            positions[i].x += delta[0];
            positions[i].y += delta[1];
            positions[i].z += delta[2];
        }

        let run_sdf_collision = if let Some(sdf) = sdf {
            let expected_sdf_len = (self.width * self.height * self.depth) as usize;
            assert_eq!(
                sdf.len(),
                expected_sdf_len,
                "SDF size mismatch: got {}, expected {}",
                sdf.len(),
                expected_sdf_len
            );

            // Upload latest positions (after density correction) and SDF
            let positions_padded: Vec<[f32; 4]> = positions
                .iter()
                .take(particle_count)
                .map(|p| [p.x, p.y, p.z, 0.0])
                .collect();
            queue.write_buffer(&self.g2p.positions_buffer, 0, bytemuck::cast_slice(&positions_padded));
            queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));

            let sdf_params = SdfCollisionParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                particle_count: particle_count as u32,
                cell_size: self.cell_size,
                dt,
                _pad0: 0,
                _pad1: 0,
            };
            queue.write_buffer(&self.sdf_collision_params_buffer, 0, bytemuck::bytes_of(&sdf_params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FLIP 3D SDF Collision Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SDF Collision 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sdf_collision_pipeline);
                pass.set_bind_group(0, &self.sdf_collision_bind_group, &[]);
                let workgroups = (g2p_count + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            true
        } else {
            false
        };

        // Download results (velocities + C matrix). Positions are optional.
        self.g2p.download(device, queue, g2p_count, velocities, c_matrices);

        if run_sdf_collision {
            let mut gpu_positions = vec![[0.0f32; 4]; particle_count];
            Self::read_buffer_vec4(device, queue, &self.g2p.positions_buffer, &mut gpu_positions, particle_count);
            for (i, pos) in gpu_positions.iter().enumerate() {
                positions[i] = glam::Vec3::new(pos[0], pos[1], pos[2]);
            }
        }
    }

    fn read_buffer_vec4(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, output: &mut [[f32; 4]], count: usize) {
        let byte_size = count * std::mem::size_of::<[f32; 4]>();

        // Create a staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Vec4 Staging"),
            size: byte_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Vec4 Buffer Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            output[..count].copy_from_slice(&slice[..count]);
        }
        staging.unmap();
    }
}
