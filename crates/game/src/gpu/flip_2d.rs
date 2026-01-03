//! GPU-accelerated 2D FLIP/PIC simulation (compute only).
//!
//! The CPU only orchestrates compute passes and uploads initial particles.
//! All simulation updates run on the GPU.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClearParams {
    u_len: u32,
    v_len: u32,
    p_len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct P2gParams {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GravityParams {
    width: u32,
    height: u32,
    gravity_dt: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridParams {
    width: u32,
    height: u32,
    inv_cell_size: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PressureParams {
    width: u32,
    height: u32,
    alpha: f32,
    rbeta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct G2pParams {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    flip_ratio: f32,
    dt: f32,
    max_velocity: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdvectParams {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    dt: f32,
    bounce: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BcParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct GpuFlip2D {
    width: u32,
    height: u32,
    cell_size: f32,
    max_particles: usize,
    particle_count: u32,
    u_len: u32,
    v_len: u32,
    p_len: u32,

    // Particle buffers (vec4 padded)
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,

    // Grid accumulators
    _u_sum_buffer: wgpu::Buffer,
    _u_weight_buffer: wgpu::Buffer,
    _v_sum_buffer: wgpu::Buffer,
    _v_weight_buffer: wgpu::Buffer,

    // Grid velocities
    grid_u_buffer: wgpu::Buffer,
    grid_v_buffer: wgpu::Buffer,
    grid_u_old_buffer: wgpu::Buffer,
    grid_v_old_buffer: wgpu::Buffer,

    // Pressure buffers
    _divergence_buffer: wgpu::Buffer,
    _pressure_a_buffer: wgpu::Buffer,
    _pressure_b_buffer: wgpu::Buffer,

    // Parameter buffers
    _clear_params_buffer: wgpu::Buffer,
    p2g_params_buffer: wgpu::Buffer,
    gravity_params_buffer: wgpu::Buffer,
    _grid_params_buffer: wgpu::Buffer,
    _pressure_params_buffer: wgpu::Buffer,
    g2p_params_buffer: wgpu::Buffer,
    advect_params_buffer: wgpu::Buffer,
    _bc_params_buffer: wgpu::Buffer,

    // Pipelines
    clear_pipeline: wgpu::ComputePipeline,
    p2g_u_pipeline: wgpu::ComputePipeline,
    p2g_v_pipeline: wgpu::ComputePipeline,
    divide_u_pipeline: wgpu::ComputePipeline,
    divide_v_pipeline: wgpu::ComputePipeline,
    gravity_pipeline: wgpu::ComputePipeline,
    divergence_pipeline: wgpu::ComputePipeline,
    pressure_pipeline: wgpu::ComputePipeline,
    gradient_u_pipeline: wgpu::ComputePipeline,
    gradient_v_pipeline: wgpu::ComputePipeline,
    bc_u_pipeline: wgpu::ComputePipeline,
    bc_v_pipeline: wgpu::ComputePipeline,
    g2p_pipeline: wgpu::ComputePipeline,
    advect_pipeline: wgpu::ComputePipeline,

    // Bind groups
    clear_bind_group: wgpu::BindGroup,
    p2g_u_bind_group: wgpu::BindGroup,
    p2g_v_bind_group: wgpu::BindGroup,
    divide_u_bind_group: wgpu::BindGroup,
    divide_v_bind_group: wgpu::BindGroup,
    gravity_bind_group: wgpu::BindGroup,
    divergence_bind_group: wgpu::BindGroup,
    pressure_bind_group_ab: wgpu::BindGroup,
    pressure_bind_group_ba: wgpu::BindGroup,
    gradient_bind_group_a: wgpu::BindGroup,
    gradient_bind_group_b: wgpu::BindGroup,
    bc_bind_group: wgpu::BindGroup,
    g2p_bind_group: wgpu::BindGroup,
    advect_bind_group: wgpu::BindGroup,

    // Tunables
    flip_ratio: f32,
    max_velocity: f32,
    bounce: f32,
}

impl GpuFlip2D {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        cell_size: f32,
        max_particles: usize,
    ) -> Self {
        let u_len = (width + 1) * height;
        let v_len = width * (height + 1);
        let p_len = width * height;

        let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particles Positions 2D"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocities_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particles Velocities 2D"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("U Sum 2D"),
            size: (u_len as usize * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let u_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("U Weight 2D"),
            size: (u_len as usize * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let v_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Sum 2D"),
            size: (v_len as usize * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let v_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Weight 2D"),
            size: (v_len as usize * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid U 2D"),
            size: (u_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid V 2D"),
            size: (v_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid U Old 2D"),
            size: (u_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid V Old 2D"),
            size: (v_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let divergence_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Divergence 2D"),
            size: (p_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pressure_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure A 2D"),
            size: (p_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pressure_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure B 2D"),
            size: (p_len as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let clear_params = ClearParams {
            u_len,
            v_len,
            p_len,
            _pad: 0,
        };
        let clear_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Params 2D"),
            contents: bytemuck::bytes_of(&clear_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let p2g_params = P2gParams {
            cell_size,
            width,
            height,
            particle_count: 0,
        };
        let p2g_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("P2G Params 2D"),
            contents: bytemuck::bytes_of(&p2g_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gravity_params = GravityParams {
            width,
            height,
            gravity_dt: 0.0,
            _pad: 0.0,
        };
        let gravity_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gravity Params 2D"),
            contents: bytemuck::bytes_of(&gravity_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_params = GridParams {
            width,
            height,
            inv_cell_size: 1.0 / cell_size,
            _pad: 0.0,
        };
        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Params 2D"),
            contents: bytemuck::bytes_of(&grid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pressure_params = PressureParams {
            width,
            height,
            alpha: -cell_size * cell_size,
            rbeta: 0.25,
        };
        let pressure_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pressure Params 2D"),
            contents: bytemuck::bytes_of(&pressure_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let g2p_params = G2pParams {
            cell_size,
            width,
            height,
            particle_count: 0,
            flip_ratio: 0.97,
            dt: 0.0,
            max_velocity: 2000.0,
            _pad: 0.0,
        };
        let g2p_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G2P Params 2D"),
            contents: bytemuck::bytes_of(&g2p_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let advect_params = AdvectParams {
            cell_size,
            width,
            height,
            particle_count: 0,
            dt: 0.0,
            bounce: 0.1,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let advect_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Advect Params 2D"),
            contents: bytemuck::bytes_of(&advect_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bc_params = BcParams {
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        };
        let bc_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BC Params 2D"),
            contents: bytemuck::bytes_of(&bc_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear_grid_2d.wgsl").into()),
        });
        let p2g_u_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Scatter U 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_scatter_u_2d.wgsl").into()),
        });
        let p2g_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Scatter V 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_scatter_v_2d.wgsl").into()),
        });
        let divide_u_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Divide U 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_divide_u_2d.wgsl").into()),
        });
        let divide_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Divide V 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_divide_v_2d.wgsl").into()),
        });
        let gravity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravity 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gravity_2d.wgsl").into()),
        });
        let divergence_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Divergence 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/divergence_2d.wgsl").into()),
        });
        let pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pressure_jacobi_2d.wgsl").into()),
        });
        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure Gradient 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pressure_gradient_2d.wgsl").into()),
        });
        let bc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Boundary 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/enforce_bc_2d.wgsl").into()),
        });
        let g2p_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G2P 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/g2p_2d.wgsl").into()),
        });
        let advect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Advect 2D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/advect_2d.wgsl").into()),
        });

        let clear_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Clear Bind Group Layout"),
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

        let clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clear Bind Group"),
            layout: &clear_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clear_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pressure_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pressure_b_buffer.as_entire_binding(),
                },
            ],
        });

        let clear_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Clear Pipeline Layout"),
            bind_group_layouts: &[&clear_bind_group_layout],
            push_constant_ranges: &[],
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Pipeline"),
            layout: Some(&clear_pipeline_layout),
            module: &clear_shader,
            entry_point: Some("clear"),
            compilation_options: Default::default(),
            cache: None,
        });

        let p2g_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("P2G Bind Group Layout"),
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

        let p2g_u_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G U Bind Group"),
            layout: &p2g_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p2g_params_buffer.as_entire_binding(),
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
                    resource: u_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: u_weight_buffer.as_entire_binding(),
                },
            ],
        });

        let p2g_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G V Bind Group"),
            layout: &p2g_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p2g_params_buffer.as_entire_binding(),
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
                    resource: v_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_weight_buffer.as_entire_binding(),
                },
            ],
        });

        let p2g_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("P2G Pipeline Layout"),
            bind_group_layouts: &[&p2g_bind_group_layout],
            push_constant_ranges: &[],
        });
        let p2g_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G Scatter U Pipeline"),
            layout: Some(&p2g_pipeline_layout),
            module: &p2g_u_shader,
            entry_point: Some("scatter_u"),
            compilation_options: Default::default(),
            cache: None,
        });
        let p2g_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G Scatter V Pipeline"),
            layout: Some(&p2g_pipeline_layout),
            module: &p2g_v_shader,
            entry_point: Some("scatter_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let divide_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Divide Bind Group Layout"),
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

        let divide_u_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Divide U Bind Group"),
            layout: &divide_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p2g_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_u_buffer.as_entire_binding(),
                },
            ],
        });

        let divide_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Divide V Bind Group"),
            layout: &divide_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p2g_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_v_buffer.as_entire_binding(),
                },
            ],
        });

        let divide_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Divide Pipeline Layout"),
            bind_group_layouts: &[&divide_bind_group_layout],
            push_constant_ranges: &[],
        });
        let divide_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Divide U Pipeline"),
            layout: Some(&divide_pipeline_layout),
            module: &divide_u_shader,
            entry_point: Some("divide_u"),
            compilation_options: Default::default(),
            cache: None,
        });
        let divide_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Divide V Pipeline"),
            layout: Some(&divide_pipeline_layout),
            module: &divide_v_shader,
            entry_point: Some("divide_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gravity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gravity Bind Group Layout"),
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
        let gravity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity Bind Group"),
            layout: &gravity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gravity_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_v_buffer.as_entire_binding(),
                },
            ],
        });
        let gravity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gravity Pipeline Layout"),
            bind_group_layouts: &[&gravity_bind_group_layout],
            push_constant_ranges: &[],
        });
        let gravity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gravity Pipeline"),
            layout: Some(&gravity_pipeline_layout),
            module: &gravity_shader,
            entry_point: Some("apply_gravity"),
            compilation_options: Default::default(),
            cache: None,
        });

        let divergence_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Divergence Bind Group Layout"),
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
        let divergence_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Divergence Bind Group"),
            layout: &divergence_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: divergence_buffer.as_entire_binding(),
                },
            ],
        });
        let divergence_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Divergence Pipeline Layout"),
            bind_group_layouts: &[&divergence_bind_group_layout],
            push_constant_ranges: &[],
        });
        let divergence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Divergence Pipeline"),
            layout: Some(&divergence_pipeline_layout),
            module: &divergence_shader,
            entry_point: Some("compute_divergence"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pressure_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pressure Bind Group Layout"),
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
        let pressure_bind_group_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pressure Bind Group A->B"),
            layout: &pressure_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pressure_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: divergence_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pressure_b_buffer.as_entire_binding(),
                },
            ],
        });
        let pressure_bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pressure Bind Group B->A"),
            layout: &pressure_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pressure_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: divergence_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pressure_a_buffer.as_entire_binding(),
                },
            ],
        });
        let pressure_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure Pipeline Layout"),
            bind_group_layouts: &[&pressure_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pressure Pipeline"),
            layout: Some(&pressure_pipeline_layout),
            module: &pressure_shader,
            entry_point: Some("pressure_jacobi"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gradient_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gradient Bind Group Layout"),
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
        let gradient_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group A"),
            layout: &gradient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pressure_a_buffer.as_entire_binding(),
                },
            ],
        });
        let gradient_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group B"),
            layout: &gradient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pressure_b_buffer.as_entire_binding(),
                },
            ],
        });
        let gradient_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Pipeline Layout"),
            bind_group_layouts: &[&gradient_bind_group_layout],
            push_constant_ranges: &[],
        });
        let gradient_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient U Pipeline"),
            layout: Some(&gradient_pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("gradient_u"),
            compilation_options: Default::default(),
            cache: None,
        });
        let gradient_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient V Pipeline"),
            layout: Some(&gradient_pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("gradient_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BC Bind Group Layout"),
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
            ],
        });
        let bc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BC Bind Group"),
            layout: &bc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bc_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_v_buffer.as_entire_binding(),
                },
            ],
        });
        let bc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BC Pipeline Layout"),
            bind_group_layouts: &[&bc_bind_group_layout],
            push_constant_ranges: &[],
        });
        let bc_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC U Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_u"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bc_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC V Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let g2p_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G2P Bind Group Layout"),
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
            ],
        });
        let g2p_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G2P Bind Group"),
            layout: &g2p_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: g2p_params_buffer.as_entire_binding(),
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
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grid_u_old_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_v_old_buffer.as_entire_binding(),
                },
            ],
        });
        let g2p_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("G2P Pipeline Layout"),
            bind_group_layouts: &[&g2p_bind_group_layout],
            push_constant_ranges: &[],
        });
        let g2p_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("G2P Pipeline"),
            layout: Some(&g2p_pipeline_layout),
            module: &g2p_shader,
            entry_point: Some("g2p"),
            compilation_options: Default::default(),
            cache: None,
        });

        let advect_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Advect Bind Group Layout"),
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
            ],
        });
        let advect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Advect Bind Group"),
            layout: &advect_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: advect_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities_buffer.as_entire_binding(),
                },
            ],
        });
        let advect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Advect Pipeline Layout"),
            bind_group_layouts: &[&advect_bind_group_layout],
            push_constant_ranges: &[],
        });
        let advect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Advect Pipeline"),
            layout: Some(&advect_pipeline_layout),
            module: &advect_shader,
            entry_point: Some("advect"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            height,
            cell_size,
            max_particles,
            particle_count: 0,
            u_len,
            v_len,
            p_len,
            positions_buffer,
            velocities_buffer,
            _u_sum_buffer: u_sum_buffer,
            _u_weight_buffer: u_weight_buffer,
            _v_sum_buffer: v_sum_buffer,
            _v_weight_buffer: v_weight_buffer,
            grid_u_buffer,
            grid_v_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            _divergence_buffer: divergence_buffer,
            _pressure_a_buffer: pressure_a_buffer,
            _pressure_b_buffer: pressure_b_buffer,
            _clear_params_buffer: clear_params_buffer,
            p2g_params_buffer,
            gravity_params_buffer,
            _grid_params_buffer: grid_params_buffer,
            _pressure_params_buffer: pressure_params_buffer,
            g2p_params_buffer,
            advect_params_buffer,
            _bc_params_buffer: bc_params_buffer,
            clear_pipeline,
            p2g_u_pipeline,
            p2g_v_pipeline,
            divide_u_pipeline,
            divide_v_pipeline,
            gravity_pipeline,
            divergence_pipeline,
            pressure_pipeline,
            gradient_u_pipeline,
            gradient_v_pipeline,
            bc_u_pipeline,
            bc_v_pipeline,
            g2p_pipeline,
            advect_pipeline,
            clear_bind_group,
            p2g_u_bind_group,
            p2g_v_bind_group,
            divide_u_bind_group,
            divide_v_bind_group,
            gravity_bind_group,
            divergence_bind_group,
            pressure_bind_group_ab,
            pressure_bind_group_ba,
            gradient_bind_group_a,
            gradient_bind_group_b,
            bc_bind_group,
            g2p_bind_group,
            advect_bind_group,
            flip_ratio: 0.97,
            max_velocity: 2000.0,
            bounce: 0.1,
        }
    }

    pub fn upload_particles(
        &mut self,
        queue: &wgpu::Queue,
        positions: &[[f32; 2]],
        velocities: &[[f32; 2]],
    ) {
        let count = positions
            .len()
            .min(velocities.len())
            .min(self.max_particles);
        self.particle_count = count as u32;

        let mut pos_data = Vec::with_capacity(count);
        let mut vel_data = Vec::with_capacity(count);
        for i in 0..count {
            pos_data.push([positions[i][0], positions[i][1], 0.0, 0.0]);
            vel_data.push([velocities[i][0], velocities[i][1], 0.0, 0.0]);
        }

        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&pos_data));
        queue.write_buffer(&self.velocities_buffer, 0, bytemuck::cast_slice(&vel_data));

        let p2g_params = P2gParams {
            cell_size: self.cell_size,
            width: self.width,
            height: self.height,
            particle_count: self.particle_count,
        };
        queue.write_buffer(&self.p2g_params_buffer, 0, bytemuck::bytes_of(&p2g_params));

        let g2p_params = G2pParams {
            cell_size: self.cell_size,
            width: self.width,
            height: self.height,
            particle_count: self.particle_count,
            flip_ratio: self.flip_ratio,
            dt: 0.0,
            max_velocity: self.max_velocity,
            _pad: 0.0,
        };
        queue.write_buffer(&self.g2p_params_buffer, 0, bytemuck::bytes_of(&g2p_params));

        let advect_params = AdvectParams {
            cell_size: self.cell_size,
            width: self.width,
            height: self.height,
            particle_count: self.particle_count,
            dt: 0.0,
            bounce: self.bounce,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.advect_params_buffer, 0, bytemuck::bytes_of(&advect_params));
    }

    pub fn step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        gravity: f32,
        pressure_iters: u32,
    ) {
        if self.particle_count == 0 {
            return;
        }

        let g2p_params = G2pParams {
            cell_size: self.cell_size,
            width: self.width,
            height: self.height,
            particle_count: self.particle_count,
            flip_ratio: self.flip_ratio,
            dt,
            max_velocity: self.max_velocity,
            _pad: 0.0,
        };
        queue.write_buffer(&self.g2p_params_buffer, 0, bytemuck::bytes_of(&g2p_params));

        let advect_params = AdvectParams {
            cell_size: self.cell_size,
            width: self.width,
            height: self.height,
            particle_count: self.particle_count,
            dt,
            bounce: self.bounce,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.advect_params_buffer, 0, bytemuck::bytes_of(&advect_params));

        let gravity_params = GravityParams {
            width: self.width,
            height: self.height,
            gravity_dt: gravity * dt,
            _pad: 0.0,
        };
        queue.write_buffer(&self.gravity_params_buffer, 0, bytemuck::bytes_of(&gravity_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Flip 2D Step Encoder"),
        });

        let particle_workgroups = (self.particle_count + 255) / 256;
        let clear_len = self.u_len.max(self.v_len).max(self.p_len);
        let clear_workgroups = (clear_len + 255) / 256;
        let u_workgroups = (self.u_len + 255) / 256;
        let v_workgroups = (self.v_len + 255) / 256;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &self.clear_bind_group, &[]);
            pass.dispatch_workgroups(clear_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G U Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p2g_u_pipeline);
            pass.set_bind_group(0, &self.p2g_u_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G V Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p2g_v_pipeline);
            pass.set_bind_group(0, &self.p2g_v_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Divide U Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_u_pipeline);
            pass.set_bind_group(0, &self.divide_u_bind_group, &[]);
            pass.dispatch_workgroups(u_workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Divide V Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_v_pipeline);
            pass.set_bind_group(0, &self.divide_v_bind_group, &[]);
            pass.dispatch_workgroups(v_workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.grid_u_buffer,
            0,
            &self.grid_u_old_buffer,
            0,
            self.u_len as u64 * std::mem::size_of::<f32>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.grid_v_buffer,
            0,
            &self.grid_v_old_buffer,
            0,
            self.v_len as u64 * std::mem::size_of::<f32>() as u64,
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravity Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravity_pipeline);
            pass.set_bind_group(0, &self.gravity_bind_group, &[]);
            pass.dispatch_workgroups(v_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Divergence Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divergence_pipeline);
            pass.set_bind_group(0, &self.divergence_bind_group, &[]);
            let wg_x = (self.width + 7) / 8;
            let wg_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        for iter in 0..pressure_iters {
            let use_ab = iter % 2 == 0;
            let bind_group = if use_ab {
                &self.pressure_bind_group_ab
            } else {
                &self.pressure_bind_group_ba
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pressure Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pressure_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let wg_x = (self.width + 7) / 8;
            let wg_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        if pressure_iters > 0 {
            let pressure_is_a = pressure_iters % 2 == 0;
            let gradient_bind_group = if pressure_is_a {
                &self.gradient_bind_group_a
            } else {
                &self.gradient_bind_group_b
            };

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Gradient U Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gradient_u_pipeline);
                pass.set_bind_group(0, gradient_bind_group, &[]);
                let wg_x = (self.width + 1 + 7) / 8;
                let wg_y = (self.height + 7) / 8;
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Gradient V Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gradient_v_pipeline);
                pass.set_bind_group(0, gradient_bind_group, &[]);
                let wg_x = (self.width + 7) / 8;
                let wg_y = (self.height + 1 + 7) / 8;
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC U Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_u_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let wg_x = (self.width + 1 + 7) / 8;
            let wg_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC V Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_v_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let wg_x = (self.width + 7) / 8;
            let wg_y = (self.height + 1 + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("G2P Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.g2p_pipeline);
            pass.set_bind_group(0, &self.g2p_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Advect Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.advect_pipeline);
            pass.set_bind_group(0, &self.advect_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}
