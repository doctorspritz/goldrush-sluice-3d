//! GPU-accelerated pressure solver for 3D FLIP simulation.
//!
//! This module implements the pressure Poisson equation solver using
//! Red-Black Gauss-Seidel with SOR (Successive Over-Relaxation).
//!
//! Pipeline:
//! 1. Compute divergence: div = ∇·u
//! 2. Solve ∇²p = -div iteratively (red-black passes)
//! 3. Apply pressure gradient: u -= ∇p

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Parameters for divergence/gradient shaders
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GridParams3D {
    width: u32,
    height: u32,
    depth: u32,
    inv_cell_size: f32,
    open_boundaries: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Parameters for pressure solver
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PressureParams3D {
    width: u32,
    height: u32,
    depth: u32,
    omega: f32, // SOR relaxation factor (1.5-1.9)
    h_sq: f32,  // cell_size^2, for Poisson equation scaling
    open_boundaries: u32, // Bitmask: 1=-X, 2=+X, 4=-Y, 8=+Y, 16=-Z, 32=+Z
    phase_omega: f32,
    _pad1: u32,
}

/// Residual of the stored pressure equation, before any velocity correction.
/// Diagnostics are opt-in and never add work or buffers to ordinary stepping.
#[derive(Debug, Clone, Default)]
pub struct PressureResidual3D {
    pub fluid_cells: usize,
    pub max_abs: f64,
    pub rms: f64,
    pub rhs_l2: f64,
    /// Undefined if a zero RHS has a nonzero residual; this is not convergence.
    pub relative_l2: Option<f64>,
    pub max_backward_error: f64,
}

/// Rectangular mixed-boundary estimate for the weighted operator. This is a
/// convergence heuristic, not an optimum for arbitrary mobility/air/solid masks.
/// A closed axis admits a constant mode; a single open end has a half-wave.
/// The fully closed case estimates the first nonconstant mode, retaining its
/// pressure nullspace and exposing incompatible sources through the residual.
fn phase_relaxation(shape: [u32; 3], open: u32, has_air: bool) -> f32 {
    let mut theta = [0.; 3];
    for (axis, angle) in theta.iter_mut().enumerate() {
        let ends = ((open >> (2 * axis)) & 3).count_ones();
        let n = shape[axis].max(1) as f32;
        *angle = match ends {
            0 => 0.,
            // Neumann at the lower face and zero pressure at the next air
            // centre: cos(theta*(j+1/2)), theta*(N+1/2)=pi/2.
            1 => std::f32::consts::PI / (2. * n + 1.),
            _ => std::f32::consts::PI / (n + 1.),
        };
    }
    if open & 63 == 0 {
        let axis = (0..3).max_by_key(|&axis| shape[axis]).unwrap();
        let n = shape[axis].max(1) as f32;
        // Air inside a closed exterior still grounds pressure. CPU air may
        // later receive fluid on the GPU, so this remains a spectral estimate,
        // not a claim that the host mask is current fluid occupancy.
        theta[axis] = std::f32::consts::PI / if has_air { 2. * n + 1. } else { n };
    }
    let rho = theta.iter().map(|v| v.cos()).sum::<f32>() / 3.;
    (2. / (1. + (1. - rho * rho).max(0.).sqrt())).clamp(1., 1.95)
}

/// GPU-based pressure solver for 3D FLIP
pub struct GpuPressure3D {
    pub compact_iterations: bool,
    pub cached_stencil: bool,
    width: u32,
    height: u32,
    depth: u32,

    // Pressure and divergence buffers (public for density projection reuse)
    pub pressure_buffer: wgpu::Buffer,
    pub divergence_buffer: wgpu::Buffer,
    pub cell_type_buffer: wgpu::Buffer, // Public so gravity shader can use it
    pressure_backup_buffer: wgpu::Buffer,

    // Parameters
    grid_params_buffer: wgpu::Buffer,
    pressure_params_buffer: wgpu::Buffer,

    // Compute pipelines
    divergence_pipeline: wgpu::ComputePipeline,
    pressure_red_pipeline: wgpu::ComputePipeline,
    pressure_black_pipeline: wgpu::ComputePipeline,
    packed_red: wgpu::ComputePipeline,
    packed_black: wgpu::ComputePipeline,
    stencil_prepare: wgpu::ComputePipeline,
    stencil_red: wgpu::ComputePipeline,
    stencil_black: wgpu::ComputePipeline,
    gradient_u_pipeline: wgpu::ComputePipeline,
    gradient_v_pipeline: wgpu::ComputePipeline,
    gradient_w_pipeline: wgpu::ComputePipeline,

    // Bind groups
    divergence_bind_group: wgpu::BindGroup,
    pressure_bind_group: wgpu::BindGroup,
    pressure_bind_group_layout: wgpu::BindGroupLayout,
    variable_mobility_allowed: bool,
    gradient_bind_group: wgpu::BindGroup,
}

impl GpuPressure3D {
    /// Create a new GPU pressure solver
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        grid_u_buffer: &wgpu::Buffer,
        grid_v_buffer: &wgpu::Buffer,
        grid_w_buffer: &wgpu::Buffer,
    ) -> Self {
        Self::new_impl(
            device,
            width,
            height,
            depth,
            grid_u_buffer,
            grid_v_buffer,
            grid_w_buffer,
            None,
        )
    }

    /// Variable face mobility for a locally eliminated multi-phase response.
    /// Each plane contains a ready header followed by its MAC coefficients.
    /// A zero U header selects the unchanged constant-mobility path, allowing
    /// position repair and phase projection in one ordered GPU submission.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_mobility(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        grid_u_buffer: &wgpu::Buffer,
        grid_v_buffer: &wgpu::Buffer,
        grid_w_buffer: &wgpu::Buffer,
        mobility: [&wgpu::Buffer; 3],
    ) -> Self {
        Self::new_impl(
            device,
            width,
            height,
            depth,
            grid_u_buffer,
            grid_v_buffer,
            grid_w_buffer,
            Some(mobility),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_impl(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        grid_u_buffer: &wgpu::Buffer,
        grid_v_buffer: &wgpu::Buffer,
        grid_w_buffer: &wgpu::Buffer,
        mobility: Option<[&wgpu::Buffer; 3]>,
    ) -> Self {
        let unused = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Constant pressure mobility header"),
            contents: &[0; 16],
            usage: wgpu::BufferUsages::STORAGE,
        });
        let variable_mobility_allowed = mobility.is_some();
        let mobility_constants = std::collections::HashMap::from([(
            "VARIABLE_MOBILITY_ALLOWED".to_string(),
            f64::from(u32::from(mobility.is_some())),
        )]);
        let mobility = mobility.unwrap_or([&unused; 3]);
        let cell_count = (width * height * depth) as usize;

        // Create shaders
        let divergence_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Divergence 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/divergence_3d.wgsl").into()),
        });

        let pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("shaders/pressure_mobility.wgsl"),
                    include_str!("shaders/pressure_3d.wgsl")
                )
                .into(),
            ),
        });

        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure Gradient 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("shaders/pressure_mobility.wgsl"),
                    include_str!("shaders/pressure_gradient_3d.wgsl")
                )
                .into(),
            ),
        });

        // Create buffers
        let pressure_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pressure 3D"),
            contents: &vec![0u8; cell_count * std::mem::size_of::<f32>()],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let pressure_backup_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure 3D Backup"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let divergence_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Divergence 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_type_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type 3D"),
            size: (cell_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let stencil_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure neighbour stencil"),
            size: (cell_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Params 3D"),
            size: std::mem::size_of::<GridParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Params 3D"),
            size: std::mem::size_of::<PressureParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create divergence bind group layout
        let divergence_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Divergence 3D Bind Group Layout"),
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
                    // 1: grid_u
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
                    // 2: grid_v
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
                    // 3: grid_w
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
                    // 4: cell_type
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
                    // 5: divergence (output)
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

        // Create pressure bind group layout
        let pressure_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pressure 3D Bind Group Layout"),
                entries: &[
                    // 0: pressure (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: divergence (read)
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
                    // 2: cell_type (read)
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
                    // 3: params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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
                ],
            });

        // Create gradient bind group layout
        let gradient_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gradient 3D Bind Group Layout"),
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
                    // 1: pressure (read)
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
                    // 2: cell_type (read)
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
                    // 3: grid_u (read_write)
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
                    // 4: grid_v (read_write)
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
                    // 5: grid_w (read_write)
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
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

        // Create bind groups
        let divergence_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Divergence 3D Bind Group"),
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
                    resource: grid_w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: divergence_buffer.as_entire_binding(),
                },
            ],
        });

        let pressure_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pressure 3D Bind Group"),
            layout: &pressure_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pressure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: divergence_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pressure_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: stencil_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mobility[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mobility[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: mobility[2].as_entire_binding(),
                },
            ],
        });

        let gradient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient 3D Bind Group"),
            layout: &gradient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_type_buffer.as_entire_binding(),
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
                    resource: grid_w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mobility[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: mobility[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: mobility[2].as_entire_binding(),
                },
            ],
        });

        // Create pipelines
        let divergence_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Divergence 3D Pipeline Layout"),
                bind_group_layouts: &[&divergence_bind_group_layout],
                push_constant_ranges: &[],
            });

        let divergence_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Divergence 3D Pipeline"),
                layout: Some(&divergence_pipeline_layout),
                module: &divergence_shader,
                entry_point: Some("compute_divergence"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pressure_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pressure 3D Pipeline Layout"),
                bind_group_layouts: &[&pressure_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pressure_red_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Red 3D Pipeline"),
                layout: Some(&pressure_pipeline_layout),
                module: &pressure_shader,
                entry_point: Some("pressure_red"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &mobility_constants,
                    ..Default::default()
                },
                cache: None,
            });

        let pressure_black_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Black 3D Pipeline"),
                layout: Some(&pressure_pipeline_layout),
                module: &pressure_shader,
                entry_point: Some("pressure_black"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &mobility_constants,
                    ..Default::default()
                },
                cache: None,
            });

        let gradient_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gradient 3D Pipeline Layout"),
                bind_group_layouts: &[&gradient_bind_group_layout],
                push_constant_ranges: &[],
            });

        let gradient_u_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gradient U 3D Pipeline"),
                layout: Some(&gradient_pipeline_layout),
                module: &gradient_shader,
                entry_point: Some("apply_gradient_u"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &mobility_constants,
                    ..Default::default()
                },
                cache: None,
            });

        let gradient_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gradient V 3D Pipeline"),
                layout: Some(&gradient_pipeline_layout),
                module: &gradient_shader,
                entry_point: Some("apply_gradient_v"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &mobility_constants,
                    ..Default::default()
                },
                cache: None,
            });

        let gradient_w_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gradient W 3D Pipeline"),
                layout: Some(&gradient_pipeline_layout),
                module: &gradient_shader,
                entry_point: Some("apply_gradient_w"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &mobility_constants,
                    ..Default::default()
                },
                cache: None,
            });

        let pipeline = |module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &mobility_constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let packed_red = pipeline(
            &pressure_shader,
            &pressure_pipeline_layout,
            "pressure_red_packed",
        );
        let packed_black = pipeline(
            &pressure_shader,
            &pressure_pipeline_layout,
            "pressure_black_packed",
        );
        let stencil_prepare = pipeline(
            &pressure_shader,
            &pressure_pipeline_layout,
            "prepare_stencil",
        );
        let stencil_red = pipeline(
            &pressure_shader,
            &pressure_pipeline_layout,
            "pressure_red_stencil",
        );
        let stencil_black = pipeline(
            &pressure_shader,
            &pressure_pipeline_layout,
            "pressure_black_stencil",
        );
        Self {
            compact_iterations: true,
            cached_stencil: true,
            stencil_prepare,
            stencil_red,
            stencil_black,
            packed_red,
            packed_black,
            width,
            height,
            depth,
            pressure_buffer,
            divergence_buffer,
            cell_type_buffer,
            pressure_backup_buffer,
            grid_params_buffer,
            pressure_params_buffer,
            divergence_pipeline,
            pressure_red_pipeline,
            pressure_black_pipeline,
            gradient_u_pipeline,
            gradient_v_pipeline,
            gradient_w_pipeline,
            divergence_bind_group,
            pressure_bind_group,
            pressure_bind_group_layout,
            variable_mobility_allowed,
            gradient_bind_group,
        }
    }

    /// Measure b-Ap for the actual stored equation, including variable MAC
    /// mobility, solid Neumann faces and zero-pressure air/open boundaries.
    /// Creates its pipeline and scratch/readback buffers only on this call.
    /// It does not clear pressure, recompute divergence, or apply a gradient.
    pub fn measure_residual(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<PressureResidual3D, String> {
        let count = u64::from(self.width) * u64::from(self.height) * u64::from(self.depth);
        if count == 0 {
            return Ok(PressureResidual3D::default());
        }
        let bytes = count * 16;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure residual diagnostic"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure residual readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pressure residual output"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pressure residual output"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure residual operator"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("shaders/pressure_mobility.wgsl"),
                    include_str!("shaders/pressure_3d.wgsl"),
                )
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pressure residual operator"),
            bind_group_layouts: &[&self.pressure_bind_group_layout, &layout],
            push_constant_ranges: &[],
        });
        let constants = std::collections::HashMap::from([(
            "VARIABLE_MOBILITY_ALLOWED".to_string(),
            f64::from(u32::from(self.variable_mobility_allowed)),
        )]);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Measure stored pressure residual"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("measure_residual"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &self.pressure_bind_group, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups(
                self.width.div_ceil(8),
                self.height.div_ceil(8),
                self.depth.div_ceil(4),
            );
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mut result = PressureResidual3D::default();
        let mut r2 = 0.;
        let mut b2 = 0.;
        let mut nonfinite = false;
        {
            let data = staging.slice(..).get_mapped_range();
            for row in bytemuck::cast_slice::<u8, [f32; 4]>(&data) {
                if row[3] == 0. {
                    continue;
                }
                result.fluid_cells += 1;
                nonfinite |= row.iter().any(|x| !x.is_finite());
                let r = f64::from(row[0]);
                let b = f64::from(row[1]);
                r2 += r * r;
                b2 += b * b;
                result.max_abs = result.max_abs.max(r.abs());
                if row[2] > 0. {
                    result.max_backward_error =
                        result.max_backward_error.max(r.abs() / f64::from(row[2]));
                }
            }
        }
        staging.unmap();
        if nonfinite {
            return Err("Nonfinite stored pressure equation or residual".into());
        }
        result.rms = (r2 / result.fluid_cells.max(1) as f64).sqrt();
        result.rhs_l2 = b2.sqrt();
        result.relative_l2 = if b2 > 0. {
            Some((r2 / b2).sqrt())
        } else if r2 == 0. {
            Some(0.)
        } else {
            None
        };
        Ok(result)
    }

    /// Upload cell types and initialize pressure to zero
    pub fn upload_cell_types(
        &self,
        queue: &wgpu::Queue,
        cell_types: &[u32],
        cell_size: f32,
        open_boundaries: u32,
    ) {
        queue.write_buffer(&self.cell_type_buffer, 0, bytemuck::cast_slice(cell_types));

        // Upload grid params
        let grid_params = GridParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            inv_cell_size: 1.0 / cell_size,
            open_boundaries,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(
            &self.grid_params_buffer,
            0,
            bytemuck::bytes_of(&grid_params),
        );

        // Upload pressure params
        // Retain the ordinary solver's existing largest-dimension estimate.
        // The weighted operator has a separate mixed-boundary estimate; neither
        // formula is a universal optimum for irregular variable coefficients.
        let max_dim = self.width.max(self.height).max(self.depth) as f32;
        let omega = 2.0 / (1.0 + (std::f32::consts::PI / max_dim).sin());
        let omega = omega.clamp(1.0, 1.95);
        let pressure_params = PressureParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            omega,
            h_sq: cell_size * cell_size, // Poisson equation needs dx²
            open_boundaries,
            phase_omega: if self.variable_mobility_allowed {
                phase_relaxation(
                    [self.width, self.height, self.depth],
                    open_boundaries,
                    cell_types.contains(&0),
                )
            } else {
                omega
            },
            _pad1: 0,
        };
        queue.write_buffer(
            &self.pressure_params_buffer,
            0,
            bytemuck::bytes_of(&pressure_params),
        );
    }

    /// Encode full pressure solve: divergence → iterations → gradient
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, iterations: u32) {
        self.encode_profiled(encoder, iterations, None);
    }

    pub fn encode_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
        timer: Option<&mut super::timing::GpuStageTimer>,
    ) {
        self.encode_with_rhs_profiled(encoder, iterations, timer, false, |_| {});
    }

    /// Insert a consistent divergence target before every pressure solve,
    /// including diagnostic reprojection. Ordinary projection uses no target.
    pub fn encode_with_rhs_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
        mut timer: Option<&mut super::timing::GpuStageTimer>,
        restore_pressure: bool,
        mut rhs: impl FnMut(&mut wgpu::CommandEncoder),
    ) {
        let bytes = (self.width * self.height * self.depth * 4) as u64;
        for solve in 0..if restore_pressure { 2 } else { 1 } {
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);

            // Compute divergence
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Divergence 3D Pass"),
                    timestamp_writes: timer
                        .as_deref_mut()
                        .map(|timer| timer.pass("Divergence 3D Pass")),
                });
                pass.set_pipeline(&self.divergence_pipeline);
                pass.set_bind_group(0, &self.divergence_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            rhs(encoder);
            self.encode_projection_profiled(encoder, iterations, timer.as_deref_mut());
            if restore_pressure && solve == 0 {
                encoder.copy_buffer_to_buffer(
                    &self.pressure_buffer,
                    0,
                    &self.pressure_backup_buffer,
                    0,
                    bytes,
                );
            }
        }
        if restore_pressure {
            encoder.copy_buffer_to_buffer(
                &self.pressure_backup_buffer,
                0,
                &self.pressure_buffer,
                0,
                bytes,
            );
        }
    }

    fn encode_projection_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
        mut timer: Option<&mut super::timing::GpuStageTimer>,
    ) {
        let workgroups_x = self.width.div_ceil(8);
        let workgroups_y = self.height.div_ceil(8);
        let workgroups_z = self.depth.div_ceil(4);
        self.encode_iterations_only_profiled(
            encoder,
            iterations,
            timer.as_deref_mut(),
            "Velocity pressure sweeps",
        );

        // Apply pressure gradient to U
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient U 3D Pass"),
                timestamp_writes: timer
                    .as_deref_mut()
                    .map(|timer| timer.pass("Gradient U 3D Pass")),
            });
            pass.set_pipeline(&self.gradient_u_pipeline);
            pass.set_bind_group(0, &self.gradient_bind_group, &[]);
            let workgroups_u_x = (self.width + 1).div_ceil(8);
            pass.dispatch_workgroups(workgroups_u_x, workgroups_y, workgroups_z);
        }

        // Apply pressure gradient to V
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient V 3D Pass"),
                timestamp_writes: timer
                    .as_deref_mut()
                    .map(|timer| timer.pass("Gradient V 3D Pass")),
            });
            pass.set_pipeline(&self.gradient_v_pipeline);
            pass.set_bind_group(0, &self.gradient_bind_group, &[]);
            let workgroups_v_y = (self.height + 1).div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, workgroups_v_y, workgroups_z);
        }

        // Apply pressure gradient to W
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient W 3D Pass"),
                timestamp_writes: timer.map(|timer| timer.pass("Gradient W 3D Pass")),
            });
            pass.set_pipeline(&self.gradient_w_pipeline);
            pass.set_bind_group(0, &self.gradient_bind_group, &[]);
            let workgroups_w_z = (self.depth + 1).div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_w_z);
        }
    }

    /// Encode two pressure solves but preserve the first pressure field for diagnostics.
    pub fn encode_with_pressure_restore(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
    ) {
        self.encode_with_pressure_restore_profiled(encoder, iterations, None);
    }

    pub fn encode_with_pressure_restore_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
        timer: Option<&mut super::timing::GpuStageTimer>,
    ) {
        self.encode_with_rhs_profiled(encoder, iterations, timer, true, |_| {});
    }

    /// Encode just the divergence pass (no pressure iterations or gradient apply).
    pub fn encode_divergence_only(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups_x = self.width.div_ceil(8);
        let workgroups_y = self.height.div_ceil(8);
        let workgroups_z = self.depth.div_ceil(4);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Divergence 3D Pass (Diagnostics)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.divergence_pipeline);
        pass.set_bind_group(0, &self.divergence_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }

    /// Clear the pressure buffer (needed before density projection)
    pub fn clear_pressure(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.pressure_buffer, 0, None);
    }

    /// Encode just the pressure iterations (no divergence compute, no gradient apply)
    ///
    /// This is used for density projection where:
    /// - divergence_buffer is pre-filled with density error
    /// - We want the resulting pressure to apply to particle positions, not grid velocities
    pub fn encode_iterations_only(&self, encoder: &mut wgpu::CommandEncoder, iterations: u32) {
        self.encode_iterations_only_profiled(encoder, iterations, None, "Pressure sweeps");
    }

    pub fn encode_iterations_only_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
        mut timer: Option<&mut super::timing::GpuStageTimer>,
        profile_label: &'static str,
    ) {
        let workgroups_x = self.width.div_ceil(8);
        let workgroups_y = self.height.div_ceil(8);
        let workgroups_z = self.depth.div_ceil(4);

        if self.compact_iterations {
            // Dispatches in a compute pass execute in order. Keep all red/black
            // sweeps and their storage dependencies, without 2*N pass encoders.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ordered red/black pressure sweeps"),
                timestamp_writes: timer.as_deref_mut().map(|timer| timer.pass(profile_label)),
            });
            pass.set_bind_group(0, &self.pressure_bind_group, &[]);
            if self.cached_stencil {
                pass.set_pipeline(&self.stencil_prepare);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            for _ in 0..iterations {
                pass.set_pipeline(if self.cached_stencil { &self.stencil_red } else { &self.packed_red });
                pass.dispatch_workgroups(self.width.div_ceil(64), self.height.div_ceil(2), self.depth.div_ceil(2));
                pass.set_pipeline(if self.cached_stencil { &self.stencil_black } else { &self.packed_black });
                pass.dispatch_workgroups(self.width.div_ceil(64), self.height.div_ceil(2), self.depth.div_ceil(2));
            }
            return;
        }
        // Red-Black Gauss-Seidel iterations
        for _ in 0..iterations {
            // Red pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Pressure Red 3D Pass"),
                    timestamp_writes: timer.as_deref_mut().map(|timer| timer.pass(profile_label)),
                });
                pass.set_pipeline(&self.pressure_red_pipeline);
                pass.set_bind_group(0, &self.pressure_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Black pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Pressure Black 3D Pass"),
                    timestamp_writes: timer.as_deref_mut().map(|timer| timer.pass(profile_label)),
                });
                pass.set_pipeline(&self.pressure_black_pipeline);
                pass.set_bind_group(0, &self.pressure_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::mem::{offset_of, size_of};
    use std::path::Path;

    struct WgslLayout {
        size: u32,
        offsets: HashMap<String, u32>,
    }

    fn wgsl_struct_layout(shader: &str, struct_name: &str) -> WgslLayout {
        let shader_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders").join(shader);
        let source = fs::read_to_string(&shader_path)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {e}", shader_path));
        let source = if shader == "pressure_3d.wgsl" {
            format!(
                "{}\n{source}",
                include_str!("shaders/pressure_mobility.wgsl")
            )
        } else {
            source
        };
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
            panic!(
                "Failed to parse {:?}: {}",
                shader_path,
                e.emit_to_string(&source)
            )
        });

        let mut layouter = naga::proc::Layouter::default();
        let gctx = naga::proc::GlobalCtx {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_expressions: &module.global_expressions,
        };
        layouter
            .update(gctx)
            .unwrap_or_else(|e| panic!("Failed to compute layout for {:?}: {e}", shader_path));

        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(struct_name))
            .unwrap_or_else(|| panic!("Struct {struct_name} not found in {:?}", shader_path));

        let members = match &ty.inner {
            naga::TypeInner::Struct { members, .. } => members,
            _ => panic!("Type {struct_name} is not a struct in {:?}", shader_path),
        };

        let mut offsets = HashMap::new();
        for member in members {
            if let Some(name) = &member.name {
                offsets.insert(name.clone(), member.offset);
            }
        }

        WgslLayout {
            size: layouter[handle].size,
            offsets,
        }
    }

    #[test]
    fn grid_params_layout_matches_wgsl() {
        let layout = wgsl_struct_layout("divergence_3d.wgsl", "Params");
        assert_eq!(layout.size as usize, size_of::<GridParams3D>());
        assert_eq!(
            *layout.offsets.get("open_boundaries").unwrap(),
            offset_of!(GridParams3D, open_boundaries) as u32
        );
        assert_eq!(
            *layout.offsets.get("inv_cell_size").unwrap(),
            offset_of!(GridParams3D, inv_cell_size) as u32
        );
    }

    #[test]
    fn pressure_params_layout_matches_wgsl() {
        let layout = wgsl_struct_layout("pressure_3d.wgsl", "Params");
        assert_eq!(layout.size as usize, size_of::<PressureParams3D>());
        assert_eq!(
            *layout.offsets.get("open_boundaries").unwrap(),
            offset_of!(PressureParams3D, open_boundaries) as u32
        );
        assert_eq!(
            *layout.offsets.get("h_sq").unwrap(),
            offset_of!(PressureParams3D, h_sq) as u32
        );
        assert_eq!(
            *layout.offsets.get("phase_omega").unwrap(),
            offset_of!(PressureParams3D, phase_omega) as u32
        );
    }
}
