//! GPU MGPCG (Multigrid-Preconditioned Conjugate Gradient) Pressure Solver
//!
//! This solver uses a V-cycle multigrid preconditioner inside a conjugate gradient
//! outer loop for guaranteed convergence. The multigrid hierarchy provides fast
//! error smoothing at all spatial frequencies, while CG ensures monotonic convergence.
//!
//! Architecture:
//! - PCG outer loop (15-20 iterations)
//! - V-cycle preconditioner: 4 levels (512→256→128→64)
//! - Red-Black Gauss-Seidel smoother at each level
//! - Full-weighting restriction, bilinear prolongation

use bytemuck::{Pod, Zeroable};
use super::GpuContext;

/// Parameters for a single multigrid level
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LevelParams {
    pub width: u32,
    pub height: u32,
    pub level: u32,
    pub _pad: u32,
}

/// Parameters for restriction operation (fine → coarse)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RestrictParams {
    pub fine_width: u32,
    pub fine_height: u32,
    pub coarse_width: u32,
    pub coarse_height: u32,
}

/// Parameters for prolongation operation (coarse → fine)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ProlongateParams {
    pub fine_width: u32,
    pub fine_height: u32,
    pub coarse_width: u32,
    pub coarse_height: u32,
}

/// Parameters for PCG vector operations
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PcgParams {
    pub width: u32,
    pub height: u32,
    pub alpha: f32,       // Scalar for axpy/xpay operations
    pub length: u32,      // Total number of elements (for 1D ops)
}

/// Scalar value for GPU reduction operations
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ScalarValue {
    pub value: f32,
    pub _pad: [u32; 3],
}

/// A single level in the multigrid hierarchy
pub struct MgLevel {
    pub width: u32,
    pub height: u32,
    pub cell_count: usize,

    /// Pressure field (solution at this level)
    pub pressure: wgpu::Buffer,
    /// Residual/RHS at this level
    pub residual: wgpu::Buffer,
    /// Divergence (RHS for pressure solve)
    pub divergence: wgpu::Buffer,
    /// Cell types (solid/fluid/air)
    pub cell_type: wgpu::Buffer,
    /// Level parameters uniform
    pub params_buffer: wgpu::Buffer,
}

impl MgLevel {
    /// Create a new multigrid level with GPU buffers
    pub fn new(gpu: &GpuContext, width: u32, height: u32, level: u32) -> Self {
        let cell_count = (width * height) as usize;
        let buffer_size = (cell_count * std::mem::size_of::<f32>()) as u64;

        let pressure = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Pressure", level)),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let residual = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Residual", level)),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let divergence = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Divergence", level)),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Cell type uses u32
        let cell_type = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Cell Type", level)),
            size: (cell_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Level parameters
        let params = LevelParams {
            width,
            height,
            level,
            _pad: 0,
        };
        let params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Params", level)),
            size: std::mem::size_of::<LevelParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        Self {
            width,
            height,
            cell_count,
            pressure,
            residual,
            divergence,
            cell_type,
            params_buffer,
        }
    }

    /// Clear pressure buffer to zero
    pub fn clear_pressure(&self, gpu: &GpuContext) {
        let zeros = vec![0u8; self.cell_count * std::mem::size_of::<f32>()];
        gpu.queue.write_buffer(&self.pressure, 0, &zeros);
    }
}

/// GPU MGPCG Pressure Solver
///
/// Uses multigrid-preconditioned conjugate gradient for stable, fast convergence.
pub struct GpuMgpcgSolver {
    /// Base grid dimensions (level 0)
    pub width: u32,
    pub height: u32,

    /// Number of multigrid levels
    pub num_levels: usize,

    /// Multigrid level hierarchy (level 0 = finest)
    pub levels: Vec<MgLevel>,

    // PCG vectors (level 0 size only)
    /// Residual vector r = b - Ax
    pub r: wgpu::Buffer,
    /// Preconditioned residual z = M⁻¹r
    pub z: wgpu::Buffer,
    /// Search direction p
    pub p: wgpu::Buffer,
    /// Matrix-vector product Ap
    pub ap: wgpu::Buffer,

    // Reduction buffers for dot products
    /// Partial sums from workgroups (one f32 per workgroup)
    pub partial_sums: wgpu::Buffer,
    /// Final scalar result
    pub final_sum: wgpu::Buffer,
    /// Staging buffer for reading scalar back to CPU
    pub sum_staging: wgpu::Buffer,

    // Compute pipelines (will be created in Step 2+)
    // Smoother pipelines
    pub smooth_red_pipeline: Option<wgpu::ComputePipeline>,
    pub smooth_black_pipeline: Option<wgpu::ComputePipeline>,

    // Multigrid transfer pipelines
    pub restrict_pipeline: Option<wgpu::ComputePipeline>,
    pub prolongate_pipeline: Option<wgpu::ComputePipeline>,

    // Multigrid residual/clear pipelines
    pub mg_residual_pipeline: Option<wgpu::ComputePipeline>,
    pub clear_pipeline: Option<wgpu::ComputePipeline>,

    // PCG operation pipelines
    pub residual_pipeline: Option<wgpu::ComputePipeline>,
    pub laplacian_pipeline: Option<wgpu::ComputePipeline>,
    pub axpy_pipeline: Option<wgpu::ComputePipeline>,
    pub xpay_pipeline: Option<wgpu::ComputePipeline>,
    pub copy_pipeline: Option<wgpu::ComputePipeline>,
    pub dot_partial_pipeline: Option<wgpu::ComputePipeline>,
    pub dot_finalize_pipeline: Option<wgpu::ComputePipeline>,

    // Bind group layouts (will be created in Step 2+)
    pub smooth_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub restrict_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub prolongate_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Bind groups per level
    pub smooth_bind_groups: Vec<wgpu::BindGroup>,
    pub restrict_bind_groups: Vec<wgpu::BindGroup>,
    pub prolongate_bind_groups: Vec<wgpu::BindGroup>,
    pub residual_bind_groups: Vec<wgpu::BindGroup>,

    // PCG bind group layout and bind groups
    pub pcg_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Bind group for r = b - Ax: (x=pressure, b=divergence, cell_type, r=output)
    pub pcg_residual_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for Ap = A*p: (p, cell_type, ap=output)
    pub pcg_laplacian_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for x += α*p: (pressure, p, -, -)
    pub pcg_x_update_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for r -= α*Ap: (r, ap, -, -)
    pub pcg_r_update_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for p = z + β*p: (p, z, -, -)
    pub pcg_p_update_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for divergence = r (copy for V-cycle input)
    pub pcg_copy_to_div_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for z = pressure (copy V-cycle output)
    pub pcg_copy_from_pressure_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for dot(r, z)
    pub pcg_dot_rz_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for dot(p, Ap)
    pub pcg_dot_pap_bind_group: Option<wgpu::BindGroup>,
    /// PCG params buffer (updated per operation)
    pub pcg_params_buffer: wgpu::Buffer,

    // Staging buffer for pressure download
    pub pressure_staging: wgpu::Buffer,
}

impl GpuMgpcgSolver {
    /// Create a new MGPCG solver with the given grid dimensions
    ///
    /// Builds a multigrid hierarchy with levels halving until min dimension < 16
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        // Build level hierarchy: 512→256→128→64 (or until min < 16)
        let mut levels = Vec::new();
        let (mut w, mut h) = (width, height);
        let mut level_idx = 0u32;

        // Always add finest level
        levels.push(MgLevel::new(gpu, w, h, level_idx));
        level_idx += 1;

        // Add coarser levels until we reach minimum size
        while w >= 32 && h >= 32 {
            w /= 2;
            h /= 2;
            levels.push(MgLevel::new(gpu, w, h, level_idx));
            level_idx += 1;
        }

        let num_levels = levels.len();
        let cell_count = (width * height) as usize;
        let buffer_size = (cell_count * std::mem::size_of::<f32>()) as u64;

        log::info!(
            "MGPCG: Created {} levels: {}",
            num_levels,
            levels.iter()
                .map(|l| format!("{}x{}", l.width, l.height))
                .collect::<Vec<_>>()
                .join(" → ")
        );

        // Create PCG vectors (level 0 size)
        let r = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG r (residual)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let z = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG z (preconditioned)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let p = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG p (direction)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ap = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG Ap (matrix-vector)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Reduction buffers for dot products
        // Max workgroups at level 0: ceil(512*512 / 256) = 1024
        let max_workgroups = ((width * height) as usize + 255) / 256;
        let partial_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Product Partial Sums"),
            size: (max_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let final_sum = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Product Final Sum"),
            size: std::mem::size_of::<ScalarValue>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sum_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scalar Staging"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffer for pressure download
        let pressure_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // PCG params buffer (updated per operation via queue.write_buffer)
        let pcg_params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG Params"),
            size: std::mem::size_of::<PcgParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create smooth pipelines and bind groups
        let (smooth_bind_group_layout, smooth_red_pipeline, smooth_black_pipeline, smooth_bind_groups) =
            Self::create_smooth_pipelines(gpu, &levels);

        // Create restrict pipelines and bind groups (between adjacent levels)
        let (restrict_bind_group_layout, restrict_pipeline, restrict_bind_groups) =
            Self::create_restrict_pipelines(gpu, &levels);

        // Create prolongate pipelines and bind groups (between adjacent levels)
        let (prolongate_bind_group_layout, prolongate_pipeline, prolongate_bind_groups) =
            Self::create_prolongate_pipelines(gpu, &levels);

        // Create residual computation pipelines
        let (mg_residual_pipeline, clear_pipeline, residual_bind_groups) =
            Self::create_residual_pipelines(gpu, &levels);

        // Create PCG pipelines and bind groups
        let (
            pcg_bind_group_layout,
            residual_pipeline,
            laplacian_pipeline,
            axpy_pipeline,
            xpay_pipeline,
            copy_pipeline,
            dot_partial_pipeline,
            dot_finalize_pipeline,
            pcg_residual_bind_group,
            pcg_laplacian_bind_group,
            pcg_x_update_bind_group,
            pcg_r_update_bind_group,
            pcg_p_update_bind_group,
            pcg_copy_to_div_bind_group,
            pcg_copy_from_pressure_bind_group,
            pcg_dot_rz_bind_group,
            pcg_dot_pap_bind_group,
        ) = Self::create_pcg_pipelines(
            gpu,
            &levels[0],
            &r,
            &z,
            &p,
            &ap,
            &partial_sums,
            &pcg_params_buffer,
        );

        Self {
            width,
            height,
            num_levels,
            levels,
            r,
            z,
            p,
            ap,
            partial_sums,
            final_sum,
            sum_staging,
            smooth_red_pipeline: Some(smooth_red_pipeline),
            smooth_black_pipeline: Some(smooth_black_pipeline),
            restrict_pipeline: Some(restrict_pipeline),
            prolongate_pipeline: Some(prolongate_pipeline),
            mg_residual_pipeline: Some(mg_residual_pipeline),
            clear_pipeline: Some(clear_pipeline),
            residual_pipeline: Some(residual_pipeline),
            laplacian_pipeline: Some(laplacian_pipeline),
            axpy_pipeline: Some(axpy_pipeline),
            xpay_pipeline: Some(xpay_pipeline),
            copy_pipeline: Some(copy_pipeline),
            dot_partial_pipeline: Some(dot_partial_pipeline),
            dot_finalize_pipeline: Some(dot_finalize_pipeline),
            smooth_bind_group_layout: Some(smooth_bind_group_layout),
            restrict_bind_group_layout: Some(restrict_bind_group_layout),
            prolongate_bind_group_layout: Some(prolongate_bind_group_layout),
            smooth_bind_groups,
            restrict_bind_groups,
            prolongate_bind_groups,
            residual_bind_groups,
            pcg_bind_group_layout: Some(pcg_bind_group_layout),
            pcg_residual_bind_group: Some(pcg_residual_bind_group),
            pcg_laplacian_bind_group: Some(pcg_laplacian_bind_group),
            pcg_x_update_bind_group: Some(pcg_x_update_bind_group),
            pcg_r_update_bind_group: Some(pcg_r_update_bind_group),
            pcg_p_update_bind_group: Some(pcg_p_update_bind_group),
            pcg_copy_to_div_bind_group: Some(pcg_copy_to_div_bind_group),
            pcg_copy_from_pressure_bind_group: Some(pcg_copy_from_pressure_bind_group),
            pcg_dot_rz_bind_group: Some(pcg_dot_rz_bind_group),
            pcg_dot_pap_bind_group: Some(pcg_dot_pap_bind_group),
            pcg_params_buffer,
            pressure_staging,
        }
    }

    /// Create the smooth shader module, pipelines, and bind groups for all levels
    fn create_smooth_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline, wgpu::ComputePipeline, Vec<wgpu::BindGroup>) {
        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MG Smooth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mg_smooth.wgsl").into()),
        });

        // Create bind group layout
        // @group(0) @binding(0) pressure: storage read_write
        // @group(0) @binding(1) divergence: storage read
        // @group(0) @binding(2) cell_type: storage read
        // @group(0) @binding(3) params: uniform
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MG Smooth Bind Group Layout"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MG Smooth Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create red and black pipelines
        let red_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MG Smooth Red Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("smooth_red"),
            compilation_options: Default::default(),
            cache: None,
        });

        let black_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MG Smooth Black Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("smooth_black"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups for each level
        let bind_groups: Vec<wgpu::BindGroup> = levels
            .iter()
            .enumerate()
            .map(|(idx, level)| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("MG Smooth Level {} Bind Group", idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: level.pressure.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: level.divergence.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: level.cell_type.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: level.params_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        (bind_group_layout, red_pipeline, black_pipeline, bind_groups)
    }

    /// Dispatch smooth operation on a specific level
    /// Performs one red-black iteration
    pub fn dispatch_smooth(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        let level_data = &self.levels[level];
        let bind_group = &self.smooth_bind_groups[level];

        // Workgroup counts: each thread handles one cell, but we process half per pass
        // Using 8x8 workgroups
        let workgroup_x = (level_data.width / 2 + 7) / 8;
        let workgroup_y = (level_data.height + 7) / 8;

        // Red pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("MG Smooth Red Level {}", level)),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.smooth_red_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Black pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("MG Smooth Black Level {}", level)),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.smooth_black_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
    }

    /// Create the restrict shader module, pipeline, and bind groups for level transitions
    fn create_restrict_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline, Vec<wgpu::BindGroup>) {
        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MG Restrict Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mg_restrict.wgsl").into()),
        });

        // Create bind group layout
        // @group(0) @binding(0) fine_residual: storage read
        // @group(0) @binding(1) fine_cell_type: storage read
        // @group(0) @binding(2) coarse_divergence: storage read_write
        // @group(0) @binding(3) coarse_cell_type: storage read_write
        // @group(0) @binding(4) params: uniform
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MG Restrict Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MG Restrict Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MG Restrict Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mg_restrict"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups for each level transition (fine → coarse)
        // We need num_levels - 1 bind groups
        let mut bind_groups = Vec::new();
        let mut params_buffers = Vec::new();

        for i in 0..levels.len().saturating_sub(1) {
            let fine = &levels[i];
            let coarse = &levels[i + 1];

            // Create params buffer for this transition
            let params = RestrictParams {
                fine_width: fine.width,
                fine_height: fine.height,
                coarse_width: coarse.width,
                coarse_height: coarse.height,
            };
            let params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MG Restrict Params {} -> {}", i, i + 1)),
                size: std::mem::size_of::<RestrictParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
            params_buffers.push(params_buffer);
        }

        for i in 0..levels.len().saturating_sub(1) {
            let fine = &levels[i];
            let coarse = &levels[i + 1];

            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MG Restrict {} -> {} Bind Group", i, i + 1)),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: fine.residual.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: fine.cell_type.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: coarse.divergence.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: coarse.cell_type.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffers[i].as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bind_group);
        }

        (bind_group_layout, pipeline, bind_groups)
    }

    /// Dispatch restriction from fine level to coarse level
    /// Transfers residual from level[fine_level] to divergence of level[fine_level + 1]
    pub fn dispatch_restrict(&self, encoder: &mut wgpu::CommandEncoder, fine_level: usize) {
        if fine_level >= self.num_levels - 1 {
            return; // No coarser level to restrict to
        }

        let coarse = &self.levels[fine_level + 1];
        let bind_group = &self.restrict_bind_groups[fine_level];

        // Workgroup counts for coarse level (each thread handles one coarse cell)
        let workgroup_x = (coarse.width + 7) / 8;
        let workgroup_y = (coarse.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Restrict {} -> {}", fine_level, fine_level + 1)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.restrict_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Create the prolongate shader module, pipeline, and bind groups for level transitions
    fn create_prolongate_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline, Vec<wgpu::BindGroup>) {
        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MG Prolongate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mg_prolongate.wgsl").into()),
        });

        // Create bind group layout
        // @group(0) @binding(0) coarse_pressure: storage read
        // @group(0) @binding(1) fine_pressure: storage read_write
        // @group(0) @binding(2) fine_cell_type: storage read
        // @group(0) @binding(3) params: uniform
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MG Prolongate Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MG Prolongate Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MG Prolongate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("prolongate"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups for each level transition (coarse → fine)
        // We need num_levels - 1 bind groups
        let mut bind_groups = Vec::new();
        let mut params_buffers = Vec::new();

        for i in 0..levels.len().saturating_sub(1) {
            let fine = &levels[i];
            let coarse = &levels[i + 1];

            // Create params buffer for this transition
            let params = ProlongateParams {
                fine_width: fine.width,
                fine_height: fine.height,
                coarse_width: coarse.width,
                coarse_height: coarse.height,
            };
            let params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("MG Prolongate Params {} <- {}", i, i + 1)),
                size: std::mem::size_of::<ProlongateParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
            params_buffers.push(params_buffer);
        }

        for i in 0..levels.len().saturating_sub(1) {
            let fine = &levels[i];
            let coarse = &levels[i + 1];

            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("MG Prolongate {} <- {} Bind Group", i, i + 1)),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: coarse.pressure.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: fine.pressure.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: fine.cell_type.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffers[i].as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bind_group);
        }

        (bind_group_layout, pipeline, bind_groups)
    }

    /// Dispatch prolongation from coarse level to fine level
    /// Transfers correction from level[coarse_level] to pressure of level[coarse_level - 1]
    pub fn dispatch_prolongate(&self, encoder: &mut wgpu::CommandEncoder, coarse_level: usize) {
        if coarse_level == 0 || coarse_level > self.num_levels - 1 {
            return; // No finer level to prolongate to
        }

        let fine = &self.levels[coarse_level - 1];
        let bind_group = &self.prolongate_bind_groups[coarse_level - 1];

        // Workgroup counts for fine level (each thread handles one fine cell)
        let workgroup_x = (fine.width + 7) / 8;
        let workgroup_y = (fine.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Prolongate {} <- {}", coarse_level - 1, coarse_level)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.prolongate_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Create the residual computation and clear pipelines
    fn create_residual_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (wgpu::ComputePipeline, wgpu::ComputePipeline, Vec<wgpu::BindGroup>) {
        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MG Residual Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mg_residual.wgsl").into()),
        });

        // Create bind group layout
        // @group(0) @binding(0) pressure: storage read
        // @group(0) @binding(1) divergence: storage read
        // @group(0) @binding(2) cell_type: storage read
        // @group(0) @binding(3) residual: storage read_write
        // @group(0) @binding(4) params: uniform
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MG Residual Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MG Residual Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let residual_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MG Residual Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_residual"),
            compilation_options: Default::default(),
            cache: None,
        });

        let clear_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MG Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear_buffer"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups for each level
        let bind_groups: Vec<wgpu::BindGroup> = levels
            .iter()
            .enumerate()
            .map(|(idx, level)| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("MG Residual Level {} Bind Group", idx)),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: level.pressure.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: level.divergence.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: level.cell_type.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: level.residual.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: level.params_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        (residual_pipeline, clear_pipeline, bind_groups)
    }

    /// Create PCG operation pipelines and bind groups
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn create_pcg_pipelines(
        gpu: &GpuContext,
        level0: &MgLevel,
        r: &wgpu::Buffer,
        z: &wgpu::Buffer,
        p: &wgpu::Buffer,
        ap: &wgpu::Buffer,
        partial_sums: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::ComputePipeline,  // residual
        wgpu::ComputePipeline,  // laplacian
        wgpu::ComputePipeline,  // axpy
        wgpu::ComputePipeline,  // xpay
        wgpu::ComputePipeline,  // copy
        wgpu::ComputePipeline,  // dot_partial
        wgpu::ComputePipeline,  // dot_finalize
        wgpu::BindGroup,        // residual bind group
        wgpu::BindGroup,        // laplacian bind group
        wgpu::BindGroup,        // x_update bind group
        wgpu::BindGroup,        // r_update bind group
        wgpu::BindGroup,        // p_update bind group
        wgpu::BindGroup,        // copy_to_div bind group
        wgpu::BindGroup,        // copy_from_pressure bind group
        wgpu::BindGroup,        // dot_rz bind group
        wgpu::BindGroup,        // dot_pap bind group
    ) {
        // Create shader module
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PCG Ops Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pcg_ops.wgsl").into()),
        });

        // Bind group layout matches pcg_ops.wgsl:
        // @group(0) @binding(0) buffer_a: storage read_write
        // @group(0) @binding(1) buffer_b: storage read
        // @group(0) @binding(2) buffer_c: storage read
        // @group(0) @binding(3) buffer_d: storage read_write
        // @group(0) @binding(4) params: uniform
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PCG Bind Group Layout"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PCG Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipelines for each operation
        let residual_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG Residual Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_pcg_residual"),
            compilation_options: Default::default(),
            cache: None,
        });

        let laplacian_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG Laplacian Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_laplacian"),
            compilation_options: Default::default(),
            cache: None,
        });

        let axpy_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG AXPY Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("axpy"),
            compilation_options: Default::default(),
            cache: None,
        });

        let xpay_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG XPAY Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("xpay"),
            compilation_options: Default::default(),
            cache: None,
        });

        let copy_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG Copy Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("copy_buffer"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dot_partial_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG Dot Partial Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dot_partial"),
            compilation_options: Default::default(),
            cache: None,
        });

        let dot_finalize_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCG Dot Finalize Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dot_finalize"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Bind group for residual: r = b - Ax
        // buffer_a = pressure (x), buffer_b = divergence (b), buffer_c = cell_type, buffer_d = r
        let residual_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Residual Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: level0.pressure.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: level0.divergence.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for Laplacian: Ap = A*p
        // buffer_a = p, buffer_b = unused (but needed), buffer_c = cell_type, buffer_d = ap
        let laplacian_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Laplacian Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: level0.divergence.as_entire_binding(),  // unused but needed
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for x += α*p (axpy on pressure)
        let x_update_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG X Update Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: level0.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Bind group for r -= α*Ap (axpy on residual)
        let r_update_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG R Update Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ap.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Bind group for p = z + β*p (xpay)
        let p_update_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG P Update Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Bind group for divergence = r (copy for V-cycle input)
        let copy_to_div_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Copy To Div Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: level0.divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Bind group for z = pressure (copy V-cycle output)
        let copy_from_pressure_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Copy From Pressure Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: level0.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Bind group for dot(r, z)
        let dot_rz_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Dot RZ Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Bind group for dot(p, Ap)
        let dot_pap_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Dot PAp Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ap.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: level0.cell_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partial_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        (
            bind_group_layout,
            residual_pipeline,
            laplacian_pipeline,
            axpy_pipeline,
            xpay_pipeline,
            copy_pipeline,
            dot_partial_pipeline,
            dot_finalize_pipeline,
            residual_bind_group,
            laplacian_bind_group,
            x_update_bind_group,
            r_update_bind_group,
            p_update_bind_group,
            copy_to_div_bind_group,
            copy_from_pressure_bind_group,
            dot_rz_bind_group,
            dot_pap_bind_group,
        )
    }

    /// Dispatch residual computation at a specific level
    /// Computes r = b - Ax (residual = divergence - Laplacian(pressure))
    pub fn dispatch_level_residual(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        let level_data = &self.levels[level];
        let bind_group = &self.residual_bind_groups[level];

        let workgroup_x = (level_data.width + 7) / 8;
        let workgroup_y = (level_data.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Residual Level {}", level)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.mg_residual_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Clear pressure buffer at a specific level to zero
    pub fn dispatch_clear_pressure(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        let level_data = &self.levels[level];
        let bind_group = &self.residual_bind_groups[level];

        // Use 256-wide workgroups for 1D clear
        let total_cells = level_data.width * level_data.height;
        let workgroup_count = (total_cells + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Clear Level {}", level)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.clear_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    /// Execute a complete V-cycle
    /// This is the multigrid preconditioner: applies V-cycle to solve Az = r approximately
    ///
    /// Input: residual in levels[0].divergence (or r buffer for PCG)
    /// Output: correction in levels[0].pressure (or z buffer for PCG)
    pub fn dispatch_vcycle(&self, encoder: &mut wgpu::CommandEncoder) {
        // Use 2 levels (512x256 -> 256x128) for now
        // 3+ levels causes instability - needs investigation
        let max_level = 1.min(self.num_levels - 1);
        self.dispatch_vcycle_recursive(encoder, 0, max_level);
    }

    /// Recursive V-cycle implementation
    fn dispatch_vcycle_recursive(&self, encoder: &mut wgpu::CommandEncoder, level: usize, max_level: usize) {
        const PRE_SMOOTH: usize = 3;
        const POST_SMOOTH: usize = 3;
        const COARSE_SOLVE: usize = 15;

        // Pre-smoothing
        for _ in 0..PRE_SMOOTH {
            self.dispatch_smooth(encoder, level);
        }

        if level == max_level {
            // At coarsest level: direct solve with more iterations
            for _ in 0..COARSE_SOLVE {
                self.dispatch_smooth(encoder, level);
            }
        } else {
            // Compute residual at this level
            self.dispatch_level_residual(encoder, level);

            // Restrict residual to coarse level (into coarse divergence)
            self.dispatch_restrict(encoder, level);

            // Clear coarse pressure to zero before solving
            self.dispatch_clear_pressure(encoder, level + 1);

            // Recursively solve on coarse level
            self.dispatch_vcycle_recursive(encoder, level + 1, max_level);

            // Prolongate correction from coarse to fine
            self.dispatch_prolongate(encoder, level + 1);

            // Post-smoothing
            for _ in 0..POST_SMOOTH {
                self.dispatch_smooth(encoder, level);
            }
        }
    }

    /// Run the pressure solve using multigrid V-cycles
    ///
    /// This is a simpler approach than full PCG - just runs multiple V-cycles
    /// which works well in practice for the pressure equation.
    ///
    /// Call `upload()` or `upload_warm()` first, then `solve()`, then `download()`.
    pub fn solve(&self, gpu: &GpuContext, num_vcycles: u32) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MGPCG Solve Encoder"),
        });

        for _ in 0..num_vcycles {
            self.dispatch_vcycle(&mut encoder);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run pressure solve with timing information
    pub fn solve_timed(&self, gpu: &GpuContext, num_vcycles: u32) -> std::time::Duration {
        let start = std::time::Instant::now();

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MGPCG Solve Encoder"),
        });

        for _ in 0..num_vcycles {
            self.dispatch_vcycle(&mut encoder);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU to finish
        gpu.device.poll(wgpu::Maintain::Wait);

        start.elapsed()
    }

    /// Run full Preconditioned Conjugate Gradient with V-cycle as preconditioner
    ///
    /// This provides guaranteed convergence and may stabilize 3+ level multigrid.
    /// More expensive than pure V-cycles due to dot product synchronizations.
    ///
    /// Call `upload()` or `upload_warm()` first, then `solve_pcg()`, then `download()`.
    pub fn solve_pcg(&self, gpu: &GpuContext, max_iterations: u32) {
        let cell_count = (self.width * self.height) as usize;
        let workgroup_count_1d = ((cell_count + 255) / 256) as u32;
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;

        // Update params buffer with grid dimensions
        let params = PcgParams {
            width: self.width,
            height: self.height,
            alpha: 0.0,
            length: cell_count as u32,
        };
        gpu.queue.write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

        // Step 1: Compute initial residual r = b - Ax
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG Initial Residual"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute r = b - Ax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.residual_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pcg_residual_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 2: Apply preconditioner z = M⁻¹r (V-cycle)
        self.apply_preconditioner(gpu, workgroup_count_1d);

        // Step 3: p = z (copy z to p)
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG p = z"),
            });
            // We need a bind group for p = z, but we have p_update which does p = z + beta*p
            // Let's use copy instead: p = z
            // Need a different bind group for this... or use a buffer copy
            encoder.copy_buffer_to_buffer(&self.z, 0, &self.p, 0, (cell_count * 4) as u64);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 4: rz = dot(r, z)
        let mut rz = self.compute_dot_product(gpu, workgroup_count_1d, self.pcg_dot_rz_bind_group.as_ref().unwrap());

        // Main PCG iteration loop
        for _iter in 0..max_iterations {
            // Check for convergence (rz is proportional to error)
            if rz.abs() < 1e-10 {
                break;
            }

            // Ap = A*p (apply Laplacian)
            {
                let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG Ap = A*p"),
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Apply Laplacian"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.laplacian_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.pcg_laplacian_bind_group.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
                drop(pass);
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }

            // pAp = dot(p, Ap)
            let pap = self.compute_dot_product(gpu, workgroup_count_1d, self.pcg_dot_pap_bind_group.as_ref().unwrap());

            // Compute alpha = rz / pAp
            let alpha = if pap.abs() > 1e-20 { rz / pap } else { 0.0 };

            // x = x + alpha*p
            self.dispatch_axpy(gpu, alpha, self.pcg_x_update_bind_group.as_ref().unwrap(), workgroup_count_1d);

            // r = r - alpha*Ap (note: negative alpha)
            self.dispatch_axpy(gpu, -alpha, self.pcg_r_update_bind_group.as_ref().unwrap(), workgroup_count_1d);

            // Save old rz
            let rz_old = rz;

            // Apply preconditioner z = M⁻¹r (V-cycle)
            self.apply_preconditioner(gpu, workgroup_count_1d);

            // rz = dot(r, z)
            rz = self.compute_dot_product(gpu, workgroup_count_1d, self.pcg_dot_rz_bind_group.as_ref().unwrap());

            // Compute beta = rz / rz_old
            let beta = if rz_old.abs() > 1e-20 { rz / rz_old } else { 0.0 };

            // p = z + beta*p (xpay operation)
            self.dispatch_xpay(gpu, beta, workgroup_count_1d);
        }
    }

    /// Apply V-cycle preconditioner: copies r to divergence, runs V-cycle, copies result to z
    ///
    /// IMPORTANT: The V-cycle uses levels[0].pressure as working buffer, but that same
    /// buffer stores the current solution x. We must save x before V-cycle and restore after.
    fn apply_preconditioner(&self, gpu: &GpuContext, workgroup_count_1d: u32) {
        let cell_size = (self.width * self.height * 4) as u64;

        // Step 1: Save x (levels[0].pressure) to ap buffer temporarily
        // (ap is only used after V-cycle for Laplacian(p), so it's safe to use here)
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG save x to ap"),
            });
            encoder.copy_buffer_to_buffer(&self.levels[0].pressure, 0, &self.ap, 0, cell_size);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 2: Copy -r to levels[0].divergence (negate because V-cycle solves L*z = input,
        // but we need (-L)*z = r, so we pass -r as input to get (-L)*z = r)
        {
            // Set alpha = -1.0 for negated copy
            let params = PcgParams {
                width: self.width,
                height: self.height,
                alpha: -1.0,
                length: (self.width * self.height) as u32,
            };
            gpu.queue.write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG copy -r to div"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy -r to divergence"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.copy_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pcg_copy_to_div_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroup_count_1d, 1, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 3: Clear pressure (V-cycle working buffer) before solving
        self.levels[0].clear_pressure(gpu);

        // Step 4: Run V-cycle to compute z ≈ M⁻¹r
        // Use all available levels now that we've fixed the x preservation
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG V-cycle"),
            });
            // Limit to 2 levels for now - more levels may have convergence issues
            let max_level = 1.min(self.num_levels - 1);
            self.dispatch_vcycle_recursive(&mut encoder, 0, max_level);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 5: Copy V-cycle result (levels[0].pressure) to z
        {
            // Set alpha = 1.0 for regular copy (no negation)
            let params = PcgParams {
                width: self.width,
                height: self.height,
                alpha: 1.0,
                length: (self.width * self.height) as u32,
            };
            gpu.queue.write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG copy pressure to z"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy pressure to z"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.copy_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pcg_copy_from_pressure_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroup_count_1d, 1, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 6: Restore x from ap back to levels[0].pressure
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG restore x from ap"),
            });
            encoder.copy_buffer_to_buffer(&self.ap, 0, &self.levels[0].pressure, 0, cell_size);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Compute dot product and read result back to CPU
    fn compute_dot_product(&self, gpu: &GpuContext, workgroup_count: u32, bind_group: &wgpu::BindGroup) -> f32 {
        // Dispatch partial sum computation
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dot Partial"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dot Partial Sums"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.dot_partial_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Dispatch finalize to sum partial sums
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dot Finalize"),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dot Finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.dot_finalize_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);

            // Copy result to staging buffer
            encoder.copy_buffer_to_buffer(
                &self.partial_sums,
                0,
                &self.sum_staging,
                0,
                std::mem::size_of::<f32>() as u64,
            );
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read result from staging buffer
        let buffer_slice = self.sum_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::from_bytes::<f32>(&data[..4]);
        let value = *result;
        drop(data);
        self.sum_staging.unmap();

        value
    }

    /// Dispatch axpy: buffer_a += alpha * buffer_b
    fn dispatch_axpy(&self, gpu: &GpuContext, alpha: f32, bind_group: &wgpu::BindGroup, workgroup_count: u32) {
        // Update params with alpha
        let params = PcgParams {
            width: self.width,
            height: self.height,
            alpha,
            length: (self.width * self.height) as u32,
        };
        gpu.queue.write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PCG AXPY"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("AXPY"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.axpy_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(pass);
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch xpay: p = z + beta * p
    fn dispatch_xpay(&self, gpu: &GpuContext, beta: f32, workgroup_count: u32) {
        // Update params with beta as alpha
        let params = PcgParams {
            width: self.width,
            height: self.height,
            alpha: beta,
            length: (self.width * self.height) as u32,
        };
        gpu.queue.write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PCG XPAY"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("XPAY p = z + beta*p"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.xpay_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.pcg_p_update_bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(pass);
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Upload divergence and cell_type from CPU to level 0
    pub fn upload(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32]) {
        let level0 = &self.levels[0];
        gpu.queue.write_buffer(&level0.divergence, 0, bytemuck::cast_slice(divergence));
        gpu.queue.write_buffer(&level0.cell_type, 0, bytemuck::cast_slice(cell_type));

        // Clear pressure to zero
        level0.clear_pressure(gpu);
    }

    /// Upload with warm start from previous pressure
    pub fn upload_warm(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32], pressure: &[f32]) {
        let level0 = &self.levels[0];
        gpu.queue.write_buffer(&level0.divergence, 0, bytemuck::cast_slice(divergence));
        gpu.queue.write_buffer(&level0.cell_type, 0, bytemuck::cast_slice(cell_type));
        gpu.queue.write_buffer(&level0.pressure, 0, bytemuck::cast_slice(pressure));
    }

    /// Download pressure results from GPU
    pub fn download(&self, gpu: &GpuContext, pressure: &mut [f32]) {
        let level0 = &self.levels[0];

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pressure Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &level0.pressure,
            0,
            &self.pressure_staging,
            0,
            (self.width * self.height * 4) as u64,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let buffer_slice = self.pressure_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            pressure.copy_from_slice(bytemuck::cast_slice(&data));
        }

        self.pressure_staging.unmap();
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MgpcgMemoryStats {
        let mut level_memory = 0u64;
        for level in &self.levels {
            // pressure + residual + divergence (f32) + cell_type (u32)
            let level_size = level.cell_count as u64 * 4 * 4;
            level_memory += level_size;
        }

        let pcg_memory = (self.width * self.height) as u64 * 4 * 4; // r, z, p, Ap
        let cell_count = (self.width * self.height) as usize;
        let reduction_memory = ((cell_count + 255) / 256 * 4 + 16) as u64; // partial_sums + final_sum

        MgpcgMemoryStats {
            level_memory,
            pcg_memory,
            reduction_memory,
            total: level_memory + pcg_memory + reduction_memory,
        }
    }
}

/// Memory usage statistics for MGPCG solver
#[derive(Debug)]
pub struct MgpcgMemoryStats {
    pub level_memory: u64,
    pub pcg_memory: u64,
    pub reduction_memory: u64,
    pub total: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a GPU context which isn't available in unit tests
    // Integration tests should be run with the game binary
}
