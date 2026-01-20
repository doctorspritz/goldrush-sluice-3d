//! GPU pipeline creation for MGPCG solver.
//!
//! This module contains all the pipeline and bind group creation methods
//! for the multigrid and PCG operations.

use super::{GpuMgpcgSolver, MgLevel, ProlongateParams, RestrictParams};
use crate::gpu::GpuContext;

impl GpuMgpcgSolver {
    /// Create the smooth shader module, pipelines, and bind groups for all levels
    pub(super) fn create_smooth_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        Vec<wgpu::BindGroup>,
    ) {
        // Create shader module
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MG Smooth Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mg_smooth.wgsl").into()),
            });

        // Create bind group layout
        // @group(0) @binding(0) pressure: storage read_write
        // @group(0) @binding(1) divergence: storage read
        // @group(0) @binding(2) cell_type: storage read
        // @group(0) @binding(3) params: uniform
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MG Smooth Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create red and black pipelines
        let red_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MG Smooth Red Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("smooth_red"),
                compilation_options: Default::default(),
                cache: None,
            });

        let black_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

    /// Create the restrict shader module, pipeline, and bind groups for level transitions
    pub(super) fn create_restrict_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::ComputePipeline,
        Vec<wgpu::BindGroup>,
    ) {
        // Create shader module
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MG Restrict Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mg_restrict.wgsl").into()),
            });

        // Create bind group layout
        // @group(0) @binding(0) fine_residual: storage read
        // @group(0) @binding(1) fine_cell_type: storage read
        // @group(0) @binding(2) coarse_divergence: storage read_write
        // @group(0) @binding(3) coarse_cell_type: storage read_write
        // @group(0) @binding(4) params: uniform
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MG Restrict Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
            gpu.queue
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
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

    /// Create the prolongate shader module, pipeline, and bind groups for level transitions
    pub(super) fn create_prolongate_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::ComputePipeline,
        Vec<wgpu::BindGroup>,
    ) {
        // Create shader module
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MG Prolongate Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mg_prolongate.wgsl").into()),
            });

        // Create bind group layout
        // @group(0) @binding(0) coarse_pressure: storage read
        // @group(0) @binding(1) fine_pressure: storage read_write
        // @group(0) @binding(2) fine_cell_type: storage read
        // @group(0) @binding(3) params: uniform
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MG Prolongate Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
            gpu.queue
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
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

    /// Create the residual computation and clear pipelines
    pub(super) fn create_residual_pipelines(
        gpu: &GpuContext,
        levels: &[MgLevel],
    ) -> (
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        Vec<wgpu::BindGroup>,
    ) {
        // Create shader module
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MG Residual Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mg_residual.wgsl").into()),
            });

        // Create bind group layout
        // @group(0) @binding(0) pressure: storage read
        // @group(0) @binding(1) divergence: storage read
        // @group(0) @binding(2) cell_type: storage read
        // @group(0) @binding(3) residual: storage read_write
        // @group(0) @binding(4) params: uniform
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MG Residual Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let residual_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MG Residual Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("compute_residual"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let clear_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
    pub(super) fn create_pcg_pipelines(
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
        wgpu::ComputePipeline, // residual
        wgpu::ComputePipeline, // laplacian
        wgpu::ComputePipeline, // axpy
        wgpu::ComputePipeline, // xpay
        wgpu::ComputePipeline, // copy
        wgpu::ComputePipeline, // dot_partial
        wgpu::ComputePipeline, // dot_finalize
        wgpu::BindGroup,       // residual bind group
        wgpu::BindGroup,       // laplacian bind group
        wgpu::BindGroup,       // x_update bind group
        wgpu::BindGroup,       // r_update bind group
        wgpu::BindGroup,       // p_update bind group
        wgpu::BindGroup,       // copy_to_div bind group
        wgpu::BindGroup,       // copy_from_pressure bind group
        wgpu::BindGroup,       // dot_rz bind group
        wgpu::BindGroup,       // dot_pap bind group
    ) {
        // Create shader module
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("PCG Ops Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pcg_ops.wgsl").into()),
            });

        // Bind group layout matches pcg_ops.wgsl:
        // @group(0) @binding(0) buffer_a: storage read_write
        // @group(0) @binding(1) buffer_b: storage read
        // @group(0) @binding(2) buffer_c: storage read
        // @group(0) @binding(3) buffer_d: storage read_write
        // @group(0) @binding(4) params: uniform
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PCG Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create pipelines for each operation
        let residual_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("PCG Residual Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("compute_pcg_residual"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let laplacian_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("PCG Laplacian Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("apply_laplacian"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let axpy_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PCG AXPY Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("axpy"),
                compilation_options: Default::default(),
                cache: None,
            });

        let xpay_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PCG XPAY Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("xpay"),
                compilation_options: Default::default(),
                cache: None,
            });

        let copy_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PCG Copy Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("copy_buffer"),
                compilation_options: Default::default(),
                cache: None,
            });

        let dot_partial_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("PCG Dot Partial Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("dot_partial"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let dot_finalize_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
                    resource: level0.divergence.as_entire_binding(), // unused but needed
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
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: level0.pressure.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for r -= α*Ap (axpy on residual)
        let r_update_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG R Update Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for p = z + β*p (xpay)
        let p_update_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG P Update Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for divergence = r (copy for V-cycle input)
        let copy_to_div_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Copy To Div Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: level0.divergence.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for z = pressure (copy V-cycle output)
        let copy_from_pressure_bind_group =
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PCG Copy From Pressure Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: z.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: level0.pressure.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: level0.cell_type.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: partial_sums.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Bind group for dot(r, z)
        let dot_rz_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Dot RZ Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind group for dot(p, Ap)
        let dot_pap_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCG Dot PAp Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.cell_type.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
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
}
