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
    _pad0: u32,
    _pad1: u32,
}

/// GPU-based pressure solver for 3D FLIP
pub struct GpuPressure3D {
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
    gradient_u_pipeline: wgpu::ComputePipeline,
    gradient_v_pipeline: wgpu::ComputePipeline,
    gradient_w_pipeline: wgpu::ComputePipeline,

    // Bind groups
    divergence_bind_group: wgpu::BindGroup,
    pressure_bind_group: wgpu::BindGroup,
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
        let cell_count = (width * height * depth) as usize;

        // Create shaders
        let divergence_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Divergence 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/divergence_3d.wgsl").into()),
        });

        let pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pressure_3d.wgsl").into()),
        });

        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pressure Gradient 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/pressure_gradient_3d.wgsl").into(),
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
                compilation_options: Default::default(),
                cache: None,
            });

        let pressure_black_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Black 3D Pipeline"),
                layout: Some(&pressure_pipeline_layout),
                module: &pressure_shader,
                entry_point: Some("pressure_black"),
                compilation_options: Default::default(),
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
                compilation_options: Default::default(),
                cache: None,
            });

        let gradient_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gradient V 3D Pipeline"),
                layout: Some(&gradient_pipeline_layout),
                module: &gradient_shader,
                entry_point: Some("apply_gradient_v"),
                compilation_options: Default::default(),
                cache: None,
            });

        let gradient_w_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gradient W 3D Pipeline"),
                layout: Some(&gradient_pipeline_layout),
                module: &gradient_shader,
                entry_point: Some("apply_gradient_w"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
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
            gradient_bind_group,
        }
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
        // NOTE: For 3D grids, optimal SOR omega is approximately:
        //   omega = 2 / (1 + sin(π/N)) where N is largest grid dimension
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
            _pad0: 0,
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
        let workgroups_x = self.width.div_ceil(8);
        let workgroups_y = self.height.div_ceil(8);
        let workgroups_z = self.depth.div_ceil(4);

        // Compute divergence
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Divergence 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divergence_pipeline);
            pass.set_bind_group(0, &self.divergence_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Red-Black Gauss-Seidel iterations
        for _ in 0..iterations {
            // Red pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Pressure Red 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pressure_red_pipeline);
                pass.set_bind_group(0, &self.pressure_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Black pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Pressure Black 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pressure_black_pipeline);
                pass.set_bind_group(0, &self.pressure_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        }

        // Apply pressure gradient to U
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient U 3D Pass"),
                timestamp_writes: None,
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
                timestamp_writes: None,
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
                timestamp_writes: None,
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
        self.encode(encoder, iterations);
        let byte_size = (self.width * self.height * self.depth * std::mem::size_of::<f32>() as u32) as u64;
        encoder.copy_buffer_to_buffer(
            &self.pressure_buffer,
            0,
            &self.pressure_backup_buffer,
            0,
            byte_size,
        );
        self.encode(encoder, iterations);
        encoder.copy_buffer_to_buffer(
            &self.pressure_backup_buffer,
            0,
            &self.pressure_buffer,
            0,
            byte_size,
        );
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
        let workgroups_x = self.width.div_ceil(8);
        let workgroups_y = self.height.div_ceil(8);
        let workgroups_z = self.depth.div_ceil(4);

        // Red-Black Gauss-Seidel iterations
        for _ in 0..iterations {
            // Red pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Pressure Red 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pressure_red_pipeline);
                pass.set_bind_group(0, &self.pressure_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Black pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Pressure Black 3D Pass"),
                    timestamp_writes: None,
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
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", shader_path, e.emit_to_string(&source)));

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
    }
}
