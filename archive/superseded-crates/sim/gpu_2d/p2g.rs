//! GPU-accelerated Particle-to-Grid (P2G) transfer using fixed-point atomic scatter.
//!
//! This module implements P2G for APIC-FLIP using compute shaders with atomic operations.
//! WebGPU/wgpu only supports atomicAdd for i32, so we encode floats as fixed-point:
//!   f32 * SCALE → i32 (scatter), then i32 / SCALE → f32 (divide)
//!
//! Two compute passes:
//! 1. Scatter: Each particle atomically adds its momentum contribution to grid nodes
//! 2. Divide: Each grid node divides accumulated momentum by weight to get velocity

use bytemuck::{Pod, Zeroable};

use super::GpuContext;

/// Fixed-point scale factor (must match shader)
const SCALE: f32 = 1_000_000.0;

/// Parameters for P2G compute shaders
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct P2gParams {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

/// GPU-based Particle-to-Grid transfer using atomic scatter
pub struct GpuP2gSolver {
    width: u32,
    height: u32,

    // Particle buffers (uploaded each frame)
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,
    c_matrices_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,

    // Grid accumulator buffers (atomic<i32>)
    u_sum_buffer: wgpu::Buffer,
    u_weight_buffer: wgpu::Buffer,
    v_sum_buffer: wgpu::Buffer,
    v_weight_buffer: wgpu::Buffer,

    // Output grid buffers (f32)
    grid_u_buffer: wgpu::Buffer,
    grid_v_buffer: wgpu::Buffer,

    // Staging buffers for readback
    grid_u_staging: wgpu::Buffer,
    grid_v_staging: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,

    // Compute pipelines
    scatter_pipeline: wgpu::ComputePipeline,
    divide_u_pipeline: wgpu::ComputePipeline,
    divide_v_pipeline: wgpu::ComputePipeline,

    // Bind groups
    scatter_bind_group_layout: wgpu::BindGroupLayout,
    divide_bind_group_layout: wgpu::BindGroupLayout,
    scatter_bind_group: wgpu::BindGroup,
    divide_bind_group: wgpu::BindGroup,

    // Current capacity
    max_particles: usize,

    // Workgroup sizes
    scatter_workgroup_size: u32,
    divide_workgroup_x: u32,
    divide_workgroup_y: u32,
}

impl GpuP2gSolver {
    /// Create a new GPU P2G solver from device and queue
    ///
    /// Arguments:
    /// - device: wgpu device
    /// - queue: wgpu queue
    /// - width: grid width
    /// - height: grid height
    /// - max_particles: maximum particle capacity
    pub fn new_headless(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        max_particles: usize,
    ) -> Self {
        Self::new_internal(device, width, height, max_particles)
    }

    /// Create a new GPU P2G solver
    ///
    /// Arguments:
    /// - gpu: GPU context
    /// - width: grid width
    /// - height: grid height
    /// - max_particles: maximum particle capacity
    pub fn new(gpu: &GpuContext, width: u32, height: u32, max_particles: usize) -> Self {
        Self::new_internal(&gpu.device, width, height, max_particles)
    }

    fn new_internal(device: &wgpu::Device, width: u32, height: u32, max_particles: usize) -> Self {
        let u_size = ((width + 1) * height) as usize;
        let v_size = (width * (height + 1)) as usize;

        // Create shader modules
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_scatter.wgsl").into()),
        });

        let divide_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Divide Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/p2g_divide.wgsl").into()),
        });

        // Create particle buffers
        let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Positions"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Velocities"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_matrices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G C-Matrices"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Materials"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create grid accumulator buffers (atomic i32)
        let u_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G U Sum"),
            size: (u_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G U Weight"),
            size: (u_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let v_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G V Sum"),
            size: (v_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let v_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G V Weight"),
            size: (v_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output grid buffers (f32)
        let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Grid U"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Grid V"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffers for readback
        let grid_u_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Grid U Staging"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Grid V Staging"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Params"),
            size: std::mem::size_of::<P2gParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create scatter bind group layout
        let scatter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("P2G Scatter Bind Group Layout"),
                    entries: &[
                        // params (uniform)
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
                        // positions (read)
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
                        // velocities (read)
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
                        // c_matrices (read)
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
                        // materials (read)
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
                        // u_sum (read_write atomic)
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
                        // u_weight (read_write atomic)
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
                        // v_sum (read_write atomic)
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
                        // v_weight (read_write atomic)
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

        // Create divide bind group layout
        let divide_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("P2G Divide Bind Group Layout"),
                    entries: &[
                        // params (uniform)
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
                        // u_sum (read)
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
                        // u_weight (read)
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
                        // v_sum (read)
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
                        // v_weight (read)
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
                        // grid_u (read_write)
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
                        // grid_v (read_write)
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

        // Create scatter bind group
        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G Scatter Bind Group"),
            layout: &scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
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
                    resource: c_matrices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: u_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: u_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: v_sum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: v_weight_buffer.as_entire_binding(),
                },
            ],
        });

        // Create divide bind group
        let divide_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G Divide Bind Group"),
            layout: &divide_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
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
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_v_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipelines
        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("P2G Scatter Pipeline Layout"),
                    bind_group_layouts: &[&scatter_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("P2G Scatter Pipeline"),
                    layout: Some(&scatter_pipeline_layout),
                    module: &scatter_shader,
                    entry_point: Some("scatter"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let divide_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("P2G Divide Pipeline Layout"),
                    bind_group_layouts: &[&divide_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let divide_u_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("P2G Divide U Pipeline"),
                    layout: Some(&divide_pipeline_layout),
                    module: &divide_shader,
                    entry_point: Some("divide_u"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let divide_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("P2G Divide V Pipeline"),
                    layout: Some(&divide_pipeline_layout),
                    module: &divide_shader,
                    entry_point: Some("divide_v"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Calculate workgroup counts (256 threads per workgroup for scatter)
        let scatter_workgroup_size = 256;
        let divide_workgroup_x = (width + 7) / 8;
        let divide_workgroup_y = (height + 7) / 8;

        Self {
            width,
            height,
            positions_buffer,
            velocities_buffer,
            c_matrices_buffer,
            materials_buffer,
            u_sum_buffer,
            u_weight_buffer,
            v_sum_buffer,
            v_weight_buffer,
            grid_u_buffer,
            grid_v_buffer,
            grid_u_staging,
            grid_v_staging,
            params_buffer,
            scatter_pipeline,
            divide_u_pipeline,
            divide_v_pipeline,
            scatter_bind_group_layout,
            divide_bind_group_layout,
            scatter_bind_group,
            divide_bind_group,
            max_particles,
            scatter_workgroup_size,
            divide_workgroup_x,
            divide_workgroup_y,
        }
    }

    /// Upload particle data to GPU
    ///
    /// Extracts particle data into SoA format for efficient GPU access.
    /// Only uploads water particles (sediment doesn't contribute to P2G).
    pub fn upload_particles(
        &self,
        gpu: &GpuContext,
        particles: &[sim::Particle],
        cell_size: f32,
    ) -> u32 {
        // Filter to water particles and extract SoA data
        let mut positions = Vec::with_capacity(particles.len());
        let mut velocities = Vec::with_capacity(particles.len());
        let mut c_matrices = Vec::with_capacity(particles.len());
        let mut materials = Vec::with_capacity(particles.len());

        for p in particles {
            // Only water contributes to grid momentum
            if p.material.is_sediment() {
                continue;
            }

            positions.push([p.position.x, p.position.y]);
            velocities.push([p.velocity.x, p.velocity.y]);
            // Mat2 is column-major: [col0.x, col0.y, col1.x, col1.y]
            c_matrices.push([
                p.affine_velocity.x_axis.x,
                p.affine_velocity.x_axis.y,
                p.affine_velocity.y_axis.x,
                p.affine_velocity.y_axis.y,
            ]);
            materials.push(0u32); // 0 = water (not sediment)
        }

        let particle_count = positions.len() as u32;

        // Upload to GPU
        gpu.queue.write_buffer(
            &self.positions_buffer,
            0,
            bytemuck::cast_slice(&positions),
        );
        gpu.queue.write_buffer(
            &self.velocities_buffer,
            0,
            bytemuck::cast_slice(&velocities),
        );
        gpu.queue.write_buffer(
            &self.c_matrices_buffer,
            0,
            bytemuck::cast_slice(&c_matrices),
        );
        gpu.queue
            .write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(&materials));

        // Upload params
        let params = P2gParams {
            cell_size,
            width: self.width,
            height: self.height,
            particle_count,
        };
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Clear accumulator buffers
        let u_size = ((self.width + 1) * self.height) as usize;
        let v_size = (self.width * (self.height + 1)) as usize;

        gpu.queue.write_buffer(
            &self.u_sum_buffer,
            0,
            &vec![0u8; u_size * std::mem::size_of::<i32>()],
        );
        gpu.queue.write_buffer(
            &self.u_weight_buffer,
            0,
            &vec![0u8; u_size * std::mem::size_of::<i32>()],
        );
        gpu.queue.write_buffer(
            &self.v_sum_buffer,
            0,
            &vec![0u8; v_size * std::mem::size_of::<i32>()],
        );
        gpu.queue.write_buffer(
            &self.v_weight_buffer,
            0,
            &vec![0u8; v_size * std::mem::size_of::<i32>()],
        );

        particle_count
    }

    /// Run P2G compute shaders
    ///
    /// Executes scatter pass (particle → grid accumulation) then divide pass (weight normalization).
    pub fn compute(&self, gpu: &GpuContext, particle_count: u32) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("P2G Compute Encoder"),
            });

        // Scatter pass: each particle contributes to 3x3 neighborhood
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Scatter Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.scatter_bind_group, &[]);
            let workgroups = (particle_count + self.scatter_workgroup_size - 1) / self.scatter_workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Divide U pass: normalize U velocities
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Divide U Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_u_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            // U grid is (width+1) x height
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Divide V pass: normalize V velocities
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Divide V Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_v_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            // V grid is width x (height+1)
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Download grid velocities from GPU
    pub fn download(&self, gpu: &GpuContext, grid_u: &mut [f32], grid_v: &mut [f32]) {
        let u_size = ((self.width + 1) * self.height) as usize;
        let v_size = (self.width * (self.height + 1)) as usize;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("P2G Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.grid_u_buffer,
            0,
            &self.grid_u_staging,
            0,
            (u_size * std::mem::size_of::<f32>()) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.grid_v_buffer,
            0,
            &self.grid_v_staging,
            0,
            (v_size * std::mem::size_of::<f32>()) as u64,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map and read U
        {
            let buffer_slice = self.grid_u_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            {
                let data = buffer_slice.get_mapped_range();
                grid_u.copy_from_slice(bytemuck::cast_slice(&data));
            }
            self.grid_u_staging.unmap();
        }

        // Map and read V
        {
            let buffer_slice = self.grid_v_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            {
                let data = buffer_slice.get_mapped_range();
                grid_v.copy_from_slice(bytemuck::cast_slice(&data));
            }
            self.grid_v_staging.unmap();
        }
    }

    /// Execute full P2G pipeline: upload → compute → download
    ///
    /// Convenience method for complete transfer. For benchmarking, use
    /// individual methods to measure each phase.
    pub fn execute(
        &self,
        gpu: &GpuContext,
        particles: &[sim::Particle],
        cell_size: f32,
        grid_u: &mut [f32],
        grid_v: &mut [f32],
    ) {
        let particle_count = self.upload_particles(gpu, particles, cell_size);
        self.compute(gpu, particle_count);
        self.download(gpu, grid_u, grid_v);
    }

    // ============ Headless variants (for benchmarking without window) ============

    /// Upload particle data to GPU (headless version)
    pub fn upload_particles_headless(
        &self,
        queue: &wgpu::Queue,
        particles: &[sim::Particle],
        cell_size: f32,
    ) -> u32 {
        // Filter to water particles and extract SoA data
        let mut positions = Vec::with_capacity(particles.len());
        let mut velocities = Vec::with_capacity(particles.len());
        let mut c_matrices = Vec::with_capacity(particles.len());
        let mut materials = Vec::with_capacity(particles.len());

        for p in particles {
            if p.material.is_sediment() {
                continue;
            }

            positions.push([p.position.x, p.position.y]);
            velocities.push([p.velocity.x, p.velocity.y]);
            c_matrices.push([
                p.affine_velocity.x_axis.x,
                p.affine_velocity.x_axis.y,
                p.affine_velocity.y_axis.x,
                p.affine_velocity.y_axis.y,
            ]);
            materials.push(0u32);
        }

        let particle_count = positions.len() as u32;

        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions));
        queue.write_buffer(&self.velocities_buffer, 0, bytemuck::cast_slice(&velocities));
        queue.write_buffer(&self.c_matrices_buffer, 0, bytemuck::cast_slice(&c_matrices));
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(&materials));

        let params = P2gParams {
            cell_size,
            width: self.width,
            height: self.height,
            particle_count,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Clear accumulator buffers
        let u_size = ((self.width + 1) * self.height) as usize;
        let v_size = (self.width * (self.height + 1)) as usize;

        queue.write_buffer(&self.u_sum_buffer, 0, &vec![0u8; u_size * std::mem::size_of::<i32>()]);
        queue.write_buffer(&self.u_weight_buffer, 0, &vec![0u8; u_size * std::mem::size_of::<i32>()]);
        queue.write_buffer(&self.v_sum_buffer, 0, &vec![0u8; v_size * std::mem::size_of::<i32>()]);
        queue.write_buffer(&self.v_weight_buffer, 0, &vec![0u8; v_size * std::mem::size_of::<i32>()]);

        particle_count
    }

    /// Run P2G compute shaders (headless version)
    pub fn compute_headless(&self, device: &wgpu::Device, queue: &wgpu::Queue, particle_count: u32) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("P2G Compute Encoder"),
        });

        // Scatter pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Scatter Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.scatter_bind_group, &[]);
            let workgroups = (particle_count + self.scatter_workgroup_size - 1) / self.scatter_workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Divide U pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Divide U Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_u_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Divide V pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Divide V Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divide_v_pipeline);
            pass.set_bind_group(0, &self.divide_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Download grid velocities from GPU (headless version)
    pub fn download_headless(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid_u: &mut [f32],
        grid_v: &mut [f32],
    ) {
        let u_size = ((self.width + 1) * self.height) as usize;
        let v_size = (self.width * (self.height + 1)) as usize;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("P2G Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.grid_u_buffer,
            0,
            &self.grid_u_staging,
            0,
            (u_size * std::mem::size_of::<f32>()) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.grid_v_buffer,
            0,
            &self.grid_v_staging,
            0,
            (v_size * std::mem::size_of::<f32>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read U
        {
            let buffer_slice = self.grid_u_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            {
                let data = buffer_slice.get_mapped_range();
                grid_u.copy_from_slice(bytemuck::cast_slice(&data));
            }
            self.grid_u_staging.unmap();
        }

        // Map and read V
        {
            let buffer_slice = self.grid_v_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            {
                let data = buffer_slice.get_mapped_range();
                grid_v.copy_from_slice(bytemuck::cast_slice(&data));
            }
            self.grid_v_staging.unmap();
        }
    }

    /// Execute full P2G pipeline (headless version)
    pub fn execute_headless(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[sim::Particle],
        cell_size: f32,
        grid_u: &mut [f32],
        grid_v: &mut [f32],
    ) {
        let particle_count = self.upload_particles_headless(queue, particles, cell_size);
        self.compute_headless(device, queue, particle_count);
        self.download_headless(device, queue, grid_u, grid_v);
    }
}
