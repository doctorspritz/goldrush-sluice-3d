use bytemuck::{Pod, Zeroable};

use super::GpuContext;

/// Parameters for pressure solve
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PressureParams {
    width: u32,
    height: u32,
    omega: f32,  // SOR relaxation factor
    _padding: u32,
}

/// GPU-based pressure solver using Checkerboard SOR
pub struct GpuPressureSolver {
    width: u32,
    height: u32,

    // GPU buffers
    pressure_buffer: wgpu::Buffer,
    divergence_buffer: wgpu::Buffer,
    cell_type_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

    // Staging buffers for CPU<->GPU transfer
    pressure_staging: wgpu::Buffer,
    divergence_staging: wgpu::Buffer,
    cell_type_staging: wgpu::Buffer,

    // Compute pipelines
    red_pipeline: wgpu::ComputePipeline,
    black_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // Workgroup counts
    workgroup_x: u32,
    workgroup_y: u32,
}

impl GpuPressureSolver {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        let cell_count = (width * height) as usize;

        // Create shader module
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Pressure Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pressure.wgsl").into()),
            });

        // Create buffers
        let pressure_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Buffer"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let divergence_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Divergence Buffer"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_type_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type Buffer"),
            size: (cell_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<PressureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffers for readback
        let pressure_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Staging"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let divergence_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Divergence Staging"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_type_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type Staging"),
            size: (cell_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pressure Bind Group Layout"),
                entries: &[
                    // pressure (read_write)
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
                    // divergence (read)
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
                    // cell_type (read)
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
                    // params (uniform)
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

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pressure Bind Group"),
            layout: &bind_group_layout,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pressure Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create pipelines for red and black passes
        let red_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Red Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("pressure_red"),
                compilation_options: Default::default(),
                cache: None,
            });

        let black_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pressure Black Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("pressure_black"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Calculate workgroup counts (8x8 workgroups)
        // Each thread handles one cell, but we process half the cells per pass
        let workgroup_x = (width / 2 + 7) / 8;
        let workgroup_y = (height + 7) / 8;

        Self {
            width,
            height,
            pressure_buffer,
            divergence_buffer,
            cell_type_buffer,
            params_buffer,
            pressure_staging,
            divergence_staging,
            cell_type_staging,
            red_pipeline,
            black_pipeline,
            bind_group,
            workgroup_x,
            workgroup_y,
        }
    }

    /// Upload divergence and cell type data to GPU
    pub fn upload(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32], omega: f32) {
        // Upload divergence
        gpu.queue.write_buffer(&self.divergence_buffer, 0, bytemuck::cast_slice(divergence));

        // Upload cell type
        gpu.queue.write_buffer(&self.cell_type_buffer, 0, bytemuck::cast_slice(cell_type));

        // Upload params
        let params = PressureParams {
            width: self.width,
            height: self.height,
            omega,
            _padding: 0,
        };
        gpu.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Clear pressure to 0 using a pre-allocated buffer would be faster,
        // but for now we do a simple clear
        gpu.queue.write_buffer(&self.pressure_buffer, 0, &vec![0u8; (self.width * self.height * 4) as usize]);
    }

    /// Upload with warm start from previous pressure
    pub fn upload_warm(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32], pressure: &[f32], omega: f32) {
        gpu.queue.write_buffer(&self.divergence_buffer, 0, bytemuck::cast_slice(divergence));
        gpu.queue.write_buffer(&self.cell_type_buffer, 0, bytemuck::cast_slice(cell_type));
        gpu.queue.write_buffer(&self.pressure_buffer, 0, bytemuck::cast_slice(pressure));

        let params = PressureParams {
            width: self.width,
            height: self.height,
            omega,
            _padding: 0,
        };
        gpu.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Run pressure solve iterations on GPU
    pub fn solve(&self, gpu: &GpuContext, iterations: u32) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pressure Solve Encoder"),
        });

        for _ in 0..iterations {
            // Red pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Red Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.red_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(self.workgroup_x, self.workgroup_y, 1);
            }

            // Black pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Black Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.black_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(self.workgroup_x, self.workgroup_y, 1);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Download pressure results from GPU
    pub fn download(&self, gpu: &GpuContext, pressure: &mut [f32]) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pressure Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.pressure_buffer,
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
}
