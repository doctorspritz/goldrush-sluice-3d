//! GPU Particle Separation (3D) - Houdini-style relaxation
//!
//! Pushes particles apart to maintain incompressibility.
//! Uses spatial hash for O(n) neighbor lookup.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    min_dist: f32,
    push_strength: f32,
    _pad: f32,
}

pub struct GpuParticleSeparation {
    params_buffer: wgpu::Buffer,
    cell_counts_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    sorted_indices_buffer: wgpu::Buffer,
    particle_cells_buffer: wgpu::Buffer,

    reset_pipeline: wgpu::ComputePipeline,
    count_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    separate_pipeline: wgpu::ComputePipeline,

    bind_group_layout: wgpu::BindGroupLayout,

    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
    max_particles: usize,
}

impl GpuParticleSeparation {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        cell_size: f32,
        max_particles: usize,
    ) -> Self {
        let num_cells = (width * height * depth) as usize;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Separation 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle_separation_3d.wgsl").into()),
        });

        // Buffers
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Separation Params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Counts"),
            size: (num_cells * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // +1 for sentinel at end (for prefix sum lookups)
        let cell_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Offsets"),
            size: ((num_cells + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorted_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Indices"),
            size: (max_particles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let particle_cells_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Cells"),
            size: (max_particles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Separation Bind Group Layout"),
            entries: &[
                // params
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
                // positions (read_write)
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
                // cell_counts
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
                // cell_offsets
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
                // sorted_indices
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
                // particle_cells
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Separation Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reset Counts Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reset_counts"),
            compilation_options: Default::default(),
            cache: None,
        });

        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Count Particles Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("count_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scatter Particles Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let separate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Separate Particles Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("separate_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            params_buffer,
            cell_counts_buffer,
            cell_offsets_buffer,
            sorted_indices_buffer,
            particle_cells_buffer,
            reset_pipeline,
            count_pipeline,
            scatter_pipeline,
            separate_pipeline,
            bind_group_layout,
            width,
            height,
            depth,
            cell_size,
            max_particles,
        }
    }

    /// Run particle separation on GPU
    ///
    /// positions_buffer: GPU buffer containing particle positions (modified in place)
    /// particle_count: Number of active particles
    /// min_dist: Minimum distance between particles
    /// push_strength: Relaxation strength (0.3 typical)
    /// iterations: Number of relaxation iterations
    pub fn separate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions_buffer: &wgpu::Buffer,
        particle_count: u32,
        min_dist: f32,
        push_strength: f32,
        iterations: u32,
    ) {
        if particle_count == 0 {
            return;
        }

        let num_cells = self.width * self.height * self.depth;

        // Update params
        let params = Params {
            cell_size: self.cell_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            min_dist,
            push_strength,
            _pad: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Separation Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cell_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.sorted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.particle_cells_buffer.as_entire_binding(),
                },
            ],
        });

        for _ in 0..iterations {
            // Pass 1: Reset counts
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.reset_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((num_cells + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // Pass 2: Count particles per cell
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.count_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((particle_count + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // Pass 3: Compute prefix sum on CPU (read back counts, compute offsets, upload)
            // This is a sync point but keeps the code simple
            self.compute_prefix_sum_cpu(device, queue, num_cells);

            // Reset counts again (they'll be used as write cursors in scatter)
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.reset_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((num_cells + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // Pass 4: Scatter particles to sorted array
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((particle_count + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // Pass 5: Compute separation forces and apply
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.separate_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((particle_count + 255) / 256, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }
    }

    fn compute_prefix_sum_cpu(&self, device: &wgpu::Device, queue: &wgpu::Queue, num_cells: u32) {
        // Create staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counts Staging"),
            size: (num_cells * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy counts to staging
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.cell_counts_buffer, 0, &staging, 0, (num_cells * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let counts: Vec<u32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        staging.unmap();

        // Compute prefix sum
        let mut offsets = vec![0u32; num_cells as usize + 1];
        let mut sum = 0u32;
        for i in 0..num_cells as usize {
            offsets[i] = sum;
            sum += counts[i];
        }
        offsets[num_cells as usize] = sum;  // Sentinel

        // Upload offsets
        queue.write_buffer(&self.cell_offsets_buffer, 0, bytemuck::cast_slice(&offsets));
    }
}
