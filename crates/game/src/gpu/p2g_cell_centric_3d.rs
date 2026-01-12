//! GPU-accelerated Cell-Centric Particle-to-Grid (P2G) transfer for 3D FLIP simulation.
//!
//! Unlike particle-centric P2G (one thread per particle, 81 atomics each),
//! this uses cell-centric dispatch (one thread per grid node, ZERO atomics).
//!
//! Requires:
//! - Sorted particles from GpuParticleSort
//! - cell_offsets buffer (exclusive prefix sum of cell counts)
//!
//! Three compute passes (one per velocity component):
//! 1. scatter_u: Each U node iterates nearby particles, accumulates, writes velocity
//! 2. scatter_v: Same for V nodes
//! 3. scatter_w: Same for W nodes
//! 4. count_particles: Count water/sediment per cell (for density projection)

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for cell-centric P2G shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct P2gCellCentricParams {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    include_sediment: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-based Cell-Centric Particle-to-Grid transfer
pub struct GpuP2gCellCentric3D {
    width: u32,
    height: u32,
    depth: u32,
    include_sediment: bool,

    // Output velocity grids (f32, written directly - no atomics)
    pub grid_u_buffer: wgpu::Buffer,
    pub grid_v_buffer: wgpu::Buffer,
    pub grid_w_buffer: wgpu::Buffer,

    // Particle counts per cell (for density projection)
    pub particle_count_buffer: wgpu::Buffer,
    pub sediment_count_buffer: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,

    // Compute pipelines
    scatter_u_pipeline: wgpu::ComputePipeline,
    scatter_v_pipeline: wgpu::ComputePipeline,
    scatter_w_pipeline: wgpu::ComputePipeline,
    count_particles_pipeline: wgpu::ComputePipeline,

    // Bind group
    bind_group: wgpu::BindGroup,
}

impl GpuP2gCellCentric3D {
    /// Create a new GPU Cell-Centric P2G solver
    ///
    /// Takes sorted particle buffers and cell_offsets from GpuParticleSort.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        include_sediment: bool,
        // Sorted particle data from GpuParticleSort
        positions_buffer: Arc<wgpu::Buffer>,
        velocities_buffer: Arc<wgpu::Buffer>,
        c_col0_buffer: Arc<wgpu::Buffer>,
        c_col1_buffer: Arc<wgpu::Buffer>,
        c_col2_buffer: Arc<wgpu::Buffer>,
        densities_buffer: Arc<wgpu::Buffer>,
        // Cell offsets from counting sort (size: cell_count + 1)
        cell_offsets_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;
        let cell_count = (width * height * depth) as usize;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("P2G Cell-Centric 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/p2g_cell_centric_3d.wgsl").into(),
            ),
        });

        // Create output velocity buffers (f32, direct writes)
        let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Cell-Centric Grid U"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Cell-Centric Grid V"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_w_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Cell-Centric Grid W"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Particle count buffers (i32, direct writes)
        let particle_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Cell-Centric Particle Count"),
            size: (cell_count * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Cell-Centric Sediment Count"),
            size: (cell_count * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Parameters buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("P2G Cell-Centric Params"),
            size: std::mem::size_of::<P2gCellCentricParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("P2G Cell-Centric Bind Group Layout"),
            entries: &[
                // 0: params (uniform)
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
                // 1: positions (read)
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
                // 2: velocities (read)
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
                // 3: c_col0 (read)
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
                // 4: c_col1 (read)
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
                // 5: c_col2 (read)
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
                // 6: densities (read)
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
                // 7: cell_offsets (read)
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
                // 8: grid_u (write)
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
                // 9: grid_v (write)
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
                // 10: grid_w (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 11: particle_count (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 12: sediment_count (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("P2G Cell-Centric Bind Group"),
            layout: &bind_group_layout,
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
                    resource: c_col0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: c_col1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: c_col2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cell_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: grid_w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: particle_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: sediment_count_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("P2G Cell-Centric Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipelines for each entry point
        let scatter_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G Cell-Centric Scatter U Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G Cell-Centric Scatter V Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("P2G Cell-Centric Scatter W Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        let count_particles_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("P2G Cell-Centric Count Particles Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("count_particles"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            width,
            height,
            depth,
            include_sediment,
            grid_u_buffer,
            grid_v_buffer,
            grid_w_buffer,
            particle_count_buffer,
            sediment_count_buffer,
            params_buffer,
            scatter_u_pipeline,
            scatter_v_pipeline,
            scatter_w_pipeline,
            count_particles_pipeline,
            bind_group,
        }
    }

    /// Prepare for encoding (upload parameters)
    pub fn prepare(&self, queue: &wgpu::Queue, particle_count: u32, cell_size: f32) {
        let params = P2gCellCentricParams {
            cell_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            include_sediment: if self.include_sediment { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Encode P2G passes into command encoder
    ///
    /// Dispatches 4 compute passes:
    /// - scatter_u: (width+1) x height x depth threads
    /// - scatter_v: width x (height+1) x depth threads
    /// - scatter_w: width x height x (depth+1) threads
    /// - count_particles: width x height x depth threads
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, _particle_count: u32) {
        // Workgroup size is 8x8x4 = 256 threads
        let wg_x = 8u32;
        let wg_y = 8u32;
        let wg_z = 4u32;

        // U grid: (width+1) x height x depth
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Cell-Centric Scatter U"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_u_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                (self.width + 1 + wg_x - 1) / wg_x,
                (self.height + wg_y - 1) / wg_y,
                (self.depth + wg_z - 1) / wg_z,
            );
        }

        // V grid: width x (height+1) x depth
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Cell-Centric Scatter V"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_v_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                (self.width + wg_x - 1) / wg_x,
                (self.height + 1 + wg_y - 1) / wg_y,
                (self.depth + wg_z - 1) / wg_z,
            );
        }

        // W grid: width x height x (depth+1)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Cell-Centric Scatter W"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_w_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                (self.width + wg_x - 1) / wg_x,
                (self.height + wg_y - 1) / wg_y,
                (self.depth + 1 + wg_z - 1) / wg_z,
            );
        }

        // Count particles: width x height x depth
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P2G Cell-Centric Count Particles"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_particles_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                (self.width + wg_x - 1) / wg_x,
                (self.height + wg_y - 1) / wg_y,
                (self.depth + wg_z - 1) / wg_z,
            );
        }
    }

    /// Get grid sizes for U, V, W buffers
    pub fn grid_sizes(&self) -> (usize, usize, usize) {
        let u_size = ((self.width + 1) * self.height * self.depth) as usize;
        let v_size = (self.width * (self.height + 1) * self.depth) as usize;
        let w_size = (self.width * self.height * (self.depth + 1)) as usize;
        (u_size, v_size, w_size)
    }
}
