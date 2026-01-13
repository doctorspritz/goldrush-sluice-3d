//! GPU-accelerated particle sorting by cell index for spatial coherence.
//!
//! Implements counting sort on GPU:
//! 1. Compute cell key for each particle
//! 2. Count particles per cell (histogram)
//! 3. Exclusive prefix sum to get cell offsets
//! 4. Scatter particles to sorted order
//!
//! After sorting, particles with the same cell index are contiguous in memory,
//! improving cache coherence and reducing atomic contention in P2G scatter.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for sorting shaders
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SortParams {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Parameters for prefix sum
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PrefixSumParams {
    element_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU particle sorter for spatial coherence optimization
pub struct GpuParticleSort {
    width: u32,
    height: u32,
    depth: u32,
    max_particles: usize,
    cell_count: usize,

    // Input buffers (shared with FLIP)
    in_positions_buffer: Arc<wgpu::Buffer>,
    in_velocities_buffer: Arc<wgpu::Buffer>,
    in_densities_buffer: Arc<wgpu::Buffer>,
    in_c_col0_buffer: Arc<wgpu::Buffer>,
    in_c_col1_buffer: Arc<wgpu::Buffer>,
    in_c_col2_buffer: Arc<wgpu::Buffer>,

    // Output buffers (sorted particles)
    pub out_positions_buffer: Arc<wgpu::Buffer>,
    pub out_velocities_buffer: Arc<wgpu::Buffer>,
    pub out_densities_buffer: Arc<wgpu::Buffer>,
    pub out_c_col0_buffer: Arc<wgpu::Buffer>,
    pub out_c_col1_buffer: Arc<wgpu::Buffer>,
    pub out_c_col2_buffer: Arc<wgpu::Buffer>,

    // Intermediate buffers
    cell_keys_buffer: wgpu::Buffer,
    cell_counts_buffer: wgpu::Buffer,
    /// Cell offsets (exclusive prefix sum of cell counts) - public for cell-centric P2G
    /// Size: cell_count + 1 (extra element for end-of-last-cell lookup)
    pub cell_offsets_buffer: Arc<wgpu::Buffer>,
    cell_counters_buffer: wgpu::Buffer,
    block_sums_buffer: wgpu::Buffer,

    // Parameters
    sort_params_buffer: wgpu::Buffer,
    prefix_sum_params_buffer: wgpu::Buffer,

    // Pipelines
    compute_keys_pipeline: wgpu::ComputePipeline,
    count_cells_pipeline: wgpu::ComputePipeline,
    local_prefix_sum_pipeline: wgpu::ComputePipeline,
    scan_block_sums_pipeline: wgpu::ComputePipeline,
    add_block_offsets_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    // Bind groups
    compute_keys_bind_group: wgpu::BindGroup,
    count_cells_bind_group: wgpu::BindGroup,
    prefix_sum_bind_group: wgpu::BindGroup,
    scatter_bind_group: wgpu::BindGroup,

    // Whether sorting is enabled
    enabled: bool,
}

impl GpuParticleSort {
    /// Create a new GPU particle sorter
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        max_particles: usize,
        in_positions_buffer: Arc<wgpu::Buffer>,
        in_velocities_buffer: Arc<wgpu::Buffer>,
        in_densities_buffer: Arc<wgpu::Buffer>,
        in_c_col0_buffer: Arc<wgpu::Buffer>,
        in_c_col1_buffer: Arc<wgpu::Buffer>,
        in_c_col2_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        let cell_count = (width * height * depth) as usize;
        let num_blocks = (cell_count + 511) / 512;

        // Create output buffers (sorted particles) - same size as input
        let out_positions_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Positions"),
            size: (max_particles * 16) as u64, // vec3 padded to vec4
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let out_velocities_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Velocities"),
            size: (max_particles * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let out_densities_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Densities"),
            size: (max_particles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let out_c_col0_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted C Col0"),
            size: (max_particles * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let out_c_col1_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted C Col1"),
            size: (max_particles * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let out_c_col2_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted C Col2"),
            size: (max_particles * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Create intermediate buffers
        let cell_keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Keys"),
            size: (max_particles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Counts"),
            size: (cell_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // cell_count + 1 elements so we can look up cell_offsets[cell + 1] for the last cell
        let cell_offsets_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Offsets"),
            size: ((cell_count + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let cell_counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Counters"),
            size: (cell_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block Sums"),
            size: (num_blocks * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create parameter buffers
        let sort_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sort Params"),
            size: std::mem::size_of::<SortParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prefix_sum_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefix Sum Params"),
            size: std::mem::size_of::<PrefixSumParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shaders
        let keys_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Sort Keys Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_sort_keys.wgsl").into(),
            ),
        });

        let count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Sort Count Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_sort_count.wgsl").into(),
            ),
        });

        let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Sort Prefix Sum Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_sort_prefix_sum.wgsl").into(),
            ),
        });

        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Sort Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_sort_scatter.wgsl").into(),
            ),
        });

        // Create bind group layouts and bind groups
        // Keys bind group
        let keys_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Keys Bind Group Layout"),
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

        let compute_keys_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Keys Bind Group"),
            layout: &keys_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_keys_buffer.as_entire_binding(),
                },
            ],
        });

        // Count bind group
        let count_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Count Bind Group Layout"),
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

        let count_cells_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Count Bind Group"),
            layout: &count_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Prefix sum bind group
        let prefix_sum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Prefix Sum Bind Group Layout"),
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

        let prefix_sum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prefix Sum Bind Group"),
            layout: &prefix_sum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_sum_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_sums_buffer.as_entire_binding(),
                },
            ],
        });

        // Scatter bind group (16 bindings)
        let scatter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scatter Bind Group Layout"),
                entries: &(0..16)
                    .map(|i| {
                        let _read_only = i < 4 || (i >= 4 && i < 10);
                        wgpu::BindGroupLayoutEntry {
                            binding: i,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: if i == 0 {
                                    wgpu::BufferBindingType::Uniform
                                } else {
                                    wgpu::BufferBindingType::Storage {
                                        read_only: i < 3 || (i >= 4 && i < 10),
                                    }
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }
                    })
                    .collect::<Vec<_>>(),
            });

        let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scatter Bind Group"),
            layout: &scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: in_positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: in_velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: in_densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: in_c_col0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: in_c_col1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: in_c_col2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: out_positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: out_velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: out_densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: out_c_col0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: out_c_col1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: out_c_col2_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipelines
        let keys_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Keys Pipeline Layout"),
            bind_group_layouts: &[&keys_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_keys_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Keys Pipeline"),
                layout: Some(&keys_pipeline_layout),
                module: &keys_shader,
                entry_point: Some("compute_keys"),
                compilation_options: Default::default(),
                cache: None,
            });

        let count_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Count Pipeline Layout"),
                bind_group_layouts: &[&count_bind_group_layout],
                push_constant_ranges: &[],
            });

        let count_cells_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Count Cells Pipeline"),
                layout: Some(&count_pipeline_layout),
                module: &count_shader,
                entry_point: Some("count_cells"),
                compilation_options: Default::default(),
                cache: None,
            });

        let prefix_sum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Prefix Sum Pipeline Layout"),
                bind_group_layouts: &[&prefix_sum_bind_group_layout],
                push_constant_ranges: &[],
            });

        let local_prefix_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Local Prefix Sum Pipeline"),
                layout: Some(&prefix_sum_pipeline_layout),
                module: &prefix_sum_shader,
                entry_point: Some("local_prefix_sum"),
                compilation_options: Default::default(),
                cache: None,
            });

        let scan_block_sums_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Scan Block Sums Pipeline"),
                layout: Some(&prefix_sum_pipeline_layout),
                module: &prefix_sum_shader,
                entry_point: Some("scan_block_sums"),
                compilation_options: Default::default(),
                cache: None,
            });

        let add_block_offsets_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Add Block Offsets Pipeline"),
                layout: Some(&prefix_sum_pipeline_layout),
                module: &prefix_sum_shader,
                entry_point: Some("add_block_offsets"),
                compilation_options: Default::default(),
                cache: None,
            });

        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Scatter Pipeline Layout"),
                bind_group_layouts: &[&scatter_bind_group_layout],
                push_constant_ranges: &[],
            });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scatter Pipeline"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_shader,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            height,
            depth,
            max_particles,
            cell_count,
            in_positions_buffer,
            in_velocities_buffer,
            in_densities_buffer,
            in_c_col0_buffer,
            in_c_col1_buffer,
            in_c_col2_buffer,
            out_positions_buffer,
            out_velocities_buffer,
            out_densities_buffer,
            out_c_col0_buffer,
            out_c_col1_buffer,
            out_c_col2_buffer,
            cell_keys_buffer,
            cell_counts_buffer,
            cell_offsets_buffer,
            cell_counters_buffer,
            block_sums_buffer,
            sort_params_buffer,
            prefix_sum_params_buffer,
            compute_keys_pipeline,
            count_cells_pipeline,
            local_prefix_sum_pipeline,
            scan_block_sums_pipeline,
            add_block_offsets_pipeline,
            scatter_pipeline,
            compute_keys_bind_group,
            count_cells_bind_group,
            prefix_sum_bind_group,
            scatter_bind_group,
            enabled: true,
        }
    }

    /// Enable or disable sorting
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if sorting is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Prepare sorting parameters
    pub fn prepare(&self, queue: &wgpu::Queue, particle_count: u32, cell_size: f32) {
        let sort_params = SortParams {
            cell_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(
            &self.sort_params_buffer,
            0,
            bytemuck::bytes_of(&sort_params),
        );

        let prefix_sum_params = PrefixSumParams {
            element_count: self.cell_count as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(
            &self.prefix_sum_params_buffer,
            0,
            bytemuck::bytes_of(&prefix_sum_params),
        );

        // Clear intermediate buffers
        queue.write_buffer(&self.cell_counts_buffer, 0, &vec![0u8; self.cell_count * 4]);
        queue.write_buffer(
            &self.cell_counters_buffer,
            0,
            &vec![0u8; self.cell_count * 4],
        );
        let num_blocks = (self.cell_count + 511) / 512;
        queue.write_buffer(&self.block_sums_buffer, 0, &vec![0u8; num_blocks * 4]);

        // Set cell_offsets[cell_count] = particle_count for cell-centric P2G
        // This allows looking up cell_offsets[idx + 1] for the last cell
        queue.write_buffer(
            &self.cell_offsets_buffer,
            (self.cell_count * 4) as u64,
            bytemuck::bytes_of(&particle_count),
        );
    }

    /// Encode sorting passes into command encoder
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        _queue: &wgpu::Queue,
        particle_count: u32,
    ) {
        if !self.enabled || particle_count == 0 {
            return;
        }

        let workgroups_particles = (particle_count + 255) / 256;
        let workgroups_cells = (self.cell_count as u32 + 511) / 512;
        let workgroups_cells_256 = (self.cell_count as u32 + 255) / 256;

        // Pass 1: Compute cell keys
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Keys Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_keys_pipeline);
            pass.set_bind_group(0, &self.compute_keys_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_particles, 1, 1);
        }

        // Pass 2: Count particles per cell
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Count Cells Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_cells_pipeline);
            pass.set_bind_group(0, &self.count_cells_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_particles, 1, 1);
        }

        // Copy counts to offsets before prefix sum
        encoder.copy_buffer_to_buffer(
            &self.cell_counts_buffer,
            0,
            &self.cell_offsets_buffer,
            0,
            (self.cell_count * 4) as u64,
        );

        // Pass 3: Local prefix sum
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Local Prefix Sum Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.local_prefix_sum_pipeline);
            pass.set_bind_group(0, &self.prefix_sum_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_cells, 1, 1);
        }

        // For multi-block prefix sum, we need to scan block sums and add them back
        if workgroups_cells > 1 {
            // Pass 4a: Scan block_sums to get exclusive prefix sum of block totals
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Scan Block Sums Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_block_sums_pipeline);
                pass.set_bind_group(0, &self.prefix_sum_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1); // Single workgroup, sequential scan
            }

            // Pass 4b: Add scanned block offsets to each element
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Add Block Offsets Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.add_block_offsets_pipeline);
                pass.set_bind_group(0, &self.prefix_sum_bind_group, &[]);
                pass.dispatch_workgroups(workgroups_cells_256, 1, 1);
            }
        }

        // Pass 5: Scatter particles to sorted order
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scatter Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &self.scatter_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_particles, 1, 1);
        }
    }

    /// Get sorted output buffers
    pub fn sorted_buffers(
        &self,
    ) -> (
        &Arc<wgpu::Buffer>, // positions
        &Arc<wgpu::Buffer>, // velocities
        &Arc<wgpu::Buffer>, // densities
        &Arc<wgpu::Buffer>, // c_col0
        &Arc<wgpu::Buffer>, // c_col1
        &Arc<wgpu::Buffer>, // c_col2
    ) {
        (
            &self.out_positions_buffer,
            &self.out_velocities_buffer,
            &self.out_densities_buffer,
            &self.out_c_col0_buffer,
            &self.out_c_col1_buffer,
            &self.out_c_col2_buffer,
        )
    }
}
