//! GPU-accelerated SPH fluid simulation using IISPH pressure solver.
//!
//! Implements "Implicit Incompressible SPH" for volume-preserving fluids.
//! Target: 1M particles at 60 FPS on Apple Silicon.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Diagnostic metrics for one simulation frame
#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
    pub particle_count: u32,
    /// Average density error as ratio (0.01 = 1%)
    pub avg_density_error: f32,
    pub max_density: f32,
    pub min_density: f32,
    pub avg_pressure: f32,
    pub max_pressure: f32,
    /// Y position diagnostics (to detect collapse)
    pub min_y: f32,
    pub max_y: f32,
    pub avg_y: f32,
    /// Y spread = max_y - min_y (should increase as particles stack)
    pub y_spread: f32,
}

/// SPH simulation parameters (uniform buffer)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SphParams {
    /// Number of active particles
    pub num_particles: u32,
    /// Kernel support radius
    pub h: f32,
    /// h²  (precomputed)
    pub h2: f32,
    /// Rest density (kg/m³)
    pub rest_density: f32,

    /// Timestep
    pub dt: f32,
    /// dt² (precomputed)
    pub dt2: f32,
    /// Gravity (typically -9.81)
    pub gravity: f32,
    /// IISPH relaxation factor (0.5 typical)
    pub omega: f32,

    /// Grid cell size (= h for spatial hash)
    pub cell_size: f32,
    /// Grid dimensions for spatial hash
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,

    /// Poly6 kernel coefficient (precomputed)
    pub poly6_coef: f32,
    /// Spiky gradient coefficient (precomputed)
    pub spiky_grad_coef: f32,
    /// Number of IISPH iterations
    pub pressure_iters: u32,
    pub _pad: u32,
}

impl SphParams {
    pub fn new(num_particles: u32, h: f32, dt: f32, grid_dims: [u32; 3]) -> Self {
        use std::f32::consts::PI;

        let h2 = h * h;
        let h3 = h * h * h;
        let h6 = h3 * h3;
        let h9 = h6 * h3;

        Self {
            num_particles,
            h,
            h2,
            rest_density: 1000.0,

            dt,
            dt2: dt * dt,
            gravity: -9.81,
            omega: 0.5,

            cell_size: h,
            grid_size_x: grid_dims[0],
            grid_size_y: grid_dims[1],
            grid_size_z: grid_dims[2],

            poly6_coef: 315.0 / (64.0 * PI * h9),
            spiky_grad_coef: -45.0 / (PI * h6),
            pressure_iters: 20,
            _pad: 0,
        }
    }
}

/// GPU SPH simulation state
pub struct GpuSph3D {
    // Parameters
    params: SphParams,
    params_buffer: wgpu::Buffer,

    // Particle buffers (SoA layout)
    pub positions: wgpu::Buffer,
    pub velocities: wgpu::Buffer,
    positions_pred: wgpu::Buffer,
    densities: wgpu::Buffer,
    pressures: wgpu::Buffer,
    d_ii: wgpu::Buffer,           // IISPH diagonal
    sum_dij_pj: wgpu::Buffer,     // IISPH off-diagonal sum

    // Debug staging buffers (for CPU readback)
    debug_density_staging: wgpu::Buffer,
    debug_pressure_staging: wgpu::Buffer,

    // Spatial hash buffers
    cell_indices: wgpu::Buffer,   // Per-particle cell hash
    particle_order: wgpu::Buffer, // Sorted particle indices
    cell_offsets: wgpu::Buffer,   // Start index per cell

    // Pipelines
    predict_hash_pipeline: wgpu::ComputePipeline,
    build_offsets_pipeline: wgpu::ComputePipeline,
    density_dii_pipeline: wgpu::ComputePipeline,
    sum_dij_pipeline: wgpu::ComputePipeline,
    update_pressure_pipeline: wgpu::ComputePipeline,
    apply_pressure_pipeline: wgpu::ComputePipeline,
    boundary_pipeline: wgpu::ComputePipeline,

    // Sort pipelines (bitonic sort stages)
    sort_local_pipeline: wgpu::ComputePipeline,
    sort_global_pipeline: wgpu::ComputePipeline,

    // Simple test pipeline (gravity + boundary only)
    simple_step_pipeline: wgpu::ComputePipeline,

    // Brute-force IISPH pipelines (O(n²) - for physics validation only)
    bf_predict_pipeline: wgpu::ComputePipeline,
    bf_density_dii_pipeline: wgpu::ComputePipeline,
    bf_sum_dij_pipeline: wgpu::ComputePipeline,
    bf_update_pressure_pipeline: wgpu::ComputePipeline,
    bf_apply_pressure_pipeline: wgpu::ComputePipeline,
    bf_boundary_pipeline: wgpu::ComputePipeline,

    // Bind groups
    bind_group: wgpu::BindGroup,

    // Limits
    max_particles: u32,
    num_cells: u32,
}

impl GpuSph3D {
    pub fn new(
        device: &wgpu::Device,
        max_particles: u32,
        h: f32,
        dt: f32,
        grid_dims: [u32; 3],
    ) -> Self {
        let params = SphParams::new(0, h, dt, grid_dims);
        let num_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];

        // Create uniform buffer
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SPH Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create particle buffers
        let create_particle_buffer = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (max_particles as usize * size) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        // Positions needs VERTEX usage for rendering
        let positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPH Positions"),
            size: (max_particles as usize * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let velocities = create_particle_buffer("SPH Velocities", 16);
        let positions_pred = create_particle_buffer("SPH Positions Predicted", 16);
        let densities = create_particle_buffer("SPH Densities", 4);
        let pressures = create_particle_buffer("SPH Pressures", 4);
        let d_ii = create_particle_buffer("SPH d_ii", 4);
        let sum_dij_pj = create_particle_buffer("SPH sum_dij_pj", 4);

        // Debug staging buffers for CPU readback
        let debug_density_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPH Debug Density Staging"),
            size: (max_particles as usize * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_pressure_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPH Debug Pressure Staging"),
            size: (max_particles as usize * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Spatial hash buffers
        let cell_indices = create_particle_buffer("SPH Cell Indices", 4);
        let particle_order = create_particle_buffer("SPH Particle Order", 4);

        let cell_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPH Cell Offsets"),
            size: ((num_cells + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SPH Bind Group Layout"),
            entries: &[
                // 0: Params (uniform)
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
                // 1: Positions
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
                // 2: Velocities
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
                // 3: Positions Predicted
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
                // 4: Densities
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
                // 5: Pressures
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
                // 6: d_ii
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
                // 7: sum_dij_pj
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
                // 8: Cell indices
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
                // 9: Particle order
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
                // 10: Cell offsets
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
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SPH Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: positions_pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: densities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pressures.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: d_ii.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: sum_dij_pj.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cell_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: particle_order.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: cell_offsets.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SPH Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shaders and create pipelines
        let create_pipeline = |label: &str, source: &str, entry: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // Load IISPH shader
        let iisph_shader = include_str!("shaders/sph_iisph.wgsl");

        let predict_hash_pipeline = create_pipeline("SPH Predict+Hash", iisph_shader, "predict_and_hash");
        let build_offsets_pipeline = create_pipeline("SPH Build Offsets", iisph_shader, "build_offsets");
        let density_dii_pipeline = create_pipeline("SPH Density+dii", iisph_shader, "compute_density_dii");
        let sum_dij_pipeline = create_pipeline("SPH Sum dij", iisph_shader, "compute_sum_dij");
        let update_pressure_pipeline = create_pipeline("SPH Update Pressure", iisph_shader, "update_pressure");
        let apply_pressure_pipeline = create_pipeline("SPH Apply Pressure", iisph_shader, "apply_pressure");
        let boundary_pipeline = create_pipeline("SPH Boundary", iisph_shader, "boundary_collision");

        // Bitonic sort - use separate shader
        let sort_shader = include_str!("shaders/sph_bitonic_sort.wgsl");
        let sort_local_pipeline = create_pipeline("SPH Sort Local", sort_shader, "bitonic_sort_local");
        let sort_global_pipeline = create_pipeline("SPH Sort Global", sort_shader, "bitonic_sort_global");

        // Simple test shader (gravity + boundary only)
        let simple_shader = include_str!("shaders/sph_simple.wgsl");
        let simple_step_pipeline = create_pipeline("SPH Simple Step", simple_shader, "simple_step");

        // Brute-force IISPH shader (O(n²) - for physics validation)
        let bf_shader = include_str!("shaders/sph_bruteforce.wgsl");
        let bf_predict_pipeline = create_pipeline("SPH BF Predict", bf_shader, "bf_predict");
        let bf_density_dii_pipeline = create_pipeline("SPH BF Density+dii", bf_shader, "bf_density_dii");
        let bf_sum_dij_pipeline = create_pipeline("SPH BF Sum dij", bf_shader, "bf_sum_dij");
        let bf_update_pressure_pipeline = create_pipeline("SPH BF Update Pressure", bf_shader, "bf_update_pressure");
        let bf_apply_pressure_pipeline = create_pipeline("SPH BF Apply Pressure", bf_shader, "bf_apply_pressure");
        let bf_boundary_pipeline = create_pipeline("SPH BF Boundary", bf_shader, "bf_boundary");

        Self {
            params,
            params_buffer,
            positions,
            velocities,
            positions_pred,
            densities,
            pressures,
            d_ii,
            sum_dij_pj,
            debug_density_staging,
            debug_pressure_staging,
            cell_indices,
            particle_order,
            cell_offsets,
            predict_hash_pipeline,
            build_offsets_pipeline,
            density_dii_pipeline,
            sum_dij_pipeline,
            update_pressure_pipeline,
            apply_pressure_pipeline,
            boundary_pipeline,
            sort_local_pipeline,
            sort_global_pipeline,
            simple_step_pipeline,
            bf_predict_pipeline,
            bf_density_dii_pipeline,
            bf_sum_dij_pipeline,
            bf_update_pressure_pipeline,
            bf_apply_pressure_pipeline,
            bf_boundary_pipeline,
            bind_group,
            max_particles,
            num_cells,
        }
    }

    /// Upload particle data from CPU (replaces all particles)
    pub fn upload_particles(
        &mut self,
        queue: &wgpu::Queue,
        positions: &[Vec3],
        velocities: &[Vec3],
    ) {
        let n = positions.len().min(self.max_particles as usize);
        self.params.num_particles = n as u32;

        // Update params
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // Convert Vec3 to padded format (vec3 -> vec4 for GPU alignment)
        let pos_padded: Vec<[f32; 4]> = positions.iter()
            .take(n)
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();
        let vel_padded: Vec<[f32; 4]> = velocities.iter()
            .take(n)
            .map(|v| [v.x, v.y, v.z, 0.0])
            .collect();

        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(&pos_padded));
        queue.write_buffer(&self.velocities, 0, bytemuck::cast_slice(&vel_padded));
    }

    /// Append new particles without overwriting existing GPU state
    /// Returns the new total particle count
    pub fn append_particles(
        &mut self,
        queue: &wgpu::Queue,
        positions: &[Vec3],
        velocities: &[Vec3],
    ) -> u32 {
        let current = self.params.num_particles as usize;
        let to_add = positions.len().min(self.max_particles as usize - current);
        if to_add == 0 { return self.params.num_particles; }

        let new_total = current + to_add;
        self.params.num_particles = new_total as u32;

        // Update params
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // Convert Vec3 to padded format
        let pos_padded: Vec<[f32; 4]> = positions.iter()
            .take(to_add)
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();
        let vel_padded: Vec<[f32; 4]> = velocities.iter()
            .take(to_add)
            .map(|v| [v.x, v.y, v.z, 0.0])
            .collect();

        // Write at offset (current * 16 bytes per vec4)
        let offset = (current * 16) as u64;
        queue.write_buffer(&self.positions, offset, bytemuck::cast_slice(&pos_padded));
        queue.write_buffer(&self.velocities, offset, bytemuck::cast_slice(&vel_padded));

        self.params.num_particles
    }

    /// Run one simple simulation step (gravity + boundary only, no SPH)
    /// Use this to debug basic particle motion before full IISPH
    pub fn step_simple(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = (self.params.num_particles + 255) / 256;
        if workgroups == 0 { return; }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SPH: Simple Step"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.simple_step_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Run one brute-force IISPH step (O(n²) - for physics validation only)
    /// Use with ~5-10k particles max due to quadratic complexity
    pub fn step_bruteforce(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = (self.params.num_particles + 255) / 256;
        if workgroups == 0 { return; }

        // 1. Predict positions + apply gravity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH BF: Predict"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bf_predict_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 2. Compute density + d_ii (brute force O(n²))
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH BF: Density + d_ii"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bf_density_dii_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 3. IISPH pressure iterations
        for _ in 0..self.params.pressure_iters {
            // 3a. Compute sum(d_ij * p_j) (brute force O(n²))
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH BF: Sum d_ij"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bf_sum_dij_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // 3b. Update pressure (Jacobi relaxation)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH BF: Update Pressure"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bf_update_pressure_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // 4. Apply pressure forces + integrate (brute force O(n²))
        // Note: Boundary collision is handled inline in bf_apply_pressure
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH BF: Apply Pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bf_apply_pressure_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Note: bf_boundary dispatch removed - boundary collision is already
        // handled inline in bf_apply_pressure kernel to avoid double-processing
    }

    /// Run one simulation step
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = (self.params.num_particles + 255) / 256;
        if workgroups == 0 { return; }

        // 1. Predict positions + compute cell hash
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH: Predict + Hash"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.predict_hash_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 2. Sort particles by cell (bitonic sort)
        self.bitonic_sort(encoder);

        // 3. Build cell offset table
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH: Build Offsets"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_offsets_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 4. Compute density + d_ii (IISPH diagonal)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH: Density + d_ii"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_dii_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 5. IISPH pressure iterations
        for _ in 0..self.params.pressure_iters {
            // 5a. Compute sum(d_ij * p_j)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH: Sum d_ij"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sum_dij_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // 5b. Update pressure (Jacobi relaxation)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH: Update Pressure"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_pressure_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // 6. Apply pressure forces + integrate
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH: Apply Pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.apply_pressure_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 7. Boundary collision (SDF)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SPH: Boundary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.boundary_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Bitonic sort for spatial hash
    fn bitonic_sort(&self, encoder: &mut wgpu::CommandEncoder) {
        let n = self.params.num_particles;
        if n <= 1 { return; }

        // Find next power of 2
        let n_padded = n.next_power_of_two();
        let workgroups = (n_padded + 255) / 256;

        // Bitonic sort: O(log²(n)) stages
        let mut k = 2u32;
        while k <= n_padded {
            let mut j = k / 2;
            while j > 0 {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("SPH: Bitonic Sort"),
                        timestamp_writes: None,
                    });
                    // For now use global sort - we'll optimize with local sort later
                    pass.set_pipeline(&self.sort_global_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                j /= 2;
            }
            k *= 2;
        }
    }

    /// Get current particle count
    pub fn num_particles(&self) -> u32 {
        self.params.num_particles
    }

    /// Get max particle capacity
    pub fn max_particles(&self) -> u32 {
        self.max_particles
    }

    /// Get rest density parameter
    pub fn rest_density(&self) -> f32 {
        self.params.rest_density
    }

    /// Set rest density parameter (updates GPU buffer)
    pub fn set_rest_density(&mut self, queue: &wgpu::Queue, rest_density: f32) {
        self.params.rest_density = rest_density;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// Set the timestep (and precomputed dt²)
    pub fn set_timestep(&mut self, queue: &wgpu::Queue, dt: f32) {
        self.params.dt = dt;
        self.params.dt2 = dt * dt;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// Get the current timestep
    pub fn timestep(&self) -> f32 {
        self.params.dt
    }

    /// Set gravity (m/s²)
    pub fn set_gravity(&mut self, queue: &wgpu::Queue, gravity: f32) {
        self.params.gravity = gravity;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// Calibrate rest_density to match actual kernel density sum
    ///
    /// Creates a test grid of particles, runs density computation, and returns
    /// the median density. This should be called once at startup.
    ///
    /// The IISPH pressure solver compares actual density against rest_density.
    /// If rest_density doesn't match what the kernel actually produces,
    /// pressure will be computed incorrectly (either always positive or always negative).
    pub fn calibrate_rest_density(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> f32 {
        // Create a 10x10x10 grid of particles at spacing h*0.5
        let spacing = self.params.h * 0.5;
        let grid_size = 10;
        let center = Vec3::new(0.3, 0.3, 0.3);

        let mut positions = Vec::with_capacity(grid_size * grid_size * grid_size);
        let mut velocities = Vec::with_capacity(grid_size * grid_size * grid_size);

        for x in 0..grid_size {
            for y in 0..grid_size {
                for z in 0..grid_size {
                    let pos = center + Vec3::new(
                        (x as f32 - grid_size as f32 / 2.0) * spacing,
                        (y as f32 - grid_size as f32 / 2.0) * spacing,
                        (z as f32 - grid_size as f32 / 2.0) * spacing,
                    );
                    positions.push(pos);
                    velocities.push(Vec3::ZERO);
                }
            }
        }

        // Upload particles
        self.upload_particles(queue, &positions, &velocities);

        // Run bf_predict kernel (copies positions to positions_pred)
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Calibration Predict"),
            });
            let workgroups = (self.params.num_particles + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calibration BF Predict"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bf_predict_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Run bf_density_dii kernel (computes density)
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Calibration Density"),
            });
            let workgroups = (self.params.num_particles + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calibration BF Density"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bf_density_dii_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Read back densities
        let densities = self.read_densities(device, queue);

        // Get median density from middle particles (avoiding boundary effects)
        // Middle particles are those in the center 6x6x6 region
        let mut middle_densities = Vec::new();
        for x in 2..8 {
            for y in 2..8 {
                for z in 2..8 {
                    let idx = x * grid_size * grid_size + y * grid_size + z;
                    if idx < densities.len() {
                        middle_densities.push(densities[idx]);
                    }
                }
            }
        }

        // Sort and get median
        middle_densities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_density = if middle_densities.is_empty() {
            1000.0 // Fallback
        } else {
            middle_densities[middle_densities.len() / 2]
        };

        // Reset particle count to 0 so bucket_test starts fresh
        self.params.num_particles = 0;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        median_density
    }

    /// Read density values from GPU to CPU (blocking)
    ///
    /// Copies from GPU storage buffer to staging buffer, then maps and reads.
    /// This is a blocking operation - use sparingly (e.g., every 60 frames for diagnostics).
    /// Read position values from GPU to CPU (blocking)
    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Vec3> {
        let n = self.params.num_particles as usize;
        if n == 0 {
            return Vec::new();
        }

        let bytes_to_copy = (n * 16) as u64; // vec4<f32> = 16 bytes

        // Create temporary staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Staging"),
            size: bytes_to_copy,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU storage to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Positions Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.positions, 0, &staging, 0, bytes_to_copy);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read staging buffer
        let slice = staging.slice(0..bytes_to_copy);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let raw: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let positions: Vec<Vec3> = raw.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
        drop(data);
        staging.unmap();

        positions
    }

    pub fn read_densities(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let n = self.params.num_particles as usize;
        if n == 0 {
            return Vec::new();
        }

        let bytes_to_copy = (n * 4) as u64;

        // Copy from GPU storage to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Densities Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.densities, 0, &self.debug_density_staging, 0, bytes_to_copy);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read staging buffer
        let slice = self.debug_density_staging.slice(0..bytes_to_copy);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let densities: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.debug_density_staging.unmap();

        densities
    }

    /// Read pressure values from GPU to CPU (blocking)
    ///
    /// Copies from GPU storage buffer to staging buffer, then maps and reads.
    /// This is a blocking operation - use sparingly (e.g., every 60 frames for diagnostics).
    pub fn read_pressures(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let n = self.params.num_particles as usize;
        if n == 0 {
            return Vec::new();
        }

        let bytes_to_copy = (n * 4) as u64;

        // Copy from GPU storage to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Pressures Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.pressures, 0, &self.debug_pressure_staging, 0, bytes_to_copy);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read staging buffer
        let slice = self.debug_pressure_staging.slice(0..bytes_to_copy);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let pressures: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.debug_pressure_staging.unmap();

        pressures
    }

    /// Compute diagnostic metrics for the current simulation state (blocking)
    ///
    /// Reads densities and pressures from GPU and computes statistics.
    /// This is a blocking operation - use sparingly (e.g., every 60 frames).
    pub fn compute_metrics(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> FrameMetrics {
        let n = self.params.num_particles;
        if n == 0 {
            return FrameMetrics {
                particle_count: 0,
                avg_density_error: 0.0,
                max_density: 0.0,
                min_density: 0.0,
                avg_pressure: 0.0,
                max_pressure: 0.0,
                min_y: 0.0,
                max_y: 0.0,
                avg_y: 0.0,
                y_spread: 0.0,
            };
        }

        let positions = self.read_positions(device, queue);
        let densities = self.read_densities(device, queue);
        let pressures = self.read_pressures(device, queue);

        let rest_density = self.params.rest_density;

        // Compute density statistics
        let mut sum_density_error = 0.0f32;
        let mut max_density = f32::MIN;
        let mut min_density = f32::MAX;

        for &rho in &densities {
            let error = (rho - rest_density).abs() / rest_density;
            sum_density_error += error;
            max_density = max_density.max(rho);
            min_density = min_density.min(rho);
        }

        let avg_density_error = sum_density_error / n as f32;

        // Compute pressure statistics
        let mut sum_pressure = 0.0f32;
        let mut max_pressure = f32::MIN;

        for &p in &pressures {
            sum_pressure += p;
            max_pressure = max_pressure.max(p);
        }

        let avg_pressure = sum_pressure / n as f32;

        // Compute Y position statistics (to detect collapse)
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut sum_y = 0.0f32;

        for pos in &positions {
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
            sum_y += pos.y;
        }

        let avg_y = sum_y / n as f32;
        let y_spread = max_y - min_y;

        FrameMetrics {
            particle_count: n,
            avg_density_error,
            max_density,
            min_density,
            avg_pressure,
            max_pressure,
            min_y,
            max_y,
            avg_y,
            y_spread,
        }
    }
}
