//! GPU-accelerated Grid-to-Particle (G2P) transfer for water particles.
//!
//! This module implements G2P for APIC-FLIP using compute shaders.
//! Unlike P2G (scatter), G2P is a gather operation where each particle reads
//! from its 3x3 grid neighborhood - no atomics needed.
//!
//! Output per particle:
//! - Updated velocity (FLIP/PIC blend)
//! - Reconstructed C matrix (APIC affine velocity)
//!
//! The FLIP delta is computed using grid_u_old/grid_v_old (pre-force velocities)
//! stored by the caller after P2G but before forces are applied.

use bytemuck::{Pod, Zeroable};

use super::GpuContext;

/// Parameters for G2P compute shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct G2pParams {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    d_inv: f32,        // APIC D inverse = 4/dx^2
    flip_ratio: f32,   // FLIP blend ratio (0.97 for water)
    dt: f32,           // Time step for velocity clamping
    max_velocity: f32, // Safety clamp (2000.0)
}

/// GPU-based Grid-to-Particle transfer using compute shaders
pub struct GpuG2pSolver {
    width: u32,
    height: u32,

    // Particle buffers
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,
    c_matrices_buffer: wgpu::Buffer,

    // Grid buffers (read-only during G2P)
    grid_u_buffer: wgpu::Buffer,      // Post-force velocities
    grid_v_buffer: wgpu::Buffer,
    grid_u_old_buffer: wgpu::Buffer,  // Pre-force velocities (for FLIP delta)
    grid_v_old_buffer: wgpu::Buffer,

    // Staging buffers for readback
    velocities_staging: wgpu::Buffer,
    c_matrices_staging: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,

    // Compute pipeline
    g2p_pipeline: wgpu::ComputePipeline,

    // Bind group
    bind_group: wgpu::BindGroup,

    // Current capacity
    max_particles: usize,

    // Workgroup size
    workgroup_size: u32,
}

impl GpuG2pSolver {
    /// Create a new GPU G2P solver
    pub fn new(gpu: &GpuContext, width: u32, height: u32, max_particles: usize) -> Self {
        Self::new_internal(&gpu.device, width, height, max_particles)
    }

    /// Create a new GPU G2P solver from device directly (for headless testing)
    pub fn new_headless(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
        max_particles: usize,
    ) -> Self {
        Self::new_internal(device, width, height, max_particles)
    }

    fn new_internal(device: &wgpu::Device, width: u32, height: u32, max_particles: usize) -> Self {
        let u_size = ((width + 1) * height) as usize;
        let v_size = (width * (height + 1)) as usize;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G2P Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/g2p.wgsl").into()),
        });

        // Create particle buffers
        let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Positions"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Velocities"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let c_matrices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P C-Matrices"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create grid buffers (uploaded from CPU each frame)
        let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Grid U"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Grid V"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Grid U Old"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Grid V Old"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create staging buffers for readback
        let velocities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Velocities Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_matrices_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P C-Matrices Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P Params"),
            size: std::mem::size_of::<G2pParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout (matches shader bindings 0-7)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G2P Bind Group Layout"),
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
                // 2: velocities (read_write)
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
                // 3: c_matrices (read_write)
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
                // 4: grid_u (read)
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
                // 5: grid_v (read)
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
                // 6: grid_u_old (read)
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
                // 7: grid_v_old (read)
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G2P Bind Group"),
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
                    resource: c_matrices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_u_old_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grid_v_old_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("G2P Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let g2p_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("G2P Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("g2p"),
            compilation_options: Default::default(),
            cache: None,
        });

        let workgroup_size = 256;

        Self {
            width,
            height,
            positions_buffer,
            velocities_buffer,
            c_matrices_buffer,
            grid_u_buffer,
            grid_v_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            velocities_staging,
            c_matrices_staging,
            params_buffer,
            g2p_pipeline,
            bind_group,
            max_particles,
            workgroup_size,
        }
    }

    /// Upload particle and grid data to GPU
    ///
    /// Returns the number of water particles uploaded and their indices.
    /// Only water particles are processed by GPU G2P.
    ///
    /// Note: grid_u_old/grid_v_old should be the grid state BEFORE forces were applied
    /// (stored after P2G). This is used for FLIP delta calculation.
    pub fn upload(
        &self,
        gpu: &GpuContext,
        particles: &[sim::Particle],
        grid_u: &[f32],
        grid_v: &[f32],
        grid_u_old: &[f32],
        grid_v_old: &[f32],
        cell_size: f32,
        dt: f32,
    ) -> (u32, Vec<usize>) {
        // Extract water particles only
        let mut positions = Vec::with_capacity(particles.len());
        let mut velocities = Vec::with_capacity(particles.len());
        let mut c_matrices = Vec::with_capacity(particles.len());
        let mut water_indices = Vec::with_capacity(particles.len());

        for (idx, p) in particles.iter().enumerate() {
            if p.material.is_sediment() {
                continue;
            }

            water_indices.push(idx);
            positions.push([p.position.x, p.position.y]);
            velocities.push([p.velocity.x, p.velocity.y]);
            c_matrices.push([
                p.affine_velocity.x_axis.x,
                p.affine_velocity.x_axis.y,
                p.affine_velocity.y_axis.x,
                p.affine_velocity.y_axis.y,
            ]);
        }

        let particle_count = positions.len() as u32;

        // Upload particle data
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

        // Upload grid data
        gpu.queue
            .write_buffer(&self.grid_u_buffer, 0, bytemuck::cast_slice(grid_u));
        gpu.queue
            .write_buffer(&self.grid_v_buffer, 0, bytemuck::cast_slice(grid_v));
        gpu.queue
            .write_buffer(&self.grid_u_old_buffer, 0, bytemuck::cast_slice(grid_u_old));
        gpu.queue
            .write_buffer(&self.grid_v_old_buffer, 0, bytemuck::cast_slice(grid_v_old));

        // Upload params
        let d_inv = 4.0 / (cell_size * cell_size); // APIC D inverse
        let params = G2pParams {
            cell_size,
            width: self.width,
            height: self.height,
            particle_count,
            d_inv,
            flip_ratio: 0.97,      // Standard FLIP ratio for water
            dt,
            max_velocity: 2000.0,  // Safety clamp
        };
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        (particle_count, water_indices)
    }

    /// Run G2P compute shader
    pub fn compute(&self, gpu: &GpuContext, particle_count: u32) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("G2P Compute Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("G2P Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.g2p_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (particle_count + self.workgroup_size - 1) / self.workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Download particle data from GPU
    ///
    /// Returns (velocities, c_matrices)
    pub fn download(
        &self,
        gpu: &GpuContext,
        particle_count: u32,
    ) -> (Vec<[f32; 2]>, Vec<[f32; 4]>) {
        let count = particle_count as usize;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("G2P Download Encoder"),
            });

        // Copy to staging
        encoder.copy_buffer_to_buffer(
            &self.velocities_buffer,
            0,
            &self.velocities_staging,
            0,
            (count * std::mem::size_of::<[f32; 2]>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.c_matrices_buffer,
            0,
            &self.c_matrices_staging,
            0,
            (count * std::mem::size_of::<[f32; 4]>()) as u64,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map and read velocities
        let velocities = {
            let buffer_slice = self.velocities_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();
            let result: Vec<[f32; 2]> = bytemuck::cast_slice(&data)[..count].to_vec();
            drop(data);
            self.velocities_staging.unmap();
            result
        };

        // Map and read c_matrices
        let c_matrices = {
            let buffer_slice = self.c_matrices_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();
            let result: Vec<[f32; 4]> = bytemuck::cast_slice(&data)[..count].to_vec();
            drop(data);
            self.c_matrices_staging.unmap();
            result
        };

        (velocities, c_matrices)
    }

    /// Execute full G2P pipeline: upload → compute → download → apply to particles
    ///
    /// Updates water particles in place with new velocities and C matrices.
    /// Sediment particles are NOT touched - they should be handled by CPU G2P.
    ///
    /// Note: grid_u_old/grid_v_old should be the grid state BEFORE forces were applied
    /// (stored after P2G). This is used for FLIP delta calculation.
    pub fn execute(
        &self,
        gpu: &GpuContext,
        particles: &mut [sim::Particle],
        grid_u: &[f32],
        grid_v: &[f32],
        grid_u_old: &[f32],
        grid_v_old: &[f32],
        cell_size: f32,
        dt: f32,
    ) {
        let (particle_count, water_indices) = self.upload(
            gpu,
            particles,
            grid_u,
            grid_v,
            grid_u_old,
            grid_v_old,
            cell_size,
            dt,
        );

        if particle_count == 0 {
            return;
        }

        self.compute(gpu, particle_count);

        let (velocities, c_matrices) = self.download(gpu, particle_count);

        // Apply results back to particles
        for (i, &idx) in water_indices.iter().enumerate() {
            let p = &mut particles[idx];
            p.velocity.x = velocities[i][0];
            p.velocity.y = velocities[i][1];
            p.affine_velocity.x_axis.x = c_matrices[i][0];
            p.affine_velocity.x_axis.y = c_matrices[i][1];
            p.affine_velocity.y_axis.x = c_matrices[i][2];
            p.affine_velocity.y_axis.y = c_matrices[i][3];
        }
    }
}
