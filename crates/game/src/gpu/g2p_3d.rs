//! GPU-accelerated Grid-to-Particle (G2P) transfer for 3D FLIP simulation.
//!
//! This module implements G2P for APIC-FLIP using 3D compute shaders.
//! Unlike P2G (scatter), G2P is a gather operation where each particle reads
//! from its 3x3x3 grid neighborhood - no atomics needed.
//!
//! Output per particle:
//! - Updated velocity (FLIP/PIC blend)
//! - Reconstructed C matrix (3x3 APIC affine velocity)
//!
//! The FLIP delta is computed using grid_*_old (pre-force velocities)
//! stored by the caller after P2G but before forces are applied.

use bytemuck::{Pod, Zeroable};

/// Parameters for G2P 3D compute shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct G2pParams3D {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    d_inv: f32,        // APIC D inverse = 4/dx^2
    flip_ratio: f32,   // FLIP blend ratio (0.97 for water)
    dt: f32,           // Time step for velocity clamping
    max_velocity: f32, // Safety clamp (2000.0)
    _padding: [f32; 3], // Align to 48 bytes
}

/// GPU-based Grid-to-Particle transfer for 3D simulation
pub struct GpuG2p3D {
    width: u32,
    height: u32,
    depth: u32,

    // Particle buffers
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,
    // C matrix columns (output)
    c_col0_buffer: wgpu::Buffer,
    c_col1_buffer: wgpu::Buffer,
    c_col2_buffer: wgpu::Buffer,

    // Grid buffers (read-only during G2P)
    grid_u_buffer: wgpu::Buffer,      // Post-force velocities
    grid_v_buffer: wgpu::Buffer,
    grid_w_buffer: wgpu::Buffer,
    grid_u_old_buffer: wgpu::Buffer,  // Pre-force velocities (for FLIP delta)
    grid_v_old_buffer: wgpu::Buffer,
    grid_w_old_buffer: wgpu::Buffer,

    // Staging buffers for readback
    velocities_staging: wgpu::Buffer,
    c_col0_staging: wgpu::Buffer,
    c_col1_staging: wgpu::Buffer,
    c_col2_staging: wgpu::Buffer,

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

impl GpuG2p3D {
    /// Create a new GPU G2P 3D solver
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        max_particles: usize,
    ) -> Self {
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G2P 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/g2p_3d.wgsl").into()),
        });

        // Create particle buffers (vec3 padded to vec4)
        let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Positions"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Velocities"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // C matrix columns (output)
        let c_col0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col0"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let c_col1_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col1"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let c_col2_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col2"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Grid buffers
        let grid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Grid U"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Grid V"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_w_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Grid W"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Grid U Old"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Grid V Old"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_w_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Grid W Old"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffers for readback
        let velocities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Velocities Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_col0_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col0 Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_col1_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col1 Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_col2_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col2 Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Params"),
            size: std::mem::size_of::<G2pParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout (matches shader bindings 0-11)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G2P 3D Bind Group Layout"),
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
                // 3-5: c_col0, c_col1, c_col2 (read_write)
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                // 6-8: grid_u, grid_v, grid_w (read)
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
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9-11: grid_u_old, grid_v_old, grid_w_old (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
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
            label: Some("G2P 3D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_col0_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: c_col1_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: c_col2_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: grid_w_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: grid_u_old_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: grid_v_old_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: grid_w_old_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("G2P 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let g2p_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("G2P 3D Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("g2p"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            height,
            depth,
            positions_buffer,
            velocities_buffer,
            c_col0_buffer,
            c_col1_buffer,
            c_col2_buffer,
            grid_u_buffer,
            grid_v_buffer,
            grid_w_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            grid_w_old_buffer,
            velocities_staging,
            c_col0_staging,
            c_col1_staging,
            c_col2_staging,
            params_buffer,
            g2p_pipeline,
            bind_group,
            max_particles,
            workgroup_size: 256,
        }
    }

    /// Upload particle and grid data to GPU
    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        positions: &[glam::Vec3],
        velocities: &[glam::Vec3],
        c_matrices: &[glam::Mat3],
        grid_u: &[f32],
        grid_v: &[f32],
        grid_w: &[f32],
        grid_u_old: &[f32],
        grid_v_old: &[f32],
        grid_w_old: &[f32],
        cell_size: f32,
        dt: f32,
    ) -> u32 {
        // Validate grid buffer sizes to prevent silent partial writes
        let expected_u_size = ((self.width + 1) * self.height * self.depth) as usize;
        let expected_v_size = (self.width * (self.height + 1) * self.depth) as usize;
        let expected_w_size = (self.width * self.height * (self.depth + 1)) as usize;

        assert_eq!(
            grid_u.len(), expected_u_size,
            "Grid U size mismatch: got {}, expected {}",
            grid_u.len(), expected_u_size
        );
        assert_eq!(
            grid_v.len(), expected_v_size,
            "Grid V size mismatch: got {}, expected {}",
            grid_v.len(), expected_v_size
        );
        assert_eq!(
            grid_w.len(), expected_w_size,
            "Grid W size mismatch: got {}, expected {}",
            grid_w.len(), expected_w_size
        );
        assert_eq!(
            grid_u_old.len(), expected_u_size,
            "Grid U old size mismatch: got {}, expected {}",
            grid_u_old.len(), expected_u_size
        );
        assert_eq!(
            grid_v_old.len(), expected_v_size,
            "Grid V old size mismatch: got {}, expected {}",
            grid_v_old.len(), expected_v_size
        );
        assert_eq!(
            grid_w_old.len(), expected_w_size,
            "Grid W old size mismatch: got {}, expected {}",
            grid_w_old.len(), expected_w_size
        );

        let particle_count = positions.len().min(self.max_particles) as u32;

        // Convert to padded vec4 format
        let positions_padded: Vec<[f32; 4]> = positions.iter()
            .take(particle_count as usize)
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();

        let velocities_padded: Vec<[f32; 4]> = velocities.iter()
            .take(particle_count as usize)
            .map(|v| [v.x, v.y, v.z, 0.0])
            .collect();

        // Extract C matrix columns
        let c_col0: Vec<[f32; 4]> = c_matrices.iter()
            .take(particle_count as usize)
            .map(|c| [c.x_axis.x, c.x_axis.y, c.x_axis.z, 0.0])
            .collect();

        let c_col1: Vec<[f32; 4]> = c_matrices.iter()
            .take(particle_count as usize)
            .map(|c| [c.y_axis.x, c.y_axis.y, c.y_axis.z, 0.0])
            .collect();

        let c_col2: Vec<[f32; 4]> = c_matrices.iter()
            .take(particle_count as usize)
            .map(|c| [c.z_axis.x, c.z_axis.y, c.z_axis.z, 0.0])
            .collect();

        // Upload particle data
        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions_padded));
        queue.write_buffer(&self.velocities_buffer, 0, bytemuck::cast_slice(&velocities_padded));
        queue.write_buffer(&self.c_col0_buffer, 0, bytemuck::cast_slice(&c_col0));
        queue.write_buffer(&self.c_col1_buffer, 0, bytemuck::cast_slice(&c_col1));
        queue.write_buffer(&self.c_col2_buffer, 0, bytemuck::cast_slice(&c_col2));

        // Upload grid data
        queue.write_buffer(&self.grid_u_buffer, 0, bytemuck::cast_slice(grid_u));
        queue.write_buffer(&self.grid_v_buffer, 0, bytemuck::cast_slice(grid_v));
        queue.write_buffer(&self.grid_w_buffer, 0, bytemuck::cast_slice(grid_w));
        queue.write_buffer(&self.grid_u_old_buffer, 0, bytemuck::cast_slice(grid_u_old));
        queue.write_buffer(&self.grid_v_old_buffer, 0, bytemuck::cast_slice(grid_v_old));
        queue.write_buffer(&self.grid_w_old_buffer, 0, bytemuck::cast_slice(grid_w_old));

        // Upload params
        let d_inv = 4.0 / (cell_size * cell_size);
        let params = G2pParams3D {
            cell_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            d_inv,
            flip_ratio: 0.97,
            dt,
            max_velocity: 2000.0,
            _padding: [0.0; 3],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        particle_count
    }

    /// Encode G2P compute pass into command encoder
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, particle_count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("G2P 3D Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.g2p_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let workgroups = (particle_count + self.workgroup_size - 1) / self.workgroup_size;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Download particle data from GPU
    pub fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
    ) {
        let count = particle_count as usize;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G2P 3D Download Encoder"),
        });

        // Copy to staging
        encoder.copy_buffer_to_buffer(&self.velocities_buffer, 0, &self.velocities_staging, 0, (count * 16) as u64);
        encoder.copy_buffer_to_buffer(&self.c_col0_buffer, 0, &self.c_col0_staging, 0, (count * 16) as u64);
        encoder.copy_buffer_to_buffer(&self.c_col1_buffer, 0, &self.c_col1_staging, 0, (count * 16) as u64);
        encoder.copy_buffer_to_buffer(&self.c_col2_buffer, 0, &self.c_col2_staging, 0, (count * 16) as u64);

        queue.submit(std::iter::once(encoder.finish()));

        // Read velocities
        let vel_data = Self::read_staging_buffer_vec4(device, &self.velocities_staging, count);
        for (i, v) in vel_data.iter().enumerate().take(count) {
            velocities[i] = glam::Vec3::new(v[0], v[1], v[2]);
        }

        // Read C matrices
        let c0_data = Self::read_staging_buffer_vec4(device, &self.c_col0_staging, count);
        let c1_data = Self::read_staging_buffer_vec4(device, &self.c_col1_staging, count);
        let c2_data = Self::read_staging_buffer_vec4(device, &self.c_col2_staging, count);

        for i in 0..count {
            c_matrices[i] = glam::Mat3::from_cols(
                glam::Vec3::new(c0_data[i][0], c0_data[i][1], c0_data[i][2]),
                glam::Vec3::new(c1_data[i][0], c1_data[i][1], c1_data[i][2]),
                glam::Vec3::new(c2_data[i][0], c2_data[i][1], c2_data[i][2]),
            );
        }
    }

    fn read_staging_buffer_vec4(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Vec<[f32; 4]> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<[f32; 4]> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}
