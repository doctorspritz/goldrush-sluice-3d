//! GPU-accelerated 3D FLIP/APIC simulation.
//!
//! This module provides a complete GPU-based 3D fluid simulation that combines:
//! - P2G (Particle-to-Grid) with atomic scatter
//! - Pressure solve (Red-Black Gauss-Seidel)
//! - G2P (Grid-to-Particle) with FLIP/PIC blend
//!
//! The simulation maintains particle data on CPU but does all heavy computation on GPU.

use super::g2p_3d::GpuG2p3D;
use super::p2g_3d::GpuP2g3D;
use super::pressure_3d::GpuPressure3D;

use bytemuck::{Pod, Zeroable};

/// Gravity application parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GravityParams3D {
    width: u32,
    height: u32,
    depth: u32,
    gravity_dt: f32,
}

/// Boundary condition parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BcParams3D {
    width: u32,
    height: u32,
    depth: u32,
    _pad: u32,
}

/// Velocity clamping parameters (matches CPU MAX_GRID_VEL for CFL stability)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ClampParams3D {
    width: u32,
    height: u32,
    depth: u32,
    max_vel: f32,
}

/// GPU-accelerated 3D FLIP simulation
pub struct GpuFlip3D {
    // Grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,

    // Sub-solvers
    p2g: GpuP2g3D,
    g2p: GpuG2p3D,
    pressure: GpuPressure3D,

    // Gravity shader
    gravity_pipeline: wgpu::ComputePipeline,
    gravity_bind_group: wgpu::BindGroup,
    gravity_params_buffer: wgpu::Buffer,

    // Boundary condition enforcement shaders
    bc_u_pipeline: wgpu::ComputePipeline,
    bc_v_pipeline: wgpu::ComputePipeline,
    bc_w_pipeline: wgpu::ComputePipeline,
    bc_bind_group: wgpu::BindGroup,
    bc_params_buffer: wgpu::Buffer,

    // Velocity clamping shaders (for CFL stability before pressure solve)
    clamp_u_pipeline: wgpu::ComputePipeline,
    clamp_v_pipeline: wgpu::ComputePipeline,
    clamp_w_pipeline: wgpu::ComputePipeline,
    clamp_bind_group: wgpu::BindGroup,
    clamp_params_buffer: wgpu::Buffer,

    // Grid velocity backup for FLIP delta
    grid_u_old_buffer: wgpu::Buffer,
    grid_v_old_buffer: wgpu::Buffer,
    grid_w_old_buffer: wgpu::Buffer,

    // Maximum particles supported
    max_particles: usize,
}

impl GpuFlip3D {
    /// Create a new GPU 3D FLIP simulation
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        cell_size: f32,
        max_particles: usize,
    ) -> Self {
        // Create P2G solver (owns the grid velocity buffers)
        let p2g = GpuP2g3D::new(device, width, height, depth, max_particles);

        // Create pressure solver (references P2G's grid buffers)
        let pressure = GpuPressure3D::new(
            device,
            width,
            height,
            depth,
            &p2g.grid_u_buffer,
            &p2g.grid_v_buffer,
            &p2g.grid_w_buffer,
        );

        // Create G2P solver
        let g2p = GpuG2p3D::new(device, width, height, depth, max_particles);

        // Create grid velocity backup buffers for FLIP delta
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;

        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid U Old 3D"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid V Old 3D"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_w_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid W Old 3D"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create gravity shader
        let gravity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gravity_3d.wgsl").into()),
        });

        let gravity_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gravity Params 3D"),
            size: std::mem::size_of::<GravityParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Note: We need a cell_type buffer for gravity - borrow from pressure solver
        // For now we'll create a simple gravity pipeline that just modifies grid_v
        let gravity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gravity 3D Bind Group Layout"),
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

        // Use the pressure solver's cell_type buffer for gravity
        let gravity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity 3D Bind Group"),
            layout: &gravity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gravity_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_v_buffer.as_entire_binding() },
            ],
        });

        let gravity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gravity 3D Pipeline Layout"),
            bind_group_layouts: &[&gravity_bind_group_layout],
            push_constant_ranges: &[],
        });

        let gravity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gravity 3D Pipeline"),
            layout: Some(&gravity_pipeline_layout),
            module: &gravity_shader,
            entry_point: Some("apply_gravity"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create boundary condition enforcement shader
        let bc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Enforce BC 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/enforce_bc_3d.wgsl").into()),
        });

        let bc_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BC Params 3D"),
            size: std::mem::size_of::<BcParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bc_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BC 3D Bind Group Layout"),
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
            ],
        });

        let bc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BC 3D Bind Group"),
            layout: &bc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bc_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p2g.grid_w_buffer.as_entire_binding() },
            ],
        });

        let bc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BC 3D Pipeline Layout"),
            bind_group_layouts: &[&bc_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bc_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC U 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC V 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC W 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create velocity clamping shader (for CFL stability before pressure solve)
        let clamp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clamp Vel 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clamp_vel_3d.wgsl").into()),
        });

        let clamp_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Clamp Params 3D"),
            size: std::mem::size_of::<ClampParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clamp_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Clamp Vel 3D Bind Group Layout"),
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
            ],
        });

        let clamp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clamp Vel 3D Bind Group"),
            layout: &clamp_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: clamp_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.grid_w_buffer.as_entire_binding() },
            ],
        });

        let clamp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Clamp Vel 3D Pipeline Layout"),
            bind_group_layouts: &[&clamp_bind_group_layout],
            push_constant_ranges: &[],
        });

        let clamp_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clamp U 3D Pipeline"),
            layout: Some(&clamp_pipeline_layout),
            module: &clamp_shader,
            entry_point: Some("clamp_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let clamp_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clamp V 3D Pipeline"),
            layout: Some(&clamp_pipeline_layout),
            module: &clamp_shader,
            entry_point: Some("clamp_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let clamp_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clamp W 3D Pipeline"),
            layout: Some(&clamp_pipeline_layout),
            module: &clamp_shader,
            entry_point: Some("clamp_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            height,
            depth,
            cell_size,
            p2g,
            g2p,
            pressure,
            gravity_pipeline,
            gravity_bind_group,
            gravity_params_buffer,
            bc_u_pipeline,
            bc_v_pipeline,
            bc_w_pipeline,
            bc_bind_group,
            bc_params_buffer,
            clamp_u_pipeline,
            clamp_v_pipeline,
            clamp_w_pipeline,
            clamp_bind_group,
            clamp_params_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            grid_w_old_buffer,
            max_particles,
        }
    }

    /// Run one simulation step
    ///
    /// This performs the full FLIP pipeline:
    /// 1. P2G: Transfer particle data to grid
    /// 2. Enforce boundary conditions (before storing old velocities!)
    /// 3. Save grid velocity (for FLIP delta)
    /// 4. Apply gravity
    /// 4.5. Clamp grid velocities (CFL stability, matches CPU MAX_GRID_VEL=10.0)
    /// 5. Pressure solve (includes divergence, iterations, gradient)
    /// 5.5. Enforce boundary conditions (after pressure gradient)
    /// 6. G2P: Transfer grid data back to particles
    pub fn step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        cell_types: &[u32],
        dt: f32,
        gravity: f32,
        pressure_iterations: u32,
    ) {
        let particle_count = positions.len().min(self.max_particles);
        if particle_count == 0 {
            return;
        }

        // Upload cell types FIRST (needed for BC enforcement)
        self.pressure.upload_cell_types(queue, cell_types, self.cell_size);

        // Upload BC params
        let bc_params = BcParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            _pad: 0,
        };
        queue.write_buffer(&self.bc_params_buffer, 0, bytemuck::bytes_of(&bc_params));

        // 1. Upload particles and run P2G
        let count = self.p2g.upload_particles(
            queue,
            positions,
            velocities,
            c_matrices,
            self.cell_size,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Step Encoder"),
        });

        // Run P2G scatter and divide
        self.p2g.encode(&mut encoder, count);

        queue.submit(std::iter::once(encoder.finish()));

        // 2. Enforce boundary conditions BEFORE storing old velocities
        // This is critical for correct FLIP delta computation!
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D BC Encoder"),
        });

        // Enforce BC on U
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC U 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_u_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on V
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC V 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_v_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on W
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC W 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_w_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 1 + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 3. Save grid velocity for FLIP delta (now with proper BCs!)
        let (u_size, v_size, w_size) = self.p2g.grid_sizes();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Grid Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p2g.grid_u_buffer, 0, &self.grid_u_old_buffer, 0, (u_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.p2g.grid_v_buffer, 0, &self.grid_v_old_buffer, 0, (v_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.p2g.grid_w_buffer, 0, &self.grid_w_old_buffer, 0, (w_size * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // 4. Apply gravity

        let gravity_params = GravityParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            gravity_dt: gravity * dt,
        };
        queue.write_buffer(&self.gravity_params_buffer, 0, bytemuck::bytes_of(&gravity_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Gravity Encoder"),
        });

        // Apply gravity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravity 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravity_pipeline);
            pass.set_bind_group(0, &self.gravity_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 4.5. Clamp grid velocities BEFORE pressure solve (CFL stability)
        // This matches CPU behavior: clamp_grid_velocities() with MAX_GRID_VEL = 10.0
        let clamp_params = ClampParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            max_vel: 10.0, // Match CPU MAX_GRID_VEL
        };
        queue.write_buffer(&self.clamp_params_buffer, 0, bytemuck::bytes_of(&clamp_params));

        // Clamp U velocities
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clamp U 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clamp_u_pipeline);
            pass.set_bind_group(0, &self.clamp_bind_group, &[]);
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Clamp V velocities
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clamp V 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clamp_v_pipeline);
            pass.set_bind_group(0, &self.clamp_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Clamp W velocities
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clamp W 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clamp_w_pipeline);
            pass.set_bind_group(0, &self.clamp_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 1 + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 5. Pressure solve (divergence → iterations → gradient)
        self.pressure.encode(&mut encoder, pressure_iterations);

        queue.submit(std::iter::once(encoder.finish()));

        // 5.5. Enforce boundary conditions AFTER pressure gradient
        // This is critical! The CPU does this (pressure::enforce_boundary_conditions after apply_pressure_gradient).
        // Without this, pressure gradient can create non-zero velocities at solid boundaries.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Post-Pressure BC Encoder"),
        });

        // Enforce BC on U
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Post-Pressure BC U 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_u_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on V
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Post-Pressure BC V 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_v_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on W
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Post-Pressure BC W 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_w_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 1 + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 6. Download grid velocities and run G2P
        // For now, we'll do a simplified version that downloads grids to CPU
        // A full GPU implementation would keep everything on GPU

        let mut grid_u = vec![0.0f32; u_size];
        let mut grid_v = vec![0.0f32; v_size];
        let mut grid_w = vec![0.0f32; w_size];
        let mut grid_u_old = vec![0.0f32; u_size];
        let mut grid_v_old = vec![0.0f32; v_size];
        let mut grid_w_old = vec![0.0f32; w_size];

        self.p2g.download(device, queue, &mut grid_u, &mut grid_v, &mut grid_w);

        // Read old velocities
        Self::read_buffer(device, queue, &self.grid_u_old_buffer, &mut grid_u_old);
        Self::read_buffer(device, queue, &self.grid_v_old_buffer, &mut grid_v_old);
        Self::read_buffer(device, queue, &self.grid_w_old_buffer, &mut grid_w_old);

        // Upload to G2P and run
        let g2p_count = self.g2p.upload(
            queue,
            positions,
            velocities,
            c_matrices,
            &grid_u,
            &grid_v,
            &grid_w,
            &grid_u_old,
            &grid_v_old,
            &grid_w_old,
            self.cell_size,
            dt,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D G2P Encoder"),
        });
        self.g2p.encode(&mut encoder, g2p_count);
        queue.submit(std::iter::once(encoder.finish()));

        // Download results
        self.g2p.download(device, queue, g2p_count, velocities, c_matrices);
    }

    fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, output: &mut [f32]) {
        // Create a staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Staging"),
            size: (output.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Buffer Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (output.len() * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            output.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();
    }
}
