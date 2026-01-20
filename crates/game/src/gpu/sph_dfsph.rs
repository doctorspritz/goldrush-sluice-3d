//! DFSPH (Divergence-Free SPH) GPU Implementation
//! 
//! This is a high-performance SPH implementation based on:
//! "Divergence-Free SPH for Incompressible and Viscous Fluids" (Bender & Koschier, 2017)
//!
//! Key optimizations over IISPH:
//! - 1-2 solver iterations instead of 8+
//! - Non-atomic cell_offsets reads
//! - Direct buffer indexing after physical sort

use std::sync::Arc;
use super::particle_sort::GpuParticleSort;

/// Frame metrics from DFSPH simulation
#[derive(Debug, Clone, Default)]
pub struct DfsphMetrics {
    pub avg_density: f32,
    pub max_density: f32,
    pub min_density: f32,
    pub density_error_percent: f32,
    pub avg_velocity: f32,
    pub max_velocity: f32,
    pub avg_kinetic_energy: f32,
    pub avg_pressure: f32,
    pub max_pressure: f32,
    pub solver_iterations: u32,
}

/// DFSPH simulation parameters (matches shader struct)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DfsphParams {
    pub num_particles: u32,
    pub h: f32,
    pub h2: f32,
    pub rest_density: f32,

    pub dt: f32,
    pub dt2: f32,
    pub gravity: f32,
    pub omega: f32,
    pub nu: f32,
    pub _pad0: f32,      // Pad block 1 to 16 bytes

    pub cell_size: f32,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,

    pub poly6_coef: f32,
    pub spiky_grad_coef: f32,
    pub particle_mass: f32,
    pub volume: f32,
}

impl DfsphParams {
    pub fn new(
        num_particles: u32,
        h: f32,
        rest_density: f32,
        dt: f32,
        grid_size: (u32, u32, u32),
        cell_size: f32,
    ) -> Self {
        let h2 = h * h;
        let h3 = h2 * h;
        let h9 = h3 * h3 * h3;
        let h6 = h3 * h3;
        
        // Standard SPH kernel coefficients
        let poly6_coef = 315.0 / (64.0 * std::f32::consts::PI * h9);
        let spiky_grad_coef = -45.0 / (std::f32::consts::PI * h6);
        
        // Particle mass and volume for 8 particles per cell target
        let particles_per_cell: f32 = 8.0;
        let _cell_volume = cell_size * cell_size * cell_size;
        let particle_spacing = cell_size / particles_per_cell.cbrt();
        let particle_volume = particle_spacing * particle_spacing * particle_spacing;
        let particle_mass = rest_density * particle_volume;
        let volume = particle_mass / rest_density;

        Self {
            num_particles,
            h,
            h2,
            rest_density,
            dt,
            dt2: dt * dt,
            gravity: 9.81,
            omega: 0.5,
            nu: 0.01,    // Realistic artificial viscosity
            _pad0: 0.0,
            cell_size,
            grid_size_x: grid_size.0,
            grid_size_y: grid_size.1,
            grid_size_z: grid_size.2,
            poly6_coef,
            spiky_grad_coef,
            particle_mass,
            volume,
        }
    }
}

/// GPU DFSPH simulation state
pub struct GpuSphDfsph {
    pub params: DfsphParams,
    num_cells: u32,
    
    // Particle buffers (physically sorted)
    positions: Arc<wgpu::Buffer>,
    velocities: Arc<wgpu::Buffer>,
    positions_pred: Arc<wgpu::Buffer>,
    densities: Arc<wgpu::Buffer>,
    
    // DFSPH-specific buffers
    _alpha: wgpu::Buffer,           // αᵢ factor
    _density_adv: wgpu::Buffer,     // ρ_adv
    pressure_rho2: Arc<wgpu::Buffer>,  // p/ρ²
    _pressure_accel: wgpu::Buffer,  // Pressure acceleration
    
    // Cell offsets (non-atomic read buffer from sorter)
    cell_offsets: wgpu::Buffer,
    
    // Neighbor list buffers
    _neighbor_counts: wgpu::Buffer,
    _neighbor_indices: wgpu::Buffer,
    
    // Sorter
    sorter: GpuParticleSort,
    
    // Parameters buffer
    params_buffer: wgpu::Buffer,
    
    // Bind group
    bind_group: wgpu::BindGroup,
    _bind_group_layout: wgpu::BindGroupLayout,
    
    // Pipelines
    predict_hash_pipeline: wgpu::ComputePipeline,
    compute_density_pipeline: wgpu::ComputePipeline,
    compute_alpha_pipeline: wgpu::ComputePipeline,
    compute_divergence_source_pipeline: wgpu::ComputePipeline,
    update_divergence_pressure_pipeline: wgpu::ComputePipeline,
    compute_density_adv_pipeline: wgpu::ComputePipeline,
    compute_pressure_accel_pipeline: wgpu::ComputePipeline,
    update_pressure_pipeline: wgpu::ComputePipeline,
    integrate_velocity_pipeline: wgpu::ComputePipeline,
    integrate_position_pipeline: wgpu::ComputePipeline,
    boundary_pipeline: wgpu::ComputePipeline,
    
    // Neighbor capping pipelines
    build_neighbors_pipeline: wgpu::ComputePipeline,
    
    // Solver settings
    pub density_iterations: u32,
    pub divergence_iterations: u32,
    pub enable_divergence_solver: bool,
}

impl GpuSphDfsph {
    pub fn new(
        device: &wgpu::Device,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        h: f32,
        max_particles: u32,
        dt: f32,
    ) -> Self {
        let cell_size = h; // Cell size = smoothing radius for optimal neighbor search
        let rest_density = 1000.0; // Water
        
        let params = DfsphParams::new(
            0, // Initially no particles
            h,
            rest_density,
            dt,
            (grid_x, grid_y, grid_z),
            cell_size,
        );
        println!("DFSPH NEW: dt={}, h={}", params.dt, params.h);
        
        let num_cells = grid_x * grid_y * grid_z;
        let particle_capacity = max_particles as usize;
        
        // Create particle buffers
        let positions = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Positions"),
            size: (particle_capacity * 16) as u64, // vec4<f32>
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        }));
        
        let velocities = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Velocities"),
            size: (particle_capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        let positions_pred = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Positions Pred"),
            size: (particle_capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        let densities = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Densities"),
            size: (particle_capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        let alpha = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Alpha"),
            size: (particle_capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let density_adv = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Density Adv"),
            size: (particle_capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let pressure_rho2 = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Pressure/rho2"),
            size: (particle_capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        let pressure_accel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Pressure Accel"),
            size: (particle_capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Cell offsets (non-atomic for reading)
        let cell_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Cell Offsets"),
            size: ((num_cells + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Neighbor list buffers (Fixed 24 neighbors per particle for 60FPS bandwidth targets)
        let neighbor_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Neighbor Counts"),
            size: (particle_capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let neighbor_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Neighbor Indices"),
            size: (particle_capacity * 24 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Parameters buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Params"),
            size: std::mem::size_of::<DfsphParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create sorter
        let sorter = GpuParticleSort::new(
            device,
            grid_x,
            grid_y,
            grid_z,
            particle_capacity,
            positions_pred.clone(), // SORT BASED ON PREDICTED POSITIONS
            velocities.clone(),
            densities.clone(),
            positions.clone(),      // Carry current positions
            pressure_rho2.clone(),  // Carry pressure/rho2
            pressure_rho2.clone(),  // Filler
        );
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DFSPH Bind Group Layout"),
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
                // 1-8: Storage buffers
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
                // 9: Cell offsets (read-only)
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
                // 10: Neighbor Counts
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
                // 11: Neighbor Indices
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
            ],
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DFSPH Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: positions_pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: densities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: alpha.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: density_adv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pressure_rho2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: pressure_accel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: neighbor_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: neighbor_indices.as_entire_binding() },
            ],
        });
        
        // Load shader
        let shader_source = include_str!("shaders/sph_dfsph.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DFSPH Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DFSPH Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create pipelines
        let create_pipeline = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        
        let predict_hash_pipeline = create_pipeline("DFSPH Predict", "predict_and_hash");
        let compute_density_pipeline = create_pipeline("DFSPH Density", "compute_density");
        let compute_alpha_pipeline = create_pipeline("DFSPH Alpha", "compute_alpha");
        let compute_divergence_source_pipeline = create_pipeline("DFSPH Div Source", "compute_divergence_source");
        let update_divergence_pressure_pipeline = create_pipeline("DFSPH Div Update", "update_divergence_pressure");
        let compute_density_adv_pipeline = create_pipeline("DFSPH Density Adv", "compute_density_adv");
        let compute_pressure_accel_pipeline = create_pipeline("DFSPH Pressure Accel", "compute_pressure_accel");
        let update_pressure_pipeline = create_pipeline("DFSPH Update Pressure", "update_pressure");
        let integrate_velocity_pipeline = create_pipeline("DFSPH Integrate Vel", "integrate_velocity");
        let integrate_position_pipeline = create_pipeline("DFSPH Integrate Pos", "integrate_position");
        let boundary_pipeline = create_pipeline("DFSPH Boundary", "boundary_collision");
        let build_neighbors_pipeline = create_pipeline("DFSPH Build Neighbors", "build_neighbor_list");
        
        Self {
            params,
            num_cells,
            positions,
            velocities,
            positions_pred,
            densities,
            _alpha: alpha,
            _density_adv: density_adv,
            pressure_rho2,
            _pressure_accel: pressure_accel,
            cell_offsets,
            _neighbor_counts: neighbor_counts,
            _neighbor_indices: neighbor_indices,
            sorter,
            params_buffer,
            bind_group,
            _bind_group_layout: bind_group_layout,
            predict_hash_pipeline,
            compute_density_pipeline,
            compute_alpha_pipeline,
            compute_divergence_source_pipeline,
            update_divergence_pressure_pipeline,
            compute_density_adv_pipeline,
            compute_pressure_accel_pipeline,
            update_pressure_pipeline,
            integrate_velocity_pipeline,
            integrate_position_pipeline,
            boundary_pipeline,
            build_neighbors_pipeline,
            density_iterations: 3,
            divergence_iterations: 1,
            enable_divergence_solver: true,
        }
    }
    
    /// Upload particles to GPU
    pub fn upload_particles(&mut self, queue: &wgpu::Queue, positions: &[glam::Vec3], velocities: &[glam::Vec3]) {
        let n = positions.len().min(velocities.len());
        self.params.num_particles = n as u32;
        
        // Convert to vec4
        let pos4: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let vel4: Vec<[f32; 4]> = velocities.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        
        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(&pos4));
        queue.write_buffer(&self.velocities, 0, bytemuck::cast_slice(&vel4));
        
        // Initialize densities to rest density
        let densities: Vec<f32> = vec![self.params.rest_density; n];
        queue.write_buffer(&self.densities, 0, bytemuck::cast_slice(&densities));
        
        // Zero out pressure
        let zeros: Vec<f32> = vec![0.0; n];
        queue.write_buffer(&self.pressure_rho2, 0, bytemuck::cast_slice(&zeros));
        
        // Upload params
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }
    
    /// Run one simulation step
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        if self.params.num_particles == 0 { return; }
        
        let workgroups = (self.params.num_particles + 255) / 256;
        if workgroups == 0 { return; }
        
        // Update params
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        
        // 1. Prepare sorter
        self.sorter.prepare(queue, self.params.num_particles, self.params.cell_size);
        
        // 2. Predict positions + hash
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Predict"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.predict_hash_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // 3. Radix sort
        self.sorter.encode(encoder, queue, self.params.num_particles);
        
        // 4. Copy sorted data back to main buffers
        let n_bytes = (self.params.num_particles as u64) * 16; // vec4
        let n_bytes_f32 = (self.params.num_particles as u64) * 4;
        let (out_predicted, out_vel, out_den, out_current, out_p, _out_fill) = self.sorter.sorted_buffers();
        
        encoder.copy_buffer_to_buffer(out_current, 0, &self.positions, 0, n_bytes);
        encoder.copy_buffer_to_buffer(out_vel, 0, &self.velocities, 0, n_bytes);
        encoder.copy_buffer_to_buffer(out_den, 0, &self.densities, 0, n_bytes_f32);
        encoder.copy_buffer_to_buffer(out_predicted, 0, &self.positions_pred, 0, n_bytes);
        encoder.copy_buffer_to_buffer(out_p, 0, &self.pressure_rho2, 0, n_bytes);
        
        // Copy cell offsets (read-only)
        let offset_bytes = ((self.num_cells + 1) as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.sorter.cell_offsets_read_buffer, 0, &self.cell_offsets, 0, offset_bytes);

        // 4.5 Build neighbor list (CACHE)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Build Neighbors"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_neighbors_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // 5. Compute density & alpha
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Density & Alpha"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            
            pass.set_pipeline(&self.compute_density_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
            
            pass.set_pipeline(&self.compute_alpha_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // 7. Divergence-Free Physics phase
        if self.enable_divergence_solver {
            // Early zero out pressure
            encoder.clear_buffer(&self.pressure_rho2, 0, None);

            // Compute divergence source
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("DFSPH: Div Source"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.compute_divergence_source_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            for _ in 0..self.divergence_iterations {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("DFSPH: Div Pressure Accel"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.compute_pressure_accel_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("DFSPH: Div Pressure Update"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.update_divergence_pressure_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }

            // Apply divergence correction immediately to velocities
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("DFSPH: Div Apply"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.integrate_velocity_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // 8. Constant Density Solve Phase
        encoder.clear_buffer(&self.pressure_rho2, 0, None);

        // Compute density advection source term
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Density Adv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_density_adv_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        for _ in 0..self.density_iterations {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("DFSPH: Pressure Accel"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.compute_pressure_accel_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("DFSPH: Update Pressure"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_pressure_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        
        // 9. Final integration & Boundary
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Integrate & Boundary"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            
            pass.set_pipeline(&self.integrate_velocity_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
            
            pass.set_pipeline(&self.integrate_position_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
            
            pass.set_pipeline(&self.boundary_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
    
    /// Get position buffer for rendering
    pub fn position_buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.positions
    }
    
    /// Get particle count
    pub fn particle_count(&self) -> u32 {
        self.params.num_particles
    }

    /// Calibrate rest density based on current particle distribution
    pub fn calibrate_rest_density(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.params.num_particles == 0 { return; }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DFSPH Calibration"),
        });
        
        // 1. Sort to ensure neighbor search works
        self.sorter.prepare(queue, self.params.num_particles, self.params.cell_size);
        // Predict positions = current positions for calibration
        queue.write_buffer(&self.positions_pred, 0, bytemuck::cast_slice(&vec![0.0f32; (self.params.num_particles * 4) as usize])); 
        // Wait, I should just copy positions to positions_pred
        encoder.copy_buffer_to_buffer(&self.positions, 0, &self.positions_pred, 0, (self.params.num_particles as u64) * 16);
        
        self.sorter.encode(&mut encoder, queue, self.params.num_particles);
        
        // Copy back sorted predicted positions (which are current positions)
        let (out_pred, _, _, _, _, _) = self.sorter.sorted_buffers();
        encoder.copy_buffer_to_buffer(out_pred, 0, &self.positions_pred, 0, (self.params.num_particles as u64) * 16);
        
        // Copy cell offsets
        let offset_bytes = ((self.num_cells + 1) as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.sorter.cell_offsets_read_buffer, 0, &self.cell_offsets, 0, offset_bytes);
        
        // 2. Build neighbor list
        {
            let workgroups = (self.params.num_particles + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Neighbor Calibration"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_neighbors_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // 3. Compute density
        {
            let workgroups = (self.params.num_particles + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DFSPH: Density Calibration"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_density_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        queue.submit(std::iter::once(encoder.finish()));
        
        // 3. Readback and find max
        let densities = self.read_densities(device, queue);
        if let Some(max_rho) = densities.iter().copied().reduce(f32::max) {
            self.params.rest_density = max_rho;
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        }
    }

    /// Set simulation timestep
    pub fn set_timestep(&mut self, dt: f32) {
        self.params.dt = dt;
        self.params.dt2 = dt * dt;
    }

    /// Readback positions from GPU
    pub fn read_positions(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<glam::Vec3> {
        let n = self.params.num_particles as usize;
        if n == 0 { return vec![]; }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        encoder.copy_buffer_to_buffer(&self.positions, 0, &staging, 0, (n * 16) as u64);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).expect("Failed to send map_async result"));
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("Failed to receive from channel").expect("Failed to map buffer for reading");
        
        let data = slice.get_mapped_range();
        let pos4: &[[f32; 4]] = bytemuck::cast_slice(&data);
        pos4.iter().map(|p| glam::Vec3::new(p[0], p[1], p[2])).collect()
    }

    /// Readback velocities from GPU
    pub fn read_velocities(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<glam::Vec3> {
        let n = self.params.num_particles as usize;
        if n == 0 { return vec![]; }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        encoder.copy_buffer_to_buffer(&self.velocities, 0, &staging, 0, (n * 16) as u64);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).expect("Failed to send map_async result"));
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("Failed to receive from channel").expect("Failed to map buffer for reading");
        
        let data = slice.get_mapped_range();
        let vel4: &[[f32; 4]] = bytemuck::cast_slice(&data);
        vel4.iter().map(|v| glam::Vec3::new(v[0], v[1], v[2])).collect()
    }

    /// Readback densities from GPU
    pub fn read_densities(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let n = self.params.num_particles as usize;
        if n == 0 { return vec![]; }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        encoder.copy_buffer_to_buffer(&self.densities, 0, &staging, 0, (n * 4) as u64);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).expect("Failed to send map_async result"));
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("Failed to receive from channel").expect("Failed to map buffer for reading");
        
        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }

    /// Readback pressures from GPU
    pub fn read_pressures(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let n = self.params.num_particles as usize;
        if n == 0 { return vec![]; }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        encoder.copy_buffer_to_buffer(&self.pressure_rho2, 0, &staging, 0, (n * 16) as u64);
        queue.submit(Some(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).expect("Failed to send map_async result"));
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("Failed to receive from channel").expect("Failed to map buffer for reading");
        
        let data = slice.get_mapped_range();
        let vec4_data: &[[f32; 4]] = bytemuck::cast_slice(&data);
        vec4_data.iter().map(|v| v[0]).collect()
    }

    /// Compute frame metrics and transfer to CPU
    pub fn compute_metrics(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> DfsphMetrics {
        if self.params.num_particles == 0 {
            return DfsphMetrics::default();
        }

        let densities = self.read_densities(device, queue);
        let velocities = self.read_velocities(device, queue);
        let pressures = self.read_pressures(device, queue);

        let n = densities.len() as f32;
        let mut sum_density_err = 0.0;
        let mut max_density = 0.0f32;
        let mut min_density = f32::MAX;
        let mut sum_vel = 0.0;
        let mut max_vel = 0.0f32;
        let mut sum_ke = 0.0;
        let mut sum_pressure = 0.0;
        let mut max_p = 0.0f32;

        let rho0 = self.params.rest_density;

        for (i, &rho) in densities.iter().enumerate() {
            let err = (rho - rho0).abs() / rho0;
            sum_density_err += err;
            if rho > max_density { max_density = rho; }
            if rho < min_density { min_density = rho; }

            let v = velocities[i];
            let speed = v.length();
            sum_vel += speed;
            if speed > max_vel { max_vel = speed; }
            sum_ke += 0.5 * self.params.particle_mass * speed * speed;
            
            let p = pressures[i] * rho * rho; 
            if p.is_nan() { println!("NAN PRESSURE at particle {}", i); }
            sum_pressure += p;
            if p > max_p { max_p = p; }
        }

        DfsphMetrics {
            avg_density: densities.iter().sum::<f32>() / n,
            max_density,
            min_density,
            density_error_percent: (sum_density_err / n) * 100.0,
            avg_velocity: sum_vel / n,
            max_velocity: max_vel,
            avg_kinetic_energy: sum_ke / n,
            avg_pressure: sum_pressure / n,
            max_pressure: max_p,
            solver_iterations: self.density_iterations,
        }
    }
}
