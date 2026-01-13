//! GPU-accelerated 3D Discrete Element Method (DEM)
//!
//! Handles particle systems from single sediment grains to large boulder meshes.
//! Integrates with GPU FLIP for fluid-particle coupling.
//!
//! Architecture:
//! - Spatial hashing for broad-phase collision detection
//! - Sphere-sphere collision for all particle types
//! - Multi-sphere clump support (2-100 spheres per object)
//! - Two-way coupling with FLIP fluid simulation

use bytemuck::{Pod, Zeroable};
use wgpu::*;
use wgpu::*;
use std::sync::Arc;

/// Maximum number of spheres per clump template
pub const MAX_SPHERES_PER_CLUMP: u32 = 100;

/// Maximum number of hash table entries
pub const HASH_TABLE_SIZE: u32 = 1 << 20; // ~1M entries

/// Empty slot marker for hash table
pub const EMPTY_SLOT: u32 = 0xffffffff;

/// Workgroup size for compute shaders
pub const WORKGROUP_SIZE: u32 = 64;

/// Particle flags
pub const PARTICLE_ACTIVE: u32 = 1u32 << 0u32;
pub const PARTICLE_STATIC: u32 = 1u32 << 1u32;

/// DEM particle data (Structure of Arrays for GPU efficiency)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuDemParticle {
    pub position: [f32; 3],         // World position (m)
    pub velocity: [f32; 3],         // Linear velocity (m/s)
    pub angular_velocity: [f32; 3], // Angular velocity (rad/s)
    pub orientation: [f32; 4],      // Rotation quaternion (xyzw)
    pub radius: f32,                // Bounding radius (m)
    pub mass: f32,                  // Total mass (kg)
    pub template_id: u32,           // Index into clump template buffer
    pub flags: u32,                 // Particle state flags
}

/// Clump template definition (read-only on GPU)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuClumpTemplate {
    pub sphere_count: u32,          // Number of spheres in this clump
    pub mass: f32,                  // Total mass of clump
    pub radius: f32,                // Bounding sphere radius
    pub inertia_inv: [[f32; 3]; 3], // Inverse inertia tensor
    pub _pad: [f32; 1],
}

/// DEM simulation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemParams {
    pub dt: f32,            // Time step (s)
    pub stiffness: f32,     // Contact stiffness (N/m)
    pub damping: f32,       // Contact damping (N·s/m)
    pub friction: f32,      // Friction coefficient
    pub gravity: [f32; 3],  // Gravity vector (m/s²)
    pub cell_size: f32,     // Spatial hash cell size (m)
    pub max_particles: u32, // Maximum particle count
    pub _pad: [f32; 2],
}

/// Spatial hash parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HashParams {
    pub table_size: u32,
    pub cell_size: f32,
    pub max_particles: u32,
    pub _pad: [u32; 1],
}

/// Hash entry for collision linked lists
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HashEntry {
    pub particle_idx: u32,
    pub next_idx: u32,
}

/// Contact information between two spheres
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuContact {
    pub particle_a: u32,  // First particle index
    pub sphere_a: u32,    // Sphere index in particle A
    pub particle_b: u32,  // Second particle index
    pub sphere_b: u32,    // Sphere index in particle B
    pub normal: [f32; 3], // Contact normal (from A to B)
    pub penetration: f32, // Penetration depth (m)
    pub normal_vel: f32,  // Relative velocity along normal
    pub _pad: [f32; 1],
}

/// GPU DEM simulator
pub struct GpuDem3D {
    device: Arc<Device>,
    queue: Arc<Queue>,
    max_particles: u32,
    current_template_count: u32,
    num_active_particles: u32,
    
    // Particle data (SoA layout)
    position_buffer: Buffer,
    velocity_buffer: Buffer,
    angular_velocity_buffer: Buffer,
    orientation_buffer: Buffer,
    radius_buffer: Buffer,
    mass_buffer: Buffer,
    template_id_buffer: Buffer,
    flags_buffer: Buffer,

    // Template storage
    template_buffer: Buffer,
    sphere_offsets_buffer: Buffer, // Local offsets for each sphere
    sphere_radii_buffer: Buffer,   // Local radii for each sphere

    // Spatial hashing
    hash_table_buffer: Buffer,
    hash_entry_buffer: Buffer,
    hash_params_buffer: Buffer,

    // Collision response
    contact_buffer: Buffer,
    force_buffer: Buffer,
    torque_buffer: Buffer,

    // Counters
    particle_counter_buffer: Buffer,
    contact_counter_buffer: Buffer,

    // Parameters
    dem_params_buffer: Buffer,

    // Compute pipelines
    broadphase_pipeline: ComputePipeline,
    collision_pipeline: ComputePipeline,
    integration_pipeline: ComputePipeline,

    // Bind groups
    broadphase_bind_group: BindGroup,
    collision_bind_group: BindGroup,
    integration_bind_group: BindGroup,
}

impl GpuDem3D {    
    /// Create a new GPU DEM simulator
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        max_particles: u32,
        max_templates: u32,
        max_contacts: u32,
    ) -> Self {
        
        // Create particle buffers
        let position_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Position Buffer"),
            size: (max_particles as u64) * 12,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocity_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Velocity Buffer"),
            size: (max_particles as u64) * 12,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let angular_velocity_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Angular Velocity Buffer"),
            size: (max_particles as u64) * 12,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let orientation_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Orientation Buffer"),
            size: (max_particles as u64) * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radius_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Radius Buffer"),
            size: (max_particles as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mass_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Mass Buffer"),
            size: (max_particles as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let template_id_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Template ID Buffer"),
            size: (max_particles as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flags_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Flags Buffer"),
            size: (max_particles as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create template buffers
        let template_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Template Buffer"),
            size: (max_templates as u64) * 64, // sizeof(GpuClumpTemplate) = 64
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_total_spheres = max_templates * MAX_SPHERES_PER_CLUMP;
        let sphere_offsets_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Sphere Offsets Buffer"),
            size: (max_total_spheres as u64) * 12,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sphere_radii_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Sphere Radii Buffer"),
            size: (max_total_spheres as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Spatial hashing buffers
        let hash_table_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Hash Table Buffer"),
            size: (HASH_TABLE_SIZE as u64) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_hash_entries = max_particles * 27; // 27 neighbors per particle
        let hash_entry_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Hash Entry Buffer"),
            size: (max_hash_entries as u64) * 8, // sizeof(HashEntry) = 8
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Contact buffer
        let contact_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Contact Buffer"),
            size: (max_contacts as u64) * 32, // sizeof(GpuContact) = 32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let force_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Force Buffer"),
            size: (max_particles as u64) * 12, // 3D force per particle
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let torque_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Torque Buffer"),
            size: (max_particles as u64) * 12, // 3D torque per particle
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Parameter and counter buffers
        let dem_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Params Buffer"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hash_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Hash Params Buffer"),
            size: 20,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_counter_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Particle Counter Buffer"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let contact_counter_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Contact Counter Buffer"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shaders
        let broadphase_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_broadphase.wgsl"));
        let collision_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_collision.wgsl"));
        let integration_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_integration.wgsl"));

        // Create pipelines
        let broadphase_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("DEM Broadphase Pipeline"),
            layout: None,
            module: &broadphase_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let collision_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("DEM Collision Pipeline"),
            layout: None,
            module: &collision_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let integration_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("DEM Integration Pipeline"),
            layout: None,
            module: &integration_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        // 1. Broadphase bind group
        let broadphase_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Broadphase Bind Group"),
            layout: &broadphase_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: position_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: radius_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: flags_buffer.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: hash_table_buffer.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: hash_entry_buffer.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: particle_counter_buffer.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: hash_params_buffer.as_entire_binding() },
            ],
        });

        // 2. Collision bind group
        let collision_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Collision Bind Group"),
            layout: &collision_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: position_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: velocity_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: radius_buffer.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: mass_buffer.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: template_id_buffer.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: sphere_offsets_buffer.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: template_buffer.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: hash_table_buffer.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: hash_entry_buffer.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: contact_buffer.as_entire_binding() },
                BindGroupEntry { binding: 10, resource: force_buffer.as_entire_binding() },
                BindGroupEntry { binding: 11, resource: dem_params_buffer.as_entire_binding() },
                BindGroupEntry { binding: 12, resource: torque_buffer.as_entire_binding() },
            ],
        });

        // 3. Integration bind group
        let integration_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Integration Bind Group"),
            layout: &integration_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: position_buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: velocity_buffer.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: angular_velocity_buffer.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: mass_buffer.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: radius_buffer.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: flags_buffer.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: force_buffer.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: dem_params_buffer.as_entire_binding() },
                BindGroupEntry { binding: 8, resource: orientation_buffer.as_entire_binding() },
                BindGroupEntry { binding: 9, resource: torque_buffer.as_entire_binding() },
            ],
        });

        let mut dem = Self {
            device,
            queue,
            max_particles,
            current_template_count: 0,
            num_active_particles: 0,
            position_buffer,
            velocity_buffer,
            angular_velocity_buffer,
            orientation_buffer,
            radius_buffer,
            mass_buffer,
            template_id_buffer,
            flags_buffer,
            template_buffer,
            sphere_offsets_buffer,
            sphere_radii_buffer,
            hash_table_buffer,
            hash_entry_buffer,
            hash_params_buffer,
            contact_buffer,
            force_buffer,
            torque_buffer,
            particle_counter_buffer,
            contact_counter_buffer,
            dem_params_buffer,
            broadphase_pipeline,
            collision_pipeline,
            integration_pipeline,
            broadphase_bind_group,
            collision_bind_group,
            integration_bind_group,
        };

        // Initialize hash table
        let empty_hash: [u32; HASH_TABLE_SIZE as usize] = [EMPTY_SLOT; HASH_TABLE_SIZE as usize];
        dem.queue
            .write_buffer(&dem.hash_table_buffer, 0, bytemuck::cast_slice(&empty_hash));

        // Initialize particle counter to 0
        dem.queue
            .write_buffer(&dem.particle_counter_buffer, 0, bytemuck::cast_slice(&[0u32; 1]));
        dem.queue
            .write_buffer(&dem.contact_counter_buffer, 0, bytemuck::cast_slice(&[0u32; 1]));

        // Initialize parameters
        let hash_params = HashParams {
            table_size: HASH_TABLE_SIZE,
            cell_size: 0.1,
            max_particles,
            _pad: [0u32; 1],
        };
        dem.queue.write_buffer(
            &dem.hash_params_buffer,
            0,
            bytemuck::cast_slice(&[hash_params]),
        );

        dem
    }

    /// Get buffer for rendering
    pub fn position_buffer(&self) -> &Buffer {
        &self.position_buffer
    }



    /// Update DEM simulation by one time step
    pub fn update(&mut self, encoder: &mut CommandEncoder, dt: f32) {
        // Update parameters
        let dem_params = DemParams {
            dt,
            stiffness: 50000.0,
            damping: 50.0,
            friction: 0.5,
            gravity: [0.0, -9.81, 0.0],
            cell_size: 0.1,
            max_particles: self.particle_count(),
            _pad: [0.0, 0.0],
        };

        self.queue.write_buffer(
            &self.dem_params_buffer,
            0,
            bytemuck::cast_slice(&[dem_params]),
        );

        let workgroup_count_x = (self.particle_count() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // 1. Broadphase (spatial hashing)
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("DEM Broadphase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.broadphase_pipeline);
            pass.set_bind_group(0, &self.broadphase_bind_group, &[]);

            // Work groups: one thread per particle
            pass.dispatch_workgroups(workgroup_count_x, 1, 1);
        }

        // 2. Collision detection
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("DEM Collision"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.collision_pipeline);
            pass.set_bind_group(0, &self.collision_bind_group, &[]);

            pass.dispatch_workgroups(workgroup_count_x, 1, 1);
        }

        // 3. Integration
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("DEM Integration"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.integration_pipeline);
            pass.set_bind_group(0, &self.integration_bind_group, &[]);

            pass.dispatch_workgroups(workgroup_count_x, 1, 1);
        }
    }

    /// Get current particle count
    pub fn particle_count(&self) -> u32 {
        self.num_active_particles
    }



    /// Add a clump template from CPU template
    pub fn add_template(&mut self, template: sim3d::ClumpTemplate3D) -> u32 {
        let gpu_template = GpuClumpTemplate {
            sphere_count: template.local_offsets.len() as u32,
            mass: template.mass,
            radius: template.bounding_radius,
            inertia_inv: [
                [template.inertia_inv_local.x_axis.x, template.inertia_inv_local.x_axis.y, template.inertia_inv_local.x_axis.z],
                [template.inertia_inv_local.y_axis.x, template.inertia_inv_local.y_axis.y, template.inertia_inv_local.y_axis.z],
                [template.inertia_inv_local.z_axis.x, template.inertia_inv_local.z_axis.y, template.inertia_inv_local.z_axis.z],
            ],
            _pad: [0.0],
        };
        
        // Calculate starting offset for sphere offsets
        let template_offset = self.current_template_count * MAX_SPHERES_PER_CLUMP;
        
        // Upload template data
        self.queue.write_buffer(&self.template_buffer, 
            (template_offset as u64) * 64, 
            bytemuck::cast_slice(&[gpu_template]));
        
        // Upload sphere offsets
        for (i, offset) in template.local_offsets.iter().enumerate() {
            let idx = template_offset + i as u32;
            if idx < template.local_offsets.len() as u32 * MAX_SPHERES_PER_CLUMP {
                self.queue.write_buffer(&self.sphere_offsets_buffer,
                    (idx as u64) * 12,
                    bytemuck::cast_slice(&[offset.x, offset.y, offset.z]));
            }
        }
        
        // For now, assume uniform radii
        for i in 0..template.local_offsets.len() {
            let idx = template_offset + i as u32;
            if idx < template.local_offsets.len() as u32 * MAX_SPHERES_PER_CLUMP {
                self.queue.write_buffer(&self.sphere_radii_buffer,
                    (idx as u64) * 4,
                    bytemuck::cast_slice(&[template.particle_radius]));
            }
        }
        
        let template_id = self.current_template_count;
        self.current_template_count += 1;
        template_id
    }
    
    /// Spawn a clump instance
    pub fn spawn_clump(&mut self, template_id: u32, position: glam::Vec3, velocity: glam::Vec3) -> Option<u32> {
        // Get current particle count
        let current_count = self.particle_count();
        
        if current_count >= self.max_particles {
            return None;
        }
        
        // Initialize particle data
        let particle = GpuDemParticle {
            position: [position.x, position.y, position.z],
            velocity: [velocity.x, velocity.y, velocity.z],
            angular_velocity: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            radius: 0.0, // Will be set from template
            mass: 0.0, // Will be set from template
            template_id,
            flags: PARTICLE_ACTIVE,
        };
        
        // Upload particle data
        let idx = current_count;
        self.queue.write_buffer(&self.position_buffer, (idx as u64) * 12, bytemuck::cast_slice(&[particle.position]));
        self.queue.write_buffer(&self.velocity_buffer, (idx as u64) * 12, bytemuck::cast_slice(&[particle.velocity]));
        self.queue.write_buffer(&self.angular_velocity_buffer, (idx as u64) * 12, bytemuck::cast_slice(&[particle.angular_velocity]));
        self.queue.write_buffer(&self.orientation_buffer, (idx as u64) * 16, bytemuck::cast_slice(&[particle.orientation]));
        self.queue.write_buffer(&self.template_id_buffer, (idx as u64) * 4, bytemuck::cast_slice(&[template_id]));
        self.queue.write_buffer(&self.flags_buffer, (idx as u64) * 4, bytemuck::cast_slice(&[particle.flags]));
        
        // Update particle count
        self.num_active_particles += 1;
        self.queue.write_buffer(&self.particle_counter_buffer, 0, bytemuck::cast_slice(&[self.num_active_particles]));
        
        // Update particle properties from template
        self.update_particle_from_template(idx, template_id);
        
        Some(idx)
    }
    
    /// Update particle properties from template (radius, mass)
    fn update_particle_from_template(&mut self, particle_idx: u32, template_id: u32) {
        // Read template data on CPU and upload relevant properties
        // In a full implementation, templates would be pre-uploaded
        // For now, use placeholder values
        let radius = 0.01; // 1cm default
        let mass = 0.001; // 1kg default
        
        self.queue.write_buffer(&self.radius_buffer, (particle_idx as u64) * 4, bytemuck::cast_slice(&[radius]));
        self.queue.write_buffer(&self.mass_buffer, (particle_idx as u64) * 4, bytemuck::cast_slice(&[mass]));
    }
    
}
