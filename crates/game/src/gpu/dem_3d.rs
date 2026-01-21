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
use std::sync::Arc;
use wgpu::*;

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
    pub position: [f32; 4],         // World position (m) + padding
    pub velocity: [f32; 4],         // Linear velocity (m/s) + padding
    pub angular_velocity: [f32; 4], // Angular velocity (rad/s) + padding
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
    pub sphere_count: u32, // Number of spheres in this clump
    pub mass: f32,         // Total mass of clump
    pub radius: f32,       // Bounding sphere radius
    pub _pad0: f32,
    pub inertia_inv: [[f32; 4]; 3], // Inverse inertia tensor (16-byte aligned columns)
}

/// DEM simulation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemParams {
    pub dt: f32,            // Time step (s)
    pub stiffness: f32,     // Contact stiffness (N/m)
    pub damping: f32,       // Contact damping (N·s/m)
    pub friction: f32,      // Friction coefficient
    pub gravity: [f32; 4],  // Gravity vector (m/s²) + padding
    pub cell_size: f32,     // Spatial hash cell size (m)
    pub max_particles: u32, // Maximum particle count
    pub pad0: f32,
    pub pad1: f32,
}

/// SDF parameters for GPU
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSdfParams {
    pub grid_offset: [f32; 4],
    pub grid_dims: [u32; 4], // width, height, depth, pad
    pub cell_size: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

/// Spatial hash parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HashParams {
    pub table_size: u32,
    pub cell_size: f32,
    pub max_particles: u32,
    pub max_hash_entries: u32,
}

/// Clear hash parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ClearParams {
    pub table_size: u32,
    pub _pad: [u32; 3],
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
    pub normal: [f32; 4], // Contact normal (from A to B) + padding
    pub penetration: f32, // Penetration depth (m)
    pub normal_vel: f32,  // Relative velocity along normal
    pub _pad: [f32; 2],
}

/// DEM-FLIP bridge parameters for coupling shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BridgeParams {
    // FLIP grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
    // Physics parameters
    pub dt: f32,
    pub drag_coefficient: f32,
    pub density_water: f32,
    pub _pad0: f32,
    pub gravity: [f32; 4],
    // DEM particle range
    pub dem_particle_count: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// GPU DEM simulator
pub struct GpuDem3D {
    device: Arc<Device>,
    queue: Arc<Queue>,
    max_particles: u32,
    current_template_count: u32,
    num_active_particles: u32,
    pub stiffness: f32,
    pub damping: f32,

    // Particle data (SoA layout)
    pub(crate) position_buffer: Buffer,
    velocity_buffer: Buffer,
    angular_velocity_buffer: Buffer,
    pub(crate) orientation_buffer: Buffer,
    radius_buffer: Buffer,
    mass_buffer: Buffer,
    pub(crate) template_id_buffer: Buffer,
    flags_buffer: Buffer,

    // Template storage
    pub(crate) template_buffer: Buffer,
    pub(crate) sphere_offsets_buffer: Buffer, // Local offsets for each sphere
    pub(crate) sphere_radii_buffer: Buffer,   // Local radii for each sphere

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
    clear_hash_pipeline: ComputePipeline,
    broadphase_pipeline: ComputePipeline,
    collision_pipeline: ComputePipeline,
    integration_pipeline: ComputePipeline,

    // Bind groups
    clear_hash_bind_group: BindGroup,
    broadphase_bind_group: BindGroup,
    collision_bind_group: BindGroup,
    integration_bind_group: BindGroup,

    // Clear params buffer
    clear_params_buffer: Buffer,

    // SDF Collision
    sdf_buffer: Option<Buffer>,
    sdf_params_buffer: Buffer,
    sdf_collision_pipeline: ComputePipeline,
    sdf_collision_bind_group_layout: BindGroupLayout,

    // DEM-FLIP Bridge (fluid coupling)
    bridge_pipeline: ComputePipeline,
    bridge_bind_group_layout: BindGroupLayout,
    bridge_params_buffer: Buffer,
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
        // Create particle buffers (using 16-byte alignment for 3D vectors)
        let position_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Position Buffer"),
            size: (max_particles as u64) * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocity_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Velocity Buffer"),
            size: (max_particles as u64) * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let angular_velocity_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Angular Velocity Buffer"),
            size: (max_particles as u64) * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let orientation_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Orientation Buffer"),
            size: (max_particles as u64) * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
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
            size: (max_total_spheres as u64) * 16,
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
            size: (max_particles as u64) * 16, // 3D force per particle
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let torque_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Torque Buffer"),
            size: (max_particles as u64) * 16, // 3D torque per particle
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

        // Clear params buffer for hash table clearing
        let clear_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM Clear Params Buffer"),
            size: 16, // ClearParams struct (table_size: u32 + padding)
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shaders
        let clear_hash_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_clear_hash.wgsl"));
        let broadphase_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_broadphase.wgsl"));
        let collision_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_collision.wgsl"));
        let integration_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_integration.wgsl"));

        // Create pipelines
        let clear_hash_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("DEM Clear Hash Pipeline"),
            layout: None,
            module: &clear_hash_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

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

        // 0. Clear hash bind group
        let clear_hash_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Clear Hash Bind Group"),
            layout: &clear_hash_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: hash_table_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: particle_counter_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: clear_params_buffer.as_entire_binding(),
                },
            ],
        });

        // 1. Broadphase bind group
        let broadphase_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Broadphase Bind Group"),
            layout: &broadphase_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: hash_table_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: hash_entry_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: particle_counter_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: hash_params_buffer.as_entire_binding(),
                },
            ],
        });

        // 2. Collision bind group
        let collision_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Collision Bind Group"),
            layout: &collision_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: velocity_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: template_id_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: sphere_offsets_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: template_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: hash_table_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: hash_entry_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: force_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: dem_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: torque_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: sphere_radii_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: orientation_buffer.as_entire_binding(),
                },
            ],
        });

        // 3. Integration bind group
        let integration_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM Integration Bind Group"),
            layout: &integration_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: velocity_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: angular_velocity_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: force_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: dem_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: orientation_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: torque_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: template_id_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: template_buffer.as_entire_binding(),
                },
            ],
        });

        // SDF collision pipeline
        let sdf_collision_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_sdf_collision.wgsl"));

        let sdf_collision_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("DEM SDF Collision Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 7,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 10,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 11,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 12,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let sdf_collision_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("DEM SDF Collision Pipeline Layout"),
                bind_group_layouts: &[&sdf_collision_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sdf_collision_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("DEM SDF Collision Pipeline"),
            layout: Some(&sdf_collision_pipeline_layout),
            module: &sdf_collision_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let sdf_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM SDF Params Buffer"),
            size: std::mem::size_of::<GpuSdfParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // DEM-FLIP Bridge pipeline
        let bridge_shader =
            device.create_shader_module(include_wgsl!("shaders/dem_flip_bridge.wgsl"));

        let bridge_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("DEM-FLIP Bridge Bind Group Layout"),
                entries: &[
                    // binding 0: dem_positions (read)
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: dem_velocities (read_write - for debug, could be read)
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: dem_flags (read)
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: dem_template_ids (read)
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 4: templates (read)
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 5: grid_u (read)
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 6: grid_v (read)
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 7: grid_w (read)
                    BindGroupLayoutEntry {
                        binding: 7,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 8: dem_forces (read_write)
                    BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 9: bridge_params (uniform)
                    BindGroupLayoutEntry {
                        binding: 9,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bridge_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("DEM-FLIP Bridge Pipeline Layout"),
            bind_group_layouts: &[&bridge_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bridge_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("DEM-FLIP Bridge Pipeline"),
            layout: Some(&bridge_pipeline_layout),
            module: &bridge_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let bridge_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("DEM-FLIP Bridge Params Buffer"),
            size: std::mem::size_of::<BridgeParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dem = Self {
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
            clear_hash_pipeline,
            broadphase_pipeline,
            collision_pipeline,
            integration_pipeline,
            clear_hash_bind_group,
            broadphase_bind_group,
            collision_bind_group,
            integration_bind_group,
            clear_params_buffer,
            sdf_buffer: None,
            sdf_params_buffer,
            sdf_collision_pipeline,
            sdf_collision_bind_group_layout,
            bridge_pipeline,
            bridge_bind_group_layout,
            bridge_params_buffer,
            stiffness: 1000.0, // Default safer stiffness
            damping: 10.0,     // Default safer damping
        };

        // Initialize hash table (use Vec to avoid stack overflow - 4MB array)
        let empty_hash = vec![EMPTY_SLOT; HASH_TABLE_SIZE as usize];
        dem.queue
            .write_buffer(&dem.hash_table_buffer, 0, bytemuck::cast_slice(&empty_hash));

        // Initialize particle counter to 0
        dem.queue.write_buffer(
            &dem.particle_counter_buffer,
            0,
            bytemuck::cast_slice(&[0u32; 1]),
        );
        dem.queue.write_buffer(
            &dem.contact_counter_buffer,
            0,
            bytemuck::cast_slice(&[0u32; 1]),
        );

        // Initialize parameters
        let hash_params = HashParams {
            table_size: HASH_TABLE_SIZE,
            cell_size: 0.1,
            max_particles,
            max_hash_entries: max_particles * 27,
        };
        dem.queue.write_buffer(
            &dem.hash_params_buffer,
            0,
            bytemuck::cast_slice(&[hash_params]),
        );

        // Initialize clear params
        let clear_params = ClearParams {
            table_size: HASH_TABLE_SIZE,
            _pad: [0u32; 3],
        };
        dem.queue.write_buffer(
            &dem.clear_params_buffer,
            0,
            bytemuck::cast_slice(&[clear_params]),
        );

        dem
    }

    /// Get buffer for rendering
    pub fn position_buffer(&self) -> &Buffer {
        &self.position_buffer
    }

    /// Step 1: Clear forces, update spatial hash, and apply particle-particle collisions
    pub fn prepare_step(&mut self, encoder: &mut CommandEncoder, dt: f32) {
        // 0. Clear forces and torques (CRITICAL: prevents explosion from accumulation)
        encoder.clear_buffer(&self.force_buffer, 0, None);
        encoder.clear_buffer(&self.torque_buffer, 0, None);

        // Update parameters with current dt
        let dem_params = DemParams {
            dt,
            stiffness: self.stiffness,
            damping: self.damping,
            friction: 0.5,
            gravity: [0.0, -9.81, 0.0, 0.0],
            cell_size: 0.1,
            max_particles: self.max_particles,
            pad0: 0.0,
            pad1: 0.0,
        };

        self.queue
            .write_buffer(&self.dem_params_buffer, 0, bytemuck::bytes_of(&dem_params));

        let workgroup_count_x = (self.max_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // 1. Clear hash table
        {
            let clear_workgroups = (HASH_TABLE_SIZE + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("DEM Clear Hash"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_hash_pipeline);
            pass.set_bind_group(0, &self.clear_hash_bind_group, &[]);
            pass.dispatch_workgroups(clear_workgroups, 1, 1);
        }

        // 2. Broadphase (spatial hashing)
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("DEM Broadphase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.broadphase_pipeline);
            pass.set_bind_group(0, &self.broadphase_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count_x, 1, 1);
        }

        // 3. Collision detection (particle-particle)
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("DEM Collision"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.collision_pipeline);
            pass.set_bind_group(0, &self.collision_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count_x, 1, 1);
        }
    }

    /// Step 2: Final integration (after all collision passes are done)
    pub fn finish_step(&mut self, encoder: &mut CommandEncoder) {
        let workgroup_count_x = (self.max_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // 4. Integration
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

    /// Update DEM simulation by one time step (compact wrapper)
    pub fn update(&mut self, encoder: &mut CommandEncoder, dt: f32) {
        self.prepare_step(encoder, dt);
        self.finish_step(encoder);
    }

    /// Apply collision response against a specific SDF (can be called multiple times for multigrid)
    pub fn apply_sdf_collision_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        sdf_buffer: &Buffer,
        params: &GpuSdfParams,
    ) {
        if self.particle_count() == 0 {
            return;
        }

        // Update SDF parameters
        self.queue
            .write_buffer(&self.sdf_params_buffer, 0, bytemuck::bytes_of(params));

        // Create temporary bind group for this SDF
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM SDF Collision Bind Group"),
            layout: &self.sdf_collision_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.position_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.velocity_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.angular_velocity_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.template_id_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: self.orientation_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: self.template_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: self.sphere_offsets_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.force_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.torque_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: self.dem_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: sdf_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: self.sdf_params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroup_count_x = (self.max_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("DEM SDF Collision"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sdf_collision_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count_x, 1, 1);
    }

    /// Apply DEM-FLIP bridge coupling pass
    ///
    /// This samples FLIP grid velocities at DEM particle positions and applies
    /// drag and buoyancy forces. Must be called between prepare_step() and finish_step()
    /// so that forces are integrated properly.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder
    /// * `grid_u` - FLIP grid U velocity buffer (width+1 x height x depth)
    /// * `grid_v` - FLIP grid V velocity buffer (width x height+1 x depth)
    /// * `grid_w` - FLIP grid W velocity buffer (width x height x depth+1)
    /// * `grid_width` - FLIP grid width
    /// * `grid_height` - FLIP grid height
    /// * `grid_depth` - FLIP grid depth
    /// * `cell_size` - FLIP cell size in meters
    /// * `dt` - Time step
    /// * `drag_coefficient` - Drag coefficient (higher = more drag, typical 1.0-10.0)
    /// * `density_water` - Water density in kg/m³ (typically 1000.0)
    pub fn apply_flip_coupling(
        &self,
        encoder: &mut CommandEncoder,
        grid_u: &Buffer,
        grid_v: &Buffer,
        grid_w: &Buffer,
        grid_width: u32,
        grid_height: u32,
        grid_depth: u32,
        cell_size: f32,
        dt: f32,
        drag_coefficient: f32,
        density_water: f32,
    ) {
        if self.particle_count() == 0 {
            return;
        }

        // Update bridge parameters
        let params = BridgeParams {
            width: grid_width,
            height: grid_height,
            depth: grid_depth,
            cell_size,
            dt,
            drag_coefficient,
            density_water,
            _pad0: 0.0,
            gravity: [0.0, -9.81, 0.0, 0.0],
            dem_particle_count: self.num_active_particles,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        self.queue
            .write_buffer(&self.bridge_params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group for this pass
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("DEM-FLIP Bridge Bind Group"),
            layout: &self.bridge_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.position_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.velocity_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.flags_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.template_id_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: self.template_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: grid_u.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: grid_v.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: grid_w.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: self.force_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.bridge_params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroup_count_x = (self.num_active_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("DEM-FLIP Bridge"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.bridge_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count_x, 1, 1);
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
            _pad0: 0.0,
            inertia_inv: [
                [
                    template.inertia_inv_local.x_axis.x,
                    template.inertia_inv_local.x_axis.y,
                    template.inertia_inv_local.x_axis.z,
                    0.0,
                ],
                [
                    template.inertia_inv_local.y_axis.x,
                    template.inertia_inv_local.y_axis.y,
                    template.inertia_inv_local.y_axis.z,
                    0.0,
                ],
                [
                    template.inertia_inv_local.z_axis.x,
                    template.inertia_inv_local.z_axis.y,
                    template.inertia_inv_local.z_axis.z,
                    0.0,
                ],
            ],
        };

        let template_id = self.current_template_count;

        // Upload template data (each template is 64 bytes)
        self.queue.write_buffer(
            &self.template_buffer,
            (template_id as u64) * 64,
            bytemuck::bytes_of(&gpu_template),
        );

        // Calculate starting offset for sphere offsets/radii
        let sphere_base_offset = template_id * MAX_SPHERES_PER_CLUMP;

        // Upload sphere offsets
        for (i, offset) in template.local_offsets.iter().enumerate() {
            if i >= MAX_SPHERES_PER_CLUMP as usize {
                break;
            }
            let idx = sphere_base_offset + i as u32;
            self.queue.write_buffer(
                &self.sphere_offsets_buffer,
                (idx as u64) * 16,
                bytemuck::cast_slice(&[offset.x, offset.y, offset.z, 0.0]),
            );
        }

        // Upload sphere radii (uniform for now)
        for i in 0..template.local_offsets.len() {
            let idx = sphere_base_offset + i as u32;
            self.queue.write_buffer(
                &self.sphere_radii_buffer,
                (idx as u64) * 4,
                bytemuck::cast_slice(&[template.particle_radius]),
            );
        }

        self.current_template_count += 1;

        template_id
    }

    /// Spawn a clump instance
    pub fn spawn_clump(
        &mut self,
        template_id: u32,
        position: glam::Vec3,
        velocity: glam::Vec3,
    ) -> Option<u32> {
        // Get current particle count
        let current_count = self.particle_count();

        if current_count >= self.max_particles {
            return None;
        }

        // Initialize particle data
        let particle = GpuDemParticle {
            position: [position.x, position.y, position.z, 0.0],
            velocity: [velocity.x, velocity.y, velocity.z, 0.0],
            angular_velocity: [0.0, 0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            radius: 0.0, // Will be set from template
            mass: 0.0,   // Will be set from template
            template_id,
            flags: PARTICLE_ACTIVE,
        };

        // Upload particle data
        let idx = current_count;
        self.queue.write_buffer(
            &self.position_buffer,
            (idx as u64) * 16,
            bytemuck::cast_slice(&[particle.position]),
        );
        self.queue.write_buffer(
            &self.velocity_buffer,
            (idx as u64) * 16,
            bytemuck::cast_slice(&[particle.velocity]),
        );
        self.queue.write_buffer(
            &self.angular_velocity_buffer,
            (idx as u64) * 16,
            bytemuck::cast_slice(&[particle.angular_velocity]),
        );
        self.queue.write_buffer(
            &self.orientation_buffer,
            (idx as u64) * 16,
            bytemuck::cast_slice(&[particle.orientation]),
        );
        self.queue.write_buffer(
            &self.template_id_buffer,
            (idx as u64) * 4,
            bytemuck::cast_slice(&[template_id]),
        );
        self.queue.write_buffer(
            &self.flags_buffer,
            (idx as u64) * 4,
            bytemuck::cast_slice(&[particle.flags]),
        );

        // Update particle count
        self.num_active_particles += 1;
        self.queue.write_buffer(
            &self.particle_counter_buffer,
            0,
            bytemuck::cast_slice(&[self.num_active_particles]),
        );

        // Update particle properties from template
        self.update_particle_from_template(idx, template_id);

        Some(idx)
    }

    /// Update particle properties from template (radius, mass)
    fn update_particle_from_template(&mut self, _particle_idx: u32, _template_id: u32) {
        // No-op: Template data is read directly on GPU now.
        // We leave mass/radius buffers as zero/placeholder for now
        // or potentially unused.
    }

    /// Read back particle data from GPU to CPU
    pub async fn readback(&self, device: &wgpu::Device) -> Vec<GpuDemParticle> {
        let count = self.num_active_particles as usize;
        if count == 0 {
            return Vec::new();
        }

        // Create staging buffers
        let pos_staging = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Position"),
            size: (count * 16) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let vel_staging = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Velocity"),
            size: (count * 16) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let ang_vel_staging = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Angular Velocity"),
            size: (count * 16) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let orient_staging = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Orientation"),
            size: (count * 16) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.position_buffer,
            0,
            &pos_staging,
            0,
            (count * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.velocity_buffer,
            0,
            &vel_staging,
            0,
            (count * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.angular_velocity_buffer,
            0,
            &ang_vel_staging,
            0,
            (count * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.orientation_buffer,
            0,
            &orient_staging,
            0,
            (count * 16) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let (tx, rx) = std::sync::mpsc::channel();

        pos_staging.slice(..).map_async(MapMode::Read, {
            let tx = tx.clone();
            move |v| tx.send(("pos", v)).unwrap()
        });
        vel_staging.slice(..).map_async(MapMode::Read, {
            let tx = tx.clone();
            move |v| tx.send(("vel", v)).unwrap()
        });
        ang_vel_staging.slice(..).map_async(MapMode::Read, {
            let tx = tx.clone();
            move |v| tx.send(("ang_vel", v)).unwrap()
        });
        orient_staging.slice(..).map_async(MapMode::Read, {
            let tx = tx.clone();
            move |v| tx.send(("orient", v)).unwrap()
        });

        device.poll(Maintain::Wait);

        // Wait for all 4 maps to complete
        for _ in 0..4 {
            let (label, result) = rx.recv().unwrap();
            result.expect(&format!("Failed to map {} buffer", label));
        }

        let pos_data = pos_staging.slice(..).get_mapped_range();
        let vel_data = vel_staging.slice(..).get_mapped_range();
        let ang_vel_data = ang_vel_staging.slice(..).get_mapped_range();
        let orient_data = orient_staging.slice(..).get_mapped_range();

        let positions: &[[f32; 4]] = bytemuck::cast_slice(&pos_data);
        let velocities: &[[f32; 4]] = bytemuck::cast_slice(&vel_data);
        let ang_vels: &[[f32; 4]] = bytemuck::cast_slice(&ang_vel_data);
        let orients: &[[f32; 4]] = bytemuck::cast_slice(&orient_data);

        let mut final_particles = Vec::with_capacity(count);
        for i in 0..count {
            final_particles.push(GpuDemParticle {
                position: positions[i],
                velocity: velocities[i],
                angular_velocity: ang_vels[i],
                orientation: orients[i],
                radius: 0.0,    // Not critical for readback sync
                mass: 0.0,      // Not critical for readback sync
                template_id: 0, // Not critical for readback sync
                flags: PARTICLE_ACTIVE,
            });
        }

        final_particles
    }
}
