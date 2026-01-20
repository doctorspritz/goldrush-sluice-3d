//! GPU-accelerated 3D FLIP/APIC simulation.
//!
//! This module provides a complete GPU-based 3D fluid simulation that combines:
//! - P2G (Particle-to-Grid) with atomic scatter
//! - Pressure solve (Red-Black Gauss-Seidel)
//! - G2P (Grid-to-Particle) with FLIP/PIC blend
//!
//! The simulation maintains particle data on CPU but does all heavy computation on GPU.

use super::g2p_3d::{GpuG2p3D, SedimentParams3D};
use super::p2g_3d::GpuP2g3D;
use super::p2g_cell_centric_3d::GpuP2gCellCentric3D;
use super::params::{
    BcParams3D, DensityCorrectParams3D, DensityErrorParams3D, DensityPositionGridParams3D,
    FlowParams3D, GravelObstacleParams3D, GravityParams3D, PorosityDragParams3D,
    SdfCollisionParams3D, SedimentCellTypeParams3D, SedimentFractionParams3D,
    SedimentPressureParams3D, VortConfineParams3D, VorticityParams3D,
};
use super::particle_sort::GpuParticleSort;
use super::pressure_3d::GpuPressure3D;
use super::readback::{ReadbackMode, ReadbackSlot};

use std::sync::Arc;
use wgpu::util::DeviceExt;

// Re-export GravelObstacle for public API
pub use super::params::GravelObstacle;

const GRAVEL_OBSTACLE_MAX: u32 = 2048;

/// GPU-accelerated 3D FLIP simulation
pub struct GpuFlip3D {
    // Grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
    /// Bitmask for open boundaries (particles can exit without clamping):
    /// Bit 0 (1): -X open, Bit 1 (2): +X open
    /// Bit 2 (4): -Y open, Bit 3 (8): +Y open
    /// Bit 4 (16): -Z open, Bit 5 (32): +Z open
    pub open_boundaries: u32,
    /// Vorticity confinement strength (default 0.05, range 0.0-0.25)
    pub vorticity_epsilon: f32,
    /// Target water particles per cell for density projection.
    pub water_rest_particles: f32,
    /// Clamp crowded surface cells (air neighbors) during density projection.
    pub density_surface_clamp: bool,
    pub density_projection_strength: f32,
    /// Separate iterations for density projection pressure solve (default 40).
    pub volume_iterations: u32,
    /// Target sediment particles per cell for porosity fraction.
    pub sediment_rest_particles: f32,
    /// Speed below which friction kicks in (m/s).
    pub sediment_friction_threshold: f32,
    /// How much to damp velocity when slow (0-1 per frame).
    pub sediment_friction_strength: f32,
    /// Downward settling speed for sediment particles.
    pub sediment_settling_velocity: f32,
    /// Vorticity lift applied to sediment when flow swirls.
    pub sediment_vorticity_lift: f32,
    /// Minimum vorticity magnitude to apply lift.
    pub sediment_vorticity_threshold: f32,
    /// Rate at which particle velocity approaches water velocity (1/s).
    /// Higher = more entrainment. Typical: 5.0-20.0. Scaled by 1/density.
    pub sediment_drag_coefficient: f32,
    /// Density threshold for gold particles (treat as gold above this).
    pub gold_density_threshold: f32,
    /// Drag multiplier applied to gold (fine gold entrains more).
    pub gold_drag_multiplier: f32,
    /// Settling velocity used for gold-specific lift (m/s).
    pub gold_settling_velocity: f32,
    /// Upward bias for flaky gold near the surface (m/s^2).
    pub gold_flake_lift: f32,
    /// Porosity-based drag applied to grid velocities.
    pub sediment_porosity_drag: f32,

    // Shared particle buffers (for readback scheduling)
    pub positions_buffer: Arc<wgpu::Buffer>,
    pub velocities_buffer: Arc<wgpu::Buffer>,
    pub(crate) c_col0_buffer: Arc<wgpu::Buffer>,
    pub(crate) c_col1_buffer: Arc<wgpu::Buffer>,
    pub(crate) c_col2_buffer: Arc<wgpu::Buffer>,
    pub densities_buffer: Arc<wgpu::Buffer>,

    // Sub-solvers
    p2g: GpuP2g3D,
    water_p2g: GpuP2g3D,
    g2p: GpuG2p3D,
    pressure: GpuPressure3D,

    // Gravity shader
    gravity_pipeline: wgpu::ComputePipeline,
    gravity_bind_group: wgpu::BindGroup,
    gravity_params_buffer: wgpu::Buffer,

    // Flow acceleration shader (for sluice downstream flow)
    flow_pipeline: wgpu::ComputePipeline,
    flow_bind_group: wgpu::BindGroup,
    flow_params_buffer: wgpu::Buffer,

    // Vorticity confinement
    vorticity_x_buffer: wgpu::Buffer,
    vorticity_y_buffer: wgpu::Buffer,
    vorticity_z_buffer: wgpu::Buffer,
    vorticity_mag_buffer: wgpu::Buffer,
    vorticity_compute_pipeline: wgpu::ComputePipeline,
    vorticity_compute_bind_group: wgpu::BindGroup,
    vorticity_confine_u_pipeline: wgpu::ComputePipeline,
    vorticity_confine_v_pipeline: wgpu::ComputePipeline,
    vorticity_confine_w_pipeline: wgpu::ComputePipeline,
    vorticity_confine_bind_group: wgpu::BindGroup,
    vorticity_params_buffer: wgpu::Buffer,
    vort_confine_params_buffer: wgpu::Buffer,

    // Sediment fraction (cell-centered)
    sediment_fraction_buffer: wgpu::Buffer,
    sediment_fraction_pipeline: wgpu::ComputePipeline,
    sediment_fraction_bind_group: wgpu::BindGroup,
    sediment_fraction_params_buffer: wgpu::Buffer,

    // Sediment pressure (Drucker-Prager)
    sediment_pressure_buffer: wgpu::Buffer,
    sediment_pressure_pipeline: wgpu::ComputePipeline,
    sediment_pressure_bind_group: wgpu::BindGroup,
    sediment_pressure_params_buffer: wgpu::Buffer,

    // Porosity drag (sediment damping on grid)
    porosity_drag_u_pipeline: wgpu::ComputePipeline,
    porosity_drag_v_pipeline: wgpu::ComputePipeline,
    porosity_drag_w_pipeline: wgpu::ComputePipeline,
    porosity_drag_bind_group: wgpu::BindGroup,
    porosity_drag_params_buffer: wgpu::Buffer,

    // Boundary condition enforcement shaders
    bc_u_pipeline: wgpu::ComputePipeline,
    bc_v_pipeline: wgpu::ComputePipeline,
    bc_w_pipeline: wgpu::ComputePipeline,
    bc_bind_group: wgpu::BindGroup,
    bc_params_buffer: wgpu::Buffer,

    // Grid velocity backup for FLIP delta
    grid_u_old_buffer: wgpu::Buffer,
    grid_v_old_buffer: wgpu::Buffer,
    grid_w_old_buffer: wgpu::Buffer,


    // Sediment density projection (granular packing)
    sediment_cell_type_pipeline: wgpu::ComputePipeline,
    sediment_cell_type_bind_group: wgpu::BindGroup,
    sediment_cell_type_params_buffer: wgpu::Buffer,

    // Density projection (volume preservation)
    density_error_buffer: wgpu::Buffer,
    position_delta_x_buffer: wgpu::Buffer,
    position_delta_y_buffer: wgpu::Buffer,
    position_delta_z_buffer: wgpu::Buffer,
    density_error_pipeline: wgpu::ComputePipeline,
    density_error_bind_group: wgpu::BindGroup,
    density_error_params_buffer: wgpu::Buffer,
    density_position_grid_pipeline: wgpu::ComputePipeline,
    density_position_grid_bind_group: wgpu::BindGroup,
    density_position_grid_params_buffer: wgpu::Buffer,
    density_correct_pipeline: wgpu::ComputePipeline,
    density_correct_bind_group: wgpu::BindGroup,
    density_correct_params_buffer: wgpu::Buffer,


    // SDF collision (advection + solid collision)
    sdf_collision_pipeline: wgpu::ComputePipeline,
    sdf_collision_bind_group: wgpu::BindGroup,
    sdf_collision_params_buffer: wgpu::Buffer,
    sdf_buffer: wgpu::Buffer,
    bed_height_buffer: Arc<wgpu::Buffer>,
    sdf_uploaded: bool,

    // Gravel obstacles (dynamic solids)
    gravel_obstacle_pipeline: wgpu::ComputePipeline,
    gravel_obstacle_bind_group: wgpu::BindGroup,
    gravel_obstacle_params_buffer: wgpu::Buffer,
    gravel_obstacle_buffer: wgpu::Buffer,
    gravel_obstacle_count: u32,
    gravel_porosity_pipeline: wgpu::ComputePipeline,
    gravel_porosity_bind_group: wgpu::BindGroup,

    // Particle sorting for cache coherence
    sorter: GpuParticleSort,
    sorted_p2g: GpuP2g3D,
    /// Cell-centric P2G (zero atomics, requires sorted particles)
    cell_centric_p2g: GpuP2gCellCentric3D,
    /// Enable particle sorting before P2G (for benchmarking)
    pub use_sorted_p2g: bool,
    /// Use cell-centric P2G (requires use_sorted_p2g = true)
    pub use_cell_centric_p2g: bool,

    // Maximum particles supported
    max_particles: usize,

    // Double-buffered async readback slots
    readback_slots: Vec<ReadbackSlot>,
    readback_cursor: usize,
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
        // Shared particle buffers for P2G/G2P
        let particle_buffer_size = (max_particles * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D Positions"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let velocities_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D Velocities"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let c_col0_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D C Col0"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let c_col1_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D C Col1"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let c_col2_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D C Col2"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let densities_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D Densities"),
            size: (max_particles * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create P2G solver (owns the grid velocity buffers)
        let p2g = GpuP2g3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            true,  // include_sediment
            false, // use_tiled_scatter (unsorted particles)
            Arc::clone(&positions_buffer),
            Arc::clone(&velocities_buffer),
            Arc::clone(&densities_buffer),
            Arc::clone(&c_col0_buffer),
            Arc::clone(&c_col1_buffer),
            Arc::clone(&c_col2_buffer),
        );

        let water_p2g = GpuP2g3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            false, // include_sediment
            false, // use_tiled_scatter (unsorted particles)
            Arc::clone(&positions_buffer),
            Arc::clone(&velocities_buffer),
            Arc::clone(&densities_buffer),
            Arc::clone(&c_col0_buffer),
            Arc::clone(&c_col1_buffer),
            Arc::clone(&c_col2_buffer),
        );

        // Create particle sorter for cache coherence optimization
        let sorter = GpuParticleSort::new(
            device,
            width,
            height,
            depth,
            max_particles,
            Arc::clone(&positions_buffer),
            Arc::clone(&velocities_buffer),
            Arc::clone(&densities_buffer),
            Arc::clone(&c_col0_buffer),
            Arc::clone(&c_col1_buffer),
            Arc::clone(&c_col2_buffer),
        );

        // Create P2G that reads from sorted particle buffers
        // NOTE: Tiled scatter with shared memory atomics didn't help - contention just moves
        // from global to shared memory. Using non-tiled shader for now.
        let sorted_p2g = GpuP2g3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            true,  // include_sediment
            false, // use_tiled_scatter - disabled, shared memory atomics don't help
            Arc::clone(&sorter.out_positions_buffer),
            Arc::clone(&sorter.out_velocities_buffer),
            Arc::clone(&sorter.out_densities_buffer),
            Arc::clone(&sorter.out_c_col0_buffer),
            Arc::clone(&sorter.out_c_col1_buffer),
            Arc::clone(&sorter.out_c_col2_buffer),
        );

        // Create cell-centric P2G (zero atomics, requires sorted particles + cell_offsets)
        let cell_centric_p2g = GpuP2gCellCentric3D::new(
            device,
            width,
            height,
            depth,
            true, // include_sediment
            Arc::clone(&sorter.out_positions_buffer),
            Arc::clone(&sorter.out_velocities_buffer),
            Arc::clone(&sorter.out_c_col0_buffer),
            Arc::clone(&sorter.out_c_col1_buffer),
            Arc::clone(&sorter.out_c_col2_buffer),
            Arc::clone(&sorter.out_densities_buffer),
            Arc::clone(&sorter.cell_offsets_buffer),
        );

        // Create grid velocity backup buffers for FLIP delta
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;

        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid U Old 3D"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid V Old 3D"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_w_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid W Old 3D"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_count = (width * height * depth) as usize;

        let sediment_pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Pressure Buffer"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity X 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vorticity_y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Y 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vorticity_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Z 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vorticity_mag_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Magnitude 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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

        // Create G2P solver (binds to P2G and old grid buffers)
        let g2p = GpuG2p3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            Arc::clone(&positions_buffer),
            Arc::clone(&velocities_buffer),
            Arc::clone(&c_col0_buffer),
            Arc::clone(&c_col1_buffer),
            Arc::clone(&c_col2_buffer),
            Arc::clone(&densities_buffer),
            &p2g.grid_u_buffer,
            &p2g.grid_v_buffer,
            &p2g.grid_w_buffer,
            &grid_u_old_buffer,
            &grid_v_old_buffer,
            &grid_w_old_buffer,
            &vorticity_mag_buffer,
            &water_p2g.grid_u_buffer,
            &water_p2g.grid_v_buffer,
            &water_p2g.grid_w_buffer,
        );

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
        let gravity_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                ],
            });

        let bed_height_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Height 3D"),
            size: (width as usize * depth as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Use the pressure solver's cell_type buffer for gravity
        let gravity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity 3D Bind Group"),
            layout: &gravity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gravity_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2g.grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bed_height_buffer.as_entire_binding(),
                },
            ],
        });

        let gravity_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

        // Create flow acceleration shader (for sluice downstream flow)
        let flow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flow 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flow_3d.wgsl").into()),
        });

        let flow_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flow Params 3D"),
            size: std::mem::size_of::<FlowParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Flow shader bindings: params, cell_type, grid_u
        let flow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Flow 3D Bind Group Layout"),
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

        let flow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Flow 3D Bind Group"),
            layout: &flow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: flow_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2g.grid_u_buffer.as_entire_binding(),
                },
            ],
        });

        let flow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flow 3D Pipeline Layout"),
            bind_group_layouts: &[&flow_bind_group_layout],
            push_constant_ranges: &[],
        });

        let flow_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flow 3D Pipeline"),
            layout: Some(&flow_pipeline_layout),
            module: &flow_shader,
            entry_point: Some("apply_flow"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Vorticity Confinement ==========
        let vorticity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vorticity_3d.wgsl").into()),
        });

        let vorticity_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Params 3D"),
            size: std::mem::size_of::<VorticityParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Vorticity 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });

        let vorticity_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vorticity 3D Bind Group"),
            layout: &vorticity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vorticity_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p2g.grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2g.grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p2g.grid_w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vorticity_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: vorticity_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: vorticity_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: vorticity_mag_buffer.as_entire_binding(),
                },
            ],
        });

        let vorticity_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Vorticity 3D Pipeline Layout"),
                bind_group_layouts: &[&vorticity_bind_group_layout],
                push_constant_ranges: &[],
            });

        let vorticity_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Vorticity 3D Pipeline"),
                layout: Some(&vorticity_pipeline_layout),
                module: &vorticity_shader,
                entry_point: Some("compute_vorticity"),
                compilation_options: Default::default(),
                cache: None,
            });

        let vorticity_confine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity Confine 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/vorticity_confine_3d.wgsl").into(),
            ),
        });

        let vort_confine_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Confine Params 3D"),
            size: std::mem::size_of::<VortConfineParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_confine_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Vorticity Confine 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });

        let vorticity_confine_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vorticity Confine 3D Bind Group"),
            layout: &vorticity_confine_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vort_confine_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vorticity_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vorticity_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vorticity_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vorticity_mag_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: p2g.grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: p2g.grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: p2g.grid_w_buffer.as_entire_binding(),
                },
            ],
        });

        let vorticity_confine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Vorticity Confine 3D Pipeline Layout"),
                bind_group_layouts: &[&vorticity_confine_bind_group_layout],
                push_constant_ranges: &[],
            });

        let vorticity_confine_u_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Vorticity Confine U 3D Pipeline"),
                layout: Some(&vorticity_confine_pipeline_layout),
                module: &vorticity_confine_shader,
                entry_point: Some("apply_confinement_u"),
                compilation_options: Default::default(),
                cache: None,
            });

        let vorticity_confine_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Vorticity Confine V 3D Pipeline"),
                layout: Some(&vorticity_confine_pipeline_layout),
                module: &vorticity_confine_shader,
                entry_point: Some("apply_confinement_v"),
                compilation_options: Default::default(),
                cache: None,
            });

        let vorticity_confine_w_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Vorticity Confine W 3D Pipeline"),
                layout: Some(&vorticity_confine_pipeline_layout),
                module: &vorticity_confine_shader,
                entry_point: Some("apply_confinement_w"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== Sediment Fraction ==========
        let sediment_fraction_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Fraction 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/sediment_fraction_3d.wgsl").into(),
            ),
        });

        let sediment_fraction_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Fraction Params 3D"),
            size: std::mem::size_of::<SedimentFractionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_fraction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Fraction 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sediment_fraction_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sediment Fraction 3D Bind Group Layout"),
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

        let sediment_fraction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Fraction 3D Bind Group"),
            layout: &sediment_fraction_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sediment_fraction_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p2g.sediment_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sediment_fraction_buffer.as_entire_binding(),
                },
            ],
        });

        let sediment_fraction_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sediment Fraction 3D Pipeline Layout"),
                bind_group_layouts: &[&sediment_fraction_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sediment_fraction_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sediment Fraction 3D Pipeline"),
                layout: Some(&sediment_fraction_pipeline_layout),
                module: &sediment_fraction_shader,
                entry_point: Some("compute_sediment_fraction"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== Sediment Pressure ==========
        let sediment_pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Pressure 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/sediment_pressure_3d.wgsl").into(),
            ),
        });

        let sediment_pressure_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Pressure Params 3D"),
            size: std::mem::size_of::<SedimentPressureParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_pressure_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sediment Pressure 3D Bind Group Layout"),
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

        let sediment_pressure_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Pressure 3D Bind Group"),
            layout: &sediment_pressure_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sediment_pressure_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p2g.sediment_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sediment_pressure_buffer.as_entire_binding(),
                },
            ],
        });

        let sediment_pressure_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sediment Pressure 3D Pipeline Layout"),
                bind_group_layouts: &[&sediment_pressure_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sediment_pressure_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sediment Pressure 3D Pipeline"),
                layout: Some(&sediment_pressure_pipeline_layout),
                module: &sediment_pressure_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== Porosity Drag ==========
        let porosity_drag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Porosity Drag 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/porosity_drag_3d.wgsl").into()),
        });

        let porosity_drag_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Porosity Drag Params 3D"),
            size: std::mem::size_of::<PorosityDragParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let porosity_drag_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Porosity Drag 3D Bind Group Layout"),
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

        let porosity_drag_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Porosity Drag 3D Bind Group"),
            layout: &porosity_drag_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: porosity_drag_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sediment_fraction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2g.grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p2g.grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p2g.grid_w_buffer.as_entire_binding(),
                },
            ],
        });

        let porosity_drag_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Porosity Drag 3D Pipeline Layout"),
                bind_group_layouts: &[&porosity_drag_bind_group_layout],
                push_constant_ranges: &[],
            });

        let porosity_drag_u_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Porosity Drag U 3D Pipeline"),
                layout: Some(&porosity_drag_pipeline_layout),
                module: &porosity_drag_shader,
                entry_point: Some("apply_porosity_u"),
                compilation_options: Default::default(),
                cache: None,
            });

        let porosity_drag_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Porosity Drag V 3D Pipeline"),
                layout: Some(&porosity_drag_pipeline_layout),
                module: &porosity_drag_shader,
                entry_point: Some("apply_porosity_v"),
                compilation_options: Default::default(),
                cache: None,
            });

        let porosity_drag_w_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Porosity Drag W 3D Pipeline"),
                layout: Some(&porosity_drag_pipeline_layout),
                module: &porosity_drag_shader,
                entry_point: Some("apply_porosity_w"),
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

        let bc_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bc_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2g.grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p2g.grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p2g.grid_w_buffer.as_entire_binding(),
                },
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

        // ========== Density Projection (Implicit Density Projection) ==========
        // Creates pipelines for density error computation and position correction



        // ========== Sediment Density Projection (Granular Packing) ==========
        let sediment_cell_type_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Cell Type 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/sediment_cell_type_3d.wgsl").into(),
            ),
        });
        let gravel_obstacle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravel Obstacle 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/gravel_obstacle_3d.wgsl").into(),
            ),
        });
        let gravel_porosity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravel Porosity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/gravel_porosity_3d.wgsl").into(),
            ),
        });

        let sediment_cell_type_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Cell Type Params 3D"),
            size: std::mem::size_of::<SedimentCellTypeParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gravel_obstacle_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gravel Obstacle Params 3D"),
            size: std::mem::size_of::<GravelObstacleParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_cell_type_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sediment Cell Type 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });
        let gravel_obstacle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gravel Obstacle 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let gravel_porosity_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gravel Porosity 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let sediment_cell_type_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Cell Type 3D Bind Group"),
            layout: &sediment_cell_type_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sediment_cell_type_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p2g.sediment_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p2g.particle_count_buffer.as_entire_binding(),
                },
            ],
        });
        let gravel_obstacle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gravel Obstacle Buffer"),
            size: (std::mem::size_of::<GravelObstacle>() * GRAVEL_OBSTACLE_MAX as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gravel_obstacle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravel Obstacle 3D Bind Group"),
            layout: &gravel_obstacle_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gravel_obstacle_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gravel_obstacle_buffer.as_entire_binding(),
                },
            ],
        });
        let gravel_porosity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravel Porosity 3D Bind Group"),
            layout: &gravel_porosity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gravel_obstacle_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sediment_fraction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gravel_obstacle_buffer.as_entire_binding(),
                },
            ],
        });

        let sediment_cell_type_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sediment Cell Type 3D Pipeline Layout"),
                bind_group_layouts: &[&sediment_cell_type_bind_group_layout],
                push_constant_ranges: &[],
            });
        let gravel_obstacle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gravel Obstacle 3D Pipeline Layout"),
                bind_group_layouts: &[&gravel_obstacle_bind_group_layout],
                push_constant_ranges: &[],
            });
        let gravel_porosity_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gravel Porosity 3D Pipeline Layout"),
                bind_group_layouts: &[&gravel_porosity_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sediment_cell_type_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sediment Cell Type 3D Pipeline"),
                layout: Some(&sediment_cell_type_pipeline_layout),
                module: &sediment_cell_type_shader,
                entry_point: Some("build_sediment_cell_type"),
                compilation_options: Default::default(),
                cache: None,
            });
        let gravel_obstacle_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gravel Obstacle 3D Pipeline"),
                layout: Some(&gravel_obstacle_pipeline_layout),
                module: &gravel_obstacle_shader,
                entry_point: Some("build_gravel_obstacles"),
                compilation_options: Default::default(),
                cache: None,
            });
        let gravel_porosity_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gravel Porosity 3D Pipeline"),
                layout: Some(&gravel_porosity_pipeline_layout),
                module: &gravel_porosity_shader,
                entry_point: Some("apply_gravel_porosity"),
                compilation_options: Default::default(),
                cache: None,
            });


        // ========== Density Projection (Volume Preservation) ==========
        let density_error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Error 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_error_3d.wgsl").into()),
        });
        let density_position_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Position Grid 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_position_grid_3d.wgsl").into()),
        });
        let density_correct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Correct 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_correct_3d.wgsl").into()),
        });

        // Density error buffer (per-cell)
        let density_error_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Error Buffer 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Position delta buffers (per-cell, 3D)
        let position_delta_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta X Buffer 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let position_delta_y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Y Buffer 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let position_delta_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Z Buffer 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Density Error pass
        let density_error_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Error Params 3D"),
            size: std::mem::size_of::<DensityErrorParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let density_error_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Error 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let density_error_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Error 3D Bind Group"),
            layout: &density_error_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_error_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.particle_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: density_error_buffer.as_entire_binding() },
            ],
        });

        let density_error_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Error 3D Pipeline Layout"),
            bind_group_layouts: &[&density_error_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_error_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Error 3D Pipeline"),
            layout: Some(&density_error_pipeline_layout),
            module: &density_error_shader,
            entry_point: Some("compute_density_error"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Density Position Grid pass
        let density_position_grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Position Grid Params 3D"),
            size: std::mem::size_of::<DensityPositionGridParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let density_position_grid_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Position Grid 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let density_position_grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Position Grid 3D Bind Group"),
            layout: &density_position_grid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_position_grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.pressure_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: position_delta_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: position_delta_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: position_delta_z_buffer.as_entire_binding() },
            ],
        });

        let density_position_grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Position Grid 3D Pipeline Layout"),
            bind_group_layouts: &[&density_position_grid_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_position_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Position Grid 3D Pipeline"),
            layout: Some(&density_position_grid_pipeline_layout),
            module: &density_position_grid_shader,
            entry_point: Some("compute_position_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Density Correct pass
        let density_correct_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Correct Params 3D"),
            size: std::mem::size_of::<DensityCorrectParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let density_correct_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Correct 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let density_correct_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Correct 3D Bind Group"),
            layout: &density_correct_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_correct_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: position_delta_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: position_delta_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: position_delta_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: velocities_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: densities_buffer.as_entire_binding() },
            ],
        });

        let density_correct_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Correct 3D Pipeline Layout"),
            bind_group_layouts: &[&density_correct_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_correct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Correct 3D Pipeline"),
            layout: Some(&density_correct_pipeline_layout),
            module: &density_correct_shader,
            entry_point: Some("correct_positions"),
            compilation_options: Default::default(),
            cache: None,
        });


        // ========== SDF Collision (Advection + Solid Collision) ==========
        let sdf_collision_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Collision 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf_collision_3d.wgsl").into()),
        });

        let sdf_collision_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Collision Params 3D"),
            size: std::mem::size_of::<SdfCollisionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize SDF buffer with "infinity" (outside) so we default to no collision if not provided
        // We use 1000.0 * cell_size as a safe "far away" distance
        let sdf_size = (width * height * depth) as usize;
        let dummy_sdf = vec![1000.0 * cell_size; sdf_size];
        let sdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SDF Buffer 3D"),
            contents: bytemuck::cast_slice(&dummy_sdf),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sdf_collision_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SDF Collision 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });

        let sdf_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Collision 3D Bind Group"),
            layout: &sdf_collision_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sdf_collision_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: g2p.positions_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: g2p.velocities_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bed_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
            ],
        });

        let sdf_collision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SDF Collision 3D Pipeline Layout"),
                bind_group_layouts: &[&sdf_collision_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sdf_collision_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SDF Collision 3D Pipeline"),
                layout: Some(&sdf_collision_pipeline_layout),
                module: &sdf_collision_shader,
                entry_point: Some("sdf_collision"),
                compilation_options: Default::default(),
                cache: None,
            });

        let readback_slots = vec![
            ReadbackSlot::new(device, max_particles),
            ReadbackSlot::new(device, max_particles),
        ];

        Self {
            width,
            height,
            depth,
            cell_size,
            open_boundaries: 0, // All boundaries closed by default
            vorticity_epsilon: 0.05,
            water_rest_particles: 8.0,
            density_surface_clamp: true,
            density_projection_strength: 1.0,  // Restore default strength
            volume_iterations: 40,
            sediment_rest_particles: 8.0,
            sediment_friction_threshold: 0.1,
            sediment_friction_strength: 0.5,
            sediment_settling_velocity: 0.05,
            sediment_vorticity_lift: 1.5,
            sediment_vorticity_threshold: 3.0,
            sediment_drag_coefficient: 6.0, // Moderate drag - particles entrain in flow
            gold_density_threshold: 10.0,
            gold_drag_multiplier: 1.0,
            gold_settling_velocity: 0.02,
            gold_flake_lift: 0.0,
            sediment_porosity_drag: 3.0,
            positions_buffer,
            velocities_buffer,
            c_col0_buffer,
            c_col1_buffer,
            c_col2_buffer,
            densities_buffer,
            p2g,
            water_p2g,
            g2p,
            pressure,
            gravity_pipeline,
            gravity_bind_group,
            gravity_params_buffer,
            flow_pipeline,
            flow_bind_group,
            flow_params_buffer,
            vorticity_x_buffer,
            vorticity_y_buffer,
            vorticity_z_buffer,
            vorticity_mag_buffer,
            vorticity_compute_pipeline,
            vorticity_compute_bind_group,
            vorticity_confine_u_pipeline,
            vorticity_confine_v_pipeline,
            vorticity_confine_w_pipeline,
            vorticity_confine_bind_group,
            vorticity_params_buffer,
            vort_confine_params_buffer,
            sediment_fraction_buffer,
            sediment_fraction_pipeline,
            sediment_fraction_bind_group,
            sediment_fraction_params_buffer,
            sediment_pressure_buffer,
            sediment_pressure_pipeline,
            sediment_pressure_bind_group,
            sediment_pressure_params_buffer,
            porosity_drag_u_pipeline,
            porosity_drag_v_pipeline,
            porosity_drag_w_pipeline,
            porosity_drag_bind_group,
            porosity_drag_params_buffer,
            bc_u_pipeline,
            bc_v_pipeline,
            bc_w_pipeline,
            bc_bind_group,
            bc_params_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            grid_w_old_buffer,
            sediment_cell_type_pipeline,
            sediment_cell_type_bind_group,
            sediment_cell_type_params_buffer,

            density_error_buffer,
            position_delta_x_buffer,
            position_delta_y_buffer,
            position_delta_z_buffer,
            density_error_pipeline,
            density_error_bind_group,
            density_error_params_buffer,
            density_position_grid_pipeline,
            density_position_grid_bind_group,
            density_position_grid_params_buffer,
            density_correct_pipeline,
            density_correct_bind_group,
            density_correct_params_buffer,

            gravel_obstacle_pipeline,
            gravel_obstacle_bind_group,
            gravel_porosity_pipeline,
            gravel_porosity_bind_group,
            sorter,
            sorted_p2g,
            cell_centric_p2g,
            use_sorted_p2g: true,        // Start with sorting enabled to test
            use_cell_centric_p2g: false, // TEMP: Disabled for baseline comparison
            gravel_obstacle_params_buffer,
            gravel_obstacle_buffer,
            gravel_obstacle_count: 0,
            sdf_collision_pipeline,
            sdf_collision_bind_group,
            sdf_collision_params_buffer,
            sdf_buffer,
            bed_height_buffer,
            sdf_uploaded: false,
            max_particles,
            readback_slots,
            readback_cursor: 0,
        }
    }

    /// Upload SDF data to GPU.
    pub fn upload_sdf(&mut self, queue: &wgpu::Queue, sdf: &[f32]) {
        if self.sdf_uploaded {
            return;
        }

        let expected_sdf_len = (self.width * self.height * self.depth) as usize;
        assert_eq!(
            sdf.len(),
            expected_sdf_len,
            "SDF size mismatch: got {}, expected {}",
            sdf.len(),
            expected_sdf_len
        );

        queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));
        self.sdf_uploaded = true;
    }

    /// Force upload SDF data to GPU (for dynamic obstacles).
    pub fn upload_sdf_force(&mut self, queue: &wgpu::Queue, sdf: &[f32]) {
        let expected_sdf_len = (self.width * self.height * self.depth) as usize;
        assert_eq!(
            sdf.len(),
            expected_sdf_len,
            "SDF size mismatch: got {}, expected {}",
            sdf.len(),
            expected_sdf_len
        );
        queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));
        self.sdf_uploaded = true;
    }

    pub fn upload_gravel_obstacles(&mut self, queue: &wgpu::Queue, obstacles: &[GravelObstacle]) {
        let count = obstacles.len().min(GRAVEL_OBSTACLE_MAX as usize);
        if count == 0 {
            self.gravel_obstacle_count = 0;
            return;
        }
        queue.write_buffer(
            &self.gravel_obstacle_buffer,
            0,
            bytemuck::cast_slice(&obstacles[..count]),
        );
        self.gravel_obstacle_count = count as u32;
    }

    fn apply_gravel_obstacles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gravel_obstacle_count == 0 {
            return;
        }

        let params = GravelObstacleParams3D::new(
            self.width,
            self.height,
            self.depth,
            self.gravel_obstacle_count,
            self.cell_size,
        );
        queue.write_buffer(
            &self.gravel_obstacle_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gravel Obstacle Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravel Obstacle Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravel_obstacle_pipeline);
            pass.set_bind_group(0, &self.gravel_obstacle_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run one simulation step (sync readback).
    ///
    /// This performs the full FLIP pipeline:
    /// 1. P2G: Transfer particle data to grid
    /// 2. Enforce boundary conditions (before storing old velocities!)
    /// 3. Save grid velocity (for FLIP delta)
    /// 4. Apply gravity (vertical)
    /// 5. Apply flow acceleration (horizontal, for sluice flow)
    /// 6. Vorticity confinement (adds rotational energy)
    /// 7. Pressure solve (includes divergence, iterations, gradient)
    /// 8. G2P: Transfer grid data back to particles
    /// 9. Optional: GPU advection + SDF collision (when `sdf` is provided)
    ///
    /// # Arguments
    /// * `flow_accel` - Downstream flow acceleration (m/s²). Set to 0.0 for closed box sims.
    ///                  For a sluice, use ~2-5 m/s² to drive water downstream.
    /// Pure GPU step (no CPU readback/upload).
    /// Assumes all data (positions, velocities, etc.) is already in GPU buffers.
    pub fn encode_step(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particle_count: u32,
        dt: f32,
    ) {
        if particle_count == 0 {
            return;
        }

        // 1. P2G
        self.p2g.prepare(queue, particle_count, self.cell_size);
        self.water_p2g
            .prepare(queue, particle_count, self.cell_size);
        self.p2g.encode(encoder, particle_count);
        self.water_p2g.encode(encoder, particle_count);

        // 2. Pressure (Simplified for now - might need more steps for stability)
        self.pressure.encode(encoder, 40); // 40 iterations

        // 3. G2P
        let sediment_params = SedimentParams3D::default();
        self.g2p
            .upload_params(queue, particle_count, self.cell_size, dt, 0.95, sediment_params);
        self.g2p.encode(encoder, particle_count);

        // 4. SDF Collision
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Collision Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sdf_collision_pipeline);
        pass.set_bind_group(0, &self.sdf_collision_bind_group, &[]);
        let workgroups = particle_count.div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        let _ = self.step_internal(
            device,
            queue,
            positions,
            velocities,
            c_matrices,
            densities,
            cell_types,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
            ReadbackMode::Sync,
        );
    }

    /// Run one simulation step without any readback (upload + GPU passes only).
    pub fn step_no_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        let _ = self.step_internal(
            device,
            queue,
            positions,
            velocities,
            c_matrices,
            densities,
            cell_types,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
            ReadbackMode::None,
        );
    }

    /// Run one simulation step and schedule an async readback.
    ///
    /// Call `try_readback` on subsequent frames to pull results without stalling.
    /// Returns false if all readback slots are still in flight.
    pub fn step_async(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) -> bool {
        self.step_internal(
            device,
            queue,
            positions,
            velocities,
            c_matrices,
            densities,
            cell_types,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
            ReadbackMode::Async,
        )
    }

    /// Try to fetch the latest async readback results without stalling.
    ///
    /// Returns Some(count) when a slot completes, otherwise None.
    pub fn try_readback(
        &mut self,
        device: &wgpu::Device,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
    ) -> Option<usize> {
        device.poll(wgpu::Maintain::Poll);
        for slot in &mut self.readback_slots {
            if let Some(count) = slot.try_read(positions, velocities, c_matrices) {
                return Some(count);
            }
        }
        None
    }

    /// Schedule a readback of the current GPU particle buffers without uploading.
    pub fn request_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: usize,
    ) -> bool {
        let count = particle_count.min(self.max_particles);
        self.schedule_readback(device, queue, count)
    }

    pub fn positions_buffer(&self) -> Arc<wgpu::Buffer> {
        self.positions_buffer.clone()
    }

    pub fn velocities_buffer(&self) -> Arc<wgpu::Buffer> {
        self.velocities_buffer.clone()
    }

    pub fn densities_buffer(&self) -> Arc<wgpu::Buffer> {
        self.densities_buffer.clone()
    }

    pub fn sdf_buffer(&self) -> &wgpu::Buffer {
        &self.sdf_buffer
    }

    pub fn bed_height_buffer(&self) -> Arc<wgpu::Buffer> {
        Arc::clone(&self.bed_height_buffer)
    }

    pub fn p2g_particle_count_buffer(&self) -> &wgpu::Buffer {
        &self.p2g.particle_count_buffer
    }

    /// Get the U velocity grid buffer (MAC staggered: width+1 x height x depth)
    pub fn grid_u_buffer(&self) -> &wgpu::Buffer {
        &self.p2g.grid_u_buffer
    }

    /// Get the V velocity grid buffer (MAC staggered: width x height+1 x depth)
    pub fn grid_v_buffer(&self) -> &wgpu::Buffer {
        &self.p2g.grid_v_buffer
    }

    /// Get the W velocity grid buffer (MAC staggered: width x height x depth+1)
    pub fn grid_w_buffer(&self) -> &wgpu::Buffer {
        &self.p2g.grid_w_buffer
    }

    fn upload_bc_and_cell_types(&self, queue: &wgpu::Queue, cell_types: &[u32]) {
        // Upload cell types FIRST (needed for BC enforcement)
        self.pressure
            .upload_cell_types(queue, cell_types, self.cell_size, self.open_boundaries);

        // Upload BC params
        let bc_params = BcParams3D::new(self.width, self.height, self.depth);
        queue.write_buffer(&self.bc_params_buffer, 0, bytemuck::bytes_of(&bc_params));
    }

    fn schedule_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: usize,
    ) -> bool {
        if self.readback_slots.is_empty() {
            return false;
        }

        let slots = self.readback_slots.len();
        for _ in 0..slots {
            let index = self.readback_cursor % slots;
            self.readback_cursor = (self.readback_cursor + 1) % slots;
            let slot = &mut self.readback_slots[index];
            if slot.schedule(
                device,
                queue,
                &self.positions_buffer,
                &self.velocities_buffer,
                &self.c_col0_buffer,
                &self.c_col1_buffer,
                &self.c_col2_buffer,
                particle_count,
            ) {
                return true;
            }
        }

        false
    }

    pub fn step_in_place(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        if particle_count == 0 {
            return;
        }

        self.upload_bc_and_cell_types(queue, cell_types);
        self.apply_gravel_obstacles(device, queue);
        self.p2g.prepare(queue, particle_count, self.cell_size);

        let _ = self.run_gpu_passes(
            device,
            queue,
            particle_count,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );
    }

    fn run_gpu_passes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) -> u32 {
        let count = particle_count;

        self.p2g.prepare(queue, count, self.cell_size);
        self.water_p2g.prepare(queue, count, self.cell_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Step Encoder"),
        });

        // Run P2G scatter and divide (optionally with particle sorting)
        let (u_size, v_size, w_size) = self.p2g.grid_sizes();
        let cell_count = (self.width * self.height * self.depth) as usize;
        if self.use_sorted_p2g {
            // Sort particles by cell index for cache coherence
            self.sorter.prepare(queue, count, self.cell_size);
            self.sorter.encode(&mut encoder, queue, count);

            if self.use_cell_centric_p2g {
                // Use cell-centric P2G (zero atomics, one thread per grid node)
                self.cell_centric_p2g.prepare(queue, count, self.cell_size);
                self.cell_centric_p2g.encode(&mut encoder, count);
                // Copy results from cell_centric_p2g to p2g buffers (rest of pipeline uses p2g buffers)
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.grid_u_buffer,
                    0,
                    &self.p2g.grid_u_buffer,
                    0,
                    (u_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.grid_v_buffer,
                    0,
                    &self.p2g.grid_v_buffer,
                    0,
                    (v_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.grid_w_buffer,
                    0,
                    &self.p2g.grid_w_buffer,
                    0,
                    (w_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.particle_count_buffer,
                    0,
                    &self.p2g.particle_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.sediment_count_buffer,
                    0,
                    &self.p2g.sediment_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
            } else {
                // Use sorted P2G that reads from sorted particle buffers
                self.sorted_p2g.prepare(queue, count, self.cell_size);
                self.sorted_p2g.encode(&mut encoder, count);
                // Copy results from sorted_p2g to p2g buffers (rest of pipeline uses p2g buffers)
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.grid_u_buffer,
                    0,
                    &self.p2g.grid_u_buffer,
                    0,
                    (u_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.grid_v_buffer,
                    0,
                    &self.p2g.grid_v_buffer,
                    0,
                    (v_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.grid_w_buffer,
                    0,
                    &self.p2g.grid_w_buffer,
                    0,
                    (w_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.particle_count_buffer,
                    0,
                    &self.p2g.particle_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.sediment_count_buffer,
                    0,
                    &self.p2g.sediment_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
            }
        } else {
            self.p2g.encode(&mut encoder, count);
        }
        self.water_p2g.encode(&mut encoder, count);

        queue.submit(std::iter::once(encoder.finish()));

        // ========== Density Projection (Volume Preservation) ==========
        // After P2G, before G2P: correct particle positions based on density error
        if self.water_rest_particles > 0.0 {
            // 1. Compute density error from particle count
            let density_error_params = DensityErrorParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                rest_density: self.water_rest_particles,
                dt,
                surface_clamp: if self.density_surface_clamp { 1 } else { 0 },
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_error_params_buffer,
                0,
                bytemuck::bytes_of(&density_error_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Density Error 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Error 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_error_pipeline);
                pass.set_bind_group(0, &self.density_error_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // 2. Copy density error to divergence buffer (pressure solver reads from divergence)
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Density Error Copy Encoder"),
            });
            encoder.copy_buffer_to_buffer(
                &self.density_error_buffer,
                0,
                &self.pressure.divergence_buffer,
                0,
                (self.width * self.height * self.depth * std::mem::size_of::<f32>() as u32) as u64,
            );
            queue.submit(std::iter::once(encoder.finish()));

            // Clear pressure before solving
            self.pressure.clear_pressure(queue);

            // 3. Solve pressure with density error as RHS
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Density Pressure 3D Encoder"),
            });
            self.pressure.encode_iterations_only(&mut encoder, self.volume_iterations);
            queue.submit(std::iter::once(encoder.finish()));

            // 4. Convert pressure to position deltas on grid
            let density_position_grid_params = DensityPositionGridParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                dt,
                strength: self.density_projection_strength,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_position_grid_params_buffer,
                0,
                bytemuck::bytes_of(&density_position_grid_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Density Position Grid 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Position Grid 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_position_grid_pipeline);
                pass.set_bind_group(0, &self.density_position_grid_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // 4. Apply position corrections to particles
            let density_correct_params = DensityCorrectParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                particle_count: count,
                cell_size: self.cell_size,
                damping: 0.95, // Damp velocities during position correction to bleed off jitter energy
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_correct_params_buffer,
                0,
                bytemuck::bytes_of(&density_correct_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Density Correct 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Correct 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_correct_pipeline);
                pass.set_bind_group(0, &self.density_correct_bind_group, &[]);
                let workgroups = count.div_ceil(256);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        let sediment_fraction_params = SedimentFractionParams3D::new(
            self.width,
            self.height,
            self.depth,
            self.sediment_rest_particles,
        );
        queue.write_buffer(
            &self.sediment_fraction_params_buffer,
            0,
            bytemuck::bytes_of(&sediment_fraction_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sediment Fraction 3D Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sediment Fraction 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sediment_fraction_pipeline);
            pass.set_bind_group(0, &self.sediment_fraction_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        if self.gravel_obstacle_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravel Porosity 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravel_porosity_pipeline);
            pass.set_bind_group(0, &self.gravel_porosity_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        let sediment_pressure_params = SedimentPressureParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            _pad0: 0,
            cell_size: self.cell_size,
            particle_mass: 1.0,
            gravity: gravity.abs(),
            buoyancy_factor: 0.62,
        };
        queue.write_buffer(
            &self.sediment_pressure_params_buffer,
            0,
            bytemuck::bytes_of(&sediment_pressure_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sediment Pressure 3D Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sediment Pressure 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sediment_pressure_pipeline);
            pass.set_bind_group(0, &self.sediment_pressure_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, 1, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));





        // 3. Save grid velocity for FLIP delta (now with proper BCs!)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Grid Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.p2g.grid_u_buffer,
            0,
            &self.grid_u_old_buffer,
            0,
            (u_size * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.grid_v_buffer,
            0,
            &self.grid_v_old_buffer,
            0,
            (v_size * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.grid_w_buffer,
            0,
            &self.grid_w_old_buffer,
            0,
            (w_size * 4) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // 4. Apply gravity

        let gravity_params = GravityParams3D::new(
            self.width,
            self.height,
            self.depth,
            gravity * dt, // gravity_dt
            self.cell_size,
        );
        queue.write_buffer(
            &self.gravity_params_buffer,
            0,
            bytemuck::bytes_of(&gravity_params),
        );

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
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = (self.height + 1).div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 5. Apply flow acceleration (for sluice downstream flow)
        // This MUST happen before pressure solve so the solver can account for the flow!
        if flow_accel.abs() > 0.0001 {
            let flow_params = FlowParams3D::new(
                self.width,
                self.height,
                self.depth,
                flow_accel * dt, // flow_accel_dt
            );
            queue.write_buffer(
                &self.flow_params_buffer,
                0,
                bytemuck::bytes_of(&flow_params),
            );

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flow 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.flow_pipeline);
            pass.set_bind_group(0, &self.flow_bind_group, &[]);
            // U grid: (width+1) x height x depth
            let workgroups_x = (self.width + 1).div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 6. Vorticity confinement (adds rotational energy)
        if self.vorticity_epsilon > 0.0 {
            let vort_params = VorticityParams3D::new(
                self.width,
                self.height,
                self.depth,
                self.cell_size,
            );
            queue.write_buffer(
                &self.vorticity_params_buffer,
                0,
                bytemuck::bytes_of(&vort_params),
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_compute_pipeline);
                pass.set_bind_group(0, &self.vorticity_compute_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            let confine_params = VortConfineParams3D::new(
                self.width,
                self.height,
                self.depth,
                self.vorticity_epsilon * self.cell_size * dt, // epsilon_h_dt
            );
            queue.write_buffer(
                &self.vort_confine_params_buffer,
                0,
                bytemuck::bytes_of(&confine_params),
            );

            // Apply confinement to U faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine U 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_u_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = (self.width + 1).div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Apply confinement to V faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine V 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_v_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = (self.height + 1).div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Apply confinement to W faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine W 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_w_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = (self.depth + 1).div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        }

        // Submit gravity/flow/vorticity passes
        queue.submit(std::iter::once(encoder.finish()));

        // 7. Enforce Boundary Conditions
        // CRITICAL FIX: Must run AFTER Gravity and BEFORE Pressure Solve.
        // Gravity adds -9.8 to V at the floor. If we don't clamp it to 0 here,
        // the pressure solver sees outflow at the floor and fails to build up pressure,
        // causing the fluid to leak/compress through the bottom.
        //
        // IMPORTANT: Separate encoder + submit ensures BC writes are committed
        // before pressure solver reads velocity buffers.
        let mut bc_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D BC Encoder"),
        });

        // Enforce BC on U
        {
            let mut pass = bc_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC U 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_u_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 1).div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on V
        {
            let mut pass = bc_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC V 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_v_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = (self.height + 1).div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on W
        {
            let mut pass = bc_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC W 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_w_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = (self.depth + 1).div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // CRITICAL: Submit BC before pressure solve to ensure velocity writes are visible
        queue.submit(std::iter::once(bc_encoder.finish()));

        // 8. Pressure solve (divergence → iterations → gradient)
        let mut pressure_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Pressure Encoder"),
        });
        self.pressure.encode(&mut pressure_encoder, pressure_iterations);

        queue.submit(std::iter::once(pressure_encoder.finish()));

        if self.sediment_porosity_drag > 0.0 {
            let drag_params = PorosityDragParams3D::new(
                self.width,
                self.height,
                self.depth,
                self.sediment_porosity_drag * dt, // drag_dt
            );
            queue.write_buffer(
                &self.porosity_drag_params_buffer,
                0,
                bytemuck::bytes_of(&drag_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Porosity Drag 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag U 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_u_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = (self.width + 1).div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag V 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_v_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = (self.height + 1).div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag W 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_w_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = (self.depth + 1).div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // 8. Run G2P using grid buffers already on GPU
        let sediment_params = SedimentParams3D {
            settling_velocity: self.sediment_settling_velocity,
            friction_threshold: self.sediment_friction_threshold,
            friction_strength: self.sediment_friction_strength,
            vorticity_lift: self.sediment_vorticity_lift,
            vorticity_threshold: self.sediment_vorticity_threshold,
            drag_coefficient: self.sediment_drag_coefficient,
            gold_density_threshold: self.gold_density_threshold,
            gold_drag_multiplier: self.gold_drag_multiplier,
            gold_settling_velocity: self.gold_settling_velocity,
            gold_flake_lift: self.gold_flake_lift,
            _pad: [0.0; 2],
        };
        let g2p_count = self
            .g2p
            .upload_params(queue, count, self.cell_size, dt, 0.95, sediment_params);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D G2P Encoder"),
        });
        self.g2p.encode(&mut encoder, g2p_count);
        queue.submit(std::iter::once(encoder.finish()));

        // ========== Sediment Density Projection (Granular Packing) ==========
        if self.sediment_rest_particles > 0.0 {
            let sediment_cell_params = SedimentCellTypeParams3D::new(
                self.width,
                self.height,
                self.depth,
            );
            queue.write_buffer(
                &self.sediment_cell_type_params_buffer,
                0,
                bytemuck::bytes_of(&sediment_cell_params),
            );

            // DISABLED for friction-only model: jamming causes infinite compression
            // because sediment marked as SOLID doesn't participate in pressure solve.
            // With friction-only, sediment flows like water but settles + has friction.
            /*
            let jamming_iterations = 5;
            for _ in 0..jamming_iterations {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Sediment Cell Type 3D Encoder"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sediment Cell Type 3D Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.sediment_cell_type_pipeline);
                    pass.set_bind_group(0, &self.sediment_cell_type_bind_group, &[]);
                    let workgroups_x = (self.width + 7) / 8;
                    let workgroups_y = (self.height + 7) / 8;
                    let workgroups_z = (self.depth + 3) / 4;
                    pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
                }
                queue.submit(std::iter::once(encoder.finish()));
            }
            */

            // DISABLED: Using voxel-based jamming instead of density projection
            /*
            let sediment_density_error_params = DensityErrorParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                rest_density: self.sediment_rest_particles,
                dt,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_error_params_buffer,
                0,
                bytemuck::bytes_of(&sediment_density_error_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Density Error 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Density Error 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sediment_density_error_pipeline);
                pass.set_bind_group(0, &self.sediment_density_error_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

            self.pressure.clear_pressure(queue);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Density Pressure 3D Encoder"),
            });
            let sediment_iterations = 15;
            self.pressure.encode_iterations_only(&mut encoder, sediment_iterations);
            queue.submit(std::iter::once(encoder.finish()));

            let grid_params = DensityPositionGridParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                dt,
                strength: self.density_projection_strength,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_position_grid_params_buffer,
                0,
                bytemuck::bytes_of(&grid_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Position Grid 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Position Grid 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_position_grid_pipeline);
                pass.set_bind_group(0, &self.density_position_grid_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

            let density_correct_params = DensityCorrectionParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                particle_count: particle_count,
                cell_size: self.cell_size,
                dt,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(
                &self.density_correct_params_buffer,
                0,
                bytemuck::bytes_of(&density_correct_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Density Correct 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Density Correct 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sediment_density_correct_pipeline);
                pass.set_bind_group(0, &self.density_correct_bind_group, &[]);
                let workgroups = (particle_count as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            */
        }

        if let Some(bed_height) = bed_height {
            let expected_len = (self.width * self.depth) as usize;
            assert_eq!(
                bed_height.len(),
                expected_len,
                "Bed height size mismatch: got {}, expected {}",
                bed_height.len(),
                expected_len
            );
            queue.write_buffer(&self.bed_height_buffer, 0, bytemuck::cast_slice(bed_height));
        }

        if let Some(sdf) = sdf {
            if !self.sdf_uploaded {
                self.upload_sdf(queue, sdf);
            }
        }

        let sdf_params = SdfCollisionParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            cell_size: self.cell_size,
            dt,
            open_boundaries: self.open_boundaries,
            _pad0: 0,
        };
        queue.write_buffer(
            &self.sdf_collision_params_buffer,
            0,
            bytemuck::bytes_of(&sdf_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D SDF Collision Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF Collision 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sdf_collision_pipeline);
            pass.set_bind_group(0, &self.sdf_collision_bind_group, &[]);
            let workgroups = g2p_count.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        g2p_count
    }

    fn step_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
        readback: ReadbackMode,
    ) -> bool {
        let particle_count = positions.len().min(self.max_particles);
        if particle_count == 0 {
            return false;
        }

        self.upload_bc_and_cell_types(queue, cell_types);
        self.apply_gravel_obstacles(device, queue);

        // 1. Upload particles and run P2G
        let count = self.p2g.upload_particles(
            queue,
            positions,
            velocities,
            densities,
            c_matrices,
            self.cell_size,
        );

        let g2p_count = self.run_gpu_passes(
            device,
            queue,
            count,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );

        match readback {
            ReadbackMode::None => true,
            ReadbackMode::Sync => {
                // Download results (velocities + C matrix + positions).
                self.g2p
                    .download(device, queue, g2p_count, velocities, c_matrices);
                let mut gpu_positions = vec![[0.0f32; 4]; particle_count];
                Self::read_buffer_vec4(
                    device,
                    queue,
                    &self.g2p.positions_buffer,
                    &mut gpu_positions,
                    particle_count,
                );
                for (i, pos) in gpu_positions.iter().enumerate() {
                    positions[i] = glam::Vec3::new(pos[0], pos[1], pos[2]);
                }
                true
            }
            ReadbackMode::Async => self.schedule_readback(device, queue, particle_count),
        }
    }

    fn read_buffer_vec4(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        output: &mut [[f32; 4]],
        count: usize,
    ) {
        let byte_size = count * std::mem::size_of::<[f32; 4]>();

        // Create a staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Vec4 Staging"),
            size: byte_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Vec4 Buffer Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size as u64);
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
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            output[..count].copy_from_slice(&slice[..count]);
        }
        staging.unmap();
    }

    /// Print density projection diagnostics (volume preservation metrics).
    pub fn print_density_projection_diagnostics(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        rest_density: f32,
    ) {
        let cell_count = (self.width * self.height * self.depth) as usize;
        if cell_count == 0 {
            return;
        }

        let byte_size = (cell_count * 4) as u64;

        let cell_type_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type Staging (Density Diagnostics)"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let particle_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Count Staging (Density Diagnostics)"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let divergence_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Error Staging (Density Diagnostics)"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pressure_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Staging (Density Diagnostics)"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Density Diagnostics Readback"),
        });
        encoder.copy_buffer_to_buffer(
            &self.pressure.cell_type_buffer,
            0,
            &cell_type_staging,
            0,
            byte_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.particle_count_buffer,
            0,
            &particle_count_staging,
            0,
            byte_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.density_error_buffer,
            0,
            &divergence_staging,
            0,
            byte_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.pressure.pressure_buffer,
            0,
            &pressure_staging,
            0,
            byte_size,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let ct_slice = cell_type_staging.slice(..);
        let part_slice = particle_count_staging.slice(..);
        let div_slice = divergence_staging.slice(..);
        let pressure_slice = pressure_staging.slice(..);

        let (ct_tx, ct_rx) = std::sync::mpsc::channel();
        let (part_tx, part_rx) = std::sync::mpsc::channel();
        let (div_tx, div_rx) = std::sync::mpsc::channel();
        let (pressure_tx, pressure_rx) = std::sync::mpsc::channel();

        ct_slice.map_async(wgpu::MapMode::Read, move |r| {
            ct_tx.send(r).unwrap();
        });
        part_slice.map_async(wgpu::MapMode::Read, move |r| {
            part_tx.send(r).unwrap();
        });
        div_slice.map_async(wgpu::MapMode::Read, move |r| {
            div_tx.send(r).unwrap();
        });
        pressure_slice.map_async(wgpu::MapMode::Read, move |r| {
            pressure_tx.send(r).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        ct_rx.recv().unwrap().unwrap();
        part_rx.recv().unwrap().unwrap();
        div_rx.recv().unwrap().unwrap();
        pressure_rx.recv().unwrap().unwrap();

        let ct_data = ct_slice.get_mapped_range();
        let part_data = part_slice.get_mapped_range();
        let div_data = div_slice.get_mapped_range();
        let pressure_data = pressure_slice.get_mapped_range();

        let cell_types: &[u32] = bytemuck::cast_slice(&ct_data);
        let particle_counts: &[i32] = bytemuck::cast_slice(&part_data);
        let density_error: &[f32] = bytemuck::cast_slice(&div_data);
        let pressure: &[f32] = bytemuck::cast_slice(&pressure_data);

        let mut solid_count = 0usize;
        let mut fluid_count = 0usize;
        let mut air_count = 0usize;

        let mut occupied_cells = 0usize;
        let mut total_particles: i64 = 0;
        let mut min_particles = i32::MAX;
        let mut max_particles = 0i32;

        let mut error_sum = 0.0f32;
        let mut error_max = 0.0f32;

        let mut pressure_sum = 0.0f32;
        let mut pressure_min = f32::INFINITY;
        let mut pressure_max = f32::NEG_INFINITY;

        for idx in 0..cell_count {
            match cell_types[idx] {
                0 => air_count += 1,
                1 => fluid_count += 1,
                2 => solid_count += 1,
                _ => {}
            }

            let count = particle_counts[idx];
            if count > 0 {
                occupied_cells += 1;
                total_particles += count as i64;
                min_particles = min_particles.min(count);
                max_particles = max_particles.max(count);
            }

            if cell_types[idx] == 1 {
                let err = density_error[idx].abs();
                error_sum += err;
                error_max = error_max.max(err);

                let p = pressure[idx];
                pressure_sum += p;
                pressure_min = pressure_min.min(p);
                pressure_max = pressure_max.max(p);
            }
        }

        let fluid_cells = fluid_count.max(1);
        let occupied = occupied_cells.max(1);
        let avg_ppc_fluid = total_particles as f32 / fluid_cells as f32;
        let avg_ppc_occupied = total_particles as f32 / occupied as f32;
        let density_ratio = total_particles as f32 / (fluid_cells as f32 * rest_density.max(1e-6));
        let error_avg = error_sum / fluid_cells as f32;
        let error_avg_norm = error_avg * dt;
        let error_max_norm = error_max * dt;
        let pressure_avg = pressure_sum / fluid_cells as f32;
        if fluid_count == 0 {
            pressure_min = 0.0;
            pressure_max = 0.0;
        }

        println!("  Density diagnostics:");
        println!(
            "    cell types: FLUID={} SOLID={} AIR={} (total={})",
            fluid_count, solid_count, air_count, cell_count
        );
        println!(
            "    particles: total={} occupied_cells={} avg_ppc_fluid={:.2} avg_ppc_occ={:.2} min={} max={}",
            total_particles,
            occupied_cells,
            avg_ppc_fluid,
            avg_ppc_occupied,
            if min_particles == i32::MAX { 0 } else { min_particles },
            max_particles
        );
        println!(
            "    density error: avg|1-rho/rest|={:.3} max|1-rho/rest|={:.3} ratio={:.3}",
            error_avg_norm, error_max_norm, density_ratio
        );
        println!(
            "    pressure: avg={:.6} min={:.6} max={:.6}",
            pressure_avg, pressure_min, pressure_max
        );
        let cell_volume = self.cell_size.powi(3);
        let rest_density = rest_density.max(1e-6);
        let estimated_water_volume = (total_particles as f32 / rest_density) * cell_volume;
        let fluid_volume = fluid_cells as f32 * cell_volume;
        let volume_ratio = if fluid_volume > 0.0 {
            estimated_water_volume / fluid_volume
        } else {
            0.0
        };
        println!(
            "    water volume: est={:.6} m^3 fluid_cells_vol={:.6} ratio={:.3}",
            estimated_water_volume, fluid_volume, volume_ratio
        );

        drop(ct_data);
        drop(part_data);
        drop(div_data);
        drop(pressure_data);
        cell_type_staging.unmap();
        particle_count_staging.unmap();
        divergence_staging.unmap();
        pressure_staging.unmap();
    }

    /// Print jamming diagnostics - call this every N frames from the example
    pub fn print_jamming_diagnostics(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Read back cell types
        let cell_type_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Count Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Count Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Jamming Diagnostics Readback"),
        });
        encoder.copy_buffer_to_buffer(
            &self.pressure.cell_type_buffer,
            0,
            &cell_type_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.sediment_count_buffer,
            0,
            &sediment_count_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.particle_count_buffer,
            0,
            &particle_count_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.vorticity_mag_buffer,
            0,
            &vorticity_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read buffers
        let ct_slice = cell_type_staging.slice(..);
        let sed_slice = sediment_count_staging.slice(..);
        let part_slice = particle_count_staging.slice(..);
        let vort_slice = vorticity_staging.slice(..);

        let (ct_tx, ct_rx) = std::sync::mpsc::channel();
        let (sed_tx, sed_rx) = std::sync::mpsc::channel();
        let (part_tx, part_rx) = std::sync::mpsc::channel();
        let (vort_tx, vort_rx) = std::sync::mpsc::channel();

        ct_slice.map_async(wgpu::MapMode::Read, move |r| {
            ct_tx.send(r).unwrap();
        });
        sed_slice.map_async(wgpu::MapMode::Read, move |r| {
            sed_tx.send(r).unwrap();
        });
        part_slice.map_async(wgpu::MapMode::Read, move |r| {
            part_tx.send(r).unwrap();
        });
        vort_slice.map_async(wgpu::MapMode::Read, move |r| {
            vort_tx.send(r).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        ct_rx.recv().unwrap().unwrap();
        sed_rx.recv().unwrap().unwrap();
        part_rx.recv().unwrap().unwrap();
        vort_rx.recv().unwrap().unwrap();

        let ct_data = ct_slice.get_mapped_range();
        let sed_data = sed_slice.get_mapped_range();
        let part_data = part_slice.get_mapped_range();
        let vort_data = vort_slice.get_mapped_range();

        let cell_types: &[u32] = bytemuck::cast_slice(&ct_data);
        let sediment_counts: &[i32] = bytemuck::cast_slice(&sed_data);
        let particle_counts: &[i32] = bytemuck::cast_slice(&part_data);
        let vorticity_mag: &[f32] = bytemuck::cast_slice(&vort_data);

        // Count cell types
        let solid_count = cell_types.iter().filter(|&&ct| ct == 2).count();
        let fluid_count = cell_types.iter().filter(|&&ct| ct == 1).count();
        let air_count = cell_types.iter().filter(|&&ct| ct == 0).count();

        // Find sample cells with sediment - sample riffle area (x=12-20) where sediment accumulates
        let mut sample_cells = Vec::new();
        let sample_i_start = 12u32.min(self.width - 1);
        let sample_i_end = 20u32.min(self.width - 1);
        for j in 0..self.height.min(8) {
            for k in (self.depth / 2).saturating_sub(2)..=(self.depth / 2 + 2).min(self.depth - 1) {
                for i in sample_i_start..=sample_i_end {
                    let idx = (k * self.width * self.height + j * self.width + i) as usize;
                    let sed_count = sediment_counts[idx];
                    let total_count = particle_counts[idx];
                    let wat_count = total_count - sed_count;
                    let cell_type = cell_types[idx];
                    let vort = vorticity_mag[idx];

                    if sed_count > 0 || total_count > 0 {
                        sample_cells.push((
                            i,
                            j,
                            k,
                            sed_count,
                            wat_count,
                            total_count,
                            cell_type,
                            vort,
                        ));
                    }
                }
            }
        }

        // Calculate vorticity statistics in riffle area
        let mut vort_sum = 0.0f32;
        let mut vort_count = 0;
        let mut vort_max = 0.0f32;
        for j in 0..self.height.min(12) {
            for k in (self.depth / 2).saturating_sub(2)..=(self.depth / 2 + 2).min(self.depth - 1) {
                for i in sample_i_start..=sample_i_end {
                    let idx = (k * self.width * self.height + j * self.width + i) as usize;
                    let vort = vorticity_mag[idx];
                    if vort > 0.0 {
                        vort_sum += vort;
                        vort_count += 1;
                        vort_max = vort_max.max(vort);
                    }
                }
            }
        }

        // Print summary
        println!("\n========== JAMMING DIAGNOSTICS ==========");
        println!(
            "Cell Types: SOLID={} FLUID={} AIR={} (total={})",
            solid_count,
            fluid_count,
            air_count,
            cell_types.len()
        );

        // Print vorticity statistics
        let vort_avg = if vort_count > 0 {
            vort_sum / vort_count as f32
        } else {
            0.0
        };
        println!("\nVorticity in riffle area:");
        println!(
            "  avg={:.4}  max={:.4}  threshold={:.2}  lift_coeff={:.2}",
            vort_avg, vort_max, self.sediment_vorticity_threshold, self.sediment_vorticity_lift
        );

        // Calculate effective lift
        let vort_excess_avg = (vort_avg - self.sediment_vorticity_threshold).max(0.0);
        let lift_factor_avg = (self.sediment_vorticity_lift * vort_excess_avg).min(0.9);
        let settling_cancellation_pct = lift_factor_avg * 100.0;
        println!(
            "  → Average lift cancels {:.1}% of settling velocity",
            settling_cancellation_pct
        );

        if !sample_cells.is_empty() {
            println!("\nSample cells (riffle area i=12-20, j=0-7):");
            println!("  (i, j, k) -> sed | water | total | vort | type");
            for (i, j, k, sed, wat, total, ct, vort) in sample_cells.iter().take(20) {
                let type_str = match ct {
                    0 => "AIR",
                    1 => "FLUID",
                    2 => "SOLID",
                    _ => "???",
                };
                let dominance = if *sed > *wat {
                    "SED>"
                } else if *wat > *sed {
                    "WAT>"
                } else {
                    "="
                };
                println!(
                    "  ({:3},{:3},{:3}) -> {:3} | {:3} | {:3} | {:.3} | {} {}",
                    i, j, k, sed, wat, total, vort, type_str, dominance
                );
            }
        }

        // Check support chains (sample column at riffle area)
        let riffle_i = 15u32.min(self.width - 1); // Middle of riffle area
        let mid_k = self.depth / 2;
        println!(
            "\nRiffle column support chain (i={}, k={}):",
            riffle_i, mid_k
        );
        for j in 0..self.height.min(12) {
            let idx = (mid_k * self.width * self.height + j * self.width + riffle_i) as usize;
            let sed_count = sediment_counts[idx];
            let total_count = particle_counts[idx];
            let wat_count = total_count - sed_count;
            let cell_type = cell_types[idx];
            let vort = vorticity_mag[idx];
            let type_str = match cell_type {
                0 => "AIR  ",
                1 => "FLUID",
                2 => "SOLID",
                _ => "???  ",
            };
            let vort_excess = (vort - self.sediment_vorticity_threshold).max(0.0);
            let lift_factor = (self.sediment_vorticity_lift * vort_excess).min(0.9);
            println!(
                "  j={:2}: {} | sed={:2} wat={:2} | vort={:.3} lift={:.0}% | {}",
                j,
                type_str,
                sed_count,
                wat_count,
                vort,
                lift_factor * 100.0,
                if sed_count > wat_count {
                    "SED>"
                } else if wat_count > sed_count {
                    "WAT>"
                } else {
                    "="
                }
            );
        }
        println!("=========================================\n");

        drop(ct_data);
        drop(sed_data);
        drop(part_data);
        drop(vort_data);
        cell_type_staging.unmap();
        sediment_count_staging.unmap();
        particle_count_staging.unmap();
        vorticity_staging.unmap();
    }
}
