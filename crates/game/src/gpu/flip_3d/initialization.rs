//! Initialization logic for GpuFlip3D.
//!
//! This module extracts the large new() function (originally 2580 lines) into a more manageable form.
//! The full initialization is kept in init_with_solvers() which is called from new().

use super::super::g2p_3d::{GpuG2p3D, SedimentParams3D};
use super::super::p2g_3d::GpuP2g3D;
use super::super::p2g_cell_centric_3d::GpuP2gCellCentric3D;
use super::super::particle_sort::GpuParticleSort;
use super::super::pressure_3d::GpuPressure3D;
use super::params::*;
use super::readback::{ReadbackSlot, ReadbackMode};
use crate::gpu::flip_3d::GpuFlip3D;

use bytemuck::{Pod, Zeroable};
use std::sync::{mpsc, Arc};
use wgpu::util::DeviceExt;

/// Complete GPU pipeline initialization
///
/// This function contains the full 2580-line initialization from the original new().
/// It's extracted here to keep the module's mod.rs focused on the public API.
pub fn init_complete_pipeline(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
    max_particles: usize,
) -> GpuFlip3D {
        assert!(cell_size > 0.0, "cell_size must be positive, got {}", cell_size);
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gravity_3d.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/flow_3d.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vorticity_3d.wgsl").into()),
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
                include_str!("../shaders/vorticity_confine_3d.wgsl").into(),
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
                include_str!("../shaders/sediment_fraction_3d.wgsl").into(),
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
                include_str!("../shaders/sediment_pressure_3d.wgsl").into(),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/porosity_drag_3d.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/enforce_bc_3d.wgsl").into()),
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

        let cell_count = (width * height * depth) as usize;

        // Create density error shader
        let density_error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Error 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/density_error_3d.wgsl").into()),
        });

        let density_error_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Error Params 3D"),
            size: std::mem::size_of::<DensityErrorParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Density error bindings: params, particle_count, cell_type, density_error (uses divergence_buffer)
        let density_error_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Density Error 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let density_error_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Error 3D Bind Group"),
            layout: &density_error_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: density_error_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p2g.particle_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pressure.divergence_buffer.as_entire_binding(),
                },
            ],
        });

        let density_error_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Density Error 3D Pipeline Layout"),
                bind_group_layouts: &[&density_error_bind_group_layout],
                push_constant_ranges: &[],
            });

        let density_error_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Density Error 3D Pipeline"),
                layout: Some(&density_error_pipeline_layout),
                module: &density_error_shader,
                entry_point: Some("compute_density_error"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== Phase 2: Density Position Grid (blub approach) ==========
        // Compute position changes on grid, then particles sample with trilinear

        // Create grid-based position delta buffers
        let position_delta_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta X Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let position_delta_y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Y Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let position_delta_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Z Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create density position grid shader
        let density_position_grid_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Density Position Grid 3D Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/density_position_grid_3d.wgsl").into(),
                ),
            });

        let density_position_grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Position Grid Params 3D"),
            size: std::mem::size_of::<DensityPositionGridParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bindings: params, pressure, cell_type, delta_x, delta_y, delta_z
        let density_position_grid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Density Position Grid 3D Bind Group Layout"),
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
                ],
            });

        let density_position_grid_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Density Position Grid 3D Bind Group"),
                layout: &density_position_grid_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: density_position_grid_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pressure.pressure_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pressure.cell_type_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: position_delta_x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: position_delta_y_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: position_delta_z_buffer.as_entire_binding(),
                    },
                ],
            });

        let density_position_grid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Density Position Grid 3D Pipeline Layout"),
                bind_group_layouts: &[&density_position_grid_bind_group_layout],
                push_constant_ranges: &[],
            });

        let density_position_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Density Position Grid 3D Pipeline"),
                layout: Some(&density_position_grid_pipeline_layout),
                module: &density_position_grid_shader,
                entry_point: Some("compute_position_grid"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== Phase 3: Particle Position Correction (trilinear sampling) ==========
        let density_correct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Correct 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/density_correct_3d.wgsl").into(),
            ),
        });

        let density_correct_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Correct Params 3D"),
            size: std::mem::size_of::<DensityCorrectionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bindings: params, delta_x, delta_y, delta_z, cell_type, positions, densities, velocities
        let density_correct_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Density Correct 3D Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let density_correct_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Correct 3D Bind Group"),
            layout: &density_correct_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: density_correct_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: position_delta_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: position_delta_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: position_delta_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: g2p.positions_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: velocities_buffer.as_ref().as_entire_binding(),
                },
            ],
        });

        let density_correct_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Density Correct 3D Pipeline Layout"),
                bind_group_layouts: &[&density_correct_bind_group_layout],
                push_constant_ranges: &[],
            });

        let density_correct_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Density Correct 3D Pipeline"),
                layout: Some(&density_correct_pipeline_layout),
                module: &density_correct_shader,
                entry_point: Some("correct_positions"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== Sediment Density Projection (Granular Packing) ==========
        let sediment_cell_type_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Cell Type 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sediment_cell_type_3d.wgsl").into(),
            ),
        });
        let gravel_obstacle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravel Obstacle 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/gravel_obstacle_3d.wgsl").into(),
            ),
        });
        let gravel_porosity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravel Porosity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/gravel_porosity_3d.wgsl").into(),
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

        // ========== Fluid Cell Expansion (Standard FLIP "7 points per particle") ==========
        let fluid_cell_expand_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Cell Expand 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fluid_cell_expand_3d.wgsl").into(),
            ),
        });

        let fluid_cell_expand_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fluid Cell Expand 3D Bind Group Layout"),
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

        let fluid_cell_expand_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Cell Expand 3D Bind Group"),
            layout: &fluid_cell_expand_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sediment_cell_type_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p2g.particle_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pressure.cell_type_buffer.as_entire_binding(),
                },
            ],
        });

        let fluid_cell_expand_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fluid Cell Expand 3D Pipeline Layout"),
                bind_group_layouts: &[&fluid_cell_expand_bind_group_layout],
                push_constant_ranges: &[],
            });

        let fluid_cell_expand_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fluid Cell Expand 3D Pipeline"),
                layout: Some(&fluid_cell_expand_pipeline_layout),
                module: &fluid_cell_expand_shader,
                entry_point: Some("expand_fluid_cells"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Velocity extrapolation: extend velocities into AIR cells near surface
        let velocity_extrap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Velocity Extrapolate 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/velocity_extrapolate_3d.wgsl").into(),
            ),
        });

        let velocity_extrap_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity Extrap Params Buffer"),
            size: std::mem::size_of::<VelocityExtrapParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create validity buffers for U, V, W faces
        let u_face_count = ((width + 1) * height * depth) as usize;
        let v_face_count = (width * (height + 1) * depth) as usize;
        let w_face_count = (width * height * (depth + 1)) as usize;

        let valid_u_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Valid U Buffer"),
            size: (u_face_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let valid_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Valid V Buffer"),
            size: (v_face_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let valid_w_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Valid W Buffer"),
            size: (w_face_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocity_extrap_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Velocity Extrap 3D Bind Group Layout"),
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
                ],
            });

        let velocity_extrap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Velocity Extrap 3D Bind Group"),
            layout: &velocity_extrap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: velocity_extrap_params_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: valid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: valid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: valid_w_buffer.as_entire_binding(),
                },
            ],
        });

        let velocity_extrap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Velocity Extrap 3D Pipeline Layout"),
                bind_group_layouts: &[&velocity_extrap_bind_group_layout],
                push_constant_ranges: &[],
            });

        let velocity_extrap_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Velocity Extrap Init Pipeline"),
                layout: Some(&velocity_extrap_pipeline_layout),
                module: &velocity_extrap_shader,
                entry_point: Some("init_valid_faces"),
                compilation_options: Default::default(),
                cache: None,
            });

        let velocity_extrap_u_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Velocity Extrap U Pipeline"),
                layout: Some(&velocity_extrap_pipeline_layout),
                module: &velocity_extrap_shader,
                entry_point: Some("extrapolate_u"),
                compilation_options: Default::default(),
                cache: None,
            });

        let velocity_extrap_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Velocity Extrap V Pipeline"),
                layout: Some(&velocity_extrap_pipeline_layout),
                module: &velocity_extrap_shader,
                entry_point: Some("extrapolate_v"),
                compilation_options: Default::default(),
                cache: None,
            });

        let velocity_extrap_w_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Velocity Extrap W Pipeline"),
                layout: Some(&velocity_extrap_pipeline_layout),
                module: &velocity_extrap_shader,
                entry_point: Some("extrapolate_w"),
                compilation_options: Default::default(),
                cache: None,
            });

        let velocity_extrap_finalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Velocity Extrap Finalize Pipeline"),
                layout: Some(&velocity_extrap_pipeline_layout),
                module: &velocity_extrap_shader,
                entry_point: Some("finalize_valid"),
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

        let sediment_density_error_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sediment Density Error 3D Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/sediment_density_error_3d.wgsl").into(),
                ),
            });

        let sediment_density_error_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sediment Density Error 3D Bind Group"),
                layout: &density_error_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: density_error_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: p2g.sediment_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pressure.cell_type_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pressure.divergence_buffer.as_entire_binding(),
                    },
                ],
            });

        let sediment_density_error_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sediment Density Error 3D Pipeline"),
                layout: Some(&density_error_pipeline_layout),
                module: &sediment_density_error_shader,
                entry_point: Some("compute_sediment_density_error"),
                compilation_options: Default::default(),
                cache: None,
            });

        let sediment_density_correct_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sediment Density Correct 3D Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/sediment_density_correct_3d.wgsl").into(),
                ),
            });

        let sediment_density_correct_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sediment Density Correct 3D Pipeline"),
                layout: Some(&density_correct_pipeline_layout),
                module: &sediment_density_correct_shader,
                entry_point: Some("correct_positions"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ========== SDF Collision (Advection + Solid Collision) ==========
        let sdf_collision_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Collision 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_collision_3d.wgsl").into()),
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

        GpuFlip3D {
            width,
            height,
            depth,
            cell_size,
            open_boundaries: 0, // All boundaries closed by default
            vorticity_epsilon: 0.05,
            flip_ratio: 0.99, // Standard FLIP ratio for water (99% FLIP, 1% PIC)
            density_projection_enabled: true, // Enable by default for main sim
            slip_factor: 1.0, // Free-slip by default (allow tangential flow at boundaries)
            water_rest_density: 8.0, // Target 8 particles per cell (2x2x2 seeding)
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
            density_error_pipeline,
            density_error_bind_group,
            density_error_params_buffer,
            density_position_grid_pipeline,
            density_position_grid_bind_group,
            density_position_grid_params_buffer,
            position_delta_x_buffer,
            position_delta_y_buffer,
            position_delta_z_buffer,
            density_correct_pipeline,
            density_correct_bind_group,
            density_correct_params_buffer,
            sediment_cell_type_pipeline,
            sediment_cell_type_bind_group,
            sediment_cell_type_params_buffer,
            fluid_cell_expand_pipeline,
            fluid_cell_expand_bind_group,
            velocity_extrap_params_buffer,
            valid_u_buffer,
            valid_v_buffer,
            valid_w_buffer,
            velocity_extrap_init_pipeline,
            velocity_extrap_u_pipeline,
            velocity_extrap_v_pipeline,
            velocity_extrap_w_pipeline,
            velocity_extrap_finalize_pipeline,
            velocity_extrap_bind_group,
            sediment_density_error_pipeline,
            sediment_density_error_bind_group,
            sediment_density_correct_pipeline,
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

