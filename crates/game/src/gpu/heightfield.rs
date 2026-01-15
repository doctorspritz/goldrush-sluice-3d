use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridVertex {
    pub position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub cell_size: f32,
    pub grid_width: u32,
    pub grid_depth: u32,
    pub time: f32,
    pub _pad: u32,
}

pub const HEIGHTFIELD_DEBUG_STATS_LEN: usize = 12;
pub const HEIGHTFIELD_DEBUG_SCALE: f32 = 1000.0;

const DBG_EROSION_CELLS: usize = 0;
const DBG_DEPOSITION_CELLS: usize = 1;
const DBG_EROSION_MAX_MM: usize = 2;
const DBG_DEPOSITION_MAX_MM: usize = 3;
const DBG_EROSION_SEDIMENT: usize = 4;
const DBG_EROSION_OVERBURDEN: usize = 5;
const DBG_EROSION_GRAVEL: usize = 6;
const DBG_EROSION_PAYDIRT: usize = 7;
const DBG_DEPOSITION_SEDIMENT: usize = 8;
const DBG_DEPOSITION_OVERBURDEN: usize = 9;
const DBG_DEPOSITION_GRAVEL: usize = 10;
const DBG_DEPOSITION_PAYDIRT: usize = 11;

#[derive(Clone, Copy, Debug, Default)]
pub struct HeightfieldErosionDebugStats {
    pub erosion_cells: u32,
    pub deposition_cells: u32,
    pub erosion_max_height: f32,
    pub deposition_max_height: f32,
    pub erosion_layers: [u32; 4],
    pub deposition_layers: [u32; 4],
}

impl HeightfieldErosionDebugStats {
    pub fn from_raw(raw: &[u32]) -> Self {
        let get = |idx: usize| -> u32 { *raw.get(idx).unwrap_or(&0) };
        Self {
            erosion_cells: get(DBG_EROSION_CELLS),
            deposition_cells: get(DBG_DEPOSITION_CELLS),
            erosion_max_height: get(DBG_EROSION_MAX_MM) as f32 / HEIGHTFIELD_DEBUG_SCALE,
            deposition_max_height: get(DBG_DEPOSITION_MAX_MM) as f32 / HEIGHTFIELD_DEBUG_SCALE,
            erosion_layers: [
                get(DBG_EROSION_SEDIMENT),
                get(DBG_EROSION_OVERBURDEN),
                get(DBG_EROSION_GRAVEL),
                get(DBG_EROSION_PAYDIRT),
            ],
            deposition_layers: [
                get(DBG_DEPOSITION_SEDIMENT),
                get(DBG_DEPOSITION_OVERBURDEN),
                get(DBG_DEPOSITION_GRAVEL),
                get(DBG_DEPOSITION_PAYDIRT),
            ],
        }
    }
}

/// GPU-accelerated Heightfield Simulation.
///
/// Manages the state for:
/// - Hydrodynamics (Shallow Water Equations)
/// - Erosion & Sediment Transport
/// - Multi-layer Geology (Bedrock, Paydirt, Overburden)
pub struct GpuHeightfield {
    width: u32,
    depth: u32,
    cell_size: f32,
    debug_flags: u32,

    // Geology Buffers
    pub bedrock_buffer: wgpu::Buffer,
    pub paydirt_buffer: wgpu::Buffer,
    pub gravel_buffer: wgpu::Buffer, // Gravel layer (erosion resistant)
    pub overburden_buffer: wgpu::Buffer,
    pub sediment_buffer: wgpu::Buffer, // Deposited sediment
    pub surface_material_buffer: wgpu::Buffer, // What material is on TOP (0=bed,1=pay,2=gravel,3=over,4=sed)
    pub settling_time_buffer: wgpu::Buffer,    // u32: frames since last disturbance (for temporal stability)

    // Water State Buffers
    pub water_depth_buffer: wgpu::Buffer,
    pub water_velocity_x_buffer: wgpu::Buffer,
    pub water_velocity_z_buffer: wgpu::Buffer,
    pub suspended_sediment_buffer: wgpu::Buffer,
    pub suspended_sediment_next_buffer: wgpu::Buffer, // Double buffer for race-free transport
    pub suspended_overburden_buffer: wgpu::Buffer,
    pub suspended_overburden_next_buffer: wgpu::Buffer,
    pub suspended_gravel_buffer: wgpu::Buffer,
    pub suspended_gravel_next_buffer: wgpu::Buffer,
    pub suspended_paydirt_buffer: wgpu::Buffer,
    pub suspended_paydirt_next_buffer: wgpu::Buffer,

    // Derived/Intermediate Buffers
    pub water_surface_buffer: wgpu::Buffer, // Calculated as Ground + Water Depth
    pub flux_x_buffer: wgpu::Buffer,
    pub flux_z_buffer: wgpu::Buffer,
    pub debug_stats_buffer: wgpu::Buffer,

    // Bind Groups
    pub params_bind_group: wgpu::BindGroup,
    pub terrain_bind_group: wgpu::BindGroup,
    pub water_bind_group: wgpu::BindGroup,

    // Pipelines
    pub surface_pipeline: wgpu::ComputePipeline,
    pub flux_pipeline: wgpu::ComputePipeline,
    pub depth_pipeline: wgpu::ComputePipeline,

    pub settling_pipeline: wgpu::ComputePipeline,
    pub erosion_pipeline: wgpu::ComputePipeline,
    pub sediment_transport_pipeline: wgpu::ComputePipeline,
    pub collapse_pipeline: wgpu::ComputePipeline,
    pub collapse_red_pipeline: wgpu::ComputePipeline,
    pub collapse_black_pipeline: wgpu::ComputePipeline,

    // Emitter Pipeline
    pub emitter_pipeline: wgpu::ComputePipeline,
    pub emitter_params_buffer: wgpu::Buffer,
    pub emitter_bind_group: wgpu::BindGroup,

    // Material Tool Pipelines
    pub material_tool_pipeline: wgpu::ComputePipeline,
    pub excavate_pipeline: wgpu::ComputePipeline,
    pub material_tool_params_buffer: wgpu::Buffer,
    pub material_tool_bind_group: wgpu::BindGroup,
    pub material_tool_terrain_bind_group: wgpu::BindGroup,

    // Params Buffer (to update every frame)
    // Params Buffer (to update every frame)
    pub params_buffer: wgpu::Buffer,

    // Rendering
    pub render_pipeline: wgpu::RenderPipeline,
    pub water_pipeline: wgpu::RenderPipeline,
    pub render_bind_group: wgpu::BindGroup,
    pub render_uniform_buffer: wgpu::Buffer,
    pub grid_vertex_buffer: wgpu::Buffer,
    pub grid_index_buffer: wgpu::Buffer,
    pub num_indices: u32,

    // Bridge Integration
    pub bridge_merge_pipeline: wgpu::ComputePipeline,
    pub bridge_merge_bind_group: Option<wgpu::BindGroup>,
    pub bridge_merge_bg_layout: wgpu::BindGroupLayout,
}

impl GpuHeightfield {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        depth: u32,
        cell_size: f32,
        initial_height: f32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let _size = (width * depth) as usize * std::mem::size_of::<f32>();

        // Helper to create valid storage buffers
        let create_storage = |label: &str, init_val: f32| -> wgpu::Buffer {
            let data = vec![init_val; (width * depth) as usize];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };

        // 1. Initialize Geology
        let bedrock = create_storage("Bedrock Buffer", initial_height * 0.5);
        let paydirt = create_storage("Paydirt Buffer", initial_height * 0.25);
        let gravel = create_storage("Gravel Buffer", initial_height * 0.05);
        let overburden = create_storage("Overburden Buffer", initial_height * 0.2);
        let sediment = create_storage("Sediment Buffer", 0.0);

        // Surface material tracker: what material is on TOP (0=bed,1=pay,2=gravel,3=over,4=sed)
        // Start with sediment (4) as default since most terrain has some sediment on top
        let surface_material_data = vec![4u32; (width * depth) as usize];
        let surface_material = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Surface Material Buffer"),
            contents: bytemuck::cast_slice(&surface_material_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Settling time tracker: frames since last disturbance (starts at 0 = freshly deposited)
        let settling_time_data = vec![0u32; (width * depth) as usize];
        let settling_time = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Settling Time Buffer"),
            contents: bytemuck::cast_slice(&settling_time_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // 2. Initialize Water
        let water_depth = create_storage("Water Depth Buffer", 0.0);
        let water_vel_x = create_storage("Water Vel X Buffer", 0.0);
        let water_vel_z = create_storage("Water Vel Z Buffer", 0.0);
        let suspended = create_storage("Suspended Sediment Buffer", 0.0);
        let suspended_next = create_storage("Suspended Sediment Next Buffer", 0.0);
        let suspended_overburden = create_storage("Suspended Overburden Buffer", 0.0);
        let suspended_overburden_next = create_storage("Suspended Overburden Next Buffer", 0.0);
        let suspended_gravel = create_storage("Suspended Gravel Buffer", 0.0);
        let suspended_gravel_next = create_storage("Suspended Gravel Next Buffer", 0.0);
        let suspended_paydirt = create_storage("Suspended Paydirt Buffer", 0.0);
        let suspended_paydirt_next = create_storage("Suspended Paydirt Next Buffer", 0.0);

        // 3. Initialize Intermediate
        let water_surface = create_storage("Water Surface Buffer", 0.0);
        let flux_x = create_storage("Flux X Buffer", 0.0);
        let flux_z = create_storage("Flux Z Buffer", 0.0);
        let debug_stats_data = vec![0u32; HEIGHTFIELD_DEBUG_STATS_LEN];
        let debug_stats_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Heightfield Debug Stats Buffer"),
            contents: bytemuck::cast_slice(&debug_stats_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Bind Group Layouts (Placeholder - will implement in next step with proper layout)
        // For now constructing the struct fields. simpler to create layout and bindgroups here.

        // 4. Uniforms
        let params_size = std::mem::size_of::<[u32; 20]>(); // Alignment padding safe size
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heightfield Params"),
            size: params_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/heightfield_water.wgsl").into()),
        });

        // Group 0: Params
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Heightfield Params Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Heightfield Params Bind Group"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Group 1: Water State (RW)
        let water_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water State Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // depth
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // vel_x
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // vel_z
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // surface
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // flux_x
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // flux_z
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_sediment
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_sediment_next (double buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_overburden
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_overburden_next
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_gravel
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_gravel_next
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_paydirt
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // suspended_paydirt_next
            ],
        });

        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Bind Group"),
            layout: &water_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: water_depth.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: water_vel_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: water_vel_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: water_surface.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: flux_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flux_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: suspended.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: suspended_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: suspended_overburden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: suspended_overburden_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: suspended_gravel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: suspended_gravel_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: suspended_paydirt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: suspended_paydirt_next.as_entire_binding(),
                },
            ],
        });

        // Group 2: Terrain (ReadWrite for erosion)
        let terrain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // bedrock
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // paydirt
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // gravel
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // overburden
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // sediment
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // surface_material
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // settling_time
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // debug_stats
            ],
        });

        let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Bind Group"),
            layout: &terrain_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bedrock.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: paydirt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gravel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: overburden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sediment.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: surface_material.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: settling_time.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: debug_stats_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Heightfield Pipeline Layout"),
            bind_group_layouts: &[&params_layout, &water_layout, &terrain_layout],
            push_constant_ranges: &[],
        });

        let surface_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Surface Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_surface"),
            compilation_options: Default::default(),
            cache: None,
        });

        let flux_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Flux Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_flux"),
            compilation_options: Default::default(),
            cache: None,
        });

        let depth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Depth Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_depth"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Erosion Shader & Pipelines
        let erosion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Erosion Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_erosion.wgsl").into(),
            ),
        });

        let settling_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Settling Pipeline"),
            layout: Some(&pipeline_layout),
            module: &erosion_shader,
            entry_point: Some("update_settling"),
            compilation_options: Default::default(),
            cache: None,
        });

        let erosion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Erosion Pipeline"),
            layout: Some(&pipeline_layout), // Reusing same layout
            module: &erosion_shader,
            entry_point: Some("update_erosion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sediment_transport_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sediment Transport Pipeline"),
                layout: Some(&pipeline_layout),
                module: &erosion_shader,
                entry_point: Some("update_sediment_transport"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Collapse Shader & Pipeline (angle of repose)
        let collapse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Collapse Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_collapse.wgsl").into(),
            ),
        });

        // Collapse needs params (group 0) and terrain (group 2), but not water (group 1)
        // We'll use a simpler layout for collapse
        let _collapse_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collapse Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let collapse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collapse Pipeline Layout"),
                bind_group_layouts: &[&params_layout, &water_layout, &terrain_layout], // Reuse full layout for compatibility
                push_constant_ranges: &[],
            });

        let collapse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Collapse Pipeline"),
            layout: Some(&collapse_pipeline_layout),
            module: &collapse_shader,
            entry_point: Some("update_collapse"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Red-black collapse pipelines for race-free updates
        let collapse_red_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Collapse Red Pipeline"),
                layout: Some(&collapse_pipeline_layout),
                module: &collapse_shader,
                entry_point: Some("update_collapse_red"),
                compilation_options: Default::default(),
                cache: None,
            });

        let collapse_black_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Collapse Black Pipeline"),
                layout: Some(&collapse_pipeline_layout),
                module: &collapse_shader,
                entry_point: Some("update_collapse_black"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Emitter Shader & Pipelines
        let emitter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Emitter Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_emitter.wgsl").into(),
            ),
        });

        // Emitter params buffer (pos/radius/rate + vel + concs + world/tile dims + origin + cell_size)
        let emitter_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Emitter Params Buffer"),
            size: 96, // Increased size for velocity
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let emitter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Emitter Bind Group Layout"),
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

        let emitter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Emitter Bind Group"),
            layout: &emitter_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: emitter_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: water_depth.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: suspended.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: suspended_overburden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: suspended_gravel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: suspended_paydirt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: water_vel_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: water_vel_z.as_entire_binding(),
                },
            ],
        });

        let emitter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Emitter Pipeline Layout"),
                bind_group_layouts: &[&emitter_layout],
                push_constant_ranges: &[],
            });

        let emitter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Emitter Pipeline"),
            layout: Some(&emitter_pipeline_layout),
            module: &emitter_shader,
            entry_point: Some("add_water"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Material Tool Shader & Pipelines
        let material_tool_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Material Tool Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_material_tool.wgsl").into(),
            ),
        });

        // Material tool params buffer (pos/radius/amount + world/tile dims + origin + cell_size/dt)
        let material_tool_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Tool Params Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Material tool bind group layout: params + terrain buffers
        let material_tool_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material Tool Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Terrain layout for material tool (same as terrain_layout but as group 1)
        let material_tool_terrain_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material Tool Terrain Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        let material_tool_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Tool Bind Group"),
            layout: &material_tool_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_tool_params_buffer.as_entire_binding(),
            }],
        });

        let material_tool_terrain_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Material Tool Terrain Bind Group"),
                layout: &material_tool_terrain_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bedrock.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: paydirt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gravel.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: overburden.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: sediment.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: surface_material.as_entire_binding(),
                    },
                ],
            });

        let material_tool_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Material Tool Pipeline Layout"),
                bind_group_layouts: &[&material_tool_layout, &material_tool_terrain_layout],
                push_constant_ranges: &[],
            });

        let material_tool_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Material Tool Pipeline"),
                layout: Some(&material_tool_pipeline_layout),
                module: &material_tool_shader,
                entry_point: Some("apply_material_tool"),
                compilation_options: Default::default(),
                cache: None,
            });

        let excavate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Excavate Pipeline"),
            layout: Some(&material_tool_pipeline_layout),
            module: &material_tool_shader,
            entry_point: Some("excavate"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Rendering Setup ---

        // 1. Grid Mesh (Static)
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for z in 0..depth {
            for x in 0..width {
                // vertices
                vertices.push(GridVertex {
                    position: [x as f32, z as f32],
                });

                // indices (quads)
                if x < width - 1 && z < depth - 1 {
                    let i = z * width + x;
                    // Triangle 1
                    indices.push(i);
                    indices.push(i + width);
                    indices.push(i + 1);
                    // Triangle 2
                    indices.push(i + 1);
                    indices.push(i + width);
                    indices.push(i + width + 1);
                }
            }
        }
        let num_indices = indices.len() as u32;

        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let grid_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // 2. Render Uniforms
        let render_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Uniform Buffer"),
            size: std::mem::size_of::<RenderUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Render Bind Group
        let render_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT, // Only used in terrain fragment shader
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX, // Only vertex needs it - sediment_load passed via interpolation
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: render_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bedrock.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: paydirt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gravel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: overburden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sediment.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: water_surface.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: water_depth.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: surface_material.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: suspended.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: suspended_overburden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: suspended_gravel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: suspended_paydirt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: water_vel_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: water_vel_z.as_entire_binding(),
                },
            ],
        });

        // 4. Pipelines
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_render.wgsl").into(),
            ),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bg_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GridVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_water"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GridVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_water"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ========== Bridge Merge Pipeline ==========
        let bridge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bridge Merge Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_bridge_merge.wgsl").into(),
            ),
        });

        let bridge_merge_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bridge Merge Transfer Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });

        let bridge_merge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bridge Merge Layout"),
                bind_group_layouts: &[&params_layout, &bridge_merge_bg_layout, &water_layout],
                push_constant_ranges: &[],
            });

        let bridge_merge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bridge Merge Pipeline"),
                layout: Some(&bridge_merge_pipeline_layout),
                module: &bridge_shader,
                entry_point: Some("merge_particles"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            width,
            depth,
            cell_size,
            debug_flags: 0,
            bedrock_buffer: bedrock,
            paydirt_buffer: paydirt,
            gravel_buffer: gravel,
            overburden_buffer: overburden,
            sediment_buffer: sediment,
            surface_material_buffer: surface_material,
            settling_time_buffer: settling_time,

            water_depth_buffer: water_depth,
            water_velocity_x_buffer: water_vel_x,
            water_velocity_z_buffer: water_vel_z,
            suspended_sediment_buffer: suspended,
            suspended_sediment_next_buffer: suspended_next,
            suspended_overburden_buffer: suspended_overburden,
            suspended_overburden_next_buffer: suspended_overburden_next,
            suspended_gravel_buffer: suspended_gravel,
            suspended_gravel_next_buffer: suspended_gravel_next,
            suspended_paydirt_buffer: suspended_paydirt,
            suspended_paydirt_next_buffer: suspended_paydirt_next,

            water_surface_buffer: water_surface,
            flux_x_buffer: flux_x,
            flux_z_buffer: flux_z,
            debug_stats_buffer,

            terrain_bind_group,
            water_bind_group,
            params_bind_group,

            surface_pipeline,
            flux_pipeline,
            depth_pipeline,
            settling_pipeline,
            erosion_pipeline,
            sediment_transport_pipeline,
            collapse_pipeline,
            collapse_red_pipeline,
            collapse_black_pipeline,

            emitter_pipeline,
            emitter_params_buffer,
            emitter_bind_group,

            material_tool_pipeline,
            excavate_pipeline,
            material_tool_params_buffer,
            material_tool_bind_group,
            material_tool_terrain_bind_group,

            params_buffer,

            render_pipeline,
            water_pipeline,
            render_bind_group,
            render_uniform_buffer,
            grid_vertex_buffer,
            grid_index_buffer,
            num_indices,
            bridge_merge_pipeline,
            bridge_merge_bind_group: None,
            bridge_merge_bg_layout,
        }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        let _ = dt;
        self.dispatch_tile(encoder, self.width, self.depth);
    }

    pub fn dispatch_tile(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tile_width: u32,
        tile_depth: u32,
    ) {
        let x_groups = tile_width.div_ceil(16);
        let z_groups = tile_depth.div_ceil(16);

        // Helper macro to dispatch a compute pass
        macro_rules! dispatch_step {
            ($label:expr, $pipeline:expr) => {{
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some($label),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.params_bind_group, &[]);
                pass.set_bind_group(1, &self.water_bind_group, &[]);
                pass.set_bind_group(2, &self.terrain_bind_group, &[]);
                pass.set_pipeline($pipeline);
                pass.dispatch_workgroups(x_groups, z_groups, 1);
            }};
        }

        // CRITICAL: Update Surface FIRST - flux needs water_surface to compute gradients!
        // water_surface = ground_height + water_depth
        // Without this, flux sees stale/zero surface heights and water won't flow.
        dispatch_step!("Update Surface", &self.surface_pipeline);

        // 1. Update Flux (Updates Velocity + Flux) - Reads Surface gradients
        dispatch_step!("Update Flux", &self.flux_pipeline);

        // 2. Update Depth (Volume Conservation) - Reads Flux, updates water_depth
        dispatch_step!("Update Depth", &self.depth_pipeline);

        // 3a. Settling (post-flux) - Reads Depth/Vel, writes Terrain/Suspended
        dispatch_step!("Update Settling", &self.settling_pipeline);

        // 3b. Erosion (post-settling) - Reads Depth/Vel/Terrain, writes Terrain/Suspended
        dispatch_step!("Update Erosion", &self.erosion_pipeline);

        // 4. Sediment Transport (flux-based advection) - Reads from current, writes to next buffer
        dispatch_step!(
            "Update Sediment Transport",
            &self.sediment_transport_pipeline
        );

        // 4b. Copy suspended_sediment_next -> suspended_sediment (swap double buffers)
        let buffer_size = (self.width * self.depth) as u64 * std::mem::size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.suspended_sediment_next_buffer,
            0,
            &self.suspended_sediment_buffer,
            0,
            buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.suspended_overburden_next_buffer,
            0,
            &self.suspended_overburden_buffer,
            0,
            buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.suspended_gravel_next_buffer,
            0,
            &self.suspended_gravel_buffer,
            0,
            buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.suspended_paydirt_next_buffer,
            0,
            &self.suspended_paydirt_buffer,
            0,
            buffer_size,
        );

        // 5. Collapse (angle of repose / slope stability) - Writes Terrain
        // Use red-black pattern for race-free updates: red cells don't neighbor other red cells
        dispatch_step!("Update Collapse Red", &self.collapse_red_pipeline);
        dispatch_step!("Update Collapse Black", &self.collapse_black_pipeline);
    }

    pub fn update_params(&self, queue: &wgpu::Queue, dt: f32) {
        self.update_params_tile(queue, dt, 0, 0, self.width, self.depth);
    }

    pub fn update_params_tile(
        &self,
        queue: &wgpu::Queue,
        dt: f32,
        origin_x: u32,
        origin_z: u32,
        tile_width: u32,
        tile_depth: u32,
    ) {
        let params: [u32; 20] = [
            self.width,
            self.depth,
            tile_width,
            tile_depth,
            origin_x,
            origin_z,
            0,
            0,
            bytemuck::cast(self.cell_size),
            bytemuck::cast(dt),
            bytemuck::cast(9.81f32),
            bytemuck::cast(0.02f32), // Manning's n coefficient (smooth channel)
            bytemuck::cast(1000.0f32), // rho_water
            bytemuck::cast(2650.0f32), // rho_sediment
            bytemuck::cast(0.001f32), // water_viscosity
            bytemuck::cast(0.045f32), // critical_shields
            bytemuck::cast(0.01f32), // k_erosion (100x CPU rate for visibility)
            bytemuck::cast(0.05f32), // max_erosion_per_step (10x CPU, dt=0.02 → 1mm/step max)
            self.debug_flags,
            bytemuck::cast(HEIGHTFIELD_DEBUG_SCALE),
        ];
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));
    }

    pub fn set_debug_flags(&mut self, flags: u32) {
        self.debug_flags = flags;
    }

    pub fn debug_flags(&self) -> u32 {
        self.debug_flags
    }

    pub fn reset_debug_stats(&self, queue: &wgpu::Queue) {
        let zeros = [0u32; HEIGHTFIELD_DEBUG_STATS_LEN];
        queue.write_buffer(&self.debug_stats_buffer, 0, bytemuck::cast_slice(&zeros));
    }

    pub async fn read_debug_stats(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> HeightfieldErosionDebugStats {
        if self.debug_flags == 0 {
            return HeightfieldErosionDebugStats::default();
        }

        let size = (HEIGHTFIELD_DEBUG_STATS_LEN * std::mem::size_of::<u32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heightfield Debug Stats Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.debug_stats_buffer, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let raw: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        HeightfieldErosionDebugStats::from_raw(&raw)
    }

    /// Update GPU emitter parameters
    pub fn update_emitter(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        rate: f32,
        sediment_conc: f32,
        overburden_conc: f32,
        gravel_conc: f32,
        paydirt_conc: f32,
        vel_x: f32,
        vel_z: f32,
        dt: f32,
        enabled: bool,
    ) {
        self.update_emitter_tile(
            queue,
            pos_x,
            pos_z,
            radius,
            rate,
            sediment_conc,
            overburden_conc,
            gravel_conc,
            paydirt_conc,
            vel_x,
            vel_z,
            dt,
            enabled,
            0,
            0,
            self.width,
            self.depth,
        );
    }

    pub fn update_emitter_tile(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        rate: f32,
        sediment_conc: f32,
        overburden_conc: f32,
        gravel_conc: f32,
        paydirt_conc: f32,
        vel_x: f32,
        vel_z: f32,
        dt: f32,
        enabled: bool,
        origin_x: u32,
        origin_z: u32,
        tile_width: u32,
        tile_depth: u32,
    ) {
        // EmitterParams struct: pos_x, pos_z, radius, rate, dt, enabled, world/tile dims,
        // origin, cell_size, concs, velocity, padding -> 96 bytes (24 u32s)
        let params: [u32; 24] = [
            bytemuck::cast(pos_x),
            bytemuck::cast(pos_z),
            bytemuck::cast(radius),
            bytemuck::cast(rate),
            bytemuck::cast(dt),
            if enabled { 1 } else { 0 },
            self.width,
            self.depth,
            tile_width,
            tile_depth,
            origin_x,
            origin_z,
            bytemuck::cast(self.cell_size),
            bytemuck::cast(sediment_conc),
            bytemuck::cast(overburden_conc),
            bytemuck::cast(gravel_conc),
            bytemuck::cast(paydirt_conc),
            bytemuck::cast(vel_x),
            bytemuck::cast(vel_z),
            0,
            0,
            0,
            0,
            0,
        ];
        queue.write_buffer(
            &self.emitter_params_buffer,
            0,
            bytemuck::cast_slice(&params),
        );
    }

    /// Dispatch emitter compute pass - call before main dispatch
    pub fn dispatch_emitter(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch_emitter_tile(encoder, self.width, self.depth);
    }

    pub fn dispatch_emitter_tile(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tile_width: u32,
        tile_depth: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Emitter Compute Pass"),
            timestamp_writes: None,
        });

        let x_groups = tile_width.div_ceil(16);
        let z_groups = tile_depth.div_ceil(16);

        pass.set_pipeline(&self.emitter_pipeline);
        pass.set_bind_group(0, &self.emitter_bind_group, &[]);
        pass.dispatch_workgroups(x_groups, z_groups, 1);
    }

    /// Update GPU material tool parameters
    /// material_type: 0=sediment, 1=overburden, 2=gravel
    pub fn update_material_tool(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        amount: f32,
        material_type: u32,
        dt: f32,
        enabled: bool,
    ) {
        self.update_material_tool_tile(
            queue,
            pos_x,
            pos_z,
            radius,
            amount,
            material_type,
            dt,
            enabled,
            0,
            0,
            self.width,
            self.depth,
        );
    }

    pub fn update_material_tool_tile(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        amount: f32,
        material_type: u32,
        dt: f32,
        enabled: bool,
        origin_x: u32,
        origin_z: u32,
        tile_width: u32,
        tile_depth: u32,
    ) {
        let params: [u32; 16] = [
            bytemuck::cast(pos_x),
            bytemuck::cast(pos_z),
            bytemuck::cast(radius),
            bytemuck::cast(amount),
            material_type,
            if enabled { 1 } else { 0 },
            self.width,
            self.depth,
            tile_width,
            tile_depth,
            origin_x,
            origin_z,
            bytemuck::cast(self.cell_size),
            bytemuck::cast(dt),
            0,
            0,
        ];
        queue.write_buffer(
            &self.material_tool_params_buffer,
            0,
            bytemuck::cast_slice(&params),
        );
    }

    /// Dispatch material tool (add/remove specific material type)
    pub fn dispatch_material_tool(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch_material_tool_tile(encoder, self.width, self.depth);
    }

    pub fn dispatch_material_tool_tile(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tile_width: u32,
        tile_depth: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Material Tool Compute Pass"),
            timestamp_writes: None,
        });

        let x_groups = tile_width.div_ceil(16);
        let z_groups = tile_depth.div_ceil(16);

        pass.set_pipeline(&self.material_tool_pipeline);
        pass.set_bind_group(0, &self.material_tool_bind_group, &[]);
        pass.set_bind_group(1, &self.material_tool_terrain_bind_group, &[]);
        pass.dispatch_workgroups(x_groups, z_groups, 1);
    }

    /// Dispatch excavation (generic dig from top layer down)
    pub fn dispatch_excavate(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch_excavate_tile(encoder, self.width, self.depth);
    }

    pub fn dispatch_excavate_tile(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tile_width: u32,
        tile_depth: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Excavate Compute Pass"),
            timestamp_writes: None,
        });

        let x_groups = tile_width.div_ceil(16);
        let z_groups = tile_depth.div_ceil(16);

        pass.set_pipeline(&self.excavate_pipeline);
        pass.set_bind_group(0, &self.material_tool_bind_group, &[]);
        pass.set_bind_group(1, &self.material_tool_terrain_bind_group, &[]);
        pass.dispatch_workgroups(x_groups, z_groups, 1);
    }

    pub fn upload_from_world(&self, queue: &wgpu::Queue, world: &sim3d::World) {
        if world.width != self.width as usize || world.depth != self.depth as usize {
            log::error!("World size mismatch in upload");
            return;
        }

        // Geology
        queue.write_buffer(
            &self.bedrock_buffer,
            0,
            bytemuck::cast_slice(&world.bedrock_elevation),
        );
        queue.write_buffer(
            &self.paydirt_buffer,
            0,
            bytemuck::cast_slice(&world.paydirt_thickness),
        );
        queue.write_buffer(
            &self.gravel_buffer,
            0,
            bytemuck::cast_slice(&world.gravel_thickness),
        );
        queue.write_buffer(
            &self.overburden_buffer,
            0,
            bytemuck::cast_slice(&world.overburden_thickness),
        );
        queue.write_buffer(
            &self.sediment_buffer,
            0,
            bytemuck::cast_slice(&world.terrain_sediment),
        );

        // Water
        // CAUTION: World stores Absolute Surface. GPU uses Depth.
        // We must calculate depth = max(0, surface - ground_height)
        let count = (self.width * self.depth) as usize;
        let mut depth_data = vec![0.0f32; count];

        for i in 0..count {
            let ground = world.bedrock_elevation[i]
                + world.paydirt_thickness[i]
                + world.gravel_thickness[i]
                + world.overburden_thickness[i]
                + world.terrain_sediment[i];
            depth_data[i] = (world.water_surface[i] - ground).max(0.0);
        }
        queue.write_buffer(
            &self.water_depth_buffer,
            0,
            bytemuck::cast_slice(&depth_data),
        );

        // Sync Water Surface as well (important for rendering if we don't wait for first dispatch)
        queue.write_buffer(
            &self.water_surface_buffer,
            0,
            bytemuck::cast_slice(&world.water_surface),
        );

        // Velocity
        // World uses staggered flow? flow_x[i] is at face?
        // Sim3d `water_flow_x` size is (W+1)*D.
        // Our GPU buffer is Cell-Centered W*D.
        // Rough approx: Take flux/flow and map to center?
        // For initialization, 0 is fine.
        // queue.write_buffer(&self.water_velocity_x_buffer, 0, ...);

        queue.write_buffer(
            &self.suspended_sediment_buffer,
            0,
            bytemuck::cast_slice(&world.suspended_sediment),
        );
        let zero_suspended = vec![0.0f32; count];
        queue.write_buffer(
            &self.suspended_overburden_buffer,
            0,
            bytemuck::cast_slice(&zero_suspended),
        );
        queue.write_buffer(
            &self.suspended_gravel_buffer,
            0,
            bytemuck::cast_slice(&zero_suspended),
        );
        queue.write_buffer(
            &self.suspended_paydirt_buffer,
            0,
            bytemuck::cast_slice(&zero_suspended),
        );
    }

    /// Upload only terrain buffers (for excavation) - does NOT touch water state
    pub fn upload_terrain_only(&self, queue: &wgpu::Queue, world: &sim3d::World) {
        if world.width != self.width as usize || world.depth != self.depth as usize {
            log::error!("World size mismatch in upload");
            return;
        }

        // Only geology - leave water/velocity untouched on GPU
        queue.write_buffer(
            &self.bedrock_buffer,
            0,
            bytemuck::cast_slice(&world.bedrock_elevation),
        );
        queue.write_buffer(
            &self.paydirt_buffer,
            0,
            bytemuck::cast_slice(&world.paydirt_thickness),
        );
        queue.write_buffer(
            &self.gravel_buffer,
            0,
            bytemuck::cast_slice(&world.gravel_thickness),
        );
        queue.write_buffer(
            &self.overburden_buffer,
            0,
            bytemuck::cast_slice(&world.overburden_thickness),
        );
        queue.write_buffer(
            &self.sediment_buffer,
            0,
            bytemuck::cast_slice(&world.terrain_sediment),
        );
    }

    pub async fn download_to_world(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &mut sim3d::World,
    ) {
        let size = (self.width * self.depth) as usize * std::mem::size_of::<f32>();

        // Helper to read buffer
        let read_buffer = |buffer: &wgpu::Buffer| -> Vec<f32> {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
            queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();
            result
        };

        // This is SLOW (doing it serially 5 times).
        // But "FPS is cooked" so this is likely faster than CPU sim.
        // Optimization: One large staging buffer and copy all regions into it?

        let bedrock = read_buffer(&self.bedrock_buffer);
        let paydirt = read_buffer(&self.paydirt_buffer);
        let gravel = read_buffer(&self.gravel_buffer);
        let overburden = read_buffer(&self.overburden_buffer);
        let sediment = read_buffer(&self.sediment_buffer);

        let water_depth = read_buffer(&self.water_depth_buffer);
        let suspended_sediment = read_buffer(&self.suspended_sediment_buffer);
        let suspended_overburden = read_buffer(&self.suspended_overburden_buffer);
        let suspended_gravel = read_buffer(&self.suspended_gravel_buffer);
        let suspended_paydirt = read_buffer(&self.suspended_paydirt_buffer);

        // Update World
        world.bedrock_elevation = bedrock;
        world.paydirt_thickness = paydirt;
        world.gravel_thickness = gravel;
        world.overburden_thickness = overburden;
        world.terrain_sediment = sediment;
        let mut suspended_total = suspended_sediment;
        for i in 0..suspended_total.len() {
            suspended_total[i] += suspended_overburden[i];
            suspended_total[i] += suspended_gravel[i];
            suspended_total[i] += suspended_paydirt[i];
        }
        world.suspended_sediment = suspended_total;

        // Calculate Surface
        for i in 0..water_depth.len() {
            let ground = world.bedrock_elevation[i]
                + world.paydirt_thickness[i]
                + world.gravel_thickness[i]
                + world.overburden_thickness[i]
                + world.terrain_sediment[i];
            world.water_surface[i] = ground + water_depth[i];
        }
    }
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        time: f32,
        draw_water: bool,
    ) {
        // Update Uniforms
        let uniforms = RenderUniforms {
            view_proj,
            camera_pos,
            cell_size: self.cell_size,
            grid_width: self.width,
            grid_depth: self.depth,
            time,
            _pad: 0,
        };
        queue.write_buffer(
            &self.render_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        // Render Pass
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Heightfield Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.render_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
        rpass.set_index_buffer(self.grid_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.num_indices, 0, 0..1);

        if draw_water {
            // Water
            rpass.set_pipeline(&self.water_pipeline);
            rpass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }

    pub fn set_bridge_buffers(
        &mut self,
        device: &wgpu::Device,
        sediment_transfer: &wgpu::Buffer,
        water_transfer: &wgpu::Buffer,
    ) {
        self.bridge_merge_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bridge Merge Bind Group"),
            layout: &self.bridge_merge_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sediment_transfer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: water_transfer.as_entire_binding(),
                },
            ],
        }));
    }

    pub fn dispatch_bridge_merge(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(bg) = &self.bridge_merge_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bridge Merge Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bridge_merge_pipeline);
            pass.set_bind_group(0, &self.params_bind_group, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.set_bind_group(2, &self.water_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(16);
            let workgroups_z = self.depth.div_ceil(16);
            pass.dispatch_workgroups(workgroups_x, workgroups_z, 1);
        }
    }
}
