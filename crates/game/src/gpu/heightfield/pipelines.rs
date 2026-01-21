//! Pipeline and bind group creation for GPU heightfield simulation.

use super::buffers::{GeologyBuffers, WaterBuffers};

/// Simulation compute pipelines.
pub struct SimulationPipelines {
    pub surface: wgpu::ComputePipeline,
    pub flux: wgpu::ComputePipeline,
    pub depth: wgpu::ComputePipeline,
    pub settling: wgpu::ComputePipeline,
    pub erosion: wgpu::ComputePipeline,
    pub sediment_transport: wgpu::ComputePipeline,
    pub collapse: wgpu::ComputePipeline,
    pub collapse_red: wgpu::ComputePipeline,
    pub collapse_black: wgpu::ComputePipeline,
}

/// Emitter pipeline and resources.
pub struct EmitterResources {
    pub pipeline: wgpu::ComputePipeline,
    pub params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// Material tool pipeline and resources.
pub struct MaterialToolResources {
    pub pipeline: wgpu::ComputePipeline,
    pub excavate_pipeline: wgpu::ComputePipeline,
    pub params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub terrain_bind_group: wgpu::BindGroup,
}

/// Bridge merge pipeline and resources.
pub struct BridgeMergeResources {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: Option<wgpu::BindGroup>,
    pub bg_layout: wgpu::BindGroupLayout,
}

/// Core bind groups for simulation.
pub struct CoreBindGroups {
    pub params: wgpu::BindGroup,
    pub water: wgpu::BindGroup,
    pub terrain: wgpu::BindGroup,
}

/// Create the params bind group layout.
pub fn create_params_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

/// Create a storage buffer bind group layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create the water state bind group layout.
pub fn create_water_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Water State Layout"),
        entries: &[
            storage_entry(0, false),  // depth
            storage_entry(1, false),  // vel_x
            storage_entry(2, false),  // vel_z
            storage_entry(3, false),  // surface
            storage_entry(4, false),  // flux_x
            storage_entry(5, false),  // flux_z
            storage_entry(6, false),  // suspended_sediment
            storage_entry(7, false),  // suspended_sediment_next
            storage_entry(8, false),  // suspended_overburden
            storage_entry(9, false),  // suspended_overburden_next
            storage_entry(10, false), // suspended_gravel
            storage_entry(11, false), // suspended_gravel_next
            storage_entry(12, false), // suspended_paydirt
            storage_entry(13, false), // suspended_paydirt_next
        ],
    })
}

/// Create the terrain bind group layout.
pub fn create_terrain_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Terrain Layout"),
        entries: &[
            storage_entry(0, false), // bedrock
            storage_entry(1, false), // paydirt
            storage_entry(2, false), // gravel
            storage_entry(3, false), // overburden
            storage_entry(4, false), // sediment
            storage_entry(5, false), // surface_material
            storage_entry(6, false), // settling_time
            storage_entry(7, false), // debug_stats
        ],
    })
}

/// Create params bind group.
pub fn create_params_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Heightfield Params Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }],
    })
}

/// Create water bind group.
pub fn create_water_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    water: &WaterBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Water Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: water.depth.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: water.velocity_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: water.velocity_z.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: water.surface.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: water.flux_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: water.flux_z.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: water.suspended_sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: water.suspended_sediment_next.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: water.suspended_overburden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: water.suspended_overburden_next.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: water.suspended_gravel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: water.suspended_gravel_next.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: water.suspended_paydirt.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: water.suspended_paydirt_next.as_entire_binding() },
        ],
    })
}

/// Create terrain bind group.
pub fn create_terrain_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    geology: &GeologyBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Terrain Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: geology.bedrock.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: geology.paydirt.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: geology.gravel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: geology.overburden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: geology.sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: geology.surface_material.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: geology.settling_time.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: geology.debug_stats.as_entire_binding() },
        ],
    })
}

/// Create all simulation pipelines.
pub fn create_simulation_pipelines(
    device: &wgpu::Device,
    params_layout: &wgpu::BindGroupLayout,
    water_layout: &wgpu::BindGroupLayout,
    terrain_layout: &wgpu::BindGroupLayout,
) -> SimulationPipelines {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Heightfield Pipeline Layout"),
        bind_group_layouts: &[params_layout, water_layout, terrain_layout],
        push_constant_ranges: &[],
    });

    // Water shader
    let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heightfield Water Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_water.wgsl").into()),
    });

    // Erosion shader
    let erosion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heightfield Erosion Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_erosion.wgsl").into()),
    });

    // Collapse shader
    let collapse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heightfield Collapse Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_collapse.wgsl").into()),
    });

    let create_compute = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    SimulationPipelines {
        surface: create_compute("Update Surface Pipeline", &water_shader, "update_surface"),
        flux: create_compute("Update Flux Pipeline", &water_shader, "update_flux"),
        depth: create_compute("Update Depth Pipeline", &water_shader, "update_depth"),
        settling: create_compute("Update Settling Pipeline", &erosion_shader, "update_settling"),
        erosion: create_compute("Update Erosion Pipeline", &erosion_shader, "update_erosion"),
        sediment_transport: create_compute("Sediment Transport Pipeline", &erosion_shader, "update_sediment_transport"),
        collapse: create_compute("Collapse Pipeline", &collapse_shader, "update_collapse"),
        collapse_red: create_compute("Collapse Red Pipeline", &collapse_shader, "update_collapse_red"),
        collapse_black: create_compute("Collapse Black Pipeline", &collapse_shader, "update_collapse_black"),
    }
}

/// Create emitter resources.
pub fn create_emitter_resources(
    device: &wgpu::Device,
    water: &WaterBuffers,
) -> EmitterResources {
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Emitter Params Buffer"),
        size: 80,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Emitter Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: water.depth.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: water.suspended_sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: water.suspended_overburden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: water.suspended_gravel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: water.suspended_paydirt.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: water.velocity_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: water.velocity_z.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Emitter Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heightfield Emitter Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_emitter.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Emitter Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("add_water"),
        compilation_options: Default::default(),
        cache: None,
    });

    EmitterResources {
        pipeline,
        params_buffer,
        bind_group,
    }
}

/// Create material tool resources.
pub fn create_material_tool_resources(
    device: &wgpu::Device,
    geology: &GeologyBuffers,
) -> MaterialToolResources {
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Material Tool Params Buffer"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let terrain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Material Tool Terrain Layout"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, false),
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Material Tool Bind Group"),
        layout: &params_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }],
    });

    let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Material Tool Terrain Bind Group"),
        layout: &terrain_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: geology.bedrock.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: geology.paydirt.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: geology.gravel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: geology.overburden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: geology.sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: geology.surface_material.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Material Tool Pipeline Layout"),
        bind_group_layouts: &[&params_layout, &terrain_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heightfield Material Tool Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_material_tool.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Material Tool Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("apply_material_tool"),
        compilation_options: Default::default(),
        cache: None,
    });

    let excavate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Excavate Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("excavate"),
        compilation_options: Default::default(),
        cache: None,
    });

    MaterialToolResources {
        pipeline,
        excavate_pipeline,
        params_buffer,
        bind_group,
        terrain_bind_group,
    }
}

/// Create bridge merge resources.
pub fn create_bridge_merge_resources(
    device: &wgpu::Device,
    params_layout: &wgpu::BindGroupLayout,
    water_layout: &wgpu::BindGroupLayout,
) -> BridgeMergeResources {
    let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bridge Merge Layout"),
        bind_group_layouts: &[params_layout, &bg_layout, water_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bridge Merge Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_bridge_merge.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bridge Merge Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("merge_particles"),
        compilation_options: Default::default(),
        cache: None,
    });

    BridgeMergeResources {
        pipeline,
        bind_group: None,
        bg_layout,
    }
}
