//! Rendering resources and methods for GPU heightfield.

use super::buffers::{GeologyBuffers, WaterBuffers};
use super::mesh::{GridMesh, GridVertex};

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

/// Rendering resources for heightfield visualization.
pub struct RenderResources {
    pub terrain_pipeline: wgpu::RenderPipeline,
    pub water_pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

/// Create a storage buffer bind group layout entry for rendering (read-only).
fn render_storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create render resources.
pub fn create_render_resources(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    geology: &GeologyBuffers,
    water: &WaterBuffers,
) -> RenderResources {
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Render Uniform Buffer"),
        size: std::mem::size_of::<RenderUniforms>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Render Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vf,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            render_storage_entry(1, vf),  // bedrock
            render_storage_entry(2, vf),  // paydirt
            render_storage_entry(3, vf),  // gravel
            render_storage_entry(4, vf),  // overburden
            render_storage_entry(5, vf),  // sediment
            render_storage_entry(6, vf),  // water_surface
            render_storage_entry(7, vf),  // water_depth
            render_storage_entry(8, wgpu::ShaderStages::FRAGMENT),  // surface_material
            render_storage_entry(9, wgpu::ShaderStages::VERTEX),    // suspended_sediment
            render_storage_entry(10, wgpu::ShaderStages::VERTEX),   // suspended_overburden
            render_storage_entry(11, wgpu::ShaderStages::VERTEX),   // suspended_gravel
            render_storage_entry(12, wgpu::ShaderStages::VERTEX),   // suspended_paydirt
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Render Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: geology.bedrock.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: geology.paydirt.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: geology.gravel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: geology.overburden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: geology.sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: water.surface.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: water.depth.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: geology.surface_material.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: water.suspended_sediment.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: water.suspended_overburden.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: water.suspended_gravel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: water.suspended_paydirt.as_entire_binding() },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Heightfield Render Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/heightfield_render.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let terrain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Terrain Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[GridVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
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
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_water"),
            buffers: &[GridVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
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

    RenderResources {
        terrain_pipeline,
        water_pipeline,
        bind_group,
        uniform_buffer,
    }
}

/// Render the heightfield.
#[allow(clippy::too_many_arguments)]
pub fn render(
    encoder: &mut wgpu::CommandEncoder,
    render: &RenderResources,
    mesh: &GridMesh,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    queue: &wgpu::Queue,
    width: u32,
    depth: u32,
    cell_size: f32,
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    draw_water: bool,
) {
    let uniforms = RenderUniforms {
        view_proj,
        camera_pos,
        cell_size,
        grid_width: width,
        grid_depth: depth,
        time,
        _pad: 0,
    };
    queue.write_buffer(&render.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

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

    rpass.set_pipeline(&render.terrain_pipeline);
    rpass.set_bind_group(0, &render.bind_group, &[]);
    rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);

    if draw_water {
        rpass.set_pipeline(&render.water_pipeline);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }
}
