//! Emitter and material tool methods for GPU heightfield.

use super::pipelines::{EmitterResources, MaterialToolResources};

/// Update emitter parameters.
#[allow(clippy::too_many_arguments)]
pub fn update_emitter(
    queue: &wgpu::Queue,
    emitter: &EmitterResources,
    width: u32,
    depth: u32,
    cell_size: f32,
    pos_x: f32,
    pos_z: f32,
    radius: f32,
    rate: f32,
    sediment_conc: f32,
    overburden_conc: f32,
    gravel_conc: f32,
    paydirt_conc: f32,
    dt: f32,
    enabled: bool,
) {
    update_emitter_tile(
        queue, emitter, width, depth, cell_size,
        pos_x, pos_z, radius, rate,
        sediment_conc, overburden_conc, gravel_conc, paydirt_conc,
        dt, enabled, 0, 0, width, depth,
    );
}

/// Update emitter parameters for a specific tile.
#[allow(clippy::too_many_arguments)]
pub fn update_emitter_tile(
    queue: &wgpu::Queue,
    emitter: &EmitterResources,
    width: u32,
    depth: u32,
    cell_size: f32,
    pos_x: f32,
    pos_z: f32,
    radius: f32,
    rate: f32,
    sediment_conc: f32,
    overburden_conc: f32,
    gravel_conc: f32,
    paydirt_conc: f32,
    dt: f32,
    enabled: bool,
    origin_x: u32,
    origin_z: u32,
    tile_width: u32,
    tile_depth: u32,
) {
    let params: [u32; 20] = [
        bytemuck::cast(pos_x),
        bytemuck::cast(pos_z),
        bytemuck::cast(radius),
        bytemuck::cast(rate),
        bytemuck::cast(dt),
        if enabled { 1 } else { 0 },
        width,
        depth,
        tile_width,
        tile_depth,
        origin_x,
        origin_z,
        bytemuck::cast(cell_size),
        bytemuck::cast(sediment_conc),
        bytemuck::cast(overburden_conc),
        bytemuck::cast(gravel_conc),
        bytemuck::cast(paydirt_conc),
        0,
        0,
        0,
    ];
    queue.write_buffer(&emitter.params_buffer, 0, bytemuck::cast_slice(&params));
}

/// Dispatch emitter compute pass.
pub fn dispatch_emitter(
    encoder: &mut wgpu::CommandEncoder,
    emitter: &EmitterResources,
    width: u32,
    depth: u32,
) {
    dispatch_emitter_tile(encoder, emitter, width, depth);
}

/// Dispatch emitter for a specific tile.
pub fn dispatch_emitter_tile(
    encoder: &mut wgpu::CommandEncoder,
    emitter: &EmitterResources,
    tile_width: u32,
    tile_depth: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Emitter Compute Pass"),
        timestamp_writes: None,
    });

    let x_groups = tile_width.div_ceil(16);
    let z_groups = tile_depth.div_ceil(16);

    pass.set_pipeline(&emitter.pipeline);
    pass.set_bind_group(0, &emitter.bind_group, &[]);
    pass.dispatch_workgroups(x_groups, z_groups, 1);
}

/// Update material tool parameters.
#[allow(clippy::too_many_arguments)]
pub fn update_material_tool(
    queue: &wgpu::Queue,
    material_tool: &MaterialToolResources,
    width: u32,
    depth: u32,
    cell_size: f32,
    pos_x: f32,
    pos_z: f32,
    radius: f32,
    amount: f32,
    material_type: u32,
    dt: f32,
    enabled: bool,
) {
    update_material_tool_tile(
        queue, material_tool, width, depth, cell_size,
        pos_x, pos_z, radius, amount, material_type, dt, enabled,
        0, 0, width, depth,
    );
}

/// Update material tool parameters for a specific tile.
#[allow(clippy::too_many_arguments)]
pub fn update_material_tool_tile(
    queue: &wgpu::Queue,
    material_tool: &MaterialToolResources,
    width: u32,
    depth: u32,
    cell_size: f32,
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
        width,
        depth,
        tile_width,
        tile_depth,
        origin_x,
        origin_z,
        bytemuck::cast(cell_size),
        bytemuck::cast(dt),
        0,
        0,
    ];
    queue.write_buffer(&material_tool.params_buffer, 0, bytemuck::cast_slice(&params));
}

/// Dispatch material tool compute pass.
pub fn dispatch_material_tool(
    encoder: &mut wgpu::CommandEncoder,
    material_tool: &MaterialToolResources,
    width: u32,
    depth: u32,
) {
    dispatch_material_tool_tile(encoder, material_tool, width, depth);
}

/// Dispatch material tool for a specific tile.
pub fn dispatch_material_tool_tile(
    encoder: &mut wgpu::CommandEncoder,
    material_tool: &MaterialToolResources,
    tile_width: u32,
    tile_depth: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Material Tool Compute Pass"),
        timestamp_writes: None,
    });

    let x_groups = tile_width.div_ceil(16);
    let z_groups = tile_depth.div_ceil(16);

    pass.set_pipeline(&material_tool.pipeline);
    pass.set_bind_group(0, &material_tool.bind_group, &[]);
    pass.set_bind_group(1, &material_tool.terrain_bind_group, &[]);
    pass.dispatch_workgroups(x_groups, z_groups, 1);
}

/// Dispatch excavation compute pass.
pub fn dispatch_excavate(
    encoder: &mut wgpu::CommandEncoder,
    material_tool: &MaterialToolResources,
    width: u32,
    depth: u32,
) {
    dispatch_excavate_tile(encoder, material_tool, width, depth);
}

/// Dispatch excavation for a specific tile.
pub fn dispatch_excavate_tile(
    encoder: &mut wgpu::CommandEncoder,
    material_tool: &MaterialToolResources,
    tile_width: u32,
    tile_depth: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Excavate Compute Pass"),
        timestamp_writes: None,
    });

    let x_groups = tile_width.div_ceil(16);
    let z_groups = tile_depth.div_ceil(16);

    pass.set_pipeline(&material_tool.excavate_pipeline);
    pass.set_bind_group(0, &material_tool.bind_group, &[]);
    pass.set_bind_group(1, &material_tool.terrain_bind_group, &[]);
    pass.dispatch_workgroups(x_groups, z_groups, 1);
}
