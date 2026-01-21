//! Simulation dispatch methods for GPU heightfield.

use super::pipelines::{CoreBindGroups, SimulationPipelines};


/// Dispatch simulation for a specific tile region.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_simulation_tile(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SimulationPipelines,
    bind_groups: &CoreBindGroups,
    width: u32,
    depth: u32,
    tile_width: u32,
    tile_depth: u32,
    suspended_sediment_buffer: &wgpu::Buffer,
    suspended_sediment_next_buffer: &wgpu::Buffer,
    suspended_overburden_buffer: &wgpu::Buffer,
    suspended_overburden_next_buffer: &wgpu::Buffer,
    suspended_gravel_buffer: &wgpu::Buffer,
    suspended_gravel_next_buffer: &wgpu::Buffer,
    suspended_paydirt_buffer: &wgpu::Buffer,
    suspended_paydirt_next_buffer: &wgpu::Buffer,
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
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, &bind_groups.water, &[]);
            pass.set_bind_group(2, &bind_groups.terrain, &[]);
            pass.set_pipeline($pipeline);
            pass.dispatch_workgroups(x_groups, z_groups, 1);
        }};
    }

    // CRITICAL: Update Surface FIRST - flux needs water_surface to compute gradients!
    dispatch_step!("Update Surface", &pipelines.surface);

    // 1. Update Flux (Updates Velocity + Flux) - Reads Surface gradients
    dispatch_step!("Update Flux", &pipelines.flux);

    // 2. Update Depth (Volume Conservation) - Reads Flux, updates water_depth
    dispatch_step!("Update Depth", &pipelines.depth);

    // 3a. Settling (post-flux) - Reads Depth/Vel, writes Terrain/Suspended
    dispatch_step!("Update Settling", &pipelines.settling);

    // 3b. Erosion (post-settling) - Reads Depth/Vel/Terrain, writes Terrain/Suspended
    dispatch_step!("Update Erosion", &pipelines.erosion);

    // 4. Sediment Transport (flux-based advection)
    dispatch_step!("Update Sediment Transport", &pipelines.sediment_transport);

    // 4b. Copy suspended_sediment_next -> suspended_sediment (swap double buffers)
    let buffer_size = (width * depth) as u64 * std::mem::size_of::<f32>() as u64;
    encoder.copy_buffer_to_buffer(suspended_sediment_next_buffer, 0, suspended_sediment_buffer, 0, buffer_size);
    encoder.copy_buffer_to_buffer(suspended_overburden_next_buffer, 0, suspended_overburden_buffer, 0, buffer_size);
    encoder.copy_buffer_to_buffer(suspended_gravel_next_buffer, 0, suspended_gravel_buffer, 0, buffer_size);
    encoder.copy_buffer_to_buffer(suspended_paydirt_next_buffer, 0, suspended_paydirt_buffer, 0, buffer_size);

    // 5. Collapse (angle of repose / slope stability)
    // Use red-black pattern for race-free updates
    dispatch_step!("Update Collapse Red", &pipelines.collapse_red);
    dispatch_step!("Update Collapse Black", &pipelines.collapse_black);
}

/// Update simulation parameters buffer.
pub fn update_params(
    queue: &wgpu::Queue,
    params_buffer: &wgpu::Buffer,
    width: u32,
    depth: u32,
    cell_size: f32,
    dt: f32,
) {
    update_params_tile(queue, params_buffer, width, depth, cell_size, dt, 0, 0, width, depth);
}

/// Update simulation parameters for a specific tile.
#[allow(clippy::too_many_arguments)]
pub fn update_params_tile(
    queue: &wgpu::Queue,
    params_buffer: &wgpu::Buffer,
    width: u32,
    depth: u32,
    cell_size: f32,
    dt: f32,
    origin_x: u32,
    origin_z: u32,
    tile_width: u32,
    tile_depth: u32,
) {
    let params: [u32; 20] = [
        width,
        depth,
        tile_width,
        tile_depth,
        origin_x,
        origin_z,
        0,
        0,
        bytemuck::cast(cell_size),
        bytemuck::cast(dt),
        bytemuck::cast(9.81f32),
        bytemuck::cast(0.02f32),    // Manning's n coefficient
        bytemuck::cast(1000.0f32),  // rho_water
        bytemuck::cast(2650.0f32),  // rho_sediment
        bytemuck::cast(0.001f32),   // water_viscosity
        bytemuck::cast(0.045f32),   // critical_shields
        bytemuck::cast(0.01f32),    // k_erosion
        bytemuck::cast(0.05f32),    // max_erosion_per_step
        0,
        0,
    ];
    queue.write_buffer(params_buffer, 0, bytemuck::cast_slice(&params));
}
