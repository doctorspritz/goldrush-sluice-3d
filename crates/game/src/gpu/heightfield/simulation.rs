//! Simulation dispatch methods for GPU heightfield.

use super::buffers::WaterBuffers;
use super::pipelines::{CoreBindGroups, SimulationPipelines};

#[derive(Clone, Copy)]
pub enum BedExchangeStage {
    Settling,
    Erosion,
}

fn copy_concentrations(
    encoder: &mut wgpu::CommandEncoder,
    water: &WaterBuffers,
    size: u64,
    to_next: bool,
) {
    for (current, next) in [
        (&water.suspended_sediment, &water.suspended_sediment_next),
        (
            &water.suspended_overburden,
            &water.suspended_overburden_next,
        ),
        (&water.suspended_gravel, &water.suspended_gravel_next),
        (&water.suspended_paydirt, &water.suspended_paydirt_next),
    ] {
        let (source, destination) = if to_next {
            (current, next)
        } else {
            (next, current)
        };
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
    }
}

/// Snapshot immutable geometry, store the requested bed change, then reconcile
/// suspension against the actual stored f32 bed delta in a separate pass.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_bed_exchange_tile(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SimulationPipelines,
    bind_groups: &CoreBindGroups,
    water: &WaterBuffers,
    cells: u32,
    tile_width: u32,
    tile_depth: u32,
    stage: BedExchangeStage,
) {
    dispatch_bed_exchange_passes(
        encoder,
        pipelines,
        bind_groups,
        tile_width,
        tile_depth,
        stage,
    );
    // Release the phase-local old-layer aliases, including inactive cells.
    copy_concentrations(encoder, water, cells as u64 * 4, true);
}

fn dispatch_bed_exchange_passes(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SimulationPipelines,
    bind_groups: &CoreBindGroups,
    tile_width: u32,
    tile_depth: u32,
    stage: BedExchangeStage,
) {
    let (candidate, reconcile) = match stage {
        BedExchangeStage::Settling => (&pipelines.settling, &pipelines.settling_reconcile),
        BedExchangeStage::Erosion => (&pipelines.erosion, &pipelines.erosion_reconcile),
    };
    for (pipeline, label) in [
        (&pipelines.bed_snapshot, "Snapshot Bed Geometry"),
        (candidate, "Store Candidate Bed Exchange"),
        (reconcile, "Reconcile Stored Bed Exchange"),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_groups.params, &[]);
        pass.set_bind_group(1, &bind_groups.water, &[]);
        pass.set_bind_group(2, &bind_groups.terrain, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(tile_width.div_ceil(16), tile_depth.div_ceil(16), 1);
    }
}

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
    water_substeps: u32,
    water: &WaterBuffers,
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

    let buffer_size = (width * depth) as u64 * std::mem::size_of::<f32>() as u64;
    encoder.clear_buffer(
        &water.suspended_sediment_next,
        buffer_size,
        Some(buffer_size * super::buffers::WATER_AUX_FLOATS as u64),
    );
    // Transport writes only the active tile but copies the complete target.
    // Preserve CPU edits and concentrations outside moving tile boundaries.
    copy_concentrations(encoder, water, buffer_size, true);
    for _ in 0..water_substeps.max(1) {
        dispatch_step!("Update Surface", &pipelines.surface);
        dispatch_step!("Update Flux", &pipelines.flux);
        dispatch_step!("Aggregate Donor Budget", &pipelines.donor_budget);
        dispatch_step!("Limit Shared Flux", &pipelines.limit_flux);
        dispatch_step!("Update Depth", &pipelines.depth);
        dispatch_step!(
            "Conservative Sediment Transport",
            &pipelines.sediment_transport
        );
        copy_concentrations(encoder, water, buffer_size, false);
    }
    // Geology evolves over the full step, after conservative water/material advection.
    for stage in [BedExchangeStage::Settling, BedExchangeStage::Erosion] {
        dispatch_bed_exchange_passes(
            encoder,
            pipelines,
            bind_groups,
            tile_width,
            tile_depth,
            stage,
        );
    }
    // No transport runs between the two bed stages. The second snapshot
    // replaces the first alias; restore the public targets only once at exit.
    copy_concentrations(encoder, water, buffer_size, true);

    dispatch_collapse_tile(encoder, pipelines, bind_groups, tile_width, tile_depth);
}

/// Snapshot donor transfers followed by a gather into each receiver, including
/// a one-cell tile halo. Every cell has exactly one writer in each pass.
pub fn dispatch_collapse_tile(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SimulationPipelines,
    bind_groups: &CoreBindGroups,
    tile_width: u32,
    tile_depth: u32,
) {
    for (pipeline, label, width, depth) in [
        (
            &pipelines.collapse_flux,
            "Collapse Donor Transfers",
            tile_width,
            tile_depth,
        ),
        (
            &pipelines.collapse_apply,
            "Gather Collapse Transfers",
            tile_width + 2,
            tile_depth + 2,
        ),
    ] {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_groups.params, &[]);
        pass.set_bind_group(1, &bind_groups.collapse, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(width.div_ceil(16), depth.div_ceil(16), 1);
    }
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
    update_params_tile(
        queue,
        params_buffer,
        width,
        depth,
        cell_size,
        dt,
        0,
        0,
        width,
        depth,
    );
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
        2, // legacy standalone caller: +X outflow; game configures its river explicitly
        2, // must match the default dispatch substep count
        bytemuck::cast(cell_size),
        bytemuck::cast(dt),
        bytemuck::cast(9.81f32),
        bytemuck::cast(0.02f32),   // Manning's n coefficient
        bytemuck::cast(1000.0f32), // rho_water
        bytemuck::cast(2650.0f32), // rho_sediment
        bytemuck::cast(0.001f32),  // water_viscosity
        bytemuck::cast(0.045f32),  // critical_shields
        bytemuck::cast(0.01f32),   // k_erosion
        bytemuck::cast(0.05f32),   // max_erosion_per_step
        0,
        0,
    ];
    queue.write_buffer(params_buffer, 0, bytemuck::cast_slice(&params));
}
