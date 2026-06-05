//! World synchronization methods for GPU heightfield.

use super::buffers::{GeologyBuffers, WaterBuffers};
use super::pipelines::{BridgeMergeResources, CoreBindGroups};

/// Upload world data to GPU buffers.
pub fn upload_from_world(
    queue: &wgpu::Queue,
    geology: &GeologyBuffers,
    water: &WaterBuffers,
    width: u32,
    depth: u32,
    world: &sim3d::World,
) {
    if world.width != width as usize || world.depth != depth as usize {
        log::error!("World size mismatch in upload");
        return;
    }

    // Geology
    queue.write_buffer(&geology.bedrock, 0, bytemuck::cast_slice(&world.bedrock_elevation));
    queue.write_buffer(&geology.paydirt, 0, bytemuck::cast_slice(&world.paydirt_thickness));
    queue.write_buffer(&geology.gravel, 0, bytemuck::cast_slice(&world.gravel_thickness));
    queue.write_buffer(&geology.overburden, 0, bytemuck::cast_slice(&world.overburden_thickness));
    queue.write_buffer(&geology.sediment, 0, bytemuck::cast_slice(&world.terrain_sediment));

    // Seed surface_material so terrain shows real geology colors before any
    // erosion/collapse pass runs (those only recompute changed cells).
    // Priority MUST match compute_surface_material() in heightfield_collapse.wgsl:49
    // and heightfield_erosion.wgsl:102.  4=sediment 3=overburden 2=gravel 1=paydirt 0=bedrock
    let count = (width * depth) as usize;
    let mut surface_mat = vec![0u32; count];
    let min_thick = 0.001f32;
    for i in 0..count {
        surface_mat[i] = if world.terrain_sediment[i] > min_thick {
            4u32
        } else if world.overburden_thickness[i] > min_thick {
            3u32
        } else if world.gravel_thickness[i] > min_thick {
            2u32
        } else if world.paydirt_thickness[i] > min_thick {
            1u32
        } else {
            0u32
        };
    }
    queue.write_buffer(&geology.surface_material, 0, bytemuck::cast_slice(&surface_mat));

    // Water - calculate depth from surface
    let mut depth_data = vec![0.0f32; count];

    for i in 0..count {
        let ground = world.bedrock_elevation[i]
            + world.paydirt_thickness[i]
            + world.gravel_thickness[i]
            + world.overburden_thickness[i]
            + world.terrain_sediment[i];
        depth_data[i] = (world.water_surface[i] - ground).max(0.0);
    }
    queue.write_buffer(&water.depth, 0, bytemuck::cast_slice(&depth_data));
    queue.write_buffer(&water.surface, 0, bytemuck::cast_slice(&world.water_surface));

    // Suspended sediment
    queue.write_buffer(&water.suspended_sediment, 0, bytemuck::cast_slice(&world.suspended_sediment));

    let zero_suspended = vec![0.0f32; count];
    queue.write_buffer(&water.suspended_overburden, 0, bytemuck::cast_slice(&zero_suspended));
    queue.write_buffer(&water.suspended_gravel, 0, bytemuck::cast_slice(&zero_suspended));
    queue.write_buffer(&water.suspended_paydirt, 0, bytemuck::cast_slice(&zero_suspended));
}

/// Upload only terrain buffers (for excavation) - does NOT touch water state.
pub fn upload_terrain_only(
    queue: &wgpu::Queue,
    geology: &GeologyBuffers,
    width: u32,
    depth: u32,
    world: &sim3d::World,
) {
    if world.width != width as usize || world.depth != depth as usize {
        log::error!("World size mismatch in upload");
        return;
    }

    queue.write_buffer(&geology.bedrock, 0, bytemuck::cast_slice(&world.bedrock_elevation));
    queue.write_buffer(&geology.paydirt, 0, bytemuck::cast_slice(&world.paydirt_thickness));
    queue.write_buffer(&geology.gravel, 0, bytemuck::cast_slice(&world.gravel_thickness));
    queue.write_buffer(&geology.overburden, 0, bytemuck::cast_slice(&world.overburden_thickness));
    queue.write_buffer(&geology.sediment, 0, bytemuck::cast_slice(&world.terrain_sediment));
}

/// Download GPU state to world.
pub async fn download_to_world(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    geology: &GeologyBuffers,
    water: &WaterBuffers,
    width: u32,
    depth: u32,
    world: &mut sim3d::World,
) {
    let size = (width * depth) as usize * std::mem::size_of::<f32>();

    // Batch all 10 buffer reads into a SINGLE command submission and a SINGLE
    // device.poll(Wait). Previously each read_buffer() did its own submit + poll,
    // i.e. 10 full GPU stalls per frame; this collapses that to one.
    let src_buffers: [&wgpu::Buffer; 10] = [
        &geology.bedrock,
        &geology.paydirt,
        &geology.gravel,
        &geology.overburden,
        &geology.sediment,
        &water.depth,
        &water.suspended_sediment,
        &water.suspended_overburden,
        &water.suspended_gravel,
        &water.suspended_paydirt,
    ];

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let stagings: Vec<wgpu::Buffer> = src_buffers
        .iter()
        .map(|buffer| {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
            staging
        })
        .collect();
    queue.submit(Some(encoder.finish()));

    // Kick off all map_async requests, then wait once.
    let mut receivers = Vec::with_capacity(stagings.len());
    for staging in &stagings {
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        receivers.push(rx);
    }
    device.poll(wgpu::Maintain::Wait);

    let mut results: Vec<Vec<f32>> = Vec::with_capacity(stagings.len());
    for (staging, rx) in stagings.iter().zip(receivers) {
        rx.recv().unwrap().unwrap();
        let data = staging.slice(..).get_mapped_range();
        results.push(bytemuck::cast_slice(&data).to_vec());
        drop(data);
        staging.unmap();
    }

    let mut it = results.into_iter();
    let bedrock = it.next().unwrap();
    let paydirt = it.next().unwrap();
    let gravel = it.next().unwrap();
    let overburden = it.next().unwrap();
    let sediment = it.next().unwrap();
    let water_depth = it.next().unwrap();
    let suspended_sediment = it.next().unwrap();
    let suspended_overburden = it.next().unwrap();
    let suspended_gravel = it.next().unwrap();
    let suspended_paydirt = it.next().unwrap();

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

    for i in 0..water_depth.len() {
        let ground = world.bedrock_elevation[i]
            + world.paydirt_thickness[i]
            + world.gravel_thickness[i]
            + world.overburden_thickness[i]
            + world.terrain_sediment[i];
        world.water_surface[i] = ground + water_depth[i];
    }
}

/// Set bridge buffers for particle merging.
pub fn set_bridge_buffers(
    device: &wgpu::Device,
    bridge: &mut BridgeMergeResources,
    sediment_transfer: &wgpu::Buffer,
    water_transfer: &wgpu::Buffer,
) {
    bridge.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bridge Merge Bind Group"),
        layout: &bridge.bg_layout,
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

/// Dispatch bridge merge pass.
pub fn dispatch_bridge_merge(
    encoder: &mut wgpu::CommandEncoder,
    bridge: &BridgeMergeResources,
    bind_groups: &CoreBindGroups,
    width: u32,
    depth: u32,
) {
    if let Some(bg) = &bridge.bind_group {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bridge Merge Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&bridge.pipeline);
        pass.set_bind_group(0, &bind_groups.params, &[]);
        pass.set_bind_group(1, bg, &[]);
        pass.set_bind_group(2, &bind_groups.water, &[]);
        let workgroups_x = width.div_ceil(16);
        let workgroups_z = depth.div_ceil(16);
        pass.dispatch_workgroups(workgroups_x, workgroups_z, 1);
    }
}
