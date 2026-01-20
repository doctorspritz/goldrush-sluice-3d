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

    // Water - calculate depth from surface
    let count = (width * depth) as usize;
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

    let read_buffer = |buffer: &wgpu::Buffer| -> Vec<f32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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

    let bedrock = read_buffer(&geology.bedrock);
    let paydirt = read_buffer(&geology.paydirt);
    let gravel = read_buffer(&geology.gravel);
    let overburden = read_buffer(&geology.overburden);
    let sediment = read_buffer(&geology.sediment);

    let water_depth = read_buffer(&water.depth);
    let suspended_sediment = read_buffer(&water.suspended_sediment);
    let suspended_overburden = read_buffer(&water.suspended_overburden);
    let suspended_gravel = read_buffer(&water.suspended_gravel);
    let suspended_paydirt = read_buffer(&water.suspended_paydirt);

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
