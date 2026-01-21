//! Heightfield SDF generation utilities.

use super::pipelines::{CoreBindGroups, SdfResources};

pub fn update_sdf_params(
    queue: &wgpu::Queue,
    sdf: &SdfResources,
    width: u32,
    depth: u32,
    height: u32,
    cell_size: f32,
) {
    let params: [u32; 8] = [
        width,
        height,
        depth,
        0,
        bytemuck::cast(cell_size),
        0,
        0,
        0,
    ];
    queue.write_buffer(&sdf.params_buffer, 0, bytemuck::cast_slice(&params));
}

pub fn dispatch_sdf(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    sdf: &SdfResources,
    bind_groups: &CoreBindGroups,
    sdf_buffer: &wgpu::Buffer,
    width: u32,
    depth: u32,
    height: u32,
) {
    let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Heightfield SDF Output Bind Group"),
        layout: &sdf.output_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: sdf_buffer.as_entire_binding(),
        }],
    });

    let workgroups_x = width.div_ceil(8);
    let workgroups_y = height.div_ceil(8);
    let workgroups_z = depth.div_ceil(8);

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Heightfield SDF Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&sdf.pipeline);
    pass.set_bind_group(0, &sdf.params_bind_group, &[]);
    pass.set_bind_group(1, &bind_groups.terrain, &[]);
    pass.set_bind_group(2, &output_bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
}
