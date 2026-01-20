// heightfield_bridge_merge.wgsl
// Merges atomic transfer buffers from 3D bridge into heightfield state.

struct Params {
    world_width: u32,
    world_depth: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read> transfer_sediment: array<i32>;
@group(1) @binding(1) var<storage, read> transfer_water: array<i32>;

@group(2) @binding(0) var<storage, read_write> sediment: array<f32>;
@group(2) @binding(1) var<storage, read_write> water_depth: array<f32>;

const SCALE: f32 = 100000.0;

@compute @workgroup_size(16, 16)
fn merge_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let z = id.y;
    if (x >= params.world_width || z >= params.world_depth) { return; }
    
    let idx = z * params.world_width + x;
    
    let s_val = f32(transfer_sediment[idx]) / SCALE;
    let w_val = f32(transfer_water[idx]) / SCALE;
    
    if (s_val > 0.0) {
        sediment[idx] += s_val;
    }
    if (w_val > 0.0) {
        water_depth[idx] += w_val;
    }
}
