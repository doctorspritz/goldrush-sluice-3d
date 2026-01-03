// DEM Bin Clear Shader - Reset bin counts to zero
//
// Run before bin_count to prepare for new frame.

struct Params {
    grid_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<u32>;

@compute @workgroup_size(256)
fn bin_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.grid_size) {
        return;
    }
    bin_counts[id.x] = 0u;
}
