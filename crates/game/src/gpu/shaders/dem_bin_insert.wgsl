// DEM Bin Insert Shader - Phase 2 of spatial hashing
//
// After CPU computes prefix sum (bin_offsets), this shader inserts each
// particle into the sorted_indices array at its correct position.
//
// Uses atomicAdd on bin_counts to get unique slot within each bin.
// bin_counts is reset to 0 before this pass (or we use a separate counter).

struct Params {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> bin_counters: array<atomic<u32>>; // Reset to 0 before dispatch
@group(0) @binding(4) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size(256)
fn bin_insert(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let gi = clamp(u32(pos.x / params.cell_size), 0u, params.grid_width - 1u);
    let gj = clamp(u32(pos.y / params.cell_size), 0u, params.grid_height - 1u);
    let bin_idx = gj * params.grid_width + gi;

    // Get unique index within this bin
    let local_idx = atomicAdd(&bin_counters[bin_idx], 1u);

    // Compute global position in sorted array
    let global_idx = bin_offsets[bin_idx] + local_idx;

    // Store particle index
    sorted_indices[global_idx] = id.x;
}
