// DEM Bin Count Shader - Phase 1 of spatial hashing
//
// Each particle atomically increments its bin counter.
// After this pass, bin_counts[i] contains number of particles in bin i.

struct Params {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bin_counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn bin_count(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let gi = clamp(u32(pos.x / params.cell_size), 0u, params.grid_width - 1u);
    let gj = clamp(u32(pos.y / params.cell_size), 0u, params.grid_height - 1u);
    let bin_idx = gj * params.grid_width + gi;

    atomicAdd(&bin_counts[bin_idx], 1u);
}
