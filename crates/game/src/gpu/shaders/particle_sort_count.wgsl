// Particle Sort Count Shader - Count particles per cell
//
// Input: cell_keys[i] = cell index for particle i
// Output: cell_counts[cell] = number of particles in that cell
//
// Uses atomics for counting

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn count_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let cell_idx = cell_keys[id.x];
    atomicAdd(&cell_counts[cell_idx], 1u);
}
