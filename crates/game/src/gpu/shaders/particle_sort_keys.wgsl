// Particle Sort Keys Shader - Compute cell index for each particle
//
// Output: cell_keys[i] = cell index for particle i
// Used as the first pass in counting sort for spatial coherence

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
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> cell_keys: array<u32>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(256)
fn compute_keys(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let cell_size = params.cell_size;

    // Compute cell indices with clamping
    let ci = u32(clamp(i32(pos.x / cell_size), 0, i32(params.width) - 1));
    let cj = u32(clamp(i32(pos.y / cell_size), 0, i32(params.height) - 1));
    let ck = u32(clamp(i32(pos.z / cell_size), 0, i32(params.depth) - 1));

    cell_keys[id.x] = cell_index(ci, cj, ck);
}
