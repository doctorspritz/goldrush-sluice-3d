// Fluid Cell Expansion Shader (3D)
//
// Conservative expansion: marks cells as FLUID only if:
// - The cell itself contains particles, OR
// - At least 2 face-adjacent neighbors contain particles
//
// This prevents over-expansion that causes volume inflation while still
// preventing gaps that cause volume collapse.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particle_count: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> cell_type: array<u32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

// Minimum neighbors with particles to expand into empty cell
const MIN_NEIGHBORS_FOR_EXPANSION: u32 = 3u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_particle_count(i: i32, j: i32, k: i32) -> i32 {
    if (i < 0 || i >= i32(params.width)) { return 0; }
    if (j < 0 || j >= i32(params.height)) { return 0; }
    if (k < 0 || k >= i32(params.depth)) { return 0; }

    let idx = cell_index(u32(i), u32(j), u32(k));
    return atomicLoad(&particle_count[idx]);
}

fn count_neighbors_with_particles(i: i32, j: i32, k: i32) -> u32 {
    var count: u32 = 0u;

    // Count 6 face neighbors with particles
    if (get_particle_count(i - 1, j, k) > 0) { count += 1u; }
    if (get_particle_count(i + 1, j, k) > 0) { count += 1u; }
    if (get_particle_count(i, j - 1, k) > 0) { count += 1u; }
    if (get_particle_count(i, j + 1, k) > 0) { count += 1u; }
    if (get_particle_count(i, j, k - 1) > 0) { count += 1u; }
    if (get_particle_count(i, j, k + 1) > 0) { count += 1u; }

    return count;
}

@compute @workgroup_size(8, 8, 4)
fn expand_fluid_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);

    // Preserve solid geometry (static obstacles, jammed sediment)
    if (cell_type[idx] == CELL_SOLID) {
        return;
    }

    // If this cell has particles, definitely FLUID
    if (get_particle_count(i32(i), i32(j), i32(k)) > 0) {
        cell_type[idx] = CELL_FLUID;
        return;
    }

    // Empty cell: only mark FLUID if surrounded by 2+ neighbors with particles
    // This prevents over-expansion at the fluid surface
    let neighbor_count = count_neighbors_with_particles(i32(i), i32(j), i32(k));
    if (neighbor_count >= MIN_NEIGHBORS_FOR_EXPANSION) {
        cell_type[idx] = CELL_FLUID;
    } else {
        cell_type[idx] = CELL_AIR;
    }
}
