// Sediment Cell Type Builder (3D)
//
// Overwrites the grid cell type buffer with sediment occupancy while preserving solids.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read> sediment_count: array<i32>;
@group(0) @binding(3) var<storage, read> water_count: array<i32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

// Jamming threshold: cells with this many sediment particles become solid
const SEDIMENT_JAM_THRESHOLD: i32 = 6;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(8, 8, 4)
fn build_sediment_cell_type(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    let ct = cell_type[idx];
    if (ct == CELL_SOLID) {
        cell_type[idx] = CELL_SOLID;
        return;
    }

    let sed_count = sediment_count[idx];
    let wat_count = water_count[idx];

    // Voxel-based jamming: treat heavily packed cells as solid obstacles
    if (sed_count >= SEDIMENT_JAM_THRESHOLD) {
        cell_type[idx] = CELL_SOLID;
    } else if (sed_count > 0 || wat_count > 0) {
        cell_type[idx] = CELL_FLUID;
    } else {
        cell_type[idx] = CELL_AIR;
    }
}
