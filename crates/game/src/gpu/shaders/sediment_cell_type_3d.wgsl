// Sediment Cell Type Builder (3D)
//
// Overwrites the grid cell type buffer with sediment occupancy while preserving solids.
// Jamming only occurs when sediment has support below (ground or other jammed cells).

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read> sediment_count: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read> particle_count: array<atomic<i32>>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

// Jamming occurs when sediment dominates AND cell has support
const MIN_SEDIMENT_FOR_JAM: i32 = 3;  // Minimum sediment particles needed

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_SOLID; }
    if (j < 0) { return CELL_SOLID; }  // Ground
    if (j >= i32(params.height)) { return CELL_AIR; }
    if (k < 0 || k >= i32(params.depth)) { return CELL_SOLID; }
    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

fn has_support_below(i: i32, j: i32, k: i32) -> bool {
    // Check if cell has solid support below (ground, static geometry, or jammed sediment)
    if (j == 0) { return true; }  // Ground level

    let below = get_cell_type(i, j - 1, k);
    return below == CELL_SOLID;
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

    // Preserve static solid geometry
    if (ct == CELL_SOLID) {
        cell_type[idx] = CELL_SOLID;
        return;
    }

    let sed_count = atomicLoad(&sediment_count[idx]);
    let total_count = atomicLoad(&particle_count[idx]);
    let wat_count = total_count - sed_count;

    // Voxel-based jamming: only jam if:
    // 1. Cell has minimum sediment particles
    // 2. Sediment dominates over water (more sediment than water)
    // 3. Cell has solid support below it
    let has_enough_sediment = sed_count >= MIN_SEDIMENT_FOR_JAM;
    let sediment_dominates = sed_count > wat_count;
    let is_supported = has_support_below(i32(i), i32(j), i32(k));

    if (has_enough_sediment && sediment_dominates && is_supported) {
        cell_type[idx] = CELL_SOLID;
    } else if (sed_count > 0 || wat_count > 0) {
        cell_type[idx] = CELL_FLUID;
    } else {
        cell_type[idx] = CELL_AIR;
    }
}
