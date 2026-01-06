// Sediment Density Error Shader (3D)
//
// Computes density error from sediment particle count per cell.
// This pass is repulsive-only (crowding) to emulate granular collision.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    rest_density: f32,
    dt: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sediment_count: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> density_error: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_AIR; }
    if (j < 0 || j >= i32(params.height)) { return CELL_AIR; }
    if (k < 0 || k >= i32(params.depth)) { return CELL_AIR; }
    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

@compute @workgroup_size(8, 8, 4)
fn compute_sediment_density_error(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    if (cell_type[idx] != CELL_FLUID) {
        density_error[idx] = 0.0;
        return;
    }

    var density = f32(max(atomicLoad(&sediment_count[idx]), 0));

    let solid_neighbor_contribution: f32 = 0.5625;
    if (get_cell_type(i32(i) + 1, i32(j), i32(k)) == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (get_cell_type(i32(i) - 1, i32(j), i32(k)) == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (get_cell_type(i32(i), i32(j) + 1, i32(k)) == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (get_cell_type(i32(i), i32(j) - 1, i32(k)) == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (get_cell_type(i32(i), i32(j), i32(k) + 1) == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (get_cell_type(i32(i), i32(j), i32(k) - 1) == CELL_SOLID) { density += solid_neighbor_contribution; }

    // Negative error = crowded. Only apply repulsion (no suction into empty space).
    var error = 1.0 - density / params.rest_density;
    error = min(0.0, error);
    error = clamp(error, -0.5, 0.0);
    error /= params.dt;

    density_error[idx] = error;
}
