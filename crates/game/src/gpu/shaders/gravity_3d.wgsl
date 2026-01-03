// Gravity Application Shader (3D)
//
// Applies gravity to U (horizontal) and V (vertical) velocity components.
// Supports tilted gravity for sluice slope simulation.
// Only affects faces adjacent to at least one fluid cell.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    gravity_y_dt: f32,  // gravity.y * dt (typically -9.81 * dt)
    gravity_x_dt: f32,  // gravity.x * dt (slope component, e.g. 1.0 * dt)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_u: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

// Get cell type with proper boundary handling for sluice:
// Both inlet and outlet are OPEN for flow-through
fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    // CLOSED boundaries: floor and side walls
    if (j < 0) { return CELL_SOLID; }  // Floor (closed)
    if (k < 0) { return CELL_SOLID; }  // Side wall (closed)
    if (u32(k) >= params.depth) { return CELL_SOLID; }  // Side wall (closed)

    // OPEN boundaries: inlet, outlet, and top → treat as AIR
    if (i < 0) { return CELL_AIR; }  // Inlet (OPEN for sluice flow!)
    if (u32(i) >= params.width) { return CELL_AIR; }  // Outlet (OPEN!)
    if (u32(j) >= params.height) { return CELL_AIR; }  // Top (OPEN!)

    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

// Apply vertical (Y) gravity to V grid
@compute @workgroup_size(8, 8, 4)
fn apply_gravity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;  // 0 to height (inclusive for V grid)
    let k = id.z;

    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    // V face is between cells [i, j-1, k] and [i, j, k]
    let bottom_type = get_cell_type(i32(i), i32(j) - 1, i32(k));
    let top_type = get_cell_type(i32(i), i32(j), i32(k));

    // Don't apply gravity to solid boundary faces
    if (bottom_type == CELL_SOLID || top_type == CELL_SOLID) {
        return;
    }

    // Only apply gravity if at least one adjacent cell is fluid
    if (bottom_type == CELL_FLUID || top_type == CELL_FLUID) {
        let idx = v_index(i, j, k);
        grid_v[idx] += params.gravity_y_dt;
    }
}

// Apply horizontal (X) gravity to U grid - for tilted sluice slope
@compute @workgroup_size(8, 8, 4)
fn apply_gravity_x(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;  // 0 to width (inclusive for U grid)
    let j = id.y;
    let k = id.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    // U face is between cells [i-1, j, k] and [i, j, k]
    let left_type = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let right_type = get_cell_type(i32(i), i32(j), i32(k));

    // Don't apply gravity to solid boundary faces
    if (left_type == CELL_SOLID || right_type == CELL_SOLID) {
        return;
    }

    // Only apply gravity if at least one adjacent cell is fluid
    if (left_type == CELL_FLUID || right_type == CELL_FLUID) {
        let idx = u_index(i, j, k);
        grid_u[idx] += params.gravity_x_dt;
    }
}
