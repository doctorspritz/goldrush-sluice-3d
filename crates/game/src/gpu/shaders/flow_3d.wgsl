// Flow Acceleration Shader (3D)
//
// Applies downstream flow acceleration to the U (horizontal/X) velocity component.
// This simulates the gravity component along a sloped channel.
// Only affects faces adjacent to at least one fluid cell.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    flow_accel_dt: f32,  // flow_accel * dt (e.g., 2.0 * dt for sluice)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_u: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

// Get cell type with proper boundary handling for sluice
fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    // CLOSED boundaries: floor and side walls
    if (j < 0) { return CELL_SOLID; }  // Floor (closed)
    if (k < 0) { return CELL_SOLID; }  // Side wall (closed)
    if (u32(k) >= params.depth) { return CELL_SOLID; }  // Side wall (closed)

    // OPEN boundaries: inlet, outlet, and top → treat as AIR
    if (i < 0) { return CELL_AIR; }  // Inlet (OPEN!)
    if (u32(i) >= params.width) { return CELL_AIR; }  // Outlet (OPEN!)
    if (u32(j) >= params.height) { return CELL_AIR; }  // Top (OPEN!)

    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

@compute @workgroup_size(8, 8, 4)
fn apply_flow(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;  // 0 to width (inclusive for U grid)
    let j = id.y;
    let k = id.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    // U face is between cells [i-1, j, k] and [i, j, k]
    let left_type = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let right_type = get_cell_type(i32(i), i32(j), i32(k));

    // Apply flow acceleration to any face with at least one fluid cell.
    // We do NOT skip solid-adjacent faces (riffles) because:
    // 1. Flow acceleration represents gravity along the slope
    // 2. Gravity acts on fluid whether or not there's a riffle nearby
    // 3. The pressure solver enforces no-penetration at solid boundaries
    //
    // OLD BUG: Skipping solid-adjacent faces created "dead zones" where
    // water pooled behind riffles with no downstream push.
    if (left_type == CELL_FLUID || right_type == CELL_FLUID) {
        let idx = u_index(i, j, k);
        grid_u[idx] += params.flow_accel_dt;
    }
}
