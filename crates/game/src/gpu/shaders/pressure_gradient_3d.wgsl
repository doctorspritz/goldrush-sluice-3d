// Pressure Gradient Application Shader (3D)
//
// Subtracts pressure gradient from velocity to enforce incompressibility:
//   u -= dt * (p[i,j,k] - p[i-1,j,k]) / dx
//   v -= dt * (p[i,j,k] - p[i,j-1,k]) / dx
//   w -= dt * (p[i,j,k] - p[i,j,k-1]) / dx
//
// Note: We don't multiply by dt here - the pressure already has correct units
// from the Poisson solve. The gradient gives velocity directly.
//
// Boundary conditions:
// - Solid boundaries: velocity set to zero (no-slip or no-penetration)
// - Air boundaries: pressure is zero, gradient is just p[fluid] / dx

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    inv_cell_size: f32,  // 1.0 / cell_size
}

// Velocity clamping removed - matches CPU behavior and allows proper hydrostatic support

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(4) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> grid_w: array<f32>;

// Cell type constants (matching sluice_3d_visual convention)
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

// Get cell type with proper boundary handling for sluice:
// Both inlet and outlet are OPEN to allow flow-through.
fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    // CLOSED boundaries: floor and side walls
    if (j < 0) { return CELL_SOLID; }  // Floor (closed)
    if (k < 0) { return CELL_SOLID; }  // Side wall (closed)
    if (u32(k) >= params.depth) { return CELL_SOLID; }  // Side wall (closed)

    // OPEN boundaries: inlet, outlet, and top → treat as AIR to allow flow
    if (i < 0) { return CELL_AIR; }  // Inlet (OPEN for sluice flow!)
    if (u32(i) >= params.width) { return CELL_AIR; }  // Outlet (OPEN!)
    if (u32(j) >= params.height) { return CELL_AIR; }  // Top (OPEN!)

    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

fn get_pressure(i: i32, j: i32, k: i32) -> f32 {
    if (i < 0 || j < 0 || k < 0 ||
        u32(i) >= params.width || u32(j) >= params.height || u32(k) >= params.depth) {
        return 0.0;
    }
    let ct = cell_type[cell_index(u32(i), u32(j), u32(k))];
    if (ct != CELL_FLUID) {
        return 0.0;  // Air and solid have zero pressure
    }
    return pressure[cell_index(u32(i), u32(j), u32(k))];
}

// Apply pressure gradient to U velocity component
// U faces are between cells [i-1,j,k] and [i,j,k]
@compute @workgroup_size(8, 8, 4)
fn apply_gradient_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;  // 0 to width (inclusive)
    let j = id.y;
    let k = id.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = u_index(i, j, k);

    // Skip inlet face (i=0): Neumann BC means no pressure gradient at inlet
    // This allows water to enter without being pushed back by pressure buildup inside.
    // The CPU also skips i=0 in apply_pressure_gradient.
    if (i == 0u) {
        return;
    }

    // Get cell types on either side of this U face
    let left_type = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let right_type = get_cell_type(i32(i), i32(j), i32(k));

    // If either side is solid, zero velocity (no-penetration)
    if (left_type == CELL_SOLID || right_type == CELL_SOLID) {
        grid_u[idx] = 0.0;
        return;
    }

    // Skip if both sides are air (no fluid to drive flow)
    if (left_type == CELL_AIR && right_type == CELL_AIR) {
        return;
    }

    // Apply pressure gradient
    let p_right = get_pressure(i32(i), i32(j), i32(k));
    let p_left = get_pressure(i32(i) - 1, i32(j), i32(k));

    // Apply gradient directly - no clamping to match CPU behavior
    grid_u[idx] = grid_u[idx] - (p_right - p_left) * params.inv_cell_size;
}

// Apply pressure gradient to V velocity component
// V faces are between cells [i,j-1,k] and [i,j,k]
@compute @workgroup_size(8, 8, 4)
fn apply_gradient_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;  // 0 to height (inclusive)
    let k = id.z;

    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    let idx = v_index(i, j, k);

    let bottom_type = get_cell_type(i32(i), i32(j) - 1, i32(k));
    let top_type = get_cell_type(i32(i), i32(j), i32(k));

    if (bottom_type == CELL_SOLID || top_type == CELL_SOLID) {
        grid_v[idx] = 0.0;
        return;
    }

    if (bottom_type == CELL_AIR && top_type == CELL_AIR) {
        return;
    }

    let p_top = get_pressure(i32(i), i32(j), i32(k));
    let p_bottom = get_pressure(i32(i), i32(j) - 1, i32(k));

    // Apply gradient directly - no clamping to match CPU behavior
    grid_v[idx] = grid_v[idx] - (p_top - p_bottom) * params.inv_cell_size;
}

// Apply pressure gradient to W velocity component
// W faces are between cells [i,j,k-1] and [i,j,k]
@compute @workgroup_size(8, 8, 4)
fn apply_gradient_w(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;  // 0 to depth (inclusive)

    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    let idx = w_index(i, j, k);

    let back_type = get_cell_type(i32(i), i32(j), i32(k) - 1);
    let front_type = get_cell_type(i32(i), i32(j), i32(k));

    if (back_type == CELL_SOLID || front_type == CELL_SOLID) {
        grid_w[idx] = 0.0;
        return;
    }

    if (back_type == CELL_AIR && front_type == CELL_AIR) {
        return;
    }

    let p_front = get_pressure(i32(i), i32(j), i32(k));
    let p_back = get_pressure(i32(i), i32(j), i32(k) - 1);

    // Apply gradient directly - no clamping to match CPU behavior
    grid_w[idx] = grid_w[idx] - (p_front - p_back) * params.inv_cell_size;
}
