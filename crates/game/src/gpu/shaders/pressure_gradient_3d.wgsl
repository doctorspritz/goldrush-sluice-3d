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
    // Bitmask for open boundaries:
    // Bit 0 (1): -X open, Bit 1 (2): +X open
    // Bit 2 (4): -Y open, Bit 3 (8): +Y open
    // Bit 4 (16): -Z open, Bit 5 (32): +Z open
    open_boundaries: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Velocity clamping removed - proper pressure solver handles stability
// (Previously clamped to ±10 m/s as safety valve)

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

// Get cell type with proper boundary handling based on open_boundaries config.
// OPEN boundary → treat as AIR (Dirichlet p=0, flow allowed)
// CLOSED boundary → treat as SOLID (Neumann, no penetration)
fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    // Check open boundary flags
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // -X boundary (inlet side)
    if (i < 0) {
        return select(CELL_SOLID, CELL_AIR, open_neg_x);
    }
    // +X boundary (outlet side)
    if (u32(i) >= params.width) {
        return select(CELL_SOLID, CELL_AIR, open_pos_x);
    }
    // -Y boundary (floor)
    if (j < 0) {
        return select(CELL_SOLID, CELL_AIR, open_neg_y);
    }
    // +Y boundary (ceiling/top)
    if (u32(j) >= params.height) {
        return select(CELL_SOLID, CELL_AIR, open_pos_y);
    }
    // -Z boundary (side wall)
    if (k < 0) {
        return select(CELL_SOLID, CELL_AIR, open_neg_z);
    }
    // +Z boundary (side wall)
    if (u32(k) >= params.depth) {
        return select(CELL_SOLID, CELL_AIR, open_pos_z);
    }

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
// Positive U = flow from left to right (+X direction)
@compute @workgroup_size(8, 8, 4)
fn apply_gradient_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;  // 0 to width (inclusive)
    let j = id.y;
    let k = id.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = u_index(i, j, k);

    // Get cell types on either side of this U face
    let left_type = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let right_type = get_cell_type(i32(i), i32(j), i32(k));

    // Case 1: Both solid - zero velocity
    if (left_type == CELL_SOLID && right_type == CELL_SOLID) {
        grid_u[idx] = 0.0;
        return;
    }

    // Case 2: Both air - no pressure gradient (no fluid)
    if (left_type == CELL_AIR && right_type == CELL_AIR) {
        return;
    }

    // Case 3: One side is solid - Neumann BC (dp/dn = 0)
    // No pressure gradient through solid walls, just enforce no-penetration
    if (left_type == CELL_SOLID) {
        // Left is solid: can't have positive velocity (from solid)
        grid_u[idx] = min(grid_u[idx], 0.0);
        return;
    }
    if (right_type == CELL_SOLID) {
        // Right is solid: can't have negative velocity (into solid)
        grid_u[idx] = max(grid_u[idx], 0.0);
        return;
    }

    // Case 4: Both sides are fluid or air - apply pressure gradient
    // This includes fluid-fluid, fluid-air, and air-fluid faces
    let p_right = get_pressure(i32(i), i32(j), i32(k));
    let p_left = get_pressure(i32(i) - 1, i32(j), i32(k));

    grid_u[idx] = grid_u[idx] - (p_right - p_left) * params.inv_cell_size;
}

// Apply pressure gradient to V velocity component
// V faces are between cells [i,j-1,k] and [i,j,k]
// Positive V = flow from bottom to top (+Y direction)
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

    // Case 1: Both solid - zero velocity
    if (bottom_type == CELL_SOLID && top_type == CELL_SOLID) {
        grid_v[idx] = 0.0;
        return;
    }

    // Case 2: Both air - no pressure gradient (no fluid)
    if (bottom_type == CELL_AIR && top_type == CELL_AIR) {
        return;
    }

    // Case 3: One side is solid - Neumann BC (dp/dn = 0)
    // No pressure gradient through solid walls, just enforce no-penetration
    if (bottom_type == CELL_SOLID) {
        // Bottom is solid: can't have positive velocity (from solid)
        grid_v[idx] = min(grid_v[idx], 0.0);
        return;
    }
    if (top_type == CELL_SOLID) {
        // Top is solid: can't have negative velocity (into solid)
        grid_v[idx] = max(grid_v[idx], 0.0);
        return;
    }

    // Case 4: Both sides are fluid or air - apply pressure gradient
    // This includes fluid-fluid, fluid-air, and air-fluid faces
    let p_top = get_pressure(i32(i), i32(j), i32(k));
    let p_bottom = get_pressure(i32(i), i32(j) - 1, i32(k));

    grid_v[idx] = grid_v[idx] - (p_top - p_bottom) * params.inv_cell_size;
}

// Apply pressure gradient to W velocity component
// W faces are between cells [i,j,k-1] and [i,j,k]
// Positive W = flow from back to front (+Z direction)
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

    // Case 1: Both solid - zero velocity
    if (back_type == CELL_SOLID && front_type == CELL_SOLID) {
        grid_w[idx] = 0.0;
        return;
    }

    // Case 2: Both air - no pressure gradient (no fluid)
    if (back_type == CELL_AIR && front_type == CELL_AIR) {
        return;
    }

    // Case 3: One side is solid - Neumann BC (dp/dn = 0)
    // No pressure gradient through solid walls, just enforce no-penetration
    if (back_type == CELL_SOLID) {
        // Back is solid: can't have positive velocity (from solid)
        grid_w[idx] = min(grid_w[idx], 0.0);
        return;
    }
    if (front_type == CELL_SOLID) {
        // Front is solid: can't have negative velocity (into solid)
        grid_w[idx] = max(grid_w[idx], 0.0);
        return;
    }

    // Case 4: Both sides are fluid or air - apply pressure gradient
    // This includes fluid-fluid, fluid-air, and air-fluid faces
    let p_front = get_pressure(i32(i), i32(j), i32(k));
    let p_back = get_pressure(i32(i), i32(j), i32(k) - 1);

    grid_w[idx] = grid_w[idx] - (p_front - p_back) * params.inv_cell_size;
}
