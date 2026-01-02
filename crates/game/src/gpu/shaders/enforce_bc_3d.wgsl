// Boundary Condition Enforcement Shader (3D)
//
// Zeros out velocity components at domain boundaries, solid cell faces,
// AND riffle terrain (floor_heights).
//
// Must be called AFTER P2G and BEFORE storing old velocities for FLIP delta.
//
// For sluice simulation:
// - CLOSED boundaries: inlet (x=0), floor (variable via floor_heights), side walls (z=0, z=depth)
// - OPEN boundaries: outlet (x=width), top (y=height) - velocities NOT zeroed
// - RIFFLE WALLS: vertical faces where floor_heights increases in +X direction

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> grid_w: array<f32>;
@group(0) @binding(5) var<storage, read> floor_heights: array<u32>;

// Get floor height at grid position (i, k)
// Returns the number of cells from y=0 to the floor surface
fn get_floor_height(i: u32, k: u32) -> u32 {
    let clamped_i = min(i, params.width - 1u);
    let clamped_k = min(k, params.depth - 1u);
    return floor_heights[clamped_k * params.width + clamped_i];
}

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

// Enforce U boundary conditions
// U faces: i from 0 to width (inclusive)
@compute @workgroup_size(8, 8, 4)
fn enforce_bc_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = u_index(i, j, k);

    // i=0 boundary: inlet (OPEN for sluice flow!)
    // Previously zeroing U[0] caused stagnation - particles lose velocity
    // because FLIP blend pulls toward zero grid velocity.
    // For open channel flow, let inlet have non-zero velocity from particles.
    // (no zeroing at i=0)

    // i=width boundary: outlet (OPEN) - do NOT zero
    // (no zeroing at i=width)

    // Zero velocities at solid cell faces (internal obstacles only)
    // U face is between cells [i-1, j, k] and [i, j, k]
    // EXCEPTION: Skip i<=1 to allow inlet inflow (don't let inlet wall kill velocity)
    if (i <= 1u) {
        return;  // Inlet region - don't zero, allow inflow
    }

    // Check for riffle wall collision:
    // U face at (i, j, k) is between cells [i-1, j, k] and [i, j, k]
    // If floor_heights[i] > floor_heights[i-1] AND j is below the riffle top,
    // this is a riffle wall - zero U velocity!
    let floor_left = get_floor_height(i - 1u, k);
    let floor_right = get_floor_height(i, k);

    if (floor_right > floor_left) {
        // There's a riffle wall between cell i-1 and cell i
        // Zero U for all j below the riffle top
        if (j < floor_right) {
            grid_u[idx] = 0.0;
            return;
        }
    }

    // Also check for solid cells (internal obstacles)
    let left_solid = cell_type[cell_index(i - 1u, j, k)] == CELL_SOLID;
    let right_solid = (i < params.width && cell_type[cell_index(i, j, k)] == CELL_SOLID);

    if (left_solid || right_solid) {
        grid_u[idx] = 0.0;
    }
}

// Enforce V boundary conditions
// V faces: j from 0 to height (inclusive)
@compute @workgroup_size(8, 8, 4)
fn enforce_bc_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    let idx = v_index(i, j, k);

    // Get floor height at this (i, k) position
    let floor_h = get_floor_height(i, k);

    // V face at j is between cells [i, j-1, k] and [i, j, k]
    // Zero V at the floor surface: when j <= floor_h
    // This means the V face is at or below the terrain surface
    if (j <= floor_h) {
        grid_v[idx] = 0.0;
        return;
    }

    // j=height boundary: top surface (OPEN) - do NOT zero
    // (commented out intentionally)

    // Zero velocities at solid cell faces
    // V face is between cells [i, j-1, k] and [i, j, k]
    let bottom_solid = (j > 0u && cell_type[cell_index(i, j - 1u, k)] == CELL_SOLID);
    let top_solid = (j < params.height && cell_type[cell_index(i, j, k)] == CELL_SOLID);

    if (bottom_solid || top_solid) {
        grid_v[idx] = 0.0;
    }
}

// Enforce W boundary conditions
// W faces: k from 0 to depth (inclusive)
@compute @workgroup_size(8, 8, 4)
fn enforce_bc_w(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    let idx = w_index(i, j, k);

    // k=0 boundary: side wall (CLOSED)
    if (k == 0u) {
        grid_w[idx] = 0.0;
        return;
    }

    // k=depth boundary: side wall (CLOSED)
    if (k == params.depth) {
        grid_w[idx] = 0.0;
        return;
    }

    // Zero velocities at solid cell faces
    // W face is between cells [i, j, k-1] and [i, j, k]
    let back_solid = (k > 0u && cell_type[cell_index(i, j, k - 1u)] == CELL_SOLID);
    let front_solid = (k < params.depth && cell_type[cell_index(i, j, k)] == CELL_SOLID);

    if (back_solid || front_solid) {
        grid_w[idx] = 0.0;
    }
}
