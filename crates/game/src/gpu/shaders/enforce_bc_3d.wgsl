// Boundary Condition Enforcement Shader (3D)
//
// Zeros out velocity components at domain boundaries and solid cell faces.
// Must be called AFTER P2G and BEFORE storing old velocities for FLIP delta.
//
// For sluice simulation:
// - CLOSED boundaries: inlet (x=0), floor (y=0), side walls (z=0, z=depth)
// - OPEN boundaries: outlet (x=width), top (y=height) - velocities NOT zeroed

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    // Bitmask for open boundaries (velocity NOT zeroed):
    // Bit 0 (1): -X open, Bit 1 (2): +X open
    // Bit 2 (4): -Y open, Bit 3 (8): +Y open
    // Bit 4 (16): -Z open, Bit 5 (32): +Z open
    open_boundaries: u32,
    // Slip factor for tangential velocities at solid boundaries:
    // 1.0 = free-slip (allow tangential flow)
    // 0.0 = no-slip (zero tangential velocity)
    // Intermediate = partial slip (damp tangential velocity)
    slip_factor: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> grid_w: array<f32>;

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

    // Check open boundary flags
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;

    // i=0 boundary: -X wall
    // CLOSED: zero velocity (no penetration)
    // OPEN: allow flow through
    if (i == 0u) {
        if (!open_neg_x) {
            grid_u[idx] = 0.0;
        }
        return;
    }

    // i=width boundary: +X wall
    // CLOSED: zero velocity (no penetration)
    // OPEN: allow flow through
    if (i == params.width) {
        if (!open_pos_x) {
            grid_u[idx] = 0.0;
        }
        return;
    }

    // Zero velocities at solid cell faces (internal obstacles)
    // U face is between cells [i-1, j, k] and [i, j, k]
    let left_solid = cell_type[cell_index(i - 1u, j, k)] == CELL_SOLID;
    let right_solid = cell_type[cell_index(i, j, k)] == CELL_SOLID;

    if (left_solid || right_solid) {
        grid_u[idx] = 0.0;
        return;
    }

    // No-slip/partial-slip at floor: damp tangential velocity when cell below is solid
    // U face at (i, j, k) is at y = (j + 0.5) * cell_size
    // For proper no-slip: tangential velocity should be zero at the wall (y=0)
    // We zero j=0..3 layers to create a thick boundary layer for hydrostatic equilibrium
    if (params.slip_factor < 1.0) {
        let open_neg_y = (params.open_boundaries & 4u) != 0u;
        var damp_factor = 1.0;

        // j<4: First four layers above domain floor - full damping
        // This creates a thick no-slip boundary layer that damps standing wave modes
        if (j < 4u && !open_neg_y) {
            damp_factor = params.slip_factor;
        }

        // j>=4: Check if either adjacent cell at j-1 level is solid (internal floor/obstacle)
        if (j >= 4u) {
            let left_below_solid = cell_type[cell_index(i - 1u, j - 1u, k)] == CELL_SOLID;
            let right_below_solid = cell_type[cell_index(i, j - 1u, k)] == CELL_SOLID;
            if (left_below_solid || right_below_solid) {
                damp_factor = params.slip_factor;
            }
        }

        if (damp_factor < 1.0) {
            grid_u[idx] *= damp_factor;
        }
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

    // Check open boundary flags
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;

    // j=0 boundary: floor (-Y)
    // CLOSED: zero velocity (no penetration)
    // OPEN: allow flow through
    if (j == 0u) {
        if (!open_neg_y) {
            grid_v[idx] = 0.0;
        }
        return;
    }

    // j=height boundary: ceiling (+Y)
    // CLOSED: zero velocity (no penetration)
    // OPEN: allow flow through (free surface)
    if (j == params.height) {
        if (!open_pos_y) {
            grid_v[idx] = 0.0;
        }
        return;
    }

    // Zero velocities at solid cell faces (internal obstacles)
    // V face is between cells [i, j-1, k] and [i, j, k]
    let bottom_solid = cell_type[cell_index(i, j - 1u, k)] == CELL_SOLID;
    let top_solid = cell_type[cell_index(i, j, k)] == CELL_SOLID;

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

    // Check open boundary flags
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // k=0 boundary: -Z wall
    // CLOSED: zero velocity (no penetration)
    // OPEN: allow flow through
    if (k == 0u) {
        if (!open_neg_z) {
            grid_w[idx] = 0.0;
        }
        return;
    }

    // k=depth boundary: +Z wall
    // CLOSED: zero velocity (no penetration)
    // OPEN: allow flow through
    if (k == params.depth) {
        if (!open_pos_z) {
            grid_w[idx] = 0.0;
        }
        return;
    }

    // Zero velocities at solid cell faces (internal obstacles)
    // W face is between cells [i, j, k-1] and [i, j, k]
    let back_solid = cell_type[cell_index(i, j, k - 1u)] == CELL_SOLID;
    let front_solid = cell_type[cell_index(i, j, k)] == CELL_SOLID;

    if (back_solid || front_solid) {
        grid_w[idx] = 0.0;
        return;
    }

    // No-slip/partial-slip at floor: damp tangential velocity when cell below is solid
    // W face at (i, j, k) is at y = (j + 0.5) * cell_size
    // For proper no-slip: tangential velocity should be zero at the wall (y=0)
    // We zero j=0..3 layers to create a thick boundary layer for hydrostatic equilibrium
    if (params.slip_factor < 1.0) {
        let open_neg_y = (params.open_boundaries & 4u) != 0u;
        var damp_factor = 1.0;

        // j<4: First four layers above domain floor - full damping
        if (j < 4u && !open_neg_y) {
            damp_factor = params.slip_factor;
        }

        // j>=4: Check if either adjacent cell at j-1 level is solid (internal floor/obstacle)
        if (j >= 4u && k > 0u) {
            let back_below_solid = cell_type[cell_index(i, j - 1u, k - 1u)] == CELL_SOLID;
            let front_below_solid = cell_type[cell_index(i, j - 1u, k)] == CELL_SOLID;
            if (back_below_solid || front_below_solid) {
                damp_factor = params.slip_factor;
            }
        }

        if (damp_factor < 1.0) {
            grid_w[idx] *= damp_factor;
        }
    }
}
