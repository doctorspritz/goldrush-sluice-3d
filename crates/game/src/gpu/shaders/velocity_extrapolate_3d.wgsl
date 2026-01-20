// Velocity Extrapolation Shader (3D)
//
// Extrapolates velocities from FLUID cells into adjacent AIR cells.
// This is critical for FLIP stability at the free surface.
//
// Per Bridson and Houdini FLIP docs:
// - Particles near the surface sample velocities from the grid
// - Without extrapolation, they sample from AIR cells with undefined velocities
// - This causes energy loss and surface collapse
//
// Run this AFTER pressure solve, BEFORE G2P.
// Run multiple passes (2-3) for deeper extrapolation into AIR.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    extrap_pass: u32,  // Current extrapolation pass (0, 1, 2...)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> grid_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> valid_u: array<u32>;  // Which U faces have valid velocity
@group(0) @binding(6) var<storage, read_write> valid_v: array<u32>;  // Which V faces have valid velocity
@group(0) @binding(7) var<storage, read_write> valid_w: array<u32>;  // Which W faces have valid velocity

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

// Cell indices for (width × height × depth) grid
fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

// U face indices: (width+1) × height × depth
fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

// V face indices: width × (height+1) × depth
fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

// W face indices: width × height × (depth+1)
fn w_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_SOLID; }
    if (j < 0 || j >= i32(params.height)) { return CELL_SOLID; }
    if (k < 0 || k >= i32(params.depth)) { return CELL_SOLID; }
    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

// On pass 0: mark faces adjacent to FLUID cells as valid
// On pass 1+: extend validity to faces adjacent to previously valid faces
@compute @workgroup_size(8, 8, 4)
fn init_valid_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // Initialize U faces
    if (i <= params.width && j < params.height && k < params.depth) {
        let u_idx = u_index(i, j, k);
        // U face (i, j, k) is between cells (i-1, j, k) and (i, j, k)
        let left = get_cell_type(i32(i) - 1, i32(j), i32(k));
        let right = get_cell_type(i32(i), i32(j), i32(k));
        // Valid if at least one adjacent cell is FLUID
        if (left == CELL_FLUID || right == CELL_FLUID) {
            valid_u[u_idx] = 1u;
        } else {
            valid_u[u_idx] = 0u;
        }
    }

    // Initialize V faces
    if (i < params.width && j <= params.height && k < params.depth) {
        let v_idx = v_index(i, j, k);
        // V face (i, j, k) is between cells (i, j-1, k) and (i, j, k)
        let below = get_cell_type(i32(i), i32(j) - 1, i32(k));
        let above = get_cell_type(i32(i), i32(j), i32(k));
        if (below == CELL_FLUID || above == CELL_FLUID) {
            valid_v[v_idx] = 1u;
        } else {
            valid_v[v_idx] = 0u;
        }
    }

    // Initialize W faces
    if (i < params.width && j < params.height && k <= params.depth) {
        let w_idx = w_index(i, j, k);
        // W face (i, j, k) is between cells (i, j, k-1) and (i, j, k)
        let back = get_cell_type(i32(i), i32(j), i32(k) - 1);
        let front = get_cell_type(i32(i), i32(j), i32(k));
        if (back == CELL_FLUID || front == CELL_FLUID) {
            valid_w[w_idx] = 1u;
        } else {
            valid_w[w_idx] = 0u;
        }
    }
}

// Extrapolate U velocities one layer into AIR
@compute @workgroup_size(8, 8, 4)
fn extrapolate_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let u_idx = u_index(i, j, k);

    // Skip if already valid
    if (valid_u[u_idx] == 1u) {
        return;
    }

    // Skip faces at SOLID boundaries - extrapolating into solid walls causes instability
    let left = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let right = get_cell_type(i32(i), i32(j), i32(k));
    if (left == CELL_SOLID || right == CELL_SOLID) {
        return;
    }

    // Check 6 neighbor U faces for valid velocity
    var sum = 0.0;
    var count = 0u;

    // -X neighbor
    if (i > 0u) {
        let n_idx = u_index(i - 1u, j, k);
        if (valid_u[n_idx] == 1u) {
            sum += grid_u[n_idx];
            count += 1u;
        }
    }
    // +X neighbor
    if (i < params.width) {
        let n_idx = u_index(i + 1u, j, k);
        if (valid_u[n_idx] == 1u) {
            sum += grid_u[n_idx];
            count += 1u;
        }
    }
    // -Y neighbor
    if (j > 0u) {
        let n_idx = u_index(i, j - 1u, k);
        if (valid_u[n_idx] == 1u) {
            sum += grid_u[n_idx];
            count += 1u;
        }
    }
    // +Y neighbor
    if (j < params.height - 1u) {
        let n_idx = u_index(i, j + 1u, k);
        if (valid_u[n_idx] == 1u) {
            sum += grid_u[n_idx];
            count += 1u;
        }
    }
    // -Z neighbor
    if (k > 0u) {
        let n_idx = u_index(i, j, k - 1u);
        if (valid_u[n_idx] == 1u) {
            sum += grid_u[n_idx];
            count += 1u;
        }
    }
    // +Z neighbor
    if (k < params.depth - 1u) {
        let n_idx = u_index(i, j, k + 1u);
        if (valid_u[n_idx] == 1u) {
            sum += grid_u[n_idx];
            count += 1u;
        }
    }

    // Extrapolate if we found valid neighbors
    if (count > 0u) {
        grid_u[u_idx] = sum / f32(count);
        valid_u[u_idx] = 2u;  // Mark as newly valid (will become 1 after pass)
    }
}

// Extrapolate V velocities one layer into AIR
@compute @workgroup_size(8, 8, 4)
fn extrapolate_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    let v_idx = v_index(i, j, k);

    // Skip if already valid
    if (valid_v[v_idx] == 1u) {
        return;
    }

    // Skip faces at SOLID boundaries - extrapolating into solid walls causes instability
    let below = get_cell_type(i32(i), i32(j) - 1, i32(k));
    let above = get_cell_type(i32(i), i32(j), i32(k));
    if (below == CELL_SOLID || above == CELL_SOLID) {
        return;
    }

    // Check 6 neighbor V faces for valid velocity
    var sum = 0.0;
    var count = 0u;

    // -X neighbor
    if (i > 0u) {
        let n_idx = v_index(i - 1u, j, k);
        if (valid_v[n_idx] == 1u) {
            sum += grid_v[n_idx];
            count += 1u;
        }
    }
    // +X neighbor
    if (i < params.width - 1u) {
        let n_idx = v_index(i + 1u, j, k);
        if (valid_v[n_idx] == 1u) {
            sum += grid_v[n_idx];
            count += 1u;
        }
    }
    // -Y neighbor
    if (j > 0u) {
        let n_idx = v_index(i, j - 1u, k);
        if (valid_v[n_idx] == 1u) {
            sum += grid_v[n_idx];
            count += 1u;
        }
    }
    // +Y neighbor
    if (j < params.height) {
        let n_idx = v_index(i, j + 1u, k);
        if (valid_v[n_idx] == 1u) {
            sum += grid_v[n_idx];
            count += 1u;
        }
    }
    // -Z neighbor
    if (k > 0u) {
        let n_idx = v_index(i, j, k - 1u);
        if (valid_v[n_idx] == 1u) {
            sum += grid_v[n_idx];
            count += 1u;
        }
    }
    // +Z neighbor
    if (k < params.depth - 1u) {
        let n_idx = v_index(i, j, k + 1u);
        if (valid_v[n_idx] == 1u) {
            sum += grid_v[n_idx];
            count += 1u;
        }
    }

    // Extrapolate if we found valid neighbors
    if (count > 0u) {
        grid_v[v_idx] = sum / f32(count);
        valid_v[v_idx] = 2u;
    }
}

// Extrapolate W velocities one layer into AIR
@compute @workgroup_size(8, 8, 4)
fn extrapolate_w(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    let w_idx = w_index(i, j, k);

    // Skip if already valid
    if (valid_w[w_idx] == 1u) {
        return;
    }

    // Skip faces at SOLID boundaries - extrapolating into solid walls causes instability
    let back = get_cell_type(i32(i), i32(j), i32(k) - 1);
    let front = get_cell_type(i32(i), i32(j), i32(k));
    if (back == CELL_SOLID || front == CELL_SOLID) {
        return;
    }

    // Check 6 neighbor W faces for valid velocity
    var sum = 0.0;
    var count = 0u;

    // -X neighbor
    if (i > 0u) {
        let n_idx = w_index(i - 1u, j, k);
        if (valid_w[n_idx] == 1u) {
            sum += grid_w[n_idx];
            count += 1u;
        }
    }
    // +X neighbor
    if (i < params.width - 1u) {
        let n_idx = w_index(i + 1u, j, k);
        if (valid_w[n_idx] == 1u) {
            sum += grid_w[n_idx];
            count += 1u;
        }
    }
    // -Y neighbor
    if (j > 0u) {
        let n_idx = w_index(i, j - 1u, k);
        if (valid_w[n_idx] == 1u) {
            sum += grid_w[n_idx];
            count += 1u;
        }
    }
    // +Y neighbor
    if (j < params.height - 1u) {
        let n_idx = w_index(i, j + 1u, k);
        if (valid_w[n_idx] == 1u) {
            sum += grid_w[n_idx];
            count += 1u;
        }
    }
    // -Z neighbor
    if (k > 0u) {
        let n_idx = w_index(i, j, k - 1u);
        if (valid_w[n_idx] == 1u) {
            sum += grid_w[n_idx];
            count += 1u;
        }
    }
    // +Z neighbor
    if (k < params.depth) {
        let n_idx = w_index(i, j, k + 1u);
        if (valid_w[n_idx] == 1u) {
            sum += grid_w[n_idx];
            count += 1u;
        }
    }

    // Extrapolate if we found valid neighbors
    if (count > 0u) {
        grid_w[w_idx] = sum / f32(count);
        valid_w[w_idx] = 2u;
    }
}

// Finalize pass: mark newly extrapolated faces (value=2) as valid (value=1)
// This is needed for multi-pass extrapolation
@compute @workgroup_size(8, 8, 4)
fn finalize_valid(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // U faces
    if (i <= params.width && j < params.height && k < params.depth) {
        let u_idx = u_index(i, j, k);
        if (valid_u[u_idx] == 2u) {
            valid_u[u_idx] = 1u;
        }
    }

    // V faces
    if (i < params.width && j <= params.height && k < params.depth) {
        let v_idx = v_index(i, j, k);
        if (valid_v[v_idx] == 2u) {
            valid_v[v_idx] = 1u;
        }
    }

    // W faces
    if (i < params.width && j < params.height && k <= params.depth) {
        let w_idx = w_index(i, j, k);
        if (valid_w[w_idx] == 2u) {
            valid_w[w_idx] = 1u;
        }
    }
}
