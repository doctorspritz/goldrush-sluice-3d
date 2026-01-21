// Pressure solver (3D) using Checkerboard SOR (Successive Over-Relaxation)
//
// Red-black ordering allows parallel updates without data races.
// In 3D, red cells have (i + j + k) % 2 == 0, black cells have (i + j + k) % 2 == 1.
//
// Uses 6-neighbor Laplacian stencil for pressure Poisson equation:
//   ∇²p = -∇·u (divergence)
//
// Boundary conditions:
// - Solid cells: Neumann BC (dp/dn = 0), pressure copied from fluid
// - Air cells: Dirichlet BC (p = 0)

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    omega: f32,  // SOR relaxation factor (typically 1.5-1.9 for 3D)
    h_sq: f32,   // cell_size^2, needed to scale divergence in Poisson equation
    open_boundaries: u32,  // Bitmask: 1=-X, 2=+X, 4=-Y, 8=+Y, 16=-Z, 32=+Z
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

// Cell type constants (matching sluice_3d_visual convention)
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn get_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn is_valid(i: u32, j: u32, k: u32) -> bool {
    return i < params.width && j < params.height && k < params.depth;
}

fn get_cell_type(i: u32, j: u32, k: u32) -> u32 {
    if (!is_valid(i, j, k)) {
        return CELL_SOLID;  // Treat out-of-bounds as solid
    }
    return cell_type[get_index(i, j, k)];
}

fn get_pressure(i: u32, j: u32, k: u32) -> f32 {
    if (!is_valid(i, j, k)) {
        return 0.0;
    }
    let ct = cell_type[get_index(i, j, k)];
    if (ct == CELL_SOLID || ct == CELL_AIR) {
        // Solid/Air cells: use zero pressure (Dirichlet for air, handled by neighbor count for solid)
        return 0.0;
    }
    return pressure[get_index(i, j, k)];
}

// Get neighbor pressure with proper boundary conditions
// - Solid: Neumann BC (dp/dn = 0) - use current cell's pressure
// - Air: Dirichlet BC (p = 0) - atmospheric pressure
// - Fluid: use actual pressure
fn get_neighbor_pressure(ni: u32, nj: u32, nk: u32, current_p: f32) -> f32 {
    if (!is_valid(ni, nj, nk)) {
        return current_p;  // Treat out-of-bounds as Neumann
    }
    let ct = cell_type[get_index(ni, nj, nk)];
    if (ct == CELL_SOLID) {
        return current_p;  // Neumann BC: use current cell pressure
    }
    if (ct == CELL_AIR) {
        return 0.0;  // Dirichlet BC: atmospheric pressure
    }
    return pressure[get_index(ni, nj, nk)];  // Fluid: actual pressure
}

// Process "red" cells: (i + j + k) % 2 == 0
@compute @workgroup_size(8, 8, 4)
fn pressure_red(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (!is_valid(i, j, k)) {
        return;
    }

    // Skip non-red cells (black cells have (i+j+k) % 2 == 1)
    if (((i + j + k) % 2u) != 0u) {
        return;
    }

    let idx = get_index(i, j, k);

    // Skip non-fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    let p = pressure[idx];

    // Gather neighbor pressures with proper boundary conditions
    // Always count 6 neighbors (proper Neumann/Dirichlet handling)
    var sum_neighbors = 0.0;

    // -X neighbor
    if (i > 0u) {
        sum_neighbors += get_neighbor_pressure(i - 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 1u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +X neighbor
    if (i < params.width - 1u) {
        sum_neighbors += get_neighbor_pressure(i + 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 2u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Y neighbor
    if (j > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j - 1u, k, p);
    } else {
        if ((params.open_boundaries & 4u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Y neighbor
    if (j < params.height - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j + 1u, k, p);
    } else {
        if ((params.open_boundaries & 8u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Z neighbor
    if (k > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j, k - 1u, p);
    } else {
        if ((params.open_boundaries & 16u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Z neighbor
    if (k < params.depth - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j, k + 1u, p);
    } else {
        if ((params.open_boundaries & 32u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // Gauss-Seidel update with SOR
    // Poisson equation: (p_neighbors - 6p) / h² = -div
    // Rearranged: p = (sum_neighbors - h² * div) / 6
    let new_p = (sum_neighbors - divergence[idx] * params.h_sq) / 6.0;
    // SOR: weighted average of old and new values
    pressure[idx] = mix(pressure[idx], new_p, params.omega);
}

// Process "black" cells: (i + j + k) % 2 == 1
@compute @workgroup_size(8, 8, 4)
fn pressure_black(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (!is_valid(i, j, k)) {
        return;
    }

    // Skip non-black cells (red cells have (i+j+k) % 2 == 0)
    if (((i + j + k) % 2u) != 1u) {
        return;
    }

    let idx = get_index(i, j, k);

    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    let p = pressure[idx];

    // Gather neighbor pressures with proper boundary conditions
    var sum_neighbors = 0.0;

    // -X neighbor
    if (i > 0u) {
        sum_neighbors += get_neighbor_pressure(i - 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 1u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +X neighbor
    if (i < params.width - 1u) {
        sum_neighbors += get_neighbor_pressure(i + 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 2u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Y neighbor
    if (j > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j - 1u, k, p);
    } else {
        if ((params.open_boundaries & 4u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Y neighbor
    if (j < params.height - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j + 1u, k, p);
    } else {
        if ((params.open_boundaries & 8u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Z neighbor
    if (k > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j, k - 1u, p);
    } else {
        if ((params.open_boundaries & 16u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Z neighbor
    if (k < params.depth - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j, k + 1u, p);
    } else {
        if ((params.open_boundaries & 32u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // Poisson equation: p = (sum_neighbors - h² * div) / 6
    let new_p = (sum_neighbors - divergence[idx] * params.h_sq) / 6.0;
    pressure[idx] = mix(pressure[idx], new_p, params.omega);
}

// Compute divergence of velocity field
// This is typically called once before the pressure iterations
@compute @workgroup_size(8, 8, 4)
fn compute_divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (!is_valid(i, j, k)) {
        return;
    }

    let idx = get_index(i, j, k);

    if (cell_type[idx] != CELL_FLUID) {
        // Note: divergence array must be read_write for this entry point
        // but we're writing to it - caller should bind appropriately
        return;
    }

    // Divergence computed from staggered velocity grid would go here
    // But typically divergence is computed by a separate shader that has
    // access to the velocity grids. This entry point is a placeholder.
}

// Apply pressure gradient to velocity (subtract gradient from velocity)
// This is called after pressure solve to make velocity divergence-free
//
// Note: This requires separate entry points for U, V, W grids since they
// have different staggered layouts. See apply_pressure_gradient_*.wgsl
