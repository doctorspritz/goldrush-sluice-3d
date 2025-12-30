// Multigrid Residual Computation
//
// Computes residual r = b - Ax where:
// - b is the divergence (RHS)
// - A is the Laplacian operator
// - x is the current pressure
//
// This is needed to transfer error to coarser levels in the V-cycle.

struct LevelParams {
    width: u32,
    height: u32,
    level: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> residual: array<f32>;
@group(0) @binding(4) var<uniform> params: LevelParams;

// Cell type constants
const CELL_SOLID: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_AIR: u32 = 2u;

fn get_index(i: u32, j: u32) -> u32 {
    return j * params.width + i;
}

fn get_pressure(i: u32, j: u32) -> f32 {
    if (i >= params.width || j >= params.height) {
        return 0.0;
    }
    let ct = cell_type[get_index(i, j)];
    if (ct == CELL_SOLID) {
        return 0.0;  // Neumann BC
    }
    return pressure[get_index(i, j)];
}

// Compute Laplacian(pressure) at cell (i, j)
// Uses the same stencil as the smoother: variable neighbor count for Neumann BCs
fn laplacian(i: u32, j: u32) -> f32 {
    let idx = get_index(i, j);
    let p_center = pressure[idx];

    var lap = 0.0;
    var neighbor_count = 0.0;

    // Left
    if (i > 0u) {
        let left_type = cell_type[get_index(i - 1u, j)];
        if (left_type != CELL_SOLID) {
            lap += get_pressure(i - 1u, j) - p_center;
            neighbor_count += 1.0;
        }
    }

    // Right
    if (i < params.width - 1u) {
        let right_type = cell_type[get_index(i + 1u, j)];
        if (right_type != CELL_SOLID) {
            lap += get_pressure(i + 1u, j) - p_center;
            neighbor_count += 1.0;
        }
    }

    // Down
    if (j > 0u) {
        let down_type = cell_type[get_index(i, j - 1u)];
        if (down_type != CELL_SOLID) {
            lap += get_pressure(i, j - 1u) - p_center;
            neighbor_count += 1.0;
        }
    }

    // Up
    if (j < params.height - 1u) {
        let up_type = cell_type[get_index(i, j + 1u)];
        if (up_type != CELL_SOLID) {
            lap += get_pressure(i, j + 1u) - p_center;
            neighbor_count += 1.0;
        }
    }

    // If no neighbors, Laplacian is 0
    // For interior fluid cells, this simplifies to:
    // lap = (p_left + p_right + p_down + p_up - 4*p_center)
    // but we use variable neighbor count for Neumann BCs
    return lap;
}

// Compute residual r = b - Ax at each cell
@compute @workgroup_size(8, 8)
fn compute_residual(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;

    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = get_index(i, j);

    // Only compute residual for fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        residual[idx] = 0.0;
        return;
    }

    // r = b - Ax
    // For the pressure equation: Laplacian(p) = div
    // So: r = div - Laplacian(p)
    //
    // But wait - our Gauss-Seidel update is:
    //   p_new = (sum_neighbors - div) / neighbor_count
    // Which solves: neighbor_count * p = sum_neighbors - div
    // Or: sum_neighbors - neighbor_count * p = div
    //
    // The residual for this is: r = div - (sum_neighbors - neighbor_count * p)
    //                             = div - Laplacian(p) where Laplacian uses this form
    //
    // Using our laplacian function which computes sum_neighbors - neighbor_count * p:
    residual[idx] = divergence[idx] - laplacian(i, j);
}

// Clear a buffer to zero (used to initialize coarse pressure before V-cycle)
@compute @workgroup_size(256)
fn clear_buffer(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total_cells = params.width * params.height;

    if (idx >= total_cells) {
        return;
    }

    residual[idx] = 0.0;  // We reuse the residual binding for clearing any buffer
}
