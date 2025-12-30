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

// Compute Laplacian(pressure) at cell (i, j) using fixed 4-neighbor stencil
// Uses Neumann BC: dp/dn = 0, implemented by mirroring pressure at boundaries/solids
// Must match the stencil used by smoother (mg_smooth.wgsl) and PCG (pcg_ops.wgsl)
fn laplacian(i: u32, j: u32) -> f32 {
    let idx = get_index(i, j);
    let p_center = pressure[idx];

    // Gather neighbor pressures with Neumann BC (mirror at solid/boundary)
    // Always use 4 neighbors for consistent stencil

    // Left - mirror if at boundary or solid
    var p_left = p_center;
    if (i > 0u) {
        let left_idx = get_index(i - 1u, j);
        if (cell_type[left_idx] != CELL_SOLID) {
            p_left = pressure[left_idx];
        }
    }

    // Right - mirror if at boundary or solid
    var p_right = p_center;
    if (i < params.width - 1u) {
        let right_idx = get_index(i + 1u, j);
        if (cell_type[right_idx] != CELL_SOLID) {
            p_right = pressure[right_idx];
        }
    }

    // Down - mirror if at boundary or solid
    var p_down = p_center;
    if (j > 0u) {
        let down_idx = get_index(i, j - 1u);
        if (cell_type[down_idx] != CELL_SOLID) {
            p_down = pressure[down_idx];
        }
    }

    // Up - mirror if at boundary or solid
    var p_up = p_center;
    if (j < params.height - 1u) {
        let up_idx = get_index(i, j + 1u);
        if (cell_type[up_idx] != CELL_SOLID) {
            p_up = pressure[up_idx];
        }
    }

    // Laplacian = (p_L + p_R + p_D + p_U - 4*p_center)
    return p_left + p_right + p_down + p_up - 4.0 * p_center;
}

// Compute residual r = b - Ax at each cell
// For the pressure equation: Laplacian(p) = div
// Residual: r = div - Laplacian(p)
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

    // r = b - Ax = div - Laplacian(p)
    // Uses fixed 4-neighbor Laplacian with Neumann BC mirroring
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
