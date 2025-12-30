// Red-Black Gauss-Seidel Smoother for Multigrid
//
// This smoother operates on any level of the multigrid hierarchy.
// Uses pure Gauss-Seidel (omega=1.0) as a smoother for multigrid,
// rather than SOR which can be unstable as a preconditioner.
//
// Solves: Laplacian(p) = div (Poisson equation)
// Where Laplacian uses the 5-point stencil with Neumann BCs at solid boundaries.

struct LevelParams {
    width: u32,
    height: u32,
    level: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<uniform> params: LevelParams;

// Cell type constants (matching sim crate CellType enum order)
const CELL_SOLID: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_AIR: u32 = 2u;

fn get_index(i: u32, j: u32) -> u32 {
    return j * params.width + i;
}

fn is_fluid(i: u32, j: u32) -> bool {
    if (i >= params.width || j >= params.height) {
        return false;
    }
    return cell_type[get_index(i, j)] == CELL_FLUID;
}

fn get_pressure(i: u32, j: u32) -> f32 {
    if (i >= params.width || j >= params.height) {
        return 0.0;
    }
    let ct = cell_type[get_index(i, j)];
    if (ct == CELL_SOLID) {
        return 0.0;  // Neumann BC: dp/dn = 0 (handled by neighbor count)
    }
    return pressure[get_index(i, j)];
}

// Gauss-Seidel update for a single cell
// Returns the updated pressure value
fn gs_update(i: u32, j: u32) {
    let idx = get_index(i, j);

    // Skip non-fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    // Gather neighbor pressures with Neumann BC handling
    var sum_neighbors = 0.0;
    var neighbor_count = 0.0;

    // Left neighbor
    if (i > 0u) {
        let left_type = cell_type[get_index(i - 1u, j)];
        if (left_type != CELL_SOLID) {
            sum_neighbors += get_pressure(i - 1u, j);
            neighbor_count += 1.0;
        }
    }

    // Right neighbor
    if (i < params.width - 1u) {
        let right_type = cell_type[get_index(i + 1u, j)];
        if (right_type != CELL_SOLID) {
            sum_neighbors += get_pressure(i + 1u, j);
            neighbor_count += 1.0;
        }
    }

    // Down neighbor
    if (j > 0u) {
        let down_type = cell_type[get_index(i, j - 1u)];
        if (down_type != CELL_SOLID) {
            sum_neighbors += get_pressure(i, j - 1u);
            neighbor_count += 1.0;
        }
    }

    // Up neighbor
    if (j < params.height - 1u) {
        let up_type = cell_type[get_index(i, j + 1u)];
        if (up_type != CELL_SOLID) {
            sum_neighbors += get_pressure(i, j + 1u);
            neighbor_count += 1.0;
        }
    }

    // Gauss-Seidel update: p = (sum_neighbors - div) / neighbor_count
    if (neighbor_count > 0.0) {
        pressure[idx] = (sum_neighbors - divergence[idx]) / neighbor_count;
    }
}

// Process "red" cells: (i + j) % 2 == 0
// Red cells are updated first, using values from black neighbors (from previous iteration)
@compute @workgroup_size(8, 8)
fn smooth_red(@builtin(global_invocation_id) id: vec3<u32>) {
    let thread_i = id.x;
    let j = id.y;

    // Map thread to red cell coordinates
    // Red cells have (i + j) % 2 == 0
    // For even j: i = 0, 2, 4, ... -> thread_i * 2
    // For odd j:  i = 1, 3, 5, ... -> thread_i * 2 + 1
    let i = thread_i * 2u + (j % 2u);

    if (i >= params.width || j >= params.height) {
        return;
    }

    gs_update(i, j);
}

// Process "black" cells: (i + j) % 2 == 1
// Black cells are updated second, using fresh values from red neighbors
@compute @workgroup_size(8, 8)
fn smooth_black(@builtin(global_invocation_id) id: vec3<u32>) {
    let thread_i = id.x;
    let j = id.y;

    // Map thread to black cell coordinates
    // Black cells have (i + j) % 2 == 1
    // For even j: i = 1, 3, 5, ... -> thread_i * 2 + 1
    // For odd j:  i = 0, 2, 4, ... -> thread_i * 2
    let i = thread_i * 2u + 1u - (j % 2u);

    if (i >= params.width || j >= params.height) {
        return;
    }

    gs_update(i, j);
}
