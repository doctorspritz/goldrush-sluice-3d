// Pressure solver using Checkerboard SOR (Successive Over-Relaxation)
// Red-black ordering allows for parallel updates

struct Params {
    width: u32,
    height: u32,
    omega: f32,  // SOR relaxation factor (typically 1.8-1.95)
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

// Cell type constants (matching sim crate CellType enum order)
const CELL_SOLID: u32 = 0u;  // CellType::Solid
const CELL_FLUID: u32 = 1u;  // CellType::Fluid
const CELL_AIR: u32 = 2u;    // CellType::Air

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
        // Solid cells: copy pressure from fluid neighbor (Neumann BC)
        return 0.0;
    }
    return pressure[get_index(i, j)];
}

// Process "red" cells: (i + j) % 2 == 0
@compute @workgroup_size(8, 8)
fn pressure_red(@builtin(global_invocation_id) id: vec3<u32>) {
    // Map thread to red cell
    let thread_i = id.x;
    let j = id.y;

    // Compute actual i for red cells
    // Red cells have (i + j) % 2 == 0
    // For even j: i = 0, 2, 4, ... -> thread_i * 2
    // For odd j:  i = 1, 3, 5, ... -> thread_i * 2 + 1
    let i = thread_i * 2u + (j % 2u);

    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = get_index(i, j);

    // Skip non-fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    // Gather neighbor pressures with boundary handling
    var p_left = 0.0;
    var p_right = 0.0;
    var p_down = 0.0;
    var p_up = 0.0;
    var neighbor_count = 0.0;

    if (i > 0u && cell_type[get_index(i - 1u, j)] != CELL_SOLID) {
        p_left = get_pressure(i - 1u, j);
        neighbor_count += 1.0;
    }
    if (i < params.width - 1u && cell_type[get_index(i + 1u, j)] != CELL_SOLID) {
        p_right = get_pressure(i + 1u, j);
        neighbor_count += 1.0;
    }
    if (j > 0u && cell_type[get_index(i, j - 1u)] != CELL_SOLID) {
        p_down = get_pressure(i, j - 1u);
        neighbor_count += 1.0;
    }
    if (j < params.height - 1u && cell_type[get_index(i, j + 1u)] != CELL_SOLID) {
        p_up = get_pressure(i, j + 1u);
        neighbor_count += 1.0;
    }

    // Gauss-Seidel update with SOR
    if (neighbor_count > 0.0) {
        let sum_neighbors = p_left + p_right + p_down + p_up;
        let new_p = (sum_neighbors - divergence[idx]) / neighbor_count;

        // SOR: weighted average of old and new values
        pressure[idx] = mix(pressure[idx], new_p, params.omega);
    }
}

// Process "black" cells: (i + j) % 2 == 1
@compute @workgroup_size(8, 8)
fn pressure_black(@builtin(global_invocation_id) id: vec3<u32>) {
    let thread_i = id.x;
    let j = id.y;

    // Black cells have (i + j) % 2 == 1
    // For even j: i = 1, 3, 5, ... -> thread_i * 2 + 1
    // For odd j:  i = 0, 2, 4, ... -> thread_i * 2
    let i = thread_i * 2u + 1u - (j % 2u);

    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = get_index(i, j);

    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    var p_left = 0.0;
    var p_right = 0.0;
    var p_down = 0.0;
    var p_up = 0.0;
    var neighbor_count = 0.0;

    if (i > 0u && cell_type[get_index(i - 1u, j)] != CELL_SOLID) {
        p_left = get_pressure(i - 1u, j);
        neighbor_count += 1.0;
    }
    if (i < params.width - 1u && cell_type[get_index(i + 1u, j)] != CELL_SOLID) {
        p_right = get_pressure(i + 1u, j);
        neighbor_count += 1.0;
    }
    if (j > 0u && cell_type[get_index(i, j - 1u)] != CELL_SOLID) {
        p_down = get_pressure(i, j - 1u);
        neighbor_count += 1.0;
    }
    if (j < params.height - 1u && cell_type[get_index(i, j + 1u)] != CELL_SOLID) {
        p_up = get_pressure(i, j + 1u);
        neighbor_count += 1.0;
    }

    if (neighbor_count > 0.0) {
        let sum_neighbors = p_left + p_right + p_down + p_up;
        let new_p = (sum_neighbors - divergence[idx]) / neighbor_count;
        pressure[idx] = mix(pressure[idx], new_p, params.omega);
    }
}
