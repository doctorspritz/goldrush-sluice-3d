// Density Position Grid Shader (3D)
//
// Computes position changes from density-derived pressure and stores on grid.
// This is the FIRST pass of density projection (blub approach).
//
// Position change for each component uses FORWARD difference:
// - delta_x[i,j,k] = (pressure[i+1,j,k] - pressure[i,j,k]) * dt
// - delta_y[i,j,k] = (pressure[i,j+1,k] - pressure[i,j,k]) * dt
// - delta_z[i,j,k] = (pressure[i,j,k+1] - pressure[i,j,k]) * dt
//
// Particles will sample with offset for proper face-centered behavior.
//
// Based on: blub/density_projection_position_change.comp

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> position_delta_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> position_delta_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> position_delta_z: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn sample_pressure(i: i32, j: i32, k: i32) -> f32 {
    // Out of bounds = 0 pressure
    if (i < 0 || i >= i32(params.width)) { return 0.0; }
    if (j < 0 || j >= i32(params.height)) { return 0.0; }
    if (k < 0 || k >= i32(params.depth)) { return 0.0; }

    let idx = cell_index(u32(i), u32(j), u32(k));
    let ct = cell_type[idx];

    // Fluid cells have pressure, others have 0
    if (ct == CELL_FLUID) {
        return pressure[idx];
    }
    return 0.0;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_AIR; }
    if (j < 0 || j >= i32(params.height)) { return CELL_AIR; }
    if (k < 0 || k >= i32(params.depth)) { return CELL_AIR; }
    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

fn compute_position_change(center_type: u32, center_pressure: f32,
                           neighbor_i: i32, neighbor_j: i32, neighbor_k: i32) -> f32 {
    let neighbor_type = get_cell_type(neighbor_i, neighbor_j, neighbor_k);
    let neighbor_pressure = sample_pressure(neighbor_i, neighbor_j, neighbor_k);

    // Position change = (neighbor_pressure - center_pressure) * dt
    // This pushes particles from high pressure (crowded) to low pressure (sparse)
    var pos_change = (neighbor_pressure - center_pressure) * params.dt;

    // Neumann boundary conditions (from blub):
    // - Solid-solid: no flow
    // - Fluid-solid or solid-fluid: no penetration
    if (center_type == CELL_SOLID || neighbor_type == CELL_SOLID) {
        pos_change = 0.0;
    }

    return pos_change;
}

@compute @workgroup_size(8, 8, 4)
fn compute_position_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    let center_type = cell_type[idx];
    let center_pressure = sample_pressure(i32(i), i32(j), i32(k));

    // Compute position changes for each direction using FORWARD difference
    // (neighbor at +1 minus center)
    let delta_x = compute_position_change(center_type, center_pressure, i32(i) + 1, i32(j), i32(k));
    let delta_y = compute_position_change(center_type, center_pressure, i32(i), i32(j) + 1, i32(k));
    let delta_z = compute_position_change(center_type, center_pressure, i32(i), i32(j), i32(k) + 1);

    position_delta_x[idx] = delta_x;
    position_delta_y[idx] = delta_y;
    position_delta_z[idx] = delta_z;
}
