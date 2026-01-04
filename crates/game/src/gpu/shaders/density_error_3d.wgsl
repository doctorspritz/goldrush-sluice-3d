// Density Error Shader (3D)
//
// Computes density error from particle count per cell.
// Output: density_error = (count - REST_COUNT) / REST_COUNT
// This is used as the RHS for a secondary pressure solve that
// pushes particles from crowded regions to empty regions.
//
// Based on: Kugelstadt et al. 2019 "Implicit Density Projection for Volume Conserving Liquids"

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    rest_density: f32,  // Target particles per cell (~8 for typical FLIP)
    dt: f32,  // Timestep for scaling (blub divides error by dt)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particle_count: array<i32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> density_error: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_AIR; }
    if (j < 0 || j >= i32(params.height)) { return CELL_AIR; }
    if (k < 0 || k >= i32(params.depth)) { return CELL_AIR; }
    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

@compute @workgroup_size(8, 8, 4)
fn compute_density_error(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);

    // Only compute for fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        density_error[idx] = 0.0;
        return;
    }

    // Start with particle count as base density
    // Note: blub computes actual density with trilinear weights, we use particle count
    var density = f32(particle_count[idx]);

    // Check neighbors for solid cells and air interfaces
    // blub contribution values: face neighbor = 0.5625 (0.75^3 / 0.75 = 0.5625 for each solid face)
    let solid_neighbor_contribution: f32 = 0.5625;

    let marker_px = get_cell_type(i32(i) + 1, i32(j), i32(k));
    let marker_mx = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let marker_py = get_cell_type(i32(i), i32(j) + 1, i32(k));
    let marker_my = get_cell_type(i32(i), i32(j) - 1, i32(k));
    let marker_pz = get_cell_type(i32(i), i32(j), i32(k) + 1);
    let marker_mz = get_cell_type(i32(i), i32(j), i32(k) - 1);

    // Add density contributions from solid neighbors (blub approach)
    // If there were particles in solid cells, they would contribute this much
    if (marker_px == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_mx == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_py == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_my == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_pz == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_mz == CELL_SOLID) { density += solid_neighbor_contribution; }

    // Check for air interface
    let has_air_neighbor = (marker_px == CELL_AIR || marker_mx == CELL_AIR ||
                            marker_py == CELL_AIR || marker_my == CELL_AIR ||
                            marker_pz == CELL_AIR || marker_mz == CELL_AIR);

    // Compute error: positive = sparse (need more), negative = crowded (too many)
    var error = 1.0 - density / params.rest_density;

    // CRITICAL: At air interface, clamp error to >= 0 (only sparse, not crowded)
    // This prevents surface cells from getting high pressure that would push particles DOWN.
    // Interior cells with high pressure will push particles TOWARD the surface instead.
    if (has_air_neighbor) {
        error = max(0.0, error);
    }

    // Clamp to prevent overshooting - don't correct more than 50% per step
    error = clamp(error, -0.5, 0.5);

    // Scale by 1/dt for consistency with pressure solver (blub approach)
    // This gets multiplied back by dt in position correction
    error /= params.dt;

    density_error[idx] = error;
}
