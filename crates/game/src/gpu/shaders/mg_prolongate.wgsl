// Multigrid Prolongation Operator
//
// Transfers correction from coarse level to fine level using bilinear interpolation.
// Adds the interpolated coarse correction to the fine pressure (not replacing it).
//
// Each fine cell samples from the 4 nearest coarse cells using bilinear weights.

struct ProlongateParams {
    fine_width: u32,
    fine_height: u32,
    coarse_width: u32,
    coarse_height: u32,
}

// Coarse level (input) - the correction computed at coarse level
@group(0) @binding(0) var<storage, read> coarse_pressure: array<f32>;

// Fine level (output) - pressure to be updated with correction
@group(0) @binding(1) var<storage, read_write> fine_pressure: array<f32>;

// Fine cell types - only update fluid cells
@group(0) @binding(2) var<storage, read> fine_cell_type: array<u32>;

@group(0) @binding(3) var<uniform> params: ProlongateParams;

// Cell type constants (must match main 3D shaders: AIR=0, FLUID=1, SOLID=2)
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn fine_index(i: u32, j: u32) -> u32 {
    return j * params.fine_width + i;
}

fn coarse_index(i: u32, j: u32) -> u32 {
    return j * params.coarse_width + i;
}

fn get_coarse_pressure(i: i32, j: i32) -> f32 {
    // Clamp to valid range
    let ci = u32(clamp(i, 0, i32(params.coarse_width) - 1));
    let cj = u32(clamp(j, 0, i32(params.coarse_height) - 1));
    return coarse_pressure[coarse_index(ci, cj)];
}

// Bilinear interpolation from coarse grid to fine grid
// Each fine cell at (fi, fj) maps to coarse coordinate (fi/2, fj/2)
// We interpolate between the 4 nearest coarse cells
@compute @workgroup_size(8, 8)
fn prolongate(@builtin(global_invocation_id) id: vec3<u32>) {
    let fi = id.x;  // Fine cell x
    let fj = id.y;  // Fine cell y

    if (fi >= params.fine_width || fj >= params.fine_height) {
        return;
    }

    let f_idx = fine_index(fi, fj);

    // Only update fluid cells
    if (fine_cell_type[f_idx] != CELL_FLUID) {
        return;
    }

    // Map fine cell to coarse coordinates
    // Fine cell (fi, fj) corresponds to coarse position (fi/2, fj/2)
    // We use cell-centered coordinates, so fine cell center at (fi + 0.5)
    // maps to coarse coordinate ((fi + 0.5) / 2 - 0.5) = fi/2 - 0.25
    //
    // For simplicity, use direct mapping: coarse cell at (ci, cj) covers
    // fine cells (ci*2, cj*2), (ci*2+1, cj*2), (ci*2, cj*2+1), (ci*2+1, cj*2+1)
    //
    // For bilinear interpolation, we find the 4 coarse cells that bracket this fine cell

    // Coarse coordinate (continuous)
    let cx = f32(fi) * 0.5;
    let cy = f32(fj) * 0.5;

    // Integer coarse indices
    let ci0 = i32(floor(cx));
    let cj0 = i32(floor(cy));

    // Fractional parts for interpolation weights
    let tx = cx - f32(ci0);
    let ty = cy - f32(cj0);

    // Sample 4 corners
    let p00 = get_coarse_pressure(ci0, cj0);
    let p10 = get_coarse_pressure(ci0 + 1, cj0);
    let p01 = get_coarse_pressure(ci0, cj0 + 1);
    let p11 = get_coarse_pressure(ci0 + 1, cj0 + 1);

    // Bilinear interpolation
    let correction = (1.0 - tx) * (1.0 - ty) * p00
                   + tx * (1.0 - ty) * p10
                   + (1.0 - tx) * ty * p01
                   + tx * ty * p11;

    // Add correction to fine pressure
    fine_pressure[f_idx] += correction;
}
