// Multigrid Restriction Operator
//
// Transfers residual from fine level to coarse level using full-weighting.
// Maps each coarse cell to a 2x2 block of fine cells, averaging fluid cells only.
//
// Also propagates cell types: coarse cell is FLUID if any fine cell is FLUID.

struct RestrictParams {
    fine_width: u32,
    fine_height: u32,
    coarse_width: u32,
    coarse_height: u32,
}

// Fine level (input)
@group(0) @binding(0) var<storage, read> fine_residual: array<f32>;
@group(0) @binding(1) var<storage, read> fine_cell_type: array<u32>;

// Coarse level (output)
@group(0) @binding(2) var<storage, read_write> coarse_divergence: array<f32>;
@group(0) @binding(3) var<storage, read_write> coarse_cell_type: array<u32>;

@group(0) @binding(4) var<uniform> params: RestrictParams;

// Cell type constants
const CELL_SOLID: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_AIR: u32 = 2u;

fn fine_index(i: u32, j: u32) -> u32 {
    return j * params.fine_width + i;
}

fn coarse_index(i: u32, j: u32) -> u32 {
    return j * params.coarse_width + i;
}

// Each coarse cell maps to a 2x2 block of fine cells
// Full-weighting restriction: average the fluid cells in the 2x2 block
@compute @workgroup_size(8, 8)
fn mg_restrict(@builtin(global_invocation_id) id: vec3<u32>) {
    let ci = id.x;  // Coarse cell x
    let cj = id.y;  // Coarse cell y

    if (ci >= params.coarse_width || cj >= params.coarse_height) {
        return;
    }

    // Fine cell coordinates (top-left of 2x2 block)
    let fi = ci * 2u;
    let fj = cj * 2u;

    // Accumulate residual from 2x2 block of fine cells
    var sum = 0.0;
    var count = 0u;
    var any_fluid = false;

    // Sample 2x2 block
    for (var dj = 0u; dj < 2u; dj++) {
        for (var di = 0u; di < 2u; di++) {
            let fii = fi + di;
            let fjj = fj + dj;

            // Bounds check
            if (fii < params.fine_width && fjj < params.fine_height) {
                let f_idx = fine_index(fii, fjj);
                let cell = fine_cell_type[f_idx];

                if (cell == CELL_FLUID) {
                    sum += fine_residual[f_idx];
                    count += 1u;
                    any_fluid = true;
                }
            }
        }
    }

    let c_idx = coarse_index(ci, cj);

    // Set coarse divergence as AVERAGE of fine residuals
    // This ensures restriction R and prolongation P satisfy R ≈ c·Pᵀ
    // Since prolongation uses bilinear interpolation (weights sum to 1),
    // restriction must also preserve amplitude via averaging
    if (count > 0u) {
        coarse_divergence[c_idx] = sum / f32(count);  // Average preserves amplitude
    } else {
        coarse_divergence[c_idx] = 0.0;
    }

    // Set coarse cell type: FLUID if any fine cell is FLUID, else AIR
    // (Solid boundaries are implicitly handled by the fine level)
    if (any_fluid) {
        coarse_cell_type[c_idx] = CELL_FLUID;
    } else {
        // Check if any fine cell was SOLID
        var any_solid = false;
        for (var dj = 0u; dj < 2u; dj++) {
            for (var di = 0u; di < 2u; di++) {
                let fii = fi + di;
                let fjj = fj + dj;
                if (fii < params.fine_width && fjj < params.fine_height) {
                    if (fine_cell_type[fine_index(fii, fjj)] == CELL_SOLID) {
                        any_solid = true;
                    }
                }
            }
        }
        coarse_cell_type[c_idx] = select(CELL_AIR, CELL_SOLID, any_solid);
    }
}
