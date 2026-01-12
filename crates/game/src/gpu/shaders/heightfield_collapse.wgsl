// heightfield_collapse.wgsl
// Red-black checkerboard slope-limited collapse with 8-neighbor distribution
// Material flows to ALL lower neighbors proportionally when slope exceeds angle of repose.
// Uses red-black pattern to prevent race conditions.

struct Params {
    world_width: u32,
    world_depth: u32,
    tile_width: u32,
    tile_depth: u32,
    origin_x: u32,
    origin_z: u32,
    _pad0: vec2<u32>,
    cell_size: f32,
    dt: f32,
    gravity: f32,
    damping: f32,
}

// Per-material maximum slope (tan of angle of repose)
const TAN_ANGLE_SEDIMENT: f32 = 0.6249;     // tan(32°)
const TAN_ANGLE_OVERBURDEN: f32 = 0.7002;   // tan(35°)
const TAN_ANGLE_GRAVEL: f32 = 0.7813;       // tan(38°)

// Transfer rate: fraction of excess transferred per second
const COLLAPSE_RATE: f32 = 4.0;

// Diagonal distance factor (sqrt(2))
const DIAG_DIST: f32 = 1.41421356;

@group(0) @binding(0) var<uniform> params: Params;

@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;
@group(2) @binding(5) var<storage, read_write> surface_material: array<u32>; // 0=bed,1=pay,2=gravel,3=over,4=sed

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.world_width + x;
}

fn get_ground_height(idx: u32) -> f32 {
    return bedrock[idx] + paydirt[idx] + gravel[idx] + overburden[idx] + sediment[idx];
}

// Determine exposed surface material
fn compute_surface_material(idx: u32) -> u32 {
    let min_thick = 0.001;
    if (sediment[idx] > min_thick) { return 4u; }
    if (overburden[idx] > min_thick) { return 3u; }
    if (gravel[idx] > min_thick) { return 2u; }
    if (paydirt[idx] > min_thick) { return 1u; }
    return 0u; // bedrock
}

// Red phase: (x + z) % 2 == 0
@compute @workgroup_size(16, 16)
fn update_collapse_red(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    // Red phase: only process cells where (x + z) is even
    if ((x + z) % 2u != 0u) { return; }

    collapse_cell(x, z);
}

// Black phase: (x + z) % 2 == 1
@compute @workgroup_size(16, 16)
fn update_collapse_black(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    // Black phase: only process cells where (x + z) is odd
    if ((x + z) % 2u != 1u) { return; }

    collapse_cell(x, z);
}

// Legacy entry point for backwards compatibility (processes all cells, has race conditions)
@compute @workgroup_size(16, 16)
fn update_collapse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    collapse_cell(x, z);
}

fn collapse_cell(x: u32, z: u32) {
    let idx = get_idx(x, z);
    let my_height = get_ground_height(idx);

    // 8 neighbors: 4 cardinal + 4 diagonal
    // Offsets and distance factors (diagonals are sqrt(2) further away)
    let offsets = array<vec2<i32>, 8>(
        vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1),  // cardinal
        vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1) // diagonal
    );
    let dist_factors = array<f32, 8>(
        1.0, 1.0, 1.0, 1.0,                // cardinal
        DIAG_DIST, DIAG_DIST, DIAG_DIST, DIAG_DIST  // diagonal
    );

    // Collect lower neighbors and their height deficits
    var lower_indices: array<u32, 8>;
    var lower_heights: array<f32, 8>;
    var lower_dists: array<f32, 8>;
    var num_lower = 0u;
    var total_deficit = 0.0;

    for (var i = 0u; i < 8u; i++) {
        let nx = i32(x) + offsets[i].x;
        let nz = i32(z) + offsets[i].y;

        if (nx < 0 || nx >= i32(params.world_width) || nz < 0 || nz >= i32(params.world_depth)) {
            continue;
        }

        let nidx = get_idx(u32(nx), u32(nz));
        let nh = get_ground_height(nidx);

        if (nh < my_height) {
            lower_indices[num_lower] = nidx;
            lower_heights[num_lower] = nh;
            lower_dists[num_lower] = dist_factors[i];
            let deficit = my_height - nh;
            total_deficit += deficit / dist_factors[i]; // Weight by inverse distance
            num_lower += 1u;
        }
    }

    // No lower neighbors
    if (num_lower == 0u) {
        return;
    }

    // Get material at this cell
    let my_sed = sediment[idx];
    let my_ob = overburden[idx];
    let my_gr = gravel[idx];

    // Transfer sediment if any neighbor exceeds angle of repose
    if (my_sed > 0.001) {
        var sed_to_transfer = 0.0;

        for (var i = 0u; i < num_lower; i++) {
            let height_diff = my_height - lower_heights[i];
            let effective_cell_size = params.cell_size * lower_dists[i];
            let slope = height_diff / effective_cell_size;

            if (slope > TAN_ANGLE_SEDIMENT) {
                let stable_diff = TAN_ANGLE_SEDIMENT * effective_cell_size;
                let excess = (height_diff - stable_diff) * 0.5;
                // Weight transfer by height deficit (steeper neighbors get more)
                let weight = (my_height - lower_heights[i]) / (total_deficit * lower_dists[i]);
                let transfer = min(excess * weight * COLLAPSE_RATE * params.dt, my_sed * 0.5 * weight);

                if (transfer > 0.0) {
                    sediment[lower_indices[i]] += transfer;
                    sed_to_transfer += transfer;
                    surface_material[lower_indices[i]] = 4u;
                }
            }
        }

        if (sed_to_transfer > 0.0) {
            sediment[idx] -= min(sed_to_transfer, my_sed);
        }
    }

    // Transfer overburden if sediment is depleted
    if (my_ob > 0.001 && sediment[idx] < 0.005) {
        var ob_to_transfer = 0.0;

        for (var i = 0u; i < num_lower; i++) {
            let height_diff = my_height - lower_heights[i];
            let effective_cell_size = params.cell_size * lower_dists[i];
            let slope = height_diff / effective_cell_size;

            if (slope > TAN_ANGLE_OVERBURDEN) {
                let stable_diff = TAN_ANGLE_OVERBURDEN * effective_cell_size;
                let excess = (height_diff - stable_diff) * 0.5;
                let weight = (my_height - lower_heights[i]) / (total_deficit * lower_dists[i]);
                let transfer = min(excess * weight * COLLAPSE_RATE * params.dt, my_ob * 0.5 * weight);

                if (transfer > 0.0) {
                    overburden[lower_indices[i]] += transfer;
                    ob_to_transfer += transfer;
                    surface_material[lower_indices[i]] = 3u;
                }
            }
        }

        if (ob_to_transfer > 0.0) {
            overburden[idx] -= min(ob_to_transfer, my_ob);
        }
    }

    // Transfer gravel if upper layers are depleted
    if (my_gr > 0.001 && sediment[idx] < 0.005 && overburden[idx] < 0.005) {
        var gr_to_transfer = 0.0;

        for (var i = 0u; i < num_lower; i++) {
            let height_diff = my_height - lower_heights[i];
            let effective_cell_size = params.cell_size * lower_dists[i];
            let slope = height_diff / effective_cell_size;

            if (slope > TAN_ANGLE_GRAVEL) {
                let stable_diff = TAN_ANGLE_GRAVEL * effective_cell_size;
                let excess = (height_diff - stable_diff) * 0.5;
                let weight = (my_height - lower_heights[i]) / (total_deficit * lower_dists[i]);
                let transfer = min(excess * weight * COLLAPSE_RATE * params.dt, my_gr * 0.5 * weight);

                if (transfer > 0.0) {
                    gravel[lower_indices[i]] += transfer;
                    gr_to_transfer += transfer;
                    surface_material[lower_indices[i]] = 2u;
                }
            }
        }

        if (gr_to_transfer > 0.0) {
            gravel[idx] -= min(gr_to_transfer, my_gr);
        }
    }

    // Update our own surface material after losing material
    surface_material[idx] = compute_surface_material(idx);
}
