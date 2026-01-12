// heightfield_collapse.wgsl
// Simple slope-limited collapse - material flows to lowest neighbor when too steep

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

@compute @workgroup_size(16, 16)
fn update_collapse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    let idx = get_idx(x, z);
    let my_height = get_ground_height(idx);

    // Find lowest neighbor
    let offsets = array<vec2<i32>, 4>(
        vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1)
    );

    var lowest_idx = idx;
    var lowest_height = my_height;

    for (var i = 0u; i < 4u; i++) {
        let nx = i32(x) + offsets[i].x;
        let nz = i32(z) + offsets[i].y;

        if (nx < 0 || nx >= i32(params.world_width) || nz < 0 || nz >= i32(params.world_depth)) {
            continue;
        }

        let nidx = get_idx(u32(nx), u32(nz));
        let nh = get_ground_height(nidx);

        if (nh < lowest_height) {
            lowest_height = nh;
            lowest_idx = nidx;
        }
    }

    // No lower neighbor
    if (lowest_idx == idx) {
        return;
    }

    let height_diff = my_height - lowest_height;
    let slope = height_diff / params.cell_size;

    // Get material at this cell
    let my_sed = sediment[idx];
    let my_ob = overburden[idx];
    let my_gr = gravel[idx];

    // Transfer sediment if slope exceeds its angle of repose
    // When material flows to destination, it lands ON TOP - update surface_material
    if (slope > TAN_ANGLE_SEDIMENT && my_sed > 0.001) {
        let stable_diff = TAN_ANGLE_SEDIMENT * params.cell_size;
        let excess = (height_diff - stable_diff) * 0.5;
        let transfer = min(excess * 0.2, my_sed * 0.15);
        if (transfer > 0.0) {
            sediment[idx] -= transfer;
            sediment[lowest_idx] += transfer;
            surface_material[lowest_idx] = 4u; // Sediment landed on top
        }
    }

    // Transfer overburden if slope exceeds its angle (and sediment is depleted)
    if (slope > TAN_ANGLE_OVERBURDEN && my_ob > 0.001 && my_sed < 0.005) {
        let stable_diff = TAN_ANGLE_OVERBURDEN * params.cell_size;
        let excess = (height_diff - stable_diff) * 0.5;
        let transfer = min(excess * 0.2, my_ob * 0.15);
        if (transfer > 0.0) {
            overburden[idx] -= transfer;
            overburden[lowest_idx] += transfer;
            surface_material[lowest_idx] = 3u; // Overburden landed on top
        }
    }

    // Transfer gravel if slope exceeds its angle (and upper layers depleted)
    if (slope > TAN_ANGLE_GRAVEL && my_gr > 0.001 && my_sed < 0.005 && my_ob < 0.005) {
        let stable_diff = TAN_ANGLE_GRAVEL * params.cell_size;
        let excess = (height_diff - stable_diff) * 0.5;
        let transfer = min(excess * 0.2, my_gr * 0.15);
        if (transfer > 0.0) {
            gravel[idx] -= transfer;
            gravel[lowest_idx] += transfer;
            surface_material[lowest_idx] = 2u; // Gravel landed on top
        }
    }
}
