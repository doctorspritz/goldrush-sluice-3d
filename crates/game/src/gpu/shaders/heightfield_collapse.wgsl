// heightfield_collapse.wgsl
// Slope collapse / angle of repose mechanics

struct Params {
    width: u32,
    depth: u32,
    _pad0: vec2<u32>,
    cell_size: f32,
    dt: f32,
    gravity: f32,
    damping: f32,
}

// Per-material maximum slope (tan of angle of repose)
const TAN_ANGLE_SEDIMENT: f32 = 0.577;      // tan(30°) - loose deposited material
const TAN_ANGLE_OVERBURDEN: f32 = 0.700;    // tan(35°) - soil/dirt
const TAN_ANGLE_GRAVEL: f32 = 1.0;          // tan(45°) - gravel (steep piles)

// Collapse rate - how fast material transfers when over limit
const K_COLLAPSE: f32 = 0.5;

@group(0) @binding(0) var<uniform> params: Params;

// Terrain layers (must match heightfield.rs binding order)
@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.width + x;
}

fn get_ground_height(idx: u32) -> f32 {
    return bedrock[idx] + paydirt[idx] + gravel[idx] + overburden[idx] + sediment[idx];
}

// Check if material can support current slope, return excess material to transfer
fn get_collapse_amount(slope: f32, tan_limit: f32, available: f32) -> f32 {
    if (slope <= tan_limit || available < 0.001) {
        return 0.0;
    }
    // Transfer proportional to how much slope exceeds limit
    let excess_slope = slope - tan_limit;
    return min(excess_slope * K_COLLAPSE * params.dt * params.cell_size, available * 0.25);
}

@compute @workgroup_size(16, 16)
fn update_collapse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    if (x >= params.width || z >= params.depth) { return; }
    
    let idx = get_idx(x, z);
    let my_height = get_ground_height(idx);
    
    // Check all 4 neighbors
    var total_transfer_sed = 0.0;
    var total_transfer_ob = 0.0;
    var total_transfer_gr = 0.0;
    
    // Neighbor offsets: left, right, back, forward
    let offsets = array<vec2<i32>, 4>(
        vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1)
    );
    
    for (var i = 0u; i < 4u; i++) {
        let nx = i32(x) + offsets[i].x;
        let nz = i32(z) + offsets[i].y;
        
        // Bounds check
        if (nx < 0 || nx >= i32(params.width) || nz < 0 || nz >= i32(params.depth)) {
            continue;
        }
        
        let nidx = get_idx(u32(nx), u32(nz));
        let neighbor_height = get_ground_height(nidx);
        
        // Only collapse downhill (from higher to lower)
        if (neighbor_height >= my_height) {
            continue;
        }
        
        let height_diff = my_height - neighbor_height;
        let slope = height_diff / params.cell_size;
        
        // Check each material layer (top to bottom)
        // Material only collapses if slope exceeds its angle of repose
        
        // 1. Sediment (loosest - collapses first)
        let sed_avail = sediment[idx];
        let sed_transfer = get_collapse_amount(slope, TAN_ANGLE_SEDIMENT, sed_avail);
        if (sed_transfer > 0.0) {
            total_transfer_sed += sed_transfer;
        }
        
        // 2. Overburden (soil)
        let ob_avail = overburden[idx];
        let ob_transfer = get_collapse_amount(slope, TAN_ANGLE_OVERBURDEN, ob_avail);
        if (ob_transfer > 0.0 && sed_avail < 0.01) { // Only collapse if sediment is depleted
            total_transfer_ob += ob_transfer;
        }
        
        // 3. Gravel (resistant - highest angle)
        let gr_avail = gravel[idx];
        let gr_transfer = get_collapse_amount(slope, TAN_ANGLE_GRAVEL, gr_avail);
        if (gr_transfer > 0.0 && ob_avail < 0.01 && sed_avail < 0.01) {
            total_transfer_gr += gr_transfer;
        }
    }
    
    // Apply transfers (divide by 4 since we checked 4 neighbors)
    // Note: This is a simplified model - proper implementation would need double buffering
    // to avoid race conditions. For now, we limit transfer rate to be stable.
    let transfer_sed = min(total_transfer_sed / 4.0, sediment[idx] * 0.1);
    let transfer_ob = min(total_transfer_ob / 4.0, overburden[idx] * 0.1);
    let transfer_gr = min(total_transfer_gr / 4.0, gravel[idx] * 0.1);
    
    // Remove from this cell (material will appear in neighbors via conservation)
    // This is an approximation - true mass conservation would need scatter/gather pattern
    sediment[idx] -= transfer_sed;
    overburden[idx] -= transfer_ob;
    gravel[idx] -= transfer_gr;
    
    // Add to lowest neighbor (simple approximation)
    // Find lowest neighbor
    var lowest_idx = idx;
    var lowest_height = my_height;
    
    for (var i = 0u; i < 4u; i++) {
        let nx = i32(x) + offsets[i].x;
        let nz = i32(z) + offsets[i].y;
        
        if (nx < 0 || nx >= i32(params.width) || nz < 0 || nz >= i32(params.depth)) {
            continue;
        }
        
        let nidx = get_idx(u32(nx), u32(nz));
        let nh = get_ground_height(nidx);
        if (nh < lowest_height) {
            lowest_height = nh;
            lowest_idx = nidx;
        }
    }
    
    if (lowest_idx != idx) {
        // Collapsed material becomes sediment in the lower cell
        sediment[lowest_idx] += transfer_sed + transfer_ob + transfer_gr;
    }
}
