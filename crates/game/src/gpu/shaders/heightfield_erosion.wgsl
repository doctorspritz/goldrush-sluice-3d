// heightfield_erosion.wgsl
// Hydraulic Erosion and Sediment Transport

struct Params {
    width: u32,
    depth: u32,
    _pad0: vec2<u32>,
    cell_size: f32,
    dt: f32,
    gravity: f32,
    damping: f32, // Reusing water params struct for now, need specific erosion params?
                  // Let's assume we pass a larger struct or a second uniform buffer.
                  // For simplicity, let's hardcode constants for now or extend Params.
}

// Extended Params for Erosion
// We'll stick to the common Params struct but maybe add fields or use constants.
const K_ENTRAIN: f32 = 0.1;           // 10x faster erosion for visible effect
const K_DEPOSIT: f32 = 0.5;           // Moderate deposition
const K_HARDNESS_BEDROCK: f32 = 0.0;  // Bedrock doesn't erode
const K_HARDNESS_PAYDIRT: f32 = 0.2;  // Slightly harder
const K_HARDNESS_OVERBURDEN: f32 = 0.8; // Berms erode relatively easily
const CAPACITY_FACTOR: f32 = 0.5;     // Higher carrying capacity
const MAX_CAPACITY: f32 = 1.0;        // Allow more sediment per cell

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> water_depth: array<f32>;
@group(1) @binding(1) var<storage, read_write> water_velocity_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> water_velocity_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> water_surface: array<f32>; // Not used directly but present in layout
@group(1) @binding(4) var<storage, read_write> flux_x: array<f32>; // Not used
@group(1) @binding(5) var<storage, read_write> flux_z: array<f32>; // Not used
@group(1) @binding(6) var<storage, read_write> suspended_sediment: array<f32>; // NEW binding for suspended

@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(3) var<storage, read_write> sediment: array<f32>;

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.width + x;
}

// 1. Erosion & Deposition
@compute @workgroup_size(16, 16)
fn update_erosion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    if (x >= params.width || z >= params.depth) { return; }
    
    let idx = get_idx(x, z);
    
    // Calculate Velocity Magnitude
    let vel_x = water_velocity_x[idx]; // Center velocity (approx from edges)
    let vel_z = water_velocity_z[idx];
    // Correct center velocity logic would average edges.
    // For x: avg(vel_x[x], vel_x[x+1]) but our grid is staggered?
    // Let's assume collocated for simplicity in this V1 port.
    
    let speed = sqrt(vel_x * vel_x + vel_z * vel_z);
    let depth = water_depth[idx];
    
    if (depth < 0.001 || speed < 0.0001) {
        return;
    }
    
    // Transport Capacity with hard cap
    let slope_factor = 1.0;
    let raw_capacity = CAPACITY_FACTOR * speed * min(depth, 1.0) * slope_factor;
    let capacity = min(raw_capacity, MAX_CAPACITY);
    
    let current_suspended = suspended_sediment[idx];
    
    if (current_suspended > capacity) {
        // Deposition
        let deposit_amount = (current_suspended - capacity) * K_DEPOSIT * params.dt;
        suspended_sediment[idx] -= deposit_amount;
        sediment[idx] += deposit_amount;
    } else {
        // Erosion
        let deficit = capacity - current_suspended;
        let entrain_target = deficit * K_ENTRAIN * params.dt;
        
        // Erode from layers (Sediment -> Overburden -> Paydirt -> Bedrock)
        // Track actual eroded mass separately
        var total_eroded = 0.0;
        var remaining_target = entrain_target;
        
        // 1. Sediment (easiest to erode)
        let sed_avail = sediment[idx];
        let sed_eroded = min(remaining_target, sed_avail);
        sediment[idx] -= sed_eroded;
        total_eroded += sed_eroded;
        remaining_target -= sed_eroded;
        
        // 2. Overburden
        if (remaining_target > 0.0) {
            let ob_avail = overburden[idx];
            let can_erode = min(remaining_target * K_HARDNESS_OVERBURDEN, ob_avail);
            overburden[idx] -= can_erode;
            total_eroded += can_erode;
            remaining_target -= can_erode / K_HARDNESS_OVERBURDEN; // Account for hardness in remaining target
        }
        
        // 3. Paydirt
         if (remaining_target > 0.0) {
            let pd_avail = paydirt[idx];
            let can_erode = min(remaining_target * K_HARDNESS_PAYDIRT, pd_avail);
            paydirt[idx] -= can_erode;
            total_eroded += can_erode;
        }
        
        // 4. Bedrock (Don't erode, it's the stable base)
        // No bedrock erosion for stability
        
        // Add ONLY what was actually eroded to suspended sediment
        suspended_sediment[idx] += total_eroded;
    }
}

// 2. Advect Sediment (Semi-Lagrangian)
@compute @workgroup_size(16, 16)
fn advect_sediment(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let z = global_id.y;
    if (x >= params.width || z >= params.depth) { return; }
    
    let idx = get_idx(x, z);
    
    // Backtrack position
    let u = water_velocity_x[idx];
    let v = water_velocity_z[idx];
    let dt = params.dt;
    
    let src_x = f32(x) - u * dt / params.cell_size;
    let src_z = f32(z) - v * dt / params.cell_size;
    
    // Bilinear Interpolate suspended_sediment at (src_x, src_z)
    // Boundary checks
    if (src_x < 0.0 || src_x >= f32(params.width) - 1.0 || 
        src_z < 0.0 || src_z >= f32(params.depth) - 1.0) {
        
        // Out of bounds: assumes 0 or clamp?
        // Let's flow out.
        // suspended_sediment[idx] = 0.0; // Clears buffer? No, output to new buffer?
        // We need double buffering for correct advection!
        // For V1, let's do simple forward Euler spread or just ignore advection to verify erosion first.
        // Actually, without double buffering, this is racy.
        return;
    }
    
    // Accessing random other cells is racy if writing to same buffer.
    // We need 'read' buffer (current) and 'write' buffer (next).
    // The current setup has only one suspended buffer.
    // For now, let's skip complex advection and trust water to carry it via flux?
    // No, SWE flux carries water mass. Suspended mass must move with water.
    // Let's implement correct Flux-Based Advection similar to Water Flux?
    // Flux_Sediment = Flux_Water * Concentration.
    // This is mass conservative and uses existing Flux buffers!
}

@compute @workgroup_size(16, 16)
fn update_sediment_transport(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Flux-Based Sediment Advection
    // Transport = Water_Flux * Concentration_upwind
    // Use upwind scheme: concentration from cell that flux flows FROM
    
    let x = global_id.x;
    let z = global_id.y;
    if (x >= params.width || z >= params.depth) { return; }
    let idx = get_idx(x, z);
    
    let depth = water_depth[idx];
    let current_sed = suspended_sediment[idx];
    
    // Calculate local concentration
    let local_conc = select(0.0, current_sed / depth, depth > 0.001);
    
    var net_sediment = 0.0;
    
    // X-direction flux
    // Inflow from left (flux_x[x-1] > 0 means flow from x-1 to x)
    if (x > 0) {
        let idx_left = get_idx(x - 1, z);
        let fx = flux_x[idx_left];
        if (fx > 0.0) {
            // Inflow from left - use left cell concentration
            let depth_left = water_depth[idx_left];
            let conc_left = select(0.0, suspended_sediment[idx_left] / depth_left, depth_left > 0.001);
            net_sediment += fx * conc_left;
        } else {
            // Outflow to left - use local concentration
            net_sediment += fx * local_conc; // fx is negative, so this subtracts
        }
    }
    
    // Outflow to right (flux_x[idx] > 0 means flow from x to x+1)
    if (x < params.width - 1) {
        let fx = flux_x[idx];
        if (fx > 0.0) {
            // Outflow to right - use local concentration
            net_sediment -= fx * local_conc;
        } else {
            // Inflow from right - use right cell concentration
            let idx_right = get_idx(x + 1, z);
            let depth_right = water_depth[idx_right];
            let conc_right = select(0.0, suspended_sediment[idx_right] / depth_right, depth_right > 0.001);
            net_sediment -= fx * conc_right; // fx is negative, so this adds
        }
    }
    
    // Z-direction flux (same pattern)
    if (z > 0) {
        let idx_back = get_idx(x, z - 1);
        let fz = flux_z[idx_back];
        if (fz > 0.0) {
            let depth_back = water_depth[idx_back];
            let conc_back = select(0.0, suspended_sediment[idx_back] / depth_back, depth_back > 0.001);
            net_sediment += fz * conc_back;
        } else {
            net_sediment += fz * local_conc;
        }
    }
    
    if (z < params.depth - 1) {
        let fz = flux_z[idx];
        if (fz > 0.0) {
            net_sediment -= fz * local_conc;
        } else {
            let idx_front = get_idx(x, z + 1);
            let depth_front = water_depth[idx_front];
            let conc_front = select(0.0, suspended_sediment[idx_front] / depth_front, depth_front > 0.001);
            net_sediment -= fz * conc_front;
        }
    }
    
    // Apply transport (scaled by cell area)
    let cell_area = params.cell_size * params.cell_size;
    let new_sed = current_sed + net_sediment / cell_area;
    
    // Clamp to non-negative and reasonable max
    suspended_sediment[idx] = clamp(new_sed, 0.0, 10.0);
}

