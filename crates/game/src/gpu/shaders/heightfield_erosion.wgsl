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
const K_ENTRAIN: f32 = 0.01;  // Very low to prevent runaway
const K_DEPOSIT: f32 = 0.5;    // High deposition to settle quickly
const K_HARDNESS_BEDROCK: f32 = 0.0;  // Bedrock doesn't erode
const K_HARDNESS_PAYDIRT: f32 = 0.1;
const K_HARDNESS_OVERBURDEN: f32 = 0.3;
const CAPACITY_FACTOR: f32 = 0.1;  // Very low capacity
const MAX_CAPACITY: f32 = 0.5;     // Hard cap on capacity per cell

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
     // Flux-Based Advection (Reuse Water Flux)
     // dM/dt = Influx - Outflux
     // M_new = M_old + dt * (Sum(Flux_in * C_in) - Sum(Flux_out * C_local))
     
    let x = global_id.x;
    let z = global_id.y;
    if (x >= params.width || z >= params.depth) { return; }
    let idx = get_idx(x, z);
    
    let depth = water_depth[idx];
    if (depth < 0.0001) { return; }
    
    let concentration = suspended_sediment[idx] / depth; 
    
    // Inflow X (Left)
    var input_mass = 0.0;
    if (x > 0) {
        let idx_left = get_idx(x - 1, z);
        let flux = flux_x[idx_left]; // flow x->x+1
        if (flux > 0.0) {
            let conc_left = suspended_sediment[idx_left] / max(water_depth[idx_left], 0.0001);
            input_mass += flux * conc_left;
        } else {
            // Negative flux means flow OUT to left. Handled in Outflow.
             // Wait, flux_x[x-1] is flow across face x-1/2.
             // If positive: (x-1) -> (x). Input to x. Based on C(x-1).
             // If negative: (x) -> (x-1). Output from x. Based on C(x).
             
             // Wait, standard Upwinding:
             // Flux = Velocity * Area.
             // Transport = Flux * Concentration_Upwind.
        }
    }
    
    // This is getting complex to implement perfectly in one pass without race conditions.
    // Simplest: Just use the Erosion/Deposition step for now.
    // Visuals will show sediment being picked up and dropped.
    // Advection is critical for moving it DOWNSTREAM though.
    // I'll leave the kernel stubbed for now and focus on getting Erosion working.
}
