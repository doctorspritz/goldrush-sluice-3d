// heightfield_erosion.wgsl
// Hydraulic Erosion and Sediment Transport

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
    damping: f32, // Reusing water params struct for now, need specific erosion params?
                  // Let's assume we pass a larger struct or a second uniform buffer.
                  // For simplicity, let's hardcode constants for now or extend Params.
}

// =============================================================================
// EROSION PHYSICS CONSTANTS
// =============================================================================
// These constants control the dramatic breach/flood dynamics.
// The model uses critical shear stress: erosion explodes when v > v_critical

// Critical velocities (m/s) - erosion rate increases above these thresholds
const V_CRIT_SEDIMENT: f32 = 0.1;       // Fresh silt/sediment - easy to erode
const V_CRIT_OVERBURDEN: f32 = 0.2;     // Soil/dirt - moderate resistance
const V_CRIT_GRAVEL: f32 = 0.5;         // Gravel - needs fast water
const V_CRIT_PAYDIRT: f32 = 0.4;        // Compacted pay layer

// Erosion rate multiplier when v > v_critical (m/s per unit shear excess)
// Tuned for visible erosion while avoiding instant terrain destruction
const K_EROSION_SEDIMENT: f32 = 0.02;     // Silt erodes readily
const K_EROSION_OVERBURDEN: f32 = 0.01;   // Dirt erodes moderately
const K_EROSION_GRAVEL: f32 = 0.003;      // Gravel resists erosion
const K_EROSION_PAYDIRT: f32 = 0.005;     // Paydirt is consolidated

// Settling/Deposition
const SETTLING_VELOCITY: f32 = 0.08;    // m/s - threshold for "slow" water (increased for more settling)
const K_DEPOSIT_FAST: f32 = 5.0;        // Deposition rate in still water (increased)
const K_DEPOSIT_SLOW: f32 = 2.0;        // Deposition rate in moving water (increased from 0.3)

// Transport capacity
const CAPACITY_FACTOR: f32 = 0.3;       // kg sediment per m³ water per m/s velocity (reduced for more deposition)
const MAX_CAPACITY: f32 = 3.0;          // Maximum sediment load (kg/m³) (reduced)

// Hardness multipliers (0 = impossible to erode, 1 = full erosion rate)
const K_HARDNESS_BEDROCK: f32 = 0.0;    // Bedrock doesn't erode
const K_HARDNESS_PAYDIRT: f32 = 0.5;
const K_HARDNESS_GRAVEL: f32 = 0.4;
const K_HARDNESS_OVERBURDEN: f32 = 1.0;
const K_HARDNESS_SEDIMENT: f32 = 1.0;   // Fresh sediment is soft

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> water_depth: array<f32>;
@group(1) @binding(1) var<storage, read_write> water_velocity_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> water_velocity_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> water_surface: array<f32>; // Not used directly but present in layout
@group(1) @binding(4) var<storage, read_write> flux_x: array<f32>;
@group(1) @binding(5) var<storage, read_write> flux_z: array<f32>;
@group(1) @binding(6) var<storage, read_write> suspended_sediment: array<f32>;
@group(1) @binding(7) var<storage, read_write> suspended_sediment_next: array<f32>; // Double buffer for race-free transport

@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;
@group(2) @binding(5) var<storage, read_write> surface_material: array<u32>; // 0=bed,1=pay,2=gravel,3=over,4=sed

// Determine what material is exposed on the surface based on layer thicknesses
fn compute_surface_material(idx: u32) -> u32 {
    let min_thick = 0.001;
    if (sediment[idx] > min_thick) { return 4u; }
    if (overburden[idx] > min_thick) { return 3u; }
    if (gravel[idx] > min_thick) { return 2u; }
    if (paydirt[idx] > min_thick) { return 1u; }
    return 0u; // bedrock
}

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.world_width + x;
}

// Get total ground height at an index
fn get_ground_height(idx: u32) -> f32 {
    return bedrock[idx] + paydirt[idx] + gravel[idx] + overburden[idx] + sediment[idx];
}

// Calculate terrain slope at a cell (gradient magnitude)
fn get_terrain_slope(x: u32, z: u32) -> f32 {
    let idx = get_idx(x, z);
    let h = get_ground_height(idx);

    // Central difference for gradient where possible
    var dh_dx = 0.0;
    var dh_dz = 0.0;

    if (x > 0u && x < params.world_width - 1u) {
        let h_left = get_ground_height(get_idx(x - 1u, z));
        let h_right = get_ground_height(get_idx(x + 1u, z));
        dh_dx = (h_right - h_left) / (2.0 * params.cell_size);
    } else if (x > 0u) {
        let h_left = get_ground_height(get_idx(x - 1u, z));
        dh_dx = (h - h_left) / params.cell_size;
    } else if (x < params.world_width - 1u) {
        let h_right = get_ground_height(get_idx(x + 1u, z));
        dh_dx = (h_right - h) / params.cell_size;
    }

    if (z > 0u && z < params.world_depth - 1u) {
        let h_back = get_ground_height(get_idx(x, z - 1u));
        let h_front = get_ground_height(get_idx(x, z + 1u));
        dh_dz = (h_front - h_back) / (2.0 * params.cell_size);
    } else if (z > 0u) {
        let h_back = get_ground_height(get_idx(x, z - 1u));
        dh_dz = (h - h_back) / params.cell_size;
    } else if (z < params.world_depth - 1u) {
        let h_front = get_ground_height(get_idx(x, z + 1u));
        dh_dz = (h_front - h) / params.cell_size;
    }

    return sqrt(dh_dx * dh_dx + dh_dz * dh_dz);
}

// =============================================================================
// CRITICAL SHEAR STRESS EROSION MODEL
// =============================================================================
// Erosion rate is proportional to (v² - v_crit²) when v > v_crit
// This creates positive feedback: erosion deepens channel → faster water → more erosion
// Result: small trickles can create runaway breaches and floods

// Get the critical velocity for the exposed surface material
fn get_critical_velocity(mat: u32) -> f32 {
    switch (mat) {
        case 4u: { return V_CRIT_SEDIMENT; }     // Fresh silt
        case 3u: { return V_CRIT_OVERBURDEN; }   // Soil/dirt
        case 2u: { return V_CRIT_GRAVEL; }       // Gravel
        case 1u: { return V_CRIT_PAYDIRT; }      // Paydirt
        default: { return 999.0; }               // Bedrock - effectively infinite
    }
}

// Get erosion rate constant for surface material
fn get_erosion_rate(mat: u32) -> f32 {
    switch (mat) {
        case 4u: { return K_EROSION_SEDIMENT; }
        case 3u: { return K_EROSION_OVERBURDEN; }
        case 2u: { return K_EROSION_GRAVEL; }
        case 1u: { return K_EROSION_PAYDIRT; }
        default: { return 0.0; }                 // Bedrock doesn't erode
    }
}

// 1. Erosion & Deposition with Critical Shear Stress
@compute @workgroup_size(16, 16)
fn update_erosion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    let idx = get_idx(x, z);
    let depth = water_depth[idx];

    // Need some water to do anything
    if (depth < 0.001) {
        return;
    }

    // Calculate velocity magnitude
    let vel_x = water_velocity_x[idx];
    let vel_z = water_velocity_z[idx];
    let speed = sqrt(vel_x * vel_x + vel_z * vel_z);
    let speed_sq = speed * speed;

    let current_suspended = suspended_sediment[idx];
    let surface_mat = surface_material[idx];

    // ==========================================================================
    // DEPOSITION: Fast settling in slow/pooled water (Stokes settling)
    // ==========================================================================
    // Sediment settles when water is calm. In a settling pond, silt drops out fast.

    if (speed < SETTLING_VELOCITY) {
        // Water is essentially still - fast deposition
        let deposit_rate = K_DEPOSIT_FAST * (1.0 - speed / SETTLING_VELOCITY);
        let deposit_amount = current_suspended * deposit_rate * params.dt;

        if (deposit_amount > 0.0001) {
            suspended_sediment[idx] -= deposit_amount;
            sediment[idx] += deposit_amount;
            surface_material[idx] = 4u; // Fresh sediment on top
        }
        return; // No erosion in still water
    }

    // ==========================================================================
    // EROSION: Critical shear stress model
    // ==========================================================================
    // Shear stress τ ∝ v². Erosion occurs when τ > τ_critical
    // Erosion rate ∝ (τ - τ_crit) = k * (v² - v_crit²)

    let v_crit = get_critical_velocity(surface_mat);
    let v_crit_sq = v_crit * v_crit;

    // Only erode if velocity exceeds critical threshold
    if (speed_sq > v_crit_sq) {
        // Shear excess: how much above the threshold we are
        let shear_excess = speed_sq - v_crit_sq;

        // Erosion rate increases with shear excess (positive feedback!)
        let k_erosion = get_erosion_rate(surface_mat);

        // Slope factor: steeper terrain erodes faster (concentrated flow)
        let terrain_slope = get_terrain_slope(x, z);
        let slope_factor = 1.0 + clamp(terrain_slope * 2.0, 0.0, 2.0);

        // Depth factor: deeper water has more erosive power
        let depth_factor = min(depth, 0.5) / 0.5; // Linear up to 0.5m

        // Total erosion potential this timestep
        let erosion_potential = k_erosion * shear_excess * slope_factor * depth_factor * params.dt;

        // Erode through layers from top to bottom
        var total_eroded = 0.0;
        var remaining = erosion_potential;

        // Layer 1: Sediment (freshly deposited silt - very easy)
        let sed_avail = sediment[idx];
        if (remaining > 0.0 && sed_avail > 0.0) {
            let erode = min(remaining, sed_avail);
            sediment[idx] -= erode;
            total_eroded += erode;
            remaining -= erode;
        }

        // Layer 2: Overburden (soil/dirt)
        let ob_avail = overburden[idx];
        if (remaining > 0.0 && ob_avail > 0.0) {
            // Overburden is a bit harder than fresh sediment
            let effective = remaining * K_HARDNESS_OVERBURDEN;
            let erode = min(effective, ob_avail);
            overburden[idx] -= erode;
            total_eroded += erode;
            remaining -= erode / K_HARDNESS_OVERBURDEN;
        }

        // Layer 3: Gravel (resistant)
        let gr_avail = gravel[idx];
        if (remaining > 0.0 && gr_avail > 0.0) {
            let effective = remaining * K_HARDNESS_GRAVEL;
            let erode = min(effective, gr_avail);
            gravel[idx] -= erode;
            total_eroded += erode;
            remaining -= erode / K_HARDNESS_GRAVEL;
        }

        // Layer 4: Paydirt (consolidated, gold-bearing)
        let pd_avail = paydirt[idx];
        if (remaining > 0.0 && pd_avail > 0.0) {
            let effective = remaining * K_HARDNESS_PAYDIRT;
            let erode = min(effective, pd_avail);
            paydirt[idx] -= erode;
            total_eroded += erode;
        }

        // Bedrock doesn't erode

        // Add eroded material to suspension
        if (total_eroded > 0.0) {
            suspended_sediment[idx] += total_eroded;
            surface_material[idx] = compute_surface_material(idx);
        }
    } else {
        // Below critical velocity but above settling - gradual deposition
        // Transport capacity decreases as velocity drops
        let capacity = CAPACITY_FACTOR * speed * min(depth, 1.0);

        if (current_suspended > capacity) {
            let excess = current_suspended - capacity;
            let deposit_amount = excess * K_DEPOSIT_SLOW * params.dt;

            if (deposit_amount > 0.0001) {
                suspended_sediment[idx] -= deposit_amount;
                sediment[idx] += deposit_amount;
                surface_material[idx] = 4u;
            }
        }
    }
}

// 2. Advect Sediment (Semi-Lagrangian)
@compute @workgroup_size(16, 16)
fn advect_sediment(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }
    
    let idx = get_idx(x, z);
    
    // Backtrack position
    let u = water_velocity_x[idx];
    let v = water_velocity_z[idx];
    let dt = params.dt;
    
    let src_x = f32(x) - u * dt / params.cell_size;
    let src_z = f32(z) - v * dt / params.cell_size;
    
    // Bilinear Interpolate suspended_sediment at (src_x, src_z)
    // Boundary checks
    if (src_x < 0.0 || src_x >= f32(params.world_width) - 1.0 || 
        src_z < 0.0 || src_z >= f32(params.world_depth) - 1.0) {
        
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
    
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }
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
    if (x < params.world_width - 1) {
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
    
    if (z < params.world_depth - 1) {
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

    // Write to NEXT buffer (double-buffering eliminates race conditions)
    // All reads are from suspended_sediment, write to suspended_sediment_next
    suspended_sediment_next[idx] = clamp(new_sed, 0.0, 10.0);
}
