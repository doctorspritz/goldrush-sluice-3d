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
    manning_n: f32,
    rho_water: f32,
    rho_sediment: f32,
    water_viscosity: f32,
    critical_shields: f32,
    k_erosion: f32,
    max_erosion_per_step: f32,
    _pad1: vec2<u32>,
}

// =============================================================================
// EROSION PHYSICS CONSTANTS
// =============================================================================
// Particle sizes (median diameter, meters)
const D50_SEDIMENT: f32 = 0.0001;    // 0.1mm fine silt
const D50_OVERBURDEN: f32 = 0.001;   // 1mm coarse sand
const D50_GRAVEL: f32 = 0.01;        // 10mm gravel
const D50_PAYDIRT: f32 = 0.002;      // 2mm compacted sand

// Hardness multipliers (resistance to erosion)
const HARDNESS_SEDIMENT: f32 = 0.5;
const HARDNESS_OVERBURDEN: f32 = 1.0;
const HARDNESS_GRAVEL: f32 = 2.0;
const HARDNESS_PAYDIRT: f32 = 5.0;

// Turbulent flow friction coefficient
const CF: f32 = 0.003;

// Settling transitions
const D50_STOKES_MAX: f32 = 0.0001;    // < 0.1mm: pure Stokes
const D50_TURBULENT_MIN: f32 = 0.001;  // > 1mm: pure turbulent
const CD_SPHERE: f32 = 0.44;           // Drag coefficient

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

// Settling velocity (Stokes/turbulent blend)
fn settling_velocity(d50: f32, g: f32, rho_p: f32, rho_f: f32, mu: f32) -> f32 {
    let vs_stokes = g * (rho_p - rho_f) * d50 * d50 / (18.0 * mu);
    let vs_turbulent = sqrt(4.0 * g * d50 * (rho_p - rho_f) / (3.0 * rho_f * CD_SPHERE));

    if (d50 < D50_STOKES_MAX) {
        return vs_stokes;
    } else if (d50 > D50_TURBULENT_MIN) {
        return vs_turbulent;
    }

    let t = (d50 - D50_STOKES_MAX) / (D50_TURBULENT_MIN - D50_STOKES_MAX);
    return vs_stokes * (1.0 - t) + vs_turbulent * t;
}

// Shear velocity: u* = sqrt(g×h×S + Cf×v²)
fn shear_velocity(depth: f32, slope: f32, vel_x: f32, vel_z: f32, g: f32) -> f32 {
    let grav_term = g * depth * slope;
    let v_sq = vel_x * vel_x + vel_z * vel_z;
    let velocity_term = CF * v_sq;
    return sqrt(grav_term + velocity_term);
}

// Bed shear stress: τ = ρf × u*²
fn shear_stress(u_star: f32, rho_f: f32) -> f32 {
    return rho_f * u_star * u_star;
}

// Shields stress: τ* = τ / (g × (ρp - ρf) × d50)
fn shields_stress(tau: f32, d50: f32, g: f32, rho_p: f32, rho_f: f32) -> f32 {
    let rho_diff = rho_p - rho_f;
    let d50_safe = max(d50, 1e-6);
    return tau / (g * rho_diff * d50_safe);
}

// Get d50 for material type
fn get_d50(mat: u32) -> f32 {
    switch (mat) {
        case 4u: { return D50_SEDIMENT; }
        case 3u: { return D50_OVERBURDEN; }
        case 2u: { return D50_GRAVEL; }
        case 1u: { return D50_PAYDIRT; }
        default: { return 0.0; }
    }
}

// Get hardness for material type
fn get_hardness(mat: u32) -> f32 {
    switch (mat) {
        case 4u: { return HARDNESS_SEDIMENT; }
        case 3u: { return HARDNESS_OVERBURDEN; }
        case 2u: { return HARDNESS_GRAVEL; }
        case 1u: { return HARDNESS_PAYDIRT; }
        default: { return 0.0; }
    }
}

// =============================================================================
// SHIELDS STRESS EROSION MODEL
// =============================================================================

// 1. Erosion & Deposition with Shields Stress
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

    if (depth < 0.001) {
        return;
    }

    let vel_x = water_velocity_x[idx];
    let vel_z = water_velocity_z[idx];
    let slope = get_terrain_slope(x, z);

    let u_star = shear_velocity(depth, slope, vel_x, vel_z, params.gravity);
    let tau = shear_stress(u_star, params.rho_water);

    var suspended = suspended_sediment[idx];
    var updated_surface = false;

    // 1) SETTLING (independent of erosion)
    if (suspended > 0.0) {
        let v_settle = settling_velocity(
            D50_SEDIMENT,
            params.gravity,
            params.rho_sediment,
            params.rho_water,
            params.water_viscosity,
        );
        let settling_rate = v_settle / depth;
        let settled_frac = min(settling_rate * params.dt, 1.0);
        let settled_amount = suspended * settled_frac;

        if (settled_amount > 0.0) {
            suspended -= settled_amount;
            sediment[idx] += settled_amount;
            updated_surface = true;
        }
    }

    // 2) EROSION (sequential layers)
    let critical = params.critical_shields;
    let max_erosion = params.max_erosion_per_step * params.dt;
    var total_eroded = 0.0;

    if (critical > 0.0 && max_erosion > 0.0) {
        // Layer 1: Sediment
        if (total_eroded < max_erosion) {
            let available = sediment[idx];
            if (available > 0.001) {
                let d50_layer = get_d50(4u);
                let shields_layer = shields_stress(
                    tau,
                    d50_layer,
                    params.gravity,
                    params.rho_sediment,
                    params.rho_water,
                );
                if (shields_layer > critical) {
                    let excess = (shields_layer - critical) / critical;
                    let erosion_rate = params.k_erosion * excess / get_hardness(4u);
                    let erode_height = min(erosion_rate * params.dt, available);
                    let budget = max_erosion - total_eroded;
                    let erode = min(erode_height, budget);
                    sediment[idx] -= erode;
                    total_eroded += erode;
                }
            }
        }

        // Layer 2: Gravel
        if (total_eroded < max_erosion) {
            let available = gravel[idx];
            if (available > 0.001) {
                let d50_layer = get_d50(2u);
                let shields_layer = shields_stress(
                    tau,
                    d50_layer,
                    params.gravity,
                    params.rho_sediment,
                    params.rho_water,
                );
                if (shields_layer > critical) {
                    let excess = (shields_layer - critical) / critical;
                    let erosion_rate = params.k_erosion * excess / get_hardness(2u);
                    let erode_height = min(erosion_rate * params.dt, available);
                    let budget = max_erosion - total_eroded;
                    let erode = min(erode_height, budget);
                    gravel[idx] -= erode;
                    total_eroded += erode;
                }
            }
        }

        // Layer 3: Overburden
        if (total_eroded < max_erosion) {
            let available = overburden[idx];
            if (available > 0.001) {
                let d50_layer = get_d50(3u);
                let shields_layer = shields_stress(
                    tau,
                    d50_layer,
                    params.gravity,
                    params.rho_sediment,
                    params.rho_water,
                );
                if (shields_layer > critical) {
                    let excess = (shields_layer - critical) / critical;
                    let erosion_rate = params.k_erosion * excess / get_hardness(3u);
                    let erode_height = min(erosion_rate * params.dt, available);
                    let budget = max_erosion - total_eroded;
                    let erode = min(erode_height, budget);
                    overburden[idx] -= erode;
                    total_eroded += erode;
                }
            }
        }

        // Layer 4: Paydirt
        if (total_eroded < max_erosion) {
            let available = paydirt[idx];
            if (available > 0.001) {
                let d50_layer = get_d50(1u);
                let shields_layer = shields_stress(
                    tau,
                    d50_layer,
                    params.gravity,
                    params.rho_sediment,
                    params.rho_water,
                );
                if (shields_layer > critical) {
                    let excess = (shields_layer - critical) / critical;
                    let erosion_rate = params.k_erosion * excess / get_hardness(1u);
                    let erode_height = min(erosion_rate * params.dt, available);
                    let budget = max_erosion - total_eroded;
                    let erode = min(erode_height, budget);
                    paydirt[idx] -= erode;
                    total_eroded += erode;
                }
            }
        }
    }

    if (total_eroded > 0.0 && depth > 1e-4) {
        suspended += total_eroded;
        updated_surface = true;
    }

    if (updated_surface) {
        suspended_sediment[idx] = max(suspended, 0.0);
        surface_material[idx] = compute_surface_material(idx);
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
