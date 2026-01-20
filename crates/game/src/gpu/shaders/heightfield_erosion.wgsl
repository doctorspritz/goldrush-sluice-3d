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
    debug_flags: u32,
    debug_scale: f32,
}

// =============================================================================
// EROSION PHYSICS CONSTANTS
// =============================================================================

// Numerical thresholds
const MIN_LAYER_THICKNESS: f32 = 0.001;  // Minimum thickness for layer existence (meters)
const MIN_WATER_DEPTH: f32 = 0.001;      // Minimum water depth for calculations (meters)
const EPSILON_FINE: f32 = 0.0001;        // Fine epsilon for numerical checks

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
const MAX_SUSPENDED: f32 = 0.3;        // Max concentration (0-1)

const DBG_EROSION_CELLS: u32 = 0u;
const DBG_DEPOSITION_CELLS: u32 = 1u;
const DBG_EROSION_MAX_MM: u32 = 2u;
const DBG_DEPOSITION_MAX_MM: u32 = 3u;
const DBG_EROSION_SEDIMENT: u32 = 4u;
const DBG_EROSION_OVERBURDEN: u32 = 5u;
const MIN_EROSION_SPEED: f32 = 0.05;
const MIN_EROSION_SLOPE: f32 = 0.0005;

// Temporal stability parameters
const STABILITY_BUILDUP_FRAMES: f32 = 30.0;  // Frames to reach max stability (~1 second at 30fps)
const STABILITY_MULTIPLIER: f32 = 0.5;        // Max threshold multiplier (50% increase)
const DBG_EROSION_GRAVEL: u32 = 6u;
const DBG_EROSION_PAYDIRT: u32 = 7u;
const DBG_DEPOSITION_SEDIMENT: u32 = 8u;
const DBG_DEPOSITION_OVERBURDEN: u32 = 9u;
const DBG_DEPOSITION_GRAVEL: u32 = 10u;
const DBG_DEPOSITION_PAYDIRT: u32 = 11u;

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> water_depth: array<f32>;
@group(1) @binding(1) var<storage, read_write> water_velocity_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> water_velocity_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> water_surface: array<f32>; // Not used directly but present in layout
@group(1) @binding(4) var<storage, read_write> flux_x: array<f32>;
@group(1) @binding(5) var<storage, read_write> flux_z: array<f32>;
@group(1) @binding(6) var<storage, read_write> suspended_sediment: array<f32>;
@group(1) @binding(7) var<storage, read_write> suspended_sediment_next: array<f32>; // Double buffer for race-free transport
@group(1) @binding(8) var<storage, read_write> suspended_overburden: array<f32>;
@group(1) @binding(9) var<storage, read_write> suspended_overburden_next: array<f32>;
@group(1) @binding(10) var<storage, read_write> suspended_gravel: array<f32>;
@group(1) @binding(11) var<storage, read_write> suspended_gravel_next: array<f32>;
@group(1) @binding(12) var<storage, read_write> suspended_paydirt: array<f32>;
@group(1) @binding(13) var<storage, read_write> suspended_paydirt_next: array<f32>;

@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;
@group(2) @binding(5) var<storage, read_write> surface_material: array<u32>; // 0=bed,1=pay,2=gravel,3=over,4=sed
@group(2) @binding(6) var<storage, read_write> settling_time: array<u32>; // frames since last disturbance
@group(2) @binding(7) var<storage, read_write> debug_stats: array<atomic<u32>>;

// Determine what material is exposed on the surface based on layer thicknesses
fn compute_surface_material(idx: u32) -> u32 {
    if (sediment[idx] > MIN_LAYER_THICKNESS) { return 4u; }
    if (overburden[idx] > MIN_LAYER_THICKNESS) { return 3u; }
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

fn debug_enabled() -> bool {
    return params.debug_flags != 0u;
}

fn debug_scale_to_u32(value: f32) -> u32 {
    let scaled = value * params.debug_scale;
    return u32(clamp(scaled, 0.0, 4294967040.0));
}

fn debug_record_erosion(amount: f32, surface: u32) {
    if (!debug_enabled() || amount <= 0.0) {
        return;
    }
    atomicAdd(&debug_stats[DBG_EROSION_CELLS], 1u);
    atomicMax(&debug_stats[DBG_EROSION_MAX_MM], debug_scale_to_u32(amount));
    switch (surface) {
        case 4u: { atomicAdd(&debug_stats[DBG_EROSION_SEDIMENT], 1u); }
        case 3u: { atomicAdd(&debug_stats[DBG_EROSION_OVERBURDEN], 1u); }
        case 2u: { atomicAdd(&debug_stats[DBG_EROSION_GRAVEL], 1u); }
        case 1u: { atomicAdd(&debug_stats[DBG_EROSION_PAYDIRT], 1u); }
        default: { }
    }
}

fn debug_record_deposit(amount: f32, layer: u32) {
    if (!debug_enabled() || amount <= 0.0) {
        return;
    }
    atomicAdd(&debug_stats[DBG_DEPOSITION_CELLS], 1u);
    atomicMax(&debug_stats[DBG_DEPOSITION_MAX_MM], debug_scale_to_u32(amount));
    switch (layer) {
        case 4u: { atomicAdd(&debug_stats[DBG_DEPOSITION_SEDIMENT], 1u); }
        case 3u: { atomicAdd(&debug_stats[DBG_DEPOSITION_OVERBURDEN], 1u); }
        case 2u: { atomicAdd(&debug_stats[DBG_DEPOSITION_GRAVEL], 1u); }
        case 1u: { atomicAdd(&debug_stats[DBG_DEPOSITION_PAYDIRT], 1u); }
        default: { }
    }
}

// =============================================================================
// SHIELDS STRESS EROSION MODEL
// =============================================================================

// 1a. Settling only (prevents oscillation with erosion)
@compute @workgroup_size(16, 16)
fn update_settling(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    let idx = get_idx(x, z);
    let depth = water_depth[idx];

    if (depth < MIN_WATER_DEPTH) {
        return;
    }

    let vel_x = water_velocity_x[idx];
    let vel_z = water_velocity_z[idx];
    let slope = get_terrain_slope(x, z);
    let u_star = shear_velocity(depth, slope, vel_x, vel_z, params.gravity);
    let tau = shear_stress(u_star, params.rho_water);

    // Only settle when Shields stress is below deposition threshold (hysteresis vs erosion)
    // Use a smooth calm-water factor to avoid hard transitions.
    let speed = sqrt(vel_x * vel_x + vel_z * vel_z);
    let calm_factor = clamp(1.0 - speed / 0.2, 0.0, 1.0);
    var updated_surface = false;

    // Sediment settling (layer-specific)
    var suspended = suspended_sediment[idx];
    if (suspended > 0.0 && depth > 1e-4) {
        let shields = shields_stress(
            tau,
            D50_SEDIMENT,
            params.gravity,
            params.rho_sediment,
            params.rho_water,
        );
        let deposition_threshold = params.critical_shields * 0.4;
        if (shields < deposition_threshold || calm_factor > 0.0) {
            let v_settle = settling_velocity(
                D50_SEDIMENT,
                params.gravity,
                params.rho_sediment,
                params.rho_water,
                params.water_viscosity,
            );
            let settling_rate = v_settle / depth;
            var settled_frac = min(settling_rate * params.dt, 0.25);
            let settle_boost = 0.04 * calm_factor;
            settled_frac = max(settled_frac, settle_boost);
            let settled_conc = suspended * settled_frac;
            if (settled_conc > 1e-6) {
                let deposit_height = settled_conc * depth;
                // Only deposit if it's meaningful (>1mm) to prevent micro-topography oscillation
                if (deposit_height > MIN_LAYER_THICKNESS) {
                    debug_record_deposit(deposit_height, 4u);
                    suspended -= settled_conc;
                    sediment[idx] += deposit_height;
                    suspended_sediment[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
                    updated_surface = true;
                }
            }
        }
    }

    // Overburden settling
    suspended = suspended_overburden[idx];
    if (suspended > 0.0 && depth > 1e-4) {
        let shields = shields_stress(
            tau,
            D50_OVERBURDEN,
            params.gravity,
            params.rho_sediment,
            params.rho_water,
        );
        let deposition_threshold = params.critical_shields * 0.5;
        if (shields < deposition_threshold || calm_factor > 0.0) {
            let v_settle = settling_velocity(
                D50_OVERBURDEN,
                params.gravity,
                params.rho_sediment,
                params.rho_water,
                params.water_viscosity,
            );
            let settling_rate = v_settle / depth;
            var settled_frac = min(settling_rate * params.dt, 0.25);
            let settle_boost = 0.05 * calm_factor;
            settled_frac = max(settled_frac, settle_boost);
            let settled_conc = suspended * settled_frac;
            if (settled_conc > 1e-6) {
                let deposit_height = settled_conc * depth;
                if (deposit_height > MIN_LAYER_THICKNESS) {
                    debug_record_deposit(deposit_height, 3u);
                    suspended -= settled_conc;
                    overburden[idx] += deposit_height;
                    suspended_overburden[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
                    updated_surface = true;
                }
            }
        }
    }

    // Gravel settling
    suspended = suspended_gravel[idx];
    if (suspended > 0.0 && depth > 1e-4) {
        let shields = shields_stress(
            tau,
            D50_GRAVEL,
            params.gravity,
            params.rho_sediment,
            params.rho_water,
        );
        let deposition_threshold = params.critical_shields * 0.8;
        if (shields < deposition_threshold || calm_factor > 0.0) {
            let v_settle = settling_velocity(
                D50_GRAVEL,
                params.gravity,
                params.rho_sediment,
                params.rho_water,
                params.water_viscosity,
            );
            let settling_rate = v_settle / depth;
            var settled_frac = min(settling_rate * params.dt, 0.35);
            let settle_boost = 0.08 * calm_factor;
            settled_frac = max(settled_frac, settle_boost);
            let settled_conc = suspended * settled_frac;
            if (settled_conc > 1e-6) {
                let deposit_height = settled_conc * depth;
                if (deposit_height > MIN_LAYER_THICKNESS) {
                    debug_record_deposit(deposit_height, 2u);
                    suspended -= settled_conc;
                    gravel[idx] += deposit_height;
                    suspended_gravel[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
                    updated_surface = true;
                }
            }
        }
    }

    // Paydirt settling
    suspended = suspended_paydirt[idx];
    if (suspended > 0.0 && depth > 1e-4) {
        let shields = shields_stress(
            tau,
            D50_PAYDIRT,
            params.gravity,
            params.rho_sediment,
            params.rho_water,
        );
        let deposition_threshold = params.critical_shields * 0.6;
        if (shields < deposition_threshold || calm_factor > 0.0) {
            let v_settle = settling_velocity(
                D50_PAYDIRT,
                params.gravity,
                params.rho_sediment,
                params.rho_water,
                params.water_viscosity,
            );
            let settling_rate = v_settle / depth;
            var settled_frac = min(settling_rate * params.dt, 0.25);
            let settle_boost = 0.05 * calm_factor;
            settled_frac = max(settled_frac, settle_boost);
            let settled_conc = suspended * settled_frac;
            if (settled_conc > 1e-6) {
                let deposit_height = settled_conc * depth;
                if (deposit_height > MIN_LAYER_THICKNESS) {
                    debug_record_deposit(deposit_height, 1u);
                    suspended -= settled_conc;
                    paydirt[idx] += deposit_height;
                    suspended_paydirt[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
                    updated_surface = true;
                }
            }
        }
    }

    if (updated_surface) {
        surface_material[idx] = compute_surface_material(idx);
        settling_time[idx] = 0u;  // Reset stability timer on new deposition
    }
}

// 1b. Erosion only (called after settling to prevent feedback)
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

    if (depth < MIN_WATER_DEPTH) {
        return;
    }

    let vel_x = water_velocity_x[idx];
    let vel_z = water_velocity_z[idx];
    
    // Skip erosion in still water (prevent oscillation from numerical water sloshing)
    let speed = sqrt(vel_x * vel_x + vel_z * vel_z);
    if (speed < 0.05) {  // <5cm/s is effectively still water
        return;
    }
    
    let slope = get_terrain_slope(x, z);

    let u_star = shear_velocity(depth, slope, vel_x, vel_z, params.gravity);
    let tau = shear_stress(u_star, params.rho_water);

    var eroded_sediment = 0.0;
    var eroded_gravel = 0.0;
    var eroded_overburden = 0.0;
    var eroded_paydirt = 0.0;
    var updated_surface = false;

    // EROSION ONLY (settling happens in separate pass)
    // Erode ONLY the exposed (top) layer.
    let critical = params.critical_shields;
    let max_erosion = params.max_erosion_per_step * params.dt;
    let surface = surface_material[idx];

    if (critical > 0.0 && max_erosion > 0.0) {
        var available = 0.0;
        var d50_layer = 0.0;
        var hardness = 1.0;

        switch (surface) {
            case 4u: { // sediment
                available = sediment[idx];
                d50_layer = get_d50(4u);
                hardness = get_hardness(4u);
            }
            case 3u: { // overburden
                available = overburden[idx];
                d50_layer = get_d50(3u);
                hardness = get_hardness(3u);
            }
            case 2u: { // gravel
                available = gravel[idx];
                d50_layer = get_d50(2u);
                hardness = get_hardness(2u);
            }
            case 1u: { // paydirt
                available = paydirt[idx];
                d50_layer = get_d50(1u);
                hardness = get_hardness(1u);
            }
            default: { }
        }

        if (available > 0.001 && d50_layer > 0.0) {
            let shields_layer = shields_stress(
                tau,
                d50_layer,
                params.gravity,
                params.rho_sediment,
                params.rho_water,
            );
            
            // Apply temporal stability bonus
            let time_settled = settling_time[idx];
            let stability_progress = min(f32(time_settled) / STABILITY_BUILDUP_FRAMES, 1.0);
            let stability_bonus = STABILITY_MULTIPLIER * stability_progress;
            let effective_critical = critical * (1.0 + stability_bonus);
            
            if (shields_layer > effective_critical) {
                let excess = (shields_layer - effective_critical) / effective_critical;
                let erosion_rate = params.k_erosion * excess / hardness;
                let erode_height = min(erosion_rate * params.dt, available);
                let erode = min(erode_height, max_erosion);

                switch (surface) {
                    case 4u: {
                        sediment[idx] -= erode;
                        eroded_sediment += erode;
                    }
                    case 3u: {
                        overburden[idx] -= erode;
                        eroded_overburden += erode;
                    }
                    case 2u: {
                        gravel[idx] -= erode;
                        eroded_gravel += erode;
                    }
                    case 1u: {
                        paydirt[idx] -= erode;
                        eroded_paydirt += erode;
                    }
                    default: { }
                }
                
                settling_time[idx] = 0u;  // Reset on erosion
            } else {
                // Increment stability counter (saturate at 255)
                settling_time[idx] = min(settling_time[idx] + 1u, 255u);
            }
        }
    }

    let eroded_total = eroded_sediment + eroded_overburden + eroded_gravel + eroded_paydirt;
    if (eroded_total > 0.0) {
        debug_record_erosion(eroded_total, surface);
    }

    // Add eroded material to suspension (stored as CONCENTRATION)
    if (depth > 1e-4) {
        if (eroded_sediment > 0.0) {
            let eroded_conc = eroded_sediment / depth;
            let suspended = suspended_sediment[idx] + eroded_conc;
            suspended_sediment[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
            updated_surface = true;
        }
        if (eroded_overburden > 0.0) {
            let eroded_conc = eroded_overburden / depth;
            let suspended = suspended_overburden[idx] + eroded_conc;
            suspended_overburden[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
            updated_surface = true;
        }
        if (eroded_gravel > 0.0) {
            let eroded_conc = eroded_gravel / depth;
            let suspended = suspended_gravel[idx] + eroded_conc;
            suspended_gravel[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
            updated_surface = true;
        }
        if (eroded_paydirt > 0.0) {
            let eroded_conc = eroded_paydirt / depth;
            let suspended = suspended_paydirt[idx] + eroded_conc;
            suspended_paydirt[idx] = clamp(suspended, 0.0, MAX_SUSPENDED);
            updated_surface = true;
        }
    }

    if (updated_surface) {
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
    let cell_area = params.cell_size * params.cell_size;

    // Sediment
    var net_sediment = 0.0;
    let local_conc = suspended_sediment[idx];

    if (x > 0) {
        let idx_left = get_idx(x - 1, z);
        let fx = flux_x[idx_left];
        if (fx > 0.0) {
            let conc_left = suspended_sediment[idx_left];
            net_sediment += fx * conc_left;
        } else {
            net_sediment += fx * local_conc;
        }
    }

    if (x < params.world_width - 1) {
        let fx = flux_x[idx];
        if (fx > 0.0) {
            net_sediment -= fx * local_conc;
        } else {
            let idx_right = get_idx(x + 1, z);
            let conc_right = suspended_sediment[idx_right];
            net_sediment -= fx * conc_right;
        }
    }

    if (z > 0) {
        let idx_back = get_idx(x, z - 1);
        let fz = flux_z[idx_back];
        if (fz > 0.0) {
            let conc_back = suspended_sediment[idx_back];
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
            let conc_front = suspended_sediment[idx_front];
            net_sediment -= fz * conc_front;
        }
    }

    if (depth > 1e-4) {
        let delta_conc = net_sediment / (cell_area * depth);
        let new_sed = local_conc + delta_conc;
        suspended_sediment_next[idx] = clamp(new_sed, 0.0, MAX_SUSPENDED);
    } else {
        suspended_sediment_next[idx] = local_conc;
    }

    // Overburden
    net_sediment = 0.0;
    let local_over = suspended_overburden[idx];

    if (x > 0) {
        let idx_left = get_idx(x - 1, z);
        let fx = flux_x[idx_left];
        if (fx > 0.0) {
            let conc_left = suspended_overburden[idx_left];
            net_sediment += fx * conc_left;
        } else {
            net_sediment += fx * local_over;
        }
    }

    if (x < params.world_width - 1) {
        let fx = flux_x[idx];
        if (fx > 0.0) {
            net_sediment -= fx * local_over;
        } else {
            let idx_right = get_idx(x + 1, z);
            let conc_right = suspended_overburden[idx_right];
            net_sediment -= fx * conc_right;
        }
    }

    if (z > 0) {
        let idx_back = get_idx(x, z - 1);
        let fz = flux_z[idx_back];
        if (fz > 0.0) {
            let conc_back = suspended_overburden[idx_back];
            net_sediment += fz * conc_back;
        } else {
            net_sediment += fz * local_over;
        }
    }

    if (z < params.world_depth - 1) {
        let fz = flux_z[idx];
        if (fz > 0.0) {
            net_sediment -= fz * local_over;
        } else {
            let idx_front = get_idx(x, z + 1);
            let conc_front = suspended_overburden[idx_front];
            net_sediment -= fz * conc_front;
        }
    }

    if (depth > 1e-4) {
        let delta_conc = net_sediment / (cell_area * depth);
        let new_sed = local_over + delta_conc;
        suspended_overburden_next[idx] = clamp(new_sed, 0.0, MAX_SUSPENDED);
    } else {
        suspended_overburden_next[idx] = local_over;
    }

    // Gravel
    net_sediment = 0.0;
    let local_gravel = suspended_gravel[idx];

    if (x > 0) {
        let idx_left = get_idx(x - 1, z);
        let fx = flux_x[idx_left];
        if (fx > 0.0) {
            let conc_left = suspended_gravel[idx_left];
            net_sediment += fx * conc_left;
        } else {
            net_sediment += fx * local_gravel;
        }
    }

    if (x < params.world_width - 1) {
        let fx = flux_x[idx];
        if (fx > 0.0) {
            net_sediment -= fx * local_gravel;
        } else {
            let idx_right = get_idx(x + 1, z);
            let conc_right = suspended_gravel[idx_right];
            net_sediment -= fx * conc_right;
        }
    }

    if (z > 0) {
        let idx_back = get_idx(x, z - 1);
        let fz = flux_z[idx_back];
        if (fz > 0.0) {
            let conc_back = suspended_gravel[idx_back];
            net_sediment += fz * conc_back;
        } else {
            net_sediment += fz * local_gravel;
        }
    }

    if (z < params.world_depth - 1) {
        let fz = flux_z[idx];
        if (fz > 0.0) {
            net_sediment -= fz * local_gravel;
        } else {
            let idx_front = get_idx(x, z + 1);
            let conc_front = suspended_gravel[idx_front];
            net_sediment -= fz * conc_front;
        }
    }

    if (depth > 1e-4) {
        let delta_conc = net_sediment / (cell_area * depth);
        let new_sed = local_gravel + delta_conc;
        suspended_gravel_next[idx] = clamp(new_sed, 0.0, MAX_SUSPENDED);
    } else {
        suspended_gravel_next[idx] = local_gravel;
    }

    // Paydirt
    net_sediment = 0.0;
    let local_paydirt = suspended_paydirt[idx];

    if (x > 0) {
        let idx_left = get_idx(x - 1, z);
        let fx = flux_x[idx_left];
        if (fx > 0.0) {
            let conc_left = suspended_paydirt[idx_left];
            net_sediment += fx * conc_left;
        } else {
            net_sediment += fx * local_paydirt;
        }
    }

    if (x < params.world_width - 1) {
        let fx = flux_x[idx];
        if (fx > 0.0) {
            net_sediment -= fx * local_paydirt;
        } else {
            let idx_right = get_idx(x + 1, z);
            let conc_right = suspended_paydirt[idx_right];
            net_sediment -= fx * conc_right;
        }
    }

    if (z > 0) {
        let idx_back = get_idx(x, z - 1);
        let fz = flux_z[idx_back];
        if (fz > 0.0) {
            let conc_back = suspended_paydirt[idx_back];
            net_sediment += fz * conc_back;
        } else {
            net_sediment += fz * local_paydirt;
        }
    }

    if (z < params.world_depth - 1) {
        let fz = flux_z[idx];
        if (fz > 0.0) {
            net_sediment -= fz * local_paydirt;
        } else {
            let idx_front = get_idx(x, z + 1);
            let conc_front = suspended_paydirt[idx_front];
            net_sediment -= fz * conc_front;
        }
    }

    if (depth > 1e-4) {
        let delta_conc = net_sediment / (cell_area * depth);
        let new_sed = local_paydirt + delta_conc;
        suspended_paydirt_next[idx] = clamp(new_sed, 0.0, MAX_SUSPENDED);
    } else {
        suspended_paydirt_next[idx] = local_paydirt;
    }
}
