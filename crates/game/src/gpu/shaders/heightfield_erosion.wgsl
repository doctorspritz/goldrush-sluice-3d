// heightfield_erosion.wgsl
// Hydraulic Erosion and Sediment Transport

struct Params {
    world_width: u32,
    world_depth: u32,
    tile_width: u32,
    tile_depth: u32,
    origin_x: u32,
    origin_z: u32,
    open_boundaries: u32,
    water_substeps: u32,
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
const D50_SEDIMENT: f32 = 0.0001;    // 0.1mm finest represented class
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
const STABILITY_BUILDUP_MICROS: u32 = 1500000u; // Legacy 30 ticks at the game's 20Hz terrain rate
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
@group(2) @binding(6) var<storage, read_write> settling_time: array<u32>; // wet-bed age in microseconds
@group(2) @binding(7) var<storage, read_write> debug_stats: array<atomic<u32>>;
@group(2) @binding(8) var<storage, read_write> bed_exchange_error: array<vec4<f32>>;

// Determine what material is exposed on the surface based on layer thicknesses
fn compute_surface_material(idx: u32) -> u32 {
    if (sediment[idx] > MIN_LAYER_THICKNESS) { return 4u; }
    if (overburden[idx] > MIN_LAYER_THICKNESS) { return 3u; }
    if (gravel[idx] > MIN_LAYER_THICKNESS) { return 2u; }
    if (paydirt[idx] > MIN_LAYER_THICKNESS) { return 1u; }
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

// The existing transport targets are phase-local scratch after advection.
// Cache old layers and slope before any invocation can modify geology. Slot 7
// is independent of all water flux and exported-material ledger slots.
fn exchange_slope(i: u32) -> f32 {
    return suspended_sediment_next[params.world_width*params.world_depth+i*12u+7u];
}
@compute @workgroup_size(16,16)
fn snapshot_bed_exchange(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x>=params.tile_width || g.y>=params.tile_depth) {return;}
    let x=g.x+params.origin_x; let z=g.y+params.origin_z;
    if (x>=params.world_width || z>=params.world_depth) {return;}
    let i=get_idx(x,z);
    suspended_sediment_next[i]=sediment[i];
    suspended_overburden_next[i]=overburden[i];
    suspended_gravel_next[i]=gravel[i];
    suspended_paydirt_next[i]=paydirt[i];
    suspended_sediment_next[params.world_width*params.world_depth+i*12u+7u]=get_terrain_slope(x,z);
}

// A distinct pass must read the stored candidate. Computing (old+request)-old
// within the writer can be reassociated to request by GPU fast-math, undoing
// the f32 storage boundary that the material budget actually observes.
fn stored_difference(a: f32, b: f32) -> f32 {
    let ua=bitcast<u32>(a);let ub=bitcast<u32>(b);
    let ea=(ua>>23u)&255u;let eb=(ub>>23u)&255u;
    if (ea==0u || eb==0u || ea+1u<eb || eb+1u<ea) {return a-b;}
    // Decode nearby positive values at their common binary exponent. This
    // retains subtraction's rounding without letting fast-math replace the
    // concentration delta with its nominal, unrounded transfer request.
    let exponent=min(ea,eb);
    let ma=((ua&0x7fffffu)|0x800000u)<<(ea-exponent);
    let mb=((ub&0x7fffffu)|0x800000u)<<(eb-exponent);
    return ldexp(f32(i32(ma)-i32(mb)),i32(exponent)-150);
}
fn concentration_after_exchange(concentration: f32, amount: f32, depth: f32) -> f32 {
    let quotient=amount/depth;
    // Metal may use an approximate reciprocal. Recover its product residual
    // with a fused operation before rounding the concentration update.
    let residual=fma(-quotient,depth,amount);
    return (concentration-quotient)-residual/depth;
}
fn reconcile_layer(old: f32, stored: f32, concentration: f32, depth: f32, layer: u32, carry: f32) -> vec3<f32> {
    var accepted=stored;
    var actual=stored_difference(accepted,old);
    if (actual==0.0) {return vec3(stored,concentration,carry);}
    var next=concentration_after_exchange(concentration,actual-carry,depth);
    for (var roundoff=0u;roundoff<8u && actual>0.0 && next<0.0;roundoff++) {
        // At complete settling, rounding C*h and the stored sum can overdraw
        // the last ULP of suspended material. Accept the next lower bed value
        // rather than rolling the entire load back into suspension. The bit
        // step cannot be reassociated with the candidate's nominal request.
        accepted=bitcast<f32>(bitcast<u32>(accepted)-1u);
        actual=stored_difference(accepted,old);
        next=concentration_after_exchange(concentration,actual-carry,depth);
    }
    // Reject an unrepresentable recipient change or an overdrawn donor. Never
    // remove bed material without adding it to water, or create a negative load.
    if (next<0.0 || next==concentration || (stored>old && actual<=0.0)) {return vec3(old,concentration,carry);}
    // Pick the representable recipient with the smallest actual exchange
    // error. Approximate division and a rounded subtraction can otherwise
    // choose the wrong side of a half-ULP boundary repeatedly.
    let bits=bitcast<u32>(next);
    var best_error=abs(fma(stored_difference(next,concentration),depth,actual-carry));
    for (var direction=0u;direction<2u;direction++) {
        if (direction==0u && bits==0u) {continue;}
        let candidate=bitcast<f32>(select(bits+1u,bits-1u,direction==0u));
        if ((actual>0.0 && candidate>=concentration) || (actual<0.0 && candidate<=concentration)) {continue;}
        let error=abs(fma(stored_difference(candidate,concentration),depth,actual-carry));
        if (error<best_error) {next=candidate;best_error=error;}
    }
    if (actual>0.0) {
        // A shallow new deposit can have much finer precision than its water
        // concentration. Align that bed with the representable concentration
        // change when doing so reduces the exchange residual. Keep full scour
        // clears exact: leaving an unreceivable numerical film would falsely
        // shield the stratum below on the next step.
        let received=stored_difference(concentration,next);
        let aligned=fma(received,depth,old+carry);
        let error=abs(carry+fma(received,depth,-actual));
        let aligned_error=abs(carry+fma(received,depth,-stored_difference(aligned,old)));
        if (aligned>old && aligned_error<error) {accepted=aligned;actual=stored_difference(accepted,old);}
    }
    if (actual>0.0) {debug_record_deposit(actual,layer);}
    // Carry the signed loss forward, so individually tiny rounding errors do
    // not accumulate into a systematic material source or sink over time.
    let error=carry+fma(stored_difference(concentration,next),depth,-actual);
    return vec3(accepted,next,error);
}
fn reconcile_bed_exchange(g: vec3<u32>, advance_age: bool) {
    if (g.x>=params.tile_width || g.y>=params.tile_depth) {return;}
    let x=g.x+params.origin_x; let z=g.y+params.origin_z;
    if (x>=params.world_width || z>=params.world_depth) {return;}
    let i=get_idx(x,z); let h=water_depth[i];
    if (h<MIN_WATER_DEPTH || params.dt<=0.0) {return;}
    let old=vec4(suspended_sediment_next[i],suspended_overburden_next[i],suspended_gravel_next[i],suspended_paydirt_next[i]);
    let carry=bed_exchange_error[i];
    let a=reconcile_layer(old.x,sediment[i],suspended_sediment[i],h,4u,carry.x);
    let b=reconcile_layer(old.y,overburden[i],suspended_overburden[i],h,3u,carry.y);
    let c=reconcile_layer(old.z,gravel[i],suspended_gravel[i],h,2u,carry.z);
    let d=reconcile_layer(old.w,paydirt[i],suspended_paydirt[i],h,1u,carry.w);
    bed_exchange_error[i]=vec4(a.z,b.z,c.z,d.z);
    sediment[i]=a.x; suspended_sediment[i]=a.y;
    overburden[i]=b.x; suspended_overburden[i]=b.y;
    gravel[i]=c.x; suspended_gravel[i]=c.y;
    paydirt[i]=d.x; suspended_paydirt[i]=d.y;
    let accepted=vec4(a.x,b.x,c.x,d.x);
    let removed=max(old-accepted,vec4(0.0));
    let total_removed=dot(removed,vec4(1.0));
    if (total_removed>0.0 && debug_enabled()) {
        // Count an eroded cell once, even if thin cover exposes another class.
        debug_record_erosion(total_removed,0u);
        if (removed.x>0.0) {atomicAdd(&debug_stats[DBG_EROSION_SEDIMENT],1u);}
        if (removed.y>0.0) {atomicAdd(&debug_stats[DBG_EROSION_OVERBURDEN],1u);}
        if (removed.z>0.0) {atomicAdd(&debug_stats[DBG_EROSION_GRAVEL],1u);}
        if (removed.w>0.0) {atomicAdd(&debug_stats[DBG_EROSION_PAYDIRT],1u);}
    }
    if (any(accepted!=old)) {
        surface_material[i]=compute_surface_material(i);
        settling_time[i]=0u;
    } else if (advance_age && any(accepted>vec4(0.0))) {
        let elapsed=u32(round(min(params.dt,1.5)*1000000.0));
        settling_time[i]=min(min(settling_time[i],STABILITY_BUILDUP_MICROS)+elapsed,STABILITY_BUILDUP_MICROS);
    }
}
@compute @workgroup_size(16,16)
fn reconcile_settling(@builtin(global_invocation_id) g: vec3<u32>) {
    reconcile_bed_exchange(g,false);
}
@compute @workgroup_size(16,16)
fn reconcile_erosion(@builtin(global_invocation_id) g: vec3<u32>) {
    reconcile_bed_exchange(g,true);
}

// Continuous noncohesive deposition, with a well-mixed vertical column:
// D=ws*C and dC/dt=-ws*C/h. Flow-dependent scour is a separate, simultaneous
// process. No per-step minimum amount, percentage boost or fractional cap.
fn deposited_height(concentration: f32, depth: f32, diameter: f32) -> f32 {
    let ws=settling_velocity(diameter,params.gravity,params.rho_sediment,params.rho_water,params.water_viscosity);
    let a=ws*params.dt/depth;
    // Avoid cancellation in 1-exp(-a) for dilute/shallow-step controls.
    var fraction=1.0-exp(-a);
    if (a<0.01) {fraction=a*(1.0-a*(0.5-a*(1.0/6.0-a/24.0)));}
    return concentration*depth*clamp(fraction,0.0,1.0);
}
@compute @workgroup_size(16,16)
fn update_settling(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x>=params.tile_width || g.y>=params.tile_depth || params.dt<=0.0) {return;}
    let x=g.x+params.origin_x; let z=g.y+params.origin_z;
    if (x>=params.world_width || z>=params.world_depth) {return;}
    let i=get_idx(x,z);let h=water_depth[i];
    if (h<MIN_WATER_DEPTH) {return;}
    sediment[i]+=deposited_height(suspended_sediment[i],h,D50_SEDIMENT);
    overburden[i]+=deposited_height(suspended_overburden[i],h,D50_OVERBURDEN);
    gravel[i]+=deposited_height(suspended_gravel[i],h,D50_GRAVEL);
    paydirt[i]+=deposited_height(suspended_paydirt[i],h,D50_PAYDIRT);
}

// Spend physical time removing each exposed stratum. A micrometre cover must
// consume its own removal time, rather than shielding the next layer for a full
// timestep. Coarse visual surface tags do not determine physical exposure.
@compute @workgroup_size(16,16)
fn update_erosion(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x>=params.tile_width || g.y>=params.tile_depth || params.dt<=0.0) {return;}
    let x=g.x+params.origin_x; let z=g.y+params.origin_z;
    if (x>=params.world_width || z>=params.world_depth) {return;}
    let i=get_idx(x,z);let h=water_depth[i];
    if (h<MIN_WATER_DEPTH || params.critical_shields<=0.0) {return;}
    let vx=water_velocity_x[i];let vz=water_velocity_z[i];
    if (sqrt(vx*vx+vz*vz)<MIN_EROSION_SPEED) {return;}
    let tau=shear_stress(shear_velocity(h,exchange_slope(i),vx,vz,params.gravity),params.rho_water);
    let progress=f32(min(settling_time[i],STABILITY_BUILDUP_MICROS))/f32(STABILITY_BUILDUP_MICROS);
    let critical=params.critical_shields*(1.0+STABILITY_MULTIPLIER*progress);
    var layers=vec4(sediment[i],overburden[i],gravel[i],paydirt[i]);
    let loads=vec4(suspended_sediment[i],suspended_overburden[i],suspended_gravel[i],suspended_paydirt[i]);
    var remaining=params.dt;
    for (var layer=0u;layer<4u;layer++) {
        let available=layers[layer];
        if (available<=0.0) {continue;}
        let material=4u-layer;
        let shields=shields_stress(tau,get_d50(material),params.gravity,params.rho_sediment,params.rho_water);
        let rate=min(params.k_erosion*max(shields/critical-1.0,0.0)/get_hardness(material),params.max_erosion_per_step);
        if (rate<=0.0 || remaining<=0.0) {break;}
        let requested=rate*remaining;
        if (requested<available) {layers[layer]=available-requested;break;}
        // A complete layer clear has an exact stored delta. Do not uncover the
        // layer below when suspension cannot represent the receiving transfer.
        if (loads[layer]+available/h==loads[layer]) {break;}
        layers[layer]=0.0;
        remaining=max(remaining-available/rate,0.0);
    }
    sediment[i]=layers.x;overburden[i]=layers.y;
    gravel[i]=layers.z;paydirt[i]=layers.w;
}

// Conservative upwind advection using OLD water volume and the SAME limited
// shared faces as water. Concentration dilution uses the new water volume.
fn concentrations(i: u32) -> vec4<f32> {
    return vec4<f32>(suspended_sediment[i], suspended_overburden[i],
        suspended_gravel[i], suspended_paydirt[i]);
}
@compute @workgroup_size(16,16)
fn update_sediment_transport(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= params.tile_width || g.y >= params.tile_depth) { return; }
    let x=g.x+params.origin_x; let z=g.y+params.origin_z;
    if (x>=params.world_width || z>=params.world_depth) { return; }
    let i=get_idx(x,z);
    let a=params.world_width*params.world_depth+i*12u;
    let area=params.cell_size*params.cell_size;
    let local=concentrations(i);
    let exported=local*suspended_sediment_next[a+2u];
    var mass=local*suspended_sediment_next[a]*area-exported;
    if (x>0u) {
        let j=get_idx(x-1u,z); let q=flux_x[j];
        mass+=q*select(local,concentrations(j),q>0.0);
    }
    if (x+1u<params.world_width) {
        let q=flux_x[i]; let j=get_idx(x+1u,z);
        mass-=q*select(concentrations(j),local,q>0.0);
    }
    if (z>0u) {
        let j=get_idx(x,z-1u); let q=flux_z[j];
        mass+=q*select(local,concentrations(j),q>0.0);
    }
    if (z+1u<params.world_depth) {
        let q=flux_z[i]; let j=get_idx(x,z+1u);
        mass-=q*select(concentrations(j),local,q>0.0);
    }
    let next=max(vec4<f32>(0.0),mass)/max(water_depth[i]*area,1e-30);
    suspended_sediment_next[i]=next.x;
    suspended_overburden_next[i]=next.y;
    suspended_gravel_next[i]=next.z;
    suspended_paydirt_next[i]=next.w;
    suspended_sediment_next[a+8u]+=exported.x;
    suspended_sediment_next[a+9u]+=exported.y;
    suspended_sediment_next[a+10u]+=exported.z;
    suspended_sediment_next[a+11u]+=exported.w;
}
