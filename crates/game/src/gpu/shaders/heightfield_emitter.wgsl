// heightfield_emitter.wgsl
// Water emitter shader - adds water to specified location

struct EmitterParams {
    pos_x: f32,
    pos_z: f32,
    radius: f32,
    rate: f32,  // volume per second
    dt: f32,
    enabled: u32,  // 0 or 1
    world_width: u32,
    world_depth: u32,
    tile_width: u32,
    tile_depth: u32,
    origin_x: u32,
    origin_z: u32,
    cell_size: f32,
    sediment_conc: f32, // volume fraction 0-1
    overburden_conc: f32,
    gravel_conc: f32,
    paydirt_conc: f32,
    vel_x: f32,
    vel_z: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: EmitterParams;
@group(0) @binding(1) var<storage, read_write> water_depth: array<f32>;
@group(0) @binding(2) var<storage, read_write> suspended_sediment: array<f32>;
@group(0) @binding(3) var<storage, read_write> suspended_overburden: array<f32>;
@group(0) @binding(4) var<storage, read_write> suspended_gravel: array<f32>;
@group(0) @binding(5) var<storage, read_write> suspended_paydirt: array<f32>;
@group(0) @binding(6) var<storage, read_write> water_velocity_x: array<f32>;
@group(0) @binding(7) var<storage, read_write> water_velocity_z: array<f32>;

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.world_width + x;
}

@compute @workgroup_size(16, 16)
fn add_water(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (params.enabled == 0u) { return; }
    
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }
    
    // Calculate distance from emitter center
    let fx = f32(x) * params.cell_size + params.cell_size * 0.5;
    let fz = f32(z) * params.cell_size + params.cell_size * 0.5;
    let dx = fx - params.pos_x;
    let dz = fz - params.pos_z;
    let dist = sqrt(dx * dx + dz * dz);
    
    if (dist <= params.radius) {
        let idx = get_idx(x, z);
        
        // Smooth falloff from center
        let falloff = 1.0 - (dist / params.radius);
        
        // Calculate volume to add this frame
        // Rate is total volume/second, spread over radius area
        let area = 3.14159 * params.radius * params.radius;
        let depth_per_cell = (params.rate * params.dt * falloff) / area;
        
        let d0 = water_depth[idx];
        let dw = depth_per_cell;
        let new_depth = d0 + dw;
        if (new_depth > 1e-6) {
            let c_sed = clamp(params.sediment_conc, 0.0, 1.0);
            let c_over = clamp(params.overburden_conc, 0.0, 1.0);
            let c_grav = clamp(params.gravel_conc, 0.0, 1.0);
            let c_pay = clamp(params.paydirt_conc, 0.0, 1.0);

            // Blend suspended concentrations
            let new_sed = (suspended_sediment[idx] * d0 + c_sed * dw) / new_depth;
            let new_over = (suspended_overburden[idx] * d0 + c_over * dw) / new_depth;
            let new_grav = (suspended_gravel[idx] * d0 + c_grav * dw) / new_depth;
            let new_pay = (suspended_paydirt[idx] * d0 + c_pay * dw) / new_depth;

            suspended_sediment[idx] = clamp(new_sed, 0.0, 1.0);
            suspended_overburden[idx] = clamp(new_over, 0.0, 1.0);
            suspended_gravel[idx] = clamp(new_grav, 0.0, 1.0);
            suspended_paydirt[idx] = clamp(new_pay, 0.0, 1.0);

            // Blend Velocity
            let old_vx = water_velocity_x[idx];
            let old_vz = water_velocity_z[idx];
            let add_vx = params.vel_x;
            let add_vz = params.vel_z;
            
            // Momentum conservation: (m1*v1 + m2*v2) / (m1+m2)
            // Here mass is proportional to depth (assuming constant density)
            let new_vx = (old_vx * d0 + add_vx * dw) / new_depth;
            let new_vz = (old_vz * d0 + add_vz * dw) / new_depth;
            
            water_velocity_x[idx] = new_vx;
            water_velocity_z[idx] = new_vz;
        }
        water_depth[idx] = new_depth;
    }
}
