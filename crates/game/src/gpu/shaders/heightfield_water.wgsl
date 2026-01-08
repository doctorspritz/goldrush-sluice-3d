// heightfield_water.wgsl
// Shallow Water Equations (SWE) solver for heightfield water simulation.

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

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> water_depth: array<f32>;
@group(1) @binding(1) var<storage, read_write> water_velocity_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> water_velocity_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> water_surface: array<f32>;
@group(1) @binding(4) var<storage, read_write> flux_x: array<f32>;
@group(1) @binding(5) var<storage, read_write> flux_z: array<f32>;

// Terrain Bind Group (Read Write for Layout Compatibility)
@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.world_width + x;
}

fn get_ground_height(idx: u32) -> f32 {
    return bedrock[idx] + paydirt[idx] + gravel[idx] + overburden[idx] + sediment[idx];
}

// 1. Calculate Water Surface Height
@compute @workgroup_size(16, 16)
fn update_surface(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }
    
    let idx = get_idx(x, z);
    let ground = get_ground_height(idx);
    let depth = water_depth[idx];
    
    water_surface[idx] = ground + depth;
}

// 2. Calculate Flux (Velocity Update)
@compute @workgroup_size(16, 16)
fn update_flux(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }
    
    let idx = get_idx(x, z);
    let cell_area = params.cell_size * params.cell_size;
    
    // X-Flux (Flow to Right: x -> x+1)
    if (x < params.world_width - 1) {
        let idx_r = get_idx(x + 1, z);
        let h_l = water_surface[idx];
        let h_r = water_surface[idx_r];
        let d_l = water_depth[idx];
        let d_r = water_depth[idx_r];
        
        // Only calculate flow if water exists
        if (d_l > 0.001 || d_r > 0.001) {
             let gradient = (h_l - h_r) / params.cell_size;
             var vel = water_velocity_x[idx];
             vel += params.gravity * gradient * params.dt;
             vel *= params.damping;
             
             // Clamp velocity
             let max_v = params.cell_size / params.dt * 0.25; // More conservative
             vel = clamp(vel, -max_v, max_v);
             
             water_velocity_x[idx] = vel;
             
             // Compute Flux (volume per timestep)
             let avg_depth = 0.5 * (d_l + d_r);
             var flux = vel * avg_depth * params.cell_size * params.dt;
             
             // CRITICAL: Limit flux to available water in source cell
             // Positive flux = flow from left to right, source is left cell
             // Negative flux = flow from right to left, source is right cell
             if (flux > 0.0) {
                 let max_flux = d_l * cell_area * 0.25; // Max 25% of cell per timestep
                 flux = min(flux, max_flux);
             } else {
                 let max_flux = d_r * cell_area * 0.25;
                 flux = max(flux, -max_flux);
             }
             
             flux_x[idx] = flux;
        } else {
             water_velocity_x[idx] = 0.0;
             flux_x[idx] = 0.0;
        }
    } else if (x == params.world_width - 1) {
         water_velocity_x[idx] = 0.0;
         flux_x[idx] = 0.0;
    }
    
    // Z-Flux (Flow Forward: z -> z+1)
    if (z < params.world_depth - 1) {
        let idx_f = get_idx(x, z + 1);
        let h_b = water_surface[idx];
        let h_f = water_surface[idx_f];
        let d_b = water_depth[idx];
        let d_f = water_depth[idx_f];
        
        if (d_b > 0.001 || d_f > 0.001) {
             let gradient = (h_b - h_f) / params.cell_size;
             var vel = water_velocity_z[idx];
             vel += params.gravity * gradient * params.dt;
             vel *= params.damping;
             
             let max_v = params.cell_size / params.dt * 0.25;
             vel = clamp(vel, -max_v, max_v);
             
             water_velocity_z[idx] = vel;
             
             let avg_depth = 0.5 * (d_b + d_f);
             var flux = vel * avg_depth * params.cell_size * params.dt;
             
             // Limit flux to available water
             if (flux > 0.0) {
                 let max_flux = d_b * cell_area * 0.25;
                 flux = min(flux, max_flux);
             } else {
                 let max_flux = d_f * cell_area * 0.25;
                 flux = max(flux, -max_flux);
             }
             
             flux_z[idx] = flux;
        } else {
             water_velocity_z[idx] = 0.0;
             flux_z[idx] = 0.0;
        }
    } else if (z == params.world_depth - 1) {
         water_velocity_z[idx] = 0.0;
         flux_z[idx] = 0.0;
    }
}

// 3. Update Depth (Volume Conservation)
@compute @workgroup_size(16, 16)
fn update_depth(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }
    
    let idx = get_idx(x, z);
    
    // Flux convention: flux_x[i] = flow from cell i to cell i+1 (positive = rightward)
    // For cell x: 
    //   - Inflow from left = max(0, flux_x[x-1]) (positive flow from x-1 to x)
    //   - Outflow to left = max(0, -flux_x[x-1]) (negative flow from x-1 to x means x flows left)
    //   - Outflow to right = max(0, flux_x[x]) (positive flow from x to x+1)
    //   - Inflow from right = max(0, -flux_x[x]) (negative flow means x+1 flows to x)
    
    var net_flux = 0.0;
    
    // X direction
    if (x > 0) {
        let f = flux_x[get_idx(x - 1, z)];
        net_flux += f; // Positive = inflow, Negative = outflow
    }
    if (x < params.world_width - 1) {
        let f = flux_x[idx];
        net_flux -= f; // Positive = outflow, Negative = inflow  
    }
    
    // Z direction
    if (z > 0) {
        let f = flux_z[get_idx(x, z - 1)];
        net_flux += f;
    }
    if (z < params.world_depth - 1) {
        let f = flux_z[idx];
        net_flux -= f;
    }
    
    // Update Depth
    let cell_area = params.cell_size * params.cell_size;
    let depth_change = net_flux / cell_area;
    
    var new_depth = water_depth[idx] + depth_change;
    
    // Clamp to prevent negative depth
    new_depth = max(0.0, new_depth);
    
    // Open Boundary: Drain at edges
    if (x == 0 || x == params.world_width - 1 || z == 0 || z == params.world_depth - 1) {
        new_depth = 0.0;
    }
    
    water_depth[idx] = new_depth;
}
