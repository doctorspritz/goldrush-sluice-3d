// heightfield_water.wgsl
// Shallow Water Equations (SWE) solver for heightfield water simulation.
// Uses Manning friction for physically accurate flow velocities.

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
    manning_n: f32, // Manning roughness coefficient (typical: 0.03 smooth, 0.05 rough)
}

@group(0) @binding(0) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read_write> water_depth: array<f32>;
@group(1) @binding(1) var<storage, read_write> water_velocity_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> water_velocity_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> water_surface: array<f32>;
@group(1) @binding(4) var<storage, read_write> flux_x: array<f32>;
@group(1) @binding(5) var<storage, read_write> flux_z: array<f32>;
@group(1) @binding(6) var<storage, read_write> suspended_sediment: array<f32>; // For bind group compatibility
@group(1) @binding(7) var<storage, read_write> suspended_sediment_next: array<f32>; // For bind group compatibility

// Terrain Bind Group (Read Write for Layout Compatibility)
@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;
@group(2) @binding(5) var<storage, read_write> surface_material: array<u32>; // For bind group compatibility

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

// Reconcile depth to preserve surface height after terrain changes.
@compute @workgroup_size(16, 16)
fn reconcile_depth(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= params.tile_width || tile_z >= params.tile_depth) { return; }
    let x = tile_x + params.origin_x;
    let z = tile_z + params.origin_z;
    if (x >= params.world_width || z >= params.world_depth) { return; }

    let idx = get_idx(x, z);
    let ground = get_ground_height(idx);
    let surface = water_surface[idx];
    let new_depth = max(0.0, surface - ground);

    water_depth[idx] = new_depth;

    // Kill residual velocities in newly dry cells to avoid oscillation.
    if (new_depth < 0.001) {
        water_velocity_x[idx] = 0.0;
        water_velocity_z[idx] = 0.0;
    }
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
             // Detect wet-dry front
             let is_wet_dry_front = (d_l > 0.01 && d_r < 0.01) || (d_r > 0.01 && d_l < 0.01);

             // Use harmonic mean of depths (better for wet-dry) unless truly zero
             let min_depth = min(d_l, d_r);
             let max_depth = max(d_l, d_r);
             let avg_depth = select(
                 0.5 * (d_l + d_r),  // Normal: arithmetic mean
                 max_depth * 0.5,    // Wet-dry: half of wet cell's depth
                 is_wet_dry_front
             );
             let effective_depth = max(avg_depth, 0.01);

             // Compute gradient from water surface heights
             let gradient = (h_l - h_r) / params.cell_size;

             // Apply gravity
             var vel = water_velocity_x[idx];
             vel += params.gravity * gradient * params.dt;

             // Apply Manning friction (implicit method for stability)
             let n2 = params.manning_n * params.manning_n;
             let friction_coeff = params.gravity * n2 / pow(effective_depth, 4.0/3.0);
             let friction_factor = 1.0 + friction_coeff * abs(vel) * params.dt;
             vel = vel / friction_factor;

             // Standard velocity clamp (CFL condition)
             let max_v = params.cell_size / params.dt * 0.25;
             vel = clamp(vel, -max_v, max_v);

             water_velocity_x[idx] = vel;

             // Compute Flux - key is limiting flux at wet-dry front
             var flux = vel * effective_depth * params.cell_size * params.dt;

             // Limit flux to available water in source cell
             // At wet-dry front, use much stricter limit to prevent pile-up
             let flux_limit_factor = select(0.25, 0.1, is_wet_dry_front);
             if (flux > 0.0) {
                 let max_flux = d_l * cell_area * flux_limit_factor;
                 flux = min(flux, max_flux);
             } else {
                 let max_flux = d_r * cell_area * flux_limit_factor;
                 flux = max(flux, -max_flux);
             }

             flux_x[idx] = flux;
        } else {
             water_velocity_x[idx] = 0.0;
             flux_x[idx] = 0.0;
        }
    } else if (x == params.world_width - 1) {
        // OPEN BOUNDARY: Allow water to flow OUT at right edge
        let d_local = water_depth[idx];
        if (d_local > 0.001) {
            var vel = water_velocity_x[idx];
            // Assume slight downhill gradient to encourage outflow
            vel += params.gravity * 0.01 * params.dt;

            // Apply Manning friction at boundary too
            let n2 = params.manning_n * params.manning_n;
            let friction_coeff = params.gravity * n2 / pow(max(d_local, 0.01), 4.0/3.0);
            let friction_factor = 1.0 + friction_coeff * abs(vel) * params.dt;
            vel = vel / friction_factor;

            // Only allow positive (outward) velocity at open boundary
            vel = max(vel, 0.0);
            let max_v = params.cell_size / params.dt * 0.25;
            vel = min(vel, max_v);
            water_velocity_x[idx] = vel;
            // Outflow flux based on local depth
            var flux = vel * d_local * params.cell_size * params.dt;
            let max_flux = d_local * cell_area * 0.5; // Allow more outflow at boundary
            flux = min(flux, max_flux);
            flux_x[idx] = flux;
        } else {
            water_velocity_x[idx] = 0.0;
            flux_x[idx] = 0.0;
        }
    }
    
    // Z-Flux (Flow Forward: z -> z+1)
    if (z < params.world_depth - 1) {
        let idx_f = get_idx(x, z + 1);
        let h_b = water_surface[idx];
        let h_f = water_surface[idx_f];
        let d_b = water_depth[idx];
        let d_f = water_depth[idx_f];
        
        if (d_b > 0.001 || d_f > 0.001) {
             // Detect wet-dry front
             let is_wet_dry_front_z = (d_b > 0.01 && d_f < 0.01) || (d_f > 0.01 && d_b < 0.01);

             // Use half of wet cell depth at wet-dry front
             let max_depth_z = max(d_b, d_f);
             let avg_depth_z = select(
                 0.5 * (d_b + d_f),  // Normal: arithmetic mean
                 max_depth_z * 0.5,  // Wet-dry: half of wet cell's depth
                 is_wet_dry_front_z
             );
             let effective_depth_z = max(avg_depth_z, 0.01);

             // Compute gradient from water surface heights
             let gradient = (h_b - h_f) / params.cell_size;

             // Apply gravity
             var vel = water_velocity_z[idx];
             vel += params.gravity * gradient * params.dt;

             // Apply Manning friction (implicit method for stability)
             let n2_z = params.manning_n * params.manning_n;
             let friction_coeff_z = params.gravity * n2_z / pow(effective_depth_z, 4.0/3.0);
             let friction_factor_z = 1.0 + friction_coeff_z * abs(vel) * params.dt;
             vel = vel / friction_factor_z;

             // Standard velocity clamp (CFL condition)
             let max_v_z = params.cell_size / params.dt * 0.25;
             vel = clamp(vel, -max_v_z, max_v_z);

             water_velocity_z[idx] = vel;

             // Compute Flux - key is limiting flux at wet-dry front
             var flux = vel * effective_depth_z * params.cell_size * params.dt;

             // Limit flux to available water in source cell
             // At wet-dry front, use stricter limit
             let flux_limit_z = select(0.25, 0.1, is_wet_dry_front_z);
             if (flux > 0.0) {
                 let max_flux = d_b * cell_area * flux_limit_z;
                 flux = min(flux, max_flux);
             } else {
                 let max_flux = d_f * cell_area * flux_limit_z;
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
        net_flux += f; // Positive = inflow from left
    }
    // Outflow to the right (includes open boundary at x=max)
    let f_out_x = flux_x[idx];
    net_flux -= f_out_x; // Positive = outflow to right

    // Z direction
    if (z > 0) {
        let f = flux_z[get_idx(x, z - 1)];
        net_flux += f; // Positive = inflow from back
    }
    if (z < params.world_depth - 1) {
        let f = flux_z[idx];
        net_flux -= f; // Positive = outflow to front
    }

    // Update Depth
    let cell_area = params.cell_size * params.cell_size;
    let depth_change = net_flux / cell_area;

    var new_depth = water_depth[idx] + depth_change;

    // Clamp to prevent negative depth
    new_depth = max(0.0, new_depth);

    water_depth[idx] = new_depth;
}
