// heightfield_emitter.wgsl
// Water emitter shader - adds water to specified location

struct EmitterParams {
    pos_x: f32,
    pos_z: f32,
    radius: f32,
    rate: f32,  // volume per second
    dt: f32,
    enabled: u32,  // 0 or 1
    width: u32,
    depth: u32,
}

@group(0) @binding(0) var<uniform> params: EmitterParams;
@group(0) @binding(1) var<storage, read_write> water_depth: array<f32>;

fn get_idx(x: u32, z: u32) -> u32 {
    return z * params.width + x;
}

@compute @workgroup_size(16, 16)
fn add_water(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (params.enabled == 0u) { return; }
    
    let x = global_id.x;
    let z = global_id.y;
    if (x >= params.width || z >= params.depth) { return; }
    
    // Calculate distance from emitter center
    let fx = f32(x);
    let fz = f32(z);
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
        
        water_depth[idx] += depth_per_cell;
    }
}
