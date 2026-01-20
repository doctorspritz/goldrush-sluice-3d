// Particle Absorption Shader
// Sinks 3D particles into the 2.5D heightfield world.

struct AbsorptionParams {
    particle_count: u32,
    world_width: u32,
    world_depth: u32,
    cell_size: f32,
    sediment_volume_per_particle: f32,
    water_volume_per_particle: f32,
    dt: f32,
    absorption_radius: f32,
}

@group(0) @binding(0) var<uniform> params: AbsorptionParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read> bed_height: array<f32>;
@group(0) @binding(5) var<storage, read_write> transfer_sediment: array<atomic<i32>>; // encoded f32 fixed point
@group(0) @binding(6) var<storage, read_write> transfer_water: array<atomic<i32>>;    // encoded f32 fixed point

const SCALE: f32 = 100000.0;

@compute @workgroup_size(256)
fn absorb_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let pid = id.x;
    if (pid >= params.particle_count) { return; }

    var p_vec = positions[pid];
    if (p_vec.w < 0.5) { return; } // Inactive/already absorbed

    let pos = p_vec.xyz;
    let vel = velocities[pid].xyz;
    let density = densities[pid];

    let ix = i32(floor(pos.x / params.cell_size));
    let iz = i32(floor(pos.z / params.cell_size));
    
    if (ix < 0 || iz < 0 || u32(ix) >= params.world_width || u32(iz) >= params.world_depth) {
        positions[pid].w = 0.0; // Out of bounds, just kill
        return;
    }

    let b_idx = u32(iz) * params.world_width + u32(ix);
    let ground = bed_height[b_idx];

    // Absorption criteria
    let dist_to_ground = pos.y - ground;
    
    // We absorb if:
    // 1. Below ground (clipping)
    // 2. Very close to ground and slow
    let speed = length(vel);
    let is_fast = speed > 0.5;
    
    if (dist_to_ground < 0.0 || (dist_to_ground < params.absorption_radius && !is_fast)) {
        if (density > 1.0) {
            // Sediment
            atomicAdd(&transfer_sediment[b_idx], i32(params.sediment_volume_per_particle * SCALE));
        } else {
            // Water
            atomicAdd(&transfer_water[b_idx], i32(params.water_volume_per_particle * SCALE));
        }
        
        // Deactivate particle
        positions[pid].w = 0.0;
        velocities[pid] = vec4<f32>(0.0);
    }
}
