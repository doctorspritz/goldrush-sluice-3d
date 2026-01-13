//! DEM Collision Detection: Narrow Phase
//!
//! Checks sphere-sphere collisions between particles found in broad phase.
//! Uses spring-damper contact model for force calculation.

@group(0) @binding(0) var<storage, read> particle_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particle_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> particle_template_ids: array<u32>;
@group(0) @binding(3) var<storage, read> sphere_offsets: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> templates: array<GpuClumpTemplate>;
@group(0) @binding(5) var<storage, read> hash_table: array<u32>;
@group(0) @binding(6) var<storage, read> hash_entries: array<HashEntry>;
@group(0) @binding(8) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(9) var<uniform> params: DemParams;
@group(0) @binding(10) var<storage, read_write> torques: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> sphere_radii: array<f32>;

struct GpuClumpTemplate {
    sphere_count: u32,
    mass: f32,
    radius: f32,
    pad0: f32,
    inertia_inv: mat3x3<f32>,
}

struct HashEntry {
    particle_idx: u32,
    next_idx: u32,
}

struct DemParams {
    dt: f32,
    stiffness: f32,
    damping: f32,
    friction: f32,
    gravity: vec4<f32>,
    cell_size: f32,
    max_particles: u32,
    pad0: f32,
    pad1: f32,
}

const EMPTY_SLOT = 0xffffffffu;
const WORKGROUP_SIZE = 64u;
const MAX_SPHERES_PER_CLUMP = 100u;
const MAX_TEMPLATES = 100u;

// Hash table size - MUST match dem_broadphase.wgsl and dem_3d.rs
const HASH_TABLE_SIZE = 1048576u; // 1 << 20

// 3D spatial hash function (must match broad phase)
fn hash_3d(coord: vec3<i32>) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    
    let x = bitcast<u32>(coord.x);
    let y = bitcast<u32>(coord.y);
    let z = bitcast<u32>(coord.z);
    
    // Use same table size as broadphase
    return ((x * p1) ^ (y * p2) ^ (z * p3)) % HASH_TABLE_SIZE;
}

// Sphere-sphere collision response
fn collide_spheres(
    pos_a: vec3<f32>,
    vel_a: vec3<f32>,
    radius_a: f32,
    pos_b: vec3<f32>,
    vel_b: vec3<f32>,
    radius_b: f32
) -> vec3<f32> {
    let delta = pos_b - pos_a;
    let dist_sq = dot(delta, delta);
    let min_dist = radius_a + radius_b;
    
    if dist_sq >= min_dist * min_dist || dist_sq < 1e-10 {
        return vec3<f32>(0.0, 0.0, 0.0); // No collision or too close to normalize
    }
    
    let dist = sqrt(dist_sq);
    let normal = delta / dist;
    let penetration = min_dist - dist;
    
    // Relative velocity along normal
    let relative_vel = vel_b - vel_a;
    let normal_vel = dot(relative_vel, normal);
    
    // Spring-damper contact model
    let spring_force = params.stiffness * penetration;
    let damper_force = params.damping * normal_vel;
    
    return -normal * (spring_force - damper_force);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles { return; }
    
    // Initialize force to zero
    forces[particle_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    torques[particle_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    let pos = particle_positions[particle_idx].xyz;
    let vel = particle_velocities[particle_idx].xyz;
    let template_id = particle_template_ids[particle_idx];
    
    // Safety check for template ID
    if template_id >= MAX_TEMPLATES { return; }
    
    let bounding_radius = templates[template_id].radius; // Bounding radius for broad checks
    
    let sphere_count = min(templates[template_id].sphere_count, MAX_SPHERES_PER_CLUMP);
    
    // Check each sphere in the clump
    var sphere_start = template_id * MAX_SPHERES_PER_CLUMP;
    for (var sphere_i = 0u; sphere_i < sphere_count; sphere_i++) {
        let local_offset = sphere_offsets[sphere_start + sphere_i].xyz;
        let sphere_radius = sphere_radii[sphere_start + sphere_i];
        let world_pos = pos + sphere_offsets[sphere_start + sphere_i].xyz; // Simplified: no rotation for now
        
        // Find which cell this sphere is in
        let cell_coord = vec3<i32>(
            i32(floor(world_pos.x / params.cell_size)),
            i32(floor(world_pos.y / params.cell_size)),
            i32(floor(world_pos.z / params.cell_size))
        );
        
        // Look up hash entries for this cell
        let hash = hash_3d(cell_coord);
        var entry_idx = hash_table[hash];
        
        // Traverse linked list with safety limit
        // Max iterations = reasonable max particles that could be in one cell
        var iterations = 0u;
        let MAX_ITERATIONS = 1000u;
        
        loop {
            if entry_idx == EMPTY_SLOT {
                break;
            }
            
            // Safety: prevent infinite loops from corrupted data
            iterations += 1u;
            if iterations > MAX_ITERATIONS {
                break;
            }
            
            let other_particle_idx = hash_entries[entry_idx].particle_idx;
            
            if other_particle_idx >= params.max_particles {
                entry_idx = hash_entries[entry_idx].next_idx;
                continue;
            }
            
            // Skip self
            if other_particle_idx == particle_idx {
                entry_idx = hash_entries[entry_idx].next_idx;
                continue;
            }
            
            let other_pos = particle_positions[other_particle_idx].xyz;
            let delta = world_pos - other_pos;
            let dist_sq = dot(delta, delta);
            
            // Quick bounding sphere check
            let other_template_id = particle_template_ids[other_particle_idx];
            if (other_template_id >= 100u) {
                entry_idx = hash_entries[entry_idx].next_idx;
                continue;
            }
            
            let other_bounding_radius = templates[other_template_id].radius;
            let min_dist = bounding_radius + other_bounding_radius;
            
            if dist_sq < min_dist * min_dist {
                // Potential collision - check each sphere in other clump
                let other_sphere_count = min(templates[other_template_id].sphere_count, MAX_SPHERES_PER_CLUMP);
                let other_sphere_start = other_template_id * MAX_SPHERES_PER_CLUMP;
                
                for (var other_sphere_i = 0u; other_sphere_i < other_sphere_count; other_sphere_i++) {
                    let other_local_offset = sphere_offsets[other_sphere_start + other_sphere_i].xyz;
                    let other_sphere_radius = sphere_radii[other_sphere_start + other_sphere_i];
                    let other_world_pos = other_pos + other_local_offset;
                    
                    // Calculate collision force
                    let force = collide_spheres(
                        world_pos, vel, sphere_radius,
                        other_world_pos, particle_velocities[other_particle_idx].xyz, other_sphere_radius
                    );
                    
                    if length(force) > 0.001 {
                        // Record contact (simplified - using particle level)
                        // In a full implementation, we'd track sphere-sphere contacts
                        let total_force = force; // Accumulate all sphere forces
                        
                        // Apply to this particle
                        forces[particle_idx] += vec4<f32>(total_force, 0.0);
                        
                        // Calculate torque: r x F
                        let delta_contact = other_world_pos - world_pos;
                        let dist_contact = length(delta_contact);
                        if (dist_contact > 1e-6) {
                            let dir = delta_contact / dist_contact; // A->B
                            let r = dir * sphere_radius;   // vector to surface
                            
                            let torque = cross(r, total_force);
                            torques[particle_idx] += vec4<f32>(torque, 0.0);
                        }
                    }
                }
            }
            
            // Move to next entry in linked list
            entry_idx = hash_entries[entry_idx].next_idx;
        }
    }
}