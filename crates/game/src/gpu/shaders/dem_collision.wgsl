//! DEM Collision Detection: Narrow Phase
//!
//! Checks sphere-sphere collisions between particles found in broad phase.
//! Uses spring-damper contact model for force calculation.

@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> particle_velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> particle_radii: array<f32>>;
@group(0) @binding(3) var<storage, read> particle_masses: array<f32>>;
@group(0) @binding(4) var<storage, read> particle_template_ids: array<u32>>;
@group(0) @binding(5) var<storage, read> sphere_offsets: array<vec3<f32>>;
@group(0) @binding(6) var<storage, read> template_sphere_counts: array<u32>>;
@group(0) @binding(7) var<storage, read> hash_table: array<u32>>;
@group(0) @binding(8) var<storage, read> hash_entries: array<HashEntry>>;
@group(0) @binding(9) var<storage, read_write> contacts: array<GpuContact>>;
@group(0) @binding(10) var<storage, read_write> forces: array<vec3<f32>>;
@group(0) @binding(11) var<uniform> params: DemParams;
@group(0) @binding(12) var<storage, read_write> torques: array<vec3<f32>>;

struct GpuContact {
    particle_a: u32,
    sphere_a: u32,
    particle_b: u32,
    sphere_b: u32,
    normal: vec3<f32>,
    penetration: f32,
    normal_vel: f32,
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
    gravity: vec3<f32>,
    cell_size: f32,
    max_particles: u32,
}

const EMPTY_SLOT = 0xffffffffu;
const WORKGROUP_SIZE = 64u;
const MAX_SPHERES_PER_CLUMP = 100u;

// 3D spatial hash function (must match broad phase)
fn hash_3d(coord: vec3<i32>) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    
    let x = bitcast<u32>(coord.x);
    let y = bitcast<u32>(coord.y);
    let z = bitcast<u32>(coord.z);
    
    // Use table size from params or a known constant
    return ((x * p1) ^ (y * p2) ^ (z * p3)) % (1u << 20);
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
    
    if dist_sq >= min_dist * min_dist {
        return vec3<f32>(0.0, 0.0, 0.0); // No collision
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
    forces[particle_idx] = vec3<f32>(0.0, 0.0, 0.0);
    torques[particle_idx] = vec3<f32>(0.0, 0.0, 0.0);
    
    let pos = particle_positions[particle_idx];
    let vel = particle_velocities[particle_idx];
    let radius = particle_radii[particle_idx];
    let template_id = particle_template_ids[particle_idx];
    
    // Get sphere count for this clump
    if template_id >= MAX_SPHERES_PER_CLUMP { return; }
    let sphere_count = template_sphere_counts[template_id];
    
    // Check each sphere in the clump
    var sphere_start = template_id * MAX_SPHERES_PER_CLUMP;
    for sphere_i in 0u..sphere_count {
        let local_offset = sphere_offsets[sphere_start + sphere_i];
        let world_pos = pos + local_offset; // Simplified: no rotation for now
        
        // Find which cell this sphere is in
        let cell_coord = vec3<i32>(
            i32(floor(world_pos.x / params.cell_size)),
            i32(floor(world_pos.y / params.cell_size)),
            i32(floor(world_pos.z / params.cell_size))
        );
        
        // Look up hash entries for this cell
        let hash = hash_3d(cell_coord);
        var entry_idx = hash_table[hash];
        
        // Check all particles in this cell
        loop {
            if entry_idx == EMPTY_SLOT || entry_idx == particle_idx {
                break;
            }
            
            let other_pos = particle_positions[entry_idx];
            let delta = world_pos - other_pos;
            let dist_sq = dot(delta, delta);
            
            // Quick bounding sphere check
            let other_radius = particle_radii[entry_idx];
            let min_dist = radius + other_radius;
            
            if dist_sq < min_dist * min_dist {
                // Potential collision - check each sphere in other clump
                let other_template_id = particle_template_ids[entry_idx];
                let other_sphere_count = template_sphere_counts[other_template_id];
                let other_sphere_start = other_template_id * MAX_SPHERES_PER_CLUMP;
                
                for other_sphere_i in 0u..other_sphere_count {
                    let other_local_offset = sphere_offsets[other_sphere_start + other_sphere_i];
                    let other_world_pos = other_pos + other_local_offset;
                    
                    // Calculate collision force
                    let force = collide_spheres(
                        world_pos, vel, radius,
                        other_world_pos, particle_velocities[entry_idx], other_radius
                    );
                    
                    if length(force) > 0.001 {
                        // Record contact (simplified - using particle level)
                        // In a full implementation, we'd track sphere-sphere contacts
                        let total_force = force; // Accumulate all sphere forces
                        
                        // Apply to this particle
                        forces[particle_idx] += total_force;
                        
                        // Calculate torque: r x F
                        // r is vector from center of mass to contact point
                        // Contact point approx: center + normal * radius
                        // Force acts on A, so normal is B->A? No, collide_spheres returns B->A.
                        // normal inside collide_spheres was A->B.
                        // We need A->Contact.
                        let delta = other_world_pos - world_pos;
                        let dist = length(delta);
                        let dir = delta / dist; // A->B
                        let r = dir * radius;   // vector to surface
                        
                        let torque = cross(r, total_force);
                        torques[particle_idx] += torque;
                    }
                }
            }
            
            // Move to next entry in linked list
            if entry_idx >= arrayLength(&hash_entries) {
                break;
            }
            entry_idx = hash_entries[entry_idx].next_idx;
        }
    }
}