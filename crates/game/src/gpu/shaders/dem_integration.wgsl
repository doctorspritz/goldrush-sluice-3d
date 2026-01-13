//! DEM Integration: Position and Velocity Update
//!
//! Integrates forces and updates particle positions.
//! Handles both linear and angular motion for clumps.

@group(0) @binding(0) var<storage, read_write> particle_positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> particle_velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> particle_angular_velocities: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> particle_masses: array<f32>>;
@group(0) @binding(4) var<storage, read> particle_radii: array<f32>>;
@group(0) @binding(5) var<storage, read> particle_flags: array<u32>>;
@group(0) @binding(6) var<storage, read> forces: array<vec3<f32>>;
@group(0) @binding(7) var<uniform> params: DemParams;
@group(0) @binding(8) var<storage, read_write> particle_orientations: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> torques: array<vec3<f32>>;

struct DemParams {
    dt: f32,
    stiffness: f32,
    damping: f32,
    friction: f32,
    gravity: vec3<f32>,
    cell_size: f32,
    max_particles: u32,
}

const WORKGROUP_SIZE = 64u;
const PARTICLE_ACTIVE = 1u;

// Quaternion multiplication (simplified)
fn quat_mul(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    let w = q1.w * q2.w - dot(q1.xyz, q2.xyz);
    let xyz = q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz);
    return vec4<f32>(xyz.x, xyz.y, xyz.z, w);
}

// Normalize quaternion
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let len_sq = dot(q, q);
    if len_sq > 0.0001 {
        return q / sqrt(len_sq);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles { return; }
    
    let flags = particle_flags[particle_idx];
    if (flags & PARTICLE_ACTIVE) == 0u { return; } // Skip inactive particles
    
    let mass = particle_masses[particle_idx];
    if mass <= 0.0 { return; } // Skip invalid particles
    
    // Get accumulated forces
    let force = forces[particle_idx];
    
    // Apply gravity (buoyancy-corrected if needed)
    let total_force = force + params.gravity * mass;
    
    // Linear motion (Euler integration)
    let accel = total_force / mass;
    let new_vel = particle_velocities[particle_idx] + accel * params.dt;
    
    // Clamp velocity for stability
    let max_vel = 50.0; // m/s
    let clamped_vel = clamp(new_vel, vec3<f32>(-max_vel, -max_vel, -max_vel), vec3<f32>(max_vel, max_vel, max_vel));
    
    // Update position
    let new_pos = particle_positions[particle_idx] + clamped_vel * params.dt;
    
    // Angular motion
    let torque = torques[particle_idx];
    let radius = particle_radii[particle_idx];
    
    // Moment of inertia for solid sphere: 2/5 * M * R^2
    let inertia = 0.4 * mass * radius * radius;
    let safe_inertia = max(inertia, 1e-6);
    
    let ang_accel = torque / safe_inertia;
    
    // Update angular velocity with damping
    let damping = 0.99; // Simple rotational drag
    let new_angular_vel = (particle_angular_velocities[particle_idx] + ang_accel * params.dt) * damping;
    
    // Update angular velocity buffer
    particle_angular_velocities[particle_idx] = new_angular_vel;
    
    let angular_delta = new_angular_vel * params.dt * 0.5; // Half angle for quaternion
    
    // Create quaternion from angular velocity
    let angle = length(angular_delta);
    if angle > 0.0001 {
        let axis = normalize(angular_delta);
        let half_angle = angle * 0.5;
        let rot_quat = vec4<f32>(
            axis.x * sin(half_angle),
            axis.y * sin(half_angle),
            axis.z * sin(half_angle),
            cos(half_angle)
        );
        
        // Update orientation (simplified - no full quaternion integration)
        // Update orientation
        let current_orient = particle_orientations[particle_idx];
        let new_orient = quat_mul(rot_quat, current_orient);
        
        particle_orientations[particle_idx] = quat_normalize(new_orient);
    }
    
    // Write back updated values
    particle_velocities[particle_idx] = clamped_vel;
    particle_positions[particle_idx] = new_pos;
    
    // Simple boundary checking (remove particles that fall too far)
    if new_pos.y < -100.0 || length(new_pos) > 1000.0 {
        // Deactivate particle
        particle_flags[particle_idx] = flags & ~PARTICLE_ACTIVE;
    }
}