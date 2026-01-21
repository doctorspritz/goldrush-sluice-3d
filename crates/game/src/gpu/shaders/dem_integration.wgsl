//! DEM Integration: Position and Velocity Update
//!
//! Integrates forces and updates particle positions.
//! Handles both linear and angular motion for clumps.

@group(0) @binding(0) var<storage, read_write> particle_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> particle_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> particle_angular_velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particle_flags: array<u32>;
@group(0) @binding(4) var<storage, read> forces: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: DemParams;
@group(0) @binding(6) var<storage, read_write> particle_orientations: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> torques: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> particle_template_ids: array<u32>;
@group(0) @binding(9) var<storage, read> templates: array<GpuClumpTemplate>;

struct GpuClumpTemplate {
    sphere_count: u32,
    mass: f32,
    radius: f32,
    particle_radius: f32,
    inertia_inv: mat3x3<f32>,
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

const WORKGROUP_SIZE = 64u;
const PARTICLE_ACTIVE = 1u;
const MAX_TEMPLATES = 100u;

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

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;

    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;

    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles { return; }
    
    let flags = particle_flags[particle_idx];
    if (flags & PARTICLE_ACTIVE) == 0u { return; } // Skip inactive particles
    
    let template_idx = particle_template_ids[particle_idx];
    if template_idx >= MAX_TEMPLATES { return; }
    
    let mass = templates[template_idx].mass;
    if mass <= 0.0 { return; } // Skip invalid particles
    
    // Get accumulated forces
    let force = forces[particle_idx].xyz;
    
    // Apply gravity (buoyancy-corrected if needed)
    let total_force = force + params.gravity.xyz * mass;
    
    // Linear motion (Euler integration)
    let accel = total_force / mass;
    let old_vel = particle_velocities[particle_idx].xyz;
    let new_vel = old_vel + accel * params.dt;
    
    // Clamp velocity for stability
    let max_vel = 50.0; // m/s
    // Clamped velocity
    let clamped_vel = clamp(new_vel, vec3<f32>(-max_vel, -max_vel, -max_vel), vec3<f32>(max_vel, max_vel, max_vel));
    
    // Update position
    let old_pos = particle_positions[particle_idx].xyz;
    // use real velocity for integration to keep simulation running somewhat? 
    // If we overwrite velocity buffer, next step uses 'force' as 'velocity'. 
    // This breaks simulation.
    // Ideally we want to see force at IMPACT.
    // If we overwrite, the particle will move with v=F.
    // F is huge? Particle flies away.
    // Position update uses 'clamped_vel' (local var). So integration uses correct vel.
    let new_pos = old_pos + clamped_vel * params.dt;
    
    // Angular motion
    let torque = torques[particle_idx].xyz;
    let orient = quat_normalize(particle_orientations[particle_idx]);
    let rot = quat_to_mat3(orient);
    let inertia_inv_local = templates[template_idx].inertia_inv;
    let inertia_inv_world = rot * inertia_inv_local * transpose(rot);
    let ang_accel = inertia_inv_world * torque;
    
    // Update angular velocity with damping
    let damping = 0.99; // Simple rotational drag
    let old_angular_vel = particle_angular_velocities[particle_idx].xyz;
    let new_angular_vel = (old_angular_vel + ang_accel * params.dt) * damping;
    
    // Update angular velocity buffer
    particle_angular_velocities[particle_idx] = vec4<f32>(new_angular_vel, 0.0);
    
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
        
        // Update orientation
        let current_orient = particle_orientations[particle_idx];
        let new_orient = quat_mul(rot_quat, current_orient);
        
        particle_orientations[particle_idx] = quat_normalize(new_orient);
    }
    
    // Write back updated values
    particle_velocities[particle_idx] = vec4<f32>(clamped_vel, 0.0);
    particle_positions[particle_idx] = vec4<f32>(new_pos, 0.0);
    
    // Simple boundary checking (remove particles that fall too far)
    if new_pos.y < -100.0 || length(new_pos) > 1000.0 {
        // Deactivate particle
        particle_flags[particle_idx] = flags & ~PARTICLE_ACTIVE;
    }
}
