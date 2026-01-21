//! DEM SDF Collision
//!
//! Handles collisions between DEM particles (clumps) and static SDF geometry.
//! Samples the SDF once per sphere in each clump and applies spring-damper forces.

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

struct SdfParams {
    grid_offset: vec4<f32>,
    grid_dims: vec4<u32>, // width, height, depth, pad
    cell_size: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<storage, read> particle_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particle_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> particle_angular_velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> particle_flags: array<u32>;
@group(0) @binding(4) var<storage, read> particle_template_ids: array<u32>;
@group(0) @binding(5) var<storage, read> particle_orientations: array<vec4<f32>>;

@group(0) @binding(6) var<storage, read> templates: array<GpuClumpTemplate>;
@group(0) @binding(7) var<storage, read> sphere_offsets: array<vec4<f32>>;

@group(0) @binding(8) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> torques: array<vec4<f32>>;

@group(0) @binding(10) var<uniform> params: DemParams;
@group(0) @binding(11) var<storage, read> sdf_buffer: array<f32>;
@group(0) @binding(12) var<uniform> sdf_params: SdfParams;

const WORKGROUP_SIZE = 64u;
const PARTICLE_ACTIVE = 1u;
const MAX_TEMPLATES = 100u;
const MAX_SPHERES_PER_CLUMP = 100u;

// Quaternion multiplication
fn quat_mul_vec3(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn sample_sdf(idx: vec3<u32>) -> f32 {
    let w = sdf_params.grid_dims.x;
    let h = sdf_params.grid_dims.y;
    let d = sdf_params.grid_dims.z;
    
    if idx.x >= w || idx.y >= h || idx.z >= d {
        return 1.0; // Far from solid outside bounds
    }
    
    let index = idx.z * w * h + idx.y * w + idx.x;
    return sdf_buffer[index];
}

struct SdfSample {
    value: f32,
    gradient: vec3<f32>,
}

fn sample_sdf_with_gradient(world_pos: vec3<f32>) -> SdfSample {
    let local_pos = world_pos - sdf_params.grid_offset.xyz;
    let cell_size = sdf_params.cell_size;
    
    // Integer samples are at cell centers (0.5, 1.5, etc)
    // So we shift by -0.5 to align grid coordinates with indices
    let fx = local_pos.x / cell_size - 0.5;
    let fy = local_pos.y / cell_size - 0.5;
    let fz = local_pos.z / cell_size - 0.5;
    
    let gw = f32(sdf_params.grid_dims.x);
    let gh = f32(sdf_params.grid_dims.y);
    let gd = f32(sdf_params.grid_dims.z);
    
    // Clamp to valid range for interpolation
    // We can go up to dim-1.0. If we go past, sample_sdf returns 1.0 (outside)
    let p = clamp(vec3<f32>(fx, fy, fz), vec3<f32>(0.0), vec3<f32>(gw - 1.001, gh - 1.001, gd - 1.001));
    
    let i = vec3<u32>(p);
    let t = p - vec3<f32>(i);
    
    // Sample 8 corners for value interpolation
    let c000 = sample_sdf(i + vec3<u32>(0, 0, 0));
    let c100 = sample_sdf(i + vec3<u32>(1, 0, 0));
    let c010 = sample_sdf(i + vec3<u32>(0, 1, 0));
    let c110 = sample_sdf(i + vec3<u32>(1, 1, 0));
    let c001 = sample_sdf(i + vec3<u32>(0, 0, 1));
    let c101 = sample_sdf(i + vec3<u32>(1, 0, 1));
    let c011 = sample_sdf(i + vec3<u32>(0, 1, 1));
    let c111 = sample_sdf(i + vec3<u32>(1, 1, 1));
    
    // Trilinear interpolation
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    
    let sdf_value = mix(c0, c1, t.z);
    
    // Proper central differences for gradient
    let h_step = 0.5;
    
    let gx_p = sample_sdf(vec3<u32>(clamp(p + vec3<f32>(h_step, 0.0, 0.0), vec3<f32>(0.0), vec3<f32>(gw-1.0, gh-1.0, gd-1.0))));
    let gx_m = sample_sdf(vec3<u32>(clamp(p - vec3<f32>(h_step, 0.0, 0.0), vec3<f32>(0.0), vec3<f32>(gw-1.0, gh-1.0, gd-1.0))));
    let grad_x = (gx_p - gx_m) / (2.0 * h_step * cell_size);
    
    let gy_p = sample_sdf(vec3<u32>(clamp(p + vec3<f32>(0.0, h_step, 0.0), vec3<f32>(0.0), vec3<f32>(gw-1.0, gh-1.0, gd-1.0))));
    let gy_m = sample_sdf(vec3<u32>(clamp(p - vec3<f32>(0.0, h_step, 0.0), vec3<f32>(0.0), vec3<f32>(gw-1.0, gh-1.0, gd-1.0))));
    let grad_y = (gy_p - gy_m) / (2.0 * h_step * cell_size);
    
    let gz_p = sample_sdf(vec3<u32>(clamp(p + vec3<f32>(0.0, 0.0, h_step), vec3<f32>(0.0), vec3<f32>(gw-1.0, gh-1.0, gd-1.0))));
    let gz_m = sample_sdf(vec3<u32>(clamp(p - vec3<f32>(0.0, 0.0, h_step), vec3<f32>(0.0), vec3<f32>(gw-1.0, gh-1.0, gd-1.0))));
    let grad_z = (gz_p - gz_m) / (2.0 * h_step * cell_size);
    
    var res: SdfSample;
    res.value = sdf_value;
    // DEBUG: Force vertical gradient to test lateral instability
    res.gradient = vec3<f32>(0.0, grad_y, 0.0);
    return res;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles { return; }
    
    let flags = particle_flags[particle_idx];
    if (flags & PARTICLE_ACTIVE) == 0u { return; }
    
    let template_idx = particle_template_ids[particle_idx];
    if template_idx >= MAX_TEMPLATES { return; }
    
    let clump_template = templates[template_idx];
    let clump_pos = particle_positions[particle_idx].xyz;
    let clump_vel = particle_velocities[particle_idx].xyz;
    let clump_ang_vel = particle_angular_velocities[particle_idx].xyz;
    let clump_orient = particle_orientations[particle_idx];
    
    var total_force = vec3<f32>(0.0);
    var total_torque = vec3<f32>(0.0);
    
    let sphere_base_offset = template_idx * MAX_SPHERES_PER_CLUMP;
    
    for (var i = 0u; i < clump_template.sphere_count; i++) {
        if (i >= MAX_SPHERES_PER_CLUMP) { break; }
        
        let local_offset = sphere_offsets[sphere_base_offset + i].xyz;
        let r = quat_mul_vec3(clump_orient, local_offset);
        let pos = clump_pos + r;
        let vel = clump_vel + cross(clump_ang_vel, r);
        let radius = clump_template.radius; // Assuming uniform sphere radius for now
        
        let sample = sample_sdf_with_gradient(pos);
        let penetration = radius - sample.value;
        
        if penetration > 0.0 && length(sample.gradient) > 1e-6 {
            let normal = normalize(sample.gradient);
            
            // Limit penetration for stability
            let capped_penetration = min(penetration, radius * 0.5);
            
            // Spring-damper model
            // v_n > 0 when moving away from surface, < 0 when approaching
            // Using -v_n as approach velocity so the formula matches dem_collision.wgsl
            let v_n = dot(vel, normal);
            let v_approach = -v_n;  // positive when approaching surface

            // Standard model: F = k*pen + c*v_approach (both terms resist penetration)
            let fn_mag = max(0.0, params.stiffness * capped_penetration + params.damping * v_approach);
            
            // Friction
            let vt = vel - normal * v_n;
            var ft = -params.stiffness * 0.5 * vt * params.dt - params.damping * 0.5 * vt; // simple tangential
            let max_ft = params.friction * fn_mag;
            if length(ft) > max_ft {
                ft = normalize(ft) * max_ft;
            }
            
            let force = normal * fn_mag + ft;
            total_force += force;
            total_torque += cross(r, force);
            
            // Rolling friction
            if length(clump_ang_vel) > 1e-6 {
                let roll = -normalize(clump_ang_vel) * (0.01 * fn_mag * radius); // 0.01 placeholder rolling friction
                total_torque += roll;
            }
        }
    }
    
    // Accumulate into buffers
    forces[particle_idx] += vec4<f32>(total_force, 0.0);
    torques[particle_idx] += vec4<f32>(total_torque, 0.0);
}
