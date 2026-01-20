// Simple SPH test - gravity + boundary only, no neighbor search
// Used to verify basic particle motion before implementing full IISPH

const PI: f32 = 3.14159265359;

struct SphParams {
    num_particles: u32,
    h: f32,
    h2: f32,
    rest_density: f32,

    dt: f32,
    dt2: f32,
    gravity: f32,
    omega: f32,

    cell_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,

    poly6_coef: f32,
    spiky_grad_coef: f32,
    pressure_iters: u32,
    particle_mass: f32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions_pred: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressures: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_ii: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressure_accel: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> cell_indices: array<u32>;
@group(0) @binding(9) var<storage, read_write> particle_order: array<u32>;
@group(0) @binding(10) var<storage, read_write> cell_offsets: array<u32>;

// Simple integration + boundary kernel
// Combines all steps into one for debugging
@compute @workgroup_size(256)
fn simple_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    // Load state
    var pos = positions[i].xyz;
    var vel = velocities[i].xyz;

    // Apply gravity
    vel.y += params.gravity * params.dt;

    // Integrate position
    pos += vel * params.dt;

    // Domain bounds
    let domain_min = vec3(params.h, params.h, params.h);
    let domain_max = vec3(
        f32(params.grid_size_x) * params.cell_size - params.h,
        f32(params.grid_size_y) * params.cell_size,  // Open top
        f32(params.grid_size_z) * params.cell_size - params.h
    );

    // Floor collision
    if (pos.y < domain_min.y) {
        pos.y = domain_min.y;
        vel.y = -vel.y * 0.3;
        vel.x *= 0.95;
        vel.z *= 0.95;
    }

    // Wall X
    if (pos.x < domain_min.x) {
        pos.x = domain_min.x;
        vel.x = -vel.x * 0.3;
    }
    if (pos.x > domain_max.x) {
        pos.x = domain_max.x;
        vel.x = -vel.x * 0.3;
    }

    // Wall Z
    if (pos.z < domain_min.z) {
        pos.z = domain_min.z;
        vel.z = -vel.z * 0.3;
    }
    if (pos.z > domain_max.z) {
        pos.z = domain_max.z;
        vel.z = -vel.z * 0.3;
    }

    // Store
    positions[i] = vec4(pos, 0.0);
    velocities[i] = vec4(vel, 0.0);
}

// Placeholders for interface compatibility
@compute @workgroup_size(256)
fn predict_and_hash(@builtin(global_invocation_id) gid: vec3<u32>) {}

@compute @workgroup_size(256)
fn build_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {}

@compute @workgroup_size(256)
fn compute_density_dii(@builtin(global_invocation_id) gid: vec3<u32>) {}

@compute @workgroup_size(256)
fn compute_sum_dij(@builtin(global_invocation_id) gid: vec3<u32>) {}

@compute @workgroup_size(256)
fn update_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {}

@compute @workgroup_size(256)
fn apply_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {}

@compute @workgroup_size(256)
fn boundary_collision(@builtin(global_invocation_id) gid: vec3<u32>) {}
