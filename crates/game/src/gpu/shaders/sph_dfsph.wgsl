// DFSPH (Divergence-Free SPH) Implementation
// Based on "Divergence-Free SPH for Incompressible and Viscous Fluids" (Bender & Koschier, 2017)

const PI: f32 = 3.14159265359;
const EPSILON: f32 = 1e-6;

struct DfsphParams {
    // Block 0
    num_particles: u32,
    h: f32,              // Smoothing radius
    h2: f32,             // h^2
    rest_density: f32,   // ρ₀ (typically 1000 for water)

    dt: f32,
    dt2: f32,            // dt^2
    gravity: f32,
    omega: f32,          // Relaxation factor (0.5 typical)
    nu: f32,             // Artificial viscosity coefficient (e.g. 0.01)
    _pad0: f32,

    // Block 2
    cell_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,

    // Block 3
    poly6_coef: f32,
    spiky_grad_coef: f32,
    particle_mass: f32,
    volume: f32,         // V = m / ρ₀
}

// Buffers (all physically sorted, direct access with i)
@group(0) @binding(0) var<uniform> params: DfsphParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions_pred: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> alpha: array<f32>;           
@group(0) @binding(6) var<storage, read_write> density_adv: array<f32>;     
@group(0) @binding(7) var<storage, read_write> pressure_rho2: array<vec4<f32>>;   
@group(0) @binding(8) var<storage, read_write> pressure_accel: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> cell_offsets: array<u32>;          
@group(0) @binding(10) var<storage, read_write> neighbor_counts: array<u32>;
@group(0) @binding(11) var<storage, read_write> neighbor_indices: array<u32>; // capacity = num_particles * 64

// ============================================================================
// SPH Kernel Functions
// ============================================================================

fn poly6(r2: f32) -> f32 {
    if (r2 >= params.h2) { return 0.0; }
    let diff = params.h2 - r2;
    return params.poly6_coef * diff * diff * diff;
}

fn spiky_grad(r: vec3<f32>, dist: f32) -> vec3<f32> {
    if (dist >= params.h || dist < EPSILON) { return vec3(0.0); }
    let diff = params.h - dist;
    // Faster: Avoid normalize(r) by using r / dist
    return (params.spiky_grad_coef * diff * diff / dist) * r;
}

// ============================================================================
// Spatial Hash Functions
// ============================================================================

fn cell_coord(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p / params.cell_size));
}

fn cell_hash(c: vec3<i32>) -> u32 {
    let cx = clamp(c.x, 0, i32(params.grid_size_x) - 1);
    let cy = clamp(c.y, 0, i32(params.grid_size_y) - 1);
    let cz = clamp(c.z, 0, i32(params.grid_size_z) - 1);
    return u32(cx) + u32(cy) * params.grid_size_x + u32(cz) * params.grid_size_x * params.grid_size_y;
}

fn is_valid_cell(c: vec3<i32>) -> bool {
    return c.x >= 0 && c.x < i32(params.grid_size_x) &&
           c.y >= 0 && c.y < i32(params.grid_size_y) &&
           c.z >= 0 && c.z < i32(params.grid_size_z);
}

// ============================================================================
// Kernel: Predict Positions + Apply Gravity + Hash
// ============================================================================

@compute @workgroup_size(256)
fn predict_and_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    // Apply gravity to velocity
    var vel = velocities[i].xyz;
    vel.y -= params.gravity * params.dt;
    velocities[i] = vec4(vel, 0.0);

    // Predict position
    let pos = positions[i].xyz;
    let pos_pred = pos + vel * params.dt;
    positions_pred[i] = vec4(pos_pred, 0.0);
}

// ============================================================================
// Kernel: Build Neighbor List
// ============================================================================

const MAX_NEIGHBORS: u32 = 24u;

@compute @workgroup_size(256)
fn build_neighbor_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let cell = cell_coord(pi);
    var count = 0u;

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell + vec3(dx, dy, dz);
                if (!is_valid_cell(neighbor_cell)) { continue; }

                let hash = cell_hash(neighbor_cell);
                let start = cell_offsets[hash];
                let end = cell_offsets[hash + 1u];

                for (var k = start; k < end; k++) {
                    if (k == i) { continue; }
                    let pj = positions_pred[k].xyz;
                    let r = pi - pj;
                    let r2 = dot(r, r);

                    if (r2 < params.h2) {
                        if (count < MAX_NEIGHBORS) {
                            neighbor_indices[i * MAX_NEIGHBORS + count] = k;
                            count++;
                        }
                    }
                }
            }
        }
    }
    neighbor_counts[i] = count;
}

// ============================================================================
// Kernel: Compute Density + XSPH Viscosity
// ============================================================================

@compute @workgroup_size(256)
fn compute_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let vi = velocities[i].xyz;
    var rho = params.particle_mass * poly6(0.0); // Self-contribution
    var v_visc = vec3(0.0);

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let vj = velocities[k].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        let w = poly6(r2);
        rho += params.particle_mass * w;
        
        // Artificial Viscosity (simple damping)
        v_visc += params.volume * (vj - vi) * w;
    }

    densities[i] = rho;
    
    // Apply VERY small viscosity to velocity (XSPH approximation)
    let nu = 0.01; 
    velocities[i] = vec4(vi + nu * v_visc, 0.0);
}

// ============================================================================
// Kernel: Compute Alpha Factor
// ============================================================================

@compute @workgroup_size(256)
fn compute_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    var sum_grad = vec3(0.0);
    var sum_grad_sq = 0.0;

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let r = pi - pj;
        let dist = length(r);

        if (dist > EPSILON) {
            let grad_j = params.particle_mass * spiky_grad(r, dist);
            sum_grad += grad_j;
            sum_grad_sq += dot(grad_j, grad_j);
        }
    }

    let denominator = dot(sum_grad, sum_grad) + sum_grad_sq;
    if (denominator > 1e-12) {
        alpha[i] = 1.0 / denominator;
    } else {
        alpha[i] = 0.0;
    }
}

// ============================================================================
// Kernel: Compute Divergence Source Term
// ============================================================================

@compute @workgroup_size(256)
fn compute_divergence_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let vi = velocities[i].xyz;
    var div = 0.0;

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let vj = velocities[k].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 > EPSILON) {
            let dist = sqrt(r2);
            let grad = spiky_grad(r, dist);
            div += params.particle_mass * dot(vi - vj, grad);
        }
    }

    // Divergence source s = drho/dt
    density_adv[i] = div; 
}

// ============================================================================
// Kernel: Update Divergence Pressure (Stiffness)
// ============================================================================

@compute @workgroup_size(256)
fn update_divergence_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let ai = pressure_accel[i].xyz;
    var aij_pj = 0.0;

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let aj = pressure_accel[k].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 > EPSILON) {
            let dist = sqrt(r2);
            let grad = spiky_grad(r, dist);
            aij_pj += params.particle_mass * dot(aj - ai, grad);
        }
    }

    let residual = density_adv[i] - params.dt * aij_pj; 
    let dp = params.omega * alpha[i] / params.dt * residual;
    
    pressure_rho2[i].x += dp;
}

// ============================================================================
// Kernel: Compute Density Advection Source Term
// ============================================================================

@compute @workgroup_size(256)
fn compute_density_adv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let vi = velocities[i].xyz;
    var div = 0.0;

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let vj = velocities[k].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 > EPSILON) {
            let dist = sqrt(r2);
            let grad = spiky_grad(r, dist);
            div += params.particle_mass * dot(vi - vj, grad);
        }
    }

    density_adv[i] = densities[i] - params.rest_density + params.dt * div;
}

// ============================================================================
// Kernel: Compute Pressure Acceleration
// ============================================================================

@compute @workgroup_size(256)
fn compute_pressure_accel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let ki = pressure_rho2[i].x;
    var accel = vec3(0.0);

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let kj = pressure_rho2[k].x;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 > EPSILON) {
            let dist = sqrt(r2);
            let grad = spiky_grad(r, dist);
            accel += params.particle_mass * (ki + kj) * grad;
        }
    }

    var a = -accel;
    let max_a = 5000.0;
    if (length(a) > max_a) {
        a = normalize(a) * max_a;
    }
    pressure_accel[i] = vec4(a, 0.0);
}

// ============================================================================
// Kernel: Update Constant Density Pressure
// ============================================================================

@compute @workgroup_size(256)
fn update_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let ai = pressure_accel[i].xyz;
    var aij_pj = 0.0;

    let count = neighbor_counts[i];
    for (var j = 0u; j < count; j++) {
        let k = neighbor_indices[i * MAX_NEIGHBORS + j];
        let pj = positions_pred[k].xyz;
        let aj = pressure_accel[k].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 > EPSILON) {
            let dist = sqrt(r2);
            let grad = spiky_grad(r, dist);
            aij_pj += params.particle_mass * dot(aj - ai, grad);
        }
    }

    let residual = max(density_adv[i] - params.dt2 * aij_pj, 0.0); 
    let dp = params.omega * alpha[i] / params.dt2 * residual;
    
    pressure_rho2[i].x = max(pressure_rho2[i].x + dp, 0.0);
}

// ============================================================================
// Kernel: Integrate Velocity + Position
// ============================================================================

@compute @workgroup_size(256)
fn integrate_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    var vel = velocities[i].xyz;
    let accel = pressure_accel[i].xyz;
    
    vel += params.dt * accel;
    
    // CFL-based speed limit: max_vel < h / dt for stability
    let max_vel = params.h * 0.5 / params.dt;  // 0.5*h per timestep for safety
    let speed = length(vel);
    if (speed > max_vel) {
        vel = vel * (max_vel / speed);
    }
    
    velocities[i] = vec4(vel, 0.0);
}

@compute @workgroup_size(256)
fn integrate_position(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let vel = velocities[i].xyz;
    var pos = positions[i].xyz;
    pos += params.dt * vel;
    positions[i] = vec4(pos, 0.0);
}

// ============================================================================
// Kernel: Boundary Collision
// ============================================================================

@compute @workgroup_size(256)
fn boundary_collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    var pos = positions[i].xyz;
    var vel = velocities[i].xyz;

    let domain_min = vec3(0.02, 0.02, 0.02); // Small inset to avoid exact zero
    let domain_max = vec3(
        f32(params.grid_size_x) * params.cell_size - 0.02,
        f32(params.grid_size_y) * params.cell_size - 0.02,
        f32(params.grid_size_z) * params.cell_size - 0.02
    );

    // X
    if (pos.x < domain_min.x) {
        pos.x = domain_min.x;
        vel.x = -vel.x * 0.3;
    }
    if (pos.x > domain_max.x) {
        pos.x = domain_max.x;
        vel.x = -vel.x * 0.3;
    }
    
    // Y
    if (pos.y < domain_min.y) {
        pos.y = domain_min.y;
        vel.y = -vel.y * 0.3;
    }
    if (pos.y > domain_max.y) {
        pos.y = domain_max.y;
        vel.y = -vel.y * 0.3;
    }
    
    // Z
    if (pos.z < domain_min.z) {
        pos.z = domain_min.z;
        vel.z = -vel.z * 0.3;
    }
    if (pos.z > domain_max.z) {
        pos.z = domain_max.z;
        vel.z = -vel.z * 0.3;
    }

    positions[i] = vec4(pos, 0.0);
    velocities[i] = vec4(vel, 0.0);
}
