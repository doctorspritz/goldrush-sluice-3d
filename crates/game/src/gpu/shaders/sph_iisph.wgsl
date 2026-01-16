// IISPH (Implicit Incompressible SPH) Implementation
// Based on "Implicit Incompressible SPH" (Ihmsen et al., 2014)
//
// Pipeline:
// 1. predict_and_hash - Predict positions, compute cell hash
// 2. sort (external bitonic sort)
// 3. build_offsets - Build cell offset table
// 4. compute_density_dii - Compute density and IISPH diagonal
// 5. LOOP: compute_sum_dij + update_pressure (Jacobi iteration)
// 6. apply_pressure - Apply pressure forces, integrate
// 7. boundary_collision - SDF collision response

const PI: f32 = 3.14159265359;
const INVALID_CELL: u32 = 0xFFFFFFFFu;

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
@group(0) @binding(10) var<storage, read_write> cell_offsets: array<atomic<u32>>;

// ============================================================================
// SPH Kernel Functions
// ============================================================================

// Poly6 kernel - used for density estimation
fn poly6(r2: f32) -> f32 {
    if (r2 >= params.h2) { return 0.0; }
    let diff = params.h2 - r2;
    return params.poly6_coef * diff * diff * diff;
}

// Spiky kernel gradient - used for pressure forces
fn spiky_grad(r: vec3<f32>, dist: f32) -> vec3<f32> {
    if (dist >= params.h || dist < 0.0001) { return vec3(0.0); }
    let diff = params.h - dist;
    return params.spiky_grad_coef * diff * diff * normalize(r);
}

// ============================================================================
// Spatial Hash Functions
// ============================================================================

fn cell_coord(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p / params.cell_size));
}

fn cell_hash(c: vec3<i32>) -> u32 {
    // Clamp to grid bounds
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
// Kernel: Predict and Hash
// ============================================================================

@compute @workgroup_size(256)
fn predict_and_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    // Load current state
    let pos = positions[i].xyz;
    var vel = velocities[i].xyz;

    // Apply damping
    vel *= 0.99;

    // Apply gravity
    vel += vec3(0.0, params.gravity, 0.0) * params.dt;

    // Predict position
    let pos_pred = pos + vel * params.dt;

    // Store predicted position and velocity
    positions_pred[i] = vec4(pos_pred, 0.0);
    velocities[i] = vec4(vel, 0.0);

    // Compute cell hash
    let cell = cell_coord(pos_pred);
    let hash = cell_hash(cell);

    cell_indices[i] = hash;
    particle_order[i] = i;

    // Initialize pressure to 0 for first iteration
    pressures[i] = 0.0;
}

// ============================================================================
// Kernel: Build Offsets (after sorting)
// ============================================================================

@compute @workgroup_size(256)
fn build_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let sorted_idx = particle_order[i];
    let my_cell = cell_indices[sorted_idx];

    // Each thread that is the first of its cell is responsible for filling
    // the gap between the previous cell and its own.
    if (i == 0u) {
        // Fill from 0 up to my_cell
        for (var c = 0u; c <= my_cell; c++) {
            atomicStore(&cell_offsets[c], 0u);
        }
    } else {
        let prev_sorted_idx = particle_order[i - 1u];
        let prev_cell = cell_indices[prev_sorted_idx];
        if (prev_cell != my_cell) {
            // Fill gaps between prev_cell+1 and my_cell
            for (var c = prev_cell + 1u; c <= my_cell; c++) {
                atomicStore(&cell_offsets[c], i);
            }
        }
    }

    // Last thread fills from its cell to the end
    if (i == params.num_particles - 1u) {
        let num_cells = params.grid_size_x * params.grid_size_y * params.grid_size_z;
        for (var c = my_cell + 1u; c <= num_cells; c++) {
            atomicStore(&cell_offsets[c], params.num_particles);
        }
    }
}

// ============================================================================
// Kernel: Compute Density and a_ii (IISPH matrix diagonal)
// ============================================================================

@compute @workgroup_size(256)
fn compute_density_dii(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let pi = positions_pred[idx].xyz;

    var rho = 0.0;
    var dii_sum = vec3(0.0);
    var dii_sum_sq = 0.0;

    let cell = cell_coord(pi);
 
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell + vec3(dx, dy, dz);
                if (!is_valid_cell(neighbor_cell)) { continue; }
 
                let hash = cell_hash(neighbor_cell);
                let start = atomicLoad(&cell_offsets[hash]);
                let end = atomicLoad(&cell_offsets[hash + 1u]);
 
                for (var k = start; k < end; k++) {
                    let j = particle_order[k];
                    let pj = positions_pred[j].xyz;
                    let r = pi - pj;
                    let r2 = dot(r, r);
 
                    if (r2 < params.h2) {
                        // Density contribution
                        rho += params.particle_mass * poly6(r2);
 
                        // d_ii contribution
                        if (r2 > 0.0001) {
                            let dist = sqrt(r2);
                            let grad = spiky_grad(r, dist);
                            let m_grad = params.particle_mass * grad;
                            dii_sum += m_grad;
                            dii_sum_sq += dot(m_grad, m_grad);
                        }
                    }
                }
            }
        }
    }
    densities[idx] = rho;

    // a_ii = -dt² * ( |sum(m_j ∇W_ij)|² + sum(|m_j ∇W_ij|²) ) / ρ²
    let safe_rho = max(rho, params.rest_density * 0.5);
    let a_ii = -params.dt2 * (dot(dii_sum, dii_sum) + dii_sum_sq) / (safe_rho * safe_rho);
    d_ii[idx] = a_ii;
}

// ============================================================================
// Kernel: Compute Pressure Acceleration
// f_i = -sum m_j * (p_i/rho_i² + p_j/rho_j²) * ∇W_ij
// ============================================================================

@compute @workgroup_size(256)
fn compute_sum_dij(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let pi = positions_pred[idx].xyz;
    let rhoi = densities[idx];
    let pi_pressure = pressures[idx];

    var accel = vec3(0.0);

    let cell = cell_coord(pi);

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell + vec3(dx, dy, dz);
                if (!is_valid_cell(neighbor_cell)) { continue; }

                let hash = cell_hash(neighbor_cell);
                let start = atomicLoad(&cell_offsets[hash]);
                let end = atomicLoad(&cell_offsets[hash + 1u]);

                for (var k = start; k < end; k++) {
                    let j = particle_order[k];
                    let pj = positions_pred[j].xyz;
                    let r = pi - pj;
                    let r2 = dot(r, r);

                    if (r2 < params.h2 && r2 > 0.0001) {
                        let dist = sqrt(r2);
                        let rhoj = densities[j];
                        let pj_pressure = pressures[j];
                        let grad = spiky_grad(r, dist);

                        let safe_rhoi = max(rhoi, params.rest_density * 0.5);
                        let safe_rhoj = max(rhoj, params.rest_density * 0.5);

                        // f_i = -m_j * (p_i/rho_i² + p_j/rho_j²) * ∇W_ij
                        // accel = f_i / m_i = f_i / m (as m_i = m_j = m)
                        // accel = -m_j/m_i * ... * grad
                        accel -= params.particle_mass * (pi_pressure / (safe_rhoi * safe_rhoi) + pj_pressure / (safe_rhoj * safe_rhoj)) * grad;
                    }
                }
            }
        }
    }

    pressure_accel[idx] = vec4(accel, 0.0);
}

// ============================================================================
// Kernel: Update Pressure (Jacobi relaxation)
// ============================================================================

@compute @workgroup_size(256)
fn update_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let pi = positions_pred[idx].xyz;
    let rho_i = densities[idx];
    let p_i = pressures[idx];
    let a_ii = d_ii[idx];
    let accel_i = pressure_accel[idx].xyz;

    // Compute current density change due to pressure:
    // d_rho/dt = sum m_j * (a_i - a_j) . ∇W_ij
    var d_rho_pressure = 0.0;

    let cell = cell_coord(pi);
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell + vec3(dx, dy, dz);
                if (!is_valid_cell(neighbor_cell)) { continue; }

                let hash = cell_hash(neighbor_cell);
                let start = atomicLoad(&cell_offsets[hash]);
                let end = atomicLoad(&cell_offsets[hash + 1u]);

                for (var k = start; k < end; k++) {
                    let j = particle_order[k];
                    let pj = positions_pred[j].xyz;
                    let r = pi - pj;
                    let r2 = dot(r, r);

                    if (r2 < params.h2 && r2 > 0.0001) {
                        let dist = sqrt(r2);
                        let grad = spiky_grad(r, dist);
                        let accel_j = pressure_accel[j].xyz;
                        
                        d_rho_pressure -= params.particle_mass * dot(accel_i - accel_j, grad);
                    }
                }
            }
        }
    }

    // Jacobi relaxation for IISPH:
    // p_new = p_i + omega * (rho_target - rho_predicted) / a_ii
    // where rho_predicted = rho_adv + dt² * d_rho_pressure
    let rho_predicted = rho_i + params.dt2 * d_rho_pressure;
    let rho_target = params.rest_density;
    
    var p_new = p_i;
    if (abs(a_ii) > 1e-9) {
        // The instruction's formula for correction and p_new is slightly different from the original comment.
        // It seems to be a specific IISPH pressure update formula.
        // Assuming rho_adv is rho_i (current density before pressure correction)
        let correction = (params.rest_density - rho_i - params.dt2 * d_rho_pressure) / a_ii;
        p_new = (1.0 - params.omega) * p_i + params.omega * correction;
    } else {
        p_new = 0.0;
    }
    pressures[idx] = max(p_new, 0.0);
}

// ============================================================================
// Kernel: Apply Pressure Forces and Integrate
// ============================================================================

@compute @workgroup_size(256)
fn apply_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let pi = positions_pred[idx].xyz;
    let rhoi = densities[idx];
    let pressi = pressures[idx];

    var f_pressure = vec3(0.0);

    let cell = cell_coord(pi);

    // Compute pressure force from neighbors
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell + vec3(dx, dy, dz);
                if (!is_valid_cell(neighbor_cell)) { continue; }

                let hash = cell_hash(neighbor_cell);
                let start = atomicLoad(&cell_offsets[hash]);
                let end = atomicLoad(&cell_offsets[hash + 1u]);

                for (var k = start; k < end; k++) {
                    let j = particle_order[k];
                    if (j == idx) { continue; }

                    let pj = positions_pred[j].xyz;
                    let r = pi - pj;
                    let r2 = dot(r, r);

                    if (r2 < params.h2 && r2 > 0.0001) {
                        let dist = sqrt(r2);
                        let rhoj = densities[j];
                        let pressj = pressures[j];
                        let grad = spiky_grad(r, dist);

                        // Symmetric pressure force with density safeguards
                        let safe_rhoi = max(rhoi, params.rest_density * 0.5);
                        let safe_rhoj = max(rhoj, params.rest_density * 0.5);
                        
                        let pressure_term = pressi / (safe_rhoi * safe_rhoi) +
                                           pressj / (safe_rhoj * safe_rhoj);
                        
                        let force = params.particle_mass * pressure_term * grad;
                        f_pressure -= force;
                    }
                }
            }
        }
    }

    // Stability: Clamp total pressure acceleration
    let accel_limit = 5000.0;
    if (length(f_pressure) > accel_limit) {
        f_pressure = normalize(f_pressure) * accel_limit;
    }

    // Update velocity with pressure force
    let vel_pred = velocities[idx].xyz;
    let vel_new = vel_pred + f_pressure * params.dt;

    // Update position
    let pos_old = positions[idx].xyz;
    let pos_new = pos_old + vel_new * params.dt;

    velocities[idx] = vec4(vel_new, 0.0);
    positions[idx] = vec4(pos_new, 0.0);
}

// ============================================================================
// Kernel: Boundary Collision (Box SDF for bucket test)
// ============================================================================

@compute @workgroup_size(256)
fn boundary_collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    var pos = positions[i].xyz;
    var vel = velocities[i].xyz;

    // Simple box boundary (bucket)
    // Domain: [0, grid_size * cell_size] in each dimension
    // Domain: [0, grid_size * cell_size] in each dimension
    let domain_min = vec3(0.0, 0.0, 0.0);
    let domain_max = vec3(
        f32(params.grid_size_x) * params.cell_size - params.h,
        f32(params.grid_size_y) * params.cell_size - params.h,  // Open top
        f32(params.grid_size_z) * params.cell_size - params.h
    );

    // Floor collision
    if (pos.y < domain_min.y) {
        pos.y = domain_min.y;
        vel.y = -vel.y * 0.3;  // Bounce with damping
        vel.x *= 0.95;  // Friction
        vel.z *= 0.95;
    }

    // Wall collisions (X)
    if (pos.x < domain_min.x) {
        pos.x = domain_min.x;
        vel.x = -vel.x * 0.3;
    }
    if (pos.x > domain_max.x) {
        pos.x = domain_max.x;
        vel.x = -vel.x * 0.3;
    }

    // Wall collisions (Z)
    if (pos.z < domain_min.z) {
        pos.z = domain_min.z;
        vel.z = -vel.z * 0.3;
    }
    if (pos.z > domain_max.z) {
        pos.z = domain_max.z;
        vel.z = -vel.z * 0.3;
    }

    // No ceiling - open top for bucket test

    positions[i] = vec4(pos, 0.0);
    velocities[i] = vec4(vel, 0.0);
}

// ============================================================================
// Bitonic Sort Kernels
// ============================================================================

// Sort key: (cell_index, particle_index) packed or just cell_index
// We sort particle_order array based on cell_indices

@compute @workgroup_size(256)
fn sort_local(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Local bitonic sort within workgroup - placeholder
    // Will use shared memory optimization later
}

@compute @workgroup_size(256)
fn sort_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Global bitonic merge - placeholder
    // Simple comparison swap for now
}
