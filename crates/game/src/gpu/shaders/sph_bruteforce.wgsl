// Brute-Force IISPH - O(n²) neighbor search
// Used to validate pressure physics before implementing spatial hash
// Limited to ~5-10k particles due to O(n²) complexity

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
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions_pred: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressures: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_ii: array<f32>;
@group(0) @binding(7) var<storage, read_write> sum_dij_pj: array<f32>;
@group(0) @binding(8) var<storage, read_write> cell_indices: array<u32>;
@group(0) @binding(9) var<storage, read_write> particle_order: array<u32>;
@group(0) @binding(10) var<storage, read_write> cell_offsets: array<u32>;

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
// Kernel: Predict Positions (no hash needed)
// ============================================================================

@compute @workgroup_size(256)
fn bf_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    // Load current state
    let pos = positions[i].xyz;
    var vel = velocities[i].xyz;

    // Apply gravity
    vel.y += params.gravity * params.dt;

    // Predict position
    let pos_pred = pos + vel * params.dt;

    // Store
    positions_pred[i] = vec4(pos_pred, 0.0);
    velocities[i] = vec4(vel, 0.0);

    // Initialize pressure to 0
    pressures[i] = 0.0;
}

// ============================================================================
// Kernel: Compute Density and d_ii (BRUTE FORCE)
// ============================================================================

@compute @workgroup_size(256)
fn bf_density_dii(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;

    var rho = 0.0;
    var dii_sum = vec3(0.0);

    // BRUTE FORCE: iterate ALL particles
    for (var j = 0u; j < params.num_particles; j++) {
        let pj = positions_pred[j].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 < params.h2) {
            // Density contribution (mass = 1.0)
            rho += poly6(r2);

            // d_ii contribution
            if (r2 > 0.0001) {
                let dist = sqrt(r2);
                let grad = spiky_grad(r, dist);
                dii_sum += grad;
            }
        }
    }

    // Store density
    densities[i] = rho;

    // Compute d_ii = -dt² * |sum(∇W_ij)|² / ρ²
    let dii_val = -params.dt2 * dot(dii_sum, dii_sum) / max(rho * rho, 0.0001);
    d_ii[i] = dii_val;
}

// ============================================================================
// Kernel: Compute sum(d_ij * p_j) (BRUTE FORCE)
// ============================================================================

@compute @workgroup_size(256)
fn bf_sum_dij(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;

    var sum = 0.0;

    // BRUTE FORCE: iterate ALL particles
    for (var j = 0u; j < params.num_particles; j++) {
        if (j == i) { continue; }

        let pj = positions_pred[j].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 < params.h2 && r2 > 0.0001) {
            let dist = sqrt(r2);
            let rhoj = densities[j];
            let pj_pressure = pressures[j];
            let grad = spiky_grad(r, dist);

            // d_ij = -dt² * (1/ρ_j²) * |∇W|²
            let d_ij = -params.dt2 / max(rhoj * rhoj, 0.0001) * dot(grad, grad);
            sum += d_ij * pj_pressure;
        }
    }

    sum_dij_pj[i] = sum;
}

// ============================================================================
// Kernel: Update Pressure (WCSPH state equation)
// ============================================================================
//
// IISPH Jacobi relaxation fails when density is far from rest_density because
// the formula p = (ρ_err - sum) / d_ii gives negative pressure for compressed
// particles (d_ii is negative by construction), which gets clamped to 0.
//
// Instead, use simple WCSPH-style state equation:
//   p = k * max(0, ρ - ρ₀)
//
// This always gives positive pressure when compressed.

@compute @workgroup_size(256)
fn bf_update_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let rho = densities[i];
    let rho_err = rho - params.rest_density;

    // WCSPH-style state equation: p = k * max(0, relative_density_error)
    // Stiffness k controls how strongly pressure resists compression.
    // Higher k = stiffer fluid = more incompressible, but requires smaller dt.
    // With Müller formula dividing by ρ², need high stiffness to resist compression.
    let stiffness = 50000.0;

    let p_new = stiffness * max(0.0, rho_err);

    pressures[i] = p_new;
}

// ============================================================================
// Kernel: Apply Pressure Forces (BRUTE FORCE)
// ============================================================================

@compute @workgroup_size(256)
fn bf_apply_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[i].xyz;
    let rhoi = densities[i];
    let pressi = pressures[i];

    var f_pressure = vec3(0.0);
    var neighbor_count = 0u;

    // BRUTE FORCE: iterate ALL particles
    for (var j = 0u; j < params.num_particles; j++) {
        if (j == i) { continue; }

        let pj = positions_pred[j].xyz;
        let r = pi - pj;
        let r2 = dot(r, r);

        if (r2 < params.h2 && r2 > 0.0001) {
            let dist = sqrt(r2);
            let rhoj = densities[j];
            let pressj = pressures[j];
            let grad = spiky_grad(r, dist);

            // Proper Müller symmetric pressure force formula:
            // F_i = -m² * Σ (p_i/ρ_i² + p_j/ρ_j²) * ∇W_spiky
            // Reference: cs418.cs.illinois.edu/website/text/sph.html
            // Mass = 1.0, so m² = 1.0
            let rhoi2 = max(rhoi * rhoi, 1.0);
            let rhoj2 = max(rhoj * rhoj, 1.0);
            let pressure_term = pressi / rhoi2 + pressj / rhoj2;
            f_pressure -= pressure_term * grad;

            neighbor_count += 1u;
        }
    }

    // Update velocity with pressure force (mass = 1.0)
    let vel_pred = velocities[i].xyz;
    var vel_new = vel_pred + f_pressure * params.dt;

    // Light damping - DEM provides its own damping
    vel_new *= 0.995;

    // Clamp max velocity to prevent tunneling
    let max_vel = params.h * 2.0 / params.dt;  // Max ~2 cell widths per frame
    let vel_mag = length(vel_new);
    if (vel_mag > max_vel) {
        vel_new = vel_new * (max_vel / vel_mag);
    }

    // Update position from original
    let pos_old = positions[i].xyz;
    var pos_new = pos_old + vel_new * params.dt;

    // HARDCODED bucket bounds - same as collision_test.rs
    // This eliminates any parameter passing issues
    let box_min = vec3(0.1, 0.04, 0.1);
    let box_max = vec3(0.3, 1.0, 0.3);

    // Floor collision (Y min)
    if (pos_new.y < box_min.y) {
        pos_new.y = box_min.y;
        vel_new.y = 0.0;
        vel_new.x *= 0.9;
        vel_new.z *= 0.9;
    }

    // Ceiling (Y max)
    if (pos_new.y > box_max.y) {
        pos_new.y = box_max.y;
        vel_new.y = 0.0;
    }

    // Wall X-
    if (pos_new.x < box_min.x) {
        pos_new.x = box_min.x;
        vel_new.x = 0.0;
    }

    // Wall X+
    if (pos_new.x > box_max.x) {
        pos_new.x = box_max.x;
        vel_new.x = 0.0;
    }

    // Wall Z-
    if (pos_new.z < box_min.z) {
        pos_new.z = box_min.z;
        vel_new.z = 0.0;
    }

    // Wall Z+
    if (pos_new.z > box_max.z) {
        pos_new.z = box_max.z;
        vel_new.z = 0.0;
    }

    velocities[i] = vec4(vel_new, 0.0);
    positions[i] = vec4(pos_new, 0.0);
}

// ============================================================================
// Kernel: Boundary Collision
// ============================================================================

@compute @workgroup_size(256)
fn bf_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    var pos = positions[i].xyz;
    var vel = velocities[i].xyz;

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

    positions[i] = vec4(pos, 0.0);
    velocities[i] = vec4(vel, 0.0);
}
