// DEM Forces Shader - Compute spring-damper contacts and integrate
//
// For each sediment particle:
// 1. Apply gravity (buoyancy-adjusted if in water)
// 2. Integrate position
// 3. Iterate 3x3 bin neighborhood for particle-particle collisions
// 4. Apply spring-damper normal force + Coulomb friction
// 5. Handle floor/SDF collision with friction
//
// Uses sorted_indices from spatial hash for O(n) collision detection.

struct DemParams {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
    dt: f32,
    gravity: f32,
    contact_stiffness: f32,
    damping_ratio: f32,
    friction_coeff: f32,
    velocity_damping: f32,
    sdf_width: u32,
    sdf_height: u32,
    water_level: f32, // -1.0 for no water, else Y level
    _pad: f32,
}

// Material densities (for buoyancy calculation)
// 0=Water, 1=Mud, 2=Sand, 3=Magnetite, 4=Gold, 5=Gravel
const DENSITIES: array<f32, 6> = array<f32, 6>(1.0, 1.8, 2.65, 5.15, 19.3, 2.5);
const FRICTIONS: array<f32, 6> = array<f32, 6>(0.0, 0.4, 0.6, 0.55, 0.45, 0.65);

@group(0) @binding(0) var<uniform> params: DemParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> radii: array<f32>;
@group(0) @binding(4) var<storage, read> materials: array<u32>;
@group(0) @binding(5) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read> sdf_data: array<f32>;

// Helper to sample SDF at clamped grid coordinates
fn get_sdf_at(i: i32, j: i32) -> f32 {
    let w = i32(params.sdf_width);
    let h = i32(params.sdf_height);
    let ci = clamp(i, 0, w - 1);
    let cj = clamp(j, 0, h - 1);
    return sdf_data[u32(cj * w + ci)];
}

// Sample SDF with bilinear interpolation
fn sample_sdf(pos: vec2<f32>) -> f32 {
    let cell_size = params.cell_size;
    let fx = pos.x / cell_size - 0.5;
    let fy = pos.y / cell_size - 0.5;

    let i0 = i32(floor(fx));
    let j0 = i32(floor(fy));
    let i1 = i0 + 1;
    let j1 = j0 + 1;

    let tx = fx - f32(i0);
    let ty = fy - f32(j0);

    let s00 = get_sdf_at(i0, j0);
    let s10 = get_sdf_at(i1, j0);
    let s01 = get_sdf_at(i0, j1);
    let s11 = get_sdf_at(i1, j1);

    return mix(mix(s00, s10, tx), mix(s01, s11, tx), ty);
}

fn sdf_gradient(pos: vec2<f32>) -> vec2<f32> {
    let eps = params.cell_size * 0.5;
    let dx = sample_sdf(pos + vec2(eps, 0.0)) - sample_sdf(pos - vec2(eps, 0.0));
    let dy = sample_sdf(pos + vec2(0.0, eps)) - sample_sdf(pos - vec2(0.0, eps));
    let len = sqrt(dx * dx + dy * dy);
    if (len < 0.0001) {
        return vec2(0.0, -1.0);
    }
    return vec2(dx, dy) / len;
}

fn compute_mass(material: u32, radius: f32) -> f32 {
    let density = DENSITIES[min(material, 5u)];
    let volume = (4.0 / 3.0) * 3.14159265 * radius * radius * radius;
    return density * volume;
}

fn effective_gravity(material: u32, in_water: bool) -> f32 {
    if (!in_water) {
        return params.gravity;
    }
    let density = DENSITIES[min(material, 5u)];
    // Buoyancy reduces effective weight: (rho_p - rho_w) / rho_p
    let buoyancy_factor = (density - 1.0) / density;
    return params.gravity * buoyancy_factor;
}

@compute @workgroup_size(256)
fn dem_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let idx = id.x;
    var pos = positions[idx];
    var vel = velocities[idx];
    let radius = radii[idx];
    let material = materials[idx];
    let mass = compute_mass(material, radius);
    let mu = FRICTIONS[min(material, 5u)];

    // Check if in water
    let in_water = params.water_level >= 0.0 && pos.y > params.water_level;

    // Apply gravity
    let g_eff = effective_gravity(material, in_water);
    vel.y += g_eff * params.dt;

    // Apply velocity damping (less in water)
    var damping = params.velocity_damping;
    if (in_water) {
        damping = 0.995;
    }
    vel *= damping;

    // Integrate position (semi-implicit Euler)
    pos += vel * params.dt;

    // Find bin for this particle
    let gi = i32(pos.x / params.cell_size);
    let gj = i32(pos.y / params.cell_size);
    let grid_w = i32(params.grid_width);
    let grid_h = i32(params.grid_height);

    // Accumulate position correction and velocity change
    var pos_correction = vec2<f32>(0.0, 0.0);
    var vel_change = vec2<f32>(0.0, 0.0);

    // DEM parameters
    let k_n = params.contact_stiffness;
    let c_ratio = params.damping_ratio;

    // Iterate 3x3 neighborhood
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let ni = gi + di;
            let nj = gj + dj;

            if (ni < 0 || ni >= grid_w || nj < 0 || nj >= grid_h) {
                continue;
            }

            let bin_idx = u32(nj * grid_w + ni);
            let bin_start = bin_offsets[bin_idx];
            let bin_end = bin_offsets[bin_idx + 1u];

            for (var k = bin_start; k < bin_end; k++) {
                let j_idx = sorted_indices[k];

                // Skip self-collision only. On GPU each thread writes only to its own
                // particle, so we must process ALL neighbors to get symmetric forces.
                // (CPU DEM can use idx < j_idx optimization since it updates both particles)
                if (j_idx == idx) {
                    continue;
                }

                let pos_j = positions[j_idx];
                let vel_j = velocities[j_idx];
                let radius_j = radii[j_idx];
                let material_j = materials[j_idx];
                let mass_j = compute_mass(material_j, radius_j);

                // Contact distance
                let contact_dist = radius + radius_j;
                let diff = pos - pos_j;
                let dist_sq = dot(diff, diff);

                if (dist_sq >= contact_dist * contact_dist || dist_sq < 0.0001) {
                    continue;
                }

                let dist = sqrt(dist_sq);
                let overlap = contact_dist - dist;

                if (overlap > 0.0) {
                    let normal = diff / dist;
                    let tangent = vec2(-normal.y, normal.x);

                    // Relative velocity
                    let rel_vel = vel - vel_j;
                    let v_n = dot(rel_vel, normal);
                    let v_t = dot(rel_vel, tangent);

                    // Effective mass
                    let m_eff = (mass * mass_j) / (mass + mass_j);

                    // Spring-damper normal force
                    let c_n = c_ratio * 2.0 * sqrt(k_n * m_eff);
                    let f_n = max(k_n * overlap - c_n * v_n, 0.0);

                    // Coulomb friction
                    let mu_avg = 0.5 * (mu + FRICTIONS[min(material_j, 5u)]);
                    let f_t_max = mu_avg * f_n;
                    var f_t = 0.0;
                    if (abs(v_t) > 0.001) {
                        f_t = -min(abs(v_t) * k_n * 0.4, f_t_max) * sign(v_t);
                    }

                    // Position correction (mass-weighted)
                    let push = overlap * 0.5;
                    let ratio_i = mass_j / (mass + mass_j);
                    pos_correction += normal * push * ratio_i;

                    // Velocity impulse
                    let impulse = (normal * f_n + tangent * f_t) * params.dt;
                    vel_change += impulse / mass;
                }
            }
        }
    }

    // Apply accumulated corrections
    pos += pos_correction;
    vel += vel_change;

    // Floor/SDF collision
    let sdf = sample_sdf(pos);
    if (sdf < radius) {
        let grad = sdf_gradient(pos);
        let push_dist = radius - sdf + 0.1;
        pos += grad * push_dist;

        // Zero velocity into floor
        let normal_vel = dot(vel, grad);
        if (normal_vel < 0.0) {
            vel -= grad * normal_vel;
        }

        // Floor friction
        let tangent = vec2(-grad.y, grad.x);
        let tangent_vel = dot(vel, tangent);
        vel -= tangent * tangent_vel * mu;
    }

    // Clamp very slow velocities to zero
    if (dot(vel, vel) < 0.25) {
        vel = vec2(0.0, 0.0);
    }

    // Write back
    positions[idx] = pos;
    velocities[idx] = vel;
}
