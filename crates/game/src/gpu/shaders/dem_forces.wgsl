// DEM Forces Shader - Position-based granular simulation
//
// Simple algorithm for sand-like behavior:
// 1. Apply gravity
// 2. Integrate position
// 3. Push overlapping particles apart (no springs, just separation)
// 4. Handle floor collision
// 5. Damp velocity for particles in contact
//
// This avoids the oscillations of spring-damper models.

struct DemParams {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
    dt: f32,
    gravity: f32,
    contact_stiffness: f32,  // Unused in PBD approach, kept for compatibility
    damping_ratio: f32,      // Used for contact damping
    friction_coeff: f32,
    velocity_damping: f32,
    sdf_width: u32,
    sdf_height: u32,
    water_level: f32,
    iteration: u32,  // 0 = first pass (apply gravity), 1+ = constraint-only passes
}

// Material densities (for buoyancy calculation)
const DENSITIES: array<f32, 6> = array<f32, 6>(1.0, 1.8, 2.65, 5.15, 19.3, 2.5);
const FRICTIONS: array<f32, 6> = array<f32, 6>(0.0, 0.4, 0.6, 0.55, 0.45, 0.65);

// Sleep system constants
const SLEEP_THRESHOLD: u32 = 10u;   // Frames of jitter before sleeping
const JITTER_SPEED_SQ: f32 = 4.0;   // Below this = jitter (~2 px/s)
const WAKE_SPEED_SQ: f32 = 200.0;   // Above this = wake neighbors (~14 px/s)

@group(0) @binding(0) var<uniform> params: DemParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> radii: array<f32>;
@group(0) @binding(4) var<storage, read> materials: array<u32>;
@group(0) @binding(5) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read> sdf_data: array<f32>;
@group(0) @binding(8) var<storage, read_write> sleep_counters: array<u32>;

fn get_sdf_at(i: i32, j: i32) -> f32 {
    let w = i32(params.sdf_width);
    let h = i32(params.sdf_height);
    let ci = clamp(i, 0, w - 1);
    let cj = clamp(j, 0, h - 1);
    return sdf_data[u32(cj * w + ci)];
}

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

fn effective_gravity(material: u32, in_water: bool) -> f32 {
    if (!in_water) {
        return params.gravity;
    }
    let density = DENSITIES[min(material, 5u)];
    let buoyancy_factor = (density - 1.0) / density;
    return params.gravity * buoyancy_factor;
}

// Hash function for surface roughness simulation
// Returns value in range [-1, 1]
fn hash_noise(seed: u32) -> f32 {
    var h = seed;
    h = h ^ (h >> 16u);
    h = h * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFu) / 32767.5 - 1.0;
}

@compute @workgroup_size(256)
fn dem_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let idx = id.x;
    let old_pos = positions[idx];
    var pos = old_pos;
    var vel = velocities[idx];
    let radius = radii[idx];
    let material = materials[idx];
    let mu = FRICTIONS[min(material, 5u)];

    // Sleep system: read current sleep counter
    var sleep_counter = sleep_counters[idx];
    let is_sleeping = sleep_counter >= SLEEP_THRESHOLD;

    // Check if particle was sleeping (low velocity = at rest)
    let was_sleeping = dot(vel, vel) < JITTER_SPEED_SQ;

    // Check if in water
    let in_water = params.water_level >= 0.0 && pos.y > params.water_level;

    // 1. Apply gravity and integrate (only on first iteration)
    if (params.iteration == 0u) {
        let g_eff = effective_gravity(material, in_water);
        vel.y += g_eff * params.dt;
        pos += vel * params.dt;
    }

    // 2. Find bin for collision detection
    let gi = i32(pos.x / params.cell_size);
    let gj = i32(pos.y / params.cell_size);
    let grid_w = i32(params.grid_width);
    let grid_h = i32(params.grid_height);

    // Track contact types separately
    var floor_contact = false;
    var particle_contact_count = 0u;
    var has_support_below = false;  // True if contacting a particle below us
    var supported_contacts = 0u;    // Count of contacts with supported particles
    var should_wake = false;        // True if hit by fast-moving particle

    // JACOBI FIX: Accumulate corrections, then average at the end
    // See: Macklin et al. "Position Based Simulation Methods" (EG 2015)
    // Formula: Δx = (ω/n) * Σ(corrections), where n = constraint count, ω = SOR factor
    var accumulated_correction = vec2<f32>(0.0, 0.0);
    var accumulated_vel_correction = vec2<f32>(0.0, 0.0);

    // 4. Particle-particle collision (position-based, no springs)
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
                if (j_idx == idx) {
                    continue;
                }

                let pos_j = positions[j_idx];
                let radius_j = radii[j_idx];
                let material_j = materials[j_idx];

                let diff = pos - pos_j;
                let dist_sq = dot(diff, diff);
                let contact_dist = radius + radius_j;

                if (dist_sq < contact_dist * contact_dist && dist_sq > 0.0001) {
                    let dist = sqrt(dist_sq);
                    let overlap = contact_dist - dist;
                    var normal = diff / dist;

                    // Skip tiny overlaps for sleeping particles (prevents creep)
                    let overlap_threshold = select(0.0, radius * 0.1, was_sleeping);
                    if (overlap < overlap_threshold) {
                        continue;
                    }

                    // Surface roughness: add small random perturbation to normal
                    // This prevents perfect vertical stacking and enables angle of repose
                    let roughness_seed = idx * 31u + j_idx * 17u + params.iteration;
                    let roughness = hash_noise(roughness_seed) * 0.15;  // ~8 degree max deviation
                    let perp = vec2(-normal.y, normal.x);
                    normal = normalize(normal + perp * roughness);

                    // Mass-weighted separation: heavy particles move less
                    let density_i = DENSITIES[min(material, 5u)];
                    let density_j = DENSITIES[min(material_j, 5u)];
                    let mass_i = density_i * radius * radius;
                    let mass_j = density_j * radius_j * radius_j;

                    // Check if j is part of a settled pile (stationary = part of stable structure)
                    let vel_j = velocities[j_idx];
                    let j_speed_sq = dot(vel_j, vel_j);
                    let j_is_stationary = j_speed_sq < 25.0;  // ~5 px/s

                    // Wake detection: if neighbor is moving fast, wake us up
                    if (j_speed_sq > WAKE_SPEED_SQ) {
                        should_wake = true;
                    }

                    // Support propagation: check if neighbor BELOW us is truly supported
                    let j_sleep = sleep_counters[j_idx];
                    let j_is_below = normal.y > 0.3;  // neighbor is below if normal points up
                    if (j_is_below && j_sleep >= SLEEP_THRESHOLD) {
                        supported_contacts += 1u;
                    }

                    // Stationary particles resist displacement but can still be pushed
                    // 3x mass = moderate resistance, allows pile restructuring
                    let j_mass_multiplier = select(1.0, 3.0, j_is_stationary);
                    let effective_mass_j = mass_j * j_mass_multiplier;
                    let total_mass = mass_i + effective_mass_j;
                    let my_fraction = effective_mass_j / total_mass;

                    // ACCUMULATE correction instead of applying directly
                    accumulated_correction += normal * overlap * my_fraction;
                    particle_contact_count += 1u;

                    // Check if this particle is below us (provides support)
                    // normal points from j to us, so if normal.y > 0.5, j is below us
                    if (normal.y > 0.5) {
                        has_support_below = true;
                    }

                    // Tangential friction - allows particles to SLIDE past each other
                    // This is critical for angle of repose behavior
                    let tangent = vec2(-normal.y, normal.x);
                    let rel_vel = vel - vel_j;
                    let v_tangent = dot(rel_vel, tangent);
                    let v_normal = dot(rel_vel, -normal);

                    // Coulomb friction: tangential force <= mu * normal force
                    // Only apply if there's relative tangential motion
                    let mu_ij = (mu + FRICTIONS[min(material_j, 5u)]) * 0.5;  // Average friction
                    if (abs(v_tangent) > 0.1 && v_normal > 0.0) {
                        let max_friction = mu_ij * v_normal;
                        let friction_impulse = min(abs(v_tangent) * 0.5, max_friction);
                        accumulated_vel_correction -= tangent * sign(v_tangent) * friction_impulse * my_fraction;
                    }

                    // Also accumulate velocity correction for normal direction
                    let v_into = dot(vel, -normal);
                    if (v_into > 0.0) {
                        accumulated_vel_correction += normal * v_into;
                    }
                }
            }
        }
    }

    // Apply AVERAGED correction (Jacobi with SOR)
    if (particle_contact_count > 0u) {
        let omega = 1.0;  // SOR factor = 1.0 for pure averaging (no over-relaxation)
        let avg_factor = omega / f32(particle_contact_count);
        pos += accumulated_correction * avg_factor;
        vel += accumulated_vel_correction * avg_factor;
    }

    // 5. Floor/SDF collision with penetration tolerance ("slop")
    // Slop creates a stable equilibrium zone where particles can rest
    // See: Box2D/Erin Catto approach - "allow some penetration to avoid contact breaking"
    let sdf = sample_sdf(pos);
    let slop = radius * 0.1;  // Allow 10% penetration as stable zone

    if (sdf < radius) {
        let grad = sdf_gradient(pos);
        let penetration = radius - sdf;

        // Only correct penetration beyond the slop zone
        // Particles settle INTO the slop, not bounce above it
        let correction = max(penetration - slop, 0.0);
        pos += grad * correction;

        floor_contact = true;

        // Zero velocity into floor
        let v_into_floor = dot(vel, -grad);
        if (v_into_floor > 0.0) {
            vel += grad * v_into_floor;
        }

        // Apply floor friction
        let tangent = vec2(-grad.y, grad.x);
        let v_tangent = dot(vel, tangent);
        let friction_force = min(abs(v_tangent), mu * abs(v_into_floor));
        vel -= tangent * sign(v_tangent) * friction_force;
    }

    // 6. Damp velocity for particles with contact
    // Use moderate damping - too strong kills lateral sliding needed for angle of repose
    let has_contact = floor_contact || has_support_below;
    if (has_contact) {
        vel *= 0.7;  // Moderate damping - allows sliding while dissipating energy
    } else if (particle_contact_count > 2u) {
        vel *= 0.9;  // Light damping for particles with multiple contacts
    }

    // 7. Support propagation sleep system
    // Support MUST propagate from floor - no support in mid-air!
    // - Floor contact = immediate support
    // - Neighbor below with support = chain support
    // - No support = counter decays (can't sleep mid-air)

    let speed_sq = dot(vel, vel);

    // True support: ONLY actual floor contact OR chain from supported neighbor below
    // Removed "near_floor" heuristic - it caused mid-air pauses near vertical walls
    // because SDF doesn't distinguish floors from walls
    let has_floor_support = floor_contact;  // Must actually touch floor
    let has_chain_support = supported_contacts >= 1u;
    let truly_supported = has_floor_support || has_chain_support;

    // Update sleep counter - ONLY supported particles can maintain high counter
    if (speed_sq > WAKE_SPEED_SQ || should_wake) {
        // Fast or woken = reset
        sleep_counter = 0u;
    } else if (!truly_supported) {
        // NO SUPPORT = counter decays (can't sleep mid-air)
        sleep_counter = sleep_counter / 2u;
    } else if (speed_sq < JITTER_SPEED_SQ && has_floor_support) {
        // Floor support + slow = fast sleep
        sleep_counter = min(sleep_counter + 3u, SLEEP_THRESHOLD * 2u);
    } else if (speed_sq < JITTER_SPEED_SQ && has_chain_support) {
        // Chain support + slow = normal sleep
        sleep_counter = min(sleep_counter + 2u, SLEEP_THRESHOLD * 2u);
    } else if (speed_sq < JITTER_SPEED_SQ * 4.0 && truly_supported) {
        // Somewhat slow + support = slow increment
        sleep_counter = min(sleep_counter + 1u, SLEEP_THRESHOLD);
    }

    // Apply sleep: zero velocity when supported and counter high
    if (sleep_counter >= SLEEP_THRESHOLD && truly_supported && !should_wake) {
        vel = vec2(0.0, 0.0);
    }

    // Write back
    positions[idx] = pos;
    velocities[idx] = vel;
    sleep_counters[idx] = sleep_counter;
}
