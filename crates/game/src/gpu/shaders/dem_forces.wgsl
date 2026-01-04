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
const IMPACT_SPEED_SQ: f32 = 900.0; // Above this = impact wake (~30 px/s)
// Chain support threshold - LOWER than SLEEP_THRESHOLD to allow faster propagation
// Floor particles reach 3 after just 1 frame (+3), enabling immediate chain support
const CHAIN_SUPPORT_THRESHOLD: u32 = 3u;

@group(0) @binding(0) var<uniform> params: DemParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> radii: array<f32>;
@group(0) @binding(4) var<storage, read> materials: array<u32>;
@group(0) @binding(5) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read> sdf_data: array<f32>;
@group(0) @binding(8) var<storage, read_write> sleep_counters: array<u32>;
// Use atomic for static_states to allow dynamic particles to mark static neighbors for wake
// Values: 0 = dynamic, 1 = static, 2 = marked-for-wake
@group(0) @binding(9) var<storage, read_write> static_states: array<atomic<u32>>;

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

    // Read static state - STATIC particles need force-threshold wake check
    // We DON'T return early anymore - we need to detect impacts that should wake us
    // Using atomic load since buffer uses atomic<u32>
    // States: 0 = dynamic, 1 = static (frozen), 2 = marked for wake
    let my_static_state = atomicLoad(&static_states[idx]);
    let was_static = my_static_state == 1u;
    let marked_for_wake = my_static_state == 2u;

    // Track maximum impact force for wake threshold check
    var max_impact_force: f32 = 0.0;

    // Sleep system: read current sleep counter
    var sleep_counter = sleep_counters[idx];
    let is_sleeping = sleep_counter >= SLEEP_THRESHOLD;

    // Handle wake marker from previous frame - clear state and counter
    // This prevents particles from immediately going back to static
    if (marked_for_wake) {
        atomicStore(&static_states[idx], 0u);  // Clear wake marker → dynamic
        sleep_counter = 0u;  // Reset counter so we don't immediately re-sleep
    }

    // Check if particle was sleeping (low velocity = at rest)
    let was_sleeping = dot(vel, vel) < JITTER_SPEED_SQ;

    // Check if in water
    let in_water = params.water_level >= 0.0 && pos.y > params.water_level;

    // 1. Apply gravity and integrate (only on first iteration)
    if (params.iteration == 0u) {
        // Skip gravity for sleeping floor particles (gravity balanced by normal force)
        // This prevents jitter where sleeping particles: get gravity → move down →
        // hit floor → get pushed up → repeat
        let sdf_now = sample_sdf(pos);
        let grad_now = sdf_gradient(pos);
        // Require nearly horizontal surface - floor gradient points UP in screen coords
        // (negative y), so grad.y < -0.9 means horizontal floor below us
        // BUG FIX: Was grad_now.y > 0.9 which is WRONG (that's a ceiling!)
        let on_floor_now = sdf_now < radius * 1.1 && grad_now.y < -0.9;
        let skip_gravity = is_sleeping && on_floor_now;

        if (!skip_gravity) {
            let g_eff = effective_gravity(material, in_water);
            vel.y += g_eff * params.dt;
        }
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

    // Level 4: Track corrections from FAST neighbors separately
    // This avoids false wakes from normal contacts in settled piles
    var impact_correction = vec2<f32>(0.0, 0.0);
    var impact_vel_correction = vec2<f32>(0.0, 0.0);
    var impact_contact_count = 0u;

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

                    // NOTE: Surface roughness hack REMOVED in Level 5
                    // Angle of repose now comes from proper Coulomb friction: tan(θ) = μ

                    // Mass-weighted separation: heavy particles move less
                    let density_i = DENSITIES[min(material, 5u)];
                    let density_j = DENSITIES[min(material_j, 5u)];
                    let mass_i = density_i * radius * radius;
                    let mass_j = density_j * radius_j * radius_j;

                    // Check if j is part of a settled pile (stationary = part of stable structure)
                    let vel_j = velocities[j_idx];
                    let j_speed_sq = dot(vel_j, vel_j);
                    let j_is_stationary = j_speed_sq < 25.0;  // ~5 px/s

                    // Wake detection: if neighbor is moving fast, track impact
                    // ONLY check in iteration 0 - later iterations see post-collision velocities
                    let j_is_fast = j_speed_sq > WAKE_SPEED_SQ;
                    if (params.iteration == 0u && j_is_fast) {
                        should_wake = true;
                        // Track impact force for force-threshold wake (Level 4)
                        // Use impulse = mass * relative_velocity_into_us
                        // normal points from j to us, so dot(vel_j, normal) > 0 means j moving toward us
                        let v_into = max(0.0, dot(vel_j, normal));
                        let j_mass = density_j * radius_j * radius_j;
                        let impulse = j_mass * v_into;
                        max_impact_force = max(max_impact_force, impulse);
                    }

                    // Track if this is an impact (for static particle wake)
                    // We use THREE signals to detect impacts:
                    // 1. j_is_fast: neighbor is moving fast (but may be racy due to GPU parallelism)
                    // 2. !j_is_stationary: neighbor is not settled (less aggressive than j_is_fast)
                    // 3. Large overlap: indicates fast collision penetration (NOT racy - uses positions)
                    //
                    // The overlap check is critical because velocity readings have race conditions.
                    // A fast projectile creates large overlap before constraint solver can push it out.
                    // Normal settled contacts have very small overlaps.
                    //
                    // Use 20% of contact_dist as threshold (contact_dist = radius + radius_j)
                    let large_overlap_threshold = contact_dist * 0.2;
                    let has_large_overlap = overlap > large_overlap_threshold;

                    // Also check: if neighbor j is itself static, we shouldn't count that as impact
                    // (two static particles touching is equilibrium, not impact)
                    let j_static_state = atomicLoad(&static_states[j_idx]);
                    let j_is_static = j_static_state == 1u;

                    // Level 4: Dynamic particles mark static neighbors for wake
                    // Wake conditions (must satisfy BOTH):
                    // 1. I am dynamic (!was_static) hitting static neighbor (j_is_static)
                    // 2. Either:
                    //    a) High impact speed (>30 px/s) + any overlap
                    //    b) Large overlap (>30%) regardless of speed
                    //
                    // This prevents:
                    // - Gold gently settling (14 px/s, small overlap) from waking
                    // - But allows projectiles (50 px/s, any overlap) to wake
                    let my_speed_sq = dot(vel, vel);
                    let is_impact_speed = my_speed_sq > IMPACT_SPEED_SQ;
                    let wake_overlap_threshold = contact_dist * 0.3;
                    let has_wake_overlap = overlap > wake_overlap_threshold;
                    let should_wake = (is_impact_speed && overlap > 0.0) || has_wake_overlap;
                    if (!was_static && j_is_static && should_wake && params.iteration == 0u) {
                        // I'm dynamic AND hitting static neighbor hard - mark for wake
                        // Value 2 means "marked for wake"
                        atomicMax(&static_states[j_idx], 2u);
                    }

                    let is_impact = (j_is_fast || !j_is_stationary || has_large_overlap) && !j_is_static;

                    // Support propagation: check if neighbor BELOW us is settling
                    // In screen coords: j below us means j.y > our y, so diff.y < 0, so normal.y < 0
                    //
                    // KEY FIX: Use CHAIN_SUPPORT_THRESHOLD (3) instead of SLEEP_THRESHOLD (10)
                    // This allows support to propagate upward in 1-2 frames instead of ~50 frames.
                    // Floor particles get +3 per frame, so they're eligible to provide support
                    // after just 1 slow frame, allowing the entire pile to build sleep counters
                    // in parallel rather than serially layer-by-layer.
                    let j_sleep = sleep_counters[j_idx];
                    let j_is_below = normal.y < -0.3;
                    if (j_is_below && j_sleep >= CHAIN_SUPPORT_THRESHOLD) {
                        supported_contacts += 1u;
                    }

                    // Stationary particles resist displacement but can still be pushed
                    // 3x mass = moderate resistance, allows pile restructuring
                    let j_mass_multiplier = select(1.0, 3.0, j_is_stationary);
                    let effective_mass_j = mass_j * j_mass_multiplier;
                    let total_mass = mass_i + effective_mass_j;
                    let my_fraction = effective_mass_j / total_mass;

                    // ACCUMULATE correction instead of applying directly
                    let this_correction = normal * overlap * my_fraction;
                    accumulated_correction += this_correction;
                    particle_contact_count += 1u;

                    // Level 4: Track impact corrections separately (for wake detection)
                    if (is_impact) {
                        impact_correction += this_correction;
                        impact_contact_count += 1u;
                    }

                    // Check if this particle is below us (provides support)
                    // normal = (pos - pos_j) / dist, points from j to us
                    // In screen coords: if j is below us (j.y > our y), then diff.y < 0, so normal.y < 0
                    if (normal.y < -0.5) {
                        has_support_below = true;
                    }

                    // Coulomb friction: F_tangent <= μ × F_normal
                    // This creates angle of repose: tan(θ) = μ
                    let tangent = vec2(-normal.y, normal.x);
                    let rel_vel = vel - vel_j;
                    let v_tangent = dot(rel_vel, tangent);

                    // Add tiny velocity perturbation to break perfect symmetry
                    // This simulates natural micro-scale surface irregularities
                    // Without this, perfectly aligned particles stack perfectly (unphysical)
                    let noise_seed = idx * 31u + j_idx * 17u + params.iteration;
                    let noise = hash_noise(noise_seed) * 0.3;  // Small perturbation
                    let v_tangent_perturbed = v_tangent + noise;

                    // For PBD: friction limits tangential velocity relative to normal overlap
                    let mu_ij = (mu + FRICTIONS[min(material_j, 5u)]) * 0.5;  // Average friction

                    // Scale factor: convert overlap to velocity-scale friction limit
                    let friction_scale = 0.5;  // Tunable: higher = more friction
                    let max_friction_vel = mu_ij * overlap * friction_scale;

                    // Apply friction: reduce tangential velocity, clamped by Coulomb limit
                    if (abs(v_tangent_perturbed) > 0.1) {
                        let friction_vel = min(abs(v_tangent_perturbed) * 0.5, max_friction_vel) * my_fraction;
                        accumulated_vel_correction -= tangent * sign(v_tangent_perturbed) * friction_vel;
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

        // Only count as floor contact if surface is roughly horizontal
        // In screen coords (y increases downward), floor gradient points UP (negative y)
        // Vertical walls (grad.y ~ 0) should NOT count as floor support
        floor_contact = grad.y < -0.7;

        // Zero velocity into floor
        let v_into_floor = dot(vel, -grad);
        if (v_into_floor > 0.0) {
            vel += grad * v_into_floor;
        }

        // Apply floor Coulomb friction: F_tangent <= μ × F_normal
        // For PBD: friction proportional to penetration (overlap with floor)
        let friction_scale = 0.5;  // Match particle-particle friction scaling
        let max_floor_friction = mu * penetration * friction_scale;

        let tangent = vec2(-grad.y, grad.x);
        let v_tangent = dot(vel, tangent);
        if (abs(v_tangent) > 0.1) {
            let friction_vel = min(abs(v_tangent) * 0.5, max_floor_friction);
            vel -= tangent * sign(v_tangent) * friction_vel;
        }
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

    // Update sleep counter - ONLY on iteration 0 to avoid 4x increment per frame
    // ONLY supported particles can maintain high counter
    if (params.iteration == 0u) {
        if (speed_sq > WAKE_SPEED_SQ || should_wake) {
            // Fast or woken = reset
            sleep_counter = 0u;
        } else if (!truly_supported && speed_sq > JITTER_SPEED_SQ) {
            // NO SUPPORT AND actually moving = counter decays (can't sleep mid-air)
            // KEY FIX: Only decay if actually moving. Slow particles shouldn't have
            // their counters destroyed due to 1-frame GPU read lag causing temporary
            // "unsupported" status. This allows the support chain to propagate.
            sleep_counter = sleep_counter / 2u;
        } else if (!truly_supported) {
            // Unsupported but slow - don't decay, just don't increment
            // This prevents mid-air freezing while allowing chain propagation
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
    }

    // Level 4: Force-threshold wake for STATIC particles
    // Static particles only wake if they receive a significant impact from a MOVING neighbor
    //
    // We use THREE methods to detect wake:
    // 1. marked_for_wake: Dynamic particle already set our state to 2 (race-condition-free)
    // 2. impact_correction: Significant push from non-stationary neighbor
    // 3. should_wake: Fast neighbor detected (may be racy but kept as backup)
    if (was_static) {
        // Check if a dynamic particle marked us for wake
        let current_state = atomicLoad(&static_states[idx]);
        let marked_for_wake = current_state == 2u;

        // Also check impact_correction (from non-stationary neighbors)
        let impact_magnitude = length(impact_correction);
        let wake_correction_threshold = radius * 0.15;
        let significant_push = impact_contact_count > 0u && impact_magnitude > wake_correction_threshold;

        let should_wake_up = marked_for_wake || significant_push || should_wake;

        if (should_wake_up) {
            // WAKE UP - become dynamic
            atomicStore(&static_states[idx], 0u);
            sleep_counter = 0u;
            // Continue to write back pos and vel with corrections applied
        } else {
            // No significant impact - STAY FROZEN
            // Discard all position/velocity changes
            positions[idx] = old_pos;
            velocities[idx] = vec2(0.0, 0.0);
            sleep_counters[idx] = sleep_counter;
            return;
        }
    }

    // Apply sleep: zero velocity when supported and counter high
    // Level 3: Also transition to STATIC state - will skip physics entirely next frame
    if (sleep_counter >= SLEEP_THRESHOLD && truly_supported && !should_wake) {
        vel = vec2(0.0, 0.0);
        atomicStore(&static_states[idx], 1u);  // Become STATIC
    }

    // Write back
    positions[idx] = pos;
    velocities[idx] = vel;
    sleep_counters[idx] = sleep_counter;
}
