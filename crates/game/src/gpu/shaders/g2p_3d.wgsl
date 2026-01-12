// G2P Shader (3D) - Grid-to-Particle transfer for water particles (APIC)
//
// Gathers velocity from grid nodes to particles using quadratic B-spline weights.
// Also reconstructs the APIC affine velocity matrix C from grid velocities.
//
// This is the inverse of P2G scatter - each particle reads from its 3x3x3 neighborhood.
// No atomics needed since each particle writes only to its own data.
//
// Grid layout (MAC staggered):
// - U velocities: stored at left YZ faces, (width+1) x height x depth
// - V velocities: stored at bottom XZ faces, width x (height+1) x depth
// - W velocities: stored at back XY faces, width x height x (depth+1)
//
// FLIP delta: Uses grid_*_old (pre-force velocities) vs grid_* (post-force)

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    d_inv: f32,          // APIC D inverse = 4/dx^2
    flip_ratio: f32,     // FLIP blend ratio (0.97 for water)
    dt: f32,             // Time step for velocity clamping
    max_velocity: f32,   // Safety clamp (2000.0)
}

struct SedimentParams {
    settling_velocity: f32,     // How fast sediment falls (m/s)
    friction_threshold: f32,    // Speed below which friction kicks in (m/s)
    friction_strength: f32,     // How much to damp when slow (0-1 per frame)
    vorticity_lift: f32,        // How much vorticity suspends sediment
    vorticity_threshold: f32,   // Minimum vorticity to lift
    drag_coefficient: f32,      // Rate at which particle approaches water velocity (1/s)
    gold_density_threshold: f32,
    gold_drag_multiplier: f32,
    gold_settling_velocity: f32,
    gold_flake_lift: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec3<f32>>;
// C matrix stored as 3 ROWS (c_col0 = row 0, etc.)
// This naming is historical; they're actually rows from the staggered APIC reconstruction
@group(0) @binding(3) var<storage, read_write> c_col0: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read_write> c_col1: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read_write> c_col2: array<vec3<f32>>;
// Post-force grid velocities
@group(0) @binding(6) var<storage, read> grid_u: array<f32>;
@group(0) @binding(7) var<storage, read> grid_v: array<f32>;
@group(0) @binding(8) var<storage, read> grid_w: array<f32>;
// Pre-force grid velocities (for FLIP delta)
@group(0) @binding(9) var<storage, read> grid_u_old: array<f32>;
@group(0) @binding(10) var<storage, read> grid_v_old: array<f32>;
@group(0) @binding(11) var<storage, read> grid_w_old: array<f32>;
@group(0) @binding(12) var<storage, read> densities: array<f32>;
@group(0) @binding(13) var<uniform> sediment_params: SedimentParams;
@group(0) @binding(14) var<storage, read> vorticity_mag: array<f32>;
@group(0) @binding(15) var<storage, read> water_grid_u: array<f32>;
@group(0) @binding(16) var<storage, read> water_grid_v: array<f32>;
@group(0) @binding(17) var<storage, read> water_grid_w: array<f32>;

// Quadratic B-spline kernel (1D)
fn quadratic_bspline_1d(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.5) {
        return 0.75 - ax * ax;
    } else if (ax < 1.5) {
        let t = 1.5 - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

// Index functions for staggered grids
fn u_index(i: i32, j: i32, k: i32) -> u32 {
    // U grid: (width+1) x height x depth
    return u32(k) * (params.width + 1u) * params.height + u32(j) * (params.width + 1u) + u32(i);
}

fn v_index(i: i32, j: i32, k: i32) -> u32 {
    // V grid: width x (height+1) x depth
    return u32(k) * params.width * (params.height + 1u) + u32(j) * params.width + u32(i);
}

fn w_index(i: i32, j: i32, k: i32) -> u32 {
    // W grid: width x height x (depth+1)
    return u32(k) * params.width * params.height + u32(j) * params.width + u32(i);
}

fn cell_index(i: i32, j: i32, k: i32) -> u32 {
    return u32(k) * params.width * params.height + u32(j) * params.width + u32(i);
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let old_particle_vel = velocities[id.x];
    let cell_size = params.cell_size;
    let density = densities[id.x];
    let is_sediment = density > 1.0;
    let width = i32(params.width);
    let height = i32(params.height);
    let depth = i32(params.depth);
    let d_inv = params.d_inv;

    // ========== Sample grid velocity and reconstruct C matrix ==========
    var new_velocity = vec3<f32>(0.0, 0.0, 0.0);
    var new_c_row0 = vec3<f32>(0.0, 0.0, 0.0);  // Row 0 of C matrix (from U velocity)
    var new_c_row1 = vec3<f32>(0.0, 0.0, 0.0);  // Row 1 of C matrix (from V velocity)
    var new_c_row2 = vec3<f32>(0.0, 0.0, 0.0);  // Row 2 of C matrix (from W velocity)
    var weight_sum = vec3<f32>(0.0, 0.0, 0.0);  // u, v, w weight sums
    var water_velocity = vec3<f32>(0.0, 0.0, 0.0);

    // Also sample old grid velocity for FLIP delta
    // NOTE: old and new use same stencil, so weights are identical
    var old_grid_vel = vec3<f32>(0.0, 0.0, 0.0);

    // ========== U component (staggered on left YZ faces) ==========
    let u_pos = pos / cell_size - vec3<f32>(0.0, 0.5, 0.5);
    let base_u = vec3<i32>(floor(u_pos));
    let frac_u = u_pos - vec3<f32>(base_u);

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base_u.z + dk;
        if (nk < 0 || nk >= depth) { continue; }
        let wz = quadratic_bspline_1d(frac_u.z - f32(dk));

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base_u.y + dj;
            if (nj < 0 || nj >= height) { continue; }
            let wy = quadratic_bspline_1d(frac_u.y - f32(dj));

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base_u.x + di;
                if (ni < 0 || ni > width) { continue; }  // U has width+1 columns

                let w = quadratic_bspline_1d(frac_u.x - f32(di)) * wy * wz;
                if (w <= 0.0) { continue; }

                let idx = u_index(ni, nj, nk);
                let u_val = grid_u[idx];
                let u_old = grid_u_old[idx];

                new_velocity.x += w * u_val;
                old_grid_vel.x += w * u_old;
                weight_sum.x += w;
                if (is_sediment) {
                    let u_water = water_grid_u[idx];
                    water_velocity.x += w * u_water;
                }

                // C matrix: row 0 (from U velocity component)
                // APIC: C[0,:] = Σ w * u * offset * d_inv
                let offset = vec3<f32>(
                    f32(ni) * cell_size - pos.x,
                    (f32(nj) + 0.5) * cell_size - pos.y,
                    (f32(nk) + 0.5) * cell_size - pos.z
                );
                new_c_row0 += offset * (w * u_val * d_inv);
            }
        }
    }

    // ========== V component (staggered on bottom XZ faces) ==========
    let v_pos = pos / cell_size - vec3<f32>(0.5, 0.0, 0.5);
    let base_v = vec3<i32>(floor(v_pos));
    let frac_v = v_pos - vec3<f32>(base_v);

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base_v.z + dk;
        if (nk < 0 || nk >= depth) { continue; }
        let wz = quadratic_bspline_1d(frac_v.z - f32(dk));

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base_v.y + dj;
            if (nj < 0 || nj > height) { continue; }  // V has height+1 rows

            let wy = quadratic_bspline_1d(frac_v.y - f32(dj));

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base_v.x + di;
                if (ni < 0 || ni >= width) { continue; }

                let w = quadratic_bspline_1d(frac_v.x - f32(di)) * wy * wz;
                if (w <= 0.0) { continue; }

                let idx = v_index(ni, nj, nk);
                let v_val = grid_v[idx];
                let v_old = grid_v_old[idx];

                new_velocity.y += w * v_val;
                old_grid_vel.y += w * v_old;
                weight_sum.y += w;
                if (is_sediment) {
                    let v_water = water_grid_v[idx];
                    water_velocity.y += w * v_water;
                }

                // C matrix: row 1 (from V velocity component)
                // APIC: C[1,:] = Σ w * v * offset * d_inv
                let offset = vec3<f32>(
                    (f32(ni) + 0.5) * cell_size - pos.x,
                    f32(nj) * cell_size - pos.y,
                    (f32(nk) + 0.5) * cell_size - pos.z
                );
                new_c_row1 += offset * (w * v_val * d_inv);
            }
        }
    }

    // ========== W component (staggered on back XY faces) ==========
    let w_pos = pos / cell_size - vec3<f32>(0.5, 0.5, 0.0);
    let base_w = vec3<i32>(floor(w_pos));
    let frac_w = w_pos - vec3<f32>(base_w);

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base_w.z + dk;
        if (nk < 0 || nk > depth) { continue; }  // W has depth+1 layers
        let wz = quadratic_bspline_1d(frac_w.z - f32(dk));

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base_w.y + dj;
            if (nj < 0 || nj >= height) { continue; }
            let wy = quadratic_bspline_1d(frac_w.y - f32(dj));

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base_w.x + di;
                if (ni < 0 || ni >= width) { continue; }

                let w = quadratic_bspline_1d(frac_w.x - f32(di)) * wy * wz;
                if (w <= 0.0) { continue; }

                let idx = w_index(ni, nj, nk);
                let w_val = grid_w[idx];
                let w_old = grid_w_old[idx];

                new_velocity.z += w * w_val;
                old_grid_vel.z += w * w_old;
                weight_sum.z += w;
                if (is_sediment) {
                    let w_water = water_grid_w[idx];
                    water_velocity.z += w * w_water;
                }

                // C matrix: row 2 (from W velocity component)
                // APIC: C[2,:] = Σ w * w * offset * d_inv
                let offset = vec3<f32>(
                    (f32(ni) + 0.5) * cell_size - pos.x,
                    (f32(nj) + 0.5) * cell_size - pos.y,
                    f32(nk) * cell_size - pos.z
                );
                new_c_row2 += offset * (w * w_val * d_inv);
            }
        }
    }

    // Normalize velocities AND C matrix by weight sum (weighted average)
    // C matrix MUST be normalized - without this, C explodes near boundaries where Σw < 1
    // NOTE: old_grid_vel uses same stencil as new_velocity, so both normalize by same weights
    if (weight_sum.x > 0.0) {
        new_velocity.x /= weight_sum.x;
        old_grid_vel.x /= weight_sum.x;
        new_c_row0 /= weight_sum.x;
        if (is_sediment) {
            water_velocity.x /= weight_sum.x;
        }
    }
    if (weight_sum.y > 0.0) {
        new_velocity.y /= weight_sum.y;
        old_grid_vel.y /= weight_sum.y;
        new_c_row1 /= weight_sum.y;
        if (is_sediment) {
            water_velocity.y /= weight_sum.y;
        }
    }
    if (weight_sum.z > 0.0) {
        new_velocity.z /= weight_sum.z;
        old_grid_vel.z /= weight_sum.z;
        new_c_row2 /= weight_sum.z;
        if (is_sediment) {
            water_velocity.z /= weight_sum.z;
        }
    }

    let is_gold = density >= sediment_params.gold_density_threshold;
    // ========== FLIP/PIC blend ==========
    let grid_delta = new_velocity - old_grid_vel;

    // Clamp delta to prevent energy explosions (5 cells per frame max)
    let max_dv = 5.0 * cell_size / params.dt;
    let delta_len_sq = dot(grid_delta, grid_delta);
    var clamped_delta = grid_delta;
    if (delta_len_sq > max_dv * max_dv) {
        clamped_delta = normalize(grid_delta) * max_dv;
    }

    let flip_velocity = old_particle_vel + clamped_delta;
    let pic_velocity = new_velocity;

    var final_velocity: vec3<f32>;

    if (is_sediment) {
        // ========== DRAG-BASED ENTRAINMENT MODEL ==========
        // Sediment is pulled toward water velocity by drag,
        // while gravity (reduced by buoyancy) pulls it down.
        // When water flows fast: drag wins → entrainment
        // When water is slow: gravity wins → settling

        let water_vel = water_velocity;  // Water-only velocity from grid
        let particle_vel = old_particle_vel;

        // CORRECTED DRAG MODEL FOR SETTLING
        // The goal: heavy particles settle upstream, light particles wash downstream.
        //
        // Key insight: in a sluice, SETTLING happens while particles are in the water
        // column, not after they hit the floor. Heavy particles sink fast and land
        // upstream; light particles drift downstream as they slowly sink.
        //
        // Problem with 1/density drag scaling: once particles hit the floor, heavy
        // particles slide (low drag) while light particles stick (high drag). WRONG!
        //
        // Fix: use UNIFORM drag for all particles. The separation happens during
        // settling through the water column, not during floor sliding.

        var drag_rate = sediment_params.drag_coefficient;  // Same drag for all!
        if (is_gold) {
            drag_rate *= sediment_params.gold_drag_multiplier;
        }
        let drag_blend = min(drag_rate * params.dt, 0.9);

        // Blend particle velocity toward water velocity (drag entrainment)
        var vel_after_drag = mix(particle_vel, water_vel, drag_blend);

        // SETTLING: This is where density matters!
        // Heavy particles sink faster, landing upstream before the flow carries them.
        // Light particles sink slowly, drifting downstream as they fall.
        //
        // Terminal velocity in water ~ sqrt((rho_p - rho_w) * g * d / (C_d * rho_w))
        // For settling, use (density - 1) as the driving factor.
        // Gold (19.3): effective weight = 18.3 units → sinks FAST
        // Sand (2.65): effective weight = 1.65 units → sinks slower (~11x slower)

        let settling_velocity = sediment_params.settling_velocity * (density - 1.0);
        if (is_gold) {
            // Gold sinks even faster (it's dense and compact)
            vel_after_drag.y -= settling_velocity * 2.0;
        } else {
            vel_after_drag.y -= settling_velocity;
        }

        final_velocity = vel_after_drag;
    } else {
        // Water: full FLIP/PIC blend
        final_velocity = params.flip_ratio * flip_velocity + (1.0 - params.flip_ratio) * pic_velocity;
    }

    // ========== SEDIMENT: Friction for slow particles ==========
    if (is_sediment) {
        // Friction: when slow, damp velocity (creates clustering/settling)
        let speed = length(final_velocity);
        if (speed < sediment_params.friction_threshold && speed > 0.001) {
            // Smooth ramp: more friction as speed approaches zero
            let friction_factor = 1.0 - speed / sediment_params.friction_threshold;
            let friction = sediment_params.friction_strength * friction_factor;
            // Apply friction uniformly - drag model handles entrainment
            final_velocity *= 1.0 - friction;
        }

        // Vorticity lift: turbulence can suspend particles
        let cell_i = clamp(i32(pos.x / cell_size), 0, width - 1);
        let cell_j = clamp(i32(pos.y / cell_size), 0, height - 1);
        let cell_k = clamp(i32(pos.z / cell_size), 0, depth - 1);
        let cell_idx = cell_index(cell_i, cell_j, cell_k);
        let vort = vorticity_mag[cell_idx];

        let settling_velocity = select(
            sediment_params.settling_velocity,
            sediment_params.gold_settling_velocity,
            is_gold
        );

        if (vort > sediment_params.vorticity_threshold) {
            let vort_excess = vort - sediment_params.vorticity_threshold;
            let lift = sediment_params.vorticity_lift * vort_excess * settling_velocity * params.dt;
            final_velocity.y += min(lift, settling_velocity * params.dt * 0.9);
        }

        if (is_gold && sediment_params.gold_flake_lift > 0.0) {
            let float_factor = smoothstep(0.0, sediment_params.friction_threshold, sediment_params.friction_threshold - speed);
            final_velocity.y += sediment_params.gold_flake_lift * float_factor * params.dt;
        }
    }

    // All particles keep APIC C matrix for angular momentum transfer
    c_col0[id.x] = new_c_row0;
    c_col1[id.x] = new_c_row1;
    c_col2[id.x] = new_c_row2;

    // Safety clamp
    let final_speed = length(final_velocity);
    if (final_speed > params.max_velocity) {
        final_velocity *= params.max_velocity / final_speed;
    }

    // ========== Write velocity ==========
    velocities[id.x] = final_velocity;
}
