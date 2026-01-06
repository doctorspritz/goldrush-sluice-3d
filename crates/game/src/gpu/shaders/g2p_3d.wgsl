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
    drag_rate: f32,
    settling_velocity: f32,
    vorticity_lift: f32,
    vorticity_threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

struct DruckerPragerParams {
    friction_coeff: f32,
    cohesion: f32,
    buoyancy_factor: f32,
    viscosity: f32,
    jammed_drag: f32,
    min_pressure: f32,
    yield_smoothing: f32,
    _pad0: f32,
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
@group(0) @binding(15) var<uniform> dp_params: DruckerPragerParams;
@group(0) @binding(16) var<storage, read> sediment_pressure: array<f32>;
@group(0) @binding(17) var<storage, read> sediment_count: array<atomic<i32>>;
@group(0) @binding(18) var<storage, read> water_count: array<atomic<i32>>;

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

// Compute shear rate magnitude from APIC C matrix (velocity gradient).
fn compute_shear_rate_from_c(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> f32 {
    // Strain rate tensor (symmetric part): D = 0.5*(C + C^T)
    let D_xx = c0.x;
    let D_yy = c1.y;
    let D_zz = c2.z;
    let D_xy = 0.5 * (c0.y + c1.x);
    let D_xz = 0.5 * (c0.z + c2.x);
    let D_yz = 0.5 * (c1.z + c2.y);

    // Second invariant: |D| = sqrt(0.5 * D:D)
    // D:D = D_xx^2 + D_yy^2 + D_zz^2 + 2*(D_xy^2 + D_xz^2 + D_yz^2)
    let D_sq = D_xx*D_xx + D_yy*D_yy + D_zz*D_zz +
               2.0*(D_xy*D_xy + D_xz*D_xz + D_yz*D_yz);

    return sqrt(0.5 * D_sq);
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let old_particle_vel = velocities[id.x];
    let cell_size = params.cell_size;
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
    }
    if (weight_sum.y > 0.0) {
        new_velocity.y /= weight_sum.y;
        old_grid_vel.y /= weight_sum.y;
        new_c_row1 /= weight_sum.y;
    }
    if (weight_sum.z > 0.0) {
        new_velocity.z /= weight_sum.z;
        old_grid_vel.z /= weight_sum.z;
        new_c_row2 /= weight_sum.z;
    }

    let density = densities[id.x];
    if (density > 1.0) {
        // ========== SEDIMENT: Drucker-Prager yield model ==========

        let cell_i = clamp(i32(pos.x / cell_size), 0, width - 1);
        let cell_j = clamp(i32(pos.y / cell_size), 0, height - 1);
        let cell_k = clamp(i32(pos.z / cell_size), 0, depth - 1);
        let cell_idx = cell_index(cell_i, cell_j, cell_k);

        // Sample sediment pressure (from column weight)
        let pressure = max(sediment_pressure[cell_idx], dp_params.min_pressure);
        let water_present = atomicLoad(&water_count[cell_idx]) > 0;

        // Compute shear rate from APIC C matrix
        let shear_rate = compute_shear_rate_from_c(new_c_row0, new_c_row1, new_c_row2);

        // Drucker-Prager yield criterion
        let yield_stress = dp_params.cohesion + pressure * dp_params.friction_coeff;
        let shear_stress = dp_params.viscosity * shear_rate;

        let stress_diff = shear_stress - yield_stress;
        let yield_ratio = clamp(
            stress_diff / (dp_params.yield_smoothing * yield_stress + 0.001),
            0.0,
            1.0
        );

        var final_velocity: vec3<f32>;

        if (yield_ratio > 0.01) {
            // YIELDING - viscoplastic flow
            let effective_drag = yield_ratio * sediment_params.drag_rate;
            let drag_factor = select(0.0, 1.0 - exp(-effective_drag * params.dt), water_present);
            final_velocity = old_particle_vel + (new_velocity - old_particle_vel) * drag_factor;

            // Reduced settling when yielding (material is flowing)
            let settling_reduction = select(1.0, 1.0 - yield_ratio * 0.7, water_present);
            final_velocity.y -= sediment_params.settling_velocity * params.dt * settling_reduction;
        } else {
            // JAMMED - damp toward rest (resist flow)
            let jam_damp = 1.0 - exp(-dp_params.jammed_drag * params.dt);
            final_velocity = old_particle_vel;
            final_velocity.x *= 1.0 - jam_damp;
            final_velocity.z *= 1.0 - jam_damp;

            // Full settling when jammed
            final_velocity.y -= sediment_params.settling_velocity * params.dt;
        }

        // Vorticity lift (suspension in turbulent flow)
        let vort = vorticity_mag[cell_idx];
        let vort_excess = max(vort - sediment_params.vorticity_threshold, 0.0);
        let lift_factor = clamp(sediment_params.vorticity_lift * vort_excess, 0.0, 0.9);
        final_velocity.y += sediment_params.settling_velocity * params.dt * lift_factor;

        // Safety clamp
        let speed = length(final_velocity);
        if (speed > params.max_velocity) {
            final_velocity *= params.max_velocity / speed;
        }

        velocities[id.x] = final_velocity;
        c_col0[id.x] = vec3<f32>(0.0, 0.0, 0.0);
        c_col1[id.x] = vec3<f32>(0.0, 0.0, 0.0);
        c_col2[id.x] = vec3<f32>(0.0, 0.0, 0.0);
        return;
    }

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

    var final_velocity = params.flip_ratio * flip_velocity + (1.0 - params.flip_ratio) * pic_velocity;

    // Safety clamp
    let speed = length(final_velocity);
    if (speed > params.max_velocity) {
        final_velocity *= params.max_velocity / speed;
    }

    // ========== Write outputs ==========
    velocities[id.x] = final_velocity;
    c_col0[id.x] = new_c_row0;  // Note: binding named c_col for historical reasons
    c_col1[id.x] = new_c_row1;
    c_col2[id.x] = new_c_row2;
}
