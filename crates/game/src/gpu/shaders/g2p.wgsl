// G2P Shader - Grid-to-Particle transfer for water particles (APIC)
//
// Gathers velocity from grid nodes to particles using quadratic B-spline weights.
// Also reconstructs the APIC affine velocity matrix C from grid velocities.
//
// This is the inverse of P2G scatter - each particle reads from its 3x3 neighborhood.
// No atomics needed since each particle writes only to its own data.
//
// Grid layout (MAC staggered):
// - U velocities: stored at left edges, (width+1) x height
// - V velocities: stored at bottom edges, width x (height+1)
//
// FLIP delta: Uses grid_u_old/grid_v_old (pre-force velocities) vs grid_u/grid_v (post-force)

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    d_inv: f32,          // APIC D inverse = 4/dx^2
    flip_ratio: f32,     // FLIP blend ratio (0.97 for water)
    dt: f32,             // Time step for velocity clamping
    max_velocity: f32,   // Safety clamp (2000.0)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> c_matrices: array<vec4<f32>>; // mat2x2 as vec4
@group(0) @binding(4) var<storage, read> grid_u: array<f32>;      // Post-force velocities
@group(0) @binding(5) var<storage, read> grid_v: array<f32>;
@group(0) @binding(6) var<storage, read> grid_u_old: array<f32>;  // Pre-force velocities (for FLIP delta)
@group(0) @binding(7) var<storage, read> grid_v_old: array<f32>;

// Quadratic B-spline kernel (1D)
// Returns weight for grid node at distance x from particle
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

// 2D quadratic B-spline (product of 1D weights)
fn quadratic_bspline(delta: vec2<f32>) -> f32 {
    return quadratic_bspline_1d(delta.x) * quadratic_bspline_1d(delta.y);
}

// U grid index: (width+1) x height
fn u_index(i: i32, j: i32, width: u32) -> u32 {
    return u32(j) * (width + 1u) + u32(i);
}

// V grid index: width x (height+1)
fn v_index(i: i32, j: i32, width: u32) -> u32 {
    return u32(j) * width + u32(i);
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let old_particle_vel = velocities[id.x];
    let cell_size = params.cell_size;
    let width = params.width;
    let height = params.height;
    let d_inv = params.d_inv;

    // ========== Sample grid velocity and reconstruct C matrix ==========
    var new_velocity = vec2<f32>(0.0, 0.0);
    var new_c = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    var u_weight_sum: f32 = 0.0;
    var v_weight_sum: f32 = 0.0;

    // Also sample old grid velocity for FLIP delta
    var old_grid_vel = vec2<f32>(0.0, 0.0);
    var u_old_weight_sum: f32 = 0.0;
    var v_old_weight_sum: f32 = 0.0;

    // ========== U component (staggered on left edges) ==========
    // U sample point is at (i, j+0.5) in cell coordinates
    let u_pos = pos / cell_size - vec2<f32>(0.0, 0.5);
    let base_i_u = i32(floor(u_pos.x));
    let base_j_u = i32(floor(u_pos.y));
    let fx_u = u_pos.x - f32(base_i_u);
    let fy_u = u_pos.y - f32(base_j_u);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_u + di;
            let nj = base_j_u + dj;

            // Bounds check: U grid is (width+1) x height
            if (ni < 0 || ni > i32(width) || nj < 0 || nj >= i32(height)) {
                continue;
            }

            let delta = vec2<f32>(fx_u - f32(di), fy_u - f32(dj));
            let w = quadratic_bspline(delta);
            if (w <= 0.0) {
                continue;
            }

            let idx = u_index(ni, nj, width);
            let u_val = grid_u[idx];
            let u_old_val = grid_u_old[idx];

            // Velocity contribution
            new_velocity.x += w * u_val;
            u_weight_sum += w;

            // Old grid velocity (same kernel)
            old_grid_vel.x += w * u_old_val;
            u_old_weight_sum += w;

            // C matrix contribution: v_i ⊗ (x_i - x_p) * w * D_inv
            // Grid node position in world coords: (ni, nj + 0.5) * cell_size
            let offset = vec2<f32>(
                f32(ni) * cell_size - pos.x,
                (f32(nj) + 0.5) * cell_size - pos.y
            );
            // Outer product contribution: adds to C's first column (x component)
            new_c[0] += offset * (w * u_val * d_inv);
        }
    }

    // ========== V component (staggered on bottom edges) ==========
    // V sample point is at (i+0.5, j) in cell coordinates
    let v_pos = pos / cell_size - vec2<f32>(0.5, 0.0);
    let base_i_v = i32(floor(v_pos.x));
    let base_j_v = i32(floor(v_pos.y));
    let fx_v = v_pos.x - f32(base_i_v);
    let fy_v = v_pos.y - f32(base_j_v);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_v + di;
            let nj = base_j_v + dj;

            // Bounds check: V grid is width x (height+1)
            if (ni < 0 || ni >= i32(width) || nj < 0 || nj > i32(height)) {
                continue;
            }

            let delta = vec2<f32>(fx_v - f32(di), fy_v - f32(dj));
            let w = quadratic_bspline(delta);
            if (w <= 0.0) {
                continue;
            }

            let idx = v_index(ni, nj, width);
            let v_val = grid_v[idx];
            let v_old_val = grid_v_old[idx];

            // Velocity contribution
            new_velocity.y += w * v_val;
            v_weight_sum += w;

            // Old grid velocity (same kernel)
            old_grid_vel.y += w * v_old_val;
            v_old_weight_sum += w;

            // C matrix contribution: adds to C's second column (y component)
            let offset = vec2<f32>(
                (f32(ni) + 0.5) * cell_size - pos.x,
                f32(nj) * cell_size - pos.y
            );
            new_c[1] += offset * (w * v_val * d_inv);
        }
    }

    // Normalize by weight sum to handle boundary clipping
    if (u_weight_sum > 0.0) {
        new_velocity.x /= u_weight_sum;
        new_c[0] /= u_weight_sum;
        old_grid_vel.x /= u_old_weight_sum;
    }
    if (v_weight_sum > 0.0) {
        new_velocity.y /= v_weight_sum;
        new_c[1] /= v_weight_sum;
        old_grid_vel.y /= v_old_weight_sum;
    }

    // ========== FLIP/PIC blend ==========
    // FLIP: v_p = v_p^old + (v_grid^new - v_grid^old)
    // PIC:  v_p = v_grid^new
    // Blend: FLIP_RATIO * FLIP + (1 - FLIP_RATIO) * PIC

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

    // Store C matrix as vec4 (column-major: [c0.x, c0.y, c1.x, c1.y])
    c_matrices[id.x] = vec4<f32>(new_c[0].x, new_c[0].y, new_c[1].x, new_c[1].y);
}
