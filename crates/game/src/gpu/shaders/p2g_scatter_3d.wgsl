// P2G Scatter Shader (3D) - Particle-to-Grid transfer using fixed-point atomics
//
// Each particle contributes momentum to a 3x3x3 neighborhood of grid nodes using
// quadratic B-spline weights (APIC transfer). WebGPU only supports atomicAdd
// for i32, so we encode f32 * SCALE → i32, then decode after accumulation.
//
// Grid layout (MAC staggered):
// - U velocities: stored at left YZ faces, (width+1) x height x depth
// - V velocities: stored at bottom XZ faces, width x (height+1) x depth
// - W velocities: stored at back XY faces, width x height x (depth+1)

// Fixed-point scale: 10^6 gives ±2147 range with 6 decimal digits precision
const SCALE: f32 = 1000000.0;

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec3<f32>>;
// C matrix stored as 3 ROWS (c_col0 = row 0, etc.)
// This naming is historical; they're actually rows from the staggered APIC reconstruction
@group(0) @binding(3) var<storage, read> c_col0: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> c_col1: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> c_col2: array<vec3<f32>>;
// Atomic accumulation buffers
@group(0) @binding(6) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> u_weight: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> v_sum: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> v_weight: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> w_sum: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> w_weight: array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> particle_count: array<atomic<i32>>;

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

// Index functions for staggered grids
fn u_index(i: u32, j: u32, k: u32) -> u32 {
    // U grid: (width+1) x height x depth
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    // V grid: width x (height+1) x depth
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    // W grid: width x height x (depth+1)
    return k * params.width * params.height + j * params.width + i;
}

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    // Cell grid: width x height x depth
    return k * params.width * params.height + j * params.width + i;
}

// Multiply C matrix by offset vector
// C is stored as 3 ROWS (not columns!): c_col0 = row 0, etc.
// So C * offset = [dot(row0, offset), dot(row1, offset), dot(row2, offset)]
fn c_mat_mul(idx: u32, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c_col0[idx], offset),
        dot(c_col1[idx], offset),
        dot(c_col2[idx], offset)
    );
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let vel = velocities[id.x];
    let cell_size = params.cell_size;
    let width = params.width;
    let height = params.height;
    let depth = params.depth;

    // ========== U component (staggered on left YZ faces) ==========
    // U sample point is at (i, j+0.5, k+0.5) in cell coordinates
    let u_pos = pos / cell_size - vec3<f32>(0.0, 0.5, 0.5);
    let base_u = vec3<i32>(floor(u_pos));
    let frac_u = u_pos - vec3<f32>(base_u);

    // Precompute 1D weights for -1, 0, +1 offsets
    let u_wx = array<f32, 3>(
        quadratic_bspline_1d(frac_u.x + 1.0),
        quadratic_bspline_1d(frac_u.x),
        quadratic_bspline_1d(frac_u.x - 1.0)
    );
    let u_wy = array<f32, 3>(
        quadratic_bspline_1d(frac_u.y + 1.0),
        quadratic_bspline_1d(frac_u.y),
        quadratic_bspline_1d(frac_u.y - 1.0)
    );
    let u_wz = array<f32, 3>(
        quadratic_bspline_1d(frac_u.z + 1.0),
        quadratic_bspline_1d(frac_u.z),
        quadratic_bspline_1d(frac_u.z - 1.0)
    );

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base_u.z + dk;
        if (nk < 0 || nk >= i32(depth)) { continue; }
        let wz = u_wz[dk + 1];

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base_u.y + dj;
            if (nj < 0 || nj >= i32(height)) { continue; }
            let wy = u_wy[dj + 1];

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base_u.x + di;
                // U grid has width+1 columns (indices 0 to width inclusive)
                if (ni < 0 || ni > i32(width)) { continue; }

                let w = u_wx[di + 1] * wy * wz;
                if (w <= 0.0) { continue; }

                // APIC: momentum = (vel + C * offset) * weight
                let node_pos = vec3<f32>(
                    f32(ni) * cell_size,
                    (f32(nj) + 0.5) * cell_size,
                    (f32(nk) + 0.5) * cell_size
                );
                let offset = node_pos - pos;
                let affine_vel = c_mat_mul(id.x, offset);
                let momentum_x = (vel.x + affine_vel.x) * w;

                let idx = u_index(u32(ni), u32(nj), u32(nk));
                atomicAdd(&u_sum[idx], i32(momentum_x * SCALE));
                atomicAdd(&u_weight[idx], i32(w * SCALE));
            }
        }
    }

    // ========== V component (staggered on bottom XZ faces) ==========
    // V sample point is at (i+0.5, j, k+0.5) in cell coordinates
    let v_pos = pos / cell_size - vec3<f32>(0.5, 0.0, 0.5);
    let base_v = vec3<i32>(floor(v_pos));
    let frac_v = v_pos - vec3<f32>(base_v);

    let v_wx = array<f32, 3>(
        quadratic_bspline_1d(frac_v.x + 1.0),
        quadratic_bspline_1d(frac_v.x),
        quadratic_bspline_1d(frac_v.x - 1.0)
    );
    let v_wy = array<f32, 3>(
        quadratic_bspline_1d(frac_v.y + 1.0),
        quadratic_bspline_1d(frac_v.y),
        quadratic_bspline_1d(frac_v.y - 1.0)
    );
    let v_wz = array<f32, 3>(
        quadratic_bspline_1d(frac_v.z + 1.0),
        quadratic_bspline_1d(frac_v.z),
        quadratic_bspline_1d(frac_v.z - 1.0)
    );

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base_v.z + dk;
        if (nk < 0 || nk >= i32(depth)) { continue; }
        let wz = v_wz[dk + 1];

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base_v.y + dj;
            // V grid has height+1 rows (indices 0 to height inclusive)
            if (nj < 0 || nj > i32(height)) { continue; }
            let wy = v_wy[dj + 1];

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base_v.x + di;
                if (ni < 0 || ni >= i32(width)) { continue; }

                let w = v_wx[di + 1] * wy * wz;
                if (w <= 0.0) { continue; }

                let node_pos = vec3<f32>(
                    (f32(ni) + 0.5) * cell_size,
                    f32(nj) * cell_size,
                    (f32(nk) + 0.5) * cell_size
                );
                let offset = node_pos - pos;
                let affine_vel = c_mat_mul(id.x, offset);
                let momentum_y = (vel.y + affine_vel.y) * w;

                let idx = v_index(u32(ni), u32(nj), u32(nk));
                atomicAdd(&v_sum[idx], i32(momentum_y * SCALE));
                atomicAdd(&v_weight[idx], i32(w * SCALE));
            }
        }
    }

    // ========== W component (staggered on back XY faces) ==========
    // W sample point is at (i+0.5, j+0.5, k) in cell coordinates
    let w_pos = pos / cell_size - vec3<f32>(0.5, 0.5, 0.0);
    let base_w = vec3<i32>(floor(w_pos));
    let frac_w = w_pos - vec3<f32>(base_w);

    let w_wx = array<f32, 3>(
        quadratic_bspline_1d(frac_w.x + 1.0),
        quadratic_bspline_1d(frac_w.x),
        quadratic_bspline_1d(frac_w.x - 1.0)
    );
    let w_wy = array<f32, 3>(
        quadratic_bspline_1d(frac_w.y + 1.0),
        quadratic_bspline_1d(frac_w.y),
        quadratic_bspline_1d(frac_w.y - 1.0)
    );
    let w_wz = array<f32, 3>(
        quadratic_bspline_1d(frac_w.z + 1.0),
        quadratic_bspline_1d(frac_w.z),
        quadratic_bspline_1d(frac_w.z - 1.0)
    );

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base_w.z + dk;
        // W grid has depth+1 layers (indices 0 to depth inclusive)
        if (nk < 0 || nk > i32(depth)) { continue; }
        let wz = w_wz[dk + 1];

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base_w.y + dj;
            if (nj < 0 || nj >= i32(height)) { continue; }
            let wy = w_wy[dj + 1];

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base_w.x + di;
                if (ni < 0 || ni >= i32(width)) { continue; }

                let w = w_wx[di + 1] * wy * wz;
                if (w <= 0.0) { continue; }

                let node_pos = vec3<f32>(
                    (f32(ni) + 0.5) * cell_size,
                    (f32(nj) + 0.5) * cell_size,
                    f32(nk) * cell_size
                );
                let offset = node_pos - pos;
                let affine_vel = c_mat_mul(id.x, offset);
                let momentum_z = (vel.z + affine_vel.z) * w;

                let idx = w_index(u32(ni), u32(nj), u32(nk));
                atomicAdd(&w_sum[idx], i32(momentum_z * SCALE));
                atomicAdd(&w_weight[idx], i32(w * SCALE));
            }
        }
    }

    // ========== Count particle in its home cell (for density projection) ==========
    // This counts each particle exactly once in the cell that contains it
    let home_i = u32(clamp(i32(pos.x / cell_size), 0, i32(width) - 1));
    let home_j = u32(clamp(i32(pos.y / cell_size), 0, i32(height) - 1));
    let home_k = u32(clamp(i32(pos.z / cell_size), 0, i32(depth) - 1));
    let home_idx = cell_index(home_i, home_j, home_k);
    atomicAdd(&particle_count[home_idx], 1);
}
