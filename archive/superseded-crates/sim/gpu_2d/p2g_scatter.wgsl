// P2G Scatter Shader - Particle-to-Grid transfer using fixed-point atomics
//
// Each particle contributes momentum to a 3x3 neighborhood of grid nodes using
// quadratic B-spline weights (APIC transfer). WebGPU only supports atomicAdd
// for i32, so we encode f32 * SCALE → i32, then decode after accumulation.
//
// Grid layout (MAC staggered):
// - U velocities: stored at left edges, (width+1) x height
// - V velocities: stored at bottom edges, width x (height+1)

// Fixed-point scale: 10^6 gives ±2147 range with 6 decimal digits precision
const SCALE: f32 = 1000000.0;

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> c_matrices: array<vec4<f32>>; // mat2x2 as vec4
@group(0) @binding(4) var<storage, read> materials: array<u32>;
@group(0) @binding(5) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> u_weight: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> v_sum: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> v_weight: array<atomic<i32>>;

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

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let vel = velocities[id.x];
    let c_vec = c_matrices[id.x];
    // Reconstruct mat2x2 from vec4 (column-major: [c0.x, c0.y, c1.x, c1.y])
    let c_mat = mat2x2<f32>(
        vec2<f32>(c_vec.x, c_vec.y),  // column 0
        vec2<f32>(c_vec.z, c_vec.w)   // column 1
    );

    // Skip sediment (materials[id.x] != 0)
    // Note: Currently all uploaded particles are water, but check anyway
    if (materials[id.x] != 0u) {
        return;
    }

    let cell_size = params.cell_size;
    let width = params.width;
    let height = params.height;

    // ========== U component (staggered on left edges) ==========
    // U sample point is at (i, j+0.5) in cell coordinates
    let u_pos = pos / cell_size - vec2<f32>(0.0, 0.5);
    let base_i_u = i32(floor(u_pos.x));
    let base_j_u = i32(floor(u_pos.y));
    let fx_u = u_pos.x - f32(base_i_u);
    let fy_u = u_pos.y - f32(base_j_u);

    // Precompute 1D weights for -1, 0, +1 offsets
    let u_wx = array<f32, 3>(
        quadratic_bspline_1d(fx_u + 1.0),
        quadratic_bspline_1d(fx_u),
        quadratic_bspline_1d(fx_u - 1.0)
    );
    let u_wy = array<f32, 3>(
        quadratic_bspline_1d(fy_u + 1.0),
        quadratic_bspline_1d(fy_u),
        quadratic_bspline_1d(fy_u - 1.0)
    );

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base_j_u + dj;
        if (nj < 0 || nj >= i32(height)) {
            continue;
        }
        let wy = u_wy[dj + 1];

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_u + di;
            // U grid has width+1 columns (indices 0 to width inclusive)
            if (ni < 0 || ni > i32(width)) {
                continue;
            }
            let w = u_wx[di + 1] * wy;
            if (w <= 0.0) {
                continue;
            }

            // APIC: momentum = (vel + C * offset) * weight
            // Grid node position in world coords
            let node_x = f32(ni) * cell_size;
            let node_y = (f32(nj) + 0.5) * cell_size;
            let offset = vec2<f32>(node_x - pos.x, node_y - pos.y);
            let affine_vel = c_mat * offset;
            let momentum_x = (vel.x + affine_vel.x) * w;

            // Buffer index: U grid is (width+1) x height
            let idx = u32(nj) * (width + 1u) + u32(ni);
            atomicAdd(&u_sum[idx], i32(momentum_x * SCALE));
            atomicAdd(&u_weight[idx], i32(w * SCALE));
        }
    }

    // ========== V component (staggered on bottom edges) ==========
    // V sample point is at (i+0.5, j) in cell coordinates
    let v_pos = pos / cell_size - vec2<f32>(0.5, 0.0);
    let base_i_v = i32(floor(v_pos.x));
    let base_j_v = i32(floor(v_pos.y));
    let fx_v = v_pos.x - f32(base_i_v);
    let fy_v = v_pos.y - f32(base_j_v);

    let v_wx = array<f32, 3>(
        quadratic_bspline_1d(fx_v + 1.0),
        quadratic_bspline_1d(fx_v),
        quadratic_bspline_1d(fx_v - 1.0)
    );
    let v_wy = array<f32, 3>(
        quadratic_bspline_1d(fy_v + 1.0),
        quadratic_bspline_1d(fy_v),
        quadratic_bspline_1d(fy_v - 1.0)
    );

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base_j_v + dj;
        // V grid has height+1 rows (indices 0 to height inclusive)
        if (nj < 0 || nj > i32(height)) {
            continue;
        }
        let wy = v_wy[dj + 1];

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_v + di;
            if (ni < 0 || ni >= i32(width)) {
                continue;
            }
            let w = v_wx[di + 1] * wy;
            if (w <= 0.0) {
                continue;
            }

            // Grid node position in world coords
            let node_x = (f32(ni) + 0.5) * cell_size;
            let node_y = f32(nj) * cell_size;
            let offset = vec2<f32>(node_x - pos.x, node_y - pos.y);
            let affine_vel = c_mat * offset;
            let momentum_y = (vel.y + affine_vel.y) * w;

            // Buffer index: V grid is width x (height+1)
            let idx = u32(nj) * width + u32(ni);
            atomicAdd(&v_sum[idx], i32(momentum_y * SCALE));
            atomicAdd(&v_weight[idx], i32(w * SCALE));
        }
    }
}
