// P2G Scatter W Shader (3D) - W component only
//
// Split from unified scatter to stay under 8 storage buffer limit.
// Each particle contributes W momentum to a 3x3x3 neighborhood.

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
@group(0) @binding(3) var<storage, read> c_col0: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> c_col1: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> c_col2: array<vec3<f32>>;
@group(0) @binding(6) var<storage, read_write> w_sum: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> w_weight: array<atomic<i32>>;

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

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn c_mat_mul(idx: u32, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c_col0[idx], offset),
        dot(c_col1[idx], offset),
        dot(c_col2[idx], offset)
    );
}

@compute @workgroup_size(256)
fn scatter_w(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let vel = velocities[id.x];
    let cell_size = params.cell_size;
    let width = params.width;
    let height = params.height;
    let depth = params.depth;

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
}
