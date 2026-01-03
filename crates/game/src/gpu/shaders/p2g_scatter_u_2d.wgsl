// P2G Scatter U Shader (2D)
// Each particle contributes U momentum to a 3x3 neighborhood.

const SCALE: f32 = 1000000.0;

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> u_weight: array<atomic<i32>>;

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

fn u_index(i: u32, j: u32) -> u32 {
    return j * (params.width + 1u) + i;
}

@compute @workgroup_size(256)
fn scatter_u(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x].xy;
    let vel = velocities[id.x].xy;

    let cell_size = params.cell_size;
    let width = i32(params.width);
    let height = i32(params.height);

    let u_pos = pos / cell_size - vec2<f32>(0.0, 0.5);
    let base = vec2<i32>(floor(u_pos));
    let frac = u_pos - vec2<f32>(base);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base.y + dj;
        if (nj < 0 || nj >= height) { continue; }
        let wy = quadratic_bspline_1d(frac.y - f32(dj));

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base.x + di;
            if (ni < 0 || ni > width) { continue; }

            let wx = quadratic_bspline_1d(frac.x - f32(di));
            let w = wx * wy;
            if (w <= 0.0) { continue; }

            let idx = u_index(u32(ni), u32(nj));
            let momentum = vel.x * w;
            atomicAdd(&u_sum[idx], i32(momentum * SCALE));
            atomicAdd(&u_weight[idx], i32(w * SCALE));
        }
    }
}
