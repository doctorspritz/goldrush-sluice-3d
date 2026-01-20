// Sediment fraction computation (3D)
//
// Converts per-cell sediment counts into a normalized fraction [0,1].

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    value: f32,  // rest_particles
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sediment_count: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> sediment_fraction: array<f32>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

@compute @workgroup_size(8, 8, 4)
fn compute_sediment_fraction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    let count = f32(atomicLoad(&sediment_count[idx]));
    let denom = max(params.value, 1.0);
    sediment_fraction[idx] = clamp(count / denom, 0.0, 1.0);
}
