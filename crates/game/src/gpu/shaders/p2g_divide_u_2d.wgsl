struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> u_sum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> u_weight: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> grid_u: array<f32>;

@compute @workgroup_size(256)
fn divide_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let len = (params.width + 1u) * params.height;
    if (idx >= len) {
        return;
    }

    let sum = f32(atomicLoad(&u_sum[idx]));
    let w = f32(atomicLoad(&u_weight[idx]));
    if (w > 0.0) {
        grid_u[idx] = sum / w;
    } else {
        grid_u[idx] = 0.0;
    }
}
