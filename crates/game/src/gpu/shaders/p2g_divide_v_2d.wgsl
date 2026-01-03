struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> v_sum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> v_weight: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> grid_v: array<f32>;

@compute @workgroup_size(256)
fn divide_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let len = params.width * (params.height + 1u);
    if (idx >= len) {
        return;
    }

    let sum = f32(atomicLoad(&v_sum[idx]));
    let w = f32(atomicLoad(&v_weight[idx]));
    if (w > 0.0) {
        grid_v[idx] = sum / w;
    } else {
        grid_v[idx] = 0.0;
    }
}
