struct ClearParams {
    u_len: u32,
    v_len: u32,
    p_len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: ClearParams;
@group(0) @binding(1) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> u_weight: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> v_sum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> v_weight: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> pressure_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> pressure_b: array<f32>;

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < params.u_len) {
        _ = atomicExchange(&u_sum[idx], 0);
        _ = atomicExchange(&u_weight[idx], 0);
    }
    if (idx < params.v_len) {
        _ = atomicExchange(&v_sum[idx], 0);
        _ = atomicExchange(&v_weight[idx], 0);
    }
    // Clear pressure each frame to prevent accumulation
    if (idx < params.p_len) {
        pressure_a[idx] = 0.0;
        pressure_b[idx] = 0.0;
    }
}
