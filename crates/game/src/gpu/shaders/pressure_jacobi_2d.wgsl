struct Params {
    width: u32,
    height: u32,
    alpha: f32,
    rbeta: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(2) var<storage, read> divergence: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure_out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn pressure_jacobi(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = j * params.width + i;
    let p_center = pressure_in[idx];

    var p_left = p_center;
    var p_right = p_center;
    var p_bottom = p_center;
    var p_top = p_center;

    if (i > 0u) {
        p_left = pressure_in[idx - 1u];
    }
    if (i + 1u < params.width) {
        p_right = pressure_in[idx + 1u];
    }
    if (j > 0u) {
        p_bottom = pressure_in[idx - params.width];
    }
    if (j + 1u < params.height) {
        p_top = pressure_in[idx + params.width];
    }

    let div = divergence[idx];
    pressure_out[idx] = (p_left + p_right + p_bottom + p_top + params.alpha * div) * params.rbeta;
}
