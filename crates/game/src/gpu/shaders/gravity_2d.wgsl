struct Params {
    width: u32,
    height: u32,
    gravity_dt: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> grid_v: array<f32>;

@compute @workgroup_size(256)
fn apply_gravity(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let len = params.width * (params.height + 1u);
    if (idx >= len) {
        return;
    }
    grid_v[idx] = grid_v[idx] + params.gravity_dt;
}
