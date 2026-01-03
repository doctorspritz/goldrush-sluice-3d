struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_v: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn enforce_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i > params.width || j >= params.height) {
        return;
    }

    if (i == 0u || i == params.width) {
        let idx = j * (params.width + 1u) + i;
        grid_u[idx] = 0.0;
    }
}

@compute @workgroup_size(8, 8, 1)
fn enforce_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= params.width || j > params.height) {
        return;
    }

    if (j == 0u || j == params.height) {
        let idx = j * params.width + i;
        grid_v[idx] = 0.0;
    }
}
