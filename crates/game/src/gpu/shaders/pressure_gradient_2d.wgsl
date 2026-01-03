struct Params {
    width: u32,
    height: u32,
    inv_cell_size: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn gradient_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i > params.width || j >= params.height) {
        return;
    }
    if (i == 0u || i == params.width) {
        return;
    }

    let idx_u = j * (params.width + 1u) + i;
    let idx_r = j * params.width + i;
    let idx_l = j * params.width + (i - 1u);

    let grad = (pressure[idx_r] - pressure[idx_l]) * params.inv_cell_size;
    grid_u[idx_u] = grid_u[idx_u] - grad;
}

@compute @workgroup_size(8, 8, 1)
fn gradient_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= params.width || j > params.height) {
        return;
    }
    if (j == 0u || j == params.height) {
        return;
    }

    let idx_v = j * params.width + i;
    let idx_t = j * params.width + i;
    let idx_b = (j - 1u) * params.width + i;

    let grad = (pressure[idx_t] - pressure[idx_b]) * params.inv_cell_size;
    grid_v[idx_v] = grid_v[idx_v] - grad;
}
