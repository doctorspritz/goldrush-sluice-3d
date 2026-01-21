// Porosity drag (3D)
//
// Applies a velocity damping based on sediment fraction per cell.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    value: f32,  // drag_dt
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sediment_fraction: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> grid_w: array<f32>;

fn cell_index(i: i32, j: i32, k: i32) -> u32 {
    return u32(k) * params.width * params.height + u32(j) * params.width + u32(i);
}

fn u_index(i: i32, j: i32, k: i32) -> u32 {
    return u32(k) * (params.width + 1u) * params.height + u32(j) * (params.width + 1u) + u32(i);
}

fn v_index(i: i32, j: i32, k: i32) -> u32 {
    return u32(k) * params.width * (params.height + 1u) + u32(j) * params.width + u32(i);
}

fn w_index(i: i32, j: i32, k: i32) -> u32 {
    return u32(k) * params.width * params.height + u32(j) * params.width + u32(i);
}

fn sediment_at(i: i32, j: i32, k: i32) -> f32 {
    if (i < 0 || i >= i32(params.width)) { return 0.0; }
    if (j < 0 || j >= i32(params.height)) { return 0.0; }
    if (k < 0 || k >= i32(params.depth)) { return 0.0; }
    return sediment_fraction[cell_index(i, j, k)];
}

fn drag_factor(fraction: f32) -> f32 {
    let drag = clamp(fraction * params.value, 0.0, 0.95);
    return 1.0 - drag;
}

@compute @workgroup_size(8, 8, 4)
fn apply_porosity_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);

    if (i < 0 || i > i32(params.width) || j < 0 || j >= i32(params.height) || k < 0 || k >= i32(params.depth)) {
        return;
    }

    let i0 = max(i - 1, 0);
    let i1 = min(i, i32(params.width) - 1);
    let frac = 0.5 * (sediment_at(i0, j, k) + sediment_at(i1, j, k));
    let idx = u_index(i, j, k);
    grid_u[idx] *= drag_factor(frac);
}

@compute @workgroup_size(8, 8, 4)
fn apply_porosity_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);

    if (i < 0 || i >= i32(params.width) || j < 0 || j > i32(params.height) || k < 0 || k >= i32(params.depth)) {
        return;
    }

    let j0 = max(j - 1, 0);
    let j1 = min(j, i32(params.height) - 1);
    let frac = 0.5 * (sediment_at(i, j0, k) + sediment_at(i, j1, k));
    let idx = v_index(i, j, k);
    grid_v[idx] *= drag_factor(frac);
}

@compute @workgroup_size(8, 8, 4)
fn apply_porosity_w(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);

    if (i < 0 || i >= i32(params.width) || j < 0 || j >= i32(params.height) || k < 0 || k > i32(params.depth)) {
        return;
    }

    let k0 = max(k - 1, 0);
    let k1 = min(k, i32(params.depth) - 1);
    let frac = 0.5 * (sediment_at(i, j, k0) + sediment_at(i, j, k1));
    let idx = w_index(i, j, k);
    grid_w[idx] *= drag_factor(frac);
}
