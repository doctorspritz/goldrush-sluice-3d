// 3D Vorticity confinement force application
//
// Applies F = epsilon * h * (N x omega) to MAC grid velocities.
// N = grad(|omega|) / |grad(|omega|)|, computed from vorticity magnitude.

struct ConfineParams {
    width: u32,
    height: u32,
    depth: u32,
    epsilon_h_dt: f32,
}

@group(0) @binding(0) var<uniform> params: ConfineParams;
@group(0) @binding(1) var<storage, read> vort_x: array<f32>;
@group(0) @binding(2) var<storage, read> vort_y: array<f32>;
@group(0) @binding(3) var<storage, read> vort_z: array<f32>;
@group(0) @binding(4) var<storage, read> vort_mag: array<f32>;
@group(0) @binding(5) var<storage, read> cell_type: array<u32>;
@group(0) @binding(6) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(7) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(8) var<storage, read_write> grid_w: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * (params.width + 1u) + k * (params.width + 1u) * params.height;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * (params.height + 1u);
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

fn compute_force(i: u32, j: u32, k: u32) -> vec3<f32> {
    if (i < 2u || i >= params.width - 2u ||
        j < 2u || j >= params.height - 2u ||
        k < 2u || k >= params.depth - 2u) {
        return vec3(0.0);
    }

    let idx = cell_index(i, j, k);
    if (cell_type[idx] != CELL_FLUID) {
        return vec3(0.0);
    }

    let has_air = cell_type[cell_index(i - 1u, j, k)] == CELL_AIR ||
        cell_type[cell_index(i + 1u, j, k)] == CELL_AIR ||
        cell_type[cell_index(i, j - 1u, k)] == CELL_AIR ||
        cell_type[cell_index(i, j + 1u, k)] == CELL_AIR ||
        cell_type[cell_index(i, j, k - 1u)] == CELL_AIR ||
        cell_type[cell_index(i, j, k + 1u)] == CELL_AIR;
    if (has_air) {
        return vec3(0.0);
    }

    let grad_x = (vort_mag[cell_index(i + 1u, j, k)] - vort_mag[cell_index(i - 1u, j, k)]) * 0.5;
    let grad_y = (vort_mag[cell_index(i, j + 1u, k)] - vort_mag[cell_index(i, j - 1u, k)]) * 0.5;
    let grad_z = (vort_mag[cell_index(i, j, k + 1u)] - vort_mag[cell_index(i, j, k - 1u)]) * 0.5;

    let grad_len = sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z) + 1e-5;
    let nx = grad_x / grad_len;
    let ny = grad_y / grad_len;
    let nz = grad_z / grad_len;

    let wx = vort_x[idx];
    let wy = vort_y[idx];
    let wz = vort_z[idx];

    let fx = (ny * wz - nz * wy) * params.epsilon_h_dt;
    let fy = (nz * wx - nx * wz) * params.epsilon_h_dt;
    let fz = (nx * wy - ny * wx) * params.epsilon_h_dt;

    return vec3(fx, fy, fz);
}

@compute @workgroup_size(8, 8, 4)
fn apply_confinement_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    if (i == 0u || i == params.width) {
        return;
    }

    let left_force = compute_force(i - 1u, j, k);
    let right_force = compute_force(i, j, k);
    let fx = 0.5 * (left_force.x + right_force.x);

    let idx = u_index(i, j, k);
    grid_u[idx] += fx;
}

@compute @workgroup_size(8, 8, 4)
fn apply_confinement_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    if (j == 0u || j == params.height) {
        return;
    }

    let bottom_force = compute_force(i, j - 1u, k);
    let top_force = compute_force(i, j, k);
    let fy = 0.5 * (bottom_force.y + top_force.y);

    let idx = v_index(i, j, k);
    grid_v[idx] += fy;
}

@compute @workgroup_size(8, 8, 4)
fn apply_confinement_w(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    if (k == 0u || k == params.depth) {
        return;
    }

    let back_force = compute_force(i, j, k - 1u);
    let front_force = compute_force(i, j, k);
    let fz = 0.5 * (back_force.z + front_force.z);

    let idx = w_index(i, j, k);
    grid_w[idx] += fz;
}
