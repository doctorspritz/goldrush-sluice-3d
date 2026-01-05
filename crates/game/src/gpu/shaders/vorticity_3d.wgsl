// 3D Vorticity (Curl) computation
//
// Computes omega = curl(v) from staggered MAC grid velocities.
// Outputs cell-centered vorticity components and magnitude.

struct VorticityParams {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
}

@group(0) @binding(0) var<uniform> params: VorticityParams;
@group(0) @binding(1) var<storage, read> grid_u: array<f32>;
@group(0) @binding(2) var<storage, read> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read> grid_w: array<f32>;
@group(0) @binding(4) var<storage, read> cell_type: array<u32>;
@group(0) @binding(5) var<storage, read_write> vort_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> vort_y: array<f32>;
@group(0) @binding(7) var<storage, read_write> vort_z: array<f32>;
@group(0) @binding(8) var<storage, read_write> vort_mag: array<f32>;

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

@compute @workgroup_size(8, 8, 4)
fn compute_vorticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);

    if (i < 1u || i >= params.width - 1u ||
        j < 1u || j >= params.height - 1u ||
        k < 1u || k >= params.depth - 1u) {
        vort_x[idx] = 0.0;
        vort_y[idx] = 0.0;
        vort_z[idx] = 0.0;
        vort_mag[idx] = 0.0;
        return;
    }

    if (cell_type[idx] != CELL_FLUID) {
        vort_x[idx] = 0.0;
        vort_y[idx] = 0.0;
        vort_z[idx] = 0.0;
        vort_mag[idx] = 0.0;
        return;
    }

    let inv_2h = 0.5 / params.cell_size;

    // omega_x = dw/dy - dv/dz
    let dw_dy = (grid_w[w_index(i, j + 1u, k)] - grid_w[w_index(i, j - 1u, k)]) * inv_2h;
    let dv_dz = (grid_v[v_index(i, j, k + 1u)] - grid_v[v_index(i, j, k - 1u)]) * inv_2h;
    let omega_x = dw_dy - dv_dz;

    // omega_y = du/dz - dw/dx
    let du_dz = (grid_u[u_index(i, j, k + 1u)] - grid_u[u_index(i, j, k - 1u)]) * inv_2h;
    let dw_dx = (grid_w[w_index(i + 1u, j, k)] - grid_w[w_index(i - 1u, j, k)]) * inv_2h;
    let omega_y = du_dz - dw_dx;

    // omega_z = dv/dx - du/dy
    let dv_dx = (grid_v[v_index(i + 1u, j, k)] - grid_v[v_index(i - 1u, j, k)]) * inv_2h;
    let du_dy = (grid_u[u_index(i, j + 1u, k)] - grid_u[u_index(i, j - 1u, k)]) * inv_2h;
    let omega_z = dv_dx - du_dy;

    vort_x[idx] = omega_x;
    vort_y[idx] = omega_y;
    vort_z[idx] = omega_z;
    vort_mag[idx] = sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);
}
