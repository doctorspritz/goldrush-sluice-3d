// Divergence Computation Shader (3D)
//
// Computes divergence of MAC staggered velocity field:
//   div = (u[i+1,j,k] - u[i,j,k] + v[i,j+1,k] - v[i,j,k] + w[i,j,k+1] - w[i,j,k]) / dx
//
// Only fluid cells get non-zero divergence. The result is used as the RHS
// of the pressure Poisson equation.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    inv_cell_size: f32,  // 1.0 / cell_size
    // Bitmask for open boundaries (not used by divergence, but needed for struct layout)
    open_boundaries: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid_u: array<f32>;
@group(0) @binding(2) var<storage, read> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read> grid_w: array<f32>;
@group(0) @binding(4) var<storage, read> cell_type: array<u32>;
@group(0) @binding(5) var<storage, read_write> divergence: array<f32>;

const CELL_FLUID: u32 = 1u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn u_index(i: u32, j: u32, k: u32) -> u32 {
    // U grid: (width+1) x height x depth
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    // V grid: width x (height+1) x depth
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    // W grid: width x height x (depth+1)
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(8, 8, 4)
fn compute_divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);

    // Only compute divergence for fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        divergence[idx] = 0.0;
        return;
    }

    // Sample velocities from staggered grid
    // U: left face at [i], right face at [i+1]
    let u_left = grid_u[u_index(i, j, k)];
    let u_right = grid_u[u_index(i + 1u, j, k)];

    // V: bottom face at [j], top face at [j+1]
    let v_bottom = grid_v[v_index(i, j, k)];
    let v_top = grid_v[v_index(i, j + 1u, k)];

    // W: back face at [k], front face at [k+1]
    let w_back = grid_w[w_index(i, j, k)];
    let w_front = grid_w[w_index(i, j, k + 1u)];

    // Divergence = (du/dx + dv/dy + dw/dz)
    let div = (u_right - u_left + v_top - v_bottom + w_front - w_back) * params.inv_cell_size;

    divergence[idx] = div;
}
