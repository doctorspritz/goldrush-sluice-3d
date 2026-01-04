struct Params {
    width: u32,
    height: u32,
    inv_cell_size: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid_u: array<f32>;
@group(0) @binding(2) var<storage, read> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn compute_divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = j * params.width + i;
    let u_idx = j * (params.width + 1u) + i;
    let v_idx = j * params.width + i;

    // Get face velocities, but respect solid boundaries
    // Left wall (i=0): u_left = 0 (no flow through solid)
    // Right wall (i=width-1): u_right = 0
    // Bottom wall (j=0): v_bottom = 0
    // Top is open, so no BC override

    var u_left = grid_u[u_idx];
    var u_right = grid_u[u_idx + 1u];
    var v_bottom = grid_v[v_idx];
    var v_top = grid_v[v_idx + params.width];

    // Apply solid wall boundary conditions
    if (i == 0u) {
        u_left = 0.0;
    }
    if (i == params.width - 1u) {
        u_right = 0.0;
    }
    if (j == 0u) {
        v_bottom = 0.0;
    }
    // Top is solid (matches BC enforcement which zeros v at j=height)
    if (j == params.height - 1u) {
        v_top = 0.0;
    }

    divergence[idx] = (u_right - u_left + v_top - v_bottom) * params.inv_cell_size;
}
