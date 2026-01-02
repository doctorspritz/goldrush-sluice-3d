// Grid Velocity Clamping Shader (3D)
//
// Clamps grid velocities to MAX_GRID_VEL to maintain CFL stability.
// This prevents the pressure solver from receiving extreme input velocities
// that could cause numerical instability or non-convergence.
//
// Must be run BEFORE the pressure solve, matching CPU behavior.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    max_vel: f32,  // Maximum velocity magnitude (10.0 matches CPU)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_w: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn clamp_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // U grid: (width+1) x height x depth
    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
    grid_u[idx] = clamp(grid_u[idx], -params.max_vel, params.max_vel);
}

@compute @workgroup_size(8, 8, 4)
fn clamp_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // V grid: width x (height+1) x depth
    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    let idx = k * params.width * (params.height + 1u) + j * params.width + i;
    grid_v[idx] = clamp(grid_v[idx], -params.max_vel, params.max_vel);
}

@compute @workgroup_size(8, 8, 4)
fn clamp_w(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // W grid: width x height x (depth+1)
    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    let idx = k * params.width * params.height + j * params.width + i;
    grid_w[idx] = clamp(grid_w[idx], -params.max_vel, params.max_vel);
}
