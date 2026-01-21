//! Apply DEM reaction impulses to the FLIP MAC grid.

struct BridgeParams {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
    dt: f32,
    drag_coefficient: f32,
    density_water: f32,
    _pad0: f32,
    gravity: vec4<f32>,
    dem_particle_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> grid: array<f32>;
@group(0) @binding(1) var<storage, read_write> reaction: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params: BridgeParams;

const SCALE: f32 = 100000.0;

@compute @workgroup_size(8, 8, 4)
fn apply_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;
    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
    let delta = f32(atomicLoad(&reaction[idx])) / SCALE;
    if (delta != 0.0) {
        grid[idx] = grid[idx] + delta;
    }
    atomicStore(&reaction[idx], 0);
}

@compute @workgroup_size(8, 8, 4)
fn apply_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;
    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    let idx = k * params.width * (params.height + 1u) + j * params.width + i;
    let delta = f32(atomicLoad(&reaction[idx])) / SCALE;
    if (delta != 0.0) {
        grid[idx] = grid[idx] + delta;
    }
    atomicStore(&reaction[idx], 0);
}

@compute @workgroup_size(8, 8, 4)
fn apply_w(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;
    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    let idx = k * params.width * params.height + j * params.width + i;
    let delta = f32(atomicLoad(&reaction[idx])) / SCALE;
    if (delta != 0.0) {
        grid[idx] = grid[idx] + delta;
    }
    atomicStore(&reaction[idx], 0);
}
