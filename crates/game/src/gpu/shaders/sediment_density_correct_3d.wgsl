// Sediment Density Position Correction Shader (3D)
//
// Applies grid-based position corrections to sediment particles only.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    dt: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> position_delta_x: array<f32>;
@group(0) @binding(2) var<storage, read> position_delta_y: array<f32>;
@group(0) @binding(3) var<storage, read> position_delta_z: array<f32>;
@group(0) @binding(4) var<storage, read> cell_type: array<u32>;
@group(0) @binding(5) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> densities: array<f32>;
@group(0) @binding(7) var<storage, read_write> velocities: array<vec4<f32>>;

const SEDIMENT_DENSITY_THRESHOLD: f32 = 1.0;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn sample_delta_x(i: i32, j: i32, k: i32) -> f32 {
    if (i < 0 || i >= i32(params.width)) { return 0.0; }
    if (j < 0 || j >= i32(params.height)) { return 0.0; }
    if (k < 0 || k >= i32(params.depth)) { return 0.0; }
    return position_delta_x[cell_index(u32(i), u32(j), u32(k))];
}

fn sample_delta_y(i: i32, j: i32, k: i32) -> f32 {
    if (i < 0 || i >= i32(params.width)) { return 0.0; }
    if (j < 0 || j >= i32(params.height)) { return 0.0; }
    if (k < 0 || k >= i32(params.depth)) { return 0.0; }
    return position_delta_y[cell_index(u32(i), u32(j), u32(k))];
}

fn sample_delta_z(i: i32, j: i32, k: i32) -> f32 {
    if (i < 0 || i >= i32(params.width)) { return 0.0; }
    if (j < 0 || j >= i32(params.height)) { return 0.0; }
    if (k < 0 || k >= i32(params.depth)) { return 0.0; }
    return position_delta_z[cell_index(u32(i), u32(j), u32(k))];
}

fn trilinear_delta_x(pos: vec3<f32>) -> f32 {
    let sample_pos = max(vec3<f32>(0.0), pos - vec3<f32>(0.5, 0.0, 0.0));

    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    let v000 = sample_delta_x(base_x, base_y, base_z);
    let v100 = sample_delta_x(base_x + 1, base_y, base_z);
    let v010 = sample_delta_x(base_x, base_y + 1, base_z);
    let v110 = sample_delta_x(base_x + 1, base_y + 1, base_z);
    let v001 = sample_delta_x(base_x, base_y, base_z + 1);
    let v101 = sample_delta_x(base_x + 1, base_y, base_z + 1);
    let v011 = sample_delta_x(base_x, base_y + 1, base_z + 1);
    let v111 = sample_delta_x(base_x + 1, base_y + 1, base_z + 1);

    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

fn trilinear_delta_y(pos: vec3<f32>) -> f32 {
    let sample_pos = max(vec3<f32>(0.0), pos - vec3<f32>(0.0, 0.5, 0.0));

    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    let v000 = sample_delta_y(base_x, base_y, base_z);
    let v100 = sample_delta_y(base_x + 1, base_y, base_z);
    let v010 = sample_delta_y(base_x, base_y + 1, base_z);
    let v110 = sample_delta_y(base_x + 1, base_y + 1, base_z);
    let v001 = sample_delta_y(base_x, base_y, base_z + 1);
    let v101 = sample_delta_y(base_x + 1, base_y, base_z + 1);
    let v011 = sample_delta_y(base_x, base_y + 1, base_z + 1);
    let v111 = sample_delta_y(base_x + 1, base_y + 1, base_z + 1);

    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

fn trilinear_delta_z(pos: vec3<f32>) -> f32 {
    let sample_pos = max(vec3<f32>(0.0), pos - vec3<f32>(0.0, 0.0, 0.5));

    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    let v000 = sample_delta_z(base_x, base_y, base_z);
    let v100 = sample_delta_z(base_x + 1, base_y, base_z);
    let v010 = sample_delta_z(base_x, base_y + 1, base_z);
    let v110 = sample_delta_z(base_x + 1, base_y + 1, base_z);
    let v001 = sample_delta_z(base_x, base_y, base_z + 1);
    let v101 = sample_delta_z(base_x + 1, base_y, base_z + 1);
    let v011 = sample_delta_z(base_x, base_y + 1, base_z + 1);
    let v111 = sample_delta_z(base_x + 1, base_y + 1, base_z + 1);

    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

@compute @workgroup_size(256)
fn correct_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let pid = id.x;

    if (pid >= params.particle_count) {
        return;
    }

    if (densities[pid] <= SEDIMENT_DENSITY_THRESHOLD) {
        return;
    }

    let pos = positions[pid].xyz;
    let vel = velocities[pid].xyz;
    let cell_size = params.cell_size;
    let grid_pos = pos / cell_size;

    let delta_x = trilinear_delta_x(grid_pos);
    let delta_y = trilinear_delta_y(grid_pos);
    let delta_z = trilinear_delta_z(grid_pos);

    let correction_scale: f32 = 15.0;
    let delta = vec3<f32>(-delta_x, -delta_y, -delta_z) * correction_scale;

    // Update position
    let corrected_pos = pos + delta;
    positions[pid] = vec4<f32>(corrected_pos, 0.0);

    // Update velocity to reflect the collision force
    // This prevents particles from overlapping again next frame
    // Use a damped velocity correction (0.2 factor) to avoid overly rigid behavior
    let velocity_correction = (delta / params.dt) * 0.2;
    let corrected_vel = vel + velocity_correction;
    velocities[pid] = vec4<f32>(corrected_vel, 0.0);
}
