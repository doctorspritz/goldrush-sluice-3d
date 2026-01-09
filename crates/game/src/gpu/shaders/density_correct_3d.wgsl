// Density Position Correction Shader (3D) - Blub-Style Trilinear Sampling
//
// Particles sample position deltas from grid using trilinear interpolation
// and apply corrections in-place on the positions buffer.
// Uses offset positions for proper staggered grid sampling (blub approach):
// - X delta sampled at pos - (0.5, 0, 0)
// - Y delta sampled at pos - (0, 0.5, 0)
// - Z delta sampled at pos - (0, 0, 0.5)
//
// Based on: blub/density_projection_correct_particles.comp

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> position_delta_x: array<f32>;
@group(0) @binding(2) var<storage, read> position_delta_y: array<f32>;
@group(0) @binding(3) var<storage, read> position_delta_z: array<f32>;
@group(0) @binding(4) var<storage, read> cell_type: array<u32>;
// Positions stored as vec4 (padded from vec3) - must match G2P buffer layout
@group(0) @binding(5) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> densities: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;
const SEDIMENT_DENSITY_THRESHOLD: f32 = 1.0;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_AIR; }
    if (j < 0 || j >= i32(params.height)) { return CELL_AIR; }
    if (k < 0 || k >= i32(params.depth)) { return CELL_AIR; }
    return cell_type[cell_index(u32(i), u32(j), u32(k))];
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

// Trilinear interpolation for position delta X (offset by -0.5 in X)
fn trilinear_delta_x(pos: vec3<f32>) -> f32 {
    // Sample position with offset for staggered grid
    let sample_pos = max(vec3<f32>(0.0), pos - vec3<f32>(0.5, 0.0, 0.0));

    // Get base cell coordinates
    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    // Get fractional coordinates for interpolation weights
    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    // Sample 8 corners
    let v000 = sample_delta_x(base_x, base_y, base_z);
    let v100 = sample_delta_x(base_x + 1, base_y, base_z);
    let v010 = sample_delta_x(base_x, base_y + 1, base_z);
    let v110 = sample_delta_x(base_x + 1, base_y + 1, base_z);
    let v001 = sample_delta_x(base_x, base_y, base_z + 1);
    let v101 = sample_delta_x(base_x + 1, base_y, base_z + 1);
    let v011 = sample_delta_x(base_x, base_y + 1, base_z + 1);
    let v111 = sample_delta_x(base_x + 1, base_y + 1, base_z + 1);

    // Trilinear interpolation
    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

// Trilinear interpolation for position delta Y (offset by -0.5 in Y)
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

// Trilinear interpolation for position delta Z (offset by -0.5 in Z)
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

    // Apply density correction to ALL particles (including sediment)
    // so they get pushed apart like water does.
    // Previously skipped sediment: if (densities[pid] > SEDIMENT_DENSITY_THRESHOLD) { return; }

    let pos = positions[pid].xyz;
    let cell_size = params.cell_size;

    // Convert world position to grid coordinates
    let grid_pos = pos / cell_size;

    // Sample position deltas using trilinear interpolation with offset
    let delta_x = trilinear_delta_x(grid_pos);
    let delta_y = trilinear_delta_y(grid_pos);
    let delta_z = trilinear_delta_z(grid_pos);

    // Grid position deltas point from center cell toward neighbor
    // With forward difference: delta = (p_neighbor - p_center) * dt
    // Crowded center has high p, so delta is NEGATIVE toward sparse neighbor
    // We want particles to move TOWARD lower pressure (sparse), so we NEGATE the delta
    let correction_scale: f32 = 15.0;  // Empirical scaling factor (increased for stronger volume conservation)
    let delta = vec3<f32>(-delta_x, -delta_y, -delta_z) * correction_scale;

    let corrected = pos + delta;
    positions[pid] = vec4<f32>(corrected, 0.0);
}
