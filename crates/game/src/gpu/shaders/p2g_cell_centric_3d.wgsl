// P2G Cell-Centric Shader (3D) - One thread per grid node, zero atomics
//
// Instead of particle-centric (one thread per particle, 81 atomics each),
// this uses cell-centric dispatch (one thread per grid node, zero atomics).
//
// Each thread:
// 1. Iterates particles in neighboring cells (using cell_offsets from counting sort)
// 2. Accumulates momentum and weight WITHOUT atomics (single writer per node)
// 3. Computes final velocity = momentum / weight
//
// Requires sorted particles and cell_offsets buffer from counting sort.

const SCALE: f32 = 100000.0;

// Kernel constants
const BSPLINE_SUPPORT_RADIUS: f32 = 1.5;  // Support range [-1.5, 1.5] for quadratic B-spline

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    include_sediment: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Sorted particle data
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> c_col0: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> c_col1: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> c_col2: array<vec3<f32>>;
@group(0) @binding(6) var<storage, read> densities: array<f32>;
// Cell offsets from counting sort (exclusive prefix sum)
@group(0) @binding(7) var<storage, read> cell_offsets: array<u32>;
// Output velocity grids (direct f32, no atomics needed)
@group(0) @binding(8) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(9) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(10) var<storage, read_write> grid_w: array<f32>;
// Particle counts per cell (for density projection)
@group(0) @binding(11) var<storage, read_write> particle_count_grid: array<i32>;
@group(0) @binding(12) var<storage, read_write> sediment_count_grid: array<i32>;

fn quadratic_bspline_1d(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.5) {
        return 0.75 - ax * ax;
    } else if (ax < BSPLINE_SUPPORT_RADIUS) {
        let t = BSPLINE_SUPPORT_RADIUS - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

fn cell_index(i: i32, j: i32, k: i32) -> u32 {
    return u32(k) * params.width * params.height + u32(j) * params.width + u32(i);
}

fn c_mat_mul(idx: u32, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c_col0[idx], offset),
        dot(c_col1[idx], offset),
        dot(c_col2[idx], offset)
    );
}

// Get particle range for a cell (start index, count)
fn get_cell_particles(ci: i32, cj: i32, ck: i32) -> vec2<u32> {
    if (ci < 0 || ci >= i32(params.width) ||
        cj < 0 || cj >= i32(params.height) ||
        ck < 0 || ck >= i32(params.depth)) {
        return vec2<u32>(0u, 0u);
    }
    let idx = cell_index(ci, cj, ck);
    let start = cell_offsets[idx];
    let end = cell_offsets[idx + 1u];
    return vec2<u32>(start, end - start);
}

// ============== U Grid (staggered at left YZ faces) ==============
// U nodes at (i, j+0.5, k+0.5), grid size (width+1) x height x depth

@compute @workgroup_size(8, 8, 4)
fn scatter_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // Bounds check for U grid: (width+1) x height x depth
    if (i > params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let cell_size = params.cell_size;

    // U node world position
    let node_pos = vec3<f32>(
        f32(i) * cell_size,
        (f32(j) + 0.5) * cell_size,
        (f32(k) + 0.5) * cell_size
    );

    var momentum_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    // Search 3x3x3 neighborhood of cells that might contain contributing particles
    // A particle at position p contributes to U nodes around base_u = floor(p/cell_size - (0, 0.5, 0.5))
    // So we need cells where particles could have base_u.x in {i-1, i, i+1}
    for (var dk: i32 = -1; dk <= 1; dk++) {
        let ck = i32(k) + dk;
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let cj = i32(j) + dj;
            for (var di: i32 = -1; di <= 1; di++) {
                // Cells that could have particles contributing to this U node
                let ci = i32(i) + di - 1;  // offset by -1 due to staggering

                let range = get_cell_particles(ci, cj, ck);
                let start = range.x;
                let count = range.y;

                // Iterate particles in this cell
                for (var p = 0u; p < count; p++) {
                    let pid = start + p;
                    if (pid >= params.particle_count) { break; }

                    let pos = positions[pid];
                    let vel = velocities[pid];
                    let density = densities[pid];

                    // Skip sediment if not included
                    if (density > 1.0 && params.include_sediment == 0u) {
                        continue;
                    }

                    // Compute weight for this particle to this U node
                    let u_pos = pos / cell_size - vec3<f32>(0.0, 0.5, 0.5);
                    let diff = vec3<f32>(f32(i), f32(j), f32(k)) - u_pos;

                    let wx = quadratic_bspline_1d(diff.x);
                    let wy = quadratic_bspline_1d(diff.y);
                    let wz = quadratic_bspline_1d(diff.z);
                    let w = wx * wy * wz;

                    if (w > 0.0) {
                        // APIC: momentum = (vel + C * offset) * weight
                        let offset = node_pos - pos;
                        let affine_vel = c_mat_mul(pid, offset);
                        let momentum_x = (vel.x + affine_vel.x) * w;

                        momentum_sum += momentum_x;
                        weight_sum += w;
                    }
                }
            }
        }
    }

    // Write final velocity (combines scatter + divide)
    let u_idx = k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
    if (weight_sum > 0.0) {
        grid_u[u_idx] = momentum_sum / weight_sum;
    } else {
        grid_u[u_idx] = 0.0;
    }
}

// ============== V Grid (staggered at bottom XZ faces) ==============
// V nodes at (i+0.5, j, k+0.5), grid size width x (height+1) x depth

@compute @workgroup_size(8, 8, 4)
fn scatter_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // Bounds check for V grid: width x (height+1) x depth
    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    let cell_size = params.cell_size;

    // V node world position
    let node_pos = vec3<f32>(
        (f32(i) + 0.5) * cell_size,
        f32(j) * cell_size,
        (f32(k) + 0.5) * cell_size
    );

    var momentum_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let ck = i32(k) + dk;
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let cj = i32(j) + dj - 1;  // offset due to staggering
            for (var di: i32 = -1; di <= 1; di++) {
                let ci = i32(i) + di;

                let range = get_cell_particles(ci, cj, ck);
                let start = range.x;
                let count = range.y;

                for (var p = 0u; p < count; p++) {
                    let pid = start + p;
                    if (pid >= params.particle_count) { break; }

                    let pos = positions[pid];
                    let vel = velocities[pid];
                    let density = densities[pid];

                    if (density > 1.0 && params.include_sediment == 0u) {
                        continue;
                    }

                    let v_pos = pos / cell_size - vec3<f32>(0.5, 0.0, 0.5);
                    let diff = vec3<f32>(f32(i), f32(j), f32(k)) - v_pos;

                    let wx = quadratic_bspline_1d(diff.x);
                    let wy = quadratic_bspline_1d(diff.y);
                    let wz = quadratic_bspline_1d(diff.z);
                    let w = wx * wy * wz;

                    if (w > 0.0) {
                        let offset = node_pos - pos;
                        let affine_vel = c_mat_mul(pid, offset);
                        let momentum_y = (vel.y + affine_vel.y) * w;

                        momentum_sum += momentum_y;
                        weight_sum += w;
                    }
                }
            }
        }
    }

    let v_idx = k * params.width * (params.height + 1u) + j * params.width + i;
    if (weight_sum > 0.0) {
        grid_v[v_idx] = momentum_sum / weight_sum;
    } else {
        grid_v[v_idx] = 0.0;
    }
}

// ============== W Grid (staggered at back XY faces) ==============
// W nodes at (i+0.5, j+0.5, k), grid size width x height x (depth+1)

@compute @workgroup_size(8, 8, 4)
fn scatter_w(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    // Bounds check for W grid: width x height x (depth+1)
    if (i >= params.width || j >= params.height || k > params.depth) {
        return;
    }

    let cell_size = params.cell_size;

    // W node world position
    let node_pos = vec3<f32>(
        (f32(i) + 0.5) * cell_size,
        (f32(j) + 0.5) * cell_size,
        f32(k) * cell_size
    );

    var momentum_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var dk: i32 = -1; dk <= 1; dk++) {
        let ck = i32(k) + dk - 1;  // offset due to staggering
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let cj = i32(j) + dj;
            for (var di: i32 = -1; di <= 1; di++) {
                let ci = i32(i) + di;

                let range = get_cell_particles(ci, cj, ck);
                let start = range.x;
                let count = range.y;

                for (var p = 0u; p < count; p++) {
                    let pid = start + p;
                    if (pid >= params.particle_count) { break; }

                    let pos = positions[pid];
                    let vel = velocities[pid];
                    let density = densities[pid];

                    if (density > 1.0 && params.include_sediment == 0u) {
                        continue;
                    }

                    let w_pos = pos / cell_size - vec3<f32>(0.5, 0.5, 0.0);
                    let diff = vec3<f32>(f32(i), f32(j), f32(k)) - w_pos;

                    let wx = quadratic_bspline_1d(diff.x);
                    let wy = quadratic_bspline_1d(diff.y);
                    let wz = quadratic_bspline_1d(diff.z);
                    let w = wx * wy * wz;

                    if (w > 0.0) {
                        let offset = node_pos - pos;
                        let affine_vel = c_mat_mul(pid, offset);
                        let momentum_z = (vel.z + affine_vel.z) * w;

                        momentum_sum += momentum_z;
                        weight_sum += w;
                    }
                }
            }
        }
    }

    let w_idx = k * params.width * params.height + j * params.width + i;
    if (weight_sum > 0.0) {
        grid_w[w_idx] = momentum_sum / weight_sum;
    } else {
        grid_w[w_idx] = 0.0;
    }
}

// ============== Particle Counting (for density projection) ==============
// One thread per cell, counts particles and sediment

@compute @workgroup_size(8, 8, 4)
fn count_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i32(i), i32(j), i32(k));
    let range = get_cell_particles(i32(i), i32(j), i32(k));
    let start = range.x;
    let count = range.y;

    var water_count: i32 = 0;
    var sediment_count: i32 = 0;

    for (var p = 0u; p < count; p++) {
        let pid = start + p;
        if (pid >= params.particle_count) { break; }

        let density = densities[pid];
        if (density > 1.0) {
            sediment_count += 1;
        } else {
            water_count += 1;
        }
    }

    particle_count_grid[idx] = water_count;
    sediment_count_grid[idx] = sediment_count;
}
