// P2G Scatter Shader (3D) - Tile-based shared memory aggregation
//
// Optimization: Instead of 81 atomic ops per particle to global memory,
// accumulate contributions in workgroup shared memory first, then flush once.
// Requires sorted particles for spatial coherence within workgroups.
//
// Tile: 4×4×4 cells, workgroup: 256 threads
// Expected ~80× reduction in atomic contention vs naive scatter.

const SCALE: f32 = 100000.0;

// Tile dimensions (cells)
const TILE_X: u32 = 4u;
const TILE_Y: u32 = 4u;
const TILE_Z: u32 = 4u;

// Staggered grid sizes within tile
const TILE_U_X: u32 = TILE_X + 1u;  // 5
const TILE_U_Y: u32 = TILE_Y;       // 4
const TILE_U_Z: u32 = TILE_Z;       // 4
const TILE_U_SIZE: u32 = TILE_U_X * TILE_U_Y * TILE_U_Z;  // 80

const TILE_V_X: u32 = TILE_X;       // 4
const TILE_V_Y: u32 = TILE_Y + 1u;  // 5
const TILE_V_Z: u32 = TILE_Z;       // 4
const TILE_V_SIZE: u32 = TILE_V_X * TILE_V_Y * TILE_V_Z;  // 80

const TILE_W_X: u32 = TILE_X;       // 4
const TILE_W_Y: u32 = TILE_Y;       // 4
const TILE_W_Z: u32 = TILE_Z + 1u;  // 5
const TILE_W_SIZE: u32 = TILE_W_X * TILE_W_Y * TILE_W_Z;  // 80

const TILE_CELLS: u32 = TILE_X * TILE_Y * TILE_Z;  // 64

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
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> c_col0: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> c_col1: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> c_col2: array<vec3<f32>>;
@group(0) @binding(6) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> u_weight: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> v_sum: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> v_weight: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> w_sum: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> w_weight: array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> particle_count: array<atomic<i32>>;
@group(0) @binding(13) var<storage, read> densities: array<f32>;
@group(0) @binding(14) var<storage, read_write> sediment_count: array<atomic<i32>>;

// Shared memory for tile accumulation (no atomics needed - each thread accumulates independently)
var<workgroup> tile_u_sum: array<atomic<i32>, 80>;      // TILE_U_SIZE
var<workgroup> tile_u_weight: array<atomic<i32>, 80>;
var<workgroup> tile_v_sum: array<atomic<i32>, 80>;      // TILE_V_SIZE
var<workgroup> tile_v_weight: array<atomic<i32>, 80>;
var<workgroup> tile_w_sum: array<atomic<i32>, 80>;      // TILE_W_SIZE
var<workgroup> tile_w_weight: array<atomic<i32>, 80>;
var<workgroup> tile_particle_count: array<atomic<i32>, 64>;  // TILE_CELLS
var<workgroup> tile_sediment_count: array<atomic<i32>, 64>;

// Tile origin in cell coordinates (set by thread 0)
var<workgroup> tile_origin: vec3<i32>;

fn quadratic_bspline_1d(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.5) {
        return 0.75 - ax * ax;
    } else if (ax < 1.5) {
        let t = 1.5 - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

// Global grid index functions
fn u_index_global(i: u32, j: u32, k: u32) -> u32 {
    return k * (params.width + 1u) * params.height + j * (params.width + 1u) + i;
}

fn v_index_global(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

fn w_index_global(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn cell_index_global(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

// Tile-local index functions (relative to tile_origin)
fn u_index_tile(li: u32, lj: u32, lk: u32) -> u32 {
    return lk * TILE_U_X * TILE_U_Y + lj * TILE_U_X + li;
}

fn v_index_tile(li: u32, lj: u32, lk: u32) -> u32 {
    return lk * TILE_V_X * TILE_V_Y + lj * TILE_V_X + li;
}

fn w_index_tile(li: u32, lj: u32, lk: u32) -> u32 {
    return lk * TILE_W_X * TILE_W_Y + lj * TILE_W_X + li;
}

fn cell_index_tile(li: u32, lj: u32, lk: u32) -> u32 {
    return lk * TILE_X * TILE_Y + lj * TILE_X + li;
}

fn c_mat_mul(idx: u32, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c_col0[idx], offset),
        dot(c_col1[idx], offset),
        dot(c_col2[idx], offset)
    );
}

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let pid = global_id.x;

    // ========== Phase 0: Initialize shared memory ==========
    // Each thread clears a few elements
    if (tid < 80u) {
        atomicStore(&tile_u_sum[tid], 0);
        atomicStore(&tile_u_weight[tid], 0);
        atomicStore(&tile_v_sum[tid], 0);
        atomicStore(&tile_v_weight[tid], 0);
        atomicStore(&tile_w_sum[tid], 0);
        atomicStore(&tile_w_weight[tid], 0);
    }
    if (tid < 64u) {
        atomicStore(&tile_particle_count[tid], 0);
        atomicStore(&tile_sediment_count[tid], 0);
    }

    // Thread 0 determines tile origin from first valid particle in workgroup
    if (tid == 0u) {
        // Find first valid particle to set tile origin
        let first_pid = workgroup_id.x * 256u;
        if (first_pid < params.particle_count) {
            let pos = positions[first_pid];
            // Tile origin: align to tile boundaries
            let cell_i = i32(pos.x / params.cell_size);
            let cell_j = i32(pos.y / params.cell_size);
            let cell_k = i32(pos.z / params.cell_size);
            // Align to tile grid (round down to nearest tile)
            tile_origin = vec3<i32>(
                (cell_i / i32(TILE_X)) * i32(TILE_X),
                (cell_j / i32(TILE_Y)) * i32(TILE_Y),
                (cell_k / i32(TILE_Z)) * i32(TILE_Z)
            );
        } else {
            tile_origin = vec3<i32>(0, 0, 0);
        }
    }

    workgroupBarrier();

    // ========== Phase 1: Accumulate particle contributions ==========
    if (pid < params.particle_count) {
        let pos = positions[pid];
        let vel = velocities[pid];
        let density = densities[pid];

        // Skip sediment if not included
        if (density > 1.0 && params.include_sediment == 0u) {
            // Still need to participate in barriers below
        } else {
            let cell_size = params.cell_size;
            let width = params.width;
            let height = params.height;
            let depth = params.depth;

            // Home cell for particle counting
            let home_i = clamp(i32(pos.x / cell_size), 0, i32(width) - 1);
            let home_j = clamp(i32(pos.y / cell_size), 0, i32(height) - 1);
            let home_k = clamp(i32(pos.z / cell_size), 0, i32(depth) - 1);

            // Local coordinates relative to tile
            let local_home_i = home_i - tile_origin.x;
            let local_home_j = home_j - tile_origin.y;
            let local_home_k = home_k - tile_origin.z;

            // Particle/sediment counting
            if (density > 1.0) {
                // Sediment particle
                if (local_home_i >= 0 && local_home_i < i32(TILE_X) &&
                    local_home_j >= 0 && local_home_j < i32(TILE_Y) &&
                    local_home_k >= 0 && local_home_k < i32(TILE_Z)) {
                    let idx = cell_index_tile(u32(local_home_i), u32(local_home_j), u32(local_home_k));
                    atomicAdd(&tile_sediment_count[idx], 1);
                } else {
                    let idx = cell_index_global(u32(home_i), u32(home_j), u32(home_k));
                    atomicAdd(&sediment_count[idx], 1);
                }
            } else {
                // Water particle
                if (local_home_i >= 0 && local_home_i < i32(TILE_X) &&
                    local_home_j >= 0 && local_home_j < i32(TILE_Y) &&
                    local_home_k >= 0 && local_home_k < i32(TILE_Z)) {
                    let idx = cell_index_tile(u32(local_home_i), u32(local_home_j), u32(local_home_k));
                    atomicAdd(&tile_particle_count[idx], 1);
                } else {
                    let idx = cell_index_global(u32(home_i), u32(home_j), u32(home_k));
                    atomicAdd(&particle_count[idx], 1);
                }
            }

            // ========== U component ==========
            let u_pos = pos / cell_size - vec3<f32>(0.0, 0.5, 0.5);
            let base_u = vec3<i32>(floor(u_pos));
            let frac_u = u_pos - vec3<f32>(base_u);

            let u_wx = array<f32, 3>(
                quadratic_bspline_1d(frac_u.x + 1.0),
                quadratic_bspline_1d(frac_u.x),
                quadratic_bspline_1d(frac_u.x - 1.0)
            );
            let u_wy = array<f32, 3>(
                quadratic_bspline_1d(frac_u.y + 1.0),
                quadratic_bspline_1d(frac_u.y),
                quadratic_bspline_1d(frac_u.y - 1.0)
            );
            let u_wz = array<f32, 3>(
                quadratic_bspline_1d(frac_u.z + 1.0),
                quadratic_bspline_1d(frac_u.z),
                quadratic_bspline_1d(frac_u.z - 1.0)
            );

            for (var dk: i32 = -1; dk <= 1; dk++) {
                let nk = base_u.z + dk;
                if (nk < 0 || nk >= i32(depth)) { continue; }
                let wz = u_wz[dk + 1];

                for (var dj: i32 = -1; dj <= 1; dj++) {
                    let nj = base_u.y + dj;
                    if (nj < 0 || nj >= i32(height)) { continue; }
                    let wy = u_wy[dj + 1];

                    for (var di: i32 = -1; di <= 1; di++) {
                        let ni = base_u.x + di;
                        if (ni < 0 || ni > i32(width)) { continue; }

                        let w = u_wx[di + 1] * wy * wz;
                        if (w <= 0.0) { continue; }

                        let node_pos = vec3<f32>(
                            f32(ni) * cell_size,
                            (f32(nj) + 0.5) * cell_size,
                            (f32(nk) + 0.5) * cell_size
                        );
                        let offset = node_pos - pos;
                        let affine_vel = c_mat_mul(pid, offset);
                        let momentum_x = (vel.x + affine_vel.x) * w;

                        // Check if within tile (U grid extends 1 extra in X)
                        let li = ni - tile_origin.x;
                        let lj = nj - tile_origin.y;
                        let lk = nk - tile_origin.z;

                        if (li >= 0 && li < i32(TILE_U_X) &&
                            lj >= 0 && lj < i32(TILE_U_Y) &&
                            lk >= 0 && lk < i32(TILE_U_Z)) {
                            // Accumulate to shared memory
                            let idx = u_index_tile(u32(li), u32(lj), u32(lk));
                            atomicAdd(&tile_u_sum[idx], i32(momentum_x * SCALE));
                            atomicAdd(&tile_u_weight[idx], i32(w * SCALE));
                        } else {
                            // Fallback to global
                            let idx = u_index_global(u32(ni), u32(nj), u32(nk));
                            atomicAdd(&u_sum[idx], i32(momentum_x * SCALE));
                            atomicAdd(&u_weight[idx], i32(w * SCALE));
                        }
                    }
                }
            }

            // ========== V component ==========
            let v_pos = pos / cell_size - vec3<f32>(0.5, 0.0, 0.5);
            let base_v = vec3<i32>(floor(v_pos));
            let frac_v = v_pos - vec3<f32>(base_v);

            let v_wx = array<f32, 3>(
                quadratic_bspline_1d(frac_v.x + 1.0),
                quadratic_bspline_1d(frac_v.x),
                quadratic_bspline_1d(frac_v.x - 1.0)
            );
            let v_wy = array<f32, 3>(
                quadratic_bspline_1d(frac_v.y + 1.0),
                quadratic_bspline_1d(frac_v.y),
                quadratic_bspline_1d(frac_v.y - 1.0)
            );
            let v_wz = array<f32, 3>(
                quadratic_bspline_1d(frac_v.z + 1.0),
                quadratic_bspline_1d(frac_v.z),
                quadratic_bspline_1d(frac_v.z - 1.0)
            );

            for (var dk: i32 = -1; dk <= 1; dk++) {
                let nk = base_v.z + dk;
                if (nk < 0 || nk >= i32(depth)) { continue; }
                let wz = v_wz[dk + 1];

                for (var dj: i32 = -1; dj <= 1; dj++) {
                    let nj = base_v.y + dj;
                    if (nj < 0 || nj > i32(height)) { continue; }
                    let wy = v_wy[dj + 1];

                    for (var di: i32 = -1; di <= 1; di++) {
                        let ni = base_v.x + di;
                        if (ni < 0 || ni >= i32(width)) { continue; }

                        let w = v_wx[di + 1] * wy * wz;
                        if (w <= 0.0) { continue; }

                        let node_pos = vec3<f32>(
                            (f32(ni) + 0.5) * cell_size,
                            f32(nj) * cell_size,
                            (f32(nk) + 0.5) * cell_size
                        );
                        let offset = node_pos - pos;
                        let affine_vel = c_mat_mul(pid, offset);
                        let momentum_y = (vel.y + affine_vel.y) * w;

                        let li = ni - tile_origin.x;
                        let lj = nj - tile_origin.y;
                        let lk = nk - tile_origin.z;

                        if (li >= 0 && li < i32(TILE_V_X) &&
                            lj >= 0 && lj < i32(TILE_V_Y) &&
                            lk >= 0 && lk < i32(TILE_V_Z)) {
                            let idx = v_index_tile(u32(li), u32(lj), u32(lk));
                            atomicAdd(&tile_v_sum[idx], i32(momentum_y * SCALE));
                            atomicAdd(&tile_v_weight[idx], i32(w * SCALE));
                        } else {
                            let idx = v_index_global(u32(ni), u32(nj), u32(nk));
                            atomicAdd(&v_sum[idx], i32(momentum_y * SCALE));
                            atomicAdd(&v_weight[idx], i32(w * SCALE));
                        }
                    }
                }
            }

            // ========== W component ==========
            let w_pos = pos / cell_size - vec3<f32>(0.5, 0.5, 0.0);
            let base_w = vec3<i32>(floor(w_pos));
            let frac_w = w_pos - vec3<f32>(base_w);

            let w_wx = array<f32, 3>(
                quadratic_bspline_1d(frac_w.x + 1.0),
                quadratic_bspline_1d(frac_w.x),
                quadratic_bspline_1d(frac_w.x - 1.0)
            );
            let w_wy = array<f32, 3>(
                quadratic_bspline_1d(frac_w.y + 1.0),
                quadratic_bspline_1d(frac_w.y),
                quadratic_bspline_1d(frac_w.y - 1.0)
            );
            let w_wz = array<f32, 3>(
                quadratic_bspline_1d(frac_w.z + 1.0),
                quadratic_bspline_1d(frac_w.z),
                quadratic_bspline_1d(frac_w.z - 1.0)
            );

            for (var dk: i32 = -1; dk <= 1; dk++) {
                let nk = base_w.z + dk;
                if (nk < 0 || nk > i32(depth)) { continue; }
                let wz = w_wz[dk + 1];

                for (var dj: i32 = -1; dj <= 1; dj++) {
                    let nj = base_w.y + dj;
                    if (nj < 0 || nj >= i32(height)) { continue; }
                    let wy = w_wy[dj + 1];

                    for (var di: i32 = -1; di <= 1; di++) {
                        let ni = base_w.x + di;
                        if (ni < 0 || ni >= i32(width)) { continue; }

                        let w = w_wx[di + 1] * wy * wz;
                        if (w <= 0.0) { continue; }

                        let node_pos = vec3<f32>(
                            (f32(ni) + 0.5) * cell_size,
                            (f32(nj) + 0.5) * cell_size,
                            f32(nk) * cell_size
                        );
                        let offset = node_pos - pos;
                        let affine_vel = c_mat_mul(pid, offset);
                        let momentum_z = (vel.z + affine_vel.z) * w;

                        let li = ni - tile_origin.x;
                        let lj = nj - tile_origin.y;
                        let lk = nk - tile_origin.z;

                        if (li >= 0 && li < i32(TILE_W_X) &&
                            lj >= 0 && lj < i32(TILE_W_Y) &&
                            lk >= 0 && lk < i32(TILE_W_Z)) {
                            let idx = w_index_tile(u32(li), u32(lj), u32(lk));
                            atomicAdd(&tile_w_sum[idx], i32(momentum_z * SCALE));
                            atomicAdd(&tile_w_weight[idx], i32(w * SCALE));
                        } else {
                            let idx = w_index_global(u32(ni), u32(nj), u32(nk));
                            atomicAdd(&w_sum[idx], i32(momentum_z * SCALE));
                            atomicAdd(&w_weight[idx], i32(w * SCALE));
                        }
                    }
                }
            }
        }
    }

    // ========== Phase 2: Flush shared memory to global ==========
    workgroupBarrier();

    // Flush U grid (80 elements, threads 0-79)
    if (tid < TILE_U_SIZE) {
        let val_sum = atomicLoad(&tile_u_sum[tid]);
        let val_weight = atomicLoad(&tile_u_weight[tid]);
        if (val_sum != 0 || val_weight != 0) {
            // Convert tile index back to global coordinates
            let li = tid % TILE_U_X;
            let lj = (tid / TILE_U_X) % TILE_U_Y;
            let lk = tid / (TILE_U_X * TILE_U_Y);
            let gi = u32(tile_origin.x) + li;
            let gj = u32(tile_origin.y) + lj;
            let gk = u32(tile_origin.z) + lk;
            // Bounds check
            if (gi <= params.width && gj < params.height && gk < params.depth) {
                let idx = u_index_global(gi, gj, gk);
                atomicAdd(&u_sum[idx], val_sum);
                atomicAdd(&u_weight[idx], val_weight);
            }
        }
    }

    // Flush V grid (80 elements, threads 80-159)
    if (tid >= 80u && tid < 160u) {
        let local_idx = tid - 80u;
        let val_sum = atomicLoad(&tile_v_sum[local_idx]);
        let val_weight = atomicLoad(&tile_v_weight[local_idx]);
        if (val_sum != 0 || val_weight != 0) {
            let li = local_idx % TILE_V_X;
            let lj = (local_idx / TILE_V_X) % TILE_V_Y;
            let lk = local_idx / (TILE_V_X * TILE_V_Y);
            let gi = u32(tile_origin.x) + li;
            let gj = u32(tile_origin.y) + lj;
            let gk = u32(tile_origin.z) + lk;
            if (gi < params.width && gj <= params.height && gk < params.depth) {
                let idx = v_index_global(gi, gj, gk);
                atomicAdd(&v_sum[idx], val_sum);
                atomicAdd(&v_weight[idx], val_weight);
            }
        }
    }

    // Flush W grid (80 elements, threads 160-239)
    if (tid >= 160u && tid < 240u) {
        let local_idx = tid - 160u;
        let val_sum = atomicLoad(&tile_w_sum[local_idx]);
        let val_weight = atomicLoad(&tile_w_weight[local_idx]);
        if (val_sum != 0 || val_weight != 0) {
            let li = local_idx % TILE_W_X;
            let lj = (local_idx / TILE_W_X) % TILE_W_Y;
            let lk = local_idx / (TILE_W_X * TILE_W_Y);
            let gi = u32(tile_origin.x) + li;
            let gj = u32(tile_origin.y) + lj;
            let gk = u32(tile_origin.z) + lk;
            if (gi < params.width && gj < params.height && gk <= params.depth) {
                let idx = w_index_global(gi, gj, gk);
                atomicAdd(&w_sum[idx], val_sum);
                atomicAdd(&w_weight[idx], val_weight);
            }
        }
    }

    // Flush particle counts (64 elements, threads 240-255 handle 4 each)
    if (tid >= 240u) {
        let base_idx = (tid - 240u) * 4u;
        for (var i = 0u; i < 4u; i++) {
            let local_idx = base_idx + i;
            if (local_idx < TILE_CELLS) {
                let val_particle = atomicLoad(&tile_particle_count[local_idx]);
                let val_sediment = atomicLoad(&tile_sediment_count[local_idx]);
                if (val_particle != 0 || val_sediment != 0) {
                    let li = local_idx % TILE_X;
                    let lj = (local_idx / TILE_X) % TILE_Y;
                    let lk = local_idx / (TILE_X * TILE_Y);
                    let gi = u32(tile_origin.x) + li;
                    let gj = u32(tile_origin.y) + lj;
                    let gk = u32(tile_origin.z) + lk;
                    if (gi < params.width && gj < params.height && gk < params.depth) {
                        let idx = cell_index_global(gi, gj, gk);
                        if (val_particle != 0) {
                            atomicAdd(&particle_count[idx], val_particle);
                        }
                        if (val_sediment != 0) {
                            atomicAdd(&sediment_count[idx], val_sediment);
                        }
                    }
                }
            }
        }
    }
}
