// Particle Separation Shader (3D) - Houdini-style relaxation
//
// Pushes particles apart when closer than min_dist to prevent compression.
// Uses spatial hash for O(n) neighbor lookup instead of O(n²).
//
// Two-pass approach:
// Pass 1 (count_particles): Count particles per cell using atomics
// Pass 2 (build_offsets): Prefix sum to get cell start offsets
// Pass 3 (scatter_particles): Write particle indices to sorted array
// Pass 4 (separate_particles): Compute separation forces using neighbor lookup

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    min_dist: f32,        // Push apart if closer than this
    push_strength: f32,   // Relaxation strength (0.3 typical)
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> particle_cells: array<u32>;

fn cell_hash(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || j < 0 || k < 0 ||
        u32(i) >= params.width || u32(j) >= params.height || u32(k) >= params.depth) {
        return 0xFFFFFFFFu;  // Invalid cell
    }
    return u32(k) * params.width * params.height + u32(j) * params.width + u32(i);
}

// Pass 1: Count particles per cell
@compute @workgroup_size(256)
fn count_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let i = i32(floor(pos.x / params.cell_size));
    let j = i32(floor(pos.y / params.cell_size));
    let k = i32(floor(pos.z / params.cell_size));

    let cell = cell_hash(i, j, k);
    particle_cells[id.x] = cell;

    if (cell != 0xFFFFFFFFu) {
        atomicAdd(&cell_counts[cell], 1u);
    }
}

// Pass 2: Compute prefix sum (done on CPU or separate shader)
// cell_offsets[i] = sum of cell_counts[0..i]

// Pass 3: Scatter particles to sorted array
@compute @workgroup_size(256)
fn scatter_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let cell = particle_cells[id.x];
    if (cell == 0xFFFFFFFFu) {
        return;
    }

    // Atomically get slot in cell's range
    let slot = atomicAdd(&cell_counts[cell], 1u);  // Reusing counts as write cursors
    let offset = cell_offsets[cell];
    sorted_indices[offset + slot] = id.x;
}

// Pass 4: Compute separation and apply directly to positions
@compute @workgroup_size(256)
fn separate_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x];
    let ci = i32(floor(pos.x / params.cell_size));
    let cj = i32(floor(pos.y / params.cell_size));
    let ck = i32(floor(pos.z / params.cell_size));

    var displacement = vec3<f32>(0.0, 0.0, 0.0);
    let min_dist_sq = params.min_dist * params.min_dist;

    // Check 3x3x3 neighborhood
    for (var dk: i32 = -1; dk <= 1; dk++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            for (var di: i32 = -1; di <= 1; di++) {
                let ni = ci + di;
                let nj = cj + dj;
                let nk = ck + dk;

                let neighbor_cell = cell_hash(ni, nj, nk);
                if (neighbor_cell == 0xFFFFFFFFu) {
                    continue;
                }

                // Get particle range for this cell
                let start = cell_offsets[neighbor_cell];
                let end = cell_offsets[neighbor_cell + 1u];

                for (var p: u32 = start; p < end; p++) {
                    let other_idx = sorted_indices[p];
                    if (other_idx == id.x) {
                        continue;
                    }

                    let other_pos = positions[other_idx];
                    let diff = pos - other_pos;
                    let dist_sq = dot(diff, diff);

                    if (dist_sq < min_dist_sq && dist_sq > 1e-10) {
                        let dist = sqrt(dist_sq);
                        let overlap = params.min_dist - dist;
                        // Each particle moves half the overlap
                        displacement += normalize(diff) * (overlap * params.push_strength * 0.5);
                    }
                }
            }
        }
    }

    // Apply displacement directly (will be clamped by advection)
    positions[id.x] = pos + displacement;
}

// Combined reset for cell_counts (run before count_particles)
@compute @workgroup_size(256)
fn reset_counts(@builtin(global_invocation_id) id: vec3<u32>) {
    let num_cells = params.width * params.height * params.depth;
    if (id.x >= num_cells) {
        return;
    }
    atomicStore(&cell_counts[id.x], 0u);
}
