// Fluid Cell Expansion Shader (3D)
//
// Conservative expansion: marks cells as FLUID only if:
// - The cell itself contains particles, OR
// - At least 2 face-adjacent neighbors contain particles
//
// This prevents over-expansion that causes volume inflation while still
// preventing gaps that cause volume collapse.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    // Bitmask for open boundaries:
    // Bit 0 (1): -X open, Bit 1 (2): +X open
    // Bit 2 (4): -Y open, Bit 3 (8): +Y open
    // Bit 4 (16): -Z open, Bit 5 (32): +Z open
    open_boundaries: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particle_count: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> cell_type: array<u32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

// Minimum neighbors with particles to expand into empty cell
const MIN_NEIGHBORS_FOR_EXPANSION: u32 = 3u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_particle_count(i: i32, j: i32, k: i32) -> i32 {
    if (i < 0 || i >= i32(params.width)) { return 0; }
    if (j < 0 || j >= i32(params.height)) { return 0; }
    if (k < 0 || k >= i32(params.depth)) { return 0; }

    let idx = cell_index(u32(i), u32(j), u32(k));
    return atomicLoad(&particle_count[idx]);
}

fn count_neighbors_with_particles(i: i32, j: i32, k: i32) -> u32 {
    var count: u32 = 0u;

    // Count 6 face neighbors with particles
    if (get_particle_count(i - 1, j, k) > 0) { count += 1u; }
    if (get_particle_count(i + 1, j, k) > 0) { count += 1u; }
    if (get_particle_count(i, j - 1, k) > 0) { count += 1u; }
    if (get_particle_count(i, j + 1, k) > 0) { count += 1u; }
    if (get_particle_count(i, j, k - 1) > 0) { count += 1u; }
    if (get_particle_count(i, j, k + 1) > 0) { count += 1u; }

    return count;
}

fn is_domain_boundary(i: u32, j: u32, k: u32) -> bool {
    return i == 0u || i == params.width - 1u ||
           j == 0u || j == params.height - 1u ||
           k == 0u || k == params.depth - 1u;
}

// Check if a boundary cell should be SOLID based on open_boundaries config.
// OPEN boundaries are treated as AIR, CLOSED boundaries as SOLID.
fn is_boundary_solid(i: u32, j: u32, k: u32) -> bool {
    // Extract open boundary flags
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // Check each boundary and return false (not solid) if it's open
    if (i == 0u && open_neg_x) { return false; }
    if (i == params.width - 1u && open_pos_x) { return false; }
    if (j == 0u && open_neg_y) { return false; }
    if (j == params.height - 1u && open_pos_y) { return false; }
    if (k == 0u && open_neg_z) { return false; }
    if (k == params.depth - 1u && open_pos_z) { return false; }

    // If we reach here and it's a boundary, it's a closed boundary -> SOLID
    return is_domain_boundary(i, j, k);
}

@compute @workgroup_size(8, 8, 4)
fn expand_fluid_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    let current_type = cell_type[idx];

    // Boundary cells are SOLID only if they're CLOSED boundaries.
    // OPEN boundaries allow particles to flow through (treated as AIR).
    if (is_boundary_solid(i, j, k)) {
        cell_type[idx] = CELL_SOLID;
        return;
    }

    // CRITICAL: SDF-marked SOLID cells stay SOLID regardless of particles.
    // This ensures pressure solver sees obstacles correctly.
    // Particles inside solid regions (from numerical errors) are handled by collision,
    // not by making the cell FLUID which would hide the obstacle from pressure.
    if (current_type == CELL_SOLID) {
        return;
    }

    // Non-solid cell with particles: mark as FLUID for pressure enforcement
    let pcount = get_particle_count(i32(i), i32(j), i32(k));
    if (pcount > 0) {
        cell_type[idx] = CELL_FLUID;
        return;
    }

    // Empty non-solid cell: mark FLUID if surrounded by enough neighbors with particles
    // This prevents gaps at the fluid surface that cause volume collapse
    let neighbor_count = count_neighbors_with_particles(i32(i), i32(j), i32(k));
    if (neighbor_count >= MIN_NEIGHBORS_FOR_EXPANSION) {
        cell_type[idx] = CELL_FLUID;
    } else {
        cell_type[idx] = CELL_AIR;
    }
}
