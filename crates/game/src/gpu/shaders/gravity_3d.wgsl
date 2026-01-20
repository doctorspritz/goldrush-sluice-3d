// Gravity Application Shader (3D)
//
// Applies gravity to the V (vertical) velocity component.
// Only affects faces adjacent to at least one fluid cell.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    gravity_dt: f32,  // gravity * dt (typically -9.81 * dt)
    cell_size: f32,
    // Bitmask for open boundaries:
    // Bit 0 (1): -X open, Bit 1 (2): +X open
    // Bit 2 (4): -Y open, Bit 3 (8): +Y open
    // Bit 4 (16): -Z open, Bit 5 (32): +Z open
    open_boundaries: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(3) var<storage, read> bed_height: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * (params.height + 1u) + j * params.width + i;
}

// Get cell type with proper boundary handling based on open_boundaries config.
// OPEN boundary → treat as AIR (Dirichlet p=0, flow allowed)
// CLOSED boundary → treat as SOLID (Neumann, no penetration)
fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    // Check open boundary flags
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // -X boundary (inlet side)
    if (i < 0) {
        return select(CELL_SOLID, CELL_AIR, open_neg_x);
    }
    // +X boundary (outlet side)
    if (u32(i) >= params.width) {
        return select(CELL_SOLID, CELL_AIR, open_pos_x);
    }
    // -Y boundary (floor)
    if (j < 0) {
        return select(CELL_SOLID, CELL_AIR, open_neg_y);
    }
    // +Y boundary (ceiling/top)
    if (u32(j) >= params.height) {
        return select(CELL_SOLID, CELL_AIR, open_pos_y);
    }
    // -Z boundary (side wall)
    if (k < 0) {
        return select(CELL_SOLID, CELL_AIR, open_neg_z);
    }
    // +Z boundary (side wall)
    if (u32(k) >= params.depth) {
        return select(CELL_SOLID, CELL_AIR, open_pos_z);
    }

    return cell_type[cell_index(u32(i), u32(j), u32(k))];
}

@compute @workgroup_size(8, 8, 4)
fn apply_gravity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;  // 0 to height (inclusive for V grid)
    let k = id.z;

    if (i >= params.width || j > params.height || k >= params.depth) {
        return;
    }

    // V face is between cells [i, j-1, k] and [i, j, k]
    let bottom_type = get_cell_type(i32(i), i32(j) - 1, i32(k));
    let top_type = get_cell_type(i32(i), i32(j), i32(k));

    // CRITICAL: Do NOT apply gravity at solid boundaries!
    // If we apply gravity to floor V faces, divergence becomes zero
    // and no hydrostatic pressure is generated.
    if (bottom_type == CELL_SOLID || top_type == CELL_SOLID) {
        return;
    }

    // Skip if both sides are air (no fluid to accelerate)
    // DISABLED: This prevents ballistic particles from accelerating!
    // Since we clear the grid every frame, applying gravity to air is safe and necessary.
    // if (bottom_type == CELL_AIR && top_type == CELL_AIR) {
    //    return;
    // }

    // Apply gravity to V faces adjacent to at least one fluid cell
    let idx = v_index(i, j, k);
    grid_v[idx] += params.gravity_dt;
}
