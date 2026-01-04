// Red-Black Gauss-Seidel with SOR
// Two passes per iteration: red cells then black cells
// Proper Neumann BC: exclude missing neighbors, adjust coefficient

struct Params {
    width: u32,
    height: u32,
    alpha: f32,
    rbeta: f32,  // Not used - we compute per-cell
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(2) var<storage, read> divergence: array<f32>;

const OMEGA: f32 = 1.9;

// Red cells: (i + j) % 2 == 0
@compute @workgroup_size(8, 8, 1)
fn solve_red(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= params.width || j >= params.height) {
        return;
    }
    if ((i + j) % 2u != 0u) {
        return;
    }

    let idx = j * params.width + i;
    let p_old = pressure[idx];
    let div = divergence[idx];

    var neighbor_sum: f32 = 0.0;
    var neighbor_count: f32 = 0.0;

    // Left neighbor (skip at left wall)
    if (i > 0u) {
        neighbor_sum += pressure[idx - 1u];
        neighbor_count += 1.0;
    }
    // Right neighbor (skip at right wall)
    if (i + 1u < params.width) {
        neighbor_sum += pressure[idx + 1u];
        neighbor_count += 1.0;
    }
    // Bottom neighbor (skip at bottom wall)
    if (j > 0u) {
        neighbor_sum += pressure[idx - params.width];
        neighbor_count += 1.0;
    }
    // Top neighbor (always valid, open boundary)
    if (j + 1u < params.height) {
        neighbor_sum += pressure[idx + params.width];
        neighbor_count += 1.0;
    }

    if (neighbor_count > 0.0) {
        // Poisson: (neighbor_sum - n*p) / dx² = div
        // => p = (neighbor_sum - dx²*div) / n
        // => p = (neighbor_sum + alpha*div) / n  [alpha = -dx²]
        let p_gs = (neighbor_sum + params.alpha * div) / neighbor_count;
        pressure[idx] = p_old + OMEGA * (p_gs - p_old);
    }
}

// Black cells: (i + j) % 2 == 1
@compute @workgroup_size(8, 8, 1)
fn solve_black(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= params.width || j >= params.height) {
        return;
    }
    if ((i + j) % 2u != 1u) {
        return;
    }

    let idx = j * params.width + i;
    let p_old = pressure[idx];
    let div = divergence[idx];

    var neighbor_sum: f32 = 0.0;
    var neighbor_count: f32 = 0.0;

    if (i > 0u) {
        neighbor_sum += pressure[idx - 1u];
        neighbor_count += 1.0;
    }
    if (i + 1u < params.width) {
        neighbor_sum += pressure[idx + 1u];
        neighbor_count += 1.0;
    }
    if (j > 0u) {
        neighbor_sum += pressure[idx - params.width];
        neighbor_count += 1.0;
    }
    if (j + 1u < params.height) {
        neighbor_sum += pressure[idx + params.width];
        neighbor_count += 1.0;
    }

    if (neighbor_count > 0.0) {
        let p_gs = (neighbor_sum + params.alpha * div) / neighbor_count;
        pressure[idx] = p_old + OMEGA * (p_gs - p_old);
    }
}
