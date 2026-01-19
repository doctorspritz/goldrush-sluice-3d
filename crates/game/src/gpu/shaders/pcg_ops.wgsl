// PCG (Preconditioned Conjugate Gradient) Vector Operations
//
// This shader provides various operations for the PCG outer loop.
// Each entry point uses the same bind group layout for simplicity.

struct PcgParams {
    width: u32,
    height: u32,
    alpha: f32,       // Scalar for axpy/xpay operations
    length: u32,      // Total number of elements (for 1D ops)
}

// Flexible buffer bindings - interpretation depends on entry point
@group(0) @binding(0) var<storage, read_write> buffer_a: array<f32>;  // x (read_write)
@group(0) @binding(1) var<storage, read> buffer_b: array<f32>;        // y (read)
@group(0) @binding(2) var<storage, read> buffer_c: array<f32>;        // z or cell_type
@group(0) @binding(3) var<storage, read_write> buffer_d: array<f32>;  // output or partial_sums
@group(0) @binding(4) var<uniform> params: PcgParams;

// Cell type constants (must match main 3D shaders: AIR=0, FLUID=1, SOLID=2)
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn get_index(i: u32, j: u32) -> u32 {
    return j * params.width + i;
}

fn get_cell_type(idx: u32) -> u32 {
    return bitcast<u32>(buffer_c[idx]);
}

fn get_pressure(i: u32, j: u32) -> f32 {
    if (i >= params.width || j >= params.height) {
        return 0.0;
    }
    let idx = get_index(i, j);
    let ct = get_cell_type(idx);
    if (ct == CELL_SOLID) {
        return 0.0;
    }
    return buffer_a[idx];
}

// Compute Laplacian of buffer_a at (i, j) using fixed 4-neighbor stencil
// Uses Neumann BC: dp/dn = 0, implemented by mirroring pressure at boundaries/solids
fn laplacian_at(i: u32, j: u32) -> f32 {
    let idx = get_index(i, j);
    let p_center = buffer_a[idx];

    // Gather neighbor pressures with Neumann BC (mirror at solid/boundary)
    // Always use 4 neighbors for consistent stencil with smoother

    // Left - mirror if at boundary or solid
    var p_left = p_center;
    if (i > 0u) {
        let left_idx = get_index(i - 1u, j);
        if (get_cell_type(left_idx) != CELL_SOLID) {
            p_left = buffer_a[left_idx];
        }
    }

    // Right - mirror if at boundary or solid
    var p_right = p_center;
    if (i < params.width - 1u) {
        let right_idx = get_index(i + 1u, j);
        if (get_cell_type(right_idx) != CELL_SOLID) {
            p_right = buffer_a[right_idx];
        }
    }

    // Down - mirror if at boundary or solid
    var p_down = p_center;
    if (j > 0u) {
        let down_idx = get_index(i, j - 1u);
        if (get_cell_type(down_idx) != CELL_SOLID) {
            p_down = buffer_a[down_idx];
        }
    }

    // Up - mirror if at boundary or solid
    var p_up = p_center;
    if (j < params.height - 1u) {
        let up_idx = get_index(i, j + 1u);
        if (get_cell_type(up_idx) != CELL_SOLID) {
            p_up = buffer_a[up_idx];
        }
    }

    // Laplacian = (p_L + p_R + p_D + p_U - 4*p_center)
    return p_left + p_right + p_down + p_up - 4.0 * p_center;
}

// ============================================================================
// Grid operations (2D)
// ============================================================================

// Compute residual: r = b - A*x where A = -Laplacian, b = -div
// The pressure equation is: Laplacian(p) = div, which becomes (-Laplacian)p = -div
// So: r = b - Ax = (-div) - (-Laplacian(p)) = Laplacian(p) - div
// buffer_a = pressure, buffer_b = divergence, buffer_c = cell_type, buffer_d = residual
@compute @workgroup_size(8, 8)
fn compute_pcg_residual(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;

    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = get_index(i, j);

    if (get_cell_type(idx) != CELL_FLUID) {
        buffer_d[idx] = 0.0;
        return;
    }

    // r = b - Ax = (-div) - (-Laplacian(p)) = Laplacian(p) - div
    buffer_d[idx] = laplacian_at(i, j) - buffer_b[idx];
}

// Apply operator A = -Laplacian: buffer_d = -Laplacian(buffer_a)
// This makes A positive semi-definite for CG convergence
// buffer_a = p, buffer_c = cell_type, buffer_d = Ap
@compute @workgroup_size(8, 8)
fn apply_laplacian(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;

    if (i >= params.width || j >= params.height) {
        return;
    }

    let idx = get_index(i, j);

    if (get_cell_type(idx) != CELL_FLUID) {
        buffer_d[idx] = 0.0;
        return;
    }

    // Return -Laplacian(p) to make the operator positive semi-definite
    buffer_d[idx] = -laplacian_at(i, j);
}

// ============================================================================
// Vector operations (1D)
// ============================================================================

// buffer_a += alpha * buffer_b
@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.length) {
        return;
    }
    buffer_a[idx] += params.alpha * buffer_b[idx];
}

// buffer_a = buffer_b + alpha * buffer_a (for p = z + beta * p)
@compute @workgroup_size(256)
fn xpay(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.length) {
        return;
    }
    buffer_a[idx] = buffer_b[idx] + params.alpha * buffer_a[idx];
}

// buffer_a = alpha * buffer_b (scaled copy, use alpha=1.0 for regular copy, alpha=-1.0 for negate)
@compute @workgroup_size(256)
fn copy_buffer(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.length) {
        return;
    }
    buffer_a[idx] = params.alpha * buffer_b[idx];
}

// ============================================================================
// Dot product with parallel reduction
// ============================================================================

var<workgroup> shared_data: array<f32, 256>;

// Compute partial dot products: buffer_d[workgroup_id] = sum(buffer_a[i] * buffer_b[i])
@compute @workgroup_size(256)
fn dot_partial(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load and multiply
    if (gid < params.length) {
        shared_data[tid] = buffer_a[gid] * buffer_b[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        buffer_d[workgroup_id.x] = shared_data[0];
    }
}

// Finalize dot product by summing partial sums in buffer_d
// Reads from buffer_d, writes final sum to buffer_d[0]
// Handles grids up to 16M cells (64K workgroups = 64K partial sums)
@compute @workgroup_size(256)
fn dot_finalize(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let num_partials = (params.length + 255u) / 256u;

    // Each thread sums multiple partial sums in a strided loop
    // This handles cases where num_partials > 256 (e.g., 512x512 grid = 1024 partials)
    var local_sum = 0.0;
    for (var i = tid; i < num_partials; i += 256u) {
        local_sum += buffer_d[i];
    }
    shared_data[tid] = local_sum;

    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write final result
    if (tid == 0u) {
        buffer_d[0] = shared_data[0];
    }
}
