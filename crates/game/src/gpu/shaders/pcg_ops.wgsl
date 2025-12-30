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

// Cell type constants (when buffer_c is used as cell_type)
const CELL_SOLID: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_AIR: u32 = 2u;

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

// Compute Laplacian of buffer_a at (i, j)
fn laplacian_at(i: u32, j: u32) -> f32 {
    let idx = get_index(i, j);
    let p_center = buffer_a[idx];

    var lap = 0.0;

    // Left
    if (i > 0u) {
        let left_idx = get_index(i - 1u, j);
        if (get_cell_type(left_idx) != CELL_SOLID) {
            lap += get_pressure(i - 1u, j) - p_center;
        }
    }

    // Right
    if (i < params.width - 1u) {
        let right_idx = get_index(i + 1u, j);
        if (get_cell_type(right_idx) != CELL_SOLID) {
            lap += get_pressure(i + 1u, j) - p_center;
        }
    }

    // Down
    if (j > 0u) {
        let down_idx = get_index(i, j - 1u);
        if (get_cell_type(down_idx) != CELL_SOLID) {
            lap += get_pressure(i, j - 1u) - p_center;
        }
    }

    // Up
    if (j < params.height - 1u) {
        let up_idx = get_index(i, j + 1u);
        if (get_cell_type(up_idx) != CELL_SOLID) {
            lap += get_pressure(i, j + 1u) - p_center;
        }
    }

    return lap;
}

// ============================================================================
// Grid operations (2D)
// ============================================================================

// Compute residual: buffer_d = buffer_b - Laplacian(buffer_a)
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

    buffer_d[idx] = buffer_b[idx] - laplacian_at(i, j);
}

// Apply Laplacian: buffer_d = Laplacian(buffer_a)
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

    buffer_d[idx] = laplacian_at(i, j);
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

// buffer_a = buffer_b
@compute @workgroup_size(256)
fn copy_buffer(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.length) {
        return;
    }
    buffer_a[idx] = buffer_b[idx];
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
@compute @workgroup_size(256)
fn dot_finalize(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let num_partials = (params.length + 255u) / 256u;

    // Load partial sums from buffer_d
    if (tid < num_partials) {
        shared_data[tid] = buffer_d[tid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Reduce
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
