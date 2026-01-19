// Particle Sort Prefix Sum Shader - Compute exclusive prefix sum of cell counts
//
// Input: cell_counts[cell] = number of particles in that cell
// Output: cell_offsets[cell] = starting index for that cell in sorted order
//
// Uses Blelloch scan algorithm within workgroups, with block totals for multi-workgroup.
// This shader handles up to 1024 elements per workgroup.

struct Params {
    element_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> shared_data: array<u32, 512>;

// Workgroup prefix sum using up-sweep and down-sweep
@compute @workgroup_size(256)
fn local_prefix_sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let block_offset = workgroup_id.x * 512u;
    let tid = local_id.x;

    // Each thread loads 2 elements
    let idx0 = block_offset + tid * 2u;
    let idx1 = block_offset + tid * 2u + 1u;

    if (idx0 < params.element_count) {
        shared_data[tid * 2u] = data[idx0];
    } else {
        shared_data[tid * 2u] = 0u;
    }
    if (idx1 < params.element_count) {
        shared_data[tid * 2u + 1u] = data[idx1];
    } else {
        shared_data[tid * 2u + 1u] = 0u;
    }

    workgroupBarrier();

    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = 256u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            shared_data[bi] += shared_data[ai];
        }
        offset *= 2u;
    }

    // Store block total and clear last element
    if (tid == 0u) {
        block_sums[workgroup_id.x] = shared_data[511u];
        shared_data[511u] = 0u;
    }

    // Down-sweep phase
    for (var d = 1u; d < 512u; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let temp = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += temp;
        }
    }

    workgroupBarrier();

    // Write results back
    if (idx0 < params.element_count) {
        data[idx0] = shared_data[tid * 2u];
    }
    if (idx1 < params.element_count) {
        data[idx1] = shared_data[tid * 2u + 1u];
    }
}

// Scan block_sums array to get exclusive prefix sum (run before add_block_offsets)
// This is a simple linear scan - works for up to ~1000 blocks
@compute @workgroup_size(1)
fn scan_block_sums() {
    // Simple sequential prefix sum on block_sums (exclusive)
    // block_sums[i] will contain sum of blocks 0..i (exclusive)
    var running_sum = 0u;

    // element_count here is the number of blocks to scan
    let num_blocks = (params.element_count + 511u) / 512u;

    for (var i = 0u; i < num_blocks; i++) {
        let old_val = block_sums[i];
        block_sums[i] = running_sum;
        running_sum += old_val;
    }
}

// Add block offsets back to local results
// CRITICAL: local_prefix_sum uses 512-element blocks (256 threads × 2 elements each)
// We must match that indexing here, NOT use our workgroup_id which is based on 256 elements
@compute @workgroup_size(256)
fn add_block_offsets(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx >= params.element_count) {
        return;
    }

    // Which 512-element block does this element belong to?
    // This MUST match the block size used in local_prefix_sum (512 elements per workgroup)
    let block_idx = idx / 512u;

    if (block_idx == 0u) {
        // First 512-element block has no offset to add
        return;
    }

    // Add the prefix sum of previous blocks (now stored in block_sums after scan_block_sums)
    data[idx] += block_sums[block_idx];
}
