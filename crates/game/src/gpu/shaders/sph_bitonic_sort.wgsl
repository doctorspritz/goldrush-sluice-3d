// Bitonic Sort for SPH Spatial Hashing
//
// Sorts particle_order array by cell_indices for cache-coherent neighbor access.
// Uses comparison-swap network pattern.

struct SphParams {
    num_particles: u32,
    h: f32,
    h2: f32,
    rest_density: f32,

    dt: f32,
    dt2: f32,
    gravity: f32,
    omega: f32,

    cell_size: f32,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,

    poly6_coef: f32,
    spiky_grad_coef: f32,
    pressure_iters: u32,
    particle_mass: f32,
}

struct SortParams {
    j: u32,      // Current step size
    k: u32,      // Current block size
    n: u32,      // Total elements (padded to power of 2)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions_pred: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressures: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_ii: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressure_accel: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> cell_indices: array<u32>;
@group(0) @binding(9) var<storage, read_write> particle_order: array<u32>;
@group(0) @binding(10) var<storage, read_write> cell_offsets: array<atomic<u32>>;
@group(0) @binding(11) var<uniform> sort_params: SortParams;

// Shared memory for local sort
var<workgroup> shared_keys: array<u32, 512>;
var<workgroup> shared_vals: array<u32, 512>;

// Get sort key for particle (cell index, with particle index as tiebreaker)
fn get_key(idx: u32) -> u32 {
    if (idx >= params.num_particles) {
        return 0xFFFFFFFFu;  // Invalid particles sort to end
    }
    return cell_indices[particle_order[idx]];
}

// Compare and swap two elements
fn compare_swap(i: u32, j: u32, dir: bool) {
    let key_i = get_key(i);
    let key_j = get_key(j);

    let should_swap = select(key_i > key_j, key_i < key_j, dir);

    if (should_swap) {
        let temp = particle_order[i];
        particle_order[i] = particle_order[j];
        particle_order[j] = temp;
    }
}

// Local bitonic sort within a workgroup (up to 512 elements)
@compute @workgroup_size(256)
fn bitonic_sort_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let local_id = lid.x;
    let group_offset = wid.x * 512u;

    // Load two elements per thread into shared memory
    let idx0 = group_offset + local_id * 2u;
    let idx1 = idx0 + 1u;

    if (idx0 < params.num_particles) {
        shared_keys[local_id * 2u] = cell_indices[particle_order[idx0]];
        shared_vals[local_id * 2u] = particle_order[idx0];
    } else {
        shared_keys[local_id * 2u] = 0xFFFFFFFFu;
        shared_vals[local_id * 2u] = idx0;
    }

    if (idx1 < params.num_particles) {
        shared_keys[local_id * 2u + 1u] = cell_indices[particle_order[idx1]];
        shared_vals[local_id * 2u + 1u] = particle_order[idx1];
    } else {
        shared_keys[local_id * 2u + 1u] = 0xFFFFFFFFu;
        shared_vals[local_id * 2u + 1u] = idx1;
    }

    workgroupBarrier();

    // Bitonic sort within shared memory
    for (var k = 2u; k <= 512u; k *= 2u) {
        for (var j = k / 2u; j > 0u; j /= 2u) {
            let i0 = local_id * 2u;
            let i1 = i0 + 1u;

            // Determine comparison partner
            let partner0 = i0 ^ j;
            let partner1 = i1 ^ j;

            // Determine sort direction
            let dir0 = ((i0 & k) == 0u);
            let dir1 = ((i1 & k) == 0u);

            // Compare and swap in shared memory
            if (partner0 > i0 && partner0 < 512u) {
                let should_swap = select(
                    shared_keys[i0] > shared_keys[partner0],
                    shared_keys[i0] < shared_keys[partner0],
                    dir0
                );
                if (should_swap) {
                    let temp_key = shared_keys[i0];
                    let temp_val = shared_vals[i0];
                    shared_keys[i0] = shared_keys[partner0];
                    shared_vals[i0] = shared_vals[partner0];
                    shared_keys[partner0] = temp_key;
                    shared_vals[partner0] = temp_val;
                }
            }

            workgroupBarrier();

            if (partner1 > i1 && partner1 < 512u) {
                let should_swap = select(
                    shared_keys[i1] > shared_keys[partner1],
                    shared_keys[i1] < shared_keys[partner1],
                    dir1
                );
                if (should_swap) {
                    let temp_key = shared_keys[i1];
                    let temp_val = shared_vals[i1];
                    shared_keys[i1] = shared_keys[partner1];
                    shared_vals[i1] = shared_vals[partner1];
                    shared_keys[partner1] = temp_key;
                    shared_vals[partner1] = temp_val;
                }
            }

            workgroupBarrier();
        }
    }

    // Write back to global memory
    if (idx0 < params.num_particles) {
        particle_order[idx0] = shared_vals[local_id * 2u];
    }
    if (idx1 < params.num_particles) {
        particle_order[idx1] = shared_vals[local_id * 2u + 1u];
    }
}

// Global bitonic merge for blocks larger than workgroup size
// This kernel is called multiple times with different j/k values
@compute @workgroup_size(256)
fn bitonic_sort_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.num_particles;

    if (i >= n) { return; }

    // For global sort, we need push constants or another way to pass j,k
    // For now, implement a simple O(n log²n) sort by calling this multiple times
    // The host code will set up the proper j,k values

    // Simple version: compare with fixed offset
    // This is called for each (j, k) pair by the host
    // Use uniforms passed from host
    let k = sort_params.k;
    let j = sort_params.j;

    let partner = i ^ j;

    // Fix: Only one thread of the pair handles the swap to avoid race conditions
    if (partner > i && partner < n) {
        // Read indices first (before any writes)
        let idx_i = particle_order[i];
        let idx_partner = particle_order[partner];
        
        // Read keys
        let key_i = cell_indices[idx_i];
        let key_partner = cell_indices[idx_partner];

        // Determine sort direction: ascending for even blocks, descending for odd
        let ascending = ((i & k) == 0u);
        
        // Should we swap? 
        let should_swap = (ascending && key_i > key_partner) || (!ascending && key_i < key_partner);

        if (should_swap) {
            // Swap indices
            particle_order[i] = idx_partner;
            particle_order[partner] = idx_i;
        }
    }
}
