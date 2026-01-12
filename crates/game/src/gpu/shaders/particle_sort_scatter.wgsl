// Particle Sort Scatter Shader - Reorder particles into sorted order by cell
//
// Input:
//   - cell_keys[i] = cell index for particle i
//   - cell_offsets[cell] = exclusive prefix sum (starting position for each cell)
//   - cell_counters[cell] = atomic counter for current position within cell
//   - Original particle data arrays
//
// Output:
//   - Sorted particle data arrays

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_keys: array<u32>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_counters: array<atomic<u32>>;

// Input particle arrays
@group(0) @binding(4) var<storage, read> in_positions: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read> in_velocities: array<vec3<f32>>;
@group(0) @binding(6) var<storage, read> in_densities: array<f32>;
@group(0) @binding(7) var<storage, read> in_c_col0: array<vec3<f32>>;
@group(0) @binding(8) var<storage, read> in_c_col1: array<vec3<f32>>;
@group(0) @binding(9) var<storage, read> in_c_col2: array<vec3<f32>>;

// Output particle arrays (sorted)
@group(0) @binding(10) var<storage, read_write> out_positions: array<vec3<f32>>;
@group(0) @binding(11) var<storage, read_write> out_velocities: array<vec3<f32>>;
@group(0) @binding(12) var<storage, read_write> out_densities: array<f32>;
@group(0) @binding(13) var<storage, read_write> out_c_col0: array<vec3<f32>>;
@group(0) @binding(14) var<storage, read_write> out_c_col1: array<vec3<f32>>;
@group(0) @binding(15) var<storage, read_write> out_c_col2: array<vec3<f32>>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let particle_idx = id.x;
    let cell_idx = cell_keys[particle_idx];

    // Get the base offset for this cell and atomically increment the counter
    let base_offset = cell_offsets[cell_idx];
    let local_offset = atomicAdd(&cell_counters[cell_idx], 1u);
    let sorted_idx = base_offset + local_offset;

    // Copy particle data to sorted position
    out_positions[sorted_idx] = in_positions[particle_idx];
    out_velocities[sorted_idx] = in_velocities[particle_idx];
    out_densities[sorted_idx] = in_densities[particle_idx];
    out_c_col0[sorted_idx] = in_c_col0[particle_idx];
    out_c_col1[sorted_idx] = in_c_col1[particle_idx];
    out_c_col2[sorted_idx] = in_c_col2[particle_idx];
}
