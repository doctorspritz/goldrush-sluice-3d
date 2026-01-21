//! DEM Broad Phase: Spatial Hashing
//! 
//! Inserts all particles into a spatial hash grid.
//! Each particle inserts itself into 27 neighboring cells (3x3x3 cube).

@group(0) @binding(0) var<storage, read> particle_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particle_flags: array<u32>;
// Hash table stores indices into hash_entries array
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> hash_entries: array<HashEntry>;
@group(0) @binding(4) var<storage, read_write> entry_counter: atomic<u32>;
@group(0) @binding(5) var<uniform> params: HashParams;
@group(0) @binding(6) var<storage, read> particle_template_ids: array<u32>;
@group(0) @binding(7) var<storage, read> templates: array<GpuClumpTemplate>;

struct GpuClumpTemplate {
    sphere_count: u32,
    mass: f32,
    radius: f32,
    particle_radius: f32,
    inertia_inv: mat3x3<f32>,
}

struct HashEntry {
    particle_idx: u32,
    next_idx: u32,
}

struct HashParams {
    table_size: u32,
    cell_size: f32,
    max_particles: u32,
    max_hash_entries: u32,
}

const EMPTY_SLOT = 0xffffffffu;
const WORKGROUP_SIZE = 64u;

fn hash_3d(coord: vec3<i32>) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    
    let x = bitcast<u32>(coord.x);
    let y = bitcast<u32>(coord.y);
    let z = bitcast<u32>(coord.z);
    
    return ((x * p1) ^ (y * p2) ^ (z * p3)) % params.table_size;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles { return; }
    
    // Skip inactive particles
    let flags = particle_flags[particle_idx];
    if (flags & 1u) == 0u { return; }
    
    let pos = particle_positions[particle_idx].xyz;
    
    let cell_coord = vec3<i32>(
        i32(floor(pos.x / params.cell_size)),
        i32(floor(pos.y / params.cell_size)),
        i32(floor(pos.z / params.cell_size))
    );
    
    // Insert into 27 cells
    // Using simple standard linked list with atomicExchange
    for (var dz = -1i; dz <= 1i; dz++) {
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                let neighbor_coord = cell_coord + vec3<i32>(dx, dy, dz);
                let hash = hash_3d(neighbor_coord);
                
                // Allocate new entry
                let entry_idx = atomicAdd(&entry_counter, 1u);
                
                // Link into list if not overflowing
                if entry_idx < params.max_hash_entries {
                    let old_head = atomicExchange(&hash_table[hash], entry_idx);
                    hash_entries[entry_idx] = HashEntry(particle_idx, old_head);
                }
            }
        }
    }
}
