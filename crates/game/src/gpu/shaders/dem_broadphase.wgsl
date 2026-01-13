//! DEM Broad Phase: Spatial Hashing
//! 
//! Inserts all particles into a spatial hash grid for O(1) neighbor queries.
//! Each particle checks 27 neighboring cells (3x3x3 cube around it).

@group(0) @binding(0) var<storage, read> particle_positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> particle_radii: array<f32>;
@group(0) @binding(2) var<storage, read> particle_flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> hash_table: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> hash_entries: array<HashEntry>;
@group(0) @binding(5) var<storage, read_write> entry_counter: atomic<u32>>;
@group(0) @binding(6) var<uniform> params: HashParams;

struct HashEntry {
    particle_idx: u32,
    next_idx: u32,
}

const EMPTY_SLOT = 0xffffffffu;
const WORKGROUP_SIZE = 64u;

// Optimized 3D spatial hash function
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
    if (flags & 1u) == 0u { return; } // PARTICLE_ACTIVE not set
    
    let pos = particle_positions[particle_idx];
    let radius = particle_radii[particle_idx];
    
    // Calculate grid cell coordinates
    let cell_coord = vec3<i32>(
        i32(floor(pos.x / params.cell_size)),
        i32(floor(pos.y / params.cell_size)),
        i32(floor(pos.z / params.cell_size))
    );
    
    // Insert particle into 27 neighboring cells (3x3x3)
    for dz in -1i32..=1i32 {
        for dy in -1i32..=1i32 {
            for dx in -1i32..=1i32 {
                let neighbor_coord = cell_coord + vec3<i32>(dx, dy, dz);
                let hash = hash_3d(neighbor_coord);
                
                // Try to claim hash slot
                let old = atomicCompareExchangeWeak(&hash_table[hash], EMPTY_SLOT, particle_idx);
                if old == EMPTY_SLOT || old == particle_idx {
                    // Success: slot was empty or already ours
                    continue;
                }
                
                // Slot occupied: add to linked list
                let entry_idx = atomicAdd(&entry_counter, 1u);
                hash_entries[entry_idx] = HashEntry {
                    particle_idx: particle_idx,
                    next_idx: old
                };
            }
        }
    }
}