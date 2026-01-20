//! DEM Hash Table Clear Shader
//!
//! Clears the spatial hash table to EMPTY_SLOT values before each frame.
//! This must be run before the broadphase pass to prevent stale/corrupted data.

@group(0) @binding(0) var<storage, read_write> hash_table: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> entry_counter: atomic<u32>;
@group(0) @binding(2) var<uniform> params: ClearParams;

struct ClearParams {
    table_size: u32,
}

const EMPTY_SLOT = 0xffffffffu;
const WORKGROUP_SIZE = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Clear hash table entries
    if idx < params.table_size {
        atomicStore(&hash_table[idx], EMPTY_SLOT);
    }
    
    // First thread resets the entry counter
    if idx == 0u {
        atomicStore(&entry_counter, 0u);
    }
}
