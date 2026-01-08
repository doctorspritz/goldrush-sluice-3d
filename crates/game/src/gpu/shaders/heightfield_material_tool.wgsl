// heightfield_material_tool.wgsl
// Add or remove specific materials at a location

struct ToolParams {
    pos_x: f32,           // World position X
    pos_z: f32,           // World position Z
    radius: f32,          // Effect radius
    amount: f32,          // Positive = add, Negative = remove (per second)
    material_type: u32,   // 0=sediment, 1=overburden, 2=gravel
    enabled: u32,         // 0 = disabled, 1 = enabled
    world_width: u32,     // World width
    world_depth: u32,     // World depth
    tile_width: u32,      // Tile width
    tile_depth: u32,      // Tile depth
    origin_x: u32,        // Tile origin X
    origin_z: u32,        // Tile origin Z
    cell_size: f32,       // Cell size
    dt: f32,              // Delta time
    _pad: vec2<f32>,      // Padding to 64 bytes
}

@group(0) @binding(0) var<uniform> tool: ToolParams;

// Terrain layers
@group(1) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(1) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(1) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(1) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(1) @binding(4) var<storage, read_write> sediment: array<f32>;

fn get_idx(x: u32, z: u32) -> u32 {
    return z * tool.world_width + x;
}

@compute @workgroup_size(16, 16)
fn apply_material_tool(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (tool.enabled == 0u) { return; }
    
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= tool.tile_width || tile_z >= tool.tile_depth) { return; }
    let x = tile_x + tool.origin_x;
    let z = tile_z + tool.origin_z;
    if (x >= tool.world_width || z >= tool.world_depth) { return; }
    
    let idx = get_idx(x, z);
    
    // Calculate world position of this cell
    let cell_x = f32(x) * tool.cell_size + tool.cell_size * 0.5;
    let cell_z = f32(z) * tool.cell_size + tool.cell_size * 0.5;
    
    // Distance from tool center
    let dx = cell_x - tool.pos_x;
    let dz = cell_z - tool.pos_z;
    let dist = sqrt(dx * dx + dz * dz);
    
    if (dist > tool.radius) { return; }
    
    // Falloff: stronger at center
    let falloff = 1.0 - (dist / tool.radius);
    let delta = tool.amount * tool.dt * falloff;
    
    // Apply to appropriate material layer
    if (tool.material_type == 0u) {
        // Sediment
        if (delta > 0.0) {
            sediment[idx] += delta;
        } else {
            sediment[idx] = max(0.0, sediment[idx] + delta);
        }
    } else if (tool.material_type == 1u) {
        // Overburden
        if (delta > 0.0) {
            overburden[idx] += delta;
        } else {
            overburden[idx] = max(0.0, overburden[idx] + delta);
        }
    } else if (tool.material_type == 2u) {
        // Gravel
        if (delta > 0.0) {
            gravel[idx] += delta;
        } else {
            gravel[idx] = max(0.0, gravel[idx] + delta);
        }
    }
}

// Generic excavation: remove from top layer first
@compute @workgroup_size(16, 16)
fn excavate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (tool.enabled == 0u) { return; }
    
    let tile_x = global_id.x;
    let tile_z = global_id.y;
    if (tile_x >= tool.tile_width || tile_z >= tool.tile_depth) { return; }
    let x = tile_x + tool.origin_x;
    let z = tile_z + tool.origin_z;
    if (x >= tool.world_width || z >= tool.world_depth) { return; }
    
    let idx = get_idx(x, z);
    
    let cell_x = f32(x) * tool.cell_size + tool.cell_size * 0.5;
    let cell_z = f32(z) * tool.cell_size + tool.cell_size * 0.5;
    
    let dx = cell_x - tool.pos_x;
    let dz = cell_z - tool.pos_z;
    let dist = sqrt(dx * dx + dz * dz);
    
    if (dist > tool.radius) { return; }
    
    let falloff = 1.0 - (dist / tool.radius);
    var remaining = abs(tool.amount) * tool.dt * falloff;
    
    // Remove from layers top to bottom: sediment -> overburden -> gravel -> paydirt
    // 1. Sediment
    let sed_remove = min(remaining, sediment[idx]);
    sediment[idx] -= sed_remove;
    remaining -= sed_remove;
    
    // 2. Overburden
    if (remaining > 0.0) {
        let ob_remove = min(remaining, overburden[idx]);
        overburden[idx] -= ob_remove;
        remaining -= ob_remove;
    }
    
    // 3. Gravel
    if (remaining > 0.0) {
        let gr_remove = min(remaining, gravel[idx]);
        gravel[idx] -= gr_remove;
        remaining -= gr_remove;
    }
    
    // 4. Paydirt (don't remove bedrock)
    if (remaining > 0.0) {
        let pd_remove = min(remaining, paydirt[idx]);
        paydirt[idx] -= pd_remove;
    }
}
