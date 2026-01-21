// heightfield_sdf_3d.wgsl
// Convert heightfield layers into a 3D signed distance field.

struct Params {
    grid_dims: vec4<u32>, // width, height, depth, pad
    cell_size: f32,
    _pad0: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;

// Terrain layers (read-only here, but layout uses read_write for compatibility)
@group(1) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(1) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(1) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(1) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(1) @binding(4) var<storage, read_write> sediment: array<f32>;
@group(1) @binding(5) var<storage, read_write> surface_material: array<u32>; // Unused

@group(2) @binding(0) var<storage, read_write> sdf: array<f32>;

fn idx_2d(x: u32, z: u32) -> u32 {
    return z * params.grid_dims.x + x;
}

fn idx_3d(x: u32, y: u32, z: u32) -> u32 {
    return z * params.grid_dims.x * params.grid_dims.y + y * params.grid_dims.x + x;
}

fn height_at(idx: u32) -> f32 {
    return bedrock[idx] + paydirt[idx] + gravel[idx] + overburden[idx] + sediment[idx];
}

@compute @workgroup_size(8, 8, 8)
fn heightfield_to_sdf(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= params.grid_dims.x || y >= params.grid_dims.y || z >= params.grid_dims.z) {
        return;
    }

    let column_idx = idx_2d(x, z);
    let height = height_at(column_idx);
    let y_center = (f32(y) + 0.5) * params.cell_size;

    let sdf_idx = idx_3d(x, y, z);
    sdf[sdf_idx] = y_center - height;
}
