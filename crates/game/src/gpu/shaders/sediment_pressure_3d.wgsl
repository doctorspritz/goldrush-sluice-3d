// Sediment Pressure Shader (3D) - Compute pressure from sediment column weight
//
// For each (x,z) column, scan from top to bottom accumulating sediment mass.
// Pressure at cell (i,j,k) = weight of all sediment above it.
// This is used for Drucker-Prager yield criterion.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
    cell_size: f32,
    particle_mass: f32,
    gravity: f32,
    buoyancy_factor: f32,  // 1 - rho_water/rho_sediment, ~0.62 for sand
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sediment_count: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> sediment_pressure: array<f32>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(8, 1, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let k = id.z;

    if (i >= params.width || k >= params.depth) {
        return;
    }

    let cell_area = params.cell_size * params.cell_size;

    // Scan from top to bottom, accumulating pressure
    var accumulated_pressure: f32 = 0.0;

    for (var j: i32 = i32(params.height) - 1; j >= 0; j--) {
        let idx = cell_index(i, u32(j), k);
        let count = f32(atomicLoad(&sediment_count[idx]));

        // Mass of sediment in this cell
        let cell_mass = count * params.particle_mass;

        // Weight (buoyancy-corrected)
        let effective_weight = cell_mass * params.gravity * params.buoyancy_factor;

        // Add to accumulated pressure (pressure = force/area)
        accumulated_pressure += effective_weight / cell_area;

        // Store pressure at this cell (pressure from sediment ABOVE, not including this cell)
        // Actually, include this cell's contribution for particles sitting in it
        sediment_pressure[idx] = accumulated_pressure;
    }
}
