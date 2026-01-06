// Bed height update (3D).

struct Params {
    width: u32,
    depth: u32,
    _pad0: vec2<u32>,
    cell_size: f32,
    dt: f32,
    bed_porosity: f32,
    max_bed_height: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bed_flux_x: array<f32>;
@group(0) @binding(2) var<storage, read> bed_flux_z: array<f32>;
@group(0) @binding(3) var<storage, read> bed_desired_delta: array<f32>;
@group(0) @binding(4) var<storage, read> bed_base_height: array<f32>;
@group(0) @binding(5) var<storage, read_write> bed_height: array<f32>;

@compute @workgroup_size(256)
fn bed_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let column_count = params.width * params.depth;
    if (idx >= column_count) {
        return;
    }

    let i = idx % params.width;
    let k = idx / params.width;

    let fx_p = select(0.0, bed_flux_x[idx + 1u], i + 1u < params.width);
    let fx_m = select(0.0, bed_flux_x[idx - 1u], i > 0u);
    let fz_p = select(0.0, bed_flux_z[idx + params.width], k + 1u < params.depth);
    let fz_m = select(0.0, bed_flux_z[idx - params.width], k > 0u);

    let div = (fx_p - fx_m + fz_p - fz_m) / (2.0 * params.cell_size);
    let bedload_delta = -div * params.dt / (1.0 - params.bed_porosity);

    let updated = bed_height[idx] + bed_desired_delta[idx] + bedload_delta;
    bed_height[idx] = clamp(updated, bed_base_height[idx], params.max_bed_height);
}
