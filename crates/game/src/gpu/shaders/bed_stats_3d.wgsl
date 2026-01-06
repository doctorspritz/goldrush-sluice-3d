// Bed stats + probe accumulation (3D).
// Keep probe offsets in sync with gpu/bed_3d.rs.

const SEDIMENT_DENSITY_THRESHOLD: f32 = 1.0;
const STAT_SCALE: f32 = 1000.0;
const VEL_SCALE: f32 = 1000.0;

const MATERIAL_STRIDE: u32 = 10u;
const ZONE_STRIDE: u32 = MATERIAL_STRIDE * 2u;
const ZONE_RIFFLE: u32 = 0u;
const ZONE_DOWNSTREAM: u32 = 1u;
const THROUGHPUT_OFFSET: u32 = ZONE_STRIDE * 2u;

const STAT_COUNT: u32 = 0u;
const STAT_SUM_Y: u32 = 1u;
const STAT_MAX_Y: u32 = 2u;
const STAT_SUM_VY: u32 = 3u;
const STAT_SDF_NEG: u32 = 4u;
const STAT_BELOW_BED: u32 = 5u;
const STAT_ABOVE_BED: u32 = 6u;
const STAT_UP: u32 = 7u;
const STAT_SUM_OFFSET: u32 = 8u;
const STAT_MAX_OFFSET: u32 = 9u;

const THROUGHPUT_TOTAL: u32 = 0u;
const THROUGHPUT_UPSTREAM: u32 = 1u;
const THROUGHPUT_AT_RIFFLE: u32 = 2u;
const THROUGHPUT_DOWNSTREAM: u32 = 3u;
const THROUGHPUT_MAX_X: u32 = 4u;
const THROUGHPUT_MAX_Y: u32 = 5u;
const THROUGHPUT_LOFTED: u32 = 6u;

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    sample_height: f32,
    bed_air_margin: f32,
    loft_height: f32,
    riffle_min_i: i32,
    riffle_max_i: i32,
    downstream_min_i: i32,
    downstream_max_i: i32,
    riffle_start_x: f32,
    riffle_end_x: f32,
    downstream_x: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> densities: array<f32>;
@group(0) @binding(4) var<storage, read> bed_height: array<f32>;
@group(0) @binding(5) var<storage, read> sdf: array<f32>;
@group(0) @binding(6) var<storage, read_write> bed_water_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> bed_sediment_count: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> bed_water_vel_sum: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> probe_stats: array<atomic<i32>>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn bed_index(i: i32, k: i32) -> u32 {
    return u32(k) * params.width + u32(i);
}

fn bed_height_at(i: i32, k: i32) -> f32 {
    let ii = clamp(i, 0, i32(params.width) - 1);
    let kk = clamp(k, 0, i32(params.depth) - 1);
    return bed_height[bed_index(ii, kk)];
}

fn sdf_at(i: i32, j: i32, k: i32) -> f32 {
    if (k < 0 || k >= i32(params.depth)) { return -params.cell_size; }
    if (i < 0 || j < 0) { return -params.cell_size; }
    if (i >= i32(params.width) || j >= i32(params.height)) { return params.cell_size; }
    return sdf[cell_index(u32(i), u32(j), u32(k))];
}

fn sample_sdf(pos: vec3<f32>) -> f32 {
    let inv_dx = 1.0 / params.cell_size;
    let fx = pos.x * inv_dx - 0.5;
    let fy = pos.y * inv_dx - 0.5;
    let fz = pos.z * inv_dx - 0.5;

    let i0 = i32(floor(fx));
    let j0 = i32(floor(fy));
    let k0 = i32(floor(fz));

    let tx = fx - f32(i0);
    let ty = fy - f32(j0);
    let tz = fz - f32(k0);

    var result = 0.0;
    for (var dk: i32 = 0; dk < 2; dk++) {
        for (var dj: i32 = 0; dj < 2; dj++) {
            for (var di: i32 = 0; di < 2; di++) {
                let ii = i0 + di;
                let jj = j0 + dj;
                let kk = k0 + dk;

                let val = sdf_at(ii, jj, kk);

                let wx = select(tx, 1.0 - tx, di == 0);
                let wy = select(ty, 1.0 - ty, dj == 0);
                let wz = select(tz, 1.0 - tz, dk == 0);

                result += val * wx * wy * wz;
            }
        }
    }

    return result;
}

fn to_fixed(value: f32) -> i32 {
    return i32(round(value * STAT_SCALE));
}

fn to_vel_fixed(value: f32) -> i32 {
    return i32(round(value * VEL_SCALE));
}

fn probe_index(zone: u32, material: u32, stat: u32) -> u32 {
    return zone * ZONE_STRIDE + material * MATERIAL_STRIDE + stat;
}

fn add_probe_stats(zone: u32, material: u32, pos: vec3<f32>, vel: vec3<f32>, bed: f32, sdf_val: f32) {
    let base = zone * ZONE_STRIDE + material * MATERIAL_STRIDE;
    let offset = pos.y - bed;
    let below_bed = pos.y < bed;
    let above_bed = offset > params.bed_air_margin;
    let moving_up = vel.y > 0.0;

    atomicAdd(&probe_stats[base + STAT_COUNT], 1);
    atomicAdd(&probe_stats[base + STAT_SUM_Y], to_fixed(pos.y));
    atomicMax(&probe_stats[base + STAT_MAX_Y], to_fixed(pos.y));
    atomicAdd(&probe_stats[base + STAT_SUM_VY], to_fixed(vel.y));
    if (sdf_val < 0.0) {
        atomicAdd(&probe_stats[base + STAT_SDF_NEG], 1);
    }
    if (below_bed) {
        atomicAdd(&probe_stats[base + STAT_BELOW_BED], 1);
    }
    if (above_bed) {
        atomicAdd(&probe_stats[base + STAT_ABOVE_BED], 1);
    }
    if (moving_up) {
        atomicAdd(&probe_stats[base + STAT_UP], 1);
    }
    atomicAdd(&probe_stats[base + STAT_SUM_OFFSET], to_fixed(offset));
    atomicMax(&probe_stats[base + STAT_MAX_OFFSET], to_fixed(offset));
}

@compute @workgroup_size(256)
fn bed_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.particle_count) {
        return;
    }

    let pos = positions[gid.x];
    let vel = velocities[gid.x];
    let density = densities[gid.x];
    let cell_size = params.cell_size;

    let i = clamp(i32(pos.x / cell_size), 0, i32(params.width) - 1);
    let k = clamp(i32(pos.z / cell_size), 0, i32(params.depth) - 1);
    let idx = bed_index(i, k);
    let bed = bed_height_at(i, k);

    let in_sample = pos.y >= bed && pos.y <= bed + params.sample_height;
    if (in_sample) {
        if (density > SEDIMENT_DENSITY_THRESHOLD) {
            atomicAdd(&bed_sediment_count[idx], 1);
        } else {
            atomicAdd(&bed_water_count[idx], 1);
            let base = idx * 3u;
            atomicAdd(&bed_water_vel_sum[base + 0u], to_vel_fixed(vel.x));
            atomicAdd(&bed_water_vel_sum[base + 1u], to_vel_fixed(vel.y));
            atomicAdd(&bed_water_vel_sum[base + 2u], to_vel_fixed(vel.z));
        }
    }

    let in_riffle = i >= params.riffle_min_i && i <= params.riffle_max_i;
    let in_downstream = i >= params.downstream_min_i && i <= params.downstream_max_i;
    if (in_riffle || in_downstream) {
        let sdf_val = sample_sdf(pos);
        let material = select(0u, 1u, density > SEDIMENT_DENSITY_THRESHOLD);
        if (in_riffle) {
            add_probe_stats(ZONE_RIFFLE, material, pos, vel, bed, sdf_val);
        }
        if (in_downstream) {
            add_probe_stats(ZONE_DOWNSTREAM, material, pos, vel, bed, sdf_val);
        }
    }

    if (density > SEDIMENT_DENSITY_THRESHOLD) {
        atomicAdd(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_TOTAL], 1);

        if (pos.x < params.riffle_start_x) {
            atomicAdd(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_UPSTREAM], 1);
        } else if (pos.x <= params.riffle_end_x) {
            atomicAdd(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_AT_RIFFLE], 1);
        } else if (pos.x > params.downstream_x) {
            atomicAdd(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_DOWNSTREAM], 1);
        }

        atomicMax(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_MAX_X], to_fixed(pos.x));
        atomicMax(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_MAX_Y], to_fixed(pos.y));

        if (vel.y > 0.0 && pos.y > bed + params.loft_height) {
            atomicAdd(&probe_stats[THROUGHPUT_OFFSET + THROUGHPUT_LOFTED], 1);
        }
    }
}
