// Bed flux + deposition/entrainment (3D).

const VEL_SCALE: f32 = 1000.0;

struct Params {
    width: u32,
    depth: u32,
    _pad0: vec2<u32>,
    cell_size: f32,
    dt: f32,
    bed_friction: f32,
    sediment_rel_density: f32,
    water_density: f32,
    sediment_grain_diameter: f32,
    shields_critical: f32,
    shields_smooth: f32,
    bedload_coeff: f32,
    entrainment_coeff: f32,
    sediment_settling_velocity: f32,
    bed_porosity: f32,
    sediment_rest_particles: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bed_water_count: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> bed_sediment_count: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read> bed_water_vel_sum: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> bed_base_height: array<f32>;
@group(0) @binding(5) var<storage, read> bed_height: array<f32>;
@group(0) @binding(6) var<storage, read_write> bed_flux_x: array<f32>;
@group(0) @binding(7) var<storage, read_write> bed_flux_z: array<f32>;
@group(0) @binding(8) var<storage, read_write> bed_desired_delta: array<f32>;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn smooth_positive(x: f32, width: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    if (width <= 0.0) {
        return x;
    }
    return x * smoothstep(0.0, width, x);
}

@compute @workgroup_size(256)
fn bed_flux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let column_count = params.width * params.depth;
    if (idx >= column_count) {
        return;
    }

    let water_count = max(atomicLoad(&bed_water_count[idx]), 0);
    let sediment_count = max(atomicLoad(&bed_sediment_count[idx]), 0);
    let base = idx * 3u;
    let sum_x = f32(atomicLoad(&bed_water_vel_sum[base + 0u])) / VEL_SCALE;
    let sum_y = f32(atomicLoad(&bed_water_vel_sum[base + 1u])) / VEL_SCALE;
    let sum_z = f32(atomicLoad(&bed_water_vel_sum[base + 2u])) / VEL_SCALE;

    var avg_vel = vec3<f32>(0.0, 0.0, 0.0);
    if (water_count > 0) {
        let inv = 1.0 / f32(water_count);
        avg_vel = vec3<f32>(sum_x, sum_y, sum_z) * inv;
    }

    let speed = length(avg_vel);
    let density_diff = (params.sediment_rel_density - 1.0) * params.water_density;
    var theta = 0.0;
    if (speed > 0.0) {
        let tau = params.water_density * params.bed_friction * speed * speed;
        theta = tau / (density_diff * 9.81 * params.sediment_grain_diameter);
    }

    let excess = smooth_positive(theta - params.shields_critical, params.shields_smooth);
    let available_height = max(bed_height[idx] - bed_base_height[idx], 0.0);
    let availability = clamp(available_height / (params.cell_size * 2.0), 0.0, 1.0);
    let bedload_mag = params.bedload_coeff * pow(excess, 1.5) * availability;

    var flow_dir = vec3<f32>(0.0, 0.0, 0.0);
    if (speed > 1e-3) {
        flow_dir = normalize(vec3<f32>(avg_vel.x, 0.0, avg_vel.z));
    }
    bed_flux_x[idx] = flow_dir.x * bedload_mag;
    bed_flux_z[idx] = flow_dir.z * bedload_mag;

    let total = water_count + sediment_count;
    let sediment_conc = select(0.0, f32(sediment_count) / f32(total), total > 0);
    let shear_factor = 1.0 - smoothstep(params.shields_critical * 0.7, params.shields_critical * 1.3, theta);
    let deposit_rate = params.sediment_settling_velocity * sediment_conc * shear_factor;
    let entrain_rate = params.entrainment_coeff * excess;
    var desired_delta = (deposit_rate - entrain_rate) * params.dt;

    let particle_height = params.cell_size / (params.sediment_rest_particles * (1.0 - params.bed_porosity));
    let max_deposit = f32(sediment_count) * particle_height;
    if (desired_delta > max_deposit) {
        desired_delta = max_deposit;
    }

    bed_desired_delta[idx] = desired_delta;
}
