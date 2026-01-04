// SDF Collision Shader (3D)
//
// Advects particles and resolves solid collisions using a precomputed SDF.
// Mirrors the CPU collision logic in sim3d::advection.

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sdf: array<f32>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn sdf_at(i: i32, j: i32, k: i32) -> f32 {
    if (i < 0 || i >= i32(params.width)) { return -params.cell_size; }
    if (j < 0 || j >= i32(params.height)) { return -params.cell_size; }
    if (k < 0 || k >= i32(params.depth)) { return -params.cell_size; }
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

fn sdf_gradient(pos: vec3<f32>) -> vec3<f32> {
    let eps = params.cell_size * 0.1;
    let dx = sample_sdf(pos + vec3<f32>(eps, 0.0, 0.0)) - sample_sdf(pos - vec3<f32>(eps, 0.0, 0.0));
    let dy = sample_sdf(pos + vec3<f32>(0.0, eps, 0.0)) - sample_sdf(pos - vec3<f32>(0.0, eps, 0.0));
    let dz = sample_sdf(pos + vec3<f32>(0.0, 0.0, eps)) - sample_sdf(pos - vec3<f32>(0.0, 0.0, eps));

    let grad = vec3<f32>(dx, dy, dz);
    let len = length(grad);
    if (len > 1e-6) {
        return grad / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

@compute @workgroup_size(256)
fn sdf_collision(@builtin(global_invocation_id) id: vec3<u32>) {
    let pid = id.x;
    if (pid >= params.particle_count) {
        return;
    }

    var pos = positions[pid].xyz;
    var vel = velocities[pid].xyz;

    // Euler advection
    pos += vel * params.dt;

    // SDF collision
    let dist = sample_sdf(pos);
    if (dist < 0.0) {
        let normal = sdf_gradient(pos);
        let penetration = -dist + params.cell_size * 0.1;
        pos += normal * penetration;

        let vel_into = dot(vel, normal);
        if (vel_into < 0.0) {
            vel -= normal * vel_into * 1.1;
        }
    }

    positions[pid] = vec4<f32>(pos, 0.0);
    velocities[pid] = vec4<f32>(vel, 0.0);
}
