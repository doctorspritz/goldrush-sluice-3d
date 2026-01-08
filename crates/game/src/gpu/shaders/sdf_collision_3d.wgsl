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
@group(0) @binding(4) var<storage, read> bed_height: array<f32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;

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

fn sample_bed_height(pos: vec3<f32>) -> f32 {
    let inv_dx = 1.0 / params.cell_size;
    let fx = pos.x * inv_dx - 0.5;
    let fz = pos.z * inv_dx - 0.5;

    let i0 = i32(floor(fx));
    let k0 = i32(floor(fz));

    let tx = fx - f32(i0);
    let tz = fz - f32(k0);

    let h00 = bed_height_at(i0, k0);
    let h10 = bed_height_at(i0 + 1, k0);
    let h01 = bed_height_at(i0, k0 + 1);
    let h11 = bed_height_at(i0 + 1, k0 + 1);

    let hx0 = mix(h00, h10, tx);
    let hx1 = mix(h01, h11, tx);
    return mix(hx0, hx1, tz);
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
        // Robust penetration clamping: don't push more than 2x cell_size per frame
        let penetration = min(-dist + params.cell_size * 0.1, params.cell_size * 2.0);
        pos += normal * penetration;

        let vel_into = dot(vel, normal);
        if (vel_into < 0.0) {
            vel -= normal * vel_into * 1.0; // Inelastic collision
        }
    }

    // Strict boundary clamping
    let margin = params.cell_size * 0.1;
    let max_x = f32(params.width) * params.cell_size - margin;
    let max_y = f32(params.height) * params.cell_size - margin;
    let max_z = f32(params.depth) * params.cell_size - margin;
    
    // Catch-all floor at y=0.1
    if (pos.y < margin) {
        pos.y = margin;
        if (vel.y < 0.0) { vel.y = 0.0; }
    }
    
    // Ceiling and walls
    pos.x = clamp(pos.x, margin, max_x);
    pos.y = clamp(pos.y, margin, max_y);
    pos.z = clamp(pos.z, margin, max_z);
    
    if (pos.y >= max_y - 0.01 && vel.y > 1e-3) {
        vel.y = 0.0; // Dampen upward momentum at ceiling
    }

    /* Removing redundant and uninitialized bed height check
    let bed = sample_bed_height(pos);
    if (pos.y < bed) {
        pos.y = bed + params.cell_size * 0.05;
        if (vel.y < 0.0) {
            vel.y = 0.0;
        }
        // Apply friction
        vel.x *= 0.7;
        vel.z *= 0.7;
    }
    */

    positions[pid] = vec4<f32>(pos, 1.0);
    velocities[pid] = vec4<f32>(vel, 0.0);
}
