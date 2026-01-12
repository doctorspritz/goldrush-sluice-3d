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
    // Bitmask for open boundaries (particles can exit without clamping):
    // Bit 0 (1): -X open, Bit 1 (2): +X open
    // Bit 2 (4): -Y open, Bit 3 (8): +Y open
    // Bit 4 (16): -Z open, Bit 5 (32): +Z open
    open_boundaries: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sdf: array<f32>;
@group(0) @binding(4) var<storage, read> bed_height: array<f32>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read> cell_type: array<u32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn is_cell_solid(i: i32, j: i32, k: i32) -> bool {
    // Check open boundary flags for out-of-bounds cells
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // X bounds - return false (not solid) if boundary is open
    if (i < 0) { return !open_neg_x; }
    if (i >= i32(params.width)) { return !open_pos_x; }
    // Y bounds - ceiling/floor
    if (j < 0) { return !open_neg_y; }
    if (j >= i32(params.height)) { return !open_pos_y; }
    // Z bounds
    if (k < 0) { return !open_neg_z; }
    if (k >= i32(params.depth)) { return !open_pos_z; }

    return cell_type[cell_index(u32(i), u32(j), u32(k))] == CELL_SOLID;
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
    let density = densities[pid];
    let is_sediment = density > 1.0;

    // Euler advection
    pos += vel * params.dt;

    // SDF collision (sediment handled by DEM)
    if (!is_sediment) {
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
    }

    // Check if particle would enter a jammed sediment cell (voxel collision)
    // Skip for sediment particles - DEM handles their collision properly
    if (!is_sediment) {
        let cell_i = i32(pos.x / params.cell_size);
        let cell_j = i32(pos.y / params.cell_size);
        let cell_k = i32(pos.z / params.cell_size);

        if (is_cell_solid(cell_i, cell_j, cell_k)) {
            // Particle is trying to enter a jammed cell - push it back
            let old_pos = positions[pid].xyz;

            // Find the nearest non-solid cell by checking neighbors
            var best_pos = old_pos;
            var found_valid = false;

            for (var dj: i32 = 1; dj <= 2; dj++) {
                for (var dk: i32 = -1; dk <= 1; dk++) {
                    for (var di: i32 = -1; di <= 1; di++) {
                        let test_i = cell_i + di;
                        let test_j = cell_j + dj;  // Prefer upward
                        let test_k = cell_k + dk;

                        if (!is_cell_solid(test_i, test_j, test_k)) {
                            let test_pos = vec3<f32>(
                                (f32(test_i) + 0.5) * params.cell_size,
                                (f32(test_j) + 0.5) * params.cell_size,
                                (f32(test_k) + 0.5) * params.cell_size
                            );
                            best_pos = test_pos;
                            found_valid = true;
                            break;
                        }
                    }
                    if (found_valid) { break; }
                }
                if (found_valid) { break; }
            }

            pos = best_pos;

            // Damp velocity when hitting jammed sediment
            vel *= 0.3;
        }
    }

    // Boundary clamping - respects open_boundaries bitmask
    // Open boundaries allow particles to exit (for transfer to adjacent grids)
    let margin = params.cell_size * 0.1;
    let max_x = f32(params.width) * params.cell_size - margin;
    let max_y = f32(params.height) * params.cell_size - margin;
    let max_z = f32(params.depth) * params.cell_size - margin;

    // Check open boundary flags
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // Floor (Y min) - usually closed
    if (pos.y < margin && !open_neg_y) {
        pos.y = margin;
        if (vel.y < 0.0) { vel.y = 0.0; }
    }

    // X boundaries
    if (pos.x < margin && !open_neg_x) {
        pos.x = margin;
        if (vel.x < 0.0) { vel.x = 0.0; }
    }
    if (pos.x > max_x && !open_pos_x) {
        pos.x = max_x;
        if (vel.x > 0.0) { vel.x = 0.0; }
    }

    // Y ceiling
    if (pos.y > max_y && !open_pos_y) {
        pos.y = max_y;
        if (vel.y > 0.0) { vel.y = 0.0; }
    }

    // Z boundaries
    if (pos.z < margin && !open_neg_z) {
        pos.z = margin;
        if (vel.z < 0.0) { vel.z = 0.0; }
    }
    if (pos.z > max_z && !open_pos_z) {
        pos.z = max_z;
        if (vel.z > 0.0) { vel.z = 0.0; }
    }

    // Sediment bed collision disabled - let sediment flow freely like water
    // let density = densities[pid];
    // if (density > 1.0) {
    //     let bed = sample_bed_height(pos);
    //     if (pos.y < bed) {
    //         pos.y = bed + params.cell_size * 0.05;
    //         if (vel.y < 0.0) {
    //             vel.y = 0.0;
    //         }
    //         vel.x *= 0.7;
    //         vel.z *= 0.7;
    //     }
    // }

    positions[pid] = vec4<f32>(pos, 1.0);
    velocities[pid] = vec4<f32>(vel, 0.0);
}
