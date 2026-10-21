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
    cpu_sediment_contacts: u32,
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
    if (is_sediment && params.cpu_sediment_contacts != 0u) {
        // Do not cancel the normal impact before DEM can compute the normal
        // reaction and its Coulomb friction budget. DEM runs every substep.
        positions[pid] = vec4<f32>(pos, positions[pid].w);
        return;
    }

    // SDF collision for ALL particles (water and sediment)
    // Sediment also gets DEM collision for friction/inter-particle, but SDF ensures
    // geometry penetration is handled even if DEM is disabled.
    let dist = sample_sdf(pos);
    if (dist < 0.0) {
        let normal = sdf_gradient(pos);
        // Penetration clamping: allow up to 10x cell_size to escape thin geometry
        let penetration = min(-dist + params.cell_size * 0.1, params.cell_size * 10.0);
        pos += normal * penetration;

        let vel_into = dot(vel, normal);
        if (vel_into < 0.0) {
            vel -= normal * vel_into * 1.0; // Inelastic collision
        }
    }

    // Resolve voxel occupancy (including packed bed) at the nearest free face.
    // Never jump to a cell centre or damp tangential flow: those operations
    // repeatedly stranded water above riffles and manufactured visible pops.
    if (!is_sediment) {
        let cell = vec3<i32>(floor(pos / params.cell_size));
        let internal = all(cell > vec3<i32>(0)) &&
            all(cell < vec3<i32>(i32(params.width)-1, i32(params.height)-1, i32(params.depth)-1));
        if (internal && is_cell_solid(cell.x, cell.y, cell.z)) {
            let skin = params.cell_size * 0.001;
            var best = pos;
            var distance_sq = 1e20;
            // Usually one ring suffices. A second ring handles a newly packed
            // bed surrounding a particle; all directions have equal priority.
            for (var ring = 1; ring <= 2; ring++) {
                for (var z = -ring; z <= ring; z++) {
                    for (var y = -ring; y <= ring; y++) {
                        for (var x = -ring; x <= ring; x++) {
                            if (max(abs(x), max(abs(y), abs(z))) != ring) { continue; }
                            let neighbour = cell + vec3<i32>(x,y,z);
                            if (is_cell_solid(neighbour.x, neighbour.y, neighbour.z)) { continue; }
                            let lo = vec3<f32>(neighbour) * params.cell_size + vec3<f32>(skin);
                            let hi = lo + vec3<f32>(params.cell_size - 2.0 * skin);
                            let candidate = clamp(pos, lo, hi);
                            let delta = candidate - pos;
                            let d2 = dot(delta, delta);
                            if (d2 < distance_sq) { best = candidate; distance_sq = d2; }
                        }
                    }
                }
                if (distance_sq < 1e19) { break; }
            }
            if (distance_sq < 1e19) {
                let normal = normalize(best - pos);
                pos = best;
                vel -= normal * min(dot(vel, normal), 0.0);
            } else {
                // No local free volume: retain the last position rather than
                // invent an upward escape or erase all momentum.
                pos = positions[pid].xyz;
            }
        }
    }

    // Boundary clamping - respects open_boundaries bitmask
    // Open boundaries allow particles to exit (for transfer to adjacent grids)
    //
    // Grid layout (8-wide example):
    //   Cell 0: [0.0, 0.1] - SOLID wall
    //   Cell 1: [0.1, 0.2] - FLUID (first valid cell)
    //   ...
    //   Cell 6: [0.6, 0.7] - FLUID (last valid cell)
    //   Cell 7: [0.7, 0.8] - SOLID wall
    //
    // Valid fluid region: cells 1 to (W-2), positions [cell_size, (W-1)*cell_size]
    let margin = params.cell_size * 0.1;
    let min_x = params.cell_size + margin;  // Just inside cell 1
    let min_y = params.cell_size + margin;  // Just inside cell 1
    let min_z = params.cell_size + margin;  // Just inside cell 1
    let max_x = f32(params.width - 1u) * params.cell_size - margin;   // Just inside cell W-2
    let max_y = f32(params.height - 1u) * params.cell_size - margin;  // Just inside cell H-2
    let max_z = f32(params.depth - 1u) * params.cell_size - margin;   // Just inside cell D-2

    // Check open boundary flags
    let open_neg_x = (params.open_boundaries & 1u) != 0u;
    let open_pos_x = (params.open_boundaries & 2u) != 0u;
    let open_neg_y = (params.open_boundaries & 4u) != 0u;
    let open_pos_y = (params.open_boundaries & 8u) != 0u;
    let open_neg_z = (params.open_boundaries & 16u) != 0u;
    let open_pos_z = (params.open_boundaries & 32u) != 0u;

    // Floor (Y min) - usually closed
    if (pos.y < min_y && !open_neg_y) {
        pos.y = min_y;
        if (vel.y < 0.0) { vel.y = 0.0; }
    }

    // X boundaries
    if (pos.x < min_x && !open_neg_x) {
        pos.x = min_x;
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
    if (pos.z < min_z && !open_neg_z) {
        pos.z = min_z;
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

    positions[pid] = vec4<f32>(pos, positions[pid].w);
    velocities[pid] = vec4<f32>(vel, 0.0);
}
