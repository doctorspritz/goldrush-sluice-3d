// 3D Particle Emitter Shader
// Spawns new particles into the FLIP simulation.
// DENSITY-AWARE: Checks existing particle density to avoid over-packing cells.

struct EmitterParams {
    pos_radius: vec4<f32>,
    vel_spread: vec4<f32>,
    counts0: vec4<u32>,
    counts1: vec4<u32>,
    misc0: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: EmitterParams;
@group(0) @binding(1) var<storage, read_write> particle_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;
@group(0) @binding(5) var<storage, read> particle_count_grid: array<f32>;  // From P2G

fn hash(u: u32) -> u32 {
    var x = u;
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

fn rand_vec3(u: u32) -> vec3<f32> {
    let h1 = hash(u);
    let h2 = hash(h1);
    let h3 = hash(h2);
    return vec3<f32>(
        f32(h1 % 1000u) / 500.0 - 1.0,
        f32(h2 % 1000u) / 500.0 - 1.0,
        f32(h3 % 1000u) / 500.0 - 1.0
    );
}

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.counts0.z * params.counts0.w + j * params.counts0.z + i;
}

@compute @workgroup_size(64)
fn spawn_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.counts0.x) { return; }

    let seed = u32(params.misc0.y * 60.0) + i;
    let offset = rand_vec3(seed) * params.pos_radius.w;
    let spawn_pos = params.pos_radius.xyz + offset;

    // DENSITY CHECK: Don't spawn if target cell is already at or above max_ppc
    let cell_i = u32(clamp(spawn_pos.x / params.misc0.z, 0.0, f32(params.counts0.z - 1u)));
    let cell_j = u32(clamp(spawn_pos.y / params.misc0.z, 0.0, f32(params.counts0.w - 1u)));
    let cell_k = u32(clamp(spawn_pos.z / params.misc0.z, 0.0, f32(params.counts1.x - 1u)));
    let idx = cell_index(cell_i, cell_j, cell_k);
    let existing_ppc = particle_count_grid[idx];

    if (existing_ppc >= params.misc0.w) {
        // Cell is full, skip this particle
        return;
    }

    // Atomic increment to find a slot
    // Note: This relies on the system having some "live" particles and adding to the end.
    // In a production system we'd manage a free list or compacting buffer.
    let pid = atomicAdd(&particle_count, 1u);
    
    if (pid >= params.counts0.y) {
        // Buffer full, just revert or ignore
        // atomicSub(&particle_count, 1u); // Risk of underflow if many threads do this
        return;
    }

    let jitter = rand_vec3(seed + 777u) * params.vel_spread.w;

    positions[pid] = vec4<f32>(spawn_pos, 1.0); // W=1.0 for "active"
    velocities[pid] = vec4<f32>(params.vel_spread.xyz + jitter, 0.0);
    densities[pid] = params.misc0.x;
}
