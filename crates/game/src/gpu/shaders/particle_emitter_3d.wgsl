// 3D Particle Emitter Shader
// Spawns new particles into the FLIP simulation.

struct EmitterParams {
    pos: vec3<f32>,
    radius: f32,
    vel: vec3<f32>,
    spread: f32,
    count: u32,
    density: f32,
    time: f32,
    max_particles: u32,
}

@group(0) @binding(0) var<uniform> params: EmitterParams;
@group(0) @binding(1) var<storage, read_write> particle_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densities: array<f32>;

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

@compute @workgroup_size(64)
fn spawn_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) { return; }

    // Atomic increment to find a slot
    // Note: This relies on the system having some "live" particles and adding to the end.
    // In a production system we'd manage a free list or compacting buffer.
    let pid = atomicAdd(&particle_count, 1u);
    
    if (pid >= params.max_particles) {
        // Buffer full, just revert or ignore
        // atomicSub(&particle_count, 1u); // Risk of underflow if many threads do this
        return;
    }

    let seed = u32(params.time * 60.0) + i;
    let offset = rand_vec3(seed) * params.radius;
    let jitter = rand_vec3(seed + 777u) * params.spread;

    positions[pid] = vec4<f32>(params.pos + offset, 1.0); // W=1.0 for "active"
    velocities[pid] = vec4<f32>(params.vel + jitter, 0.0);
    densities[pid] = params.density;
}
