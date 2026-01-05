// Advect 3D Shader - GPU particle advection
//
// Integrates particle positions using velocity: pos += vel * dt
// Also applies density correction deltas if provided.
//
// This keeps particles fully on GPU - no CPU round-trip needed.

struct Params {
    dt: f32,
    particle_count: u32,
    apply_density_correction: u32,  // 0 = no, 1 = yes
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec3<f32>>;
// Density correction deltas (vec4 for alignment, only xyz used)
@group(0) @binding(3) var<storage, read> position_deltas: array<vec4<f32>>;

@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let idx = id.x;
    var pos = positions[idx];
    let vel = velocities[idx];

    // Euler integration
    pos += vel * params.dt;

    // Apply density correction if enabled
    if (params.apply_density_correction != 0u) {
        let delta = position_deltas[idx];
        pos.x += delta.x;
        pos.y += delta.y;
        pos.z += delta.z;
    }

    positions[idx] = pos;
}
