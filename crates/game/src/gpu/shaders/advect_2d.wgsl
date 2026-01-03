struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    dt: f32,
    bounce: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;

@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    var pos = positions[id.x].xy;
    var vel = velocities[id.x].xy;

    pos = pos + vel * params.dt;

    let min_x = params.cell_size * 0.5;
    let max_x = (f32(params.width) - 0.5) * params.cell_size;
    let min_y = params.cell_size * 0.5;
    let max_y = (f32(params.height) - 0.5) * params.cell_size;

    if (pos.x < min_x) {
        pos.x = min_x;
        vel.x = abs(vel.x) * params.bounce;
    } else if (pos.x > max_x) {
        pos.x = max_x;
        vel.x = -abs(vel.x) * params.bounce;
    }

    if (pos.y < min_y) {
        pos.y = min_y;
        vel.y = abs(vel.y) * params.bounce;
    } else if (pos.y > max_y) {
        pos.y = max_y;
        vel.y = -abs(vel.y) * params.bounce;
    }

    positions[id.x] = vec4<f32>(pos, 0.0, 0.0);
    velocities[id.x] = vec4<f32>(vel, 0.0, 0.0);
}
