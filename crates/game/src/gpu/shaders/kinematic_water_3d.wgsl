struct Params {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    dt: f32,
    gravity: f32,
    flow_accel: f32,
    deck_slope: f32,
    deck_origin: vec3<f32>,
    deck_length: f32,
    deck_width: f32,
    water_hole_radius: f32,
    hole_spacing: f32,
    gutter_y: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> densities: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn in_water_hole(pos: vec3<f32>) -> bool {
    let local_x = pos.x - params.deck_origin.x;
    let local_z = pos.z - params.deck_origin.z;
    let gx = round(local_x / params.hole_spacing);
    let gz = round(local_z / params.hole_spacing);
    let cx = gx * params.hole_spacing;
    let cz = gz * params.hole_spacing;
    let dx = local_x - cx;
    let dz = local_z - cz;
    return dx * dx + dz * dz <= params.water_hole_radius * params.water_hole_radius;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    if (densities[idx] > 1.0) {
        return;
    }

    var pos = positions[idx].xyz;
    var vel = velocities[idx].xyz;

    vel.y = vel.y + params.gravity * params.dt;
    vel.x = vel.x + params.flow_accel * params.dt;
    pos = pos + vel * params.dt;

    let deck_x_min = params.deck_origin.x;
    let deck_x_max = params.deck_origin.x + params.deck_length;
    let deck_z_min = params.deck_origin.z;
    let deck_z_max = params.deck_origin.z + params.deck_width;

    let over_deck = pos.x >= deck_x_min && pos.x <= deck_x_max
        && pos.z >= deck_z_min && pos.z <= deck_z_max;
    if (over_deck) {
        let deck_height = params.deck_origin.y + params.deck_slope * (pos.x - params.deck_origin.x);
        let over_hole = in_water_hole(pos);
        if (over_hole && pos.y < deck_height) {
            vel.y = vel.y + params.gravity * 0.2 * params.dt;
        }
        if (pos.y < deck_height && !over_hole) {
            pos.y = deck_height + 0.002;
            if (vel.y < 0.0) {
                vel.y = 0.0;
            }
        }
    }

    if (pos.y < params.gutter_y) {
        pos.y = params.gutter_y + 0.001;
        if (vel.y < 0.0) {
            vel.y = 0.0;
        }
    }

    positions[idx] = vec4<f32>(pos, 1.0);
    velocities[idx] = vec4<f32>(vel, 0.0);
}
