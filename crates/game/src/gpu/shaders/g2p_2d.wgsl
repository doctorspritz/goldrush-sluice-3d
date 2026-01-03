struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    flip_ratio: f32,
    dt: f32,
    max_velocity: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> grid_u: array<f32>;
@group(0) @binding(4) var<storage, read> grid_v: array<f32>;
@group(0) @binding(5) var<storage, read> grid_u_old: array<f32>;
@group(0) @binding(6) var<storage, read> grid_v_old: array<f32>;

fn quadratic_bspline_1d(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.5) {
        return 0.75 - ax * ax;
    } else if (ax < 1.5) {
        let t = 1.5 - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

fn u_index(i: u32, j: u32) -> u32 {
    return j * (params.width + 1u) + i;
}

fn v_index(i: u32, j: u32) -> u32 {
    return j * params.width + i;
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let pos = positions[id.x].xy;
    let old_vel = velocities[id.x].xy;

    let width = i32(params.width);
    let height = i32(params.height);
    let cell_size = params.cell_size;

    var pic_vel = vec2<f32>(0.0, 0.0);
    var old_grid = vec2<f32>(0.0, 0.0);
    var weight_sum = vec2<f32>(0.0, 0.0);

    // U component
    let u_pos = pos / cell_size - vec2<f32>(0.0, 0.5);
    let base_u = vec2<i32>(floor(u_pos));
    let frac_u = u_pos - vec2<f32>(base_u);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base_u.y + dj;
        if (nj < 0 || nj >= height) { continue; }
        let wy = quadratic_bspline_1d(frac_u.y - f32(dj));

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_u.x + di;
            if (ni < 0 || ni > width) { continue; }
            let wx = quadratic_bspline_1d(frac_u.x - f32(di));
            let w = wx * wy;
            if (w <= 0.0) { continue; }

            let idx = u_index(u32(ni), u32(nj));
            pic_vel.x += w * grid_u[idx];
            old_grid.x += w * grid_u_old[idx];
            weight_sum.x += w;
        }
    }

    // V component
    let v_pos = pos / cell_size - vec2<f32>(0.5, 0.0);
    let base_v = vec2<i32>(floor(v_pos));
    let frac_v = v_pos - vec2<f32>(base_v);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base_v.y + dj;
        if (nj < 0 || nj > height) { continue; }
        let wy = quadratic_bspline_1d(frac_v.y - f32(dj));

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_v.x + di;
            if (ni < 0 || ni >= width) { continue; }
            let wx = quadratic_bspline_1d(frac_v.x - f32(di));
            let w = wx * wy;
            if (w <= 0.0) { continue; }

            let idx = v_index(u32(ni), u32(nj));
            pic_vel.y += w * grid_v[idx];
            old_grid.y += w * grid_v_old[idx];
            weight_sum.y += w;
        }
    }

    if (weight_sum.x > 0.0) {
        pic_vel.x = pic_vel.x / weight_sum.x;
        old_grid.x = old_grid.x / weight_sum.x;
    } else {
        pic_vel.x = old_vel.x;
        old_grid.x = old_vel.x;
    }

    if (weight_sum.y > 0.0) {
        pic_vel.y = pic_vel.y / weight_sum.y;
        old_grid.y = old_grid.y / weight_sum.y;
    } else {
        pic_vel.y = old_vel.y;
        old_grid.y = old_vel.y;
    }

    let flip_vel = old_vel + (pic_vel - old_grid);
    let vel = pic_vel * (1.0 - params.flip_ratio) + flip_vel * params.flip_ratio;

    let speed = length(vel);
    let clamped = select(vel, vel * (params.max_velocity / speed), speed > params.max_velocity);

    velocities[id.x] = vec4<f32>(clamped, 0.0, 0.0);
}
