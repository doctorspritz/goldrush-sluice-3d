struct Params {
    width: u32,
    height: u32,
    depth: u32,
    obstacle_count: u32,
    cell_size: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GravelObstacle {
    position: vec3<f32>,
    radius: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> sediment_fraction: array<f32>;
@group(0) @binding(2) var<storage, read> obstacles: array<GravelObstacle>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(8, 8, 4)
fn apply_gravel_porosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    let center = vec3<f32>(
        (f32(i) + 0.5) * params.cell_size,
        (f32(j) + 0.5) * params.cell_size,
        (f32(k) + 0.5) * params.cell_size,
    );

    var max_frac = sediment_fraction[idx];
    var n: u32 = 0u;
    loop {
        if (n >= params.obstacle_count) {
            break;
        }
        let obs = obstacles[n];
        let d = center - obs.position;
        let dist = length(d);
        if (dist < obs.radius) {
            let local = clamp((obs.radius - dist) / (params.cell_size * 0.5), 0.0, 1.0);
            let gravel_frac = local * 0.6;
            max_frac = max(max_frac, gravel_frac);
        }
        n = n + 1u;
    }

    sediment_fraction[idx] = max_frac;
}
