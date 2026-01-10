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
@group(0) @binding(1) var<storage, read_write> cell_type: array<u32>;
@group(0) @binding(2) var<storage, read> obstacles: array<GravelObstacle>;

const CELL_SOLID: u32 = 2u;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

@compute @workgroup_size(8, 8, 4)
fn build_gravel_obstacles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);
    if (cell_type[idx] == CELL_SOLID) {
        return;
    }

    let center = vec3<f32>(
        (f32(i) + 0.5) * params.cell_size,
        (f32(j) + 0.5) * params.cell_size,
        (f32(k) + 0.5) * params.cell_size,
    );

    var n: u32 = 0u;
    loop {
        if (n >= params.obstacle_count) {
            break;
        }
        let obs = obstacles[n];
        let d = center - obs.position;
        if (dot(d, d) <= obs.radius * obs.radius) {
            cell_type[idx] = CELL_SOLID;
            break;
        }
        n = n + 1u;
    }
}
