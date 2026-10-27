// P2G Divide Shader (3D) - Normalize accumulated momentum by weight
//
// After scatter pass, each grid node has:
//   sum = Σ(momentum_i * weight_i) encoded as i32
//   weight = Σ(weight_i) encoded as i32
//
// This shader computes: velocity = (sum / SCALE) / (weight / SCALE) = sum / weight
//
// Three entry points:
// - divide_u: processes U velocity grid ((width+1) x height x depth)
// - divide_v: processes V velocity grid (width x (height+1) x depth)
// - divide_w: processes W velocity grid (width x height x (depth+1))

const INV_SCALE: f32 = 0.00001;

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count:u32, include_sediment:u32, paired_water:u32, pad1:u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> u_sum: array<i32>;
@group(0) @binding(2) var<storage, read> u_weight: array<i32>;
@group(0) @binding(3) var<storage, read> v_sum: array<i32>;
@group(0) @binding(4) var<storage, read> v_weight: array<i32>;
@group(0) @binding(5) var<storage, read> w_sum: array<i32>;
@group(0) @binding(6) var<storage, read> w_weight: array<i32>;
@group(0) @binding(7) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(8) var<storage, read_write> grid_v: array<f32>;
@group(0) @binding(9) var<storage, read_write> grid_w: array<f32>;

fn water_offset(n:u32)->u32 {return ((n+63u)/64u)*64u;}
@compute @workgroup_size(8, 8, 4)
fn divide_u(@builtin(global_invocation_id) id: vec3<u32>) {
    // U grid is (width+1) x height x depth
    if (id.x > params.width || id.y >= params.height || id.z >= params.depth) {
        return;
    }

    let idx = id.z * (params.width + 1u) * params.height + id.y * (params.width + 1u) + id.x;
    var integer_sum=u_sum[idx];var integer_weight=u_weight[idx];
    if params.paired_water!=0u {
        let water_idx=idx+water_offset((params.width+1u)*params.height*params.depth);
        integer_sum+=u_sum[water_idx];integer_weight+=u_weight[water_idx];
    }
    let sum=f32(integer_sum)*INV_SCALE;let w=f32(integer_weight)*INV_SCALE;

    if (w > 0.0) {
        grid_u[idx] = sum / w;
    } else {
        grid_u[idx] = 0.0;
    }
    if params.paired_water!=0u {
        let water_idx=idx+water_offset((params.width+1u)*params.height*params.depth);
        let sum=f32(u_sum[water_idx])*INV_SCALE;let weight=f32(u_weight[water_idx])*INV_SCALE;
        if weight>0.0 {grid_u[water_idx]=sum/weight;}else{grid_u[water_idx]=0.0;}
        // Supported stationary water is distinct from an unsupported zero.
        grid_u[water_idx+(params.width+1u)*params.height*params.depth]=select(0.0,1.0,u_weight[water_idx]>0);
    } else if params.include_sediment==0u {
        grid_u[idx+(params.width+1u)*params.height*params.depth]=select(0.0,1.0,integer_weight>0);
    }
}

@compute @workgroup_size(8, 8, 4)
fn divide_v(@builtin(global_invocation_id) id: vec3<u32>) {
    // V grid is width x (height+1) x depth
    if (id.x >= params.width || id.y > params.height || id.z >= params.depth) {
        return;
    }

    let idx = id.z * params.width * (params.height + 1u) + id.y * params.width + id.x;
    var integer_sum=v_sum[idx];var integer_weight=v_weight[idx];
    if params.paired_water!=0u {
        let water_idx=idx+water_offset(params.width*(params.height+1u)*params.depth);
        integer_sum+=v_sum[water_idx];integer_weight+=v_weight[water_idx];
    }
    let sum=f32(integer_sum)*INV_SCALE;let w=f32(integer_weight)*INV_SCALE;

    if (w > 0.0) {
        grid_v[idx] = sum / w;
    } else {
        grid_v[idx] = 0.0;
    }
    if params.paired_water!=0u {
        let water_idx=idx+water_offset(params.width*(params.height+1u)*params.depth);
        let sum=f32(v_sum[water_idx])*INV_SCALE;let weight=f32(v_weight[water_idx])*INV_SCALE;
        if weight>0.0 {grid_v[water_idx]=sum/weight;}else{grid_v[water_idx]=0.0;}
        // Supported stationary water is distinct from an unsupported zero.
        grid_v[water_idx+params.width*(params.height+1u)*params.depth]=select(0.0,1.0,v_weight[water_idx]>0);
    } else if params.include_sediment==0u {
        grid_v[idx+params.width*(params.height+1u)*params.depth]=select(0.0,1.0,integer_weight>0);
    }
}

@compute @workgroup_size(8, 8, 4)
fn divide_w(@builtin(global_invocation_id) id: vec3<u32>) {
    // W grid is width x height x (depth+1)
    if (id.x >= params.width || id.y >= params.height || id.z > params.depth) {
        return;
    }

    let idx = id.z * params.width * params.height + id.y * params.width + id.x;
    var integer_sum=w_sum[idx];var integer_weight=w_weight[idx];
    if params.paired_water!=0u {
        let water_idx=idx+water_offset(params.width*params.height*(params.depth+1u));
        integer_sum+=w_sum[water_idx];integer_weight+=w_weight[water_idx];
    }
    let sum=f32(integer_sum)*INV_SCALE;let w=f32(integer_weight)*INV_SCALE;

    if (w > 0.0) {
        grid_w[idx] = sum / w;
    } else {
        grid_w[idx] = 0.0;
    }
    if params.paired_water!=0u {
        let water_idx=idx+water_offset(params.width*params.height*(params.depth+1u));
        let sum=f32(w_sum[water_idx])*INV_SCALE;let weight=f32(w_weight[water_idx])*INV_SCALE;
        if weight>0.0 {grid_w[water_idx]=sum/weight;}else{grid_w[water_idx]=0.0;}
        // Supported stationary water is distinct from an unsupported zero.
        grid_w[water_idx+params.width*params.height*(params.depth+1u)]=select(0.0,1.0,w_weight[water_idx]>0);
    } else if params.include_sediment==0u {
        grid_w[idx+params.width*params.height*(params.depth+1u)]=select(0.0,1.0,integer_weight>0);
    }
}
