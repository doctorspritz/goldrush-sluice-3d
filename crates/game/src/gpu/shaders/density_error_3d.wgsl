override PERIODIC_Z: bool = false;
override PHASE_RUNTIME_ALLOWED:bool=false;
@group(0) @binding(4) var<storage,read> represented_volume:array<f32>;
// Density Error Shader (3D)
//
// Computes density error from particle count per cell.
// Output: density_error = (count - REST_COUNT) / REST_COUNT
// This is used as the RHS for a secondary pressure solve that
// pushes particles from crowded regions to empty regions.
//
// Based on: Kugelstadt et al. 2019 "Implicit Density Projection for Volume Conserving Liquids"

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    rest_density: f32,  // Target particles per cell (~8 for typical FLIP)
    dt: f32,  // Timestep for scaling (blub divides error by dt)
    surface_clamp: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particle_count: array<i32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<storage, read_write> density_error: array<f32>;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn periodic_k(k: i32) -> u32 {
    if PERIODIC_Z { return u32(((k % i32(params.depth)) + i32(params.depth)) % i32(params.depth)); }
    return u32(k);
}

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn get_cell_type(i: i32, j: i32, k: i32) -> u32 {
    if (i < 0 || i >= i32(params.width)) { return CELL_AIR; }
    if (j < 0 || j >= i32(params.height)) { return CELL_AIR; }
    if (!PERIODIC_Z && (k < 0 || k >= i32(params.depth))) { return CELL_AIR; }
    return cell_type[cell_index(u32(i), u32(j), periodic_k(k))];
}

@compute @workgroup_size(8, 8, 4)
fn compute_density_error(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (i >= params.width || j >= params.height || k >= params.depth) {
        return;
    }

    let idx = cell_index(i, j, k);

    // Only compute for fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        density_error[idx] = 0.0;
        return;
    }

    if PHASE_RUNTIME_ALLOWED {density_error[idx]=phase_error(id);return;}

    // Start with particle count as base density
    // Note: blub computes actual density with trilinear weights, we use particle count
    var density = f32(particle_count[idx]);

    // Check neighbors for solid cells and air interfaces
    // blub contribution values: face neighbor = 0.5625 (0.75^3 / 0.75 = 0.5625 for each solid face)
    let solid_neighbor_contribution: f32 = 0.5625;

    let marker_px = get_cell_type(i32(i) + 1, i32(j), i32(k));
    let marker_mx = get_cell_type(i32(i) - 1, i32(j), i32(k));
    let marker_py = get_cell_type(i32(i), i32(j) + 1, i32(k));
    let marker_my = get_cell_type(i32(i), i32(j) - 1, i32(k));
    let marker_pz = get_cell_type(i32(i), i32(j), i32(k) + 1);
    let marker_mz = get_cell_type(i32(i), i32(j), i32(k) - 1);

    // Add density contributions from solid neighbors (blub approach)
    // If there were particles in solid cells, they would contribute this much
    if (marker_px == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_mx == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_py == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_my == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_pz == CELL_SOLID) { density += solid_neighbor_contribution; }
    if (marker_mz == CELL_SOLID) { density += solid_neighbor_contribution; }

    // Check for air interface
    let has_air_neighbor = (marker_px == CELL_AIR || marker_mx == CELL_AIR ||
                            marker_py == CELL_AIR || marker_my == CELL_AIR ||
                            marker_pz == CELL_AIR || marker_mz == CELL_AIR);

    // Compute error: positive = sparse (need more), negative = crowded (too many)
    // Hardcoded rest density to rule out uniform corruption
    // Hardcoded rest density to rule out uniform corruption
    let rest_density = params.rest_density; 
    var error = (density / rest_density) - 1.0;

    // Interface Correction:
    // At the fluid-air interface, we MUST prevent suction (negative error).
    // An air neighbor implies this cell is at the surface. 
    // If it's sparse (error < 0), simple FLIP would try to pull neighbors in (suction).
    // ...
    // By clamping error to >= 0, we ensure surface cells can only PUSH (if crowded), never PULL.
    if (has_air_neighbor) {
        error = max(0.0, error);
    } 
    // Interior cells can have positive error (cohesion) to maintain volume.

    // Scale by 1/dt for consistency with pressure solver
    error /= params.dt;

    // Stricter clamp to prevent excessive kicks and maintain stability
    error = clamp(error, -200.0, 200.0);

    density_error[idx] = error;
}

// Quadratic-kernel integral over a unit voxel: 2/3 for the centre interval,
// 1/6 for each neighbour. Include edges/corners, rather than a per-face guess.
fn phase_cell_type(q:vec3<i32>)->u32 {
    let bits=array<u32,3>(1u,4u,16u);
    let shape=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
    var outside=false;
    for(var a=0u;a<3u;a++) {
        if q[a]<0 {
            if (params._pad2&bits[a])==0u {return CELL_SOLID;}
            outside=true;
        }
        if q[a]>=shape[a] {
            if (params._pad2&(bits[a]*2u))==0u {return CELL_SOLID;}
            outside=true;
        }
    }
    if outside {return CELL_AIR;}
    return cell_type[cell_index(u32(q.x),u32(q.y),u32(q.z))];
}
struct PhaseVolume {moving:f32,fixed:f32,free_surface:bool};
fn phase_volume(id:vec3<u32>)->PhaseVolume {
    let index=cell_index(id.x,id.y,id.z);var fixed=0.;var air=false;
    let center=vec3<i32>(id);
    for(var z=-1;z<=1;z++) {for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
        let ct=phase_cell_type(center+vec3(x,y,z));
        if ct==CELL_SOLID {
            let w=select(1./6.,2./3.,x==0)*select(1./6.,2./3.,y==0)*select(1./6.,2./3.,z==0);
            fixed+=params.rest_density*w;
        }
        if abs(x)+abs(y)+abs(z)==1 && ct==CELL_AIR {air=true;}
    }}}
    return PhaseVolume(represented_volume[index],fixed,air);
}
fn phase_error(id:vec3<u32>)->f32 {
    let state=phase_volume(id);
    var error=(state.moving+state.fixed)/params.rest_density-1.;
    if state.free_surface {error=max(error,0.);}
    return clamp(error/params.dt,-200.,200.);
}

@group(0) @binding(5) var<storage,read_write> velocity_divergence:array<f32>;
@group(0) @binding(6) var<storage,read_write> volume_status:array<atomic<u32>>;

// Linearize continuity about the incompressible reference volume R:
// r_new = r - dt*R*div(q). Boundary-reflected volume has no phantom solid
// fill and conserves total represented material at closed walls. Thus
// div(q)=(r-R)/(dt*R), with no net source in a full closed box. Only an
// underfilled free surface omits negative feedback. This remains a first-order
// stabilization; convergence and contact/pressure calibration are required.
fn source_finite(x:f32)->bool {return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
@compute @workgroup_size(8,8,4)
fn add_density_rhs(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id>=vec3(params.width,params.height,params.depth)) {return;}
    let index=cell_index(id.x,id.y,id.z);
    if cell_type[index]!=CELL_FLUID || atomicLoad(&volume_status[0])!=0u {return;}
    let strength=bitcast<f32>(params._pad3);
    let moving=represented_volume[index];
    if !source_finite(params.dt) || params.dt<=0. || !source_finite(params.rest_density) || params.rest_density<=0.
       || !source_finite(strength) || strength<0. || strength>1.
       || !source_finite(moving) || moving<=0. {
        atomicOr(&volume_status[0],16u);return;
    }
    var excess=moving-params.rest_density;
    let q=vec3<i32>(id);var surface=!volume_home_water(q);
    for(var axis=0u;axis<3u;axis++) {
        var unit=vec3(0);unit[axis]=1;
        surface=surface || !volume_home_water(q+unit) || !volume_home_water(q-unit);
    }
    if surface {excess=max(excess,0.);}
    let volume_rate=strength*excess/(params.rest_density*params.dt);
    let result=velocity_divergence[index]-volume_rate;
    if !source_finite(volume_rate) || !source_finite(result) {atomicOr(&volume_status[0],16u);return;}
    velocity_divergence[index]=result;
}

@group(0) @binding(7) var<storage,read> volume_water_markers:array<u32>;
fn volume_home_water(q:vec3<i32>)->bool {
 let shape=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 let kind=phase_cell_type(q);
 if kind==CELL_SOLID {return true;}
 if any(q<vec3(0)) || any(q>=shape) {return false;}
 return (volume_water_markers[cell_index(u32(q.x),u32(q.y),u32(q.z))]&0x7fffffffu)>0u;
}
