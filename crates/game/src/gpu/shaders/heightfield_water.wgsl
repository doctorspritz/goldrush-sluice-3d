// Local inertial SWE with hydrostatic wet/dry reconstruction and shared,
// donor-limited face volumes. Every interior transfer is used by both cells.
struct Params {
    world_width: u32, world_depth: u32,
    tile_width: u32, tile_depth: u32,
    origin_x: u32, origin_z: u32,
    open_boundaries: u32, water_substeps: u32,
    cell_size: f32, dt: f32, gravity: f32, manning_n: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read_write> water_depth: array<f32>;
@group(1) @binding(1) var<storage, read_write> water_velocity_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> water_velocity_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> water_surface: array<f32>;
@group(1) @binding(4) var<storage, read_write> flux_x: array<f32>;
@group(1) @binding(5) var<storage, read_write> flux_z: array<f32>;
// Scratch follows sediment-next, avoiding another storage binding.
// Per cell: old depth, donor scale, drain, pad; cumulative X/Z/drain/pad;
// cumulative exported material volumes (written by sediment transport).
@group(1) @binding(7) var<storage, read_write> aux: array<f32>;
@group(2) @binding(0) var<storage, read_write> bedrock: array<f32>;
@group(2) @binding(1) var<storage, read_write> paydirt: array<f32>;
@group(2) @binding(2) var<storage, read_write> gravel: array<f32>;
@group(2) @binding(3) var<storage, read_write> overburden: array<f32>;
@group(2) @binding(4) var<storage, read_write> sediment: array<f32>;
fn idx(x: u32, z: u32) -> u32 { return z * params.world_width + x; }
fn scratch(i: u32) -> u32 { return params.world_width * params.world_depth + i * 12u; }
fn ground(i: u32) -> f32 { return bedrock[i]+paydirt[i]+gravel[i]+overburden[i]+sediment[i]; }
fn step_dt() -> f32 { return params.dt / f32(max(params.water_substeps, 1u)); }
fn velocity(v: f32, slope: f32, depth: f32) -> f32 {
    if (depth <= 1e-6) { return 0.0; }
    let accelerated = v + params.gravity * slope * step_dt();
    let friction = params.gravity * params.manning_n * params.manning_n / pow(max(depth, 1e-5), 4.0/3.0);
    let limit = 0.45 * params.cell_size / step_dt();
    return clamp(accelerated / (1.0 + friction * abs(accelerated) * step_dt()), -limit, limit);
}
fn face_depth(a: u32, b: u32) -> f32 {
    let sill = max(ground(a), ground(b));
    return 0.5 * (max(0.0, water_surface[a]-sill) + max(0.0, water_surface[b]-sill));
}
fn drain(i: u32, inner: u32, outward_velocity: f32) -> f32 {
    let slope = (water_surface[inner]-water_surface[i]) / params.cell_size;
    return max(0.0, velocity(outward_velocity, slope, water_depth[i])) * water_depth[i] * params.cell_size * step_dt();
}
fn cell(g: vec3<u32>) -> vec2<u32> { return g.xy + vec2<u32>(params.origin_x, params.origin_z); }
fn outside(g: vec3<u32>) -> bool {
    let p = cell(g);
    return g.x >= params.tile_width || g.y >= params.tile_depth || p.x >= params.world_width || p.y >= params.world_depth;
}
@compute @workgroup_size(16,16)
fn update_surface(@builtin(global_invocation_id) g: vec3<u32>) {
    if (outside(g)) { return; }
    let p=cell(g); let i=idx(p.x,p.y);
    water_surface[i]=ground(i)+water_depth[i];
    aux[scratch(i)]=water_depth[i];
}
@compute @workgroup_size(16,16)
fn update_flux(@builtin(global_invocation_id) g: vec3<u32>) {
    if (outside(g)) { return; }
    let p=cell(g); let x=p.x; let z=p.y; let i=idx(x,z);
    // Read only our own old face velocities; neighbours update their own.
    let old_x=water_velocity_x[i]; let old_z=water_velocity_z[i];
    var boundary=0.0;
    if (x==0u && (params.open_boundaries & 1u)!=0u) { boundary+=drain(i,idx(1u,z),-old_x); }
    if (x+1u==params.world_width && (params.open_boundaries & 2u)!=0u) { boundary+=drain(i,idx(x-1u,z),old_x); }
    if (z==0u && (params.open_boundaries & 4u)!=0u) { boundary+=drain(i,idx(x,1u),-old_z); }
    if (z+1u==params.world_depth && (params.open_boundaries & 8u)!=0u) { boundary+=drain(i,idx(x,z-1u),old_z); }
    aux[scratch(i)+2u]=boundary;
    var vx=0.0; var vz=0.0; var qx=0.0; var qz=0.0;
    if (x+1u<params.world_width) {
        let r=idx(x+1u,z); let d=face_depth(i,r);
        vx=velocity(old_x,(water_surface[i]-water_surface[r])/params.cell_size,d);
        qx=vx*d*params.cell_size*step_dt();
    }
    if (z+1u<params.world_depth) {
        let r=idx(x,z+1u); let d=face_depth(i,r);
        vz=velocity(old_z,(water_surface[i]-water_surface[r])/params.cell_size,d);
        qz=vz*d*params.cell_size*step_dt();
    }
    water_velocity_x[i]=vx; water_velocity_z[i]=vz;
    flux_x[i]=qx; flux_z[i]=qz;
}
@compute @workgroup_size(16,16)
fn donor_budget(@builtin(global_invocation_id) g: vec3<u32>) {
    if (outside(g)) { return; }
    let p=cell(g); let x=p.x; let z=p.y; let i=idx(x,z); let a=scratch(i);
    var outgoing=max(flux_x[i],0.0)+max(flux_z[i],0.0)+aux[a+2u];
    if (x>0u) { outgoing+=max(-flux_x[idx(x-1u,z)],0.0); }
    if (z>0u) { outgoing+=max(-flux_z[idx(x,z-1u)],0.0); }
    let available=water_depth[i]*params.cell_size*params.cell_size;
    aux[a+1u]=min(1.0,available*(1.0-1e-6)/max(outgoing,1e-30));
}
@compute @workgroup_size(16,16)
fn limit_flux(@builtin(global_invocation_id) g: vec3<u32>) {
    if (outside(g)) { return; }
    let p=cell(g); let x=p.x; let z=p.y; let i=idx(x,z); let a=scratch(i);
    var sx=aux[a+1u]; var sz=sx;
    if (flux_x[i]<0.0 && x+1u<params.world_width) { sx=aux[scratch(idx(x+1u,z))+1u]; }
    if (flux_z[i]<0.0 && z+1u<params.world_depth) { sz=aux[scratch(idx(x,z+1u))+1u]; }
    flux_x[i]*=sx; flux_z[i]*=sz; aux[a+2u]*=aux[a+1u];
    aux[a+4u]+=flux_x[i]; aux[a+5u]+=flux_z[i]; aux[a+6u]+=aux[a+2u];
}
@compute @workgroup_size(16,16)
fn update_depth(@builtin(global_invocation_id) g: vec3<u32>) {
    if (outside(g)) { return; }
    let p=cell(g); let x=p.x; let z=p.y; let i=idx(x,z);
    var net=-flux_x[i]-flux_z[i]-aux[scratch(i)+2u];
    if (x>0u) { net+=flux_x[idx(x-1u,z)]; }
    if (z>0u) { net+=flux_z[idx(x,z-1u)]; }
    water_depth[i]=max(0.0,water_depth[i]+net/(params.cell_size*params.cell_size));
}
