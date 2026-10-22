// Conservative 8-neighbour slope collapse. The first pass reads an immutable
// geology snapshot and writes only its own outgoing transfers. The second
// gathers equal-and-opposite transfers and writes only its own geology cell.
// A red/black schedule is insufficient: diagonal sources and shared receivers
// can still race even when sources have the same checkerboard colour.
struct Params {
    world_width:u32, world_depth:u32, tile_width:u32, tile_depth:u32,
    origin_x:u32, origin_z:u32, _pad0:vec2<u32>,
    cell_size:f32, dt:f32, gravity:f32, damping:f32,
}
const TAN_ANGLE_SEDIMENT:f32=0.6249;
const TAN_ANGLE_OVERBURDEN:f32=0.7002;
const TAN_ANGLE_GRAVEL:f32=0.7813;
const COLLAPSE_RATE:f32=4.;
const OFFSETS=array<vec2<i32>,8>(vec2(-1,0),vec2(1,0),vec2(0,-1),vec2(0,1),vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
const OPPOSITE=array<u32,8>(1u,0u,3u,2u,7u,6u,5u,4u);
@group(0) @binding(0) var<uniform> params:Params;
@group(1) @binding(0) var<storage,read_write> bedrock:array<f32>;
@group(1) @binding(1) var<storage,read_write> paydirt:array<f32>;
@group(1) @binding(2) var<storage,read_write> gravel:array<f32>;
@group(1) @binding(3) var<storage,read_write> overburden:array<f32>;
@group(1) @binding(4) var<storage,read_write> sediment:array<f32>;
@group(1) @binding(5) var<storage,read_write> surface_material:array<u32>;
// Each source owns 8 vec4s: sediment, overburden, gravel thickness; W reserved.
@group(1) @binding(6) var<storage,read_write> collapse_flux:array<vec4<f32>>;
fn in_world(p:vec2<i32>)->bool {
    return all(p>=vec2(0)) && p.x<i32(params.world_width) && p.y<i32(params.world_depth);
}
fn in_tile(p:vec2<i32>)->bool {
    let q=p-vec2<i32>(i32(params.origin_x),i32(params.origin_z));
    return in_world(p) && all(q>=vec2(0)) && q.x<i32(params.tile_width) && q.y<i32(params.tile_depth);
}
fn idx(p:vec2<i32>)->u32 {return u32(p.y)*params.world_width+u32(p.x);}
fn ground(i:u32)->f32 {return bedrock[i]+paydirt[i]+gravel[i]+overburden[i]+sediment[i];}
fn transfer(available:f32,angle:f32,difference:f32,distance:f32,weight:f32)->f32 {
    let excess=max(difference-angle*distance,0.)*0.5;
    return min(excess*weight*COLLAPSE_RATE*max(params.dt,0.),available*0.5*weight);
}
@compute @workgroup_size(16,16)
fn update_collapse_flux(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.tile_width || id.y>=params.tile_depth {return;}
    let p=vec2<i32>(id.xy+vec2(params.origin_x,params.origin_z));
    if !in_world(p) {return;}
    let i=idx(p);let height=ground(i);
    var deficits:array<f32,8>;
    var distances:array<f32,8>;
    var total=0.;
    for(var n=0u;n<8u;n++) {
        let q=p+OFFSETS[n];
        collapse_flux[i*8u+n]=vec4(0.);
        distances[n]=select(1.,1.41421356,n>=4u);
        if in_world(q) {deficits[n]=max(height-ground(idx(q)),0.);}
        total+=deficits[n]/distances[n];
    }
    if total<=0. {return;}
    var remaining=vec3(sediment[i],overburden[i],gravel[i]);
    // Keep the original exposure thresholds, determined from the donor's own
    // planned losses rather than another invocation's concurrent writes.
    for(var layer=0u;layer<3u;layer++) {
        if remaining[layer]<=0.001 {continue;}
        if layer>=1u && remaining.x>=0.005 {continue;}
        if layer==2u && remaining.y>=0.005 {continue;}
        let angles=vec3(TAN_ANGLE_SEDIMENT,TAN_ANGLE_OVERBURDEN,TAN_ANGLE_GRAVEL);
        var outgoing=0.;
        for(var n=0u;n<8u;n++) {
            let weight=deficits[n]/(total*distances[n]);
            let amount=transfer(remaining[layer],angles[layer],deficits[n],params.cell_size*distances[n],weight);
            collapse_flux[i*8u+n][layer]=amount;
            outgoing+=amount;
        }
        remaining[layer]-=outgoing;
    }
}
fn source_flux(p:vec2<i32>,direction:u32)->vec3<f32> {
    // Ignore inactive sources: a moving tile cannot reuse stale transfers.
    if !in_tile(p) {return vec3(0.);}
    return collapse_flux[idx(p)*8u+direction].xyz;
}
@compute @workgroup_size(16,16)
fn apply_collapse_flux(@builtin(global_invocation_id) id:vec3<u32>) {
    // Receivers outside the donor tile still receive their transfers. World
    // edges stay closed for solids; the apply region includes a one-cell halo.
    if id.x>=params.tile_width+2u || id.y>=params.tile_depth+2u {return;}
    let p=vec2<i32>(id.xy)+vec2<i32>(i32(params.origin_x),i32(params.origin_z))-vec2(1);
    if !in_world(p) {return;}
    var outgoing=vec3(0.);var incoming=vec3(0.);
    for(var n=0u;n<8u;n++) {
        outgoing+=source_flux(p,n);
        incoming+=source_flux(p+OFFSETS[n],OPPOSITE[n]);
    }
    let i=idx(p);
    let next=vec3(sediment[i],overburden[i],gravel[i])-outgoing+incoming;
    sediment[i]=next.x;overburden[i]=next.y;gravel[i]=next.z;
    var material=0u;
    if paydirt[i]>0.001 {material=1u;}
    if next.z>0.001 {material=2u;}
    if next.y>0.001 {material=3u;}
    if next.x>0.001 {material=4u;}
    surface_material[i]=material;
}
