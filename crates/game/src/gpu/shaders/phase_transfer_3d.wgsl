// Local backward-Euler drag eliminated jointly with pressure. Relative water
// density is one. Body predictors use full gravity; pressure supplies buoyancy.
struct SedimentParams {
 settling_velocity:f32,friction_threshold:f32,friction_strength:f32,vorticity_lift:f32,
 vorticity_threshold:f32,drag_coefficient:f32,gold_density_threshold:f32,gold_drag_multiplier:f32,
 gold_settling_velocity:f32,gold_flake_lift:f32,gravity:f32,pad:f32,
}
struct Params {
 width:u32,height:u32,depth:u32,count:u32,
 cell_size:f32,dt:f32,gravity:f32,reference_volume:f32,
 capacity:u32,nu:u32,nv:u32,nw:u32,
 sediment:SedimentParams,
}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> velocities:array<vec3<f32>>;
@group(0) @binding(3) var<storage,read> row0:array<vec3<f32>>;
@group(0) @binding(4) var<storage,read> row1:array<vec3<f32>>;
@group(0) @binding(5) var<storage,read> row2:array<vec3<f32>>;
@group(0) @binding(6) var<storage,read_write> metadata:array<f32>;
@group(0) @binding(7) var<storage,read_write> wide_accumulators:array<WideSum>;
@group(0) @binding(8) var<storage,read_write> grid_u:array<f32>;
@group(0) @binding(9) var<storage,read_write> grid_v:array<f32>;
@group(0) @binding(10) var<storage,read_write> grid_w:array<f32>;
@group(0) @binding(11) var<storage,read_write> water_u:array<f32>;
@group(0) @binding(12) var<storage,read_write> water_v:array<f32>;
@group(0) @binding(13) var<storage,read_write> water_w:array<f32>;
@group(0) @binding(14) var<storage,read_write> mobility_u:array<f32>;
@group(0) @binding(15) var<storage,read_write> mobility_v:array<f32>;
@group(0) @binding(16) var<storage,read_write> mobility_w:array<f32>;
fn fail(bits:u32) {atomicOr(&wide_accumulators[0].low,bits);}
fn finite(x:f32)->bool {return x==x && abs(x)<=3.402823e38;}
fn checked_add(index:u32,value:f32) {
 // Every logical face moment gets at most one contribution from each particle.
 // Count rather than reservation makes accepted results allocation independent.
 if !finite(value) || abs(value)>WIDE_MAX_UNITS/f32(max(params.count,1u)) {fail(1u);return;}
 wide_add(index,value);
}
fn weight(x:f32)->f32 {
 let a=abs(x);if a<.5 {return .75-a*a;}if a<1.5 {return .5*(1.5-a)*(1.5-a);}return 0.;
}
fn relaxation(rho:f32,diameter:f32)->vec2<f32> {
 if diameter<0. {return vec2(-1.,1.);}
 let g=abs(params.gravity);let submerged=g*max(0.,1.-1./rho);
 let gold=rho>params.sediment.gold_density_threshold;
 var terminal=select(params.sediment.settling_velocity,params.sediment.gold_settling_velocity,gold);
 var gamma=params.sediment.drag_coefficient;
 if terminal>0. && submerged>0. {gamma=submerged/terminal;}
 if gold {gamma*=params.sediment.gold_drag_multiplier;}
 if diameter>0. {
  terminal=(rho-1.)*g*diameter*diameter/(18.e-6+sqrt(.75*(rho-1.)*g*diameter*diameter*diameter));
  gamma=select(18.e-6/max(rho*diameter*diameter,1.e-12),submerged/max(terminal,1.e-5),submerged>0.);
 }
 let a=gamma*params.dt;
 if !finite(gamma) || gamma<0. || !finite(a) {fail(2u);return vec2(0.,1.);}
 return vec2(a/(1.+a),1./(1.+a));
}
@compute @workgroup_size(128)
fn scatter(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let id=invocation.x;if id>=params.count {return;}
 let p=positions[id];let rho=metadata[id];let diameter=metadata[params.capacity+id];
 if !all(p==p) || !finite(p.x) || !finite(p.y) || !finite(p.z) || !finite(p.w)
  || !finite(rho) || rho<1. || !finite(diameter) || !all(velocities[id]==velocities[id]) {fail(2u);return;}
 let solid=rho>1.;let volume=select(p.w,1.,p.w<=0.);
 var bc=vec2(0.,1.);if solid {bc=relaxation(rho,diameter);}
 metadata[2u*params.capacity+id]=bc.x;metadata[3u*params.capacity+id]=bc.y;
 if solid && bc.x<0. {return;}
 let rows=mat3x3<f32>(row0[id],row1[id],row2[id]);
 var face_base=0u;
 for(var axis=0u;axis<3u;axis++) {
  var stagger=vec3(.5);stagger[axis]=0.;
  var shape=vec3<u32>(params.width,params.height,params.depth);shape[axis]+=1u;
  let grid_position=p.xyz/params.cell_size-stagger;
  let center=vec3<i32>(floor(grid_position+vec3(.5)));let frac=grid_position-vec3<f32>(center);
  for(var dz=-1;dz<=1;dz++) {for(var dy=-1;dy<=1;dy++) {for(var dx=-1;dx<=1;dx++) {
   let delta=vec3(dx,dy,dz);let q=center+delta;
   if any(q<vec3(0)) || any(q>=vec3<i32>(shape)) {continue;}
   let w=weight(frac.x-f32(dx))*weight(frac.y-f32(dy))*weight(frac.z-f32(dz));if w<=0. {continue;}
   let offset=(vec3<f32>(q)+stagger)*params.cell_size-p.xyz;
   let body=select(0.,params.gravity*params.dt,axis==1u);
   let velocity=velocities[id][axis]+dot(rows[axis],offset)+select(0.,body,solid);
   let v=volume*w;let index=1u+8u*(face_base+(u32(q.z)*shape.y+u32(q.y))*shape.x+u32(q.x));
   if solid {
    checked_add(index+2u,v);checked_add(index+3u,v*bc.x);checked_add(index+4u,v*rho*bc.x);
    checked_add(index+5u,v*rho*bc.x*velocity);checked_add(index+6u,v*bc.y/rho);checked_add(index+7u,v*bc.y*velocity);
   } else {checked_add(index,v);checked_add(index+1u,v*velocity);}
  }}}
  face_base+=shape.x*shape.y*shape.z;
 }
}
fn axis_index(global:u32)->vec3<u32> {
 if global<params.nu {return vec3(0u,global,params.nu);}
 if global<params.nu+params.nv {return vec3(1u,global-params.nu,params.nv);}
 return vec3(2u,global-params.nu-params.nv,params.nw);
}
fn write_grid(axis:u32,index:u32,value:f32) {
 if axis==0u {grid_u[index]=value;}else if axis==1u {grid_v[index]=value;}else {grid_w[index]=value;}
}
fn read_grid(axis:u32,index:u32)->f32 {
 if axis==0u {return grid_u[index];}if axis==1u {return grid_v[index];}return grid_w[index];
}
fn write_water(axis:u32,index:u32,value:f32) {
 if axis==0u {water_u[index]=value;}else if axis==1u {water_v[index]=value;}else {water_w[index]=value;}
}
fn write_mobility(axis:u32,index:u32,value:f32) {
 if axis==0u {mobility_u[index+1u]=value;}else if axis==1u {mobility_v[index+1u]=value;}else {mobility_w[index+1u]=value;}
}
@compute @workgroup_size(128)
fn normalise(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let global=invocation.x;if global>=params.nu+params.nv+params.nw {return;}
 let face=axis_index(global);let index=1u+8u*global;let vw=wide_read(index);var velocity=0.;
 if vw>0. {velocity=wide_read(index+1u)/vw;}
 write_grid(face.x,face.y,velocity);write_water(face.x,face.y,velocity);
}
@compute @workgroup_size(128)
fn assemble(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let global=invocation.x;if global>=params.nu+params.nv+params.nw {return;}
 let face=axis_index(global);let index=1u+8u*global;let vw=wide_read(index);
 let offset=((face.z+4u+63u)/64u)*64u+4u*face.y;
 var aux=vec4(0.);var mobility=0.;var q=0.;
 if vw>0. && atomicLoad(&wide_accumulators[0].low)==0u {
  let volume=vw+wide_read(index+2u);let a=vw+wide_read(index+3u);let m=vw+wide_read(index+4u);
  let scale=params.reference_volume/volume;
  let u0=(vw*read_grid(face.x,face.y)+wide_read(index+5u))/m;
  q=(a*u0+wide_read(index+7u))/volume;
  mobility=scale/volume*(a*a/m+wide_read(index+6u));let kw=scale*a/m;
  aux=vec4(u0,kw/mobility,q,scale/mobility);
  if !finite(q) || !finite(mobility) || mobility<=0. || !all(aux==aux) || any(abs(aux)>vec4(3.402823e38)) {
   fail(4u);aux=vec4(0.);mobility=0.;q=0.;
  }
 }
 write_grid(face.x,face.y,q);write_mobility(face.x,face.y,mobility);
 for(var c=0u;c<4u;c++) {write_water(face.x,offset+c,aux[c]);}
}
@compute @workgroup_size(1)
fn publish() {
 water_u[params.nu]=f32(atomicLoad(&wide_accumulators[0].low));
 mobility_u[0]=1.;mobility_v[0]=1.;mobility_w[0]=1.;
}
