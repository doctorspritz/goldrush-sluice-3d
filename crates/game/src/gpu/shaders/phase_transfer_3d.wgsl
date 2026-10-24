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
// Integer exponent inspection remains reliable for NaNs under GPU floating
// point optimisation; an ordered greater-than check alone misses them.
fn finite(x:f32)->bool {return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}
fn checked_add(index:u32,value:f32) {
 // Every logical face moment gets at most one contribution from each particle.
 // Count rather than reservation makes accepted results allocation independent.
 if !finite(value) || abs(value)>WIDE_MAX_UNITS/f32(max(params.count,1u)) {fail(1u);return;}
 wide_add(index,value);
}
// Reflected stencil entries can share a destination. Bound the whole-particle
// moment before weighting; the positive quadratic weights sum to at most one,
// so folded contributions retain the signed-64 accumulator margin.
fn scatter_add(index:u32,value:f32,particle_moment:f32) {
 if !finite(particle_moment) || abs(particle_moment)>WIDE_MAX_UNITS/f32(max(params.count,1u)) {fail(1u);return;}
 checked_add(index,value);
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
  || !finite(rho) || rho<1. || !finite(diameter) || !finite(velocities[id].x) || !finite(velocities[id].y) || !finite(velocities[id].z) {fail(2u);return;}
 let solid=rho>1.;let volume=select(p.w,1.,p.w<=0.);
 var bc=vec2(0.,1.);if solid {bc=relaxation(rho,diameter);}
 metadata[2u*params.capacity+id]=bc.x;metadata[3u*params.capacity+id]=bc.y;
 if solid && bc.x<0. {return;}
 let rows=mat3x3<f32>(row0[id],row1[id],row2[id]);
 let walls=voxel_wall_bounds(p.xyz);
 var face_base=0u;
 for(var axis=0u;axis<3u;axis++) {
  var stagger=vec3(.5);stagger[axis]=0.;
  var shape=vec3<u32>(params.width,params.height,params.depth);shape[axis]+=1u;
  let grid_position=p.xyz/params.cell_size-stagger;
  let center=vec3<i32>(floor(grid_position+vec3(.5)));let frac=grid_position-vec3<f32>(center);
  for(var dz=-1;dz<=1;dz++) {for(var dy=-1;dy<=1;dy++) {for(var dx=-1;dx<=1;dx++) {
   let delta=vec3(dx,dy,dz);let q=center+delta;
   let reflected=voxel_wall_sample(mac_wall_sample(q,axis,u32(params.sediment.pad)),axis,walls);
   if !reflected.valid {continue;}
   let destination=reflected.node;
   let w=weight(frac.x-f32(dx))*weight(frac.y-f32(dy))*weight(frac.z-f32(dz));if w<=0. {continue;}
   let offset=(vec3<f32>(q)+stagger)*params.cell_size-p.xyz;
   let body=select(0.,params.gravity*params.dt,axis==1u);
   let velocity=(velocities[id][axis]+dot(rows[axis],offset)+select(0.,body,solid))*reflected.sign;
   let v=volume*w;let index=1u+8u*(face_base+(u32(destination.z)*shape.y+u32(destination.y))*shape.x+u32(destination.x));
   if solid {
    scatter_add(index+2u,v,volume);scatter_add(index+3u,v*bc.x,volume*bc.x);scatter_add(index+4u,v*rho*bc.x,volume*rho*bc.x);
    scatter_add(index+5u,v*rho*bc.x*velocity,volume*rho*bc.x*velocity);scatter_add(index+6u,v*bc.y/rho,volume*bc.y/rho);scatter_add(index+7u,v*bc.y*velocity,volume*bc.y*velocity);
   } else {scatter_add(index,v,volume);scatter_add(index+1u,v*velocity,volume*velocity);}
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
  if !finite(q) || !finite(mobility) || mobility<=0. || !finite(aux.x) || !finite(aux.y) || !finite(aux.z) || !finite(aux.w) {
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

fn cell_moment_start()->u32 {return 1u+8u*(params.nu+params.nv+params.nw);}
fn cell_marker_offset()->u32 {
 let end=((params.nu+4u+63u)/64u)*64u+4u*params.nu;
 return ((end+63u)/64u)*64u;
}
fn cell_volume_offset()->u32 {
 return ((cell_marker_offset()+params.width*params.height*params.depth+63u)/64u)*64u;
}
@compute @workgroup_size(128)
fn count_cells(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let id=invocation.x;if id>=params.count {return;}
 let p=positions[id];let rho=metadata[id];
 if !all(p==p) || !finite(p.x) || !finite(p.y) || !finite(p.z) || !finite(p.w) || !finite(rho) || rho<1. {fail(2u);return;}
 let shape=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 let home=vec3<i32>(floor(p.xyz/params.cell_size));
 if rho<=1. && all(home>=vec3(0)) && all(home<shape) {
  let index=(u32(home.z)*params.height+u32(home.y))*params.width+u32(home.x);
  atomicAdd(&wide_accumulators[cell_moment_start()+2u*index].low,1u);
 }
 let volume=select(p.w,1.,p.w<=0.);
 let q=p.xyz/params.cell_size-vec3(.5);let center=vec3<i32>(floor(q+vec3(.5)));let frac=q-vec3<f32>(center);
 for(var dz=-1;dz<=1;dz++) {for(var dy=-1;dy<=1;dy++) {for(var dx=-1;dx<=1;dx++) {
  let cell=center+vec3(dx,dy,dz);if any(cell<vec3(0)) || any(cell>=shape) {continue;}
  let w=weight(frac.x-f32(dx))*weight(frac.y-f32(dy))*weight(frac.z-f32(dz));if w<=0. {continue;}
  let index=(u32(cell.z)*params.height+u32(cell.y))*params.width+u32(cell.x);
  checked_add(cell_moment_start()+2u*index+1u,volume*w);
 }}}
}
@compute @workgroup_size(128)
fn decode_cells(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let index=invocation.x;if index>=params.width*params.height*params.depth {return;}
 let moment=cell_moment_start()+2u*index;
 water_u[cell_marker_offset()+index]=bitcast<f32>(atomicLoad(&wide_accumulators[moment].low));
 water_u[cell_volume_offset()+index]=wide_read(moment+1u);
}

@compute @workgroup_size(128)
fn validate_particles(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let id=invocation.x;if id>=params.count {return;}
 let fields=array<vec3<f32>,5>(positions[id].xyz,velocities[id],row0[id],row1[id],row2[id]);
 for(var i=0u;i<5u;i++) {
  let value=fields[i];
  if !finite(value.x) || !finite(value.y) || !finite(value.z) {fail(8u);return;}
 }
}

// Runtime-only no-flux volume transfer. A blocked stencil transfer stays in
// its fluid home cell. For uniform cell quadrature, off-diagonal transfers
// retain their symmetric kernel weights and rejected weights add only to the
// diagonal. This preserves constant density even at voxel corners; sending
// diagonal blocked weights to other fluid nodes does not. Open exits retain
// their lost kernel fraction. No physical particle is moved by this operator.
@group(0) @binding(17) var<storage,read> volume_cell_types:array<u32>;
fn volume_cell_index(q:vec3<i32>)->u32 {return (u32(q.z)*params.height+u32(q.y))*params.width+u32(q.x);}
fn volume_admissible(q:vec3<i32>)->bool {
 let shape=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 if any(q<vec3(0)) || any(q>=shape) {return false;}
 return volume_cell_types[volume_cell_index(q)]!=2u;
}
fn volume_open_exit(q:vec3<i32>)->bool {
 let shape=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));let open=u32(params.sediment.pad);
 for(var axis=0u;axis<3u;axis++) {
  if q[axis]<0 && (open&(1u<<(axis*2u)))!=0u {return true;}
  if q[axis]>=shape[axis] && (open&(2u<<(axis*2u)))!=0u {return true;}
 }return false;
}
fn reflected_volume(q:vec3<i32>,home:vec3<i32>,value:f32,particle:u32) {
 if volume_open_exit(q) {return;}
 if volume_admissible(q) {checked_add(cell_moment_start()+2u*volume_cell_index(q)+1u,value);return;}
 if !volume_admissible(home) {atomicMax(&wide_accumulators[0].high,particle+1u);fail(32u);return;}
 checked_add(cell_moment_start()+2u*volume_cell_index(home)+1u,value);
}
@compute @workgroup_size(128)
fn count_cells_reflected(@builtin(global_invocation_id) invocation:vec3<u32>) {
 let id=invocation.x;if id>=params.count {return;}
 let p=positions[id];let rho=metadata[id];
 if !finite(p.x) || !finite(p.y) || !finite(p.z) || !finite(p.w) || !finite(rho) || rho<1. {fail(2u);return;}
 let shape=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));let home=vec3<i32>(floor(p.xyz/params.cell_size));
 if rho<=1. && all(home>=vec3(0)) && all(home<shape) {atomicAdd(&wide_accumulators[cell_moment_start()+2u*volume_cell_index(home)].low,1u);}
 let volume=select(p.w,1.,p.w<=0.);
 // Folding can send several stencil weights to one cell. Bound the whole
 // particle volume, rather than assuming one contribution per particle/cell.
 if volume>WIDE_MAX_UNITS/f32(max(params.count,1u)) {fail(1u);return;}
 // A positive-SDF particle can occupy the fluid part of a cell whose centre
 // is marked solid. Conservatively lump that subcell support into adjacent
 // admissible stencil cells. This does not move the particle, erase geometry,
 // or admit a point lying inside the physical SDF.
 if !volume_admissible(home) {
  let inside=all(home>=vec3(0)) && all(home<shape);
  if inside {
   let distance=sample_sdf(p.xyz);
   if !finite(distance) || distance<0. {atomicMax(&wide_accumulators[0].high,id+1u);fail(32u);return;}
  } else if !volume_open_exit(home) {atomicMax(&wide_accumulators[0].high,id+1u);fail(32u);return;}
  let q=p.xyz/params.cell_size-vec3(.5);let center=vec3<i32>(floor(q+vec3(.5)));let frac=q-vec3<f32>(center);
  var available=0.;var blocked=0.;
  for(var z=-1;z<=1;z++) {for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
   let node=center+vec3(x,y,z);let w=weight(frac.x-f32(x))*weight(frac.y-f32(y))*weight(frac.z-f32(z));
   if volume_open_exit(node) {continue;}
   if volume_admissible(node) {available+=w;} else {blocked+=w;}
  }}}
  if available<=0. {
   if blocked>0. {atomicMax(&wide_accumulators[0].high,id+1u);fail(32u);}return;
  }
  let scale=(available+blocked)/available;
  for(var z=-1;z<=1;z++) {for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
   let node=center+vec3(x,y,z);if !volume_admissible(node) {continue;}
   let w=weight(frac.x-f32(x))*weight(frac.y-f32(y))*weight(frac.z-f32(z));
   checked_add(cell_moment_start()+2u*volume_cell_index(node)+1u,volume*w*scale);
  }}}
  return;
 }

 let q=p.xyz/params.cell_size-vec3(.5);let center=vec3<i32>(floor(q+vec3(.5)));let frac=q-vec3<f32>(center);
 for(var dz=-1;dz<=1;dz++) {for(var dy=-1;dy<=1;dy++) {for(var dx=-1;dx<=1;dx++) {
  let w=weight(frac.x-f32(dx))*weight(frac.y-f32(dy))*weight(frac.z-f32(dz));
  if w>0. {reflected_volume(center+vec3(dx,dy,dz),home,volume*w,id);}
 }}}
}

@group(0) @binding(18) var<storage,read> sdf:array<f32>;
fn cell_index(i:u32,j:u32,k:u32)->u32 {return (k*params.height+j)*params.width+i;}

// Reuse the upper bit of the raw marker plane for immutable solid geometry.
// Count <= capacity; the 16-byte metadata binding bounds capacity below 2^28.
// The volume plane and all authoritative material counts retain their meaning.
fn voxel_geometry_solid(q:vec3<i32>)->bool {
 let n=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 if any(q<vec3(0)) || any(q>=n) {return false;}
 return (bitcast<u32>(water_u[cell_marker_offset()+volume_cell_index(q)])&0x80000000u)!=0u;
}
@compute @workgroup_size(128)
fn publish_voxel_geometry(@builtin(global_invocation_id) q:vec3<u32>) {
 let i=q.x;if i>=params.width*params.height*params.depth {return;}
 let index=cell_marker_offset()+i;
 let count=bitcast<u32>(water_u[index])&0x7fffffffu;
 water_u[index]=bitcast<f32>(count|select(0u,0x80000000u,volume_cell_types[i]==2u));
}
