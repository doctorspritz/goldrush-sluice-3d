// Stationary free-slip domain-wall extension for the staggered APIC stencil.
// Keep the original virtual-node offset: scatter folds ghost moments inward,
// gather samples the same reflected value with odd normal/even tangential parity.
struct MacWallSample { node:vec3<i32>, sign:f32, valid:bool, }
fn mac_wall_sample(node:vec3<i32>,component:u32,open:u32)->MacWallSample {
 let n=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 var shape=n;shape[component]+=1;
 var q=node;var parity=1.;
 // Keep Z virtual until after local voxel reflection. Wrapping it here would
 // turn a wall across the seam into a distant stencil node.
 if !PERIODIC_Z && all(node>=vec3(0)) && all(node<shape) {return MacWallSample(node,1.,true);}
 for(var axis=0u;axis<3u;axis++) {
  if PERIODIC_Z && axis==2u {continue;}
  let normal=axis==component;let last=n[axis]-select(1,0,normal);
  if q[axis]<0 {
   if (open&(1u<<(2u*axis)))!=0u {return MacWallSample(node,1.,false);}
   q[axis]=-q[axis]-select(1,0,normal);
   if normal {parity=-parity;}
  } else if q[axis]>last {
   if (open&(2u<<(2u*axis)))!=0u {return MacWallSample(node,1.,false);}
   q[axis]=2*n[axis]-q[axis]-select(1,0,normal);
   if normal {parity=-parity;}
  }
  if q[axis]<0 || q[axis]>last {return MacWallSample(node,1.,false);}
 }
 return MacWallSample(q,parity,true);
}

// Paired free-slip extension at the planar faces of the actual voxel mask.
// The mask is published with the raw water markers before either transfer.
struct VoxelWallBounds { lower:vec3<i32>, upper:vec3<i32>, }
fn voxel_wall_bounds(point:vec3<f32>)->VoxelWallBounds {
 let n=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 let home=vec3<i32>(floor(point/params.cell_size));
 var bounds=VoxelWallBounds(vec3(0),n);
 if PERIODIC_Z {bounds.lower.z=home.z-2;bounds.upper.z=home.z+3;}
 // A valid SDF point can lie in a masked cell. A voxel-plane extension cannot
 // describe that cut cell, so retain its existing support instead of moving it.
 var canonical_home=home;
 if PERIODIC_Z {canonical_home=mac_storage_node(home);}
 if any(canonical_home<vec3(0)) || any(canonical_home>=n) || voxel_geometry_solid(home) {return bounds;}
 for(var axis=0u;axis<3u;axis++) {
  var e=vec3(0);e[axis]=1;
  if voxel_geometry_solid(home-e) {bounds.lower[axis]=home[axis];}
  if voxel_geometry_solid(home+e) {bounds.upper[axis]=home[axis]+1;}
 }
 return bounds;
}
fn voxel_face_solid(node:vec3<i32>,component:u32)->bool {
 var e=vec3(0);e[component]=1;
 let n=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 // A wall that reaches the computational edge also owns that outer face.
 // Its missing outside cell is not an opening through the solid voxel.
 if PERIODIC_Z && component==2u {return voxel_geometry_solid(node) && voxel_geometry_solid(node-e);}
 if node[component]==0 {return voxel_geometry_solid(node);}
 if node[component]==n[component] {return voxel_geometry_solid(node-e);}
 return voxel_geometry_solid(node) && voxel_geometry_solid(node-e);
}
fn voxel_wall_sample(sample:MacWallSample,component:u32,bounds:VoxelWallBounds)->MacWallSample {
 if !sample.valid {return sample;}
 let n=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 if !PERIODIC_Z && all(bounds.lower==vec3(0)) && all(bounds.upper==n) {return sample;}
 var last=bounds.upper-vec3(1);last[component]+=1;
 // Most nodes of even a wall-adjacent stencil lie in fluid. Avoid geometry
 // loads there; the reflection below would leave every coordinate unchanged.
 if all(sample.node>=bounds.lower) && all(sample.node<=last) {return sample;}
 // Never extend a plane through open fluid beside a finite block corner.
 if !voxel_face_solid(sample.node,component) {return sample;}
 var q=sample.node;var parity=sample.sign;
 for(var axis=0u;axis<3u;axis++) {
  let normal=axis==component;let shift=select(1,0,normal);
  if q[axis]<bounds.lower[axis] {
   q[axis]=2*bounds.lower[axis]-q[axis]-shift;
   if normal {parity=-parity;}
  } else if q[axis]>bounds.upper[axis]-shift {
   q[axis]=2*bounds.upper[axis]-q[axis]-shift;
   if normal {parity=-parity;}
  }
 }
 var shape=n;shape[component]+=1;
 let address=mac_storage_node(q);
 if any(address<vec3(0)) || any(address>=shape) || voxel_face_solid(q,component) {return sample;}
 return MacWallSample(q,parity,true);
}

// Apply periodic topology only when forming a storage address. Distances,
// offsets and reflected wall planes retain the original local image.
fn mac_storage_node(node:vec3<i32>)->vec3<i32> {
 var q=node;
 if PERIODIC_Z {q.z=((q.z%i32(params.depth))+i32(params.depth))%i32(params.depth);}
 return q;
}
