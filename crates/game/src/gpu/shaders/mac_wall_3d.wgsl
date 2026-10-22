// Stationary free-slip domain-wall extension for the staggered APIC stencil.
// Keep the original virtual-node offset: scatter folds ghost moments inward,
// gather samples the same reflected value with odd normal/even tangential parity.
struct MacWallSample { node:vec3<i32>, sign:f32, valid:bool, }
fn mac_wall_sample(node:vec3<i32>,component:u32,open:u32)->MacWallSample {
 let n=vec3<i32>(i32(params.width),i32(params.height),i32(params.depth));
 var shape=n;shape[component]+=1;
 if all(node>=vec3(0)) && all(node<shape) {return MacWallSample(node,1.,true);}
 var q=node;var parity=1.;
 for(var axis=0u;axis<3u;axis++) {
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
