override PERIODIC_Z:bool=false;
// Pressure unknowns follow occupied water or a complete carried-water stencil.
// Six supported faces avoid pressure impulses on faces with no water carrier.
struct Params { width:u32, height:u32, depth:u32, open_boundaries:u32, }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> wide_accumulators:array<WideSum>;
@group(0) @binding(2) var<storage,read_write> cell_type:array<u32>;
fn has_water(face:u32)->bool {
 var physical=face;
 let nu=(params.width+1u)*params.height*params.depth;
 let nv=params.width*(params.height+1u)*params.depth;
 let cells=params.width*params.height*params.depth;
 if PERIODIC_Z && face>=nu+nv+cells {physical-=cells;}
 return wide_read(1u+8u*physical)>0.;
}
@compute @workgroup_size(8,8,4)
fn classify(@builtin(global_invocation_id) q:vec3<u32>) {
 if any(q>=vec3(params.width,params.height,params.depth)) {return;}
 let cell=(q.z*params.height+q.y)*params.width+q.x;
 if cell_type[cell]==2u {return;}
 let open=params.open_boundaries;
 if (q.x==0u && (open&1u)!=0u) || (q.x+1u==params.width && (open&2u)!=0u)
 || (q.y==0u && (open&4u)!=0u) || (q.y+1u==params.height && (open&8u)!=0u)
 || (!PERIODIC_Z && ((q.z==0u && (open&16u)!=0u) || (q.z+1u==params.depth && (open&32u)!=0u))) {
  cell_type[cell]=0u;return;
 }
 let nu=(params.width+1u)*params.height*params.depth;
 let nv=params.width*(params.height+1u)*params.depth;
 let nw=params.width*params.height*(params.depth+1u);
 let home=1u+8u*(nu+nv+nw)+2u*cell;
 let u=(q.z*params.height+q.y)*(params.width+1u)+q.x;
 let v=nu+(q.z*(params.height+1u)+q.y)*params.width+q.x;
 let w=nu+nv+(q.z*params.height+q.y)*params.width+q.x;
 let supported=has_water(u) && has_water(u+1u) && has_water(v) && has_water(v+params.width)
  && has_water(w) && has_water(w+params.width*params.height);
 let occupied=atomicLoad(&wide_accumulators[home].low)>0u;
 let volume=wide_read(home+1u);
 if atomicLoad(&wide_accumulators[0].low)!=0u {return;}
 // A real occupied water cell must have its six carried MAC faces. Refuse
 // inconsistent/unrepresentable support instead of publishing a phantom force.
 if occupied && (!supported || volume<=0.) {
  atomicOr(&wide_accumulators[0].low,4u);return;
 }
 cell_type[cell]=select(0u,1u,supported && volume>0.);
}
