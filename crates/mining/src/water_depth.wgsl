struct Uniforms {clip_from_world:mat4x4<f32>,world_from_view:mat4x4<f32>,view_from_clip:mat4x4<f32>,view_from_world:mat4x4<f32>,viewport:vec4<f32>,sediment:vec4<f32>};
struct Particle {center_radius:vec4<f32>,velocity:vec4<f32>,shape:vec4<f32>,axis_x:vec4<f32>,axis_y:vec4<f32>,axis_z:vec4<f32>,slope_x:vec4<f32>,slope_z:vec4<f32>,flow_z:vec4<f32>,previous_center:vec4<f32>,previous_x:vec4<f32>,previous_y:vec4<f32>,previous_z:vec4<f32>,previous_shape:vec4<f32>};
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<uniform> u:Uniforms;
@group(0) @binding(2) var front_depth:texture_2d<f32>;
@group(0) @binding(3) var opaque_depth:texture_depth_2d;
@group(0) @binding(4) var joined_depth:texture_2d<f32>;
// Presentation only, matching terrain_water::MIN_VISIBLE_DEPTH_M.
const MIN_FILM_DEPTH:f32=0.0001;
struct Varying {
 @builtin(position) clip:vec4<f32>,
 @location(0) world:vec3<f32>,
 @location(1) @interpolate(flat) center_radius:vec4<f32>,
 @location(2) @interpolate(flat) shape:vec4<f32>,
 @location(3) @interpolate(flat) axis_x:vec3<f32>,
 @location(4) @interpolate(flat) axis_y:vec3<f32>,
 @location(5) @interpolate(flat) axis_z:vec3<f32>,
 @location(6) terrain:vec2<f32>,
 @location(7) velocity:vec2<f32>,
 @location(8) surface:vec4<f32>,
 @location(9) @interpolate(flat) lateral:vec4<f32>
};
fn quad_value(values:vec4<f32>,uv:vec2<f32>)->f32 {
 return mix(mix(values.x,values.y,uv.x),mix(values.z,values.w,uv.x),uv.y);
}
@vertex fn particle_vertex(@builtin(vertex_index) id:u32,@builtin(instance_index) instance:u32)->Varying {
 let corners=array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
 var p=particles[instance];var local=corners[id%6u];
 if p.shape.z!=1. {
  p.center_radius=mix(p.previous_center,p.center_radius,u.viewport.w);
  p.velocity=vec4(mix(p.slope_x.xyz,p.velocity.xyz,u.viewport.w),p.velocity.w);
  p.axis_x=mix(p.previous_x,p.axis_x,u.viewport.w);
  p.axis_y=mix(p.previous_y,p.axis_y,u.viewport.w);
  p.axis_z=mix(p.previous_z,p.axis_z,u.viewport.w);
  // Interpolated eigenframes retain orthogonal ellipsoid axes. Equivalent
  // signs/permutations were aligned once when the snapshot pair was built.
  let x=normalize(p.axis_x.xyz);
  let y=normalize(p.axis_y.xyz-x*dot(x,p.axis_y.xyz));
  let z=normalize(cross(x,y))*select(-1.,1.,dot(cross(x,y),p.axis_z.xyz)>=0.);
  p.axis_x=vec4(x*length(p.axis_x.xyz),p.axis_x.w);
  p.axis_y=vec4(y*length(p.axis_y.xyz),p.axis_y.w);
  p.axis_z=vec4(z*length(p.axis_z.xyz),p.axis_z.w);
  let kind=p.shape.z;p.shape=mix(p.previous_shape,p.shape,u.viewport.w);p.shape.z=kind;
 }
 let radius=p.center_radius.w;let center=p.center_radius.xyz;
 let ax=p.axis_x.xyz*radius;let ay=p.axis_y.xyz*radius;let az=p.axis_z.xyz*radius;
 let extent=max(length(ax),max(length(ay),length(az)))*1.02;
 var world=center+(u.world_from_view[0].xyz*local.x+u.world_from_view[1].xyz*local.y)*extent;
 var terrain=vec2(0.);var surface=vec4(p.axis_x.w,p.axis_y.w,p.axis_z.w,p.shape.w);var flow=p.velocity.xz;
 if p.shape.z==1. {
  let subdivisions=u32(p.shape.w);let square=id/6u;
  let uv=(vec2(f32(square%subdivisions),f32(square/subdivisions))+(local+1.)*0.5)/f32(subdivisions);
  local=uv*2.-1.;
  world=vec3(p.center_radius.x+local.x*radius,quad_value(p.axis_x,uv),p.center_radius.z+local.y*radius);
  terrain=vec2(quad_value(p.axis_y,uv),quad_value(p.axis_z,uv));
  flow=vec2(quad_value(p.velocity,uv),quad_value(p.flow_z,uv));
  surface=vec4(normalize(vec3(-quad_value(p.slope_x,uv),1.,-quad_value(p.slope_z,uv))),1.);
 }
 var out:Varying;out.clip=u.clip_from_world*vec4(world,1.);out.world=world;
 out.center_radius=vec4(center,radius);out.shape=p.shape;out.axis_x=ax;out.axis_y=ay;out.axis_z=az;out.terrain=terrain;out.velocity=flow;out.surface=surface;out.lateral=p.slope_z;return out;
}
// Visible ray interval inside the actual channel walls. Kernel smoothing
// never creates liquid on the outer steel face. A zero plane keeps non-channel
// and legacy diagnostic instances unbounded; the accepted outlet bypasses it.
fn channel_interval(eye:vec3<f32>,ray:vec3<f32>,plane:vec4<f32>)->vec2<f32> {
 let o=dot(vec4(eye,1.),plane);let d=dot(ray,plane.xyz);
 if abs(d)<0.000001 {
  return select(vec2(1.,-1.),vec2(-1e20,1e20),abs(o)<=1.);
 }
 let a=(-1.-o)/d;let b=(1.-o)/d;return vec2(min(a,b),max(a,b));
}
fn channel_profile_scale(ray:vec3<f32>,normal:vec3<f32>,view_cos:f32)->f32 {
 return max(1.,view_cos/max(abs(dot(ray,normalize(normal))),0.15));
}
struct Depth {@location(0) linear:vec4<f32>,@location(1) normal:vec4<f32>,@builtin(frag_depth) depth:f32};
@fragment fn particle_depth(v:Varying)->Depth {
 // Negative tags identify the channel; positive tags carry the airborne
 // outlet's optical sample. River selection uses its separate depth buffer.
 var world=v.world;var optical=select(1.+v.shape.y,-1.-v.shape.y,v.shape.z==0.);
 if v.shape.z==1. {
  if v.terrain.x<=MIN_FILM_DEPTH {discard;}
  // Positive marker plus class-aware extinction in inverse metres; clear=1.
  optical=1.+v.terrain.y;
 } else {
  // Ray/ellipsoid intersection reconstructs a surface from neighbourhood
  // covariance. The solver positions, velocities and APIC state are untouched.
  let eye=u.world_from_view[3].xyz;let ray=normalize(v.world-eye);
  let origin=eye-v.center_radius.xyz;
  let o=vec3(dot(origin,v.axis_x)/dot(v.axis_x,v.axis_x),dot(origin,v.axis_y)/dot(v.axis_y,v.axis_y),dot(origin,v.axis_z)/dot(v.axis_z,v.axis_z));
  let d=vec3(dot(ray,v.axis_x)/dot(v.axis_x,v.axis_x),dot(ray,v.axis_y)/dot(v.axis_y,v.axis_y),dot(ray,v.axis_z)/dot(v.axis_z,v.axis_z));
  let a=dot(d,d);let closest=-dot(o,d)/a;let q=o+d*closest;let squared=1.-dot(q,q);
  if squared<0. {discard;}
  var t=closest-sqrt(squared/a);
  if v.shape.z==0. {
   let interval=channel_interval(eye,ray,v.lateral);
   t=max(t,interval.x);
   if t>min(closest+sqrt(squared/a),interval.y) {discard;}
  }
  if t<=0. {discard;}world=eye+ray*t;
 }
 let clip=u.clip_from_world*vec4(world,1.);var out:Depth;
 // The profile samples normal-distance bands. At a grazing angle a fixed
 // camera-Z search stops in the front kernel, before reaching the joined sheet.
 // Store one shared scale on the nearest envelope so all overlapping kernels
 // evaluate identical probe positions; channel concentration lives in columns.
 if v.shape.z==0. {
  let ray=normalize(world-u.world_from_view[3].xyz);
  let view_cos= -(u.view_from_world*vec4(ray,0.)).z;
  optical= -channel_profile_scale(ray,v.surface.xyz,view_cos);
 }
 out.linear=vec4(-(u.view_from_world*vec4(world,1.)).z,optical,v.velocity);out.normal=v.surface;
 // River alpha carries physical vertical depth, not view-ray thickness.
 if v.shape.z==1. {out.normal.w=v.terrain.x;}
 out.depth=clip.z/clip.w;return out;
}
// A normalized poly6 column tapers continuously at a kernel's silhouette.
// Its integral over the projected ellipsoid is the same represented volume
// as the uniform chord, rather than losing water when softening the edge.
fn column_primitive(l:f32,s:f32)->f32 {
 let l2=l*l;
 return l*(s*s*s+l2*(-s*s+l2*(0.6*s-l2/7.)));
}
fn column_integral(s:f32,a:f32,lo:f32,hi:f32)->f32 {
 return max(6.5625*(column_primitive(hi,s)-column_primitive(lo,s))/sqrt(a),0.);
}
// Outward gradient of the normalized 3D poly6 density. Summing gradients
// reconstructs the interface of overlapping water, including volumetric crests;
// a covariance plane is only a reliable normal for a thin sheet.
fn density_gradient(point:vec3<f32>,center:vec3<f32>,ax:vec3<f32>,ay:vec3<f32>,az:vec3<f32>,volume_scale:f32)->vec3<f32> {
 let r=point-center;
 let q=vec3(dot(r,ax)/dot(ax,ax),dot(r,ay)/dot(ay,ay),dot(r,az)/dot(az,az));
 let support=max(1.-dot(q,q),0.);
 return (ax*q.x/dot(ax,ax)+ay*q.y/dot(ay,ay)+az*q.z/dot(az,az))*(39.375*volume_scale*support*support);
}
// Same normalized kernel as the optical volume integral. Density is additive:
// the interface belongs to all overlapping water, never the nearest sphere.
fn kernel_density(point:vec3<f32>,center:vec3<f32>,ax:vec3<f32>,ay:vec3<f32>,az:vec3<f32>,scale:f32)->f32 {
 let r=point-center;
 let q=vec3(dot(r,ax)/dot(ax,ax),dot(r,ay)/dot(ay,ay),dot(r,az)/dot(az,az));
 let support=max(1.-dot(q,q),0.);
 return 6.5625*scale*support*support*support;
}
// A compact isolated drop has high normalized density even though it has no
// connected-water support. It must not choose the interface for a whole bulk
// column behind it. Confidence applies only to surface selection: the optical
// volume integral and sparse ray/ellipsoid silhouette remain untouched.
fn interface_density(density:f32,bulk_support:f32)->f32 {
 return density*clamp(bulk_support,0.,1.);
}
// Fixed view-depth probes bracket a connected crest even when a sparse front
// particle sits ahead of it. No extra particle pass or simulation grid is used.
const PROFILE_DEPTHS:array<f32,8>=array(0.,0.002,0.005,0.009,0.015,0.023,0.035,0.051);
struct Column {@location(0) optical:vec4<f32>,@location(1) normal:vec4<f32>,@location(2) near:vec4<f32>,@location(3) far:vec4<f32>};
@fragment fn particle_thickness(v:Varying)->Column {
 let eye=u.world_from_view[3].xyz;let ray=normalize(v.world-eye);let origin=eye-v.center_radius.xyz;
 let o=vec3(dot(origin,v.axis_x)/dot(v.axis_x,v.axis_x),dot(origin,v.axis_y)/dot(v.axis_y,v.axis_y),dot(origin,v.axis_z)/dot(v.axis_z,v.axis_z));
 let d=vec3(dot(ray,v.axis_x)/dot(v.axis_x,v.axis_x),dot(ray,v.axis_y)/dot(v.axis_y,v.axis_y),dot(ray,v.axis_z)/dot(v.axis_z,v.axis_z));
 let a=dot(d,d);let closest=-dot(o,d)/a;let q=o+d*closest;let squared=1.-dot(q,q);
 if squared<0. {discard;}
 let pixel=vec2<i32>(v.clip.xy);
 let front=textureLoad(front_depth,pixel,0);
 let channel=v.shape.z==0.;
 if front.r<=0. || (front.g< -0.5)!=channel {discard;}
 let half=sqrt(squared/a);
 var path=2.*half*v.shape.x;
 var normal=vec4(0.);
 var surface=vec2(0.);
 var profile:array<f32,8>;
 if channel {
  // Integrate only the visible water column. A submerged kernel behind a
  // riffle/wall cannot contribute sediment to the water in front of it.
  let full=vec2<i32>(textureDimensions(opaque_depth));
  let q=clamp(vec2<i32>(v.clip.xy*vec2<f32>(full)/vec2<f32>(textureDimensions(front_depth))),vec2(0),full-vec2(1));
  let z=textureLoad(opaque_depth,q,0);
  let xy=(vec2<f32>(q)+0.5)/vec2<f32>(full);
  let opaque=u.view_from_clip*vec4(xy.x*2.-1.,1.-xy.y*2.,z,1.);
  let ray_scale=-(u.view_from_world*vec4(ray,0.)).z;
  var back=closest+half;
  if z>0. {back=min(back,(-opaque.z/max(opaque.w,0.000001))/max(ray_scale,0.000001));}
  let interval=channel_interval(eye,ray,v.lateral);
  back=min(back,interval.y);
  let front_t=max(max(closest-half,0.),interval.x);
  if back<=front_t {discard;}
  let support=sqrt(squared);
  path=column_integral(squared,a,clamp((front_t-closest)*sqrt(a),-support,support),clamp((back-closest)*sqrt(a),-support,support))*v.shape.x;
  // Probe just inside the visible interface, where a compact kernel has a
  // nonzero gradient. The probe is bounded by the opaque surface, so hidden
  // slurry or water behind the equipment cannot supply a surface normal.
  let inset=min(0.004,0.35*min(length(v.axis_x),min(length(v.axis_y),length(v.axis_z))));
  let probe=min(front.r/max(ray_scale,0.000001)+inset,back);
  let gradient=density_gradient(eye+ray*probe,v.center_radius.xyz,v.axis_x,v.axis_y,v.axis_z,v.shape.x);
  normal=vec4(gradient,length(gradient));
  // Sample a shared density field along the visible ray. Reusing the two
  // existing channels adds no target or pass. Opaque objects bound both probes.
  let t0=(front.r+0.004)/max(ray_scale,0.000001);
  let t1=(front.r+0.012)/max(ray_scale,0.000001);
  // A probe behind an opaque floor contains no visible liquid. Clamping its
  // position to the floor would mislabel that density as 12 mm-deep water and
  // bury shallow films when resolving the interface.
  surface=vec2(
   select(0.,interface_density(kernel_density(eye+ray*t0,v.center_radius.xyz,v.axis_x,v.axis_y,v.axis_z,v.shape.x),v.surface.w),t0<=back),
   select(0.,interface_density(kernel_density(eye+ray*t1,v.center_radius.xyz,v.axis_x,v.axis_y,v.axis_z,v.shape.x),v.surface.w),t1<=back));
  for(var i=0u;i<8u;i++) {
   let t=(front.r+PROFILE_DEPTHS[i]*max(-front.g,1.))/max(ray_scale,0.000001);
   if t<=back {profile[i]=interface_density(kernel_density(eye+ray*t,v.center_radius.xyz,v.axis_x,v.axis_y,v.axis_z,v.shape.x),v.surface.w);}
  }
 }
 var out:Column;out.optical=vec4(path,path*v.shape.y,surface);out.normal=normal;
 out.near=vec4(profile[0],profile[1],profile[2],profile[3]);
 out.far=vec4(profile[4],profile[5],profile[6],profile[7]);return out;
}

// Integrate each represented kernel over the visible eye-ray interval. Clipping
// a sum by its mean concentration retains sediment from hidden front layers.
fn visible_kernel_column(s:f32,a:f32,closest:f32,front:f32,back:f32,scale:f32)->f32 {
 let half=sqrt(max(s,0.)/a);
 let lo=max(closest-half,front);let hi=min(closest+half,back);
 if hi<=lo {return 0.;}
 return column_integral(s,a,(lo-closest)*sqrt(a),(hi-closest)*sqrt(a))*scale;
}
struct VisibleColumn {@location(0) optical:vec4<f32>,@location(1) normal:vec4<f32>};
@fragment fn particle_visible_thickness(v:Varying)->VisibleColumn {
 let eye=u.world_from_view[3].xyz;let ray=normalize(v.world-eye);let origin=eye-v.center_radius.xyz;
 let o=vec3(dot(origin,v.axis_x)/dot(v.axis_x,v.axis_x),dot(origin,v.axis_y)/dot(v.axis_y,v.axis_y),dot(origin,v.axis_z)/dot(v.axis_z,v.axis_z));
 let d=vec3(dot(ray,v.axis_x)/dot(v.axis_x,v.axis_x),dot(ray,v.axis_y)/dot(v.axis_y,v.axis_y),dot(ray,v.axis_z)/dot(v.axis_z,v.axis_z));
 let a=dot(d,d);let closest=-dot(o,d)/a;let q=o+d*closest;let squared=1.-dot(q,q);
 if squared<0. {discard;}
 let pixel=vec2<i32>(v.clip.xy);let raw=textureLoad(front_depth,pixel,0);
 let channel=v.shape.z==0.;
 if raw.r<=0. || (raw.g< -0.5)!=channel {discard;}
 // Reproduce the accepted detached/outlet chord exactly, in the same order.
 var path=2.*sqrt(squared/a)*v.shape.x;var normal=vec4(0.);
 if channel {
  let joined=textureLoad(joined_depth,pixel,0);
  if joined.r<=0. || joined.a<=0.5 {discard;}
  let full=vec2<i32>(textureDimensions(opaque_depth));
  let q=clamp(vec2<i32>(v.clip.xy*vec2<f32>(full)/vec2<f32>(textureDimensions(front_depth))),vec2(0),full-vec2(1));
  let z=textureLoad(opaque_depth,q,0);let xy=(vec2<f32>(q)+0.5)/vec2<f32>(full);
  let opaque=u.view_from_clip*vec4(xy.x*2.-1.,1.-xy.y*2.,z,1.);
  let ray_scale=max(-(u.view_from_world*vec4(ray,0.)).z,0.000001);
  let interval=channel_interval(eye,ray,v.lateral);
  let front=max(max(joined.r/ray_scale,0.),interval.x);
  var back=interval.y;
  if z>0. {back=min(back,(-opaque.z/max(opaque.w,0.000001))/ray_scale);}
  path=visible_kernel_column(squared,a,closest,front,back,v.shape.x);
  if back>front {
   // Every overlapping kernel differentiates the same joined point. The old
   // front-envelope inset gave a resolved crest the normal of foreground
   // spray. Match the bulk-weighted scalar field that selected the interface;
   // a vanishing gradient stays untrusted and retains the existing fallback.
   let gradient=density_gradient(eye+ray*front,v.center_radius.xyz,v.axis_x,v.axis_y,v.axis_z,v.shape.x)*clamp(v.surface.w,0.,1.);
   normal=vec4(gradient,length(gradient));
  }
 }
 var out:VisibleColumn;out.optical=vec4(path,path*v.shape.y,0.,0.);out.normal=normal;return out;
}
