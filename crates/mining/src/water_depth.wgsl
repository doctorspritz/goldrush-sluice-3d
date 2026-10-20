struct Uniforms {clip_from_world:mat4x4<f32>,world_from_view:mat4x4<f32>,view_from_clip:mat4x4<f32>,view_from_world:mat4x4<f32>,viewport:vec4<f32>,sediment:vec4<f32>};
struct Particle {center_radius:vec4<f32>,velocity:vec4<f32>,shape:vec4<f32>,axis_x:vec4<f32>,axis_y:vec4<f32>,axis_z:vec4<f32>,slope_x:vec4<f32>,slope_z:vec4<f32>};
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<uniform> u:Uniforms;
struct Varying {
 @builtin(position) clip:vec4<f32>,
 @location(0) world:vec3<f32>,
 @location(1) @interpolate(flat) center_radius:vec4<f32>,
 @location(2) @interpolate(flat) shape:vec4<f32>,
 @location(3) @interpolate(flat) axis_x:vec3<f32>,
 @location(4) @interpolate(flat) axis_y:vec3<f32>,
 @location(5) @interpolate(flat) axis_z:vec3<f32>,
 @location(6) terrain:vec2<f32>,
 @location(7) @interpolate(flat) velocity:vec2<f32>,
 @location(8) surface:vec4<f32>
};
@vertex fn particle_vertex(@builtin(vertex_index) id:u32,@builtin(instance_index) instance:u32)->Varying {
 let corners=array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
 let p=particles[instance];let local=corners[id];let radius=p.center_radius.w;
 let prediction=select(u.viewport.w,u.sediment.w,p.shape.z==2.);
 let center=p.center_radius.xyz+p.velocity.xyz*prediction;
 let ax=p.axis_x.xyz*radius;let ay=p.axis_y.xyz*radius;let az=p.axis_z.xyz*radius;
 let extent=max(length(ax),max(length(ay),length(az)))*1.02;
 var world=center+(u.world_from_view[0].xyz*local.x+u.world_from_view[1].xyz*local.y)*extent;
 var terrain=vec2(0.);var surface=vec4(p.axis_x.w,p.axis_y.w,p.axis_z.w,p.shape.w);
 if p.shape.z==1. {
  let corner=u32(local.x>0.)+2u*u32(local.y>0.);
  world=vec3(p.center_radius.x+local.x*radius,p.axis_x[corner],p.center_radius.z+local.y*radius);
  terrain=vec2(p.axis_y[corner],p.axis_z[corner]);
  surface=vec4(normalize(vec3(-p.slope_x[corner],1.,-p.slope_z[corner])),1.);
 }
 var out:Varying;out.clip=u.clip_from_world*vec4(world,1.);out.world=world;
 out.center_radius=vec4(center,radius);out.shape=p.shape;out.axis_x=ax;out.axis_y=ay;out.axis_z=az;out.terrain=terrain;out.velocity=p.velocity.xz;out.surface=surface;return out;
}
struct Depth {@location(0) linear:vec4<f32>,@location(1) normal:vec4<f32>,@builtin(frag_depth) depth:f32};
@fragment fn particle_depth(v:Varying)->Depth {
 var world=v.world;var optical=select(0.,-1.,v.shape.z==0.);
 if v.shape.z==1. {
  if v.terrain.x<0.0008 {discard;}
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
  let t=closest-sqrt(squared/a);if t<=0. {discard;}world=eye+ray*t;
 }
 let clip=u.clip_from_world*vec4(world,1.);var out:Depth;
 out.linear=vec4(-(u.view_from_world*vec4(world,1.)).z,optical,v.velocity);out.normal=v.surface;out.depth=clip.z/clip.w;return out;
}
@fragment fn particle_thickness(v:Varying)->@location(0) f32 {
 let eye=u.world_from_view[3].xyz;let ray=normalize(v.world-eye);let origin=eye-v.center_radius.xyz;
 let o=vec3(dot(origin,v.axis_x)/dot(v.axis_x,v.axis_x),dot(origin,v.axis_y)/dot(v.axis_y,v.axis_y),dot(origin,v.axis_z)/dot(v.axis_z,v.axis_z));
 let d=vec3(dot(ray,v.axis_x)/dot(v.axis_x,v.axis_x),dot(ray,v.axis_y)/dot(v.axis_y,v.axis_y),dot(ray,v.axis_z)/dot(v.axis_z,v.axis_z));
 let a=dot(d,d);let closest=-dot(o,d)/a;let q=o+d*closest;let squared=1.-dot(q,q);
 if squared<0. {discard;}
 return 2.*sqrt(squared/a)*v.shape.x;
}
