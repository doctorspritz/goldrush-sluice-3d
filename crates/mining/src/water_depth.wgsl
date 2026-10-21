struct Uniforms {clip_from_world:mat4x4<f32>,world_from_view:mat4x4<f32>,view_from_clip:mat4x4<f32>,view_from_world:mat4x4<f32>,viewport:vec4<f32>,sediment:vec4<f32>};
struct Particle {center_radius:vec4<f32>,velocity:vec4<f32>,shape:vec4<f32>,axis_x:vec4<f32>,axis_y:vec4<f32>,axis_z:vec4<f32>,slope_x:vec4<f32>,slope_z:vec4<f32>,flow_z:vec4<f32>};
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<uniform> u:Uniforms;
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
 @location(8) surface:vec4<f32>
};
fn quad_value(values:vec4<f32>,uv:vec2<f32>)->f32 {
 return mix(mix(values.x,values.y,uv.x),mix(values.z,values.w,uv.x),uv.y);
}
@vertex fn particle_vertex(@builtin(vertex_index) id:u32,@builtin(instance_index) instance:u32)->Varying {
 let corners=array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
 let p=particles[instance];var local=corners[id%6u];let radius=p.center_radius.w;
 let prediction=select(u.viewport.w,u.sediment.w,p.shape.z==2.);
 let center=p.center_radius.xyz+p.velocity.xyz*prediction;
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
 out.center_radius=vec4(center,radius);out.shape=p.shape;out.axis_x=ax;out.axis_y=ay;out.axis_z=az;out.terrain=terrain;out.velocity=flow;out.surface=surface;return out;
}
struct Depth {@location(0) linear:vec4<f32>,@location(1) normal:vec4<f32>,@builtin(frag_depth) depth:f32};
@fragment fn particle_depth(v:Varying)->Depth {
 // Negative tags identify the channel; positive tags carry the airborne
 // outlet's optical sample. River selection uses its separate depth buffer.
 var world=v.world;var optical=select(1.+v.shape.y,-1.-v.shape.y,v.shape.z==0.);
 if v.shape.z==1. {
  if v.terrain.x<=MIN_FILM_DEPTH {discard;}
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
 out.linear=vec4(-(u.view_from_world*vec4(world,1.)).z,optical,v.velocity);out.normal=v.surface;
 // River alpha carries physical vertical depth, not view-ray thickness.
 if v.shape.z==1. {out.normal.w=v.terrain.x;}
 out.depth=clip.z/clip.w;return out;
}
@fragment fn particle_thickness(v:Varying)->@location(0) f32 {
 let eye=u.world_from_view[3].xyz;let ray=normalize(v.world-eye);let origin=eye-v.center_radius.xyz;
 let o=vec3(dot(origin,v.axis_x)/dot(v.axis_x,v.axis_x),dot(origin,v.axis_y)/dot(v.axis_y,v.axis_y),dot(origin,v.axis_z)/dot(v.axis_z,v.axis_z));
 let d=vec3(dot(ray,v.axis_x)/dot(v.axis_x,v.axis_x),dot(ray,v.axis_y)/dot(v.axis_y,v.axis_y),dot(ray,v.axis_z)/dot(v.axis_z,v.axis_z));
 let a=dot(d,d);let closest=-dot(o,d)/a;let q=o+d*closest;let squared=1.-dot(q,q);
 if squared<0. {discard;}
 return 2.*sqrt(squared/a)*v.shape.x;
}
