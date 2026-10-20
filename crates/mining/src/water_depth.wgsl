struct Uniforms { clip_from_world:mat4x4<f32>,world_from_view:mat4x4<f32>,view_from_clip:mat4x4<f32>,view_from_world:mat4x4<f32>,viewport:vec4<f32>,sediment:vec4<f32> };
struct Particle {center_radius:vec4<f32>,velocity:vec4<f32>,shape:vec4<f32>};
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<uniform> u:Uniforms;
struct Varying {@builtin(position) clip:vec4<f32>,@location(0) local:vec2<f32>,@location(1) center_radius:vec4<f32>,@location(2) shape:vec4<f32>};
@vertex fn particle_vertex(@builtin(vertex_index) id:u32,@builtin(instance_index) instance:u32)->Varying {
 let corners=array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
 let particle=particles[instance];let prediction=select(u.viewport.w,u.sediment.y,particle.shape.z>0.);
 let p=vec4(particle.center_radius.xyz+particle.velocity.xyz*prediction,particle.center_radius.w);let local=corners[id];
 let slope=vec2(particle.shape.x,particle.shape.w);
 let extent=p.w*select(1.,sqrt(1.+dot(slope,slope)),particle.shape.z>0.);
 let world=p.xyz+(u.world_from_view[0].xyz*local.x+u.world_from_view[1].xyz*local.y)*extent;
 var out:Varying;out.clip=u.clip_from_world*vec4(world,1.);out.local=local;out.center_radius=p;out.shape=particle.shape;return out;
}
struct Depth {@location(0) linear:vec2<f32>,@builtin(frag_depth) depth:f32};
@fragment fn particle_depth(v:Varying)->Depth {
 let r2=dot(v.local,v.local);if r2>1. {discard;}
 var world=v.center_radius.xyz+(u.world_from_view[0].xyz*v.local.x+u.world_from_view[1].xyz*v.local.y+u.world_from_view[2].xyz*sqrt(max(0.,1.-r2)))*v.center_radius.w;
 if v.shape.z>0. {
  // Tangent discs blend the solver's surface samples without adding the
  // lattice-shaped waves produced by curved caps on a nearly level river.
  let center=v.center_radius.xyz;let radius=v.center_radius.w;
  let slope=vec2(v.shape.x,v.shape.w);let normal=vec3(-slope.x,1.,-slope.y);
  let extent=radius*sqrt(1.+dot(slope,slope));
  let plane=center+(u.world_from_view[0].xyz*v.local.x+u.world_from_view[1].xyz*v.local.y)*extent;
  let ray=normalize(plane-u.world_from_view[3].xyz);let denom=dot(ray,normal);
  if abs(denom)<0.00001 {discard;}
  world=plane+ray*(dot(center-plane,normal)/denom);
  let delta=world.xz-center.xz;if dot(delta,delta)>radius*radius {discard;}

 }
 let clip=u.clip_from_world*vec4(world,1.);var out:Depth;out.linear=vec2(-(u.view_from_world*vec4(world,1.)).z,select(0.,1.+v.shape.y,v.shape.z>0.));out.depth=clip.z/clip.w;return out;
}
