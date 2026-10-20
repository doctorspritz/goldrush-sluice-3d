struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32> };
struct Particle { center_radius: vec4<f32>, velocity: vec4<f32> };
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> u: Uniforms;
struct Varying { @builtin(position) clip: vec4<f32>, @location(0) local: vec2<f32>, @location(1) center_radius: vec4<f32> };
@vertex fn particle_vertex(@builtin(vertex_index) id: u32, @builtin(instance_index) instance: u32) -> Varying {
    let corners = array<vec2<f32>,6>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
    let particle = particles[instance]; let p = vec4(particle.center_radius.xyz + particle.velocity.xyz * u.viewport.w, particle.center_radius.w); let local = corners[id];
    let world = p.xyz + (u.world_from_view[0].xyz * local.x + u.world_from_view[1].xyz * local.y) * p.w;
    var out: Varying; out.clip = u.clip_from_world * vec4(world,1.); out.local = local; out.center_radius = p; return out;
}
struct Depth { @location(0) linear: f32, @builtin(frag_depth) depth: f32 };
@fragment fn particle_depth(v: Varying) -> Depth {
    let r2 = dot(v.local,v.local); if r2 > 1. { discard; }
    let world = v.center_radius.xyz + (u.world_from_view[0].xyz * v.local.x + u.world_from_view[1].xyz * v.local.y + u.world_from_view[2].xyz * sqrt(max(0.,1.-r2))) * v.center_radius.w;
    let clip = u.clip_from_world * vec4(world,1.); var out: Depth;
    out.linear = -(u.view_from_world * vec4(world,1.)).z; out.depth = clip.z/clip.w; return out;
}
