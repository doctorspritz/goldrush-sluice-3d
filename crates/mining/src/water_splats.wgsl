// Analytic sphere splats: billboard geometry, curved per-fragment depth.
#import bevy_pbr::mesh_view_bindings::view

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(3) center_radius: vec4<f32>,
    @location(4) velocity: vec4<f32>,
};
struct Varying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) center_radius: vec4<f32>,
    @location(2) speed: f32,
};
@vertex fn vertex(v: Vertex) -> Varying {
    var out: Varying;
    let local = v.position.xy * 2.0;
    let right = view.world_from_view[0].xyz;
    let up = view.world_from_view[1].xyz;
    let world = v.center_radius.xyz + (right * local.x + up * local.y) * v.center_radius.w;
    out.clip_position = view.clip_from_world * vec4(world, 1.0);
    out.local = local;
    out.center_radius = v.center_radius;
    out.speed = length(v.velocity.xyz);
    return out;
}
struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};
@fragment fn fragment(v: Varying) -> Fragment {
    let r2 = dot(v.local, v.local);
    if r2 > 1.0 { discard; }
    let front = sqrt(max(0.0, 1.0 - r2));
    let normal = normalize(view.world_from_view[0].xyz * v.local.x
        + view.world_from_view[1].xyz * v.local.y + view.world_from_view[2].xyz * front);
    let world = v.center_radius.xyz + normal * v.center_radius.w;
    let clip = view.clip_from_world * vec4(world, 1.0);
    let light = normalize(vec3(0.4, 0.85, 0.3));
    let diffuse = 0.55 + 0.45 * max(dot(normal, light), 0.0);
    let fresnel = pow(1.0 - front, 3.0);
    let half_vector = normalize(light + view.world_from_view[2].xyz);
    let specular = pow(max(dot(normal, half_vector), 0.0), 64.0);
    let body = mix(vec3(0.025, 0.16, 0.17), vec3(0.075, 0.27, 0.26), clamp(v.speed, 0.0, 1.0));
    var out: Fragment;
    out.color = vec4(body * diffuse + vec3(0.38, 0.55, 0.60) * fresnel * 0.18 + vec3(specular * 0.35), 1.0);
    out.depth = clip.z / clip.w;
    return out;
}
