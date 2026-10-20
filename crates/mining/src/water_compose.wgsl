struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32> };
struct Screen { @builtin(position) position: vec4<f32> };
@vertex fn fullscreen(@builtin(vertex_index) id: u32) -> Screen { var s: Screen; let xy = vec2(f32((id << 1u) & 2u), f32(id & 2u)); s.position = vec4(xy * 2.0 - 1.0, 0.0, 1.0); return s; }
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depths: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> u: Uniforms;
fn depth_at(q: vec2<i32>) -> f32 { return textureLoad(depths,clamp(q,vec2(0),vec2<i32>(textureDimensions(depths))-vec2(1)),0).r; }
// Bilateral upsampling keeps the half-resolution smoothing buffer from
// producing two-pixel shading blocks. Empty pixels contribute only coverage.
fn sample_depth(pixel: vec2<f32>) -> vec2<f32> {
    let p = pixel * vec2<f32>(textureDimensions(depths))/u.viewport.xy - 0.5;
    let q = vec2<i32>(floor(p)); let f = fract(p);
    let offsets = array<vec2<i32>,4>(vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
    let weights = array<f32,4>((1.-f.x)*(1.-f.y),f.x*(1.-f.y),(1.-f.x)*f.y,f.x*f.y);
    let reference = depth_at(vec2<i32>(round(p)));
    var total = 0.; var weight = 0.;
    for (var i=0u; i<4u; i++) {
        let d = depth_at(q+offsets[i]);
        if d>0. && (reference<=0. || abs(d-reference)<u.viewport.z*2.) { total += d*weights[i]; weight += weights[i]; }
    }
    return vec2(total/max(weight,0.0001),weight);
}
fn view_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel/u.viewport.xy;
    let ray = u.view_from_clip * vec4(uv.x*2.-1.,1.-uv.y*2.,1.,1.);
    return ray.xyz * (depth / max(-ray.z,0.0001));
}
@fragment fn compose(s: Screen) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(s.position.xy); let original = textureLoad(scene,pixel,0);
    let q = s.position.xy; let sample = sample_depth(q);
    let depth = sample.x; if depth <= 0. || sample.y <= 0.05 { return original; }
    let position = view_point(q,depth); let world = u.world_from_view * vec4(position,1.);
    let clip = u.clip_from_world * world;
    if clip.z/clip.w <= textureLoad(scene_depth,pixel,0) + 0.000001 { return original; }
    let left = sample_depth(q-vec2(1.,0.)).x; let right = sample_depth(q+vec2(1.,0.)).x;
    let up = sample_depth(q-vec2(0.,1.)).x; let down = sample_depth(q+vec2(0.,1.)).x;
    var dx = view_point(q+vec2(1.,0.),select(depth,right,right>0.)) - position;
    var dy = view_point(q+vec2(0.,1.),select(depth,down,down>0.)) - position;
    if left > 0. && (right <= 0. || abs(left-depth)<abs(right-depth)) { dx = position-view_point(q-vec2(1.,0.),left); }
    if up > 0. && (down <= 0. || abs(up-depth)<abs(down-depth)) { dy = position-view_point(q-vec2(0.,1.),up); }
    var n = normalize(cross(dy,dx)); if n.z < 0. { n = -n; }
    let normal = normalize((u.world_from_view * vec4(n,0.)).xyz);
    let eye = normalize(-position); let fresnel = 0.035 + 0.50*pow(1.-max(dot(n,eye),0.),4.);
    let light = normalize(vec3(0.35,0.85,0.3));
    let half_vector = normalize(light + normalize(u.world_from_view[3].xyz-world.xyz));
    let specular = pow(max(dot(normal,half_vector),0.),72.) * 0.38;
    let sky = mix(vec3(0.21,0.36,0.43),vec3(0.62,0.76,0.82),clamp(normal.y*0.5+0.5,0.,1.));
    let body = mix(original.rgb,vec3(0.035,0.18,0.18),0.68);
    return vec4(mix(original.rgb,mix(body,sky,fresnel)+vec3(specular),clamp(sample.y,0.,1.)),original.a);
}
