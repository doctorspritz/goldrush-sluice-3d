struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32>, sediment: vec4<f32> };
struct Screen { @builtin(position) position: vec4<f32> };
@vertex fn fullscreen(@builtin(vertex_index) id: u32) -> Screen { var s: Screen; let xy = vec2(f32((id << 1u) & 2u), f32(id & 2u)); s.position = vec4(xy * 2.0 - 1.0, 0.0, 1.0); return s; }
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depths: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> u: Uniforms;
@group(0) @binding(4) var raw_depths: texture_2d<f32>;
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
    // Smoothing must not pull hidden water forward over an opaque side wall.
    // Test original particle depth as well as the reconstructed surface.
    let raw_pixel = clamp(vec2<i32>(q * vec2<f32>(textureDimensions(raw_depths)) / u.viewport.xy), vec2(0), vec2<i32>(textureDimensions(raw_depths))-vec2(1));
    let raw_depth = textureLoad(raw_depths,raw_pixel,0).r;
    if raw_depth <= 0. { return original; }
    let opaque = textureLoad(scene_depth,pixel,0);
    let raw_clip = u.clip_from_world * u.world_from_view * vec4(view_point(q,raw_depth),1.);
    if raw_clip.z/raw_clip.w <= opaque + 0.000001 { return original; }
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
    let opaque_view = u.view_from_clip * vec4(q.x/u.viewport.x*2.-1.,1.-q.y/u.viewport.y*2.,opaque,1.);
    var thickness = 0.04;
    if opaque>0. && abs(opaque_view.w)>0.00001 { thickness=clamp(-opaque_view.z/opaque_view.w-depth,0.,0.5); }
    let concentration=u.sediment.x;
    let extinction=vec3(4.,2.,0.9)+vec3(45.,80.,120.)*concentration*30.;
    let transmission=exp(-extinction*thickness);
    let scattering=mix(vec3(0.035,0.18,0.18),vec3(0.22,0.13,0.055),clamp(concentration*20.,0.,1.));
    let body=original.rgb*transmission+scattering*(vec3(1.)-transmission);
    return vec4(mix(original.rgb,mix(body,sky,fresnel)+vec3(specular),clamp(sample.y,0.,1.)),original.a);
}
