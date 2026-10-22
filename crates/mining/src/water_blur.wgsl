struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32>, sediment: vec4<f32> };
struct Screen { @builtin(position) position: vec4<f32> };
@vertex fn fullscreen(@builtin(vertex_index) id: u32) -> Screen { var s: Screen; let xy = vec2(f32((id << 1u) & 2u), f32(id & 2u)); s.position = vec4(xy * 2.0 - 1.0, 0.0, 1.0); return s; }
@group(0) @binding(0) var depths: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var columns:texture_2d<f32>;
fn shared_surface_depth(raw:vec2<f32>,moments:vec2<f32>)->f32 {
    if raw.x<=0. || raw.y>= -0.5 || moments.y<=0.000001 {return raw.x;}
    return raw.x+clamp(moments.x/moments.y,0.,0.012);
}
@fragment fn reconstruct(s:Screen)->@location(0) f32 {
    let q=vec2<i32>(s.position.xy);
    return shared_surface_depth(textureLoad(depths,q,0).rg,textureLoad(columns,q,0).ba);
}
fn smooth_depth(pixel: vec2<i32>, direction: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depths)); let center = textureLoad(depths,pixel,0).r;
    if center <= 0. { return 0.; }
    var total = 0.; var weights = 0.;
    for (var i = -7; i <= 7; i++) {
        let q = clamp(pixel + direction * i, vec2(0),size-vec2(1)); let sample = textureLoad(depths,q,0).r;
        if sample <= 0. { continue; }
        let distance = (sample-center)/u.viewport.z;
        let weight = exp(-f32(i*i)/24. - distance*distance);
        total += sample*weight; weights += weight;
    }
    return total/max(weights,0.0001);
}
@fragment fn blur_x(s: Screen) -> @location(0) f32 { return smooth_depth(vec2<i32>(s.position.xy),vec2(1,0)); }
@fragment fn blur_y(s: Screen) -> @location(0) f32 { return smooth_depth(vec2<i32>(s.position.xy),vec2(0,1)); }
