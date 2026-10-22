struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32>, sediment: vec4<f32> };
struct Screen { @builtin(position) position: vec4<f32> };
@vertex fn fullscreen(@builtin(vertex_index) id: u32) -> Screen { var s: Screen; let xy = vec2(f32((id << 1u) & 2u), f32(id & 2u)); s.position = vec4(xy * 2.0 - 1.0, 0.0, 1.0); return s; }
@group(0) @binding(0) var depths: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var columns:texture_2d<f32>;
fn shared_surface_depth(raw:vec2<f32>,density:vec2<f32>)->f32 {
    if raw.x<=0. || raw.y>= -0.5 {return raw.x;}
    // Half rest density is the visual liquid/air interface. Piecewise linear
    // crossing is bounded to the original 12 mm front-layer footprint. Fade
    // weak support continuously: switching between a crossing and a sparse
    // film must never jump the interface by the full probe separation.
    let iso_density=0.5;
    if density.x>=iso_density {return raw.x+0.004*iso_density/max(density.x,0.000001);}
    if density.y>=iso_density {
        return raw.x+0.004+0.008*(iso_density-density.x)/max(density.y-density.x,0.000001);
    }
    return raw.x+0.012*smoothstep(0.05,iso_density,max(density.x,density.y));
}
@fragment fn reconstruct(s:Screen)->@location(0) f32 {
    let q=vec2<i32>(s.position.xy);
    return shared_surface_depth(textureLoad(depths,q,0).rg,textureLoad(columns,q,0).ba);
}
fn smooth_depth(pixel: vec2<i32>, direction: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depths)); let center = textureLoad(depths,pixel,0).r;
    if center <= 0. { return 0.; }
    // The summed density probes identify connected channel water. Its joined
    // interface needs a broader footprint than detached spray; otherwise the
    // remaining kernel domes reappear as circular crest highlights. Outlet
    // columns have zero density probes and retain the existing 15-tap filter.
    let density=textureLoad(columns,pixel,0).ba;
    let bulk=smoothstep(0.05,0.5,max(density.x,density.y));
    var total = 0.; var weights = 0.;
    for (var i = -11; i <= 11; i++) {
        if abs(i)>7 && bulk==0. {continue;}
        let q = clamp(pixel + direction * i, vec2(0),size-vec2(1)); let sample = textureLoad(depths,q,0).r;
        if sample <= 0. { continue; }
        let distance = (sample-center)/u.viewport.z;
        let tail=select(bulk,1.,abs(i)<=7);
        let weight = tail*exp(-f32(i*i)/mix(24.,48.,bulk) - distance*distance);
        total += sample*weight; weights += weight;
    }
    return total/max(weights,0.0001);
}
@fragment fn blur_x(s: Screen) -> @location(0) f32 { return smooth_depth(vec2<i32>(s.position.xy),vec2(1,0)); }
@fragment fn blur_y(s: Screen) -> @location(0) f32 { return smooth_depth(vec2<i32>(s.position.xy),vec2(0,1)); }
