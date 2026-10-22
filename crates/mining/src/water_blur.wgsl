struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32>, sediment: vec4<f32> };
struct Screen { @builtin(position) position: vec4<f32> };
@vertex fn fullscreen(@builtin(vertex_index) id: u32) -> Screen { var s: Screen; let xy = vec2(f32((id << 1u) & 2u), f32(id & 2u)); s.position = vec4(xy * 2.0 - 1.0, 0.0, 1.0); return s; }
@group(0) @binding(0) var depths: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var columns:texture_2d<f32>;
@group(0) @binding(3) var density_near:texture_2d<f32>;
@group(0) @binding(4) var density_far:texture_2d<f32>;
const PROFILE_DEPTHS:array<f32,8>=array(0.,0.002,0.005,0.009,0.015,0.023,0.035,0.051);
fn shared_surface_depth(raw:vec2<f32>,near:vec4<f32>,far:vec4<f32>)->f32 {
    if raw.x<=0. || raw.y>= -0.5 {return raw.x;}
    let density=array(near.x,near.y,near.z,near.w,far.x,far.y,far.z,far.w);
    let iso=0.5;
    var peak=0.;var peak_depth=0.;
    for(var i=0u;i<8u;i++) {
        if density[i]>=iso {
            if i==0u {return raw.x;}
            let t=clamp((iso-density[i-1u])/max(density[i]-density[i-1u],0.000001),0.,1.);
            return raw.x+mix(PROFILE_DEPTHS[i-1u],PROFILE_DEPTHS[i],t);
        }
        if density[i]>peak {peak=density[i];peak_depth=PROFILE_DEPTHS[i];}
    }
    // Retain a dilute film continuously as peak density approaches the level
    // set. Opaque geometry already clips every probe; never sample below it.
    return raw.x+peak_depth*smoothstep(0.05,iso,peak);
}
@fragment fn reconstruct(s:Screen)->@location(0) f32 {
    let q=vec2<i32>(s.position.xy);
    return shared_surface_depth(textureLoad(depths,q,0).rg,textureLoad(density_near,q,0),textureLoad(density_far,q,0));
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
