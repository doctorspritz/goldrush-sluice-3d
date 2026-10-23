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
// Keep depth and the two additive optical moments in one reconstruction.
// Dividing/filtering opacity separately leaves particle-sized mud patches on
// an otherwise smooth interface. Alpha distinguishes channel from outlet.
@fragment fn reconstruct(s:Screen)->@location(0) vec4<f32> {
    let q=vec2<i32>(s.position.xy);
    let raw=textureLoad(depths,q,0);
    let near=textureLoad(density_near,q,0);let far=textureLoad(density_far,q,0);
    let peak=max(max(max(near.x,near.y),max(near.z,near.w)),
                 max(max(far.x,far.y),max(far.z,far.w)));
    let channel=select(0.,1.+smoothstep(0.05,0.5,peak),raw.g< -0.5);
    return vec4(shared_surface_depth(raw.rg,near,far),textureLoad(columns,q,0).rg,channel);
}
fn smooth_surface(pixel: vec2<i32>, direction: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(depths)); let center = textureLoad(depths,pixel,0);
    if center.r <= 0. { return vec4(0.); }
    // The summed density probes identify connected channel water. Its joined
    // interface needs a broader footprint than detached spray; otherwise the
    // remaining kernel domes reappear as circular crest highlights. Outlet
    // columns have zero density probes and retain the existing 15-tap filter.
    let near=textureLoad(density_near,pixel,0);
    let far=textureLoad(density_far,pixel,0);
    // A connected layer may start behind a front spray particle. Use the same
    // full density profile that selected the interface, rather than only the
    // old 4/12 mm probes, which misclassified the delayed crest as a drop.
    let peak=max(max(max(near.x,near.y),max(near.z,near.w)),
                 max(max(far.x,far.y),max(far.z,far.w)));
    let bulk=smoothstep(0.05,0.5,peak);
    // View depth changes rapidly along an oblique sheet. Comparing every tap
    // to the centre depth mistakes that plane slope for a discontinuity and
    // keeps the small kernel domes. Fit a projective tangent (inverse depth is
    // affine on a plane), then reject residuals about it. Detached water keeps
    // its original range test and arithmetic depth filter.
    var slope=0.;var plane_origin=1./center.r;
    if bulk>0. && center.a>0.5 {
        let left=textureLoad(depths,clamp(pixel-direction*6,vec2(0),size-vec2(1)),0);
        let right=textureLoad(depths,clamp(pixel+direction*6,vec2(0),size-vec2(1)),0);
        if left.r>0. && right.r>0. && left.a>0.5 && right.a>0.5 {
            slope=(1./right.r-1./left.r)/12.;
            plane_origin=(1./right.r+1./left.r)*0.5;
        }
    }
    // A fixed pixel radius becomes a broad, round mound at grazing angles.
    // Bound connected-water diffusion to a 30 mm tangent footprint per pass.
    // The projective tangent includes both image spacing and sheet slope.
    var tangent_step2=0.;
    if bulk>0. && center.a>0.5 {
        let uv=(vec2<f32>(pixel)+0.5)/vec2<f32>(size);
        let ray=u.view_from_clip*vec4(uv.x*2.-1.,1.-uv.y*2.,1.,1.);
        let direction_view=vec3(2.*u.view_from_clip[0].x*f32(direction.x)/f32(size.x),
                              -2.*u.view_from_clip[1].y*f32(direction.y)/f32(size.y),0.);
        let z=1./max(plane_origin,0.000001);
        let next_z=1./max(plane_origin+slope,0.000001);
        let tangent=ray.xyz/max(-ray.z,0.000001)*(next_z-z)+direction_view*next_z;
        tangent_step2=dot(tangent,tangent);
    }
    var total = vec4(0.); var weights = 0.;var inverse_depth=0.;
    for (var i = -11; i <= 11; i++) {
        if abs(i)>7 && bulk==0. {continue;}
        let q = clamp(pixel + direction * i, vec2(0),size-vec2(1)); let sample = textureLoad(depths,q,0);
        if sample.r <= 0. || (sample.a>0.5)!=(center.a>0.5) { continue; }
        let expected=1./max(plane_origin+slope*f32(i),0.000001);
        let distance = (sample.r-mix(center.r,expected,bulk))/u.viewport.z;
        let tail=select(bulk,1.,abs(i)<=7);
        let spatial=max(f32(i*i)/mix(24.,48.,bulk),f32(i*i)*tangent_step2*bulk/(0.030*0.030));
        let weight = tail*exp(-spatial - distance*distance);
        total += sample*weight; weights += weight;inverse_depth+=weight/sample.r;
    }
    let filtered=total/max(weights,0.0001);
    return vec4(mix(filtered.r,weights/max(inverse_depth,0.000001),bulk),filtered.gb,center.a);
}
@fragment fn blur_x(s: Screen) -> @location(0) vec4<f32> { return smooth_surface(vec2<i32>(s.position.xy),vec2(1,0)); }
@fragment fn blur_y(s: Screen) -> @location(0) vec4<f32> { return smooth_surface(vec2<i32>(s.position.xy),vec2(0,1)); }
