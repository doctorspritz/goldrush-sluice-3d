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
            return raw.x+mix(PROFILE_DEPTHS[i-1u],PROFILE_DEPTHS[i],t)*max(-raw.y,1.);
        }
        if density[i]>peak {peak=density[i];peak_depth=PROFILE_DEPTHS[i];}
    }
    // Retain a dilute film continuously as peak density approaches the level
    // set. Opaque geometry already clips every probe; never sample below it.
    return raw.x+peak_depth*max(-raw.y,1.)*smoothstep(0.05,iso,peak);
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
// Estimate the sheet outside the central kernel bump. Each side supplies a
// one-sided slope; a flat layer step has no matching tangent across its gap.
// Blend uncertain fits back to the centre instead of switching the whole
// reconstruction at a curvature threshold.
fn sheet_axis(pixel:vec2<i32>,direction:vec2<i32>,size:vec2<i32>,center:f32)->vec4<f32> {
    let a=clamp(pixel-direction*6,vec2(0),size-vec2(1));
    let b=clamp(pixel-direction*3,vec2(0),size-vec2(1));
    let c=clamp(pixel+direction*3,vec2(0),size-vec2(1));
    let d=clamp(pixel+direction*6,vec2(0),size-vec2(1));
    let av=textureLoad(depths,a,0);let bv=textureLoad(depths,b,0);
    let cv=textureLoad(depths,c,0);let dv=textureLoad(depths,d,0);
    // Clamped border stencils do not have the symmetric quadratic metric.
    if any(a!=pixel-direction*6) || any(d!=pixel+direction*6) ||
       min(min(av.r,bv.r),min(cv.r,dv.r))<=0. ||
       min(min(av.a,bv.a),min(cv.a,dv.a))<=0.5 {
        return vec4(0.,1./center,0.,0.);
    }
    let dl=(1./bv.r-1./av.r)/3.;let dr=(1./dv.r-1./cv.r)/3.;
    let slope=(dl+dr)*0.5;
    let curvature=(dr-dl)/9.;
    // Evaluate both one-sided fits at the centre. A genuine flat layer
    // step disagrees here; a smooth crest has opposite slopes and curvature.
    // The old minmod plane discarded that curvature and flattened the fit
    // toward the shoulders, fragmenting its normal and optical footprint.
    let lo=1./bv.r+3.*slope-4.5*curvature;
    let hi=1./cv.r-3.*slope-4.5*curvature;
    let tolerance=max(max(abs(dl),abs(dr))*1.5,0.00001);
    let confidence=1.-smoothstep(tolerance,tolerance*2.,abs(lo-hi));
    return vec4(slope*confidence,mix(1./center,(lo+hi)*0.5,confidence),
                curvature*confidence,confidence);
}
// Symmetric pair exchange preserves the sum of both additive optical
// moments on a flat finite sheet. The harmonic water support tends to zero
// continuously as a film appears/disappears; no domain-sized reflection jump.
fn optical_support(a:f32,b:f32)->f32 {
    let total=a+b;
    return clamp(select(0.,4.*(a*b)/max(total*total,1e-30),total>0.),0.,1.);
}
fn optical_point(pixel:vec2<i32>,size:vec2<i32>,depth:f32)->vec3<f32> {
    let uv=(vec2<f32>(pixel)+0.5)/vec2<f32>(size);
    let ray=u.view_from_clip*vec4(uv.x*2.-1.,1.-uv.y*2.,1.,1.);
    return ray.xyz*(depth/max(-ray.z,0.000001));
}
fn optical_exchange_inner(pixel:vec2<i32>,direction:vec2<i32>,center:vec4<f32>,visible:bool)->vec2<f32> {
    let size=vec2<i32>(textureDimensions(depths));
    let origin=optical_point(pixel,size,center.r);
    var result=center.gb;
    for(var side=0;side<2;side++) {
        let sign=select(-1,1,side==1);
        var previous=center;var connection=1.;
        for(var distance=1;distance<=4;distance++) {
            let q=pixel+direction*(sign*distance);
            if any(q<vec2(0)) || any(q>=size) {break;}
            var sample=textureLoad(depths,q,0);
            if sample.r<=0. || sample.a<=0.5 {break;}
            if visible {sample.g=textureLoad(columns,q,0).r;sample.b=textureLoad(columns,q,0).g;}
            connection=min(connection,optical_support(previous.g,sample.g));
            let delta=optical_point(q,size,sample.r)-origin;
            // Eight neighbours, each at most 1/8: the update is positive.
            // Its one-pass world-space RMS displacement is bounded by
            // 45 mm/sqrt(e) < 30 mm, independent of the projected pixel size.
            let weight=0.125*connection*exp(-dot(delta,delta)/(0.045*0.045));
            result+=(sample.gb-center.gb)*weight;
            previous=sample;
        }
    }
    return result;
}
fn optical_exchange(pixel:vec2<i32>,direction:vec2<i32>,center:vec4<f32>)->vec2<f32> {
    return optical_exchange_inner(pixel,direction,center,false);
}
// Seed from the newly integrated visible moments, then perform the same
// positive symmetric exchange. Geometry is frozen before integration: neither
// exchange may move the interface and expose a different set of ray intervals.
@fragment fn visible_optical_x(s:Screen)->@location(0) vec4<f32> {
    let pixel=vec2<i32>(s.position.xy);var center=textureLoad(depths,pixel,0);
    if center.r<=0. || center.a<=0.5 {return center;}
    center.g=textureLoad(columns,pixel,0).r;center.b=textureLoad(columns,pixel,0).g;
    return vec4(center.r,optical_exchange_inner(pixel,vec2(1,0),center,true),center.a);
}
@fragment fn visible_optical_y(s:Screen)->@location(0) vec4<f32> {
    let pixel=vec2<i32>(s.position.xy);let center=textureLoad(depths,pixel,0);
    if center.r<=0. || center.a<=0.5 {return center;}
    return vec4(center.r,optical_exchange(pixel,vec2(0,1),center),center.a);
}
fn smooth_surface_inner(pixel: vec2<i32>, direction: vec2<i32>,exchange:bool) -> vec4<f32> {
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
    var slope=0.;var curvature=0.;var plane_origin=1./center.r;
    if bulk>0. && center.a>0.5 {
        let tangent=sheet_axis(pixel,direction,size,center.r);
        slope=tangent.x;plane_origin=tangent.y;curvature=tangent.z;
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
    var total = vec4(0.); var weights = 0.;var residual_depth=0.;
    for (var i = -11; i <= 11; i++) {
        if abs(i)>7 && bulk==0. {continue;}
        let q = clamp(pixel + direction * i, vec2(0),size-vec2(1)); let sample = textureLoad(depths,q,0);
        if sample.r <= 0. || (sample.a>0.5)!=(center.a>0.5) { continue; }
        let expected=1./max(plane_origin+slope*f32(i)+0.5*curvature*f32(i*i),0.000001);
        let distance = (sample.r-mix(center.r,expected,bulk))/u.viewport.z;
        let tail=select(bulk,1.,abs(i)<=7);
        let spatial=max(f32(i*i)/mix(24.,48.,bulk),f32(i*i)*tangent_step2*bulk/(0.030*0.030));
        let weight = tail*exp(-spatial - distance*distance);
        total += sample*weight; weights += weight;
        // Filter only the unresolved residual. Averaging the quadratic itself
        // lowers narrow crests and spreads their shoulders on every pass.
        // A resolved projective crest should survive exactly, like a plane.
        let fitted_inverse=plane_origin+slope*f32(i)+0.5*curvature*f32(i*i);
        residual_depth+=(1./sample.r-fitted_inverse)*weight;
    }
    let filtered=total/max(weights,0.0001);
    var moments=filtered.gb;
    if center.a>0.5 {moments=center.gb;if exchange {moments=optical_exchange(pixel,direction,center);}}
    let fitted_depth=1./max(plane_origin+residual_depth/max(weights,0.0001),0.000001);
    return vec4(mix(filtered.r,fitted_depth,bulk),moments,center.a);
}
// Establish the shared geometry before using its distances to exchange optics.
// Four geometry filters retain the original shape. Two final passes exchange
// optics and polish unresolved depth noise, reusing ping/pong. They never add
// water coverage or re-filter the outlet.
fn smooth_surface(pixel:vec2<i32>,direction:vec2<i32>)->vec4<f32> {
    return smooth_surface_inner(pixel,direction,true);
}
fn geometry_surface(pixel:vec2<i32>,direction:vec2<i32>)->vec4<f32> {
    let center=textureLoad(depths,pixel,0);
    if center.r>0. && center.a>0.5 {
        let near=textureLoad(density_near,pixel,0);
        let far=textureLoad(density_far,pixel,0);
        let peak=max(max(max(near.x,near.y),max(near.z,near.w)),
                     max(max(far.x,far.y),max(far.z,far.w)));
        let connected=smoothstep(0.05,0.5,peak)*smoothstep(0.1,0.7,center.a-1.);
        if connected>0. {
            // Broad shoulder fits retain a parabola but flatten resolved
            // travelling ridges before the final normal/optical gather. Use
            // the compact high-order filter for a supported connected sheet.
            // It removes sampling noise without repeatedly widening the wave;
            // water/sediment moments retain their separate conservative pass.
            let depth=resolved_depth(pixel,direction,center);
            if connected==1. {return vec4(depth,center.gba);}
            let sparse=smooth_surface_inner(pixel,direction,false);
            return vec4(mix(sparse.r,depth,connected),sparse.gba);
        }
    }
    return smooth_surface_inner(pixel,direction,false);
}
// Reject unresolved alternating texels without broadening a resolved crest.
// This compact sixth-difference filter exactly retains degree-five inverse
// depth profiles. Its response is 1-sin(frequency/2)^6: a pixel-scale ripple
// vanishes while a ten-texel wave retains over 99.9% of its amplitude.
fn resolved_depth(pixel:vec2<i32>,direction:vec2<i32>,center:vec4<f32>)->f32 {
    let size=vec2<i32>(textureDimensions(depths));
    var values:array<f32,7>;var support=1.;var variation=0.;var jump=0.;
    for(var i=0;i<7;i++) {
        let q=pixel+direction*(i-3);
        if any(q<vec2(0)) || any(q>=size) {return center.r;}
        let tap=textureLoad(depths,q,0);
        if tap.r<=0. || tap.a<=0.5 {return center.r;}
        values[i]=1./tap.r;support=min(support,smoothstep(0.1,0.7,tap.a-1.));
        if i>0 {let change=abs(values[i]-values[i-1]);variation+=change;jump=max(jump,change);}
    }
    // An isolated jump between layers owns most of the local variation.
    // Alternating sampling noise does not. Do not smooth across that jump.
    let continuous=1.-smoothstep(0.5,0.8,jump/max(variation,0.000001));
    let sixth=(-values[0]+6.*values[1]-15.*values[2]+20.*values[3]
               -15.*values[4]+6.*values[5]-values[6])/64.;
    let inverse=1./center.r-sixth*support*continuous;
    return select(center.r,1./max(inverse,0.000001),inverse>0.);
}
fn optical_surface(pixel:vec2<i32>,direction:vec2<i32>)->vec4<f32> {
    let center=textureLoad(depths,pixel,0);
    if center.r<=0. {return vec4(0.);}
    if center.a<=0.5 {return center;}
    return vec4(resolved_depth(pixel,direction,center),optical_exchange(pixel,direction,center),center.a);
}
@fragment fn geometry_x(s:Screen)->@location(0) vec4<f32> {return geometry_surface(vec2<i32>(s.position.xy),vec2(1,0));}
@fragment fn geometry_y(s:Screen)->@location(0) vec4<f32> {return geometry_surface(vec2<i32>(s.position.xy),vec2(0,1));}
@fragment fn optical_x(s:Screen)->@location(0) vec4<f32> {return optical_surface(vec2<i32>(s.position.xy),vec2(1,0));}
@fragment fn optical_y(s:Screen)->@location(0) vec4<f32> {return optical_surface(vec2<i32>(s.position.xy),vec2(0,1));}
@fragment fn blur_x(s: Screen) -> @location(0) vec4<f32> { return smooth_surface(vec2<i32>(s.position.xy),vec2(1,0)); }
@fragment fn blur_y(s: Screen) -> @location(0) vec4<f32> { return smooth_surface(vec2<i32>(s.position.xy),vec2(0,1)); }
