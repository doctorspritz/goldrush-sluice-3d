struct Uniforms { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>, view_from_world: mat4x4<f32>, viewport: vec4<f32>, sediment: vec4<f32> };
struct Screen { @builtin(position) position: vec4<f32> };
@vertex fn fullscreen(@builtin(vertex_index) id: u32) -> Screen { var s: Screen; let xy = vec2(f32((id << 1u) & 2u), f32(id & 2u)); s.position = vec4(xy * 2.0 - 1.0, 0.0, 1.0); return s; }
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depths: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> u: Uniforms;
@group(0) @binding(4) var raw_depths: texture_2d<f32>;
@group(0) @binding(5) var fluid_thickness:texture_2d<f32>;
@group(0) @binding(6) var surface_normals:texture_2d<f32>;
@group(0) @binding(7) var river_depths:texture_2d<f32>;
@group(0) @binding(8) var river_normals:texture_2d<f32>;
@group(0) @binding(9) var normal_sums:texture_2d<f32>;
// One interpolating sheet owns subpixel channel position and its tangent.
// The wider shoulder fit is useful at sparse edges, but bends a resolved
// non-quadratic ridge into a cap and gives its slurry a different visibility.
struct ChannelPatch { inverse:f32, slope:vec2<f32>, curvature:vec3<f32>, confidence:f32 };
fn cubic_value(t:f32)->vec4<f32> {
    return vec4(-0.5*t+t*t-0.5*t*t*t,1.-2.5*t*t+1.5*t*t*t,
                0.5*t+2.*t*t-1.5*t*t*t,-0.5*t*t+0.5*t*t*t);
}
fn cubic_slope(t:f32)->vec4<f32> {
    return vec4(-0.5+2.*t-1.5*t*t,-5.*t+4.5*t*t,
                0.5+4.*t-4.5*t*t,-t+1.5*t*t);
}
fn cubic_curvature(t:f32)->vec4<f32> {
    return vec4(2.-3.*t,-5.+9.*t,4.-9.*t,-1.+3.*t);
}
fn channel_patch(pixel:vec2<f32>)->ChannelPatch {
    var out:ChannelPatch;
    let size=vec2<i32>(textureDimensions(depths));
    let p=pixel*vec2<f32>(size)/u.viewport.xy-0.5;
    let base=vec2<i32>(floor(p));let f=fract(p);
    if any(base-vec2(1)<vec2(0)) || any(base+vec2(2)>=size) {return out;}
    var rows:array<vec4<f32>,4>;var support=1.;
    for(var y=0;y<4;y++){for(var x=0;x<4;x++){
        let tap=textureLoad(depths,base+vec2(x-1,y-1),0);
        if tap.r<=0. || tap.a<=0.5 {return out;}
        rows[y][x]=1./tap.r;
        support=min(support,smoothstep(0.1,0.7,tap.a-1.));
    }}
    let wx=cubic_value(f.x);let wy=cubic_value(f.y);
    let dx=cubic_slope(f.x);let dy=cubic_slope(f.y);
    let ddx=cubic_curvature(f.x);let ddy=cubic_curvature(f.y);
    var values=vec4(0.);var slopes=vec4(0.);var curves=vec4(0.);
    var third=0.;
    for(var y=0;y<4;y++){
        values[y]=dot(rows[y],wx);slopes[y]=dot(rows[y],dx);curves[y]=dot(rows[y],ddx);
        third=max(third,abs(dot(rows[y],vec4(-1.,3.,-3.,1.))));
    }
    for(var x=0;x<4;x++){
        third=max(third,abs(-rows[0][x]+3.*rows[1][x]-3.*rows[2][x]+rows[3][x]));
    }
    out.inverse=dot(values,wy);out.slope=vec2(dot(slopes,wy),dot(values,dy));
    out.curvature=vec3(dot(curves,wy),dot(values,ddy),dot(slopes,dy));
    if out.inverse<=0. {return out;}
    let z=1./out.inverse;let origin=view_point(pixel,z);let ray=origin/z;
    let step=u.viewport.xy/vec2<f32>(size);
    let tx=view_point(pixel+vec2(step.x,0.),z)-origin-ray*(z*z*out.slope.x);
    let ty=view_point(pixel+vec2(0.,step.y),z)-origin-ray*(z*z*out.slope.y);
    // Measure the non-smooth part in metres normal to the sheet. A steep
    // continuous plane passes; a real step between flat layers cannot supply
    // a cubic tangent, new coverage, or another layer's muddy column.
    let normal_per_inverse=z*z*abs(dot(ray,normalize(cross(ty,tx))));
    out.confidence=support*(1.-smoothstep(0.005,0.025,third*normal_per_inverse));
    return out;
}
// Bilateral upsampling keeps the half-resolution smoothing buffer from
// producing two-pixel shading blocks. Empty pixels contribute only coverage.
fn sample_depth(pixel: vec2<f32>) -> vec2<f32> {
    let p = pixel * vec2<f32>(textureDimensions(depths))/u.viewport.xy - 0.5;
    let q = vec2<i32>(floor(p)); let f = fract(p);
    let offsets = array<vec2<i32>,4>(vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
    let weights = array<f32,4>((1.-f.x)*(1.-f.y),f.x*(1.-f.y),(1.-f.x)*f.y,f.x*f.y);
    let reference = textureLoad(depths,clamp(vec2<i32>(round(p)),vec2(0),vec2<i32>(textureDimensions(depths))-vec2(1)),0);
    let channel=reference.a>0.5;
    var inverse_slope=vec2(0.);var normal_per_depth=1.;
    if channel && reference.r>0. {
        // A steep continuous crest can change camera depth by much more than
        // the edge threshold in one texel. Limit two-sided reciprocal-depth
        // slopes so real layer steps cannot masquerade as a tilted sheet.
        let size=vec2<i32>(textureDimensions(depths));
        let center=clamp(vec2<i32>(round(p)),vec2(0),size-vec2(1));
        for(var axis=0u;axis<2u;axis++) {
            let offset=select(vec2(1,0),vec2(0,1),axis==1u);
            let a=center-offset;let b=center+offset;
            if any(a<vec2(0)) || any(b>=size) {continue;}
            let left=textureLoad(depths,a,0);let right=textureLoad(depths,b,0);
            if left.r>0. && right.r>0. && left.a>0.5 && right.a>0.5 {
                let dl=1./reference.r-1./left.r;
                let dr=1./right.r-1./reference.r;
                inverse_slope[axis]=select(0.,sign(dl)*min(abs(dl),abs(dr)),dl*dr>0.);
            }
        }
        let origin=view_point(pixel,reference.r);let ray=origin/reference.r;
        let step=u.viewport.xy/vec2<f32>(size);
        let tx=view_point(pixel+vec2(step.x,0.),reference.r)-origin
            -ray*(reference.r*reference.r*inverse_slope.x);
        let ty=view_point(pixel+vec2(0.,step.y),reference.r)-origin
            -ray*(reference.r*reference.r*inverse_slope.y);
        normal_per_depth=max(abs(dot(ray,normalize(cross(ty,tx)))),0.0001);
    }
    var total = 0.; var weight = 0.;var inverse_depth=0.;
    for (var i=0u; i<4u; i++) {
        let tap=textureLoad(depths,clamp(q+offsets[i],vec2(0),vec2<i32>(textureDimensions(depths))-vec2(1)),0);
        let d=tap.r;
        if d<=0. {continue;}
        var range_weight=1.;
        if channel {
            if tap.a<=0.5 {continue;}
            let distance=vec2<f32>(q+offsets[i])-round(p);
            let expected=1./max(1./reference.r+dot(inverse_slope,distance),0.000001);
            let residual=abs(d-expected)*normal_per_depth;
            // Taper geometric discontinuities; a hard camera-Z cutoff made
            // both coverage and the visible muddy path jump at pixel phases.
            range_weight=1.-smoothstep(u.viewport.z,u.viewport.z*2.,residual);
        } else if reference.r>0. && abs(d-reference.r)>=u.viewport.z*2. {
            continue;
        }
        let w=weights[i]*range_weight;
        total += d*w;weight += w;inverse_depth+=w/d;
    }
    var depth=select(total/max(weight,0.0001),weight/max(inverse_depth,0.000001),channel);
    if channel && reference.r>0. && weight>0.9999 {
        let sheet=channel_patch(pixel);
        if sheet.confidence>0. {depth=mix(depth,1./sheet.inverse,sheet.confidence);}
    }
    return vec2(depth,weight);
}
// A secant through two separate layers is not a water tangent. Compare
// one-sided reciprocal-depth slopes outside the centre kernel, and use the
// agreement of their extrapolated planes as continuous fit confidence.
fn channel_axis(pixel:vec2<i32>,direction:vec2<i32>,size:vec2<i32>,depth:f32)->vec4<f32> {
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
        return vec4(0.,1./depth,0.,0.);
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
    return vec4(slope*confidence,(lo+hi)*0.5,
                curvature*confidence,confidence);
}
// Axis stencils cannot see mixed curvature. Without this term an oblique
// crest has a different normal/optical footprint from the same crest viewed
// along a screen axis. Require all four diagonal samples to agree with one
// curved sheet, so an unrelated layer cannot supply its missing curvature.
fn channel_cross(pixel:vec2<i32>,size:vec2<i32>,origin:f32,slope:vec2<f32>,curvature:vec2<f32>)->f32 {
    let offsets=array<vec2<i32>,4>(vec2(-3,-3),vec2(3,-3),vec2(-3,3),vec2(3,3));
    var values:array<f32,4>;var mixed=0.;
    for(var i=0u;i<4u;i++) {
        let q=pixel+offsets[i];
        if any(q<vec2(0)) || any(q>=size) {return 0.;}
        let tap=textureLoad(depths,q,0);
        if tap.r<=0. || tap.a<=0.5 {return 0.;}
        let offset=vec2<f32>(offsets[i]);
        values[i]=1./tap.r-origin-dot(slope,offset)-0.5*dot(curvature,offset*offset);
        mixed+=values[i]/(offset.x*offset.y)*0.25;
    }
    var error=0.;
    for(var i=0u;i<4u;i++) {
        error=max(error,abs(values[i]-mixed*f32(offsets[i].x*offsets[i].y)));
    }
    let tolerance=max(max(abs(slope.x),abs(slope.y))*1.5,0.00001);
    return mixed*(1.-smoothstep(tolerance,tolerance*2.,error));
}
struct ChannelSurface {normal:vec4<f32>,column:vec2<f32>,flow:vec2<f32>};
// Gather every texel in the footprint. Striding over alternating texels makes
// the surface and its turbidity change when the half-size pixel phase changes.
// Share visibility tests between the two additive fields, before normalizing.
fn channel_surface(pixel:vec2<f32>,depth:f32)->ChannelSurface {
    var sum=vec3(0.);var weight=0.;
    var position_sum=vec3(0.);var second=vec3(0.);var cross_depth=vec2(0.);var plane_weight=0.;
    var column=vec2(0.);var column_weight=0.;
    var flow_sum=vec2(0.);var flow_weight=0.;
    let size=vec2<i32>(textureDimensions(surface_normals));
    let p=pixel*vec2<f32>(size)/u.viewport.xy-0.5;
    let center=vec2<i32>(round(p));
    // Inverse view depth is affine on an inclined sheet. Reject optical and
    // normal samples about that sheet, rather than about the camera-Z depth
    // of one pixel. The latter cuts a continuous crest into opacity islands.
    var inverse_origin=1./depth;var inverse_slope=vec2(0.);var inverse_curvature=vec2(0.);
    var anchors=0.;var anchor_sum=0.;var anchor_weight=0.;var curvature_confidence=1.;
    for(var axis=0u;axis<2u;axis++) {
        let direction=select(vec2(1,0),vec2(0,1),axis==1u);
        let tangent=channel_axis(center,direction,size,depth);
        inverse_slope[axis]=tangent.x;inverse_curvature[axis]=tangent.z;
        anchor_sum+=tangent.y*tangent.w;anchor_weight+=tangent.w;
        anchors+=select(0.,1.,tangent.w==1.);
        curvature_confidence=min(curvature_confidence,tangent.w);
    }
    var mixed_curvature=0.;
    if curvature_confidence>0. {
        mixed_curvature=channel_cross(center,size,anchor_sum/anchor_weight,inverse_slope,inverse_curvature)*curvature_confidence;
    }
    if anchor_weight>0. {
        // Translate the quadratic and its tangent together to the actual
        // full-size pixel. Moving just the intercept leaves a different slope
        // on either side of a half-size texel phase boundary.
        let offset=p-vec2<f32>(center);
        let fitted=anchor_sum/anchor_weight+dot(inverse_slope,offset)
            +0.5*dot(inverse_curvature,offset*offset)+mixed_curvature*offset.x*offset.y;
        inverse_origin=mix(1./depth,fitted,min(anchor_weight,1.));
        inverse_slope+=inverse_curvature*offset+mixed_curvature*offset.yx;
    }
    let sheet=channel_patch(pixel);
    if sheet.confidence>0. {
        inverse_origin=mix(inverse_origin,sheet.inverse,sheet.confidence);
        inverse_slope=mix(inverse_slope,sheet.slope,sheet.confidence);
        inverse_curvature=mix(inverse_curvature,sheet.curvature.xy,sheet.confidence);
        mixed_curvature=mix(mixed_curvature,sheet.curvature.z,sheet.confidence);
    }
    // One screen texel spans much more liquid along a grazing sheet than
    // across it. Use its projective tangent metric for every final gather,
    // including normals and additive optical moments. A circular pixel filter
    // otherwise turns thin wave ridges and suspension into broad round caps.
    let step=u.viewport.xy/vec2<f32>(size);
    let origin=view_point(pixel,depth);
    let ray=origin/depth;
    let tx=view_point(pixel+vec2(step.x,0.),depth)-origin-ray*(depth*depth*inverse_slope.x);
    let ty=view_point(pixel+vec2(0.,step.y),depth)-origin-ray*(depth*depth*inverse_slope.y);
    // The image-plane footprint is the minimum antialiasing support. Do not
    // enlarge that support with the grazing-angle stretch we are correcting.
    let image_step=max(2.*abs(u.view_from_clip[0].x)*depth/f32(size.x),
                       2.*abs(u.view_from_clip[1].y)*depth/f32(size.y));
    let optical_radius=max(0.030,1.4*image_step);
    let normal_radius=max(0.045,1.4*image_step);
    // A depth residual is along camera Z. Range rejection belongs to the
    // liquid's local tangent plane; at grazing angles a centimetre of normal
    // displacement can span many camera-depth centimetres. Camera-Z rejection
    // breaks a curved sheet into isolated normal and turbidity caps.
    let tangent_normal=normalize(cross(ty,tx));
    let normal_per_depth=max(abs(dot(ray,tangent_normal)),0.0001);
    for(var y=-4;y<=4;y++){for(var x=-4;x<=4;x++){
        let q=center+vec2(x,y);
        if any(q<vec2(0)) || any(q>=size) {continue;}
        let raw=textureLoad(raw_depths,q,0);
        if raw.r<=0. || raw.g>= -0.5 {continue;}
        let distance=vec2<f32>(q)-p;
        let expected=1./max(inverse_origin+dot(inverse_slope,distance)+0.5*dot(inverse_curvature,distance*distance)+mixed_curvature*distance.x*distance.y,0.000001);
        // A profile spans 51 mm and the local filter another 35 mm. Taper
        // this visibility gate instead of switching whole muddy texels on/off.
        let reach=0.051*max(-raw.g,1.);
        let visibility=1.-smoothstep(reach,reach+0.035,abs(raw.r-expected));
        let filtered=textureLoad(depths,q,0);
        let shared_depth=filtered.r;
        if shared_depth<=0. {continue;}
        let residual=(shared_depth-expected)*normal_per_depth;
        let delta=residual/0.035;
        // A truncated Gaussian has nonzero weight at the gathered edge:
        // rounding the footprint centre then switches an entire row/column.
        // Compact C2 support reaches zero before a texel can enter or leave.
        let edge=clamp(vec2(4.5)-abs(distance),vec2(0.),vec2(1.));
        let window=edge*edge*edge*(edge*(edge*6.-vec2(15.))+vec2(10.));
        let taper=window.x*window.y*visibility;
        let tangent=tx*distance.x+ty*distance.y;
        let metric=dot(tangent,tangent);
        let normal_spatial=max(dot(distance,distance)/24.,metric/(normal_radius*normal_radius));
        let w=taper*exp(-normal_spatial-delta*delta);
        let normal=textureLoad(normal_sums,q,0);
        sum+=normal.xyz*w;weight+=normal.w*w;
        {
            // A hard 35 mm inlier cutoff switched whole normal-fit rows as a
            // crest moved. Compact C2 residual support keeps that fit continuous.
            let fit_edge=clamp(1.-abs(residual)/0.070,0.,1.);
            let fit_window=fit_edge*fit_edge*fit_edge*(fit_edge*(fit_edge*6.-15.)+10.);
            let pw=taper*fit_window*exp(-max(dot(distance,distance)/12.,normal_spatial)-delta*delta);
            let z=1./shared_depth-1./depth-0.5*dot(inverse_curvature,distance*distance)
                -mixed_curvature*distance.x*distance.y;
            position_sum+=vec3(distance,z)*pw;
            second+=vec3(distance.x*distance.x,distance.x*distance.y,distance.y*distance.y)*pw;
            cross_depth+=distance*z*pw;plane_weight+=pw;
        }
        let optical_spatial=max(dot(distance,distance)/12.5,metric/(optical_radius*optical_radius));
        let cw=taper*exp(-optical_spatial-delta*delta);
        // The same four passes already reconstructed these additive moments
        // with the interface. Keep the ratio until after the final gather.
        column+=filtered.gb*cw;column_weight+=cw;
        // The nearest kernel is an envelope, not a velocity field. Transport
        // surface detail with the joined water column, so switching the front
        // particle cannot switch the direction/phase of an entire muddy patch.
        let fw=cw*max(filtered.g,0.);
        flow_sum+=raw.ba*fw;flow_weight+=fw;
    }}
    var out:ChannelSurface;
    out.normal=vec4(0.);
    if weight>0.000001 && length(sum)>0.000001 {
        out.normal=vec4(normalize(sum),clamp(length(sum)/weight,0.,1.));
    }
    // Fit a local plane through the joined depth field. Differencing adjacent
    // half-size texels magnifies subpixel phase; fitting the whole footprint
    // preserves the travelling wave slope without each kernel's shiny dome.
    if plane_weight>0.000001 {
        let mean=position_sum/plane_weight;
        let cov=second/plane_weight-vec3(mean.x*mean.x,mean.x*mean.y,mean.y*mean.y);
        let rhs=cross_depth/plane_weight-mean.xy*mean.z;
        // A grazing footprint can be almost one-dimensional. A hard rank
        // cutoff fell back to each particle's rounded gradient. Stabilize
        // only the unresolved direction toward the independently sampled
        // projective tangent; a well-resolved fit keeps its measured slope.
        // Require both tangent anchors before stabilizing an under-resolved
        // fit. Isolated spray has no such sheet and keeps its curved normal.
        let regularizer=select(0.,max(max(cov.x,cov.z)*0.0001,0.000001),anchors==2.);
        let fitted_cov=cov+vec3(regularizer,0.,regularizer);
        let fitted_rhs=rhs+inverse_slope*regularizer;
        let determinant=fitted_cov.x*fitted_cov.z-fitted_cov.y*fitted_cov.y;
        if determinant>select(0.001,0.,anchors==2.) {
            let slope=vec2(fitted_cov.z*fitted_rhs.x-fitted_cov.y*fitted_rhs.y,
                           fitted_cov.x*fitted_rhs.y-fitted_cov.y*fitted_rhs.x)/determinant;
            let step=u.viewport.xy/vec2<f32>(size);
            // Fit reciprocal depth, then unproject its tangent. Fitting raw
            // depth curves a truly planar sheet and adds false cap highlights.
            let inverse_depth=1./depth+mean.z-dot(slope,mean.xy);
            let origin=view_point(pixel,1./max(inverse_depth,0.000001));
            let dx=view_point(pixel+vec2(step.x,0.),1./max(inverse_depth+slope.x,0.000001))-origin;
            let dy=view_point(pixel+vec2(0.,step.y),1./max(inverse_depth+slope.y,0.000001))-origin;
            var n=normalize(cross(dy,dx));if n.z<0. {n = -n;}
            var world_n=normalize((u.world_from_view*vec4(n,0.)).xyz);
            if dot(world_n,(u.world_from_view*vec4(-origin,0.)).xyz)<0. {world_n = -world_n;}
            // The fitted shared interface owns the connected-water normal.
            // Mixing a kernel gradient back in restores a dome highlight for
            // each particle, even when the reconstructed surface is smooth.
            // A front spray kernel may have no gradient where the connected
            // layer starts. Its missing gradient must not veto a valid fit.
            out.normal=vec4(world_n,select(1.,out.normal.w,out.normal.w>0.));
        }
    }
    if sheet.confidence>0. {
        let z=1./sheet.inverse;let origin=view_point(pixel,z);let ray=origin/z;
        let tx=view_point(pixel+vec2(step.x,0.),z)-origin-ray*(z*z*sheet.slope.x);
        let ty=view_point(pixel+vec2(0.,step.y),z)-origin-ray*(z*z*sheet.slope.y);
        var n=normalize(cross(ty,tx));if n.z<0. {n = -n;}
        var world_n=normalize((u.world_from_view*vec4(n,0.)).xyz);
        if dot(world_n,(u.world_from_view*vec4(-origin,0.)).xyz)<0. {world_n = -world_n;}
        if out.normal.w>0. {world_n=normalize(mix(out.normal.xyz,world_n,sheet.confidence));}
        out.normal=vec4(world_n,select(sheet.confidence,out.normal.w,out.normal.w>0.));
    }
    out.column=column/max(column_weight,0.000001);
    out.flow=flow_sum/max(flow_weight,0.000001);
    return out;
}
fn reconstructed_normal(pixel:vec2<f32>,depth:f32)->vec4<f32> {
    return channel_surface(pixel,depth).normal;
}
// Interpolate additive moments before dividing. Switching the front-most
// particle must not switch the colour of an entire overlapping water column.
fn optical_column(pixel:vec2<f32>,depth:f32,channel:bool)->vec2<f32> {
    let size=vec2<i32>(textureDimensions(fluid_thickness));
    let p=pixel*vec2<f32>(size)/u.viewport.xy-0.5;
    if channel {
        return channel_surface(pixel,depth).column;
    }
    let base=vec2<i32>(floor(p));let f=fract(p);
    var sum=vec2(0.);var weight=0.;
    for(var y=0;y<2;y++){for(var x=0;x<2;x++){
        let q=clamp(base+vec2(x,y),vec2(0),size-vec2(1));
        let raw=textureLoad(raw_depths,q,0);
        if raw.r<=0. || (raw.g< -0.5)!=channel {continue;}
        let delta=(raw.r-depth)/0.035;
        let w=select(1.-f.x,f.x,x==1)*select(1.-f.y,f.y,y==1)*exp(-delta*delta);
        sum+=textureLoad(fluid_thickness,q,0).rg*w;weight+=w;
    }}
    return sum/max(weight,0.000001);
}
fn view_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel/u.viewport.xy;
    let ray = u.view_from_clip * vec4(uv.x*2.-1.,1.-uv.y*2.,1.,1.);
    return ray.xyz * (depth / max(-ray.z,0.0001));
}
fn ripple_gradient(q:vec2<f32>,footprint:f32)->vec2<f32> {
    // Gaussian band filtering prevents narrow highlights from aliasing at a
    // grazing angle. Frequency is optical detail, never extra water volume.
    return vec2(0.8,0.6)*cos(dot(q,vec2(27.,20.)))*exp(-0.5*1130.*footprint*footprint)
        +vec2(-0.4,0.9)*cos(dot(q,vec2(-63.,142.)))*0.36*exp(-0.5*24133.*footprint*footprint)
        +vec2(0.9,-0.3)*cos(dot(q,vec2(194.,-65.)))*0.12*exp(-0.5*41861.*footprint*footprint);
}
// Millimetre-scale travelling surface detail, projected onto the actual sheet.
// This changes reflection/refraction only, never depth, volume,
// concentration or particle motion. Band filtering removes unresolved glints.
fn channel_wave(q:vec2<f32>,k:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec2<f32> {
    // Integrate each wave over its actual pixel ellipse. Using the largest
    // grazing footprint for every direction erased even resolved cross-flow
    // detail, leaving smooth inflated-looking shoulders on the riffle waves.
    let phase=vec2(dot(k,dx),dot(k,dy));
    return normalize(k)*cos(dot(q,k))*exp(-0.5*dot(phase,phase));
}
fn channel_ripple_gradient(q:vec2<f32>,dx:vec2<f32>,dy:vec2<f32>)->vec2<f32> {
    // Incommensurate bands avoid repeating parallel stripes on every crest.
    // Coarser resolved detail survives while subpixel capillary glints vanish.
    return channel_wave(q,vec2(51.,19.),dx,dy)*0.55
        +channel_wave(q,vec2(-34.,67.),dx,dy)*0.40
        +channel_wave(q,vec2(150.,33.),dx,dy)*0.65
        +channel_wave(q,vec2(77.,-112.),dx,dy)*0.35
        +channel_wave(q,vec2(291.,72.),dx,dy)*0.15;
}
fn channel_ripple_normal_footprint(normal:vec3<f32>,world:vec3<f32>,flow:vec2<f32>,time:f32,dx:vec2<f32>,dy:vec2<f32>,confidence:f32)->vec3<f32> {
    let velocity=vec3(flow.x,0.,flow.y);
    let projected=velocity-normal*dot(normal,velocity);
    let speed=length(projected);
    // A stationary sheet has no invented boiling animation. The bounded
    // fallback keeps zero-speed and near-vertical sheets finite.
    if speed<0.0001 {return normal;}
    // World-anchored phases do not rotate about the world origin when a crest
    // changes slope. Only the small gradient is projected onto its tangent.
    let uv=world.xz;
    let phase0=fract(time*0.5);let phase1=fract(time*0.5+0.5);
    let weight0=1.-abs(phase0*2.-1.);
    let gradient=mix(channel_ripple_gradient(uv-flow*(phase1*2.),dx,dy),
                     channel_ripple_gradient(uv-flow*(phase0*2.),dx,dy),weight0);
    let slope=vec3(gradient.x,0.,gradient.y);
    let tangent=slope-normal*dot(slope,normal);
    let amplitude=0.14*smoothstep(0.02,0.45,speed)*clamp(confidence,0.,1.);
    return normalize(normal-tangent*amplitude);
}
fn channel_ripple_normal(normal:vec3<f32>,world:vec3<f32>,flow:vec2<f32>,time:f32,footprint:f32,confidence:f32)->vec3<f32> {
    return channel_ripple_normal_footprint(normal,world,flow,time,vec2(footprint,0.),vec2(0.,footprint),confidence);
}
fn refracted_scene(pixel:vec2<f32>,normal:vec3<f32>,depth:f32,thickness:f32,original:vec3<f32>)->vec3<f32> {
    let position=view_point(pixel,depth);
    let normal_view=normalize((u.view_from_world*vec4(normal,0.)).xyz);
    let transmitted=refract(normalize(position),normal_view,1./1.333);
    let under=position+transmitted*(thickness/max(-transmitted.z,0.05));
    let projected=u.clip_from_world*u.world_from_view*vec4(under,1.);
    let uv=vec2(projected.x/projected.w*0.5+0.5,0.5-projected.y/projected.w*0.5);
    let offset=clamp(uv*u.viewport.xy-pixel,vec2(-10.),vec2(10.));
    // Refracted coordinates are pixel centres. Nearest-texel reads made
    // bed/grain colour jump whenever a moving wave crossed a pixel boundary.
    let texel=pixel+offset-0.5;let base=vec2<i32>(floor(texel));let f=fract(texel);
    var colour=vec3(0.);
    for(var y=0;y<2;y++){for(var x=0;x<2;x++){
        let q=clamp(base+vec2(x,y),vec2(0),vec2<i32>(u.viewport.xy)-vec2(1));
        let z=textureLoad(scene_depth,q,0);
        let xy=vec2((f32(q.x)+0.5)/u.viewport.x*2.-1.,1.-(f32(q.y)+0.5)/u.viewport.y*2.);
        let p=u.view_from_clip*vec4(xy,z,1.);
        // Validate every tap: interpolation must not drag walls/player into water.
        let behind=z>0. && -p.z/max(p.w,0.000001)>depth+0.001;
        let w=select(1.-f.x,f.x,x==1)*select(1.-f.y,f.y,y==1);
        colour+=select(original,textureLoad(scene,q,0).rgb,behind)*w;
    }}
    return colour;
}
// Both inputs are distances along the eye ray. The particle volume integral
// starts at the raw envelope; the joined interface can be farther back. A
// visible liquid interval cannot contain more water than its geometric length.
// Keep concentration when clipping: the caller scales both optical moments.
fn visible_column_path(column:f32,view_gap:f32,channel:bool)->f32 {
    return min(max(view_gap,0.),select(max(column,0.0002),max(column,0.),channel));
}
// Dielectric GGX sun reflection. The old broad, additive Phong lobe lit every
// standing wave like a plastic cap. Fresnel and masking bound reflected energy;
// measured pixel normal variance supplies only unresolved slope roughness.
fn channel_sun(normal:vec3<f32>,eye:vec3<f32>,light:vec3<f32>,variance:f32)->f32 {
    let nv=max(dot(normal,eye),0.0001);let nl=max(dot(normal,light),0.);
    if nl<=0. {return 0.;}
    let h=normalize(eye+light);let nh=max(dot(normal,h),0.);let vh=max(dot(eye,h),0.);
    let alpha2=clamp(0.0004+0.5*variance,0.0004,1.);
    let d=1.+nh*nh*(alpha2-1.);
    let distribution=alpha2/(3.14159265*d*d);
    let masking=2.*nv/(nv+sqrt(alpha2+(1.-alpha2)*nv*nv))
        *2.*nl/(nl+sqrt(alpha2+(1.-alpha2)*nl*nl));
    let fresnel=0.02037+0.97963*pow(1.-vh,5.);
    return distribution*masking*fresnel/(4.*nv)*0.6;
}
@fragment fn compose(s: Screen) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(s.position.xy); let original = textureLoad(scene,pixel,0);
    // Uniform capture-only selection; ordinary gameplay always uses mode zero.
    if u.sediment.w==1. {return original;}
    let q = s.position.xy;
    // Smoothing must not pull hidden water forward over an opaque side wall.
    // Test original particle depth as well as the reconstructed surface.
    let raw_pixel = clamp(vec2<i32>(q * vec2<f32>(textureDimensions(raw_depths)) / u.viewport.xy), vec2(0), vec2<i32>(textureDimensions(raw_depths))-vec2(1));
    var optical_sample=textureLoad(raw_depths,raw_pixel,0);
    let river_sample=textureLoad(river_depths,pixel,0);
    let river_surface=textureLoad(river_normals,pixel,0);
    let opaque=textureLoad(scene_depth,pixel,0);
    let opaque_view=u.view_from_clip*vec4(q.x/u.viewport.x*2.-1.,1.-q.y/u.viewport.y*2.,opaque,1.);
    // Reverse-Z sky has w=0; keep neighbouring derivative lanes finite.
    let opaque_depth=select(0.,-opaque_view.z/max(opaque_view.w,0.000001),opaque>0.);
    let contact=max(opaque_depth-river_sample.r,0.);
    // Derivatives must be evaluated before any per-pixel early return. Both
    // signed-film clipping and the opaque bank intersection get a pixel-wide
    // transition, rather than nearest half-resolution shoreline coverage.
    let film_width=max(0.0005,fwidth(river_surface.w));
    let contact_width=max(0.0005,fwidth(contact));
    let river=river_sample.r>0. && (optical_sample.r<=0. || river_sample.r<optical_sample.r);
    if river {optical_sample=river_sample;}
    let raw_depth=optical_sample.r;
    if raw_depth<=0. {return original;}
    let in_channel=!river && optical_sample.g< -0.5;
    var sample=vec2(raw_depth,1.);
    if !river {sample=sample_depth(q);}
    let depth=sample.x;if depth<=0. || sample.y<=0.05 {return original;}
    if river || in_channel {
        // A fixed reverse-Z epsilon grows into centimetres with camera range.
        // Use 0.1 mm in view space, with no forward bias over solid equipment.
        if opaque>0. && (depth>=opaque_depth-0.0001 || (!river && raw_depth>=opaque_depth-0.0001)) {return original;}
    } else {
        // Retain the accepted airborne-outlet occlusion path.
        let raw_clip = u.clip_from_world * u.world_from_view * vec4(view_point(q,raw_depth),1.);
        if raw_clip.z/raw_clip.w <= opaque + 0.000001 {return original;}
    }
    let position = view_point(q,depth); let world = u.world_from_view * vec4(position,1.);
    if !river && !in_channel {
        let clip = u.clip_from_world * world;
        if clip.z/clip.w <= opaque + 0.000001 {return original;}
    }
    var normal=normalize(select(textureLoad(surface_normals,raw_pixel,0).xyz,river_surface.xyz,river));
    var surface_confidence=0.;
    var channel_column=vec2(0.);
    var flow=optical_sample.ba;
    if in_channel {
        let joined=channel_surface(q,depth);
        let surface=joined.normal;
        channel_column=joined.column;
        flow=joined.flow;
        surface_confidence=surface.w;
        if surface.w>0. {
            var surface_normal=surface.xyz;if dot(surface_normal,u.world_from_view[3].xyz-world.xyz)<0. {surface_normal = -surface_normal;}
            normal=surface_normal;
        }
    }
    // A valid joined normal already owns the connected surface. Only compute
    // the four extra depth samples for detached water or a failed sheet fit.
    if !river && (!in_channel || surface_confidence<=0.) {
    let left = sample_depth(q-vec2(1.,0.)).x; let right = sample_depth(q+vec2(1.,0.)).x;
    let up = sample_depth(q-vec2(0.,1.)).x; let down = sample_depth(q+vec2(0.,1.)).x;
    var dx = view_point(q+vec2(1.,0.),select(depth,right,right>0.)) - position;
    var dy = view_point(q+vec2(0.,1.),select(depth,down,down>0.)) - position;
    if left > 0. && (right <= 0. || abs(left-depth)<abs(right-depth)) { dx = position-view_point(q-vec2(1.,0.),left); }
    if up > 0. && (down <= 0. || abs(up-depth)<abs(down-depth)) { dy = position-view_point(q-vec2(0.,1.),up); }
    var n = normalize(cross(dy,dx)); if n.z < 0. { n = -n; }
    normal = normalize((u.world_from_view * vec4(n,0.)).xyz);
    }
    // Dual-phase advection blends at resets, so a varying flow field cannot
    // stretch the normal detail indefinitely. Simulation time also freezes it.
    let eye_world=normalize(u.world_from_view[3].xyz-world.xyz);
    let pixel_size=depth*2.*abs(u.view_from_clip[1].y)/u.viewport.y;
    let footprint=pixel_size/max(abs(dot(normal,eye_world)),0.12);
    // Differentiate the continuous tangent plane, not the noisy raw envelope.
    // Each direction retains its own projected footprint at a grazing crest.
    let tangent_origin=view_point(q,depth);
    let world_dx=(u.world_from_view*vec4(view_point(q+vec2(1.,0.),depth)-tangent_origin,0.)).xyz;
    let world_dy=(u.world_from_view*vec4(view_point(q+vec2(0.,1.),depth)-tangent_origin,0.)).xyz;
    let view_ray=world.xyz-u.world_from_view[3].xyz;
    let tangent_denom=dot(normal,view_ray);
    let tangent_safe=abs(tangent_denom)>0.0001;
    let dx=world_dx-view_ray*(dot(normal,world_dx)/select(1.,tangent_denom,tangent_safe));
    let dy=world_dy-view_ray*(dot(normal,world_dy)/select(1.,tangent_denom,tangent_safe));
    if in_channel {
        normal=channel_ripple_normal_footprint(normal,world.xyz,flow,u.sediment.z,
            dx.xz,dy.xz,surface_confidence*select(0.,1.,tangent_safe));
    } else if river {
        let phase0=fract(u.sediment.z*0.5);let phase1=fract(u.sediment.z*0.5+0.5);
        let weight0=1.-abs(phase0*2.-1.);
        let q0=world.xz-flow*(phase0*2.);let q1=world.xz-flow*(phase1*2.);
        let gradient=mix(ripple_gradient(q1,footprint),ripple_gradient(q0,footprint),weight0);
        normal=normalize(normal+vec3(gradient.x,0.,gradient.y)*select(0.065*surface_confidence,0.055,river));
    }
    let fresnel=0.02+0.98*pow(1.-clamp(dot(normal,eye_world),0.,1.),5.);
    // Match the scene sun (from -20,35,-15 toward 17,0,17). The
    // disconnected channel highlight used to face the opposite river bank.
    let light=normalize(select(vec3(0.35,0.85,0.3),vec3(-37.,35.,-32.),in_channel));
    let half_vector=normalize(light+eye_world);
    // Broaden only unresolved variation measured across screen pixels. A
    // fixed broad channel lobe painted a soft bright cap on every resolved
    // standing wave. Normal confidence is not surface roughness.
    let nx=dpdx(normal);let ny=dpdy(normal);
    let variance=max(dot(nx,nx),dot(ny,ny));
    var specular=pow(max(dot(normal,half_vector),0.),96.)*0.32;
    if in_channel {specular=channel_sun(normal,eye_world,light,variance);}
    let reflection=reflect(-eye_world,normal);
    let sky=mix(vec3(0.19,0.25,0.24),vec3(0.48,0.66,0.76),smoothstep(-0.1,0.65,reflection.y));

    // Optical columns are integrated in world metres along the eye ray, not
    // camera-Z metres. Mixing those units made oblique crests alternate
    // between background and muddy scattering as the floor clip took over.
    let ray_cosine=select(1.,max(-normalize(position).z,0.0001),in_channel);
    var thickness = 0.04;
    if opaque>0. && abs(opaque_view.w)>0.00001 { thickness=clamp((-opaque_view.z/opaque_view.w-depth)/ray_cosine,0.,0.5); }
    var column=vec2(0.);
    if !river {
        if in_channel {column=channel_column;} else {column=optical_column(q,raw_depth,false);}
        let visible=visible_column_path(column.x,thickness,in_channel);
        // Clip water and mineral optical depth together. Never divide by the
        // shortened path while retaining all of its old sediment: that creates
        // dense tan patches where reconstructed crests meet shallow geometry.
        if in_channel {column*=visible/max(column.x,0.000001);}
        thickness=visible;
    }
    let optical=optical_sample.g;
    // River keeps its existing concentration convention. Channel extinction
    // comes from local represented fine-grain cross section per water volume.
    let beta=select(column.y/max(column.x,0.000001),max(optical-1.,0.)*2400.,river);
    // Particle water uses reduced scattering for unresolved forward-scattered
    // light, rather than treating every grain cross section as absorption.
    // g=.85 and albedo=.95 are presentation approximations, not measured soil
    // coefficients; concentration and represented sediment remain unchanged.
    let transport=select((1.-0.95)+0.95*(1.-0.85),1.,river);
    let extinction=vec3(0.8,0.24,0.12)+vec3(0.5625,1.,1.5)*beta*transport;
    let transmission=exp(-extinction*thickness);
    var scattering=mix(vec3(0.035,0.18,0.18),vec3(0.22,0.13,0.055),clamp(beta/120.,0.,1.));
    if in_channel {
        // Mix water and mineral source terms by their extinction, rather than
        // applying a second arbitrary concentration-to-colour threshold. The
        // Beer-Lambert column already controls how much suspended dirt is seen.
        let water_extinction=vec3(0.8,0.24,0.12);
        let mineral_extinction=vec3(0.5625,1.,1.5)*beta*transport;
        scattering=(water_extinction*vec3(0.035,0.18,0.18)
            +mineral_extinction*vec3(0.22,0.16,0.095))/extinction;
    }
    if in_channel {
        // Suspended particles scatter light throughout the water column. A
        // Lambert term from each reconstructed surface bump paints false mud
        // patches onto that volume. Use the diffuse environment irradiance;
        // reflection, refraction and the sun still follow the actual surface.
        scattering*=0.8;
    } else if !river {
        scattering*=0.35+0.65*max(dot(normal,light),0.);
    }
    var background=original.rgb;
    if river || in_channel {background=refracted_scene(q,normal,depth,thickness*ray_cosine,background);}
    let body=background*transmission+scattering*(vec3(1.)-transmission);
    // A uniform 0.5 mm film stays fully covered. At a sloping bank use the
    // full-resolution contact and film gradients to soften only the last pixel.
    let shore=smoothstep(0.0001,film_width,river_surface.w)
        *select(1.,smoothstep(0.0001,contact_width,contact),opaque>0.);
    let coverage=clamp(sample.y,0.,1.)*select(1.,shore,river);
    if u.sediment.w==2. {return vec4(mix(original.rgb,normal*0.5+vec3(0.5),coverage),original.a);}
    if u.sediment.w==3. {return vec4(mix(original.rgb,vec3(clamp(thickness*2.,0.,1.)),coverage),original.a);}
    if u.sediment.w==4. {return vec4(mix(original.rgb,vec3(clamp(beta/1200.,0.,1.)),coverage),original.a);}
    return vec4(mix(original.rgb,mix(body,sky,fresnel)+vec3(specular),coverage),original.a);
}
