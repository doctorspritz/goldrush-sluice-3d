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
struct ChannelSurface {normal:vec4<f32>,column:vec2<f32>};
// Gather every texel in the footprint. Striding over alternating texels makes
// the surface and its turbidity change when the half-size pixel phase changes.
// Share visibility tests between the two additive fields, before normalizing.
fn channel_surface(pixel:vec2<f32>,depth:f32)->ChannelSurface {
    var sum=vec3(0.);var weight=0.;
    var position_sum=vec3(0.);var second=vec3(0.);var cross_depth=vec2(0.);var plane_weight=0.;
    var column=vec2(0.);var column_weight=0.;
    let size=vec2<i32>(textureDimensions(surface_normals));
    let p=pixel*vec2<f32>(size)/u.viewport.xy-0.5;
    let center=vec2<i32>(round(p));
    let pixel_m=depth*4.*abs(u.view_from_clip[1].y)/u.viewport.y;
    let radius=clamp(0.030/max(pixel_m,0.00001),2.5,5.);
    for(var y=-4;y<=4;y++){for(var x=-4;x<=4;x++){
        let q=clamp(center+vec2(x,y),vec2(0),size-vec2(1));
        let raw=textureLoad(raw_depths,q,0);
        if raw.r<=0. || raw.g>= -0.5 {continue;}
        let delta=(raw.r-depth)/0.035;
        let distance=vec2<f32>(q)-p;
        let w=exp(-dot(distance,distance)/24.-delta*delta);
        let normal=textureLoad(normal_sums,q,0);
        sum+=normal.xyz*w;weight+=normal.w*w;
        let shared_depth=textureLoad(depths,q,0).r;
        if shared_depth>0. && abs(shared_depth-depth)<0.035 {
            let pw=exp(-dot(distance,distance)/12.-delta*delta);
            let z=shared_depth-depth;
            position_sum+=vec3(distance,z)*pw;
            second+=vec3(distance.x*distance.x,distance.x*distance.y,distance.y*distance.y)*pw;
            cross_depth+=distance*z*pw;plane_weight+=pw;
        }
        let cw=exp(-2.*dot(distance,distance)/(radius*radius)-delta*delta);
        column+=textureLoad(fluid_thickness,q,0).rg*cw;column_weight+=cw;
    }}
    var out:ChannelSurface;
    out.normal=vec4(0.);
    if weight>0.000001 && length(sum)>0.000001 {
        out.normal=vec4(normalize(sum),clamp(length(sum)/weight,0.,1.));
    }
    // Fit a local plane through the joined depth field. Differencing adjacent
    // half-size texels magnifies subpixel phase; fitting the whole footprint
    // preserves the travelling wave slope without each kernel's shiny dome.
    if plane_weight>0.000001 && out.normal.w>0. {
        let mean=position_sum/plane_weight;
        let cov=second/plane_weight-vec3(mean.x*mean.x,mean.x*mean.y,mean.y*mean.y);
        let rhs=cross_depth/plane_weight-mean.xy*mean.z;
        let determinant=cov.x*cov.z-cov.y*cov.y;
        if determinant>0.001 {
            let slope=vec2(cov.z*rhs.x-cov.y*rhs.y,cov.x*rhs.y-cov.y*rhs.x)/determinant;
            let step=u.viewport.xy/vec2<f32>(size);
            let origin=view_point(pixel,depth+mean.z);
            let dx=view_point(pixel+vec2(step.x,0.),depth+mean.z+slope.x)-origin;
            let dy=view_point(pixel+vec2(0.,step.y),depth+mean.z+slope.y)-origin;
            var n=normalize(cross(dy,dx));if n.z<0. {n = -n;}
            var world_n=normalize((u.world_from_view*vec4(n,0.)).xyz);
            if dot(world_n,out.normal.xyz)<0. {world_n = -world_n;}
            // The fitted shared interface owns the connected-water normal.
            // Mixing a kernel gradient back in restores a dome highlight for
            // each particle, even when the reconstructed surface is smooth.
            out.normal=vec4(world_n,out.normal.w);
        }
    }
    out.column=column/max(column_weight,0.000001);
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
fn refracted_scene(pixel:vec2<f32>,normal:vec3<f32>,depth:f32,thickness:f32,original:vec3<f32>)->vec3<f32> {
    let position=view_point(pixel,depth);
    let normal_view=normalize((u.view_from_world*vec4(normal,0.)).xyz);
    let transmitted=refract(normalize(position),normal_view,1./1.333);
    let under=position+transmitted*(thickness/max(-transmitted.z,0.05));
    let projected=u.clip_from_world*u.world_from_view*vec4(under,1.);
    let uv=vec2(projected.x/projected.w*0.5+0.5,0.5-projected.y/projected.w*0.5);
    let offset=clamp(uv*u.viewport.xy-pixel,vec2(-10.),vec2(10.));
    let q=clamp(vec2<i32>(pixel+offset),vec2(0),vec2<i32>(u.viewport.xy)-vec2(1));
    let z=textureLoad(scene_depth,q,0);
    let xy=vec2((f32(q.x)+0.5)/u.viewport.x*2.-1.,1.-(f32(q.y)+0.5)/u.viewport.y*2.);
    let p=u.view_from_clip*vec4(xy,z,1.);
    // Refraction cannot pull an opaque side wall/player in front of the water.
    if z>0. && -p.z/p.w>depth+0.001 {return textureLoad(scene,q,0).rgb;}
    return original;
}
@fragment fn compose(s: Screen) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(s.position.xy); let original = textureLoad(scene,pixel,0);
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
    if !river {
    let left = sample_depth(q-vec2(1.,0.)).x; let right = sample_depth(q+vec2(1.,0.)).x;
    let up = sample_depth(q-vec2(0.,1.)).x; let down = sample_depth(q+vec2(0.,1.)).x;
    var dx = view_point(q+vec2(1.,0.),select(depth,right,right>0.)) - position;
    var dy = view_point(q+vec2(0.,1.),select(depth,down,down>0.)) - position;
    if left > 0. && (right <= 0. || abs(left-depth)<abs(right-depth)) { dx = position-view_point(q-vec2(1.,0.),left); }
    if up > 0. && (down <= 0. || abs(up-depth)<abs(down-depth)) { dy = position-view_point(q-vec2(0.,1.),up); }
    var n = normalize(cross(dy,dx)); if n.z < 0. { n = -n; }
    normal = normalize((u.world_from_view * vec4(n,0.)).xyz);
    }
    var surface_confidence=0.;
    var channel_column=vec2(0.);
    if in_channel {
        let joined=channel_surface(q,raw_depth);
        let surface=joined.normal;
        channel_column=joined.column;
        surface_confidence=surface.w;
        if surface.w>0. {
            var surface_normal=surface.xyz;if dot(surface_normal,u.world_from_view[3].xyz-world.xyz)<0. {surface_normal = -surface_normal;}
            normal=surface_normal;
        }
    }
    let flow=optical_sample.ba;
    // Dual-phase advection blends at resets, so a varying flow field cannot
    // stretch the normal detail indefinitely. Simulation time also freezes it.
    let eye_world=normalize(u.world_from_view[3].xyz-world.xyz);
    let pixel_size=depth*2.*abs(u.view_from_clip[1].y)/u.viewport.y;
    let footprint=pixel_size/max(abs(dot(normal,eye_world)),0.12);
    if river || in_channel {
        let phase0=fract(u.sediment.z*0.5);let phase1=fract(u.sediment.z*0.5+0.5);
        let weight0=1.-abs(phase0*2.-1.);
        let q0=world.xz-flow*(phase0*2.);let q1=world.xz-flow*(phase1*2.);
        let gradient=mix(ripple_gradient(q1,footprint),ripple_gradient(q0,footprint),weight0);
        normal=normalize(normal+vec3(gradient.x,0.,gradient.y)*select(0.065*surface_confidence,0.055,river));
    }
    let fresnel=0.02+0.98*pow(1.-clamp(dot(normal,eye_world),0.,1.),5.);
    let light=normalize(vec3(0.35,0.85,0.3));
    let half_vector=normalize(light+eye_world);
    // Unresolved normal variation broadens the sun lobe instead of producing
    // a bright bead for each kernel. Scale its peak to retain the lobe energy.
    // Confidence measures agreement of neighbouring gradients, not optical
    // roughness. A single symmetric kernel can have perfect confidence, so
    // using it to sharpen the sun creates a bright round cap on every crest.
    // Keep a broad unresolved-wave lobe throughout connected channel water.
    let exponent=select(96.,24.,in_channel);
    let specular=pow(max(dot(normal,half_vector),0.),exponent)*0.32*(exponent+2.)/98.;
    let reflection=reflect(-eye_world,normal);
    let sky=mix(vec3(0.19,0.25,0.24),vec3(0.48,0.66,0.76),smoothstep(-0.1,0.65,reflection.y));

    var thickness = 0.04;
    if opaque>0. && abs(opaque_view.w)>0.00001 { thickness=clamp(-opaque_view.z/opaque_view.w-depth,0.,0.5); }
    var column=vec2(0.);
    if !river {
        if in_channel {column=channel_column;} else {column=optical_column(q,raw_depth,false);}
        thickness=min(thickness,max(column.x,0.0002));
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
    if river || in_channel {background=refracted_scene(q,normal,depth,thickness,background);}
    let body=background*transmission+scattering*(vec3(1.)-transmission);
    // A uniform 0.5 mm film stays fully covered. At a sloping bank use the
    // full-resolution contact and film gradients to soften only the last pixel.
    let shore=smoothstep(0.0001,film_width,river_surface.w)
        *select(1.,smoothstep(0.0001,contact_width,contact),opaque>0.);
    let coverage=clamp(sample.y,0.,1.)*select(1.,shore,river);
    return vec4(mix(original.rgb,mix(body,sky,fresnel)+vec3(specular),coverage),original.a);
}
