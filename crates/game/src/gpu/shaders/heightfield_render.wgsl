struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    cell_size: f32,
    grid_width: u32,
    grid_depth: u32,
    time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> bedrock: array<f32>;
@group(0) @binding(2) var<storage, read> paydirt: array<f32>;
@group(0) @binding(3) var<storage, read> gravel: array<f32>;
@group(0) @binding(4) var<storage, read> overburden: array<f32>;
@group(0) @binding(5) var<storage, read> sediment: array<f32>;

@group(0) @binding(6) var<storage, read> water_surface_buf: array<f32>;
@group(0) @binding(7) var<storage, read> water_depth_buf: array<f32>;

struct VertexInput {
    @location(0) grid_pos: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
}

struct WaterVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) depth: f32, // water depth
    @location(2) normal: vec3<f32>,
}

fn get_height(x: u32, z: u32) -> f32 {
    let idx = z * uniforms.grid_width + x;
    return bedrock[idx] + paydirt[idx] + gravel[idx] + overburden[idx] + sediment[idx];
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let gx = u32(in.grid_pos.x);
    let gz = u32(in.grid_pos.y);
    let idx = gz * uniforms.grid_width + gx;
    
    // Sample layers
    let b = bedrock[idx];
    let p = paydirt[idx];
    let g = gravel[idx];
    let o = overburden[idx];
    let s = sediment[idx];
    let h = b + p + g + o + s;
    
    // World position
    let world_pos = vec3<f32>(
        in.grid_pos.x * uniforms.cell_size,
        h,
        in.grid_pos.y * uniforms.cell_size
    );
    
    // Compute normal
    let hL = select(h, get_height(gx - 1u, gz), gx > 0u);
    let hR = select(h, get_height(gx + 1u, gz), gx < uniforms.grid_width - 1u);
    let hD = select(h, get_height(gx, gz - 1u), gz > 0u);
    let hU = select(h, get_height(gx, gz + 1u), gz < uniforms.grid_depth - 1u);
    
    // dx = (2*cell, hR-hL, 0)
    // dz = (0, hU-hD, 2*cell)
    // cross(dz, dx) -> normal = (hL-hR, 2.0*cell, hD-hU)
    let normal = normalize(vec3<f32>(hL - hR, 2.0 * uniforms.cell_size, hD - hU));
    
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.normal = normal;
    
    // Determine base color based on geology layers
    let c_bedrock = vec3<f32>(0.3, 0.3, 0.35);    // Dark Grey
    let c_paydirt = vec3<f32>(0.6, 0.5, 0.2);     // Gold/Brown
    let c_gravel = vec3<f32>(0.5, 0.5, 0.5);      // Grey
    let c_overburden = vec3<f32>(0.4, 0.3, 0.2);  // Brown
    let c_sediment = vec3<f32>(0.7, 0.6, 0.4);    // Light Brown/Sand
    
    // Smoothly blend colors based on layer thicknesses
    // Threshold (0.005 to 0.03) ensures colors don't pop instantly
    var color = c_bedrock;
    color = mix(color, c_paydirt, smoothstep(0.005, 0.03, p));
    color = mix(color, c_gravel, smoothstep(0.005, 0.03, g));
    color = mix(color, c_overburden, smoothstep(0.005, 0.03, o));
    color = mix(color, c_sediment, smoothstep(0.005, 0.03, s));
    
    out.base_color = color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.base_color;
    
    // Lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    let ambient = 0.3;
    let lighting = ambient + diffuse * 0.7;
    
    // Side darkening
    let slope = 1.0 - in.normal.y;
    let slope_darken = 1.0 - slope * 0.5;
    
    return vec4<f32>(color * lighting * slope_darken, 1.0);
}

fn get_water_surface(x: u32, z: u32) -> f32 {
    let idx = z * uniforms.grid_width + x;
    return water_surface_buf[idx];
}

fn get_water_depth(x: u32, z: u32) -> f32 {
    let idx = z * uniforms.grid_width + x;
    return water_depth_buf[idx];
}

@vertex
fn vs_water(in: VertexInput) -> WaterVertexOutput {
    var out: WaterVertexOutput;
    
    let gx = u32(in.grid_pos.x);
    let gz = u32(in.grid_pos.y);
    let idx = gz * uniforms.grid_width + gx;
    
    let ground = get_height(gx, gz);
    let surface = get_water_surface(gx, gz);
    let depth = surface - ground;
    
    // Determine render height
    var h = surface;
    var render_depth = depth;
    
    // Step 1: Stability Snapping (Shelf Logic)
    // We snap the current vertex to neighbor water levels if we are shallow or dry.
    // This bridges the "jagged" grid edge to a clean intersection with terrain.
    // We use a larger threshold (0.05) for snapping than for existence (0.001) to provide stability.
    if (depth <= 0.05) {
        var snap_h = h;
        var found_neighbor = false;
        
        // Find highest wet neighbor
        if (gx > 0u) {
            let dL = get_water_depth(gx - 1u, gz);
            if (dL > 0.001) { snap_h = max(snap_h, get_water_surface(gx - 1u, gz)); found_neighbor = true; }
        }
        if (gx < uniforms.grid_width - 1u) {
            let dR = get_water_depth(gx + 1u, gz);
            if (dR > 0.001) { snap_h = max(snap_h, get_water_surface(gx + 1u, gz)); found_neighbor = true; }
        }
        if (gz > 0u) {
            let dD = get_water_depth(gx, gz - 1u);
            if (dD > 0.001) { snap_h = max(snap_h, get_water_surface(gx, gz - 1u)); found_neighbor = true; }
        }
        if (gz < uniforms.grid_depth - 1u) {
            let dU = get_water_depth(gx, gz + 1u);
            if (dU > 0.001) { snap_h = max(snap_h, get_water_surface(gx, gz + 1u)); found_neighbor = true; }
        }
        
        // If neighbor is significantly higher, snap to its surface
        if (found_neighbor && snap_h > h + 0.005) {
             h = snap_h;
             render_depth = h - ground;
        }
    }
    
    // Neighbors for normal calculation
    let hL_g = get_height(gx - 1u, gz);
    let hR_g = get_height(gx + 1u, gz);
    let hD_g = get_height(gx, gz - 1u);
    let hU_g = get_height(gx, gz + 1u);
    
    // Treat dry neighbor surface as CURRENT height to maintain flat surface normal at edges
    var hL = get_water_surface(gx - 1u, gz);
    if (hL < hL_g + 0.001) { hL = h; }
    if (gx == 0u) { hL = h; }

    var hR = get_water_surface(gx + 1u, gz);
    if (hR < hR_g + 0.001) { hR = h; }
    if (gx == uniforms.grid_width - 1u) { hR = h; }

    var hD = get_water_surface(gx, gz - 1u);
    if (hD < hD_g + 0.001) { hD = h; }
    if (gz == 0u) { hD = h; }

    var hU = get_water_surface(gx, gz + 1u);
    if (hU < hU_g + 0.001) { hU = h; }
    if (gz == uniforms.grid_depth - 1u) { hU = h; }

    let normal = normalize(vec3<f32>(hL - hR, 2.0 * uniforms.cell_size, hD - hU));
    
    let world_pos = vec3<f32>(
        in.grid_pos.x * uniforms.cell_size,
        h + 0.002, // Tiny lift to prevent Z-fighting artifacts with terrain ground
        in.grid_pos.y * uniforms.cell_size
    );
    
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.depth = render_depth;
    out.normal = normal;
    
    return out;
}

@fragment
fn fs_water(in: WaterVertexOutput) -> @location(0) vec4<f32> {
    if (in.depth <= 0.001) {
        discard;
    }
    
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);
    
    // Ripples
    let t = uniforms.time;
    let p = in.world_pos.xz;
    // Simple composite wave
    let w1 = sin(p.x * 2.0 + t * 1.5) * 0.5 + sin(p.y * 1.7 + t * 1.5) * 0.5;
    let w2 = sin(p.x * 4.3 - t * 2.3) * 0.3 + sin(p.y * 3.8 + t * 2.1) * 0.3;
    let wave_h = w1 + w2;
    // Approximated derivative/normal perturbation
    let n_perturb = vec3<f32>(
        (cos(p.x * 2.0 + t * 1.5) + cos(p.x * 4.3 - t * 2.3) * 1.5) * 0.1,
        1.0, 
        (cos(p.y * 1.7 + t * 1.5) + cos(p.y * 3.8 + t * 2.1) * 1.5) * 0.1
    );
    
    let base_normal = normalize(in.normal);
    // Combine base normal (sloped due to terrain flow) with ripples
    // We assume base_normal is approx up, so just add perturbation
    let normal = normalize(base_normal + n_perturb * 0.3);

    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // Fresnel
    let NdotV = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - NdotV, 3.0);
    
    // Specular (Roughness increased for visibility)
    let half_dir = normalize(light_dir + view_dir);
    let NdotH = max(dot(normal, half_dir), 0.0);
    let specular = pow(NdotH, 60.0) * 0.8; 
    
    // Base color
    let deep_color = vec3<f32>(0.1, 0.2, 0.5);
    let shallow_color = vec3<f32>(0.3, 0.5, 0.8);
    let foam_color = vec3<f32>(0.9, 0.95, 1.0);
    
    let water_base = mix(shallow_color, deep_color, clamp(in.depth * 0.5, 0.0, 1.0));
    
    // Foam at edges (depth < 0.05)
    let p_noise = in.world_pos.xz * 10.0;
    let foam_noise = sin(p_noise.x + t * 0.5) * cos(p_noise.y - t * 0.3) * 0.5 + 0.5;
    
    let foam_reach = 0.05;
    var foam_factor = 1.0 - smoothstep(0.0, foam_reach, in.depth);
    foam_factor = foam_factor * (0.5 + 0.5 * foam_noise); // Soften and break up the line
    
    let water_color = mix(water_base, foam_color, foam_factor * 0.3); // Dial down foam intensity
    
    // Opacity: Smooth fade-in at shorelines
    // alpha_fade goes from 0.0 at depth 0.0 to 1.0 at depth 0.1
    let alpha_fade = smoothstep(0.0, 0.1, in.depth);
    let alpha_base = alpha_fade * clamp(in.depth * 0.5 + 0.3, 0.0, 0.85);
    let alpha = max(alpha_base, foam_factor * 0.4); // Foam is less opaque
    
    // Combine
    let sky_color = vec3<f32>(0.8, 0.9, 1.0);
    let final_color = mix(water_color, sky_color, fresnel * 0.4) + specular;
    
    return vec4<f32>(final_color, alpha + fresnel * 0.2 + specular * 0.5);
}
