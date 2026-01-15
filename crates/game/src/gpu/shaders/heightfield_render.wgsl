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
@group(0) @binding(8) var<storage, read> surface_material_buf: array<u32>; // 0=bed,1=pay,2=gravel,3=over,4=sed
@group(0) @binding(9) var<storage, read> suspended_sediment_buf: array<f32>; // kg/m² suspended in water
@group(0) @binding(10) var<storage, read> suspended_overburden_buf: array<f32>;
@group(0) @binding(11) var<storage, read> suspended_gravel_buf: array<f32>;
@group(0) @binding(12) var<storage, read> suspended_paydirt_buf: array<f32>;
@group(0) @binding(13) var<storage, read> water_velocity_x_buf: array<f32>;
@group(0) @binding(14) var<storage, read> water_velocity_z_buf: array<f32>;

struct VertexInput {
    @location(0) grid_pos: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct WaterVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) depth: f32, // water depth
    @location(2) normal: vec3<f32>,
    @location(3) sediment_load: f32, // suspended sediment concentration
    @location(4) velocity_mag: f32, // water velocity magnitude (m/s)
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

    return out;
}

// Simple hash function for procedural noise
fn hash(p: vec3<f32>) -> f32 {
    let p3 = fract(p * 0.1031);
    let p3d = p3 + vec3<f32>(dot(p3, p3.yzx + 33.33));
    return fract((p3d.x + p3d.y) * p3d.z);
}

// Value noise with smooth interpolation
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);

    // 8 corners of the cube
    let n000 = hash(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash(i + vec3<f32>(1.0, 1.0, 1.0));

    // Trilinear interpolation
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
    );
}

// Fractal Brownian Motion for multi-scale detail
fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise3d(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Voronoi noise for rocky/cracked patterns
fn voronoi(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    var min_dist = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let point = hash(vec3<f32>(i + neighbor, 0.0));
            let diff = neighbor + point - f;
            let dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }
    return min_dist;
}

// ============ PROCEDURAL MATERIAL TEXTURES ============

// BEDROCK: Hard, cracked rock with dark veins
fn texture_bedrock(p: vec3<f32>, detail: f32) -> vec3<f32> {
    let base_color = vec3<f32>(0.28, 0.28, 0.32);
    let vein_color = vec3<f32>(0.15, 0.15, 0.18);
    let highlight_color = vec3<f32>(0.38, 0.38, 0.42);

    // Large-scale rock structure
    let rock_noise = fbm(p * 3.0, 3);

    // Crack pattern using voronoi
    let cracks = voronoi(p.xz * 8.0);
    let crack_lines = smoothstep(0.0, 0.08, cracks);

    // Fine grain detail
    let fine_detail = noise3d(p * 40.0) * 0.15 * detail;

    // Combine
    var color = mix(vein_color, base_color, crack_lines);
    color = mix(color, highlight_color, rock_noise * 0.3);
    color = color + fine_detail;

    return color;
}

// PAYDIRT: Gold-bearing gravel with visible stones and gold flecks
fn texture_paydirt(p: vec3<f32>, detail: f32) -> vec3<f32> {
    let base_color = vec3<f32>(0.55, 0.45, 0.22);
    let stone_color = vec3<f32>(0.45, 0.4, 0.3);
    let gold_color = vec3<f32>(0.85, 0.7, 0.2);

    // Gravel/stone pattern
    let stones = voronoi(p.xz * 15.0);
    let stone_mask = smoothstep(0.1, 0.25, stones);

    // Variation in color
    let color_var = fbm(p * 5.0, 2) * 0.2;

    // Gold flecks (rare, bright spots)
    let gold_noise = noise3d(p * 50.0);
    let gold_flecks = smoothstep(0.85, 0.9, gold_noise) * detail;

    // Fine sandy detail
    let fine_detail = noise3d(p * 30.0) * 0.1 * detail;

    var color = mix(stone_color, base_color, stone_mask);
    color = color + color_var;
    color = mix(color, gold_color, gold_flecks * 0.8);
    color = color + fine_detail;

    return color;
}

// GRAVEL: Grey stones and pebbles
fn texture_gravel(p: vec3<f32>, detail: f32) -> vec3<f32> {
    let base_color = vec3<f32>(0.5, 0.48, 0.45);
    let dark_stone = vec3<f32>(0.35, 0.33, 0.3);
    let light_stone = vec3<f32>(0.6, 0.58, 0.55);

    // Pebble pattern (smaller voronoi cells)
    let pebbles = voronoi(p.xz * 20.0);
    let pebble_shade = smoothstep(0.0, 0.3, pebbles);

    // Color variation between stones
    let stone_id = floor(p.xz * 20.0);
    let stone_color_var = hash(vec3<f32>(stone_id, 0.0));

    // Fine grain
    let fine = noise3d(p * 35.0) * 0.08 * detail;

    var color = mix(dark_stone, light_stone, stone_color_var);
    color = color * (0.7 + pebble_shade * 0.3);
    color = color + fine;

    return color;
}

// OVERBURDEN: Earthy dirt with organic matter
fn texture_overburden(p: vec3<f32>, detail: f32) -> vec3<f32> {
    let base_color = vec3<f32>(0.4, 0.32, 0.22);
    let dark_dirt = vec3<f32>(0.3, 0.22, 0.15);
    let light_dirt = vec3<f32>(0.5, 0.4, 0.28);

    // Chunky dirt pattern
    let chunks = fbm(p * 8.0, 3);

    // Organic matter streaks
    let organic = noise3d(p * 12.0 + vec3<f32>(0.0, 100.0, 0.0));
    let organic_mask = smoothstep(0.4, 0.6, organic) * 0.15;

    // Fine grain detail
    let fine = noise3d(p * 25.0) * 0.1 * detail;

    var color = mix(dark_dirt, light_dirt, chunks);
    color = color - organic_mask; // Darker organic spots
    color = color + fine;

    return color;
}

// SEDIMENT: Fine sandy deposits
fn texture_sediment(p: vec3<f32>, detail: f32) -> vec3<f32> {
    let base_color = vec3<f32>(0.7, 0.6, 0.4);
    let wet_color = vec3<f32>(0.55, 0.45, 0.3);
    let dry_color = vec3<f32>(0.8, 0.72, 0.5);

    // Ripple patterns (from water flow)
    let ripple_x = sin(p.x * 30.0 + p.z * 5.0) * 0.5 + 0.5;
    let ripple_z = sin(p.z * 25.0 + p.x * 3.0) * 0.5 + 0.5;
    let ripples = (ripple_x * ripple_z) * 0.15 * detail;

    // Moisture variation
    let moisture = fbm(p * 4.0, 2);

    // Very fine grain
    let fine = noise3d(p * 60.0) * 0.05 * detail;

    var color = mix(wet_color, dry_color, moisture * 0.5 + 0.3);
    color = color + ripples;
    color = color + fine;

    return color;
}

// MIXED/TRANSITION: Muddy mix for material boundaries
// Uses a unified noise pattern that doesn't oscillate
fn texture_mixed(p: vec3<f32>, detail: f32, base_colors: array<vec3<f32>, 5>, weights: array<f32, 5>) -> vec3<f32> {
    // Compute weighted average base color (no texture patterns)
    var avg_color = vec3<f32>(0.0);
    for (var i = 0; i < 5; i++) {
        avg_color = avg_color + base_colors[i] * weights[i];
    }

    // Single unified noise pattern for mixed areas
    let mix_noise = fbm(p * 6.0, 3);
    let fine_noise = noise3d(p * 20.0) * 0.08 * detail;

    // Muddy/disturbed appearance
    let dark_mud = avg_color * 0.7;
    let light_mud = avg_color * 1.1;

    var color = mix(dark_mud, light_mud, mix_noise);

    // Add some clumpy variation
    let clumps = voronoi(p.xz * 12.0);
    let clump_shade = smoothstep(0.0, 0.2, clumps) * 0.15;
    color = color + clump_shade;

    color = color + fine_noise;

    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance from camera for LOD
    let dist_to_camera = length(uniforms.camera_pos - in.world_pos);

    // Detail level fades with distance (1.0 close, 0.0 far)
    let detail_level = smoothstep(40.0, 8.0, dist_to_camera);

    // Sample layer thicknesses directly from buffers using world position
    // This gives stable per-pixel values without vertex interpolation artifacts
    let wx = in.world_pos.x;
    let wz = in.world_pos.z;

    // Convert world pos to grid coordinates
    let gx = wx / uniforms.cell_size;
    let gz = wz / uniforms.cell_size;

    // Get integer grid coords and fractions for bilinear interpolation
    let x0 = u32(max(floor(gx), 0.0));
    let z0 = u32(max(floor(gz), 0.0));
    let x1 = min(x0 + 1u, uniforms.grid_width - 1u);
    let z1 = min(z0 + 1u, uniforms.grid_depth - 1u);

    let fx = fract(gx);
    let fz = fract(gz);

    // Compute indices for 4 corners
    let i00 = z0 * uniforms.grid_width + x0;
    let i10 = z0 * uniforms.grid_width + x1;
    let i01 = z1 * uniforms.grid_width + x0;
    let i11 = z1 * uniforms.grid_width + x1;

    // Sample and interpolate each layer
    let t_paydirt = mix(mix(paydirt[i00], paydirt[i10], fx), mix(paydirt[i01], paydirt[i11], fx), fz);
    let t_gravel = mix(mix(gravel[i00], gravel[i10], fx), mix(gravel[i01], gravel[i11], fx), fz);
    let t_overburden = mix(mix(overburden[i00], overburden[i10], fx), mix(overburden[i01], overburden[i11], fx), fz);
    let t_sediment = mix(mix(sediment[i00], sediment[i10], fx), mix(sediment[i01], sediment[i11], fx), fz);

    // Base colors for each material - VERY DISTINCT for visual clarity
    let c_bedrock = vec3<f32>(0.25, 0.25, 0.30);    // Dark grey rock
    let c_paydirt = vec3<f32>(0.70, 0.55, 0.20);    // Gold-brown (bright)
    let c_gravel = vec3<f32>(0.50, 0.50, 0.55);     // Blue-grey stone
    let c_overburden = vec3<f32>(0.40, 0.28, 0.15); // Rich brown dirt
    let c_sediment = vec3<f32>(0.90, 0.80, 0.55);   // Bright tan/sand

    // SURFACE MATERIAL tells us what was deposited on top
    // Sample from the nearest grid cell (no interpolation for discrete material type)
    let gxi = u32(clamp(gx + 0.5, 0.0, f32(uniforms.grid_width - 1u)));
    let gzi = u32(clamp(gz + 0.5, 0.0, f32(uniforms.grid_depth - 1u)));
    let surf_idx = gzi * uniforms.grid_width + gxi;
    let surf_mat = surface_material_buf[surf_idx];

    // Get the thickness of the surface material
    var surface_thickness = 0.0;
    var surface_color = c_bedrock;
    if (surf_mat == 1u) {
        surface_thickness = t_paydirt;
        surface_color = c_paydirt;
    } else if (surf_mat == 2u) {
        surface_thickness = t_gravel;
        surface_color = c_gravel;
    } else if (surf_mat == 3u) {
        surface_thickness = t_overburden;
        surface_color = c_overburden;
    } else if (surf_mat == 4u) {
        surface_thickness = t_sediment;
        surface_color = c_sediment;
    }

    // Compute weighted blend of ALL materials based on thickness
    // This gives us the "underlying mixture" color
    let total_loose = t_paydirt + t_gravel + t_overburden + t_sediment + 0.001;
    let w_paydirt = t_paydirt / total_loose;
    let w_gravel = t_gravel / total_loose;
    let w_overburden = t_overburden / total_loose;
    let w_sediment = t_sediment / total_loose;

    let mixture_color = c_paydirt * w_paydirt
                      + c_gravel * w_gravel
                      + c_overburden * w_overburden
                      + c_sediment * w_sediment;

    // Blend surface material with underlying mixture based on surface layer thickness
    // Thin surface layer = more mixture showing through
    // coverage_depth: how thick before fully opaque (in meters)
    let coverage_depth = 0.08;
    let surface_opacity = clamp(surface_thickness / coverage_depth, 0.0, 1.0);

    // If almost no loose material, show bedrock
    let loose_total = t_paydirt + t_gravel + t_overburden + t_sediment;
    let bedrock_blend = smoothstep(0.01, 0.05, loose_total);

    // Final color: surface material on top of mixture, with bedrock underneath
    var base_color = mix(c_bedrock, mixture_color, bedrock_blend);
    base_color = mix(base_color, surface_color, surface_opacity * bedrock_blend);

    // Apply ONE unified procedural detail pattern to the blended color
    let noise_large = fbm(in.world_pos * 5.0, 3) - 0.5;
    let noise_fine = noise3d(in.world_pos * 25.0) - 0.5;

    // Voronoi for subtle stone/grain pattern
    let stones = voronoi(in.world_pos.xz * 12.0);
    let stone_detail = (smoothstep(0.0, 0.15, stones) - 0.5) * 0.1;

    // Combine detail (fades with distance)
    let detail = (noise_large * 0.15 + noise_fine * 0.08 * detail_level + stone_detail * detail_level);

    var color = base_color + detail;

    // ============ LIGHTING ============

    let light_dir = normalize(vec3<f32>(0.4, 0.9, 0.25));
    let n = normalize(in.normal);
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = 0.35;
    let lighting = ambient + diffuse * 0.65;

    // Slope-based darkening (steeper = darker)
    let slope = 1.0 - n.y;
    let slope_darken = 1.0 - slope * 0.35;

    // Crevice/cavity darkening
    let crevice = clamp(slope * slope * 0.25, 0.0, 0.12);

    // Apply lighting
    var lit_color = color * lighting * slope_darken;
    lit_color = lit_color - vec3<f32>(crevice);

    // Specular highlight (wet surfaces - more sediment = shinier)
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let wetness = smoothstep(0.0, 0.02, t_sediment);
    let spec_power = mix(16.0, 64.0, wetness);
    let spec_strength = mix(0.1, 0.25, wetness);
    let spec = pow(max(dot(n, half_dir), 0.0), spec_power) * spec_strength;
    lit_color = lit_color + vec3<f32>(spec);

    // Distance fog
    let fog_color = vec3<f32>(0.7, 0.75, 0.85);
    let fog_start = 25.0;
    let fog_end = 120.0;
    let fog_factor = smoothstep(fog_start, fog_end, dist_to_camera);
    lit_color = mix(lit_color, fog_color, fog_factor * 0.55);

    return vec4<f32>(lit_color, 1.0);
}

fn get_water_surface(x: u32, z: u32) -> f32 {
    let idx = z * uniforms.grid_width + x;
    return water_surface_buf[idx];
}

fn get_water_depth(x: u32, z: u32) -> f32 {
    let idx = z * uniforms.grid_width + x;
    return water_depth_buf[idx];
}

fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    return max(lo, min(v, hi));
}

fn get_water_surface_smooth(x: u32, z: u32) -> f32 {
    let max_x = i32(uniforms.grid_width) - 1;
    let max_z = i32(uniforms.grid_depth) - 1;
    let cx = i32(x);
    let cz = i32(z);
    var sum = 0.0;
    var count = 0.0;
    for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
        for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
            let sx = clamp_i32(cx + dx, 0, max_x);
            let sz = clamp_i32(cz + dz, 0, max_z);
            let idx = u32(sz) * uniforms.grid_width + u32(sx);
            let d = water_depth_buf[idx];
            if (d > 0.001) {
                sum += water_surface_buf[idx];
                count += 1.0;
            }
        }
    }
    if (count > 0.0) {
        return sum / count;
    }
    return get_water_surface(x, z);
}

fn get_suspended_sediment(x: u32, z: u32) -> f32 {
    let idx = z * uniforms.grid_width + x;
    return suspended_sediment_buf[idx]
        + suspended_overburden_buf[idx]
        + suspended_gravel_buf[idx]
        + suspended_paydirt_buf[idx];
}

@vertex
fn vs_water(in: VertexInput) -> WaterVertexOutput {
    var out: WaterVertexOutput;
    
    let gx = u32(in.grid_pos.x);
    let gz = u32(in.grid_pos.y);
    let idx = gz * uniforms.grid_width + gx;
    
    let ground = get_height(gx, gz);
    let surface_raw = get_water_surface(gx, gz);
    let surface = get_water_surface_smooth(gx, gz);
    let depth = surface_raw - ground;
    
    // Determine render height
    var h = surface;
    var render_depth = surface - ground;
    
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
            if (dL > 0.001) {
                snap_h = max(snap_h, get_water_surface_smooth(gx - 1u, gz));
                found_neighbor = true;
            }
        }
        if (gx < uniforms.grid_width - 1u) {
            let dR = get_water_depth(gx + 1u, gz);
            if (dR > 0.001) {
                snap_h = max(snap_h, get_water_surface_smooth(gx + 1u, gz));
                found_neighbor = true;
            }
        }
        if (gz > 0u) {
            let dD = get_water_depth(gx, gz - 1u);
            if (dD > 0.001) {
                snap_h = max(snap_h, get_water_surface_smooth(gx, gz - 1u));
                found_neighbor = true;
            }
        }
        if (gz < uniforms.grid_depth - 1u) {
            let dU = get_water_depth(gx, gz + 1u);
            if (dU > 0.001) {
                snap_h = max(snap_h, get_water_surface_smooth(gx, gz + 1u));
                found_neighbor = true;
            }
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
    var hL = get_water_surface_smooth(gx - 1u, gz);
    if (hL < hL_g + 0.001) { hL = h; }
    if (gx == 0u) { hL = h; }

    var hR = get_water_surface_smooth(gx + 1u, gz);
    if (hR < hR_g + 0.001) { hR = h; }
    if (gx == uniforms.grid_width - 1u) { hR = h; }

    var hD = get_water_surface_smooth(gx, gz - 1u);
    if (hD < hD_g + 0.001) { hD = h; }
    if (gz == 0u) { hD = h; }

    var hU = get_water_surface_smooth(gx, gz + 1u);
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

    // Sample suspended sediment - compute sediment concentration (kg/m³)
    // suspended_sediment is in kg/m², divide by depth to get concentration
    let suspended = get_suspended_sediment(gx, gz);
    let safe_depth = max(depth, 0.01);
    out.sediment_load = suspended / safe_depth; // kg/m³ concentration

    // Sample water velocity for visualization
    // Velocities are stored at cell faces - average adjacent faces for cell-center velocity
    var avg_vel_x = 0.0;
    var avg_vel_z = 0.0;
    
    if (depth > 0.01) {  // Only compute velocity where there's water
        // X-velocity: average of left face (idx-1) and right face (idx)
        let vel_x_right = water_velocity_x_buf[idx];
        var vel_x_left = 0.0;
        if (gx > 0u) {
            vel_x_left = water_velocity_x_buf[idx - 1u];
        }
        avg_vel_x = (vel_x_left + vel_x_right) * 0.5;
        
        // Z-velocity: average of back face (idx-width) and front face (idx)
        let vel_z_front = water_velocity_z_buf[idx];
        var vel_z_back = 0.0;
        if (gz > 0u) {
            vel_z_back = water_velocity_z_buf[idx - uniforms.grid_width];
        }
        avg_vel_z = (vel_z_back + vel_z_front) * 0.5;
    }
    
    out.velocity_mag = sqrt(avg_vel_x * avg_vel_x + avg_vel_z * avg_vel_z);

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
    
    // Base color - clean water
    let deep_color = vec3<f32>(0.1, 0.2, 0.5);
    let shallow_color = vec3<f32>(0.3, 0.5, 0.8);
    let foam_color = vec3<f32>(0.9, 0.95, 1.0);

    // Muddy sediment colors
    let light_mud = vec3<f32>(0.6, 0.45, 0.25);  // Light muddy brown
    let heavy_mud = vec3<f32>(0.35, 0.25, 0.15); // Dark muddy brown

    // Compute sediment factor (0=clean, 1=fully muddy)
    // sediment_load is kg/m³. More sensitive: visible tint at 0.05 kg/m³, fully muddy at 0.2 kg/m³
    let sediment_factor = clamp(in.sediment_load * 5.0, 0.0, 1.0);

    // Clean water base (depth-based)
    let clean_base = mix(shallow_color, deep_color, clamp(in.depth * 0.5, 0.0, 1.0));

    // Muddy water base (also depth-based: lighter when shallow)
    let muddy_base = mix(light_mud, heavy_mud, clamp(in.depth * 0.5, 0.0, 1.0));

    // Blend clean to muddy based on sediment
    let water_base = mix(clean_base, muddy_base, sediment_factor);
    
    // VELOCITY VISUALIZATION: override color based on speed
    // Only show velocity colors where there's meaningful water depth (>5cm)
    // Shallow water can have unstable velocity values
    let vel = select(0.0, in.velocity_mag, in.depth > 0.05);
    let vel_color_still = vec3<f32>(0.1, 0.2, 0.6);   // Blue - still
    let vel_color_slow = vec3<f32>(0.0, 0.6, 0.6);    // Cyan - slow (near erosion threshold)
    let vel_color_medium = vec3<f32>(0.2, 0.8, 0.2);  // Green - medium
    let vel_color_fast = vec3<f32>(0.9, 0.7, 0.1);    // Yellow - fast
    let vel_color_rapid = vec3<f32>(0.9, 0.2, 0.1);   // Red - rapid
    
    var velocity_color = vel_color_still;
    if (vel < 0.05) {
        velocity_color = mix(vel_color_still, vel_color_slow, vel / 0.05);
    } else if (vel < 0.10) {
        velocity_color = mix(vel_color_slow, vel_color_medium, (vel - 0.05) / 0.05);
    } else if (vel < 0.25) {
        velocity_color = mix(vel_color_medium, vel_color_fast, (vel - 0.10) / 0.15);
    } else {
        velocity_color = mix(vel_color_fast, vel_color_rapid, clamp((vel - 0.25) / 0.25, 0.0, 1.0));
    }
    
    // Blend velocity color with water_base based on depth confidence
    // Shallow water uses normal water color, deeper water shows velocity
    let depth_confidence = smoothstep(0.03, 0.15, in.depth);
    let final_water_base = mix(water_base, velocity_color, depth_confidence);
    
    // Foam at edges (depth < 0.05)
    let p_noise = in.world_pos.xz * 10.0;
    let foam_noise = sin(p_noise.x + t * 0.5) * cos(p_noise.y - t * 0.3) * 0.5 + 0.5;
    
    let foam_reach = 0.05;
    var foam_factor = 1.0 - smoothstep(0.0, foam_reach, in.depth);
    foam_factor = foam_factor * (0.5 + 0.5 * foam_noise); // Soften and break up the line
    
    let water_color = mix(final_water_base, foam_color, foam_factor * 0.3); // Dial down foam intensity
    
    // Opacity: Smooth fade-in at shorelines
    // alpha_fade goes from 0.0 at depth 0.0 to 1.0 at depth 0.1
    let alpha_fade = smoothstep(0.0, 0.1, in.depth);
    let alpha_base = alpha_fade * clamp(in.depth * 0.5 + 0.3, 0.0, 0.85);
    var alpha = max(alpha_base, foam_factor * 0.4); // Foam is less opaque

    // Muddy water is more opaque
    alpha = mix(alpha, min(alpha + 0.3, 0.95), sediment_factor);

    // Combine - reduce fresnel/sky reflection for muddy water
    let sky_color = vec3<f32>(0.8, 0.9, 1.0);
    let effective_fresnel = fresnel * (1.0 - sediment_factor * 0.7); // Muddy water less reflective
    let final_color = mix(water_color, sky_color, effective_fresnel * 0.4) + specular * (1.0 - sediment_factor * 0.5);

    // If camera is underwater, reduce opacity so terrain remains visible.
    if (uniforms.camera_pos.y < in.world_pos.y) {
        alpha *= 0.25;
    }

    let final_alpha = clamp(alpha + effective_fresnel * 0.2 + specular * 0.3, 0.0, 0.85);
    return vec4<f32>(final_color, final_alpha);
}
