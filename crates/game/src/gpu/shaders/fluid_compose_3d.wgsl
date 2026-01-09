struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    texel_size: vec2<f32>,
    particle_radius: f32,
    blur_depth_falloff: f32,
    camera_pos: vec3<f32>,
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var depth_tex: texture_2d<f32>;
@group(1) @binding(1) var depth_sampler: sampler;
// Potentially background texture for refraction?
// @group(1) @binding(2) var scene_color: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

fn view_pos_from_depth(uv: vec2<f32>, view_z: f32) -> vec3<f32> {
    // Reconstruct view position from Linear View Z and UV
    // We need to unproject.
    // Screen Space (NDC) -> View Space
    
    // Easier way with linear Z in view space:
    // View Ray * Z
    
    // Let's compute NDC and unpack
    // But we strictly have 'view_z'.
    
    // Standard way with inverse projection:
    // We need standard depth (0..1) for inv_proj, but we stored LINEAR VIEW Z.
    
    // Alternative: We know the view ray direction for this UV.
    // let ndc_xy = uv * 2.0 - 1.0;
    // let clip = vec4(ndc_xy, 1.0, 1.0); or similar.
    // 
    // Let's rely on inv_proj.
    // But inv_proj expects non-linear depth from the depth buffer usually.
    // Here our texture IS linear view Z.
    // So we can compute x_view and y_view using:
    // x_view = view_z * (ndc_x / proj[0][0])
    // y_view = view_z * (ndc_y / proj[1][1])
    
    // UV (0,0) is Top-Left. WGPU NDC Y is Up. 
    // So UV (0,0) -> NDC (-1, 1)
    // NDC X = uv.x * 2.0 - 1.0
    // NDC Y = (1.0 - uv.y) * 2.0 - 1.0 = 1.0 - uv.y * 2.0
    
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    
    // View Space reconstruction for perspective:
    // x_view = (ndc_x * -view_z) / proj[0][0]
    // y_view = (ndc_y * -view_z) / proj[1][1]
    
    let x_view = (ndc.x * -view_z) / uniforms.proj[0][0];
    let y_view = (ndc.y * -view_z) / uniforms.proj[1][1];
    
    return vec3<f32>(x_view, y_view, view_z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, depth_sampler, in.uv).r;
    
    // Handle background (assuming cleared to very negative Z)
    if (depth < -1000.0) {
        discard;
    }
    
    let view_pos = view_pos_from_depth(in.uv, depth);
    
    // Normal reconstruction via finite differences
    let uv_r = in.uv + vec2<f32>(uniforms.texel_size.x, 0.0);
    let uv_u = in.uv + vec2<f32>(0.0, uniforms.texel_size.y);
    
    let depth_r = textureSample(depth_tex, depth_sampler, uv_r).r;
    let depth_u = textureSample(depth_tex, depth_sampler, uv_u).r;
    
    // If neighbors are background, use center depth to prevent normal artifacts at edges
    var final_depth_r = depth_r;
    if (depth_r < -1000.0) { final_depth_r = depth; }
    var final_depth_u = depth_u;
    if (depth_u < -1000.0) { final_depth_u = depth; }
    
    let view_pos_r = view_pos_from_depth(uv_r, final_depth_r);
    let view_pos_u = view_pos_from_depth(uv_u, final_depth_u);
    
    let ddx = view_pos_r - view_pos;
    let ddy = view_pos_u - view_pos;
    
    var N = normalize(cross(ddx, ddy)); // View space normal
    
    // Safety check for normal
    if (length(cross(ddx, ddy)) < 0.0001) {
        N = vec3<f32>(0.0, 0.0, 1.0);
    }
    
    // Simple Lighting
    let L = normalize(vec3<f32>(0.5, 1.0, 0.5)); // Light dir in View Space (approx)
    
    // Specular
    let V = normalize(-view_pos); // View vector
    let H = normalize(L + V);
    let spec = pow(max(dot(N, H), 0.0), 64.0);
    
    // Fresnel
    let F0 = 0.02; // Water
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    
    // Brighter debug colors
    let water_color = vec3<f32>(0.1, 0.6, 1.0); // Brighter blue
    let sky_color = vec3<f32>(0.8, 0.9, 1.0);
    
    let diff = max(dot(N, L), 0.0);
    
    let color = mix(water_color * (diff * 0.8 + 0.2), sky_color, fresnel) + vec3<f32>(spec);
    
    return vec4<f32>(color, 0.8); // Alpha 0.8
}
