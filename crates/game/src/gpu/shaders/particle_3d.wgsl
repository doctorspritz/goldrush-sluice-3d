// particle_3d.wgsl
// Renders 3D particles as billboards

struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> densities: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32,
) -> VertexOutput {
    let p_vec = positions[i_idx];
    if (p_vec.w < 0.5) {
        return VertexOutput(vec4<f32>(0.0), vec4<f32>(0.0), vec2<f32>(0.0));
    }

    let pos = p_vec.xyz;
    let density = densities[i_idx];
    
    // Choose color based on density
    var color = vec4<f32>(0.2, 0.5, 1.0, 1.0); // Vibrant Water
    if (density > 1.5) {
        color = vec4<f32>(0.6, 0.4, 0.2, 1.0); // Earthy Tailings
    }

    let size = 0.1;
    
    // Billboard logic: quad centered at pos
    let uv = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0)
    )[v_idx];
    
    // Transform world position to view space
    let view_pos = uniforms.view * vec4<f32>(pos, 1.0);
    
    // Expand billboard in view space
    let billboard_view_pos = view_pos.xyz + vec3<f32>(uv.x, uv.y, 0.0) * size;
    
    var out: VertexOutput;
    out.clip_position = uniforms.proj * vec4<f32>(billboard_view_pos, 1.0);
    out.color = color;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.color.a < 0.1) { discard; }
    
    // Circle shape
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }
    
    let normal_z = sqrt(1.0 - dist * dist);
    let lighting = max(0.3, normal_z);
    
    return vec4<f32>(in.color.rgb * lighting, 1.0);
}
