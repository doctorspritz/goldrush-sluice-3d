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
@group(1) @binding(0) var input_tex: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> output_tex: array<f32>; // Compute? Or render pass?
// The plan specified "Fragment Shader" for blur (multipass), which is easier for ping-ponging 
// without storage textures (requires format support).
// Let's implement as Fragment Shader for Fullscreen Quad.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y); // Flip Y for texture coords
    return out;
}

@group(1) @binding(0) var texture_in: texture_2d<f32>;
@group(1) @binding(1) var sampler_in: sampler;

// Specialized constant for direction? Or uniform?
@group(2) @binding(0) var<uniform> direction: vec2<f32>; // (1,0) or (0,1)

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let center_depth = textureSample(texture_in, sampler_in, in.uv).r;
    
    // Background handling (if depth is very large/far)
    if (center_depth < -1000.0) { return center_depth; } // Assume cleared to -Inf or similar far value?
    // Actually, linear View Z is usually negative. Far plane is very negative.
    
    let blur_radius = 5.0; // Kernels size
    let sigma_spatial = 3.0; // Gaussian sigma
    let sigma_depth = uniforms.blur_depth_falloff; // Edge preservation
    
    var sum_weight = 0.0;
    var sum_val = 0.0;
    
    let limits = textureDimensions(texture_in);
    let texel_size = uniforms.texel_size;

    for (var i = -blur_radius; i <= blur_radius; i += 1.0) {
        let offset = direction * i * texel_size;
        let sample_uv = in.uv + offset;
        
        // Bounds check? Sampler clamps usually.
        
        let sample_depth = textureSample(texture_in, sampler_in, sample_uv).r;
        
        // Spatial weight (Gaussian)
        let w_spatial = exp(-(i * i) / (2.0 * sigma_spatial * sigma_spatial));
        
        // Range weight (Depth difference)
        let diff = sample_depth - center_depth;
        let w_range = exp(-(diff * diff) / (2.0 * sigma_depth * sigma_depth));
        
        let weight = w_spatial * w_range;
        
        sum_val += sample_depth * weight;
        sum_weight += weight;
    }
    
    if (sum_weight > 0.0001) {
        return sum_val / sum_weight;
    } else {
        return center_depth;
    }
}
