struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    texel_size: vec2<f32>,
    particle_radius: f32, // Radius in world units
    blur_depth_falloff: f32,
    camera_pos: vec3<f32>,
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> positions: array<vec4<f32>>; // .w used for active check

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_pos: vec3<f32>,     // Center of sphere in view space
    @location(2) sphere_radius_screen: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> VertexOutput {
    let p_vec = positions[i_idx];
    
    // Skip inactive particles (though draw call should limit count, checking w is safer)
    // Assuming .w is active flag or type? In GpuFlip3D it's just x,y,z,pad. 
    // Wait, GpuFlip3D uses [f32; 4] for alignment but .w might be garbage or unused.
    // The plan said: "Uses active_particles (CPU count) for draw call range."
    // So we don't strictly need to check .w if we trust the draw count.
    
    let world_pos = p_vec.xyz;
    let radius = uniforms.particle_radius;
    
    // View space position of sphere center
    let view_pos_vec = uniforms.view * vec4<f32>(world_pos, 1.0);
    let view_pos = view_pos_vec.xyz;
    
    // Point Sprite Expansion
    // Quad vertices [-1, -1] to [1, 1]
    let uv = vec2<f32>(
        f32((v_idx << 1u) & 2u),
        f32(v_idx & 2u)
    ) * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    // Note: The specific UV mapping for a triangle strip or index buffer:
    // With gl_VertexID 0..3:
    // 0: (-1, 1)  1: (-1,-1)  2: ( 1, 1)  3: ( 1,-1) (Triangle Strip)
    // or indexed quad. Let's assume indexed quad 0,1,2, 0,2,3 or just 0..4 non-indexed strip.
    // Simpler explicit array for 4 vertices (Triangle Strip):
    let corner = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, 1.0), // Top Left
        vec2<f32>(-1.0, -1.0), // Bottom Left
        vec2<f32>(1.0, 1.0),  // Top Right
        vec2<f32>(1.0, -1.0)  // Bottom Right
    )[v_idx];

    // Compute billboard size based on radius and projection
    // We want the quad to cover the sphere in screen space.
    // At view Z, world radius R subtends angle ~ R/Z.
    
    let out_pos = view_pos + vec3<f32>(corner.x * radius, corner.y * radius, 0.0);
    
    var out: VertexOutput;
    out.clip_position = uniforms.proj * vec4<f32>(out_pos, 1.0);
    out.uv = corner;
    out.view_pos = view_pos; // Center of sphere
    out.sphere_radius_screen = radius; // Pass world radius, compute screen offset in fragment? 
    // Actually, we need exact sphere depth.
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let uv = in.uv;
    let dist_sq = dot(uv, uv);
    if (dist_sq > 1.0) {
        discard;
    }

    // Reconstruction of sphere surface in View Space
    // x^2 + y^2 + z^2 = r^2
    // We demonstrate the surface depth at this pixel.
    
    let z_sphere = sqrt(1.0 - dist_sq) * uniforms.particle_radius;
    
    // View space Z is negative looking forward in RHS (OpenGL/wgpu style usually)
    // Verify coordinate system: 
    // wgpu/glam assume RHS: -Z is forward.
    // So 'view_pos' z is negative. 
    // The surface of the sphere closer to camera has LARGER Z (less negative).
    let linear_depth = in.view_pos.z + z_sphere;
    
    // Output linear view-space depth
    return linear_depth;
}
