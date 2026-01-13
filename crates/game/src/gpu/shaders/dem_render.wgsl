struct DemUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: DemUniforms;

@group(1) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> orientations: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> template_ids: array<u32>;
@group(1) @binding(3) var<storage, read> templates: array<GpuClumpTemplate>;
@group(1) @binding(4) var<storage, read> sphere_offsets: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read> sphere_radii: array<f32>;

struct GpuClumpTemplate {
    sphere_count: u32,
    mass: f32,
    radius: f32,
    pad0: f32,
    inertia_inv: mat3x3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

const MAX_SPHERES_PER_CLUMP: u32 = 100u;

// Quaternion rotation
fn rotate_vector(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_idx: u32,
    @builtin(vertex_index) vertex_idx: u32
) -> VertexOutput {
    let particle_idx = instance_idx / MAX_SPHERES_PER_CLUMP;
    let sphere_idx = instance_idx % MAX_SPHERES_PER_CLUMP;
    
    // Check if particle active? Need flags buffer?
    // DemRenderer didn't bind flags. 
    // Assuming inactive particles have pos far away or we accept rendering them 
    // (inactive usually moved to special place or flagged).
    // Let's assume active for now, or position check.
    
    let template_id = template_ids[particle_idx];
    
    var output: VertexOutput;
    
    if template_id >= 100u { 
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return output;
    }
    
    let sphere_count = templates[template_id].sphere_count;
    
    if sphere_idx >= sphere_count {
        output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0); // Discard
        return output;
    }
    
    let pos = positions[particle_idx].xyz;
    
    // Check if active by position (hacky but saves binding flags)
    if pos.y < -900.0 {
         output.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
         return output;
    }
    
    let orient = orientations[particle_idx];
    
    let template_offset = template_id * MAX_SPHERES_PER_CLUMP;
    let local_offset = sphere_offsets[template_offset + sphere_idx].xyz;
    let radius = sphere_radii[template_offset + sphere_idx];
    
    let rotated_offset = rotate_vector(orient, local_offset);
    let center = pos + rotated_offset;
    
    // Billboarding
    // Extract camera right/up vectors from View matrix
    // View matrix rows 0, 1, 2 are Right, Up, Forward (assuming row-major in WGSL?)
    // In WGSL mat4x4 is column-major.
    // So Column 0 is Right, Column 1 is Up.
    // Transpose of View is roughly Camera -> World rotation.
    // inv_view = view^-1.
    // right = vec3(view[0].x, view[1].x, view[2].x)
    // up = vec3(view[0].y, view[1].y, view[2].y)
    let right = vec3<f32>(uniforms.view[0].x, uniforms.view[1].x, uniforms.view[2].x);
    let up = vec3<f32>(uniforms.view[0].y, uniforms.view[1].y, uniforms.view[2].y);
    
    // Quad vertices
    var uv = vec2<f32>(0.0, 0.0);
    switch vertex_idx {
        case 0u: { uv = vec2<f32>(-1.0, -1.0); }
        case 1u: { uv = vec2<f32>( 1.0, -1.0); }
        case 2u: { uv = vec2<f32>(-1.0,  1.0); }
        case 3u: { uv = vec2<f32>( 1.0,  1.0); }
        default: {}
    }
    
    let vertex_pos = center + (right * uv.x + up * uv.y) * radius;
    
    output.clip_position = uniforms.proj * uniforms.view * vec4<f32>(vertex_pos, 1.0);
    output.uv = uv;
    
    // Color based on particle index for debugging
    let r = f32(particle_idx % 10u) / 10.0;
    let g = f32((particle_idx / 10u) % 10u) / 10.0;
    let b = 0.5;
    output.color = vec3<f32>(r, g, b);
    
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circle mask
    let r_sq = dot(in.uv, in.uv);
    if r_sq > 1.0 {
        discard;
    }
    
    // Simple shading (fake normal)
    let z = sqrt(1.0 - r_sq);
    let normal = vec3<f32>(in.uv, z);
    let light = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(normal, light), 0.2);
    
    return vec4<f32>(in.color * diffuse, 1.0);
}
