// Particle rendering shader
// Renders soft circles using instanced quads

struct Uniforms {
    projection: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec2<f32>,  // World position (instance)
    @location(1) color: vec4<f32>,     // RGBA color (instance)
    @location(2) size: f32,            // Particle size in pixels (instance)
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Quad vertices (2 triangles)
    var quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
    );

    var quad_uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );

    let quad_pos = quad_positions[input.vertex_index];
    let uv = quad_uvs[input.vertex_index];

    // Scale quad by particle size (in world units)
    let size_world = input.size / uniforms.viewport_size.x * 2.0;
    let world_pos = input.position + quad_pos * size_world;

    var output: VertexOutput;
    output.position = uniforms.projection * vec4<f32>(world_pos, 0.0, 1.0);
    output.color = input.color;
    output.uv = uv;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Soft circle: distance from center
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(input.uv - center);

    // Smooth falloff at edge
    let alpha = 1.0 - smoothstep(0.35, 0.5, dist);

    if (alpha < 0.01) {
        discard;
    }

    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
