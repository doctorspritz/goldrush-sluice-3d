// Terrain rendering shader
// Renders solid rectangles for terrain geometry

struct Uniforms {
    projection: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec2<f32>,  // World position (instance) - center of rectangle
    @location(1) color: vec4<f32>,     // RGBA color (instance)
    @location(2) size_x: f32,          // Rectangle width in world units
    @location(3) size_y: f32,          // Rectangle height in world units
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Quad vertices (2 triangles) - unit square from -0.5 to 0.5
    var quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
    );

    let quad_pos = quad_positions[input.vertex_index];

    // Scale by size (width, height) and translate to position
    let world_pos = input.position + quad_pos * vec2<f32>(input.size_x, input.size_y);

    var output: VertexOutput;
    output.position = uniforms.projection * vec4<f32>(world_pos, 0.0, 1.0);
    output.color = input.color;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Solid fill - no soft edges for terrain
    return input.color;
}
