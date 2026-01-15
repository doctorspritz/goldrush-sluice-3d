use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Pos3Color4Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl Pos3Color4Vertex {
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub const BASIC_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

pub const SEDIMENT_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_pos: vec3<f32>,
    @location(3) instance_scale: f32,
    @location(4) instance_rot: vec4<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let scaled = in.position * in.instance_scale;
    let world_pos = in.instance_pos + quat_rotate(in.instance_rot, scaled);
    let normal = normalize(quat_rotate(in.instance_rot, in.normal));
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.2));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let view_dir = normalize(uniforms.camera_pos - world_pos);
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
    let shade = 0.35 + 0.65 * diffuse;
    let tint = in.color.rgb * shade + vec3<f32>(0.08) * rim;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(tint, in.color.a);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
"#;

pub fn build_rock_mesh() -> Vec<MeshVertex> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let inv_len = 1.0 / (1.0 + phi * phi).sqrt();
    let a = inv_len;
    let b = phi * inv_len;

    let mut verts = [
        Vec3::new(-a, b, 0.0),
        Vec3::new(a, b, 0.0),
        Vec3::new(-a, -b, 0.0),
        Vec3::new(a, -b, 0.0),
        Vec3::new(0.0, -a, b),
        Vec3::new(0.0, a, b),
        Vec3::new(0.0, -a, -b),
        Vec3::new(0.0, a, -b),
        Vec3::new(b, 0.0, -a),
        Vec3::new(b, 0.0, a),
        Vec3::new(-b, 0.0, -a),
        Vec3::new(-b, 0.0, a),
    ];

    let seed = 0xB2D4_09A7_u32;
    for (idx, pos) in verts.iter_mut().enumerate() {
        let idx_u = idx as u32;
        let radial = 1.0 + 0.08 * hash_to_unit(seed ^ idx_u.wrapping_mul(11));
        let lateral = Vec3::new(
            hash_to_unit(seed ^ idx_u.wrapping_mul(13)),
            hash_to_unit(seed ^ idx_u.wrapping_mul(17)),
            hash_to_unit(seed ^ idx_u.wrapping_mul(19)),
        ) * 0.04;
        *pos = (*pos * radial) + lateral;
    }

    let mut max_len = 0.0_f32;
    for pos in &verts {
        max_len = max_len.max(pos.length());
    }
    if max_len > 0.0 {
        for pos in &mut verts {
            *pos /= max_len;
        }
    }

    let indices: [[usize; 3]; 20] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    let mut vertices = Vec::with_capacity(indices.len() * 3);
    for tri in indices {
        let va = verts[tri[0]];
        let vb = verts[tri[1]];
        let vc = verts[tri[2]];
        let normal = (vb - va).cross(vc - va).normalize();
        for pos in [va, vb, vc] {
            vertices.push(MeshVertex {
                position: pos.to_array(),
                normal: normal.to_array(),
            });
        }
    }

    vertices
}

fn hash_to_unit(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    let unit = x as f32 / u32::MAX as f32;
    unit * 2.0 - 1.0
}

pub fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
