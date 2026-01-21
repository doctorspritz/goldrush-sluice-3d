//! Grid mesh generation for GPU heightfield rendering.

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridVertex {
    pub position: [f32; 2],
}

impl GridVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GridVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        }
    }
}

/// Generated grid mesh data.
pub struct GridMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl GridMesh {
    /// Generate a grid mesh for the given dimensions.
    pub fn new(device: &wgpu::Device, width: u32, depth: u32) -> Self {
        let mut vertices = Vec::with_capacity((width * depth) as usize);
        let mut indices = Vec::new();

        for z in 0..depth {
            for x in 0..width {
                vertices.push(GridVertex {
                    position: [x as f32, z as f32],
                });

                if x < width - 1 && z < depth - 1 {
                    let i = z * width + x;
                    // Triangle 1
                    indices.push(i);
                    indices.push(i + width);
                    indices.push(i + 1);
                    // Triangle 2
                    indices.push(i + 1);
                    indices.push(i + width);
                    indices.push(i + width + 1);
                }
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }
}
