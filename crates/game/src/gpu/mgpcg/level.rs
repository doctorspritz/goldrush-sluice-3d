//! Multigrid level data structure.
//!
//! Each level in the multigrid hierarchy contains GPU buffers for pressure,
//! residual, divergence, and cell type data at that resolution.

use super::LevelParams;
use crate::gpu::GpuContext;

/// A single level in the multigrid hierarchy
pub struct MgLevel {
    pub width: u32,
    pub height: u32,
    pub cell_count: usize,

    /// Pressure field (solution at this level)
    pub pressure: wgpu::Buffer,
    /// Residual/RHS at this level
    pub residual: wgpu::Buffer,
    /// Divergence (RHS for pressure solve)
    pub divergence: wgpu::Buffer,
    /// Cell types (solid/fluid/air)
    pub cell_type: wgpu::Buffer,
    /// Level parameters uniform
    pub params_buffer: wgpu::Buffer,
}

impl MgLevel {
    /// Create a new multigrid level with GPU buffers
    pub fn new(gpu: &GpuContext, width: u32, height: u32, level: u32) -> Self {
        let cell_count = (width * height) as usize;
        let buffer_size = (cell_count * std::mem::size_of::<f32>()) as u64;

        let pressure = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Pressure", level)),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let residual = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Residual", level)),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let divergence = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Divergence", level)),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Cell type uses u32
        let cell_type = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Cell Type", level)),
            size: (cell_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Level parameters
        let params = LevelParams {
            width,
            height,
            level,
            _pad: 0,
        };
        let params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("MG Level {} Params", level)),
            size: std::mem::size_of::<LevelParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        Self {
            width,
            height,
            cell_count,
            pressure,
            residual,
            divergence,
            cell_type,
            params_buffer,
        }
    }

    /// Clear pressure buffer to zero
    pub fn clear_pressure(&self, gpu: &GpuContext) {
        let zeros = vec![0u8; self.cell_count * std::mem::size_of::<f32>()];
        gpu.queue.write_buffer(&self.pressure, 0, &zeros);
    }
}
