//! GPU MGPCG (Multigrid-Preconditioned Conjugate Gradient) Pressure Solver
//!
//! This solver uses a V-cycle multigrid preconditioner inside a conjugate gradient
//! outer loop for guaranteed convergence. The multigrid hierarchy provides fast
//! error smoothing at all spatial frequencies, while CG ensures monotonic convergence.
//!
//! Architecture:
//! - PCG outer loop (15-20 iterations)
//! - V-cycle preconditioner: 4 levels (512→256→128→64)
//! - Red-Black Gauss-Seidel smoother at each level
//! - Full-weighting restriction, bilinear prolongation

use super::GpuContext;
use bytemuck::{Pod, Zeroable};

mod level;
mod pcg;
mod pipelines;
mod vcycle;

pub use level::MgLevel;

/// Parameters for a single multigrid level
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LevelParams {
    pub width: u32,
    pub height: u32,
    pub level: u32,
    pub _pad: u32,
}

/// Parameters for restriction operation (fine → coarse)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RestrictParams {
    pub fine_width: u32,
    pub fine_height: u32,
    pub coarse_width: u32,
    pub coarse_height: u32,
}

/// Parameters for prolongation operation (coarse → fine)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ProlongateParams {
    pub fine_width: u32,
    pub fine_height: u32,
    pub coarse_width: u32,
    pub coarse_height: u32,
}

/// Parameters for PCG vector operations
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PcgParams {
    pub width: u32,
    pub height: u32,
    pub alpha: f32,  // Scalar for axpy/xpay operations
    pub length: u32, // Total number of elements (for 1D ops)
}

/// Scalar value for GPU reduction operations
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ScalarValue {
    pub value: f32,
    pub _pad: [u32; 3],
}

/// GPU MGPCG Pressure Solver
///
/// Uses multigrid-preconditioned conjugate gradient for stable, fast convergence.
pub struct GpuMgpcgSolver {
    /// Base grid dimensions (level 0)
    pub width: u32,
    pub height: u32,

    /// Number of multigrid levels
    pub num_levels: usize,

    /// Multigrid level hierarchy (level 0 = finest)
    pub levels: Vec<MgLevel>,

    // PCG vectors (level 0 size only)
    /// Residual vector r = b - Ax
    pub r: wgpu::Buffer,
    /// Preconditioned residual z = M⁻¹r
    pub z: wgpu::Buffer,
    /// Search direction p
    pub p: wgpu::Buffer,
    /// Matrix-vector product Ap
    pub ap: wgpu::Buffer,

    // Reduction buffers for dot products
    /// Partial sums from workgroups (one f32 per workgroup)
    pub partial_sums: wgpu::Buffer,
    /// Final scalar result
    pub final_sum: wgpu::Buffer,
    /// Staging buffer for reading scalar back to CPU
    pub sum_staging: wgpu::Buffer,

    // Compute pipelines (will be created in Step 2+)
    // Smoother pipelines
    pub smooth_red_pipeline: Option<wgpu::ComputePipeline>,
    pub smooth_black_pipeline: Option<wgpu::ComputePipeline>,

    // Multigrid transfer pipelines
    pub restrict_pipeline: Option<wgpu::ComputePipeline>,
    pub prolongate_pipeline: Option<wgpu::ComputePipeline>,

    // Multigrid residual/clear pipelines
    pub mg_residual_pipeline: Option<wgpu::ComputePipeline>,
    pub clear_pipeline: Option<wgpu::ComputePipeline>,

    // PCG operation pipelines
    pub residual_pipeline: Option<wgpu::ComputePipeline>,
    pub laplacian_pipeline: Option<wgpu::ComputePipeline>,
    pub axpy_pipeline: Option<wgpu::ComputePipeline>,
    pub xpay_pipeline: Option<wgpu::ComputePipeline>,
    pub copy_pipeline: Option<wgpu::ComputePipeline>,
    pub dot_partial_pipeline: Option<wgpu::ComputePipeline>,
    pub dot_finalize_pipeline: Option<wgpu::ComputePipeline>,

    // Bind group layouts (will be created in Step 2+)
    pub smooth_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub restrict_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub prolongate_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Bind groups per level
    pub smooth_bind_groups: Vec<wgpu::BindGroup>,
    pub restrict_bind_groups: Vec<wgpu::BindGroup>,
    pub prolongate_bind_groups: Vec<wgpu::BindGroup>,
    pub residual_bind_groups: Vec<wgpu::BindGroup>,

    // PCG bind group layout and bind groups
    pub pcg_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Bind group for r = b - Ax: (x=pressure, b=divergence, cell_type, r=output)
    pub pcg_residual_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for Ap = A*p: (p, cell_type, ap=output)
    pub pcg_laplacian_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for x += α*p: (pressure, p, -, -)
    pub pcg_x_update_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for r -= α*Ap: (r, ap, -, -)
    pub pcg_r_update_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for p = z + β*p: (p, z, -, -)
    pub pcg_p_update_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for divergence = r (copy for V-cycle input)
    pub pcg_copy_to_div_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for z = pressure (copy V-cycle output)
    pub pcg_copy_from_pressure_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for dot(r, z)
    pub pcg_dot_rz_bind_group: Option<wgpu::BindGroup>,
    /// Bind group for dot(p, Ap)
    pub pcg_dot_pap_bind_group: Option<wgpu::BindGroup>,
    /// PCG params buffer (updated per operation)
    pub pcg_params_buffer: wgpu::Buffer,

    // Staging buffer for pressure download
    pub pressure_staging: wgpu::Buffer,
}

impl GpuMgpcgSolver {
    /// Create a new MGPCG solver with the given grid dimensions
    ///
    /// Builds a multigrid hierarchy with levels halving until min dimension < 16
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        // Build level hierarchy: 512→256→128→64 (or until min < 16)
        let mut levels = Vec::new();
        let (mut w, mut h) = (width, height);
        let mut level_idx = 0u32;

        // Always add finest level
        levels.push(MgLevel::new(gpu, w, h, level_idx));
        level_idx += 1;

        // Add coarser levels until we reach minimum size
        while w >= 32 && h >= 32 {
            w /= 2;
            h /= 2;
            levels.push(MgLevel::new(gpu, w, h, level_idx));
            level_idx += 1;
        }

        let num_levels = levels.len();
        let cell_count = (width * height) as usize;
        let buffer_size = (cell_count * std::mem::size_of::<f32>()) as u64;

        log::info!(
            "MGPCG: Created {} levels: {}",
            num_levels,
            levels
                .iter()
                .map(|l| format!("{}x{}", l.width, l.height))
                .collect::<Vec<_>>()
                .join(" → ")
        );

        // Create PCG vectors (level 0 size)
        let r = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG r (residual)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let z = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG z (preconditioned)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let p = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG p (direction)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ap = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG Ap (matrix-vector)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Reduction buffers for dot products
        // Max workgroups at level 0: ceil(512*512 / 256) = 1024
        let max_workgroups = ((width * height) as usize).div_ceil(256);
        let partial_sums = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Product Partial Sums"),
            size: (max_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let final_sum = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dot Product Final Sum"),
            size: std::mem::size_of::<ScalarValue>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sum_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scalar Staging"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffer for pressure download
        let pressure_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // PCG params buffer (updated per operation via queue.write_buffer)
        let pcg_params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PCG Params"),
            size: std::mem::size_of::<PcgParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create smooth pipelines and bind groups
        let (
            smooth_bind_group_layout,
            smooth_red_pipeline,
            smooth_black_pipeline,
            smooth_bind_groups,
        ) = Self::create_smooth_pipelines(gpu, &levels);

        // Create restrict pipelines and bind groups (between adjacent levels)
        let (restrict_bind_group_layout, restrict_pipeline, restrict_bind_groups) =
            Self::create_restrict_pipelines(gpu, &levels);

        // Create prolongate pipelines and bind groups (between adjacent levels)
        let (prolongate_bind_group_layout, prolongate_pipeline, prolongate_bind_groups) =
            Self::create_prolongate_pipelines(gpu, &levels);

        // Create residual computation pipelines
        let (mg_residual_pipeline, clear_pipeline, residual_bind_groups) =
            Self::create_residual_pipelines(gpu, &levels);

        // Create PCG pipelines and bind groups
        let (
            pcg_bind_group_layout,
            residual_pipeline,
            laplacian_pipeline,
            axpy_pipeline,
            xpay_pipeline,
            copy_pipeline,
            dot_partial_pipeline,
            dot_finalize_pipeline,
            pcg_residual_bind_group,
            pcg_laplacian_bind_group,
            pcg_x_update_bind_group,
            pcg_r_update_bind_group,
            pcg_p_update_bind_group,
            pcg_copy_to_div_bind_group,
            pcg_copy_from_pressure_bind_group,
            pcg_dot_rz_bind_group,
            pcg_dot_pap_bind_group,
        ) = Self::create_pcg_pipelines(
            gpu,
            &levels[0],
            &r,
            &z,
            &p,
            &ap,
            &partial_sums,
            &pcg_params_buffer,
        );

        Self {
            width,
            height,
            num_levels,
            levels,
            r,
            z,
            p,
            ap,
            partial_sums,
            final_sum,
            sum_staging,
            smooth_red_pipeline: Some(smooth_red_pipeline),
            smooth_black_pipeline: Some(smooth_black_pipeline),
            restrict_pipeline: Some(restrict_pipeline),
            prolongate_pipeline: Some(prolongate_pipeline),
            mg_residual_pipeline: Some(mg_residual_pipeline),
            clear_pipeline: Some(clear_pipeline),
            residual_pipeline: Some(residual_pipeline),
            laplacian_pipeline: Some(laplacian_pipeline),
            axpy_pipeline: Some(axpy_pipeline),
            xpay_pipeline: Some(xpay_pipeline),
            copy_pipeline: Some(copy_pipeline),
            dot_partial_pipeline: Some(dot_partial_pipeline),
            dot_finalize_pipeline: Some(dot_finalize_pipeline),
            smooth_bind_group_layout: Some(smooth_bind_group_layout),
            restrict_bind_group_layout: Some(restrict_bind_group_layout),
            prolongate_bind_group_layout: Some(prolongate_bind_group_layout),
            smooth_bind_groups,
            restrict_bind_groups,
            prolongate_bind_groups,
            residual_bind_groups,
            pcg_bind_group_layout: Some(pcg_bind_group_layout),
            pcg_residual_bind_group: Some(pcg_residual_bind_group),
            pcg_laplacian_bind_group: Some(pcg_laplacian_bind_group),
            pcg_x_update_bind_group: Some(pcg_x_update_bind_group),
            pcg_r_update_bind_group: Some(pcg_r_update_bind_group),
            pcg_p_update_bind_group: Some(pcg_p_update_bind_group),
            pcg_copy_to_div_bind_group: Some(pcg_copy_to_div_bind_group),
            pcg_copy_from_pressure_bind_group: Some(pcg_copy_from_pressure_bind_group),
            pcg_dot_rz_bind_group: Some(pcg_dot_rz_bind_group),
            pcg_dot_pap_bind_group: Some(pcg_dot_pap_bind_group),
            pcg_params_buffer,
            pressure_staging,
        }
    }

    /// Upload divergence and cell_type from CPU to level 0
    pub fn upload(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32]) {
        let level0 = &self.levels[0];
        gpu.queue
            .write_buffer(&level0.divergence, 0, bytemuck::cast_slice(divergence));
        gpu.queue
            .write_buffer(&level0.cell_type, 0, bytemuck::cast_slice(cell_type));

        // Clear pressure to zero
        level0.clear_pressure(gpu);
    }

    /// Upload with warm start from previous pressure
    pub fn upload_warm(
        &self,
        gpu: &GpuContext,
        divergence: &[f32],
        cell_type: &[u32],
        pressure: &[f32],
    ) {
        let level0 = &self.levels[0];
        gpu.queue
            .write_buffer(&level0.divergence, 0, bytemuck::cast_slice(divergence));
        gpu.queue
            .write_buffer(&level0.cell_type, 0, bytemuck::cast_slice(cell_type));
        gpu.queue
            .write_buffer(&level0.pressure, 0, bytemuck::cast_slice(pressure));
    }

    /// Download pressure results from GPU
    pub fn download(&self, gpu: &GpuContext, pressure: &mut [f32]) {
        let level0 = &self.levels[0];

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Pressure Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &level0.pressure,
            0,
            &self.pressure_staging,
            0,
            (self.width * self.height * 4) as u64,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let buffer_slice = self.pressure_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            pressure.copy_from_slice(bytemuck::cast_slice(&data));
        }

        self.pressure_staging.unmap();
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MgpcgMemoryStats {
        let mut level_memory = 0u64;
        for level in &self.levels {
            // pressure + residual + divergence (f32) + cell_type (u32)
            let level_size = level.cell_count as u64 * 4 * 4;
            level_memory += level_size;
        }

        let pcg_memory = (self.width * self.height) as u64 * 4 * 4; // r, z, p, Ap
        let cell_count = (self.width * self.height) as usize;
        let reduction_memory = (cell_count.div_ceil(256) * 4 + 16) as u64; // partial_sums + final_sum

        MgpcgMemoryStats {
            level_memory,
            pcg_memory,
            reduction_memory,
            total: level_memory + pcg_memory + reduction_memory,
        }
    }
}

/// Memory usage statistics for MGPCG solver
#[derive(Debug)]
pub struct MgpcgMemoryStats {
    pub level_memory: u64,
    pub pcg_memory: u64,
    pub reduction_memory: u64,
    pub total: u64,
}

#[cfg(test)]
mod tests {

    // Note: These tests require a GPU context which isn't available in unit tests
    // Integration tests should be run with the game binary
}
