//! V-cycle multigrid operations.
//!
//! This module contains the V-cycle dispatch methods for the multigrid solver,
//! including smooth, restrict, prolongate, and residual operations.

use super::GpuMgpcgSolver;
use crate::gpu::GpuContext;

impl GpuMgpcgSolver {
    /// Dispatch smooth operation on a specific level
    /// Performs one red-black iteration
    pub fn dispatch_smooth(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        let level_data = &self.levels[level];
        let bind_group = &self.smooth_bind_groups[level];

        // Workgroup counts: each thread handles one cell, but we process half per pass
        // Using 8x8 workgroups
        let workgroup_x = (level_data.width / 2).div_ceil(8);
        let workgroup_y = level_data.height.div_ceil(8);

        // Red pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("MG Smooth Red Level {}", level)),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.smooth_red_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Black pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("MG Smooth Black Level {}", level)),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.smooth_black_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }
    }

    /// Dispatch restriction from fine level to coarse level
    /// Transfers residual from level[fine_level] to divergence of level[fine_level + 1]
    pub fn dispatch_restrict(&self, encoder: &mut wgpu::CommandEncoder, fine_level: usize) {
        if fine_level >= self.num_levels - 1 {
            return; // No coarser level to restrict to
        }

        let coarse = &self.levels[fine_level + 1];
        let bind_group = &self.restrict_bind_groups[fine_level];

        // Workgroup counts for coarse level (each thread handles one coarse cell)
        let workgroup_x = coarse.width.div_ceil(8);
        let workgroup_y = coarse.height.div_ceil(8);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Restrict {} -> {}", fine_level, fine_level + 1)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.restrict_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Dispatch prolongation from coarse level to fine level
    /// Transfers correction from level[coarse_level] to pressure of level[coarse_level - 1]
    pub fn dispatch_prolongate(&self, encoder: &mut wgpu::CommandEncoder, coarse_level: usize) {
        if coarse_level == 0 || coarse_level > self.num_levels - 1 {
            return; // No finer level to prolongate to
        }

        let fine = &self.levels[coarse_level - 1];
        let bind_group = &self.prolongate_bind_groups[coarse_level - 1];

        // Workgroup counts for fine level (each thread handles one fine cell)
        let workgroup_x = fine.width.div_ceil(8);
        let workgroup_y = fine.height.div_ceil(8);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!(
                "MG Prolongate {} <- {}",
                coarse_level - 1,
                coarse_level
            )),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.prolongate_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Dispatch residual computation at a specific level
    /// Computes r = b - Ax (residual = divergence - Laplacian(pressure))
    pub fn dispatch_level_residual(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        let level_data = &self.levels[level];
        let bind_group = &self.residual_bind_groups[level];

        let workgroup_x = level_data.width.div_ceil(8);
        let workgroup_y = level_data.height.div_ceil(8);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Residual Level {}", level)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.mg_residual_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    /// Clear pressure buffer at a specific level to zero
    pub fn dispatch_clear_pressure(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        let level_data = &self.levels[level];
        let bind_group = &self.residual_bind_groups[level];

        // Use 256-wide workgroups for 1D clear
        let total_cells = level_data.width * level_data.height;
        let workgroup_count = total_cells.div_ceil(256);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("MG Clear Level {}", level)),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.clear_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    /// Execute a complete V-cycle
    /// This is the multigrid preconditioner: applies V-cycle to solve Az = r approximately
    ///
    /// Input: residual in levels[0].divergence (or r buffer for PCG)
    /// Output: correction in levels[0].pressure (or z buffer for PCG)
    pub fn dispatch_vcycle(&self, encoder: &mut wgpu::CommandEncoder) {
        // Use 2 levels (512x256 -> 256x128) for now
        // 3+ levels causes instability - needs investigation
        let max_level = 1.min(self.num_levels - 1);
        self.dispatch_vcycle_recursive(encoder, 0, max_level);
    }

    /// Recursive V-cycle implementation
    pub(super) fn dispatch_vcycle_recursive(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: usize,
        max_level: usize,
    ) {
        const PRE_SMOOTH: usize = 3;
        const POST_SMOOTH: usize = 3;
        const COARSE_SOLVE: usize = 15;

        // Pre-smoothing
        for _ in 0..PRE_SMOOTH {
            self.dispatch_smooth(encoder, level);
        }

        if level == max_level {
            // At coarsest level: direct solve with more iterations
            for _ in 0..COARSE_SOLVE {
                self.dispatch_smooth(encoder, level);
            }
        } else {
            // Compute residual at this level
            self.dispatch_level_residual(encoder, level);

            // Restrict residual to coarse level (into coarse divergence)
            self.dispatch_restrict(encoder, level);

            // Clear coarse pressure to zero before solving
            self.dispatch_clear_pressure(encoder, level + 1);

            // Recursively solve on coarse level
            self.dispatch_vcycle_recursive(encoder, level + 1, max_level);

            // Prolongate correction from coarse to fine
            self.dispatch_prolongate(encoder, level + 1);

            // Post-smoothing
            for _ in 0..POST_SMOOTH {
                self.dispatch_smooth(encoder, level);
            }
        }
    }

    /// Run the pressure solve using multigrid V-cycles
    ///
    /// This is a simpler approach than full PCG - just runs multiple V-cycles
    /// which works well in practice for the pressure equation.
    ///
    /// Call `upload()` or `upload_warm()` first, then `solve()`, then `download()`.
    pub fn solve(&self, gpu: &GpuContext, num_vcycles: u32) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MGPCG Solve Encoder"),
            });

        for _ in 0..num_vcycles {
            self.dispatch_vcycle(&mut encoder);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run pressure solve with timing information
    pub fn solve_timed(&self, gpu: &GpuContext, num_vcycles: u32) -> std::time::Duration {
        let start = std::time::Instant::now();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MGPCG Solve Encoder"),
            });

        for _ in 0..num_vcycles {
            self.dispatch_vcycle(&mut encoder);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU to finish
        gpu.device.poll(wgpu::Maintain::Wait);

        start.elapsed()
    }
}
