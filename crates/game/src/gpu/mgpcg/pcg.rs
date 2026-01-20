//! Preconditioned Conjugate Gradient (PCG) solver.
//!
//! This module implements the PCG algorithm with V-cycle multigrid as preconditioner.

use super::{GpuMgpcgSolver, PcgParams};
use crate::gpu::GpuContext;

impl GpuMgpcgSolver {
    /// Run full Preconditioned Conjugate Gradient with V-cycle as preconditioner
    ///
    /// This provides guaranteed convergence and may stabilize 3+ level multigrid.
    /// More expensive than pure V-cycles due to dot product synchronizations.
    ///
    /// Call `upload()` or `upload_warm()` first, then `solve_pcg()`, then `download()`.
    pub fn solve_pcg(&self, gpu: &GpuContext, max_iterations: u32) {
        let cell_count = (self.width * self.height) as usize;
        let workgroup_count_1d = cell_count.div_ceil(256) as u32;
        let workgroup_x = self.width.div_ceil(8);
        let workgroup_y = self.height.div_ceil(8);

        // Update params buffer with grid dimensions
        let params = PcgParams {
            width: self.width,
            height: self.height,
            alpha: 0.0,
            length: cell_count as u32,
        };
        gpu.queue
            .write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

        // Step 1: Compute initial residual r = b - Ax
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG Initial Residual"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute r = b - Ax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.residual_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pcg_residual_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 2: Apply preconditioner z = M⁻¹r (V-cycle)
        self.apply_preconditioner(gpu, workgroup_count_1d);

        // Step 3: p = z (copy z to p)
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG p = z"),
                });
            // We need a bind group for p = z, but we have p_update which does p = z + beta*p
            // Let's use copy instead: p = z
            // Need a different bind group for this... or use a buffer copy
            encoder.copy_buffer_to_buffer(&self.z, 0, &self.p, 0, (cell_count * 4) as u64);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 4: rz = dot(r, z)
        let mut rz = self.compute_dot_product(
            gpu,
            workgroup_count_1d,
            self.pcg_dot_rz_bind_group.as_ref().unwrap(),
        );

        // Main PCG iteration loop
        for _iter in 0..max_iterations {
            // Check for convergence (rz is proportional to error)
            if rz.abs() < 1e-10 {
                break;
            }

            // Ap = A*p (apply Laplacian)
            {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("PCG Ap = A*p"),
                        });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Apply Laplacian"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.laplacian_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, self.pcg_laplacian_bind_group.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
                drop(pass);
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }

            // pAp = dot(p, Ap)
            let pap = self.compute_dot_product(
                gpu,
                workgroup_count_1d,
                self.pcg_dot_pap_bind_group.as_ref().unwrap(),
            );

            // Compute alpha = rz / pAp
            let alpha = if pap.abs() > 1e-20 { rz / pap } else { 0.0 };

            // x = x + alpha*p
            self.dispatch_axpy(
                gpu,
                alpha,
                self.pcg_x_update_bind_group.as_ref().unwrap(),
                workgroup_count_1d,
            );

            // r = r - alpha*Ap (note: negative alpha)
            self.dispatch_axpy(
                gpu,
                -alpha,
                self.pcg_r_update_bind_group.as_ref().unwrap(),
                workgroup_count_1d,
            );

            // Save old rz
            let rz_old = rz;

            // Apply preconditioner z = M⁻¹r (V-cycle)
            self.apply_preconditioner(gpu, workgroup_count_1d);

            // rz = dot(r, z)
            rz = self.compute_dot_product(
                gpu,
                workgroup_count_1d,
                self.pcg_dot_rz_bind_group.as_ref().unwrap(),
            );

            // Compute beta = rz / rz_old
            let beta = if rz_old.abs() > 1e-20 {
                rz / rz_old
            } else {
                0.0
            };

            // p = z + beta*p (xpay operation)
            self.dispatch_xpay(gpu, beta, workgroup_count_1d);
        }
    }

    /// Apply V-cycle preconditioner: copies r to divergence, runs V-cycle, copies result to z
    ///
    /// IMPORTANT: The V-cycle uses levels[0].pressure as working buffer, but that same
    /// buffer stores the current solution x. We must save x before V-cycle and restore after.
    fn apply_preconditioner(&self, gpu: &GpuContext, workgroup_count_1d: u32) {
        let cell_size = (self.width * self.height * 4) as u64;

        // Step 1: Save x (levels[0].pressure) to ap buffer temporarily
        // (ap is only used after V-cycle for Laplacian(p), so it's safe to use here)
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG save x to ap"),
                });
            encoder.copy_buffer_to_buffer(&self.levels[0].pressure, 0, &self.ap, 0, cell_size);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 2: Copy -r to levels[0].divergence (negate because V-cycle solves L*z = input,
        // but we need (-L)*z = r, so we pass -r as input to get (-L)*z = r)
        {
            // Set alpha = -1.0 for negated copy
            let params = PcgParams {
                width: self.width,
                height: self.height,
                alpha: -1.0,
                length: (self.width * self.height),
            };
            gpu.queue
                .write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG copy -r to div"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy -r to divergence"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.copy_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.pcg_copy_to_div_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(workgroup_count_1d, 1, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 3: Clear pressure (V-cycle working buffer) before solving
        self.levels[0].clear_pressure(gpu);

        // Step 4: Run V-cycle to compute z ≈ M⁻¹r
        // Use all available levels now that we've fixed the x preservation
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG V-cycle"),
                });
            // Limit to 2 levels for now - more levels may have convergence issues
            let max_level = 1.min(self.num_levels - 1);
            self.dispatch_vcycle_recursive(&mut encoder, 0, max_level);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 5: Copy V-cycle result (levels[0].pressure) to z
        {
            // Set alpha = 1.0 for regular copy (no negation)
            let params = PcgParams {
                width: self.width,
                height: self.height,
                alpha: 1.0,
                length: (self.width * self.height),
            };
            gpu.queue
                .write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG copy pressure to z"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy pressure to z"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.copy_pipeline.as_ref().unwrap());
            pass.set_bind_group(
                0,
                self.pcg_copy_from_pressure_bind_group.as_ref().unwrap(),
                &[],
            );
            pass.dispatch_workgroups(workgroup_count_1d, 1, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Step 6: Restore x from ap back to levels[0].pressure
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PCG restore x from ap"),
                });
            encoder.copy_buffer_to_buffer(&self.ap, 0, &self.levels[0].pressure, 0, cell_size);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Compute dot product and read result back to CPU
    fn compute_dot_product(
        &self,
        gpu: &GpuContext,
        workgroup_count: u32,
        bind_group: &wgpu::BindGroup,
    ) -> f32 {
        // Dispatch partial sum computation
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Dot Partial"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dot Partial Sums"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.dot_partial_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
            drop(pass);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Dispatch finalize to sum partial sums
        {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Dot Finalize"),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dot Finalize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.dot_finalize_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);

            // Copy result to staging buffer
            encoder.copy_buffer_to_buffer(
                &self.partial_sums,
                0,
                &self.sum_staging,
                0,
                std::mem::size_of::<f32>() as u64,
            );
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }

        // Read result from staging buffer
        let buffer_slice = self.sum_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::from_bytes::<f32>(&data[..4]);
        let value = *result;
        drop(data);
        self.sum_staging.unmap();

        value
    }

    /// Dispatch axpy: buffer_a += alpha * buffer_b
    fn dispatch_axpy(
        &self,
        gpu: &GpuContext,
        alpha: f32,
        bind_group: &wgpu::BindGroup,
        workgroup_count: u32,
    ) {
        // Update params with alpha
        let params = PcgParams {
            width: self.width,
            height: self.height,
            alpha,
            length: (self.width * self.height),
        };
        gpu.queue
            .write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG AXPY"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("AXPY"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.axpy_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(pass);
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch xpay: p = z + beta * p
    fn dispatch_xpay(&self, gpu: &GpuContext, beta: f32, workgroup_count: u32) {
        // Update params with beta as alpha
        let params = PcgParams {
            width: self.width,
            height: self.height,
            alpha: beta,
            length: (self.width * self.height),
        };
        gpu.queue
            .write_buffer(&self.pcg_params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG XPAY"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("XPAY p = z + beta*p"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.xpay_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.pcg_p_update_bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(pass);
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
