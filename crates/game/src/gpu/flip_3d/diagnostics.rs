//! Physics and jamming diagnostics for GPU FLIP simulations.
//!
//! This module provides tools for reading back and analyzing GPU buffers
//! to validate physics calculations and diagnose sediment jamming patterns.

use super::GpuFlip3D;
use std::sync::mpsc;

/// Physics diagnostic data for testing
#[derive(Debug, Clone)]
pub struct PhysicsDiagnostics {
    pub max_divergence: f32,
    pub avg_divergence: f32,
    pub max_pressure: f32,
    pub avg_pressure: f32,
    pub fluid_cell_count: usize,
    pub divergence_values: Vec<f32>,
    pub pressure_values: Vec<f32>,
}

impl GpuFlip3D {
    /// Print jamming diagnostics - call this every N frames from the example
    pub fn print_jamming_diagnostics(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Read back cell types
        let cell_type_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Count Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Count Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Staging"),
            size: (self.width * self.height * self.depth * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Jamming Diagnostics Readback"),
        });
        encoder.copy_buffer_to_buffer(
            &self.pressure.cell_type_buffer,
            0,
            &cell_type_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.sediment_count_buffer,
            0,
            &sediment_count_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.particle_count_buffer,
            0,
            &particle_count_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.vorticity_mag_buffer,
            0,
            &vorticity_staging,
            0,
            (self.width * self.height * self.depth * 4) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read buffers
        let ct_slice = cell_type_staging.slice(..);
        let sed_slice = sediment_count_staging.slice(..);
        let part_slice = particle_count_staging.slice(..);
        let vort_slice = vorticity_staging.slice(..);

        let (ct_tx, ct_rx) = mpsc::channel();
        let (sed_tx, sed_rx) = mpsc::channel();
        let (part_tx, part_rx) = mpsc::channel();
        let (vort_tx, vort_rx) = mpsc::channel();

        ct_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = ct_tx.send(r);
        });
        sed_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sed_tx.send(r);
        });
        part_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = part_tx.send(r);
        });
        vort_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = vort_tx.send(r);
        });

        device.poll(wgpu::Maintain::Wait);
        if super::super::await_buffer_map(ct_rx).is_err()
            || super::super::await_buffer_map(sed_rx).is_err()
            || super::super::await_buffer_map(part_rx).is_err()
            || super::super::await_buffer_map(vort_rx).is_err()
        {
            log::error!("GPU jamming diagnostics readback failed");
            return;
        }

        let ct_data = ct_slice.get_mapped_range();
        let sed_data = sed_slice.get_mapped_range();
        let part_data = part_slice.get_mapped_range();
        let vort_data = vort_slice.get_mapped_range();

        let cell_types: &[u32] = bytemuck::cast_slice(&ct_data);
        let sediment_counts: &[i32] = bytemuck::cast_slice(&sed_data);
        let particle_counts: &[i32] = bytemuck::cast_slice(&part_data);
        let vorticity_mag: &[f32] = bytemuck::cast_slice(&vort_data);

        // Count cell types
        let solid_count = cell_types.iter().filter(|&&ct| ct == 2).count();
        let fluid_count = cell_types.iter().filter(|&&ct| ct == 1).count();
        let air_count = cell_types.iter().filter(|&&ct| ct == 0).count();

        // Find sample cells with sediment - sample riffle area (x=12-20) where sediment accumulates
        let mut sample_cells = Vec::new();
        let sample_i_start = 12u32.min(self.width - 1);
        let sample_i_end = 20u32.min(self.width - 1);
        for j in 0..self.height.min(8) {
            for k in (self.depth / 2).saturating_sub(2)..=(self.depth / 2 + 2).min(self.depth - 1) {
                for i in sample_i_start..=sample_i_end {
                    let idx = (k * self.width * self.height + j * self.width + i) as usize;
                    let sed_count = sediment_counts[idx];
                    let total_count = particle_counts[idx];
                    let wat_count = total_count - sed_count;
                    let cell_type = cell_types[idx];
                    let vort = vorticity_mag[idx];

                    if sed_count > 0 || total_count > 0 {
                        sample_cells.push((
                            i,
                            j,
                            k,
                            sed_count,
                            wat_count,
                            total_count,
                            cell_type,
                            vort,
                        ));
                    }
                }
            }
        }

        // Calculate vorticity statistics in riffle area
        let mut vort_sum = 0.0f32;
        let mut vort_count = 0;
        let mut vort_max = 0.0f32;
        for j in 0..self.height.min(12) {
            for k in (self.depth / 2).saturating_sub(2)..=(self.depth / 2 + 2).min(self.depth - 1) {
                for i in sample_i_start..=sample_i_end {
                    let idx = (k * self.width * self.height + j * self.width + i) as usize;
                    let vort = vorticity_mag[idx];
                    if vort > 0.0 {
                        vort_sum += vort;
                        vort_count += 1;
                        vort_max = vort_max.max(vort);
                    }
                }
            }
        }

        // Print summary
        println!("\n========== JAMMING DIAGNOSTICS ==========");
        println!(
            "Cell Types: SOLID={} FLUID={} AIR={} (total={})",
            solid_count,
            fluid_count,
            air_count,
            cell_types.len()
        );

        // Print vorticity statistics
        let vort_avg = if vort_count > 0 {
            vort_sum / vort_count as f32
        } else {
            0.0
        };
        println!("\nVorticity in riffle area:");
        println!(
            "  avg={:.4}  max={:.4}  threshold={:.2}  lift_coeff={:.2}",
            vort_avg, vort_max, self.sediment_vorticity_threshold, self.sediment_vorticity_lift
        );

        // Calculate effective lift
        let vort_excess_avg = (vort_avg - self.sediment_vorticity_threshold).max(0.0);
        let lift_factor_avg = (self.sediment_vorticity_lift * vort_excess_avg).min(0.9);
        let settling_cancellation_pct = lift_factor_avg * 100.0;
        println!(
            "  → Average lift cancels {:.1}% of settling velocity",
            settling_cancellation_pct
        );

        if !sample_cells.is_empty() {
            println!("\nSample cells (riffle area i=12-20, j=0-7):");
            println!("  (i, j, k) -> sed | water | total | vort | type");
            for (i, j, k, sed, wat, total, ct, vort) in sample_cells.iter().take(20) {
                let type_str = match ct {
                    0 => "AIR",
                    1 => "FLUID",
                    2 => "SOLID",
                    _ => "???",
                };
                let dominance = if *sed > *wat {
                    "SED>"
                } else if *wat > *sed {
                    "WAT>"
                } else {
                    "="
                };
                println!(
                    "  ({:3},{:3},{:3}) -> {:3} | {:3} | {:3} | {:.3} | {} {}",
                    i, j, k, sed, wat, total, vort, type_str, dominance
                );
            }
        }

        // Check support chains (sample column at riffle area)
        let riffle_i = 15u32.min(self.width - 1); // Middle of riffle area
        let mid_k = self.depth / 2;
        println!(
            "\nRiffle column support chain (i={}, k={}):",
            riffle_i, mid_k
        );
        for j in 0..self.height.min(12) {
            let idx = (mid_k * self.width * self.height + j * self.width + riffle_i) as usize;
            let sed_count = sediment_counts[idx];
            let total_count = particle_counts[idx];
            let wat_count = total_count - sed_count;
            let cell_type = cell_types[idx];
            let vort = vorticity_mag[idx];
            let type_str = match cell_type {
                0 => "AIR  ",
                1 => "FLUID",
                2 => "SOLID",
                _ => "???  ",
            };
            let vort_excess = (vort - self.sediment_vorticity_threshold).max(0.0);
            let lift_factor = (self.sediment_vorticity_lift * vort_excess).min(0.9);
            println!(
                "  j={:2}: {} | sed={:2} wat={:2} | vort={:.3} lift={:.0}% | {}",
                j,
                type_str,
                sed_count,
                wat_count,
                vort,
                lift_factor * 100.0,
                if sed_count > wat_count {
                    "SED>"
                } else if wat_count > sed_count {
                    "WAT>"
                } else {
                    "="
                }
            );
        }
        println!("=========================================\n");

        drop(ct_data);
        drop(sed_data);
        drop(part_data);
        drop(vort_data);
        cell_type_staging.unmap();
        sediment_count_staging.unmap();
        particle_count_staging.unmap();
        vorticity_staging.unmap();
    }

    /// Read back divergence and pressure for physics validation
    pub fn read_physics_diagnostics(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> PhysicsDiagnostics {
        let cell_count = (self.width * self.height * self.depth) as usize;
        let buffer_size = (cell_count * std::mem::size_of::<f32>()) as u64;

        // Create staging buffers
        let div_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Divergence Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pressure_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_type_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Type Staging"),
            size: (cell_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics Diagnostics Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.pressure.divergence_buffer, 0, &div_staging, 0, buffer_size);
        encoder.copy_buffer_to_buffer(&self.pressure.pressure_buffer, 0, &pressure_staging, 0, buffer_size);
        encoder.copy_buffer_to_buffer(
            &self.pressure.cell_type_buffer,
            0,
            &cell_type_staging,
            0,
            (cell_count * std::mem::size_of::<u32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Map buffers
        let div_slice = div_staging.slice(..);
        let pressure_slice = pressure_staging.slice(..);
        let cell_type_slice = cell_type_staging.slice(..);

        let (div_tx, div_rx) = mpsc::channel();
        let (p_tx, p_rx) = mpsc::channel();
        let (ct_tx, ct_rx) = mpsc::channel();

        div_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = div_tx.send(r); });
        pressure_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = p_tx.send(r); });
        cell_type_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = ct_tx.send(r); });

        device.poll(wgpu::Maintain::Wait);
        if super::super::await_buffer_map(div_rx).is_err()
            || super::super::await_buffer_map(p_rx).is_err()
            || super::super::await_buffer_map(ct_rx).is_err()
        {
            log::error!("GPU pressure diagnostics readback failed");
            return PhysicsDiagnostics {
                max_divergence: 0.0,
                avg_divergence: 0.0,
                max_pressure: 0.0,
                avg_pressure: 0.0,
                fluid_cell_count: 0,
                divergence_values: Vec::new(),
                pressure_values: Vec::new(),
            };
        }

        let div_data = div_slice.get_mapped_range();
        let p_data = pressure_slice.get_mapped_range();
        let ct_data = cell_type_slice.get_mapped_range();

        let divergence: &[f32] = bytemuck::cast_slice(&div_data);
        let pressure: &[f32] = bytemuck::cast_slice(&p_data);
        let cell_types: &[u32] = bytemuck::cast_slice(&ct_data);

        // Calculate statistics for fluid cells only
        let mut max_div = 0.0f32;
        let mut sum_div = 0.0f32;
        let mut max_p = 0.0f32;
        let mut sum_p = 0.0f32;
        let mut fluid_count = 0usize;
        let mut div_values = Vec::new();
        let mut p_values = Vec::new();

        for i in 0..cell_count {
            if cell_types[i] == 1 { // FLUID
                let div = divergence[i].abs();
                let p = pressure[i];
                max_div = max_div.max(div);
                sum_div += div;
                max_p = max_p.max(p);
                sum_p += p;
                div_values.push(divergence[i]);
                p_values.push(p);
                fluid_count += 1;
            }
        }

        drop(div_data);
        drop(p_data);
        drop(ct_data);
        div_staging.unmap();
        pressure_staging.unmap();
        cell_type_staging.unmap();

        let avg_div = if fluid_count > 0 { sum_div / fluid_count as f32 } else { 0.0 };
        let avg_p = if fluid_count > 0 { sum_p / fluid_count as f32 } else { 0.0 };

        PhysicsDiagnostics {
            max_divergence: max_div,
            avg_divergence: avg_div,
            max_pressure: max_p,
            avg_pressure: avg_p,
            fluid_cell_count: fluid_count,
            divergence_values: div_values,
            pressure_values: p_values,
        }
    }

    /// Compute and read post-correction divergence (re-runs divergence shader on current grid velocities)
    pub fn compute_post_correction_divergence(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> f32 {
        // Re-run divergence computation on current grid velocities
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Post-correction Divergence Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Post-correction Divergence Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pressure.divergence_pipeline);
            pass.set_bind_group(0, &self.pressure.divergence_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Read back divergence values
        let diag = self.read_physics_diagnostics(device, queue);
        diag.max_divergence
    }
}
