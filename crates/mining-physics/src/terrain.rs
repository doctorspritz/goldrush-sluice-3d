use super::Hydraulics;
use mining_sim::{
    site::{CELL, WIDTH},
    Session, Site, Vec3,
};
impl Hydraulics {
    pub(super) fn upload_terrain(&self, s: &Session) {
        let h = &self.terrain;
        h.upload_from_world(&self.queue, &s.site.world);
        let a = &s.site.hydraulic;
        for (buffer, data) in [
            (&h.water.velocity_x, &a.velocity_x),
            (&h.water.velocity_z, &a.velocity_z),
            (&h.water.flux_x, &a.flux_x),
            (&h.water.flux_z, &a.flux_z),
            (&h.water.suspended_sediment, &a.suspended_sediment),
            (&h.water.suspended_overburden, &a.suspended_overburden),
            (&h.water.suspended_gravel, &a.suspended_gravel),
            (&h.water.suspended_paydirt, &a.suspended_paydirt),
        ] {
            if !data.is_empty() {
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(data));
            }
        }
        if !a.settling_time.is_empty() {
            self.queue.write_buffer(
                &h.geology.settling_time,
                0,
                bytemuck::cast_slice(&a.settling_time),
            );
        }
    }
    pub(super) fn step_terrain(&mut self, s: &mut Session, dt: f32) -> Result<(), String> {
        let h = &self.terrain;
        let w = &mut s.site.world;
        // Excavation/deposition and the pump edit the most recent readback. Only
        // geology/depth are uploaded here: preserve momentum and every separate
        // suspension channel. Full upload_from_world would silently zero three.
        w.add_inflow(
            Vec3::new(Site::river_center(30.5), 0.0, 30.5),
            0.12,
            0.8,
            dt,
        );
        h.upload_terrain_only(&self.queue, w);
        if s.site.hydraulic.impact_dirty {
            self.queue.write_buffer(
                &h.water.velocity_x,
                0,
                bytemuck::cast_slice(&s.site.hydraulic.velocity_x),
            );
            self.queue.write_buffer(
                &h.water.velocity_z,
                0,
                bytemuck::cast_slice(&s.site.hydraulic.velocity_z),
            );
            for (buffer, data) in [
                (
                    &h.water.suspended_sediment,
                    &s.site.hydraulic.suspended_sediment,
                ),
                (
                    &h.water.suspended_overburden,
                    &s.site.hydraulic.suspended_overburden,
                ),
                (
                    &h.water.suspended_gravel,
                    &s.site.hydraulic.suspended_gravel,
                ),
                (
                    &h.water.suspended_paydirt,
                    &s.site.hydraulic.suspended_paydirt,
                ),
            ] {
                if !data.is_empty() {
                    self.queue
                        .write_buffer(buffer, 0, bytemuck::cast_slice(data));
                }
            }
        }
        let depth: Vec<f32> = (0..WIDTH * WIDTH)
            .map(|i| {
                let ground = w.bedrock_elevation[i]
                    + w.paydirt_thickness[i]
                    + w.gravel_thickness[i]
                    + w.overburden_thickness[i]
                    + w.terrain_sediment[i];
                (w.water_surface[i] - ground).max(0.0)
            })
            .collect();
        self.queue
            .write_buffer(&h.water.depth, 0, bytemuck::cast_slice(&depth));
        let previous_paydirt = w.paydirt_thickness.clone();
        let previous_suspended = s.site.hydraulic.suspended_paydirt.clone();
        h.update_params(&self.queue, dt);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Original complete heightfield step"),
            });
        h.dispatch(&mut encoder, dt);
        // All stages of the original pipeline are dispatched by h.dispatch:
        // surface, flux, depth, settling, erosion, conservative transport, collapse.
        let buffers = [
            &h.geology.bedrock,
            &h.geology.paydirt,
            &h.geology.gravel,
            &h.geology.overburden,
            &h.geology.sediment,
            &h.water.depth,
            &h.water.velocity_x,
            &h.water.velocity_z,
            &h.water.flux_x,
            &h.water.flux_z,
            &h.water.suspended_sediment,
            &h.water.suspended_overburden,
            &h.water.suspended_gravel,
            &h.water.suspended_paydirt,
            &h.geology.settling_time,
        ];
        let bytes = (WIDTH * WIDTH * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complete heightfield readback"),
            size: bytes * buffers.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (i, b) in buffers.iter().enumerate() {
            encoder.copy_buffer_to_buffer(b, 0, &staging, i as u64 * bytes, bytes);
        }
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            match rx.try_recv() {
                Ok(result) => {
                    result.map_err(|e| e.to_string())?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err("Terrain readback disconnected".into())
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if std::time::Instant::now() >= deadline {
                return Err("Terrain readback timed out".into());
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        let mapped = staging.slice(..).get_mapped_range();
        let mut arrays: Vec<Vec<f32>> = mapped
            .chunks(bytes as usize)
            .take(14)
            .map(|b| bytemuck::cast_slice(b).to_vec())
            .collect();
        let times: Vec<u32> = bytemuck::cast_slice(&mapped[14 * bytes as usize..]).to_vec();
        drop(mapped);
        staging.unmap();
        if arrays.iter().flatten().any(|v| !v.is_finite()) {
            return Err("Non-finite original terrain solver output".into());
        }
        // Replace the compatibility World view only after every read succeeds.
        let mut next = || std::mem::take(&mut arrays.remove(0));
        w.bedrock_elevation = next();
        w.paydirt_thickness = next();
        w.gravel_thickness = next();
        w.overburden_thickness = next();
        w.terrain_sediment = next();
        let depth = next();
        let a = &mut s.site.hydraulic;
        a.impact_dirty = false;
        a.velocity_x = next();
        a.velocity_z = next();
        a.flux_x = next();
        a.flux_z = next();
        a.suspended_sediment = next();
        a.suspended_overburden = next();
        a.suspended_gravel = next();
        a.suspended_paydirt = next();
        a.settling_time = times;
        for (i, &water_depth) in depth.iter().enumerate() {
            w.water_surface[i] = w.bedrock_elevation[i]
                + w.paydirt_thickness[i]
                + w.gravel_thickness[i]
                + w.overburden_thickness[i]
                + w.terrain_sediment[i]
                + water_depth;
            w.suspended_sediment[i] = a.suspended_sediment[i]
                + a.suspended_overburden[i]
                + a.suspended_gravel[i]
                + a.suspended_paydirt[i];
            let x = i % WIDTH;
            let z = i / WIDTH;
            w.water_flow_x[z * (WIDTH + 1) + x] = a.velocity_x[i];
            w.water_flow_z[z * WIDTH + x] = a.velocity_z[i];
        }
        // Trace gold is a conservative passive tag of the original paydirt
        // channel in the coarse world. In the sluice, density separation belongs
        // entirely to the original FLIP/DEM solver, not this tracer.
        if a.suspended_gold_g.is_empty() {
            a.suspended_gold_g = vec![0.0; WIDTH * WIDTH];
        }
        for (i, &water_depth) in depth.iter().enumerate() {
            let change = w.paydirt_thickness[i] - previous_paydirt[i];
            if change < 0.0 && previous_paydirt[i] > 0.0 {
                let moved =
                    s.site.gold_g[i] * (-change / previous_paydirt[i]).clamp(0.0, 1.0) as f64;
                s.site.gold_g[i] -= moved;
                a.suspended_gold_g[i] += moved;
            } else if change > 0.0 {
                let suspended = previous_suspended.get(i).copied().unwrap_or(0.0) * water_depth;
                let fraction = (change / (suspended + change).max(1e-9)).clamp(0.0, 1.0) as f64;
                let moved = a.suspended_gold_g[i] * fraction;
                a.suspended_gold_g[i] -= moved;
                s.site.gold_g[i] += moved;
            }
        }
        // Conservative upwind transfer using the actual solver flux and depth.
        let old = a.suspended_gold_g.clone();
        for (i, &water_depth) in depth.iter().enumerate() {
            let x = i % WIDTH;
            let z = i / WIDTH;
            let mut out = Vec::with_capacity(4);
            if x + 1 < WIDTH && a.flux_x[i] > 0.0 {
                out.push((i + 1, a.flux_x[i]));
            }
            if x > 0 && a.flux_x[i - 1] < 0.0 {
                out.push((i - 1, -a.flux_x[i - 1]));
            }
            if z + 1 < WIDTH && a.flux_z[i] > 0.0 {
                out.push((i + WIDTH, a.flux_z[i]));
            }
            if z > 0 && a.flux_z[i - WIDTH] < 0.0 {
                out.push((i - WIDTH, -a.flux_z[i - WIDTH]));
            }
            let total = out.iter().map(|(_, q)| *q as f64).sum::<f64>();
            let fraction = (total * dt as f64
                / (water_depth as f64 * (CELL * CELL) as f64).max(1e-12))
            .clamp(0.0, 1.0);
            let budget = old[i] * fraction;
            a.suspended_gold_g[i] -= budget;
            let mut remaining = budget;
            let count = out.len();
            for (n, (j, q)) in out.into_iter().enumerate() {
                let moved = if n + 1 == count {
                    remaining
                } else {
                    (budget * q as f64 / total.max(1e-12)).min(remaining)
                };
                remaining -= moved;
                a.suspended_gold_g[j] += moved;
            }
        }
        a.steps += 1;
        s.site.revision += 1;
        Ok(())
    }
}
