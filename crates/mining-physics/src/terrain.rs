use super::Hydraulics;
use mining_sim::{
    site::{CELL, WIDTH},
    Session, Site, Vec3,
};
impl Hydraulics {
    fn upload_bed_remainders(&self, s: &Session) -> Vec<[f64; 4]> {
        let area = (CELL as f64).powi(2);
        let mut retained = vec![[0.; 4]; WIDTH * WIDTH];
        let data: Vec<[f32; 4]> = (0..WIDTH * WIDTH)
            .map(|i| {
                std::array::from_fn(|ch| {
                    let volume = s
                        .site
                        .hydraulic
                        .bed_remainder_m3
                        .get(i)
                        .map(|v| v[ch])
                        .unwrap_or(0.);
                    let mut height = (volume / area) as f32;
                    if height as f64 * area > volume {
                        height = height.next_down();
                    }
                    retained[i][ch] = (volume - height as f64 * area).max(0.);
                    height
                })
            })
            .collect();
        self.queue.write_buffer(
            &self.terrain.geology.bed_remainder,
            0,
            bytemuck::cast_slice(&data),
        );
        retained
    }
    pub(super) fn upload_terrain(&self, s: &Session) {
        let h = &self.terrain;
        h.upload_from_world(&self.queue, &s.site.world);
        self.upload_bed_remainders(s);
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
        let zeros;
        let errors = if a.bed_exchange_error.is_empty() {
            zeros = vec![[0f32; 4]; WIDTH * WIDTH];
            &zeros
        } else {
            &a.bed_exchange_error
        };
        self.queue.write_buffer(
            &h.geology.bed_exchange_error,
            0,
            bytemuck::cast_slice(errors),
        );
    }
    pub(super) fn step_terrain(&mut self, s: &mut Session, dt: f32) -> Result<(), String> {
        let h = &self.terrain;
        // Excavation/deposition and the pump edit the most recent readback. Only
        // geology/depth are uploaded here: preserve momentum and every separate
        // suspension channel. Full upload_from_world would silently zero three.
        let added = s.site.add_clean_inflow(
            Vec3::new(Site::river_center(30.5), 0.0, 30.5),
            0.12,
            0.8,
            dt,
        );
        s.water_balance.external_inflow_m3 += added;
        let retained_bed_precision = self.upload_bed_remainders(s);
        let w = &mut s.site.world;
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
        let area = (CELL as f64).powi(2);
        let previous_paydirt: Vec<f64> = w
            .paydirt_thickness
            .iter()
            .enumerate()
            .map(|(i, h)| {
                *h as f64
                    + s.site
                        .hydraulic
                        .bed_remainder_m3
                        .get(i)
                        .map(|v| v[3] / area)
                        .unwrap_or(0.)
            })
            .collect();
        let max_depth = depth.iter().copied().fold(0.0_f32, f32::max);
        let max_speed = s
            .site
            .hydraulic
            .velocity_x
            .iter()
            .chain(&s.site.hydraulic.velocity_z)
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        let substeps = (dt * (max_speed + (9.81 * max_depth).sqrt()) / (0.45 * CELL))
            .ceil()
            .max(2.0) as u32;
        if substeps > 64 {
            return Err("Terrain water exceeded the supported CFL substep budget".into());
        }
        // River bed descends toward -Z. The other borders are closed banks/upstream.
        let boundaries = if w.params.open_boundaries { 4 } else { 0 };
        h.update_water_params(&self.queue, dt, boundaries, w.params.manning_n, substeps);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Original complete heightfield step"),
            });
        h.dispatch_water_substeps(&mut encoder, substeps);
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
            size: bytes * (buffers.len() as u64 + 8 + 12),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (i, b) in buffers.iter().enumerate() {
            encoder.copy_buffer_to_buffer(b, 0, &staging, i as u64 * bytes, bytes);
        }
        encoder.copy_buffer_to_buffer(
            &h.geology.bed_exchange_error,
            0,
            &staging,
            bytes * buffers.len() as u64,
            bytes * 4,
        );
        encoder.copy_buffer_to_buffer(
            &h.geology.bed_remainder,
            0,
            &staging,
            bytes * (buffers.len() as u64 + 4),
            bytes * 4,
        );
        encoder.copy_buffer_to_buffer(
            &h.water.suspended_sediment_next,
            bytes,
            &staging,
            bytes * (buffers.len() as u64 + 8),
            bytes * 12,
        );
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
        let times: Vec<u32> =
            bytemuck::cast_slice(&mapped[14 * bytes as usize..15 * bytes as usize]).to_vec();
        let errors: Vec<[f32; 4]> =
            bytemuck::cast_slice(&mapped[15 * bytes as usize..19 * bytes as usize]).to_vec();
        let bed_remainders: Vec<[f32; 4]> =
            bytemuck::cast_slice(&mapped[19 * bytes as usize..23 * bytes as usize]).to_vec();
        let aux: Vec<f32> = bytemuck::cast_slice(&mapped[23 * bytes as usize..]).to_vec();
        drop(mapped);
        staging.unmap();
        if arrays
            .iter()
            .flatten()
            .chain(&aux)
            .chain(errors.iter().flatten())
            .chain(bed_remainders.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err("Non-finite original terrain solver output".into());
        }
        if arrays[5].iter().any(|v| *v < 0.) || bed_remainders.iter().flatten().any(|v| *v < 0.) {
            return Err("Negative original terrain water/bed precision output".into());
        }
        // Replace the compatibility World view only after every read succeeds.
        let mut next = || std::mem::take(&mut arrays.remove(0));
        w.bedrock_elevation = next();
        w.paydirt_thickness = next();
        w.gravel_thickness = next();
        w.overburden_thickness = next();
        w.terrain_sediment = next();
        let next_depth = next();
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
        a.bed_exchange_error = errors;
        a.bed_remainder_m3 = bed_remainders
            .iter()
            .zip(retained_bed_precision)
            .map(|(gpu, cpu)| std::array::from_fn(|ch| gpu[ch] as f64 * area + cpu[ch]))
            .collect();
        for (i, cell) in aux.as_chunks::<12>().0.iter().enumerate() {
            a.flux_x[i] = cell[4];
            a.flux_z[i] = cell[5];
            s.water_balance.boundary_outflow_m3 += cell[6] as f64;
            for channel in 0..4 {
                a.exported_material_m3[channel] += cell[8 + channel] as f64;
            }
        }
        for i in 0..next_depth.len() {
            // Collapse/scour may reduce the large layer's precision scale.
            // Fold an oversized small component into geometry before building
            // the water surface, preserving the complete canonical inventory.
            let ground = w.ground_height(i % WIDTH, i / WIDTH).abs();
            let ground_quantum = ground.next_up() as f64 - ground as f64;
            for (ch, layer) in [
                &mut w.terrain_sediment,
                &mut w.overburden_thickness,
                &mut w.gravel_thickness,
                &mut w.paydirt_thickness,
            ]
            .into_iter()
            .enumerate()
            {
                let high = layer[i];
                let low = a.bed_remainder_m3[i][ch];
                let quantum = (high.next_up() as f64 - high as f64).max(ground_quantum) * area;
                if low > quantum {
                    let volume = high as f64 * area + low;
                    let mut height = (volume / area) as f32;
                    if height as f64 * area > volume {
                        height = height.next_down();
                    }
                    layer[i] = height;
                    a.bed_remainder_m3[i][ch] = (volume - height as f64 * area).max(0.);
                }
            }
            let x = i % WIDTH;
            let z = i / WIDTH;
            w.water_flow_x[z * (WIDTH + 1) + x] = a.velocity_x[i];
            w.water_flow_z[z * WIDTH + x] = a.velocity_z[i];
        }
        s.site.project_terrain_water(&next_depth);
        let w = &mut s.site.world;
        let a = &mut s.site.hydraulic;
        // Trace gold is a conservative passive tag of the original paydirt
        // channel in the coarse world. In the sluice, density separation belongs
        // entirely to the original FLIP/DEM solver, not this tracer.
        if a.suspended_gold_g.is_empty() {
            a.suspended_gold_g = vec![0.0; WIDTH * WIDTH];
        }
        for (i, &previous) in previous_paydirt.iter().enumerate() {
            let water_depth = w.water_depth(i % WIDTH, i / WIDTH);
            let change = w.paydirt_thickness[i] as f64 + a.bed_remainder_m3[i][3] / area - previous;
            if change < 0.0 && previous_paydirt[i] > 0.0 {
                let moved = s.site.gold_g[i] * (-change / previous_paydirt[i]).clamp(0.0, 1.0);
                s.site.gold_g[i] -= moved;
                a.suspended_gold_g[i] += moved;
            } else if change > 0.0 {
                let suspended = a.suspended_paydirt.get(i).copied().unwrap_or(0.0) as f64
                    * water_depth as f64
                    + a.suspended_remainder_m3
                        .get(i)
                        .map(|v| v[3] / area)
                        .unwrap_or(0.);
                let fraction = (change / (suspended + change).max(1e-30)).clamp(0.0, 1.0);
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
            let mut out = Vec::with_capacity(5);
            if x + 1 < WIDTH && a.flux_x[i] > 0.0 {
                out.push((Some(i + 1), a.flux_x[i]));
            }
            if x > 0 && a.flux_x[i - 1] < 0.0 {
                out.push((Some(i - 1), -a.flux_x[i - 1]));
            }
            if z + 1 < WIDTH && a.flux_z[i] > 0.0 {
                out.push((Some(i + WIDTH), a.flux_z[i]));
            }
            if z > 0 && a.flux_z[i - WIDTH] < 0.0 {
                out.push((Some(i - WIDTH), -a.flux_z[i - WIDTH]));
            }
            if aux[i * 12 + 6] > 0.0 {
                out.push((None, aux[i * 12 + 6]));
            }
            let total = out.iter().map(|(_, q)| *q as f64).sum::<f64>();
            let fraction =
                (total / (water_depth as f64 * (CELL * CELL) as f64).max(1e-12)).clamp(0.0, 1.0);
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
                if let Some(j) = j {
                    a.suspended_gold_g[j] += moved;
                } else {
                    a.exported_gold_g += moved;
                }
            }
        }
        for i in 0..WIDTH * WIDTH {
            if w.water_depth(i % WIDTH, i / WIDTH) == 0. {
                s.site.gold_g[i] += a.suspended_gold_g[i];
                a.suspended_gold_g[i] = 0.;
            }
        }
        a.steps += 1;
        s.site.revision += 1;
        s.site.reconcile_soil_water();
        Ok(())
    }
}
