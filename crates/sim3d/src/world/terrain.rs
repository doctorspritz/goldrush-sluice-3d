//! Terrain manipulation: excavation, material addition, and collapse.

use super::{ExcavationResult, TerrainMaterial, World};

impl World {
    /// Update terrain collapse based on angle of repose.
    /// Returns true if any material moved.
    pub fn update_terrain_collapse(&mut self) -> bool {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let transfer_rate = self.params.collapse_transfer_rate;
        let max_outflow = self.params.collapse_max_outflow;

        self.collapse_deltas.fill(0.0);

        let neighbors: [(i32, i32, f32); 8] = [
            (1, 0, cell_size),
            (-1, 0, cell_size),
            (0, 1, cell_size),
            (0, -1, cell_size),
            (1, 1, cell_size * std::f32::consts::SQRT_2),
            (1, -1, cell_size * std::f32::consts::SQRT_2),
            (-1, 1, cell_size * std::f32::consts::SQRT_2),
            (-1, -1, cell_size * std::f32::consts::SQRT_2),
        ];

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let h = self.ground_height(x, z);

                let material = self.surface_material(x, z);
                if material == TerrainMaterial::Bedrock {
                    continue;
                }
                let angle_tan = material.angle_of_repose().tan();

                let mut neighbor_transfers: [(usize, f32); 8] = [(0, 0.0); 8];
                let mut transfer_count = 0;
                let mut total_out = 0.0_f32;

                for &(dx, dz, dist) in neighbors.iter() {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;

                    if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                        continue;
                    }

                    let nidx = self.idx(nx as usize, nz as usize);
                    let nh = self.ground_height(nx as usize, nz as usize);
                    let diff = h - nh;
                    let max_diff = angle_tan * dist;

                    if diff > max_diff {
                        let transfer = transfer_rate * (diff - max_diff);
                        neighbor_transfers[transfer_count] = (nidx, transfer);
                        transfer_count += 1;
                        total_out += transfer;
                    }
                }

                if total_out <= 0.0 {
                    continue;
                }

                let sediment_available = self.terrain_sediment[idx];
                let max_out = sediment_available * max_outflow;
                let scale = if total_out > max_out && total_out > 0.0 {
                    max_out / total_out
                } else {
                    1.0
                };

                for i in 0..transfer_count {
                    let (nidx, transfer) = neighbor_transfers[i];
                    let scaled_transfer = transfer * scale;
                    self.collapse_deltas[idx] -= scaled_transfer;
                    self.collapse_deltas[nidx] += scaled_transfer;
                }
            }
        }

        let mut changed = false;
        for idx in 0..self.terrain_sediment.len() {
            let delta = self.collapse_deltas[idx];
            if delta.abs() <= 1e-6 {
                continue;
            }

            let x = idx % self.width;
            let z = idx / self.width;

            let old_sediment = self.terrain_sediment[idx];
            // Calc ground before sediment change, but using current layers
            let current_ground = self.ground_height(x, z);
            let ground_base_layers = current_ground - old_sediment;

            let had_water = self.water_surface[idx] > current_ground + 1e-4;

            let new_sediment = (old_sediment + delta).max(0.0);
            self.terrain_sediment[idx] = new_sediment;
            changed = true;

            let new_ground = ground_base_layers + new_sediment;
            if had_water {
                let delta_ground = new_ground - current_ground;
                self.water_surface[idx] += delta_ground;
            } else {
                self.water_surface[idx] = new_ground;
            }
        }

        changed
    }

    /// Excavate terrain at world position.
    /// Returns list of (position, volume, material) for particle spawning.
    pub fn excavate(
        &mut self,
        world_pos: glam::Vec3,
        radius: f32,
        dig_depth: f32,
    ) -> Vec<ExcavationResult> {
        let mut results = Vec::new();

        let cx = (world_pos.x / self.cell_size) as i32;
        let cz = (world_pos.z / self.cell_size) as i32;
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let r_sq = (radius / self.cell_size).powi(2);
        let cell_area = self.cell_size * self.cell_size;

        for dz in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }

                let x = cx + dx;
                let z = cz + dz;

                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.depth as i32 {
                    continue;
                }

                let x = x as usize;
                let z = z as usize;
                let idx = self.idx(x, z);

                let sed = self.terrain_sediment[idx];

                // Determine starting ground height for water calculation
                let ground_before = self.ground_height(x, z);
                let _had_water = self.water_surface[idx] > ground_before + 1e-4;

                let mut remaining_dig = dig_depth;

                // 1. Remove Sediment
                if remaining_dig > 0.0 && sed > 0.0 {
                    let dug = remaining_dig.min(sed);
                    self.terrain_sediment[idx] -= dug;
                    remaining_dig -= dug;
                    if dug > 0.0 {
                        results.push(ExcavationResult {
                            position: glam::Vec3::new(
                                world_pos.x,
                                ground_before - dug * 0.5,
                                world_pos.z,
                            ),
                            volume: dug * cell_area,
                            material: TerrainMaterial::Sand,
                        });
                    }
                }

                // 2. Remove Overburden
                let ob = self.overburden_thickness[idx];
                if remaining_dig > 0.0 && ob > 0.0 {
                    let dug = remaining_dig.min(ob);
                    self.overburden_thickness[idx] -= dug;
                    remaining_dig -= dug;
                    if dug > 0.0 {
                        results.push(ExcavationResult {
                            position: glam::Vec3::new(
                                world_pos.x,
                                ground_before - sed - dug * 0.5,
                                world_pos.z,
                            ),
                            volume: dug * cell_area,
                            material: TerrainMaterial::Dirt,
                        });
                    }
                }

                // 3. Remove Paydirt
                let pd = self.paydirt_thickness[idx];
                if remaining_dig > 0.0 && pd > 0.0 {
                    let dug = remaining_dig.min(pd);
                    self.paydirt_thickness[idx] -= dug;
                    remaining_dig -= dug;
                    if dug > 0.0 {
                        results.push(ExcavationResult {
                            position: glam::Vec3::new(
                                world_pos.x,
                                ground_before - sed - ob - dug * 0.5,
                                world_pos.z,
                            ),
                            volume: dug * cell_area,
                            material: TerrainMaterial::Gravel,
                        });
                    }
                }

                // 4. Bedrock (Stop or chip?)
                // For now, bedrock is invincible.
                let _ = remaining_dig; // Silence unused warning; ready for bedrock digging

                let ground_after = self.ground_height(x, z);
                // Water surface should never go below ground.
                // If we dug under water, water surface stays (fills the hole).
                // Only clamp if surface would be underground.
                if self.water_surface[idx] < ground_after {
                    self.water_surface[idx] = ground_after;
                }

                if results.last().is_some() {
                    // Hack to return something for particles if needed,
                    // but we are pushing multiple.
                }
            }
        }

        results
    }

    /// Add material to terrain (building berms, dumping).
    pub fn add_material(
        &mut self,
        world_pos: glam::Vec3,
        radius: f32,
        height: f32,
        _material: TerrainMaterial,
    ) {
        let cx = (world_pos.x / self.cell_size) as i32;
        let cz = (world_pos.z / self.cell_size) as i32;
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let r_sq = (radius / self.cell_size).powi(2);

        for dz in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }

                let x = cx + dx;
                let z = cz + dz;

                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.depth as i32 {
                    continue;
                }

                let idx = self.idx(x as usize, z as usize);
                let ground_before = self.ground_height(x as usize, z as usize);
                let had_water = self.water_surface[idx] > ground_before + 1e-4;

                self.terrain_sediment[idx] += height;
                // Material tracking is implicit in sediment now

                let ground_after = self.ground_height(x as usize, z as usize);
                if had_water {
                    let delta = ground_after - ground_before;
                    self.water_surface[idx] += delta;
                } else {
                    self.water_surface[idx] = ground_after;
                }
            }
        }
    }
}
