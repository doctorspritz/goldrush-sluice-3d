//! Unified world simulation: terrain + water + sediment settling.
//!
//! This is the "background" simulation for areas outside active particle zones.
//! Everything is heightfield-based for performance.

use glam::Vec3;

/// World simulation parameters.
#[derive(Clone, Debug)]
pub struct WorldParams {
    /// Angle of repose for dry material (radians).
    pub angle_of_repose: f32,
    /// Fraction of excess height transferred per collapse step.
    pub collapse_transfer_rate: f32,
    /// Maximum fraction of cell height that can flow out per step.
    pub collapse_max_outflow: f32,
    /// Gravity (m/s^2).
    pub gravity: f32,
    /// Water flow damping (0-1, higher = more damping).
    pub water_damping: f32,
    /// Sediment settling velocity (m/s).
    pub settling_velocity: f32,
    /// Bed porosity (0-1, fraction that is void space).
    pub bed_porosity: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            angle_of_repose: 35.0_f32.to_radians(),
            collapse_transfer_rate: 0.35,
            collapse_max_outflow: 0.5,
            gravity: 9.81,
            water_damping: 0.02,
            settling_velocity: 0.01,
            bed_porosity: 0.4,
        }
    }
}

/// Material types for terrain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TerrainMaterial {
    #[default]
    Dirt,
    Gravel,
    Sand,
    Clay,
    Bedrock,
}

impl TerrainMaterial {
    /// Angle of repose for this material (radians).
    pub fn angle_of_repose(self) -> f32 {
        match self {
            TerrainMaterial::Dirt => 35.0_f32.to_radians(),
            TerrainMaterial::Gravel => 38.0_f32.to_radians(),
            TerrainMaterial::Sand => 32.0_f32.to_radians(),
            TerrainMaterial::Clay => 45.0_f32.to_radians(),
            TerrainMaterial::Bedrock => 90.0_f32.to_radians(),
        }
    }

    /// Density relative to water.
    pub fn density(self) -> f32 {
        match self {
            TerrainMaterial::Dirt => 1.8,
            TerrainMaterial::Gravel => 2.0,
            TerrainMaterial::Sand => 1.6,
            TerrainMaterial::Clay => 1.9,
            TerrainMaterial::Bedrock => 2.5,
        }
    }
}

/// Unified world state.
#[derive(Clone, Debug)]
pub struct World {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,

    // Terrain layers
    /// Base terrain height (bedrock, original ground) - only changes via excavation.
    pub terrain_base: Vec<f32>,
    /// Sediment layer height (deposited material) - grows from settling.
    pub terrain_sediment: Vec<f32>,
    /// Material type per cell (affects angle of repose).
    pub terrain_material: Vec<TerrainMaterial>,

    // Water layer
    /// Water surface height (absolute, not depth).
    pub water_surface: Vec<f32>,
    /// Water flow velocity X component (at cell faces).
    pub water_flow_x: Vec<f32>,
    /// Water flow velocity Z component (at cell faces).
    pub water_flow_z: Vec<f32>,
    /// Suspended sediment concentration (volume fraction 0-1).
    pub suspended_sediment: Vec<f32>,

    // Working buffers (avoid allocation in update loop)
    collapse_deltas: Vec<f32>,

    // Parameters
    pub params: WorldParams,
}

impl World {
    /// Create a new world with flat terrain.
    pub fn new(width: usize, depth: usize, cell_size: f32, initial_height: f32) -> Self {
        let cell_count = width * depth;
        let flow_x_count = (width + 1) * depth;
        let flow_z_count = width * (depth + 1);

        Self {
            width,
            depth,
            cell_size,
            terrain_base: vec![initial_height; cell_count],
            terrain_sediment: vec![0.0; cell_count],
            terrain_material: vec![TerrainMaterial::Dirt; cell_count],
            water_surface: vec![0.0; cell_count],
            water_flow_x: vec![0.0; flow_x_count],
            water_flow_z: vec![0.0; flow_z_count],
            suspended_sediment: vec![0.0; cell_count],
            collapse_deltas: vec![0.0; cell_count],
            params: WorldParams::default(),
        }
    }

    /// Cell index from (x, z) coordinates.
    #[inline]
    pub fn idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Flow X index (faces between cells in X direction).
    #[inline]
    fn flow_x_idx(&self, x: usize, z: usize) -> usize {
        z * (self.width + 1) + x
    }

    /// Flow Z index (faces between cells in Z direction).
    #[inline]
    fn flow_z_idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Total ground height (base + sediment).
    #[inline]
    pub fn ground_height(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        self.terrain_base[idx] + self.terrain_sediment[idx]
    }

    /// Water depth at cell.
    #[inline]
    pub fn water_depth(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        (self.water_surface[idx] - self.ground_height(x, z)).max(0.0)
    }

    /// World bounds.
    pub fn world_size(&self) -> Vec3 {
        Vec3::new(
            self.width as f32 * self.cell_size,
            100.0,
            self.depth as f32 * self.cell_size,
        )
    }

    /// Convert world position to cell coordinates.
    pub fn world_to_cell(&self, pos: Vec3) -> Option<(usize, usize)> {
        let x = (pos.x / self.cell_size) as i32;
        let z = (pos.z / self.cell_size) as i32;

        if x >= 0 && x < self.width as i32 && z >= 0 && z < self.depth as i32 {
            Some((x as usize, z as usize))
        } else {
            None
        }
    }

    /// Main update step.
    pub fn update(&mut self, dt: f32) {
        self.update_terrain_collapse();
        self.update_water_flow(dt);
        self.update_sediment_settling(dt);
    }

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

                let material = self.terrain_material[idx];
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

            let old_sediment = self.terrain_sediment[idx];
            let old_ground = self.terrain_base[idx] + old_sediment;
            let had_water = self.water_surface[idx] > old_ground + 1e-4;

            let new_sediment = (old_sediment + delta).max(0.0);
            self.terrain_sediment[idx] = new_sediment;
            changed = true;

            let new_ground = self.terrain_base[idx] + new_sediment;
            if had_water {
                let delta_ground = new_ground - old_ground;
                self.water_surface[idx] += delta_ground;
            } else {
                self.water_surface[idx] = new_ground;
            }
        }

        changed
    }
}

/// Information about material to spawn as particles.
#[derive(Clone, Debug)]
pub struct ExcavationResult {
    pub position: Vec3,
    pub volume: f32,
    pub material: TerrainMaterial,
}

impl World {
    /// Excavate terrain at world position.
    /// Returns list of (position, volume, material) for particle spawning.
    pub fn excavate(
        &mut self,
        world_pos: Vec3,
        radius: f32,
        dig_depth: f32,
    ) -> Vec<ExcavationResult> {
        let mut results = Vec::new();

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

                let x = x as usize;
                let z = z as usize;
                let idx = self.idx(x, z);

                let sed = self.terrain_sediment[idx];
                let base = self.terrain_base[idx];
                let material = self.terrain_material[idx];
                let ground_before = sed + base;
                let had_water = self.water_surface[idx] > ground_before + 1e-4;

                let mut remaining_dig = dig_depth;
                let mut total_dug = 0.0;

                if sed > 0.0 && remaining_dig > 0.0 {
                    let dug = remaining_dig.min(sed);
                    self.terrain_sediment[idx] -= dug;
                    remaining_dig -= dug;
                    total_dug += dug;
                }

                if remaining_dig > 0.0 && material != TerrainMaterial::Bedrock {
                    let dug = remaining_dig.min(base);
                    self.terrain_base[idx] -= dug;
                    total_dug += dug;
                }

                let ground_after = self.terrain_base[idx] + self.terrain_sediment[idx];
                if had_water {
                    let delta = ground_after - ground_before;
                    self.water_surface[idx] += delta;
                } else {
                    self.water_surface[idx] = ground_after;
                }

                if total_dug > 0.0 {
                    let cell_area = self.cell_size * self.cell_size;
                    let volume = total_dug * cell_area;

                    results.push(ExcavationResult {
                        position: Vec3::new(
                            (x as f32 + 0.5) * self.cell_size,
                            self.ground_height(x, z) + 0.1,
                            (z as f32 + 0.5) * self.cell_size,
                        ),
                        volume,
                        material,
                    });
                }
            }
        }

        results
    }

    /// Add material to terrain (building berms, dumping).
    pub fn add_material(
        &mut self,
        world_pos: Vec3,
        radius: f32,
        height: f32,
        material: TerrainMaterial,
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
                let ground_before = self.terrain_base[idx] + self.terrain_sediment[idx];
                let had_water = self.water_surface[idx] > ground_before + 1e-4;

                self.terrain_sediment[idx] += height;
                self.terrain_material[idx] = material;

                let ground_after = self.terrain_base[idx] + self.terrain_sediment[idx];
                if had_water {
                    let delta = ground_after - ground_before;
                    self.water_surface[idx] += delta;
                } else {
                    self.water_surface[idx] = ground_after;
                }
            }
        }
    }

    /// Update water flow using simplified shallow water equations.
    pub fn update_water_flow(&mut self, dt: f32) {
        let g = self.params.gravity;
        let damping = 1.0 - self.params.water_damping;
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let ground = self.ground_height(x, z);
                if self.water_surface[idx] < ground {
                    self.water_surface[idx] = ground;
                }
            }
        }

        let max_velocity = cell_size / dt * 0.25;

        for z in 0..depth {
            for x in 1..width {
                let flow_idx = self.flow_x_idx(x, z);
                let idx_l = self.idx(x - 1, z);
                let idx_r = self.idx(x, z);

                let h_l = self.water_surface[idx_l];
                let h_r = self.water_surface[idx_r];

                let depth_l = self.water_depth(x - 1, z);
                let depth_r = self.water_depth(x, z);

                if depth_l < 0.001 && depth_r < 0.001 {
                    self.water_flow_x[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_l - h_r) / cell_size;
                self.water_flow_x[flow_idx] += g * gradient * dt;
                self.water_flow_x[flow_idx] *= damping;

                self.water_flow_x[flow_idx] =
                    self.water_flow_x[flow_idx].clamp(-max_velocity, max_velocity);
            }

            let left_idx = self.flow_x_idx(0, z);
            let right_idx = self.flow_x_idx(width, z);
            self.water_flow_x[left_idx] = 0.0;
            self.water_flow_x[right_idx] = 0.0;
        }

        for z in 1..depth {
            for x in 0..width {
                let flow_idx = self.flow_z_idx(x, z);
                let idx_b = self.idx(x, z - 1);
                let idx_f = self.idx(x, z);

                let h_b = self.water_surface[idx_b];
                let h_f = self.water_surface[idx_f];

                let depth_b = self.water_depth(x, z - 1);
                let depth_f = self.water_depth(x, z);

                if depth_b < 0.001 && depth_f < 0.001 {
                    self.water_flow_z[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_b - h_f) / cell_size;
                self.water_flow_z[flow_idx] += g * gradient * dt;
                self.water_flow_z[flow_idx] *= damping;

                self.water_flow_z[flow_idx] =
                    self.water_flow_z[flow_idx].clamp(-max_velocity, max_velocity);
            }
        }

        for x in 0..width {
            let back_idx = self.flow_z_idx(x, 0);
            let front_idx = self.flow_z_idx(x, depth);
            self.water_flow_z[back_idx] = 0.0;
            self.water_flow_z[front_idx] = 0.0;
        }

        let cell_area = cell_size * cell_size;

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);

                let depth_here = self.water_depth(x, z);
                let depth_left = if x > 0 { self.water_depth(x - 1, z) } else { depth_here };
                let depth_right = if x + 1 < width {
                    self.water_depth(x + 1, z)
                } else {
                    depth_here
                };
                let depth_back = if z > 0 { self.water_depth(x, z - 1) } else { depth_here };
                let depth_front = if z + 1 < depth {
                    self.water_depth(x, z + 1)
                } else {
                    depth_here
                };

                let flux_left = self.water_flow_x[self.flow_x_idx(x, z)]
                    * 0.5
                    * (depth_left + depth_here);
                let flux_right = self.water_flow_x[self.flow_x_idx(x + 1, z)]
                    * 0.5
                    * (depth_right + depth_here);
                let flux_back = self.water_flow_z[self.flow_z_idx(x, z)]
                    * 0.5
                    * (depth_back + depth_here);
                let flux_front = self.water_flow_z[self.flow_z_idx(x, z + 1)]
                    * 0.5
                    * (depth_front + depth_here);

                let volume_change =
                    (flux_left - flux_right + flux_back - flux_front) * cell_size * dt;
                let height_change = volume_change / cell_area;

                self.water_surface[idx] += height_change;

                let ground = self.ground_height(x, z);
                self.water_surface[idx] = self.water_surface[idx].max(ground);
            }
        }

        self.advect_suspended_sediment(dt);
    }

    fn advect_suspended_sediment(&mut self, dt: f32) {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;

        let mut new_sediment = self.suspended_sediment.clone();

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let water_depth = self.water_depth(x, z);

                if water_depth < 0.001 {
                    new_sediment[idx] = 0.0;
                    continue;
                }

                let vx = (self.water_flow_x[self.flow_x_idx(x, z)]
                    + self.water_flow_x[self.flow_x_idx(x + 1, z)]) * 0.5;
                let vz = (self.water_flow_z[self.flow_z_idx(x, z)]
                    + self.water_flow_z[self.flow_z_idx(x, z + 1)]) * 0.5;

                let mut c = self.suspended_sediment[idx];

                if vx > 0.0 && x > 0 {
                    let c_upwind = self.suspended_sediment[self.idx(x - 1, z)];
                    c -= vx * dt / cell_size * (c - c_upwind);
                } else if vx < 0.0 && x + 1 < width {
                    let c_upwind = self.suspended_sediment[self.idx(x + 1, z)];
                    c -= vx * dt / cell_size * (c_upwind - c);
                }

                if vz > 0.0 && z > 0 {
                    let c_upwind = self.suspended_sediment[self.idx(x, z - 1)];
                    c -= vz * dt / cell_size * (c - c_upwind);
                } else if vz < 0.0 && z + 1 < depth {
                    let c_upwind = self.suspended_sediment[self.idx(x, z + 1)];
                    c -= vz * dt / cell_size * (c_upwind - c);
                }

                new_sediment[idx] = c.max(0.0);
            }
        }

        self.suspended_sediment = new_sediment;
    }

    /// Add water at a position.
    pub fn add_water(&mut self, world_pos: Vec3, volume: f32) {
        if let Some((x, z)) = self.world_to_cell(world_pos) {
            let idx = self.idx(x, z);
            let cell_area = self.cell_size * self.cell_size;
            let height_add = volume / cell_area;

            let ground = self.ground_height(x, z);
            if self.water_surface[idx] < ground {
                self.water_surface[idx] = ground;
            }

            self.water_surface[idx] += height_add;
        }
    }

    /// Add sediment-laden water (from active zone outflow).
    pub fn add_sediment_water(
        &mut self,
        world_pos: Vec3,
        water_volume: f32,
        sediment_volume: f32,
    ) {
        if let Some((x, z)) = self.world_to_cell(world_pos) {
            let idx = self.idx(x, z);
            let cell_area = self.cell_size * self.cell_size;

            let height_add = water_volume / cell_area;
            let ground = self.ground_height(x, z);
            if self.water_surface[idx] < ground {
                self.water_surface[idx] = ground;
            }
            self.water_surface[idx] += height_add;

            let water_depth = self.water_depth(x, z);
            if water_depth > 0.001 {
                let conc_add = sediment_volume / (cell_area * water_depth);
                self.suspended_sediment[idx] = (self.suspended_sediment[idx] + conc_add).min(0.5);
            }
        }
    }

    /// Remove water at a position (pumping, drainage).
    /// Returns actual volume removed.
    pub fn remove_water(&mut self, world_pos: Vec3, max_volume: f32) -> f32 {
        if let Some((x, z)) = self.world_to_cell(world_pos) {
            let idx = self.idx(x, z);
            let cell_area = self.cell_size * self.cell_size;

            let depth = self.water_depth(x, z);
            let available = depth * cell_area;
            let removed = available.min(max_volume);

            let height_remove = removed / cell_area;
            self.water_surface[idx] -= height_remove;

            removed
        } else {
            0.0
        }
    }

    /// Get total water volume in world.
    pub fn total_water_volume(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        let mut total = 0.0;

        for z in 0..self.depth {
            for x in 0..self.width {
                total += self.water_depth(x, z) * cell_area;
            }
        }

        total
    }

    /// Update sediment settling (suspended -> terrain_sediment).
    pub fn update_sediment_settling(&mut self, dt: f32) {
        let settling_velocity = self.params.settling_velocity;
        let bed_porosity = self.params.bed_porosity;
        let cell_area = self.cell_size * self.cell_size;

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);
                let depth = self.water_depth(x, z);

                if depth < 0.001 {
                    self.suspended_sediment[idx] = 0.0;
                    continue;
                }

                let conc = self.suspended_sediment[idx];
                if conc < 0.0001 {
                    continue;
                }

                let settling_height = settling_velocity * dt;
                let settled_fraction = (settling_height / depth).min(1.0);

                let settled_volume = conc * cell_area * depth * settled_fraction;

                self.suspended_sediment[idx] *= 1.0 - settled_fraction;

                let solid_fraction = 1.0 - bed_porosity;
                let bed_height_increase = settled_volume / (cell_area * solid_fraction);
                self.terrain_sediment[idx] += bed_height_increase;

                let max_sediment = self.water_surface[idx] - self.terrain_base[idx];
                if self.terrain_sediment[idx] > max_sediment {
                    self.terrain_sediment[idx] = max_sediment.max(0.0);
                }
            }
        }
    }

    /// Get total deposited sediment volume.
    pub fn total_sediment_volume(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        self.terrain_sediment.iter().sum::<f32>() * cell_area
    }

    /// Get vertex data for terrain mesh rendering.
    pub fn terrain_vertices(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut positions = Vec::with_capacity(self.width * self.depth);
        let mut colors = Vec::with_capacity(self.width * self.depth);

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);
                let height = self.ground_height(x, z);
                let sediment = self.terrain_sediment[idx];

                positions.push([
                    (x as f32 + 0.5) * self.cell_size,
                    height,
                    (z as f32 + 0.5) * self.cell_size,
                ]);

                let sediment_ratio = (sediment / 2.0).min(1.0);
                let base_color = match self.terrain_material[idx] {
                    TerrainMaterial::Dirt => [0.4, 0.3, 0.2],
                    TerrainMaterial::Gravel => [0.5, 0.5, 0.5],
                    TerrainMaterial::Sand => [0.8, 0.7, 0.5],
                    TerrainMaterial::Clay => [0.6, 0.4, 0.3],
                    TerrainMaterial::Bedrock => [0.3, 0.3, 0.35],
                };

                let sediment_color = [0.6, 0.5, 0.4];

                colors.push([
                    base_color[0] * (1.0 - sediment_ratio) + sediment_color[0] * sediment_ratio,
                    base_color[1] * (1.0 - sediment_ratio) + sediment_color[1] * sediment_ratio,
                    base_color[2] * (1.0 - sediment_ratio) + sediment_color[2] * sediment_ratio,
                ]);
            }
        }

        (positions, colors)
    }

    /// Get vertex data for water surface mesh.
    pub fn water_vertices(&self) -> (Vec<[f32; 3]>, Vec<[f32; 4]>) {
        let mut positions = Vec::new();
        let mut colors = Vec::new();

        for z in 0..self.depth {
            for x in 0..self.width {
                let depth = self.water_depth(x, z);
                if depth < 0.01 {
                    continue;
                }

                let idx = self.idx(x, z);
                let height = self.water_surface[idx];
                let turbidity = self.suspended_sediment[idx];

                positions.push([
                    (x as f32 + 0.5) * self.cell_size,
                    height,
                    (z as f32 + 0.5) * self.cell_size,
                ]);

                let alpha = (depth / 2.0).min(0.8);
                let brown = turbidity.min(0.5) * 2.0;

                colors.push([
                    0.2 + brown * 0.4,
                    0.4 + brown * 0.2,
                    0.8 - brown * 0.4,
                    alpha,
                ]);
            }
        }

        (positions, colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_creation() {
        let world = World::new(10, 10, 1.0, 5.0);
        assert_eq!(world.width, 10);
        assert_eq!(world.depth, 10);
        assert!((world.ground_height(5, 5) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_excavation() {
        let mut world = World::new(10, 10, 1.0, 5.0);
        let results = world.excavate(Vec3::new(5.0, 0.0, 5.0), 1.0, 1.0);

        assert!(!results.is_empty());
        assert!(world.ground_height(5, 5) < 5.0);
    }

    #[test]
    fn test_terrain_collapse() {
        let mut world = World::new(10, 10, 1.0, 0.0);

        world.terrain_sediment[world.idx(5, 5)] = 10.0;

        for _ in 0..100 {
            world.update_terrain_collapse();
        }

        let center_height = world.terrain_sediment[world.idx(5, 5)];
        let neighbor_height = world.terrain_sediment[world.idx(5, 6)];

        assert!(center_height < 10.0);
        assert!(neighbor_height > 0.0);
    }

    #[test]
    fn test_water_leveling() {
        let mut world = World::new(10, 10, 1.0, 0.0);

        world.add_water(Vec3::new(1.0, 0.0, 1.0), 100.0);

        let initial_total = world.total_water_volume();

        for _ in 0..1000 {
            world.update_water_flow(0.016);
        }

        let final_total = world.total_water_volume();

        assert!((initial_total - final_total).abs() < 1.0);

        let depth_00 = world.water_depth(0, 0);
        let depth_99 = world.water_depth(9, 9);
        assert!((depth_00 - depth_99).abs() < 0.5);
    }

    #[test]
    fn test_sediment_settling() {
        let mut world = World::new(10, 10, 1.0, 0.0);

        world.add_sediment_water(Vec3::new(5.0, 0.0, 5.0), 10.0, 1.0);

        let initial_suspended = world.suspended_sediment[world.idx(5, 5)];
        assert!(initial_suspended > 0.0);

        for _ in 0..1000 {
            world.update_sediment_settling(0.016);
        }

        let final_suspended = world.suspended_sediment[world.idx(5, 5)];
        let final_bed = world.terrain_sediment[world.idx(5, 5)];

        assert!(final_suspended < initial_suspended);
        assert!(final_bed > 0.0);
    }
}
