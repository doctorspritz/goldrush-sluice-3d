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
    /// Water flow damping (0-1, higher = more damping). Deprecated: use manning_n instead.
    pub water_damping: f32,
    /// Manning roughness coefficient (typical: 0.03 for smooth, 0.05 for rough).
    pub manning_n: f32,
    /// Sediment settling velocity (m/s).
    pub settling_velocity: f32,
    /// Bed porosity (0-1, fraction that is void space).
    pub bed_porosity: f32,
    /// Hardness of overburden (erosion resistance multiplier).
    pub hardness_overburden: f32,
    /// Hardness of paydirt (erosion resistance multiplier).
    pub hardness_paydirt: f32,
    /// Hardness of deposited sediment (erosion resistance multiplier).
    pub hardness_sediment: f32,
    /// Hardness of gravel layer (erosion resistance multiplier).
    pub hardness_gravel: f32,
    /// Open boundaries - if true, water drains at edges. If false, closed system.
    pub open_boundaries: bool,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            angle_of_repose: 35.0_f32.to_radians(),
            collapse_transfer_rate: 0.35,
            collapse_max_outflow: 0.5,
            gravity: 9.81,
            water_damping: 0.0, // Deprecated, use manning_n
            manning_n: 0.03, // Smooth channel roughness
            settling_velocity: 0.01,
            bed_porosity: 0.4,
            hardness_overburden: 1.0,
            hardness_paydirt: 5.0,
            hardness_sediment: 0.5, // Loose sediment easier than overburden
            hardness_gravel: 2.0,   // Between overburden (1.0) and paydirt (5.0)
            open_boundaries: true, // Default to open for simulation
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

/// A higher-resolution region for adaptive LOD simulation.
/// Covers a rectangular area of the coarse grid at finer resolution.
#[derive(Clone, Debug)]
pub struct FineRegion {
    /// Bounds in coarse cell coordinates (inclusive).
    pub coarse_x_min: usize,
    pub coarse_z_min: usize,
    pub coarse_x_max: usize,
    pub coarse_z_max: usize,

    /// Subdivision factor (e.g., 4 = 4x4 fine cells per coarse cell).
    pub scale: usize,

    /// Fine grid dimensions.
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,

    // Terrain layers (interpolated from coarse, then simulated)
    pub bedrock_elevation: Vec<f32>,
    pub paydirt_thickness: Vec<f32>,
    pub gravel_thickness: Vec<f32>,
    pub overburden_thickness: Vec<f32>,
    pub terrain_sediment: Vec<f32>,

    // Water state
    pub water_surface: Vec<f32>,
    pub water_flow_x: Vec<f32>,
    pub water_flow_z: Vec<f32>,
    pub suspended_sediment: Vec<f32>,

    // Working buffers
    collapse_deltas: Vec<f32>,
    advection_mass_buffer: Vec<f32>,
    advection_delta_buffer: Vec<f32>,
    advection_outflow_buffer: Vec<f32>,
}

impl FineRegion {
    /// Create a new fine region covering the specified coarse cells.
    pub fn new(
        coarse_x_min: usize,
        coarse_z_min: usize,
        coarse_x_max: usize,
        coarse_z_max: usize,
        scale: usize,
        coarse_cell_size: f32,
    ) -> Self {
        let coarse_width = coarse_x_max - coarse_x_min + 1;
        let coarse_depth = coarse_z_max - coarse_z_min + 1;

        let width = coarse_width * scale;
        let depth = coarse_depth * scale;
        let cell_size = coarse_cell_size / scale as f32;

        let cell_count = width * depth;
        let flow_x_count = (width + 1) * depth;
        let flow_z_count = width * (depth + 1);

        Self {
            coarse_x_min,
            coarse_z_min,
            coarse_x_max,
            coarse_z_max,
            scale,
            width,
            depth,
            cell_size,

            bedrock_elevation: vec![0.0; cell_count],
            paydirt_thickness: vec![0.0; cell_count],
            gravel_thickness: vec![0.0; cell_count],
            overburden_thickness: vec![0.0; cell_count],
            terrain_sediment: vec![0.0; cell_count],

            water_surface: vec![0.0; cell_count],
            water_flow_x: vec![0.0; flow_x_count],
            water_flow_z: vec![0.0; flow_z_count],
            suspended_sediment: vec![0.0; cell_count],

            collapse_deltas: vec![0.0; cell_count],
            advection_mass_buffer: vec![0.0; cell_count],
            advection_delta_buffer: vec![0.0; cell_count],
            advection_outflow_buffer: vec![0.0; cell_count],
        }
    }

    /// Cell index from local (x, z) coordinates.
    #[inline]
    pub fn idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Flow X index.
    #[inline]
    pub fn flow_x_idx(&self, x: usize, z: usize) -> usize {
        z * (self.width + 1) + x
    }

    /// Flow Z index.
    #[inline]
    pub fn flow_z_idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Total ground height at local coordinates.
    #[inline]
    pub fn ground_height(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        self.bedrock_elevation[idx]
            + self.paydirt_thickness[idx]
            + self.gravel_thickness[idx]
            + self.overburden_thickness[idx]
            + self.terrain_sediment[idx]
    }

    /// Water depth at local coordinates.
    #[inline]
    pub fn water_depth(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        (self.water_surface[idx] - self.ground_height(x, z)).max(0.0)
    }

    /// World position of fine cell origin.
    pub fn world_origin(&self, coarse_cell_size: f32) -> Vec3 {
        Vec3::new(
            self.coarse_x_min as f32 * coarse_cell_size,
            0.0,
            self.coarse_z_min as f32 * coarse_cell_size,
        )
    }

    /// Convert world position to local fine cell coordinates.
    pub fn world_to_local(&self, pos: Vec3, coarse_cell_size: f32) -> Option<(usize, usize)> {
        let origin = self.world_origin(coarse_cell_size);
        let local_x = (pos.x - origin.x) / self.cell_size;
        let local_z = (pos.z - origin.z) / self.cell_size;

        if local_x >= 0.0
            && local_x < self.width as f32
            && local_z >= 0.0
            && local_z < self.depth as f32
        {
            Some((local_x as usize, local_z as usize))
        } else {
            None
        }
    }

    /// Check if world position is within this fine region.
    pub fn contains_world_pos(&self, pos: Vec3, coarse_cell_size: f32) -> bool {
        self.world_to_local(pos, coarse_cell_size).is_some()
    }

    /// Get the surface material at local coordinates.
    pub fn surface_material(&self, x: usize, z: usize) -> TerrainMaterial {
        let idx = self.idx(x, z);
        if self.terrain_sediment[idx] > 0.001 {
            TerrainMaterial::Sand
        } else if self.overburden_thickness[idx] > 0.001 {
            TerrainMaterial::Dirt
        } else if self.gravel_thickness[idx] > 0.001 {
            TerrainMaterial::Gravel
        } else if self.paydirt_thickness[idx] > 0.001 {
            TerrainMaterial::Clay
        } else {
            TerrainMaterial::Bedrock
        }
    }

    /// Update terrain collapse (angle of repose) for fine region.
    pub fn update_collapse(&mut self, transfer_rate: f32, max_outflow: f32) -> bool {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;

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

        for z in 1..depth - 1 {
            for x in 1..width - 1 {
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
                    let nx = (x as i32 + dx) as usize;
                    let nz = (z as i32 + dz) as usize;

                    if nx >= width || nz >= depth {
                        continue;
                    }

                    let nidx = self.idx(nx, nz);
                    let nh = self.ground_height(nx, nz);
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

        // Apply deltas
        let mut changed = false;
        for z in 1..depth - 1 {
            for x in 1..width - 1 {
                let idx = self.idx(x, z);
                let delta = self.collapse_deltas[idx];
                if delta.abs() <= 1e-6 {
                    continue;
                }

                let old_sediment = self.terrain_sediment[idx];
                let current_ground = self.ground_height(x, z);
                let ground_base = current_ground - old_sediment;

                let had_water = self.water_surface[idx] > current_ground + 1e-4;

                let new_sediment = (old_sediment + delta).max(0.0);
                self.terrain_sediment[idx] = new_sediment;
                changed = true;

                let new_ground = ground_base + new_sediment;
                if had_water {
                    let delta_ground = new_ground - current_ground;
                    self.water_surface[idx] += delta_ground;
                } else {
                    self.water_surface[idx] = new_ground;
                }
            }
        }

        changed
    }

    /// Update erosion and sediment transport for fine region.
    pub fn update_erosion(&mut self, dt: f32, hardness_overburden: f32, hardness_paydirt: f32, hardness_sediment: f32, hardness_gravel: f32) {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let cell_area = cell_size * cell_size;

        let k_entrain = 2.0;
        let k_deposit = 0.5;

        for z in 1..depth - 1 {
            for x in 1..width - 1 {
                let idx = self.idx(x, z);

                // Calculate water velocity magnitude
                let flow_x_left = self.water_flow_x[self.flow_x_idx(x, z)];
                let flow_x_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
                let flow_z_up = self.water_flow_z[self.flow_z_idx(x, z)];
                let flow_z_down = self.water_flow_z[self.flow_z_idx(x, z + 1)];

                let vel_x = (flow_x_left + flow_x_right) * 0.5;
                let vel_z = (flow_z_up + flow_z_down) * 0.5;
                let speed = (vel_x * vel_x + vel_z * vel_z).sqrt();

                let water_depth = self.water_depth(x, z);
                if water_depth < 0.01 {
                    continue;
                }

                // Transport capacity
                let transport_capacity = speed * speed * water_depth * 0.5;

                let suspended_conc = self.suspended_sediment[idx];
                let suspended_vol = suspended_conc * water_depth * cell_area;

                if suspended_vol > transport_capacity {
                    // Deposition
                    let deposit_vol = (suspended_vol - transport_capacity) * k_deposit * dt;
                    let deposit_vol = deposit_vol.min(suspended_vol);
                    let deposit_height = (deposit_vol / cell_area).min(1.0 * dt);
                    let actual_deposit_vol = deposit_height * cell_area;

                    self.terrain_sediment[idx] += deposit_height;

                    let new_suspended_vol = suspended_vol - actual_deposit_vol;
                    if water_depth > 1e-4 {
                        self.suspended_sediment[idx] = new_suspended_vol / (water_depth * cell_area);
                    } else {
                        self.suspended_sediment[idx] = 0.0;
                    }
                } else {
                    // Calculate critical velocity based on next material to erode
                    let critical_velocity = if self.terrain_sediment[idx] > 0.001 {
                        0.3 // Soft sediment
                    } else if self.gravel_thickness[idx] > 0.001 {
                        0.8 // Gravel
                    } else if self.overburden_thickness[idx] > 0.001 {
                        0.5 // Overburden
                    } else {
                        1.2 // Paydirt (very hard)
                    };

                    if speed > critical_velocity {
                        // Erosion (only if fast enough)
                        let deficit_vol = transport_capacity - suspended_vol;
                        let entrain_target_vol = deficit_vol * k_entrain * dt;
                        let entrain_target_height = (entrain_target_vol / cell_area).min(0.5 * dt);

                        let mut remaining_demand = entrain_target_height;
                        let mut total_eroded_vol = 0.0;

                        // Erode sediment first
                        if remaining_demand > 0.0 {
                            let available = self.terrain_sediment[idx];
                            let resistance = hardness_sediment;
                            let take = (remaining_demand / resistance.max(0.1)).min(available);
                            self.terrain_sediment[idx] -= take;
                            total_eroded_vol += take * cell_area;
                            remaining_demand -= take * resistance;
                        }

                        // Erode gravel
                        if remaining_demand > 0.0 {
                            let available = self.gravel_thickness[idx];
                            let resistance = hardness_gravel;
                            let take = (remaining_demand / resistance.max(1.0)).min(available);
                            self.gravel_thickness[idx] -= take;
                            total_eroded_vol += take * cell_area;
                            remaining_demand -= take * resistance;
                        }

                        // Erode overburden
                        if remaining_demand > 0.0 {
                            let available = self.overburden_thickness[idx];
                            let take = (remaining_demand / hardness_overburden.max(1.0)).min(available);
                            self.overburden_thickness[idx] -= take;
                            total_eroded_vol += take * cell_area;
                            remaining_demand -= take * hardness_overburden;
                        }

                        // Erode paydirt
                        if remaining_demand > 0.0 {
                            let available = self.paydirt_thickness[idx];
                            let take = (remaining_demand / hardness_paydirt.max(1.0)).min(available);
                            self.paydirt_thickness[idx] -= take;
                            total_eroded_vol += take * cell_area;
                        }

                        // Add to suspended
                        let new_suspended_vol = suspended_vol + total_eroded_vol;
                        if water_depth > 1e-4 {
                            self.suspended_sediment[idx] = new_suspended_vol / (water_depth * cell_area);
                        }
                    }
                }
            }
        }
    }
}

/// Unified world state.
#[derive(Clone, Debug)]
pub struct World {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,

    /// Optional fine-resolution region for adaptive LOD.
    pub fine_region: Option<FineRegion>,

    // Geological Layers
    pub bedrock_elevation: Vec<f32>, // The hard floor (Base height)
    pub paydirt_thickness: Vec<f32>, // Gold-bearing layer
    pub gravel_thickness: Vec<f32>,  // Gravel layer (resistant to erosion)
    pub overburden_thickness: Vec<f32>, // Soil/Dirt layer (easy to erode)

    // Sediment (Transient)
    pub terrain_sediment: Vec<f32>,

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
    advection_mass_buffer: Vec<f32>,
    advection_delta_buffer: Vec<f32>,
    advection_outflow_buffer: Vec<f32>,

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
            fine_region: None,

            // Default Geology:
            // 50% Bedrock (Deep base)
            // 25% Paydirt (Gold bearing layer)
            // 5% Gravel (Resistant layer)
            // 20% Overburden (Top soil)
            bedrock_elevation: vec![initial_height * 0.5; cell_count],
            paydirt_thickness: vec![initial_height * 0.25; cell_count],
            gravel_thickness: vec![initial_height * 0.05; cell_count],
            overburden_thickness: vec![initial_height * 0.2; cell_count],

            terrain_sediment: vec![0.0; cell_count],

            water_surface: vec![0.0; cell_count],
            water_flow_x: vec![0.0; flow_x_count],
            water_flow_z: vec![0.0; flow_z_count],
            suspended_sediment: vec![0.0; cell_count],
            collapse_deltas: vec![0.0; cell_count],
            advection_mass_buffer: vec![0.0; cell_count],
            advection_delta_buffer: vec![0.0; cell_count],
            advection_outflow_buffer: vec![0.0; cell_count],
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
    pub fn flow_x_idx(&self, x: usize, z: usize) -> usize {
        z * (self.width + 1) + x
    }

    /// Flow Z index (faces between cells in Z direction).
    #[inline]
    pub fn flow_z_idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Total ground height (base + all layers).
    #[inline]
    pub fn ground_height(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        self.bedrock_elevation[idx]
            + self.paydirt_thickness[idx]
            + self.gravel_thickness[idx]
            + self.overburden_thickness[idx]
            + self.terrain_sediment[idx]
    }

    #[inline]
    pub fn surface_material(&self, x: usize, z: usize) -> TerrainMaterial {
        let idx = self.idx(x, z);
        if self.terrain_sediment[idx] > 0.01 {
            return TerrainMaterial::Sand; // Or generic sediment
        }
        if self.overburden_thickness[idx] > 0.01 {
            return TerrainMaterial::Dirt;
        }
        if self.paydirt_thickness[idx] > 0.01 {
            return TerrainMaterial::Gravel;
        }
        TerrainMaterial::Bedrock
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
        // Coarse grid simulation
        self.update_terrain_collapse();
        self.update_water_flow(dt);
        self.update_sediment_advection(dt);
        self.update_erosion(dt);

        // Fine region simulation (if active)
        self.update_fine_region(dt);
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
                            position: Vec3::new(
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
                            position: Vec3::new(
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
                            position: Vec3::new(
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
        world_pos: Vec3,
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

    /// Add water inflow (source) to the terrain.
    pub fn add_inflow(&mut self, world_pos: Vec3, rate: f32, radius: f32, dt: f32) {
        let volume_per_step = rate * dt;
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        let r_sq = (radius / self.cell_size).powi(2);

        // Count valid cells to distribute evenly
        let mut count = 0;
        let cx = (world_pos.x / self.cell_size) as i32;
        let cz = (world_pos.z / self.cell_size) as i32;

        // First pass: count
        for dz in -cell_radius..=cell_radius {
            for dx in -cell_radius..=cell_radius {
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }
                let x = cx + dx;
                let z = cz + dz;
                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.depth as i32 {
                    continue;
                }
                count += 1;
            }
        }

        if count == 0 {
            return;
        }
        let vol_per_cell = volume_per_step / count as f32;
        let height_increase = vol_per_cell / (self.cell_size * self.cell_size);

        // Second pass: apply
        for dz in -cell_radius..=cell_radius {
            for dx in -cell_radius..=cell_radius {
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
                let ground = self.ground_height(x as usize, z as usize);

                // Ensure surface is at least ground
                if self.water_surface[idx] < ground {
                    self.water_surface[idx] = ground;
                }

                self.water_surface[idx] += height_increase;
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
        let cell_area = cell_size * cell_size;

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let ground = self.ground_height(x, z);
                if self.water_surface[idx] < ground {
                    self.water_surface[idx] = ground;
                }
            }
        }

        let max_velocity = cell_size / dt * 0.5;

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

        // Flux limiting: ensure no cell outputs more volume than it has
        self.collapse_deltas.fill(1.0); // Reuse buffer for limits

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let depth_here = self.water_depth(x, z);

                // if depth_here < 1e-4 { continue; } // Unsafe optimization removed

                let available_volume = depth_here * cell_area;
                let mut total_outflow_volume = 0.0;

                // Right (x+1)
                let v_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
                if v_right > 0.0 {
                    let depth_right = if x + 1 < width {
                        self.water_depth(x + 1, z)
                    } else {
                        depth_here
                    };
                    let flux = v_right * 0.5 * (depth_here + depth_right) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                // Left (x)
                let v_left = self.water_flow_x[self.flow_x_idx(x, z)];
                if v_left < 0.0 {
                    let depth_left = if x > 0 {
                        self.water_depth(x - 1, z)
                    } else {
                        depth_here
                    };
                    let flux = -v_left * 0.5 * (depth_left + depth_here) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                // Front (z+1)
                let v_front = self.water_flow_z[self.flow_z_idx(x, z + 1)];
                if v_front > 0.0 {
                    let depth_front = if z + 1 < depth {
                        self.water_depth(x, z + 1)
                    } else {
                        depth_here
                    };
                    let flux = v_front * 0.5 * (depth_here + depth_front) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                // Back (z)
                let v_back = self.water_flow_z[self.flow_z_idx(x, z)];
                if v_back < 0.0 {
                    let depth_back = if z > 0 {
                        self.water_depth(x, z - 1)
                    } else {
                        depth_here
                    };
                    let flux = -v_back * 0.5 * (depth_back + depth_here) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                if total_outflow_volume > available_volume {
                    self.collapse_deltas[idx] = available_volume / total_outflow_volume;
                }
            }
        }

        // Apply Limiters to Flow
        for z in 0..depth {
            for x in 0..width + 1 {
                // Flow X faces
                let flow_idx = self.flow_x_idx(x, z);
                let vel = self.water_flow_x[flow_idx];
                if vel > 0.0 && x > 0 {
                    let donor_idx = self.idx(x - 1, z);
                    self.water_flow_x[flow_idx] *= self.collapse_deltas[donor_idx];
                } else if vel < 0.0 && x < width {
                    let donor_idx = self.idx(x, z);
                    self.water_flow_x[flow_idx] *= self.collapse_deltas[donor_idx];
                }
            }
        }
        for z in 0..depth + 1 {
            // Flow Z faces
            for x in 0..width {
                let flow_idx = self.flow_z_idx(x, z);
                let vel = self.water_flow_z[flow_idx];
                if vel > 0.0 && z > 0 {
                    let donor_idx = self.idx(x, z - 1);
                    self.water_flow_z[flow_idx] *= self.collapse_deltas[donor_idx];
                } else if vel < 0.0 && z < depth {
                    let donor_idx = self.idx(x, z);
                    self.water_flow_z[flow_idx] *= self.collapse_deltas[donor_idx];
                }
            }
        }

        // 5. Volume Update
        // Reuse collapse_deltas for height changes to verify conservation (avoid race condition)
        self.collapse_deltas.fill(0.0);

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);

                let depth_here = self.water_depth(x, z);
                let depth_left = if x > 0 {
                    self.water_depth(x - 1, z)
                } else {
                    depth_here
                };
                let depth_right = if x + 1 < width {
                    self.water_depth(x + 1, z)
                } else {
                    depth_here
                };
                let depth_back = if z > 0 {
                    self.water_depth(x, z - 1)
                } else {
                    depth_here
                };
                let depth_front = if z + 1 < depth {
                    self.water_depth(x, z + 1)
                } else {
                    depth_here
                };

                let flux_left =
                    self.water_flow_x[self.flow_x_idx(x, z)] * 0.5 * (depth_left + depth_here);
                let flux_right =
                    self.water_flow_x[self.flow_x_idx(x + 1, z)] * 0.5 * (depth_right + depth_here);
                let flux_back =
                    self.water_flow_z[self.flow_z_idx(x, z)] * 0.5 * (depth_back + depth_here);
                let flux_front =
                    self.water_flow_z[self.flow_z_idx(x, z + 1)] * 0.5 * (depth_front + depth_here);

                let volume_change =
                    (flux_left - flux_right + flux_back - flux_front) * cell_size * dt;

                let height_change = volume_change / cell_area;
                self.collapse_deltas[idx] = height_change;
            }
        }

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                self.water_surface[idx] += self.collapse_deltas[idx];
                let ground = self.ground_height(x, z);
                self.water_surface[idx] = self.water_surface[idx].max(ground);
            }
        }

        // Open Boundary Condition: Edges are sinks (water flow off map)
        if self.params.open_boundaries {
            for z in 0..depth {
                let idx_left = self.idx(0, z);
                let idx_right = self.idx(width - 1, z);
                self.water_surface[idx_left] = self.ground_height(0, z);
                self.water_surface[idx_right] = self.ground_height(width - 1, z);
            }
            for x in 0..width {
                let idx_back = self.idx(x, 0);
                let idx_front = self.idx(x, depth - 1);
                self.water_surface[idx_back] = self.ground_height(x, 0);
                self.water_surface[idx_front] = self.ground_height(x, depth - 1);
            }
        }
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
    pub fn add_sediment_water(&mut self, world_pos: Vec3, water_volume: f32, sediment_volume: f32) {
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

                let base_height = self.bedrock_elevation[idx]
                    + self.paydirt_thickness[idx]
                    + self.overburden_thickness[idx];
                let max_sediment = self.water_surface[idx] - base_height;
                if self.terrain_sediment[idx] > max_sediment {
                    self.terrain_sediment[idx] = max_sediment.max(0.0);
                }
            }
        }
    }
    /// Advect suspended sediment using water flow.
    pub fn update_sediment_advection(&mut self, dt: f32) {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let cell_area = cell_size * cell_size;

        // Take buffers out to avoid borrow checker conflicts with self methods
        let mut mass_buffer = std::mem::take(&mut self.advection_mass_buffer);
        let mut delta_buffer = std::mem::take(&mut self.advection_delta_buffer);
        // We need a 'total outflow' buffer for limiting
        let mut outflow_buffer = std::mem::take(&mut self.advection_outflow_buffer);

        // Ensure size
        if mass_buffer.len() != width * depth {
            mass_buffer.resize(width * depth, 0.0);
        }
        if delta_buffer.len() != width * depth {
            delta_buffer.resize(width * depth, 0.0);
        }
        if outflow_buffer.len() != width * depth {
            outflow_buffer.resize(width * depth, 0.0);
        }

        // 1. Calculate current mass in each cell
        mass_buffer.fill(0.0);

        for i in 0..width * depth {
            let (x, z) = (i % width, i / width);
            let d = self.water_depth(x, z);
            // Use tiny threshold to avoid processing dry cells
            if d > 0.001 && self.suspended_sediment[i] > 0.0001 {
                mass_buffer[i] = self.suspended_sediment[i] * d * cell_area;
            }
        }

        delta_buffer.fill(0.0);

        // 2. Accumulate Desired Fluxes
        // Since we can't modify delta directly without knowing total, we store FLUXES first?
        // Or simpler: Compute all fluxes, accumulate "Total Outflow" per cell, then scale.

        // It's a staggered grid. We need to store fluxes at faces to apply them later.
        // But we don't have a face-buffer.
        // Allocating face buffers every frame is expensive (which caused the lag before).
        // Optimization: Single Pass with "Safe" flux?
        // Or: Just use the 'velocity' stored in `water_flow`? Yes.

        // Stability loop:
        // A. Calc max time step or limit velocities?
        // B. Calculate total potential outflow for each cell based on velocity field.

        outflow_buffer.fill(0.0);

        // X-Outflows
        for z in 0..depth {
            for x in 0..=width {
                let flow_idx = self.flow_x_idx(x, z);
                if flow_idx >= self.water_flow_x.len() {
                    continue;
                }
                let vel = self.water_flow_x[flow_idx];
                if vel.abs() < 1e-4 {
                    continue;
                }

                // If vel > 0, outflow from (x-1, z)
                if vel > 0.0 {
                    if x > 0 {
                        // Depth at face? Upwind depth.
                        let up_idx = self.idx(x - 1, z);
                        let d = self.water_depth(x - 1, z);
                        let face_area = d * cell_size;
                        let vol_flux = vel * face_area * dt;
                        outflow_buffer[up_idx] += vol_flux;
                    }
                } else {
                    // vel < 0, outflow from (x, z)
                    if x < width {
                        let up_idx = self.idx(x, z);
                        let d = self.water_depth(x, z);
                        let face_area = d * cell_size;
                        let vol_flux = vel.abs() * face_area * dt;
                        outflow_buffer[up_idx] += vol_flux;
                    }
                }
            }
        }
        // Z-Outflows
        for x in 0..width {
            for z in 0..=depth {
                let flow_idx = self.flow_z_idx(x, z);
                if flow_idx >= self.water_flow_z.len() {
                    continue;
                }
                let vel = self.water_flow_z[flow_idx];
                if vel.abs() < 1e-4 {
                    continue;
                }

                if vel > 0.0 {
                    // Outflow from z-1
                    if z > 0 {
                        let up_idx = self.idx(x, z - 1);
                        let d = self.water_depth(x, z - 1);
                        let face_area = d * cell_size;
                        let vol_flux = vel * face_area * dt;
                        outflow_buffer[up_idx] += vol_flux;
                    }
                } else {
                    // Outflow from z
                    if z < depth {
                        let up_idx = self.idx(x, z);
                        let d = self.water_depth(x, z);
                        let face_area = d * cell_size;
                        let vol_flux = vel.abs() * face_area * dt;
                        outflow_buffer[up_idx] += vol_flux;
                    }
                }
            }
        }

        // 3. Apply Fluxes with Limiting
        // Now revisit faces and move mass, scaled by `min(1.0, vol_available / vol_outflow)`

        // X-Fluxes
        for z in 0..depth {
            for x in 0..=width {
                let flow_idx = self.flow_x_idx(x, z);
                if flow_idx >= self.water_flow_x.len() {
                    continue;
                }
                let vel = self.water_flow_x[flow_idx];
                if vel.abs() < 1e-4 {
                    continue;
                }

                let (src_idx, dst_idx, flux_vol) = if vel > 0.0 {
                    if x == 0 {
                        continue;
                    }
                    let src = self.idx(x - 1, z);
                    let d = self.water_depth(x - 1, z); // Re-access is slow, but safe
                    (
                        src,
                        if x < width {
                            Some(self.idx(x, z))
                        } else {
                            None
                        },
                        vel * d * cell_size * dt,
                    )
                } else {
                    if x == width {
                        continue;
                    }
                    let src = self.idx(x, z);
                    let d = self.water_depth(x, z);
                    (
                        src,
                        if x > 0 {
                            Some(self.idx(x - 1, z))
                        } else {
                            None
                        },
                        vel.abs() * d * cell_size * dt,
                    )
                };

                // Limiter
                let total_out = outflow_buffer[src_idx];
                let cell_vol = self.water_depth(src_idx % width, src_idx / width) * cell_area; // Approx

                let scale = if total_out > cell_vol {
                    cell_vol / total_out  // Limit all fluxes proportionally
                } else {
                    1.0  // No limiting needed
                };

                // Mass Flux = Volume Flux * Concentration * Scale
                // Conc = Mass / Vol
                // Flux Mass = (Flux Vol * Scale) * (Mass / Vol)
                //           = (Flux Vol / Vol) * Scale * Mass
                // If scale == Vol / TotalOut, then Flux Mass = (Flux Vol / Total Out) * Mass.
                // This ensures Sum(Flux Mass) <= Mass. Perfect.

                let src_mass = mass_buffer[src_idx];
                if src_mass < 1e-6 {
                    continue;
                }

                let actual_mass_moved = if cell_vol > 1e-9 {
                    (flux_vol * scale / cell_vol) * src_mass  // flux_vol*scale is actual volume moved, divide by cell_vol to get fraction of mass
                } else {
                    0.0
                };

                delta_buffer[src_idx] -= actual_mass_moved;
                if let Some(dst) = dst_idx {
                    delta_buffer[dst] += actual_mass_moved;
                }
            }
        }

        // Z-Fluxes
        for x in 0..width {
            for z in 0..=depth {
                let flow_idx = self.flow_z_idx(x, z);
                if flow_idx >= self.water_flow_z.len() {
                    continue;
                }
                let vel = self.water_flow_z[flow_idx];
                if vel.abs() < 1e-4 {
                    continue;
                }

                let (src_idx, dst_idx, flux_vol) = if vel > 0.0 {
                    if z == 0 {
                        continue;
                    }
                    let src = self.idx(x, z - 1);
                    let d = self.water_depth(x, z - 1);
                    (
                        src,
                        if z < depth {
                            Some(self.idx(x, z))
                        } else {
                            None
                        },
                        vel * d * cell_size * dt,
                    )
                } else {
                    if z == depth {
                        continue;
                    }
                    let src = self.idx(x, z);
                    let d = self.water_depth(x, z);
                    (
                        src,
                        if z > 0 {
                            Some(self.idx(x, z - 1))
                        } else {
                            None
                        },
                        vel.abs() * d * cell_size * dt,
                    )
                };

                let total_out = outflow_buffer[src_idx];
                let cell_vol = self.water_depth(src_idx % width, src_idx / width) * cell_area;

                let scale = if total_out > cell_vol {
                    cell_vol / total_out  // Limit all fluxes proportionally
                } else {
                    1.0  // No limiting needed
                };

                let src_mass = mass_buffer[src_idx];
                if src_mass < 1e-6 {
                    continue;
                }

                let actual_mass_moved = if cell_vol > 1e-9 {
                    (flux_vol * scale / cell_vol) * src_mass
                } else {
                    0.0
                };

                delta_buffer[src_idx] -= actual_mass_moved;
                if let Some(dst) = dst_idx {
                    delta_buffer[dst] += actual_mass_moved;
                }
            }
        }

        // 4. Apply and Update Concentration
        for i in 0..width * depth {
            // Skip if no change and no mass (optimization)
            if mass_buffer[i] == 0.0 && delta_buffer[i] == 0.0 {
                continue;
            }

            let (x, z) = (i % width, i / width);
            let d = self.water_depth(x, z);

            let new_mass = (mass_buffer[i] + delta_buffer[i]).max(0.0);

            if d > 0.001 {
                self.suspended_sediment[i] = new_mass / (d * cell_area);
                // Safety clamp: Mud can't be more than 50% of volume?
                if self.suspended_sediment[i] > 0.5 {
                    self.suspended_sediment[i] = 0.5;
                }
            } else {
                if new_mass > 0.0 {
                    // Deposit everything if dry
                    let deposit_h = new_mass / cell_area;
                    self.terrain_sediment[i] += deposit_h;
                    self.suspended_sediment[i] = 0.0;
                } else {
                    self.suspended_sediment[i] = 0.0;
                }
            }
        }

        // Put buffers back for reuse
        self.advection_mass_buffer = mass_buffer;
        self.advection_delta_buffer = delta_buffer;
        self.advection_outflow_buffer = outflow_buffer;
    }

    /// Update erosion and sediment transport.
    pub fn update_erosion(&mut self, dt: f32) {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let cell_area = cell_size * cell_size;

        let k_entrain = 2.0; // Entrainment coefficient (Increased for visibility with dt scaling)
        let k_deposit = 0.5; // Deposition coefficient

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);

                // 1. Calculate Water Velocity Magnitude
                let flow_x_left = self.water_flow_x[self.flow_x_idx(x, z)];
                let flow_x_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
                let flow_z_up = self.water_flow_z[self.flow_z_idx(x, z)];
                let flow_z_down = self.water_flow_z[self.flow_z_idx(x, z + 1)];

                let vel_x = (flow_x_left + flow_x_right) * 0.5;
                let vel_z = (flow_z_up + flow_z_down) * 0.5;
                let speed = (vel_x * vel_x + vel_z * vel_z).sqrt();

                let water_depth = self.water_depth(x, z);
                if water_depth < 0.01 {
                    continue;
                }

                // 2. Transport Capacity vs Load
                // Capacity = k * v^2 * d (Simple stream power approx)
                let transport_capacity = speed * speed * water_depth * 0.5;

                let suspended_conc = self.suspended_sediment[idx];
                let suspended_vol = suspended_conc * water_depth * cell_area;

                if suspended_vol > transport_capacity {
                    // Deposition (Too much mud!)
                    let deposit_vol = (suspended_vol - transport_capacity) * k_deposit * dt; // per step
                                                                                             // Clamp to what we have
                    let deposit_vol = deposit_vol.min(suspended_vol);
                    let deposit_height = (deposit_vol / cell_area).min(1.0 * dt);
                    let actual_deposit_vol = deposit_height * cell_area;

                    self.terrain_sediment[idx] += deposit_height;

                    let new_suspended_vol = suspended_vol - actual_deposit_vol;
                    if water_depth > 1e-4 {
                        self.suspended_sediment[idx] =
                            new_suspended_vol / (water_depth * cell_area);
                    } else {
                        self.suspended_sediment[idx] = 0.0;
                    }
                } else {
                    // Erosion (Hungry water!)
                    // Calculate critical velocity for erosion based on next material to erode
                    let critical_velocity = if self.terrain_sediment[idx] > 0.001 {
                        0.3 // Soft sediment
                    } else if self.gravel_thickness[idx] > 0.001 {
                        0.8 // Gravel
                    } else if self.overburden_thickness[idx] > 0.001 {
                        0.5 // Overburden
                    } else {
                        1.2 // Paydirt (very hard)
                    };

                    // CRITICAL: Only erode if speed > material-specific threshold
                    if speed > critical_velocity {
                        let deficit_vol = transport_capacity - suspended_vol;
                        // How much do we WANT to erode
                        let entrain_target_vol = deficit_vol * k_entrain * dt;
                        let entrain_target_height = entrain_target_vol / cell_area;

                        // Limit erosion to e.g. 0.1m/s
                        let entrain_target_height = entrain_target_height.min(0.5 * dt);

                        let mut remaining_demand = entrain_target_height;
                        let mut total_eroded_vol = 0.0;

                        // Erode Sediment
                        if remaining_demand > 0.0 {
                            let available = self.terrain_sediment[idx];
                            let resistance = self.params.hardness_sediment;
                            let take = (remaining_demand / resistance.max(0.1)).min(available);
                            self.terrain_sediment[idx] -= take;
                            total_eroded_vol += take * cell_area;
                            remaining_demand -= take * resistance;
                        }

                        // Erode Gravel
                        if remaining_demand > 0.0 {
                            let available = self.gravel_thickness[idx];
                            let resistance = self.params.hardness_gravel;
                            let take = (remaining_demand / resistance.max(1.0)).min(available);
                            self.gravel_thickness[idx] -= take;
                            total_eroded_vol += take * cell_area;
                            remaining_demand -= take * resistance;
                        }

                        // Erode Overburden (Harder)
                        if remaining_demand > 0.0 {
                            let available = self.overburden_thickness[idx];
                            // Scale demand by hardness? No, scale TAKE by hardness.
                            // Effectively: We try to take X.
                            // If material is hard, we only succeed in taking X / Hardness?
                            // Or: capacity is used up faster?
                            // Simple: resistance factor.
                            let resistance = self.params.hardness_overburden; // e.g. 1.0
                            let take = (remaining_demand / resistance.max(1.0)).min(available);

                            self.overburden_thickness[idx] -= take;
                            total_eroded_vol += take * cell_area;
                            // Does digging harder material use up more transport capacity?
                            // Physically yes. We used up `demand` of energy to get `take` result.
                            remaining_demand -= take * resistance; // Consumed demand
                        }

                        // Erode Paydirt (Very Hard)
                        if remaining_demand > 0.0 {
                            let available = self.paydirt_thickness[idx];
                            let resistance = self.params.hardness_paydirt; // e.g. 5.0
                            let take = (remaining_demand / resistance.max(1.0)).min(available);

                            self.paydirt_thickness[idx] -= take;
                            total_eroded_vol += take * cell_area;
                        }

                        // Add to suspended
                        let new_suspended_vol = suspended_vol + total_eroded_vol;
                        if water_depth > 1e-4 {
                            self.suspended_sediment[idx] =
                                new_suspended_vol / (water_depth * cell_area);
                        }
                    }
                }
            }
        }
    }

    /// Get total deposited sediment volume.
    pub fn total_sediment_volume(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        self.terrain_sediment.iter().sum::<f32>() * cell_area
    }

    /// Get total suspended sediment mass.
    pub fn total_suspended_mass(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        self.suspended_sediment
            .iter()
            .enumerate()
            .map(|(i, &conc)| {
                let x = i % self.width;
                let z = i / self.width;
                let depth = self.water_depth(x, z);
                conc * depth * cell_area
            })
            .sum()
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
                let base_color = match self.surface_material(x, z) {
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

    // =========================================================================
    // Fine Region (Adaptive LOD) Methods
    // =========================================================================

    /// Create a fine region centered on world position with given radius in cells.
    /// Scale determines subdivision (e.g., 4 = 4x4 fine cells per coarse cell).
    pub fn create_fine_region(&mut self, center: Vec3, radius_cells: usize, scale: usize) {
        let (cx, cz) = match self.world_to_cell(center) {
            Some(c) => c,
            None => return,
        };

        let x_min = cx.saturating_sub(radius_cells);
        let z_min = cz.saturating_sub(radius_cells);
        let x_max = (cx + radius_cells).min(self.width - 1);
        let z_max = (cz + radius_cells).min(self.depth - 1);

        let mut fine = FineRegion::new(x_min, z_min, x_max, z_max, scale, self.cell_size);
        self.interpolate_to_fine_region(&mut fine);
        self.fine_region = Some(fine);
    }

    /// Remove the fine region (return to coarse-only simulation).
    pub fn remove_fine_region(&mut self) {
        self.fine_region = None;
    }

    /// Check if a fine region exists and covers the given world position.
    pub fn has_fine_region_at(&self, pos: Vec3) -> bool {
        self.fine_region
            .as_ref()
            .map(|f| f.contains_world_pos(pos, self.cell_size))
            .unwrap_or(false)
    }

    /// Interpolate coarse grid data to fine region using bilinear interpolation.
    fn interpolate_to_fine_region(&self, fine: &mut FineRegion) {
        let scale = fine.scale as f32;

        for fz in 0..fine.depth {
            for fx in 0..fine.width {
                // Map fine cell center to coarse coordinates
                let fine_center_x = (fx as f32 + 0.5) / scale;
                let fine_center_z = (fz as f32 + 0.5) / scale;

                // Coarse cell containing this point
                let cx = fine.coarse_x_min + fine_center_x.floor() as usize;
                let cz = fine.coarse_z_min + fine_center_z.floor() as usize;

                // For simplicity, use nearest-neighbor for terrain layers
                // (bilinear can cause issues at boundaries)
                let cx = cx.min(self.width - 1);
                let cz = cz.min(self.depth - 1);
                let cidx = self.idx(cx, cz);
                let fidx = fine.idx(fx, fz);

                fine.bedrock_elevation[fidx] = self.bedrock_elevation[cidx];
                fine.paydirt_thickness[fidx] = self.paydirt_thickness[cidx];
                fine.gravel_thickness[fidx] = self.gravel_thickness[cidx];
                fine.overburden_thickness[fidx] = self.overburden_thickness[cidx];
                fine.terrain_sediment[fidx] = self.terrain_sediment[cidx];

                // Water state - use bilinear for smoother results
                let (h, conc) = self.bilinear_sample_water(
                    fine.coarse_x_min as f32 + fine_center_x,
                    fine.coarse_z_min as f32 + fine_center_z,
                );
                fine.water_surface[fidx] = h;
                fine.suspended_sediment[fidx] = conc;
            }
        }

        // Initialize velocities to zero (will be driven by boundary conditions)
        fine.water_flow_x.fill(0.0);
        fine.water_flow_z.fill(0.0);
    }

    /// Bilinear sample water surface and sediment concentration at fractional coarse coordinates.
    fn bilinear_sample_water(&self, cx: f32, cz: f32) -> (f32, f32) {
        let x0 = (cx.floor() as usize).min(self.width - 1);
        let z0 = (cz.floor() as usize).min(self.depth - 1);
        let x1 = (x0 + 1).min(self.width - 1);
        let z1 = (z0 + 1).min(self.depth - 1);

        let tx = cx - cx.floor();
        let tz = cz - cz.floor();

        let h00 = self.water_surface[self.idx(x0, z0)];
        let h10 = self.water_surface[self.idx(x1, z0)];
        let h01 = self.water_surface[self.idx(x0, z1)];
        let h11 = self.water_surface[self.idx(x1, z1)];

        let h = h00 * (1.0 - tx) * (1.0 - tz)
            + h10 * tx * (1.0 - tz)
            + h01 * (1.0 - tx) * tz
            + h11 * tx * tz;

        let c00 = self.suspended_sediment[self.idx(x0, z0)];
        let c10 = self.suspended_sediment[self.idx(x1, z0)];
        let c01 = self.suspended_sediment[self.idx(x0, z1)];
        let c11 = self.suspended_sediment[self.idx(x1, z1)];

        let c = c00 * (1.0 - tx) * (1.0 - tz)
            + c10 * tx * (1.0 - tz)
            + c01 * (1.0 - tx) * tz
            + c11 * tx * tz;

        (h, c)
    }

    /// Update fine region simulation with boundary conditions from coarse grid.
    pub fn update_fine_region(&mut self, dt: f32) {
        if self.fine_region.is_none() {
            return;
        }

        // Apply boundary conditions from coarse grid
        self.apply_fine_boundary_conditions();

        // Step the fine region simulation
        let fine = self.fine_region.as_mut().unwrap();

        // Water flow (shallow water equations)
        Self::update_fine_water_flow(fine, dt, self.params.gravity, self.params.water_damping);

        // Terrain collapse (angle of repose)
        fine.update_collapse(
            self.params.collapse_transfer_rate,
            self.params.collapse_max_outflow,
        );

        // Erosion and sediment transport
        fine.update_erosion(
            dt,
            self.params.hardness_overburden,
            self.params.hardness_paydirt,
            self.params.hardness_sediment,
            self.params.hardness_gravel,
        );
    }

    /// Apply coarse grid water state to fine region boundaries.
    fn apply_fine_boundary_conditions(&mut self) {
        // Extract needed values from fine region first
        let (scale, fine_depth, fine_width, coarse_x_min, coarse_z_min, coarse_x_max, coarse_z_max) = {
            let fine = match self.fine_region.as_ref() {
                Some(f) => f,
                None => return,
            };
            (
                fine.scale,
                fine.depth,
                fine.width,
                fine.coarse_x_min,
                fine.coarse_z_min,
                fine.coarse_x_max,
                fine.coarse_z_max,
            )
        };

        let coarse_width = self.width;
        let coarse_depth = self.depth;

        // Collect boundary values from coarse grid
        let mut left_values = Vec::with_capacity(fine_depth);
        let mut right_values = Vec::with_capacity(fine_depth);
        let mut back_values = Vec::with_capacity(fine_width);
        let mut front_values = Vec::with_capacity(fine_width);

        // Left boundary (x = 0)
        for fz in 0..fine_depth {
            let cz = (coarse_z_min + fz / scale).min(coarse_depth - 1);
            let cx = coarse_x_min.saturating_sub(1);
            let cidx = cz * coarse_width + cx;
            left_values.push(self.water_surface[cidx]);
        }

        // Right boundary (x = width - 1)
        for fz in 0..fine_depth {
            let cz = (coarse_z_min + fz / scale).min(coarse_depth - 1);
            let cx = (coarse_x_max + 1).min(coarse_width - 1);
            let cidx = cz * coarse_width + cx;
            right_values.push(self.water_surface[cidx]);
        }

        // Back boundary (z = 0)
        for fx in 0..fine_width {
            let cx = (coarse_x_min + fx / scale).min(coarse_width - 1);
            let cz = coarse_z_min.saturating_sub(1);
            let cidx = cz * coarse_width + cx;
            back_values.push(self.water_surface[cidx]);
        }

        // Front boundary (z = depth - 1)
        for fx in 0..fine_width {
            let cx = (coarse_x_min + fx / scale).min(coarse_width - 1);
            let cz = (coarse_z_max + 1).min(coarse_depth - 1);
            let cidx = cz * coarse_width + cx;
            front_values.push(self.water_surface[cidx]);
        }

        // Now apply to fine region
        let fine = self.fine_region.as_mut().unwrap();

        for (fz, &val) in left_values.iter().enumerate() {
            let fidx = fz * fine.width;
            fine.water_surface[fidx] = val;
        }

        for (fz, &val) in right_values.iter().enumerate() {
            let fidx = fz * fine.width + (fine.width - 1);
            fine.water_surface[fidx] = val;
        }

        for (fx, &val) in back_values.iter().enumerate() {
            fine.water_surface[fx] = val;
        }

        for (fx, &val) in front_values.iter().enumerate() {
            let fidx = (fine.depth - 1) * fine.width + fx;
            fine.water_surface[fidx] = val;
        }
    }

    /// Simplified water flow update for fine region.
    fn update_fine_water_flow(fine: &mut FineRegion, dt: f32, gravity: f32, damping_param: f32) {
        let g = gravity;
        let damping = 1.0 - damping_param;
        let width = fine.width;
        let depth = fine.depth;
        let cell_size = fine.cell_size;
        let cell_area = cell_size * cell_size;

        // Ensure water surface >= ground
        for z in 0..depth {
            for x in 0..width {
                let idx = fine.idx(x, z);
                let ground = fine.ground_height(x, z);
                if fine.water_surface[idx] < ground {
                    fine.water_surface[idx] = ground;
                }
            }
        }

        let max_velocity = cell_size / dt * 0.5;

        // Update X velocities
        for z in 0..depth {
            for x in 1..width {
                let flow_idx = fine.flow_x_idx(x, z);
                let idx_l = fine.idx(x - 1, z);
                let idx_r = fine.idx(x, z);

                let h_l = fine.water_surface[idx_l];
                let h_r = fine.water_surface[idx_r];

                let depth_l = fine.water_depth(x - 1, z);
                let depth_r = fine.water_depth(x, z);

                if depth_l < 0.001 && depth_r < 0.001 {
                    fine.water_flow_x[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_l - h_r) / cell_size;
                fine.water_flow_x[flow_idx] += g * gradient * dt;
                fine.water_flow_x[flow_idx] *= damping;
                fine.water_flow_x[flow_idx] =
                    fine.water_flow_x[flow_idx].clamp(-max_velocity, max_velocity);
            }

            // Boundary velocities
            let idx0 = fine.flow_x_idx(0, z);
            let idx_width = fine.flow_x_idx(width, z);
            fine.water_flow_x[idx0] = 0.0;
            fine.water_flow_x[idx_width] = 0.0;
        }

        // Update Z velocities
        for z in 1..depth {
            for x in 0..width {
                let flow_idx = fine.flow_z_idx(x, z);
                let idx_b = fine.idx(x, z - 1);
                let idx_f = fine.idx(x, z);

                let h_b = fine.water_surface[idx_b];
                let h_f = fine.water_surface[idx_f];

                let depth_b = fine.water_depth(x, z - 1);
                let depth_f = fine.water_depth(x, z);

                if depth_b < 0.001 && depth_f < 0.001 {
                    fine.water_flow_z[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_b - h_f) / cell_size;
                fine.water_flow_z[flow_idx] += g * gradient * dt;
                fine.water_flow_z[flow_idx] *= damping;
                fine.water_flow_z[flow_idx] =
                    fine.water_flow_z[flow_idx].clamp(-max_velocity, max_velocity);
            }
        }

        // Boundary Z velocities
        for x in 0..width {
            let idx0 = fine.flow_z_idx(x, 0);
            let idx_depth = fine.flow_z_idx(x, depth);
            fine.water_flow_z[idx0] = 0.0;
            fine.water_flow_z[idx_depth] = 0.0;
        }

        // Volume update (simplified - no flux limiting for now)
        fine.collapse_deltas.fill(0.0);

        for z in 0..depth {
            for x in 0..width {
                let idx = z * width + x;

                let depth_here = fine.water_depth(x, z);
                let depth_left = if x > 0 { fine.water_depth(x - 1, z) } else { depth_here };
                let depth_right = if x + 1 < width { fine.water_depth(x + 1, z) } else { depth_here };
                let depth_back = if z > 0 { fine.water_depth(x, z - 1) } else { depth_here };
                let depth_front = if z + 1 < depth { fine.water_depth(x, z + 1) } else { depth_here };

                // Pre-compute flow indices to avoid borrow checker issues
                let flow_x_left = z * (width + 1) + x;
                let flow_x_right = z * (width + 1) + x + 1;
                let flow_z_back = z * width + x;
                let flow_z_front = (z + 1) * width + x;

                let flux_left = fine.water_flow_x[flow_x_left] * 0.5 * (depth_left + depth_here);
                let flux_right = fine.water_flow_x[flow_x_right] * 0.5 * (depth_right + depth_here);
                let flux_back = fine.water_flow_z[flow_z_back] * 0.5 * (depth_back + depth_here);
                let flux_front = fine.water_flow_z[flow_z_front] * 0.5 * (depth_front + depth_here);

                let volume_change = (flux_left - flux_right + flux_back - flux_front) * cell_size * dt;
                let height_change = volume_change / cell_area;
                fine.collapse_deltas[idx] = height_change;
            }
        }

        // Apply height changes
        for z in 0..depth {
            for x in 0..width {
                let idx = z * width + x;
                fine.water_surface[idx] += fine.collapse_deltas[idx];
                let ground = fine.ground_height(x, z);
                fine.water_surface[idx] = fine.water_surface[idx].max(ground);
            }
        }
    }

    /// Get ground height at world position, using fine region if available.
    pub fn ground_height_at(&self, pos: Vec3) -> f32 {
        if let Some(ref fine) = self.fine_region {
            if let Some((fx, fz)) = fine.world_to_local(pos, self.cell_size) {
                return fine.ground_height(fx, fz);
            }
        }

        // Fall back to coarse
        if let Some((cx, cz)) = self.world_to_cell(pos) {
            self.ground_height(cx, cz)
        } else {
            0.0
        }
    }

    /// Get water depth at world position, using fine region if available.
    pub fn water_depth_at(&self, pos: Vec3) -> f32 {
        if let Some(ref fine) = self.fine_region {
            if let Some((fx, fz)) = fine.world_to_local(pos, self.cell_size) {
                return fine.water_depth(fx, fz);
            }
        }

        // Fall back to coarse
        if let Some((cx, cz)) = self.world_to_cell(pos) {
            self.water_depth(cx, cz)
        } else {
            0.0
        }
    }

    /// Get terrain and water vertices for the fine region (for rendering).
    pub fn fine_region_terrain_vertices(&self) -> Option<(Vec<[f32; 3]>, Vec<[f32; 3]>)> {
        let fine = self.fine_region.as_ref()?;

        let origin_x = fine.coarse_x_min as f32 * self.cell_size;
        let origin_z = fine.coarse_z_min as f32 * self.cell_size;

        let mut positions = Vec::with_capacity(fine.width * fine.depth);
        let mut colors = Vec::with_capacity(fine.width * fine.depth);

        for z in 0..fine.depth {
            for x in 0..fine.width {
                let height = fine.ground_height(x, z);

                positions.push([
                    origin_x + (x as f32 + 0.5) * fine.cell_size,
                    height,
                    origin_z + (z as f32 + 0.5) * fine.cell_size,
                ]);

                // Simple brown color for terrain
                colors.push([0.45, 0.35, 0.25]);
            }
        }

        Some((positions, colors))
    }

    /// Get water vertices for the fine region.
    pub fn fine_region_water_vertices(&self) -> Option<(Vec<[f32; 3]>, Vec<[f32; 4]>)> {
        let fine = self.fine_region.as_ref()?;

        let origin_x = fine.coarse_x_min as f32 * self.cell_size;
        let origin_z = fine.coarse_z_min as f32 * self.cell_size;

        let mut positions = Vec::new();
        let mut colors = Vec::new();

        for z in 0..fine.depth {
            for x in 0..fine.width {
                let depth = fine.water_depth(x, z);
                if depth < 0.01 {
                    continue;
                }

                let idx = fine.idx(x, z);
                let height = fine.water_surface[idx];
                let turbidity = fine.suspended_sediment[idx];

                positions.push([
                    origin_x + (x as f32 + 0.5) * fine.cell_size,
                    height,
                    origin_z + (z as f32 + 0.5) * fine.cell_size,
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

        Some((positions, colors))
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

        let idx_center = world.idx(5, 5);
        world.terrain_sediment[idx_center] = 10.0;

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

        let diff = (initial_total - final_total).abs();
        assert!(diff < 1.0, "Water volume changed by {}, initial={}, final={}", diff, initial_total, final_total);

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

    #[test]
    fn test_digging_dry_land_stays_dry() {
        let mut world = World::new(3, 3, 1.0, 10.0);
        let center = Vec3::new(1.5, 10.0, 1.5);

        // Dig center
        world.excavate(center, 0.5, 1.0);

        let idx = world.idx(1, 1);
        let ground = world.ground_height(1, 1);
        let surface = world.water_surface[idx];
        let depth = world.water_depth(1, 1);

        // Should be 9.0
        assert!((ground - 9.0).abs() < 0.001, "Ground should be 9.0, got {}", ground);
        // Surface should match ground (dry)
        assert!((surface - ground).abs() < 0.001, "Water surface {} should match ground {}", surface, ground);
        assert!(depth < 0.001, "Depth should be 0, got {}", depth);
    }

    #[test]
    fn test_digging_near_water_flow() {
        let mut world = World::new(3, 3, 1.0, 10.0);
        // Add water to (0,1) - Left neighbor of center (1,1)
        world.add_water(Vec3::new(0.5, 10.0, 1.5), 5.0); // 5m depth. Surface 15.0

        // Verify Setup
        // Neighbor (0,1)
        let idx_n = world.idx(0, 1);
        assert!((world.water_surface[idx_n] - 15.0).abs() < 0.001);

        // Center is dry (1,1)
        assert!(world.water_depth(1, 1) < 0.001);

        // Dig center
        world.excavate(Vec3::new(1.5, 10.0, 1.5), 0.5, 2.0); // Dig 2m -> Height 8

        // Check immediate state
        let idx = world.idx(1, 1);
        let ground = world.ground_height(1, 1);
        let surface = world.water_surface[idx];
        assert!((ground - 8.0).abs() < 0.001);
        // If dry, surface was 10. Excavating dry -> surface becomes ground (8).
        assert!((surface - ground).abs() < 0.001, "Immediate surface should match ground");

        // Update flow once
        world.update_water_flow(0.1);

        // Water should flow in from neighbor (15) to center (8).
        // Center surface should rise.
        let surface_after = world.water_surface[idx];

        // Hypothesis: User claims it rises to "height of removed ground" (10.0 ?? or neighbor 15.0?).
        // If it rises to 15.0 instantly, that's bad.
        // If it rises to 10.0 instantly, that's also bad (arbitrary old ground).

        println!("Surface before: {}, after: {}", surface, surface_after);
        assert!(surface_after > ground, "Water should flow in");
        assert!(surface_after < 14.5, "Water shouldn't level instantly");
    }

    #[test]
    fn test_erosion_velocity_threshold() {
        // Test 1: Low velocity should not erode
        let mut world = World::new(10, 10, 1.0, 10.0);  // Ground at 10m
        let idx = world.idx(5, 5);
        world.terrain_sediment[idx] = 1.0;
        world.water_surface[idx] = 11.0;  // 1m deep water
        // Erosion calculates speed from averaged flows: vel_x = (flow_x_left + flow_x_right) * 0.5
        // To get 0.2 m/s, set both flows to 0.4 m/s (below 0.3 threshold)
        let flow_left = world.flow_x_idx(5, 5);
        let flow_right = world.flow_x_idx(6, 5);
        world.water_flow_x[flow_left] = 0.4;
        world.water_flow_x[flow_right] = 0.4;

        let initial_sediment = world.terrain_sediment[idx];
        world.update_erosion(0.1);

        // Should not erode at low velocity (0.2 m/s < 0.3 threshold)
        assert_eq!(world.terrain_sediment[idx], initial_sediment, "Sediment should not erode below threshold");

        // Test 2: High velocity should erode (fresh world to avoid suspended sediment from previous test)
        let mut world2 = World::new(10, 10, 1.0, 10.0);  // Ground at 10m
        let idx2 = world2.idx(5, 5);
        world2.terrain_sediment[idx2] = 1.0;
        world2.water_surface[idx2] = 11.0;  // 1m deep water
        // To get 0.4 m/s average (above 0.3 threshold), set both flows to 0.8 m/s
        let flow_left2 = world2.flow_x_idx(5, 5);
        let flow_right2 = world2.flow_x_idx(6, 5);
        world2.water_flow_x[flow_left2] = 0.8;
        world2.water_flow_x[flow_right2] = 0.8;

        let initial_sediment2 = world2.terrain_sediment[idx2];
        world2.update_erosion(0.1);

        // Should erode sediment
        let final_sediment2 = world2.terrain_sediment[idx2];
        assert!(final_sediment2 < initial_sediment2, "Sediment should erode above threshold (initial={}, final={}, suspended={})",
            initial_sediment2, final_sediment2, world2.suspended_sediment[idx2]);
    }

    #[test]
    fn test_gravel_erosion_with_hardness() {
        let mut world = World::new(10, 10, 1.0, 0.0);
        let idx = world.idx(5, 5);

        // Set up gravel layer only
        world.gravel_thickness[idx] = 1.0;
        world.water_surface[idx] = 2.0;
        // To get 1.0 m/s average velocity (above 0.8 gravel threshold), set both flows to 2.0 m/s
        let flow_left = world.flow_x_idx(5, 5);
        let flow_right = world.flow_x_idx(6, 5);
        world.water_flow_x[flow_left] = 2.0;
        world.water_flow_x[flow_right] = 2.0;

        let initial_gravel = world.gravel_thickness[idx];
        world.update_erosion(0.1);

        // Gravel should erode
        assert!(world.gravel_thickness[idx] < initial_gravel, "Gravel should erode at high velocity");

        // Verify it's harder to erode than sediment (hardness_gravel = 2.0 vs hardness_sediment = 0.5)
        let gravel_eroded = initial_gravel - world.gravel_thickness[idx];

        // Reset with sediment
        let mut world2 = World::new(10, 10, 1.0, 0.0);
        let idx2 = world2.idx(5, 5);
        world2.terrain_sediment[idx2] = 1.0;
        world2.water_surface[idx2] = 2.0;
        let flow_left2 = world2.flow_x_idx(5, 5);
        let flow_right2 = world2.flow_x_idx(6, 5);
        world2.water_flow_x[flow_left2] = 2.0;
        world2.water_flow_x[flow_right2] = 2.0;

        let initial_sediment = world2.terrain_sediment[idx2];
        world2.update_erosion(0.1);
        let sediment_eroded = initial_sediment - world2.terrain_sediment[idx2];

        // Sediment should erode more than gravel due to lower hardness
        assert!(sediment_eroded > gravel_eroded, "Sediment should erode more than gravel");
    }

    #[test]
    fn test_sediment_hardness_parameter() {
        let mut world = World::new(10, 10, 1.0, 0.0);
        let idx = world.idx(5, 5);

        // Verify hardness_sediment is configured
        assert_eq!(world.params.hardness_sediment, 0.5, "Default hardness_sediment should be 0.5");
        assert_eq!(world.params.hardness_gravel, 2.0, "Default hardness_gravel should be 2.0");

        // Set up deposited sediment
        world.terrain_sediment[idx] = 1.0;
        world.water_surface[idx] = 2.0;
        // To get 0.5 m/s average velocity, set both flows to 1.0 m/s
        let flow_left = world.flow_x_idx(5, 5);
        let flow_right = world.flow_x_idx(6, 5);
        world.water_flow_x[flow_left] = 1.0;
        world.water_flow_x[flow_right] = 1.0;

        let initial = world.terrain_sediment[idx];
        world.update_erosion(0.1);
        let eroded_soft = initial - world.terrain_sediment[idx];

        // Test with harder sediment
        let mut world2 = World::new(10, 10, 1.0, 0.0);
        world2.params.hardness_sediment = 2.0;
        let idx2 = world2.idx(5, 5);
        world2.terrain_sediment[idx2] = 1.0;
        world2.water_surface[idx2] = 2.0;
        let flow_left2 = world2.flow_x_idx(5, 5);
        let flow_right2 = world2.flow_x_idx(6, 5);
        world2.water_flow_x[flow_left2] = 1.0;
        world2.water_flow_x[flow_right2] = 1.0;

        let initial2 = world2.terrain_sediment[idx2];
        world2.update_erosion(0.1);
        let eroded_hard = initial2 - world2.terrain_sediment[idx2];

        // Softer sediment should erode more
        assert!(eroded_soft > eroded_hard, "Softer sediment should erode more");
    }

    #[test]
    fn test_sediment_advection_mass_conservation() {
        let mut world = World::new(5, 5, 1.0, 0.0);

        // Simple test: single cell with mass, NO flows anywhere
        // Use 0.3 concentration (below 0.5 clamp limit)
        let idx = world.idx(2, 2);
        world.suspended_sediment[idx] = 0.3; // Safe concentration below 0.5 clamp
        world.water_surface[idx] = 2.0; // 2m deep water
        // ALL flows default to 0.0 - no movement

        let initial_total_mass = world.total_suspended_mass();

        // Run advection with zero flow - mass should NOT change
        world.update_sediment_advection(0.1);

        let final_total_mass = world.total_suspended_mass();

        // With zero flow, mass must be perfectly conserved
        let mass_error = ((final_total_mass - initial_total_mass) / initial_total_mass).abs();
        assert!(mass_error < 0.001, "Mass conservation violated with ZERO flow: {}% error (initial={}, final={})", mass_error * 100.0, initial_total_mass, final_total_mass);
    }

    #[test]
    fn test_erosion_layer_order() {
        let mut world = World::new(10, 10, 1.0, 0.0);
        let idx = world.idx(5, 5);

        // Set up multi-layer cell
        world.terrain_sediment[idx] = 0.1;
        world.gravel_thickness[idx] = 0.1;
        world.overburden_thickness[idx] = 0.1;
        world.paydirt_thickness[idx] = 1.0;

        // High velocity water (2.0 m/s average needs 4.0 m/s flows)
        world.water_surface[idx] = 3.0;
        let flow_left = world.flow_x_idx(5, 5);
        let flow_right = world.flow_x_idx(6, 5);
        world.water_flow_x[flow_left] = 4.0;
        world.water_flow_x[flow_right] = 4.0;

        // Run erosion
        for _ in 0..10 {
            world.update_erosion(0.1);
        }

        // Sediment should be gone first
        assert!(world.terrain_sediment[idx] < 0.01, "Sediment should erode first");

        // Gravel should be partially eroded
        assert!(world.gravel_thickness[idx] < 0.1, "Gravel should erode after sediment");

        // Overburden might be touched
        // Paydirt should be mostly intact due to high hardness
        assert!(world.paydirt_thickness[idx] > 0.9, "Paydirt should erode last");
    }

    #[test]
    fn test_fine_region_gravel_erosion() {
        let mut world = World::new(20, 20, 1.0, 0.0);

        // Create fine region centered at (10, 10) with radius 5 cells and scale 4
        let center = Vec3::new(10.0, 0.0, 10.0);
        world.create_fine_region(center, 5, 4);

        if let Some(ref mut fine) = world.fine_region {
            let idx = fine.idx(8, 8);

            // Set up gravel layer
            fine.gravel_thickness[idx] = 1.0;
            fine.water_surface[idx] = 2.0;
            // To get 1.0 m/s average velocity, set both flows to 2.0 m/s
            let flow_left = fine.flow_x_idx(8, 8);
            let flow_right = fine.flow_x_idx(9, 8);
            fine.water_flow_x[flow_left] = 2.0;
            fine.water_flow_x[flow_right] = 2.0;

            let initial_gravel = fine.gravel_thickness[idx];

            fine.update_erosion(
                0.1,
                world.params.hardness_overburden,
                world.params.hardness_paydirt,
                world.params.hardness_sediment,
                world.params.hardness_gravel,
            );

            // Gravel should erode
            assert!(fine.gravel_thickness[idx] < initial_gravel, "Fine region gravel should erode");
        } else {
            panic!("Fine region not created");
        }
    }
}
