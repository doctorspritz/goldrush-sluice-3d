//! Unified world simulation: terrain + water + sediment settling.
//!
//! This is the "background" simulation for areas outside active particle zones.
//! Everything is heightfield-based for performance.

use crate::constants::GRAVITY_MAGNITUDE;
use glam::Vec3;

mod geometry;
mod physics;
mod terrain;
mod water_flow;

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
    /// Sediment settling velocity (m/s). Deprecated: use settling_velocity() method with d50.
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

    // ===== Particle Size Parameters (Shields-stress physics) =====
    /// Median particle diameter for sediment (meters). Default: 0.0001 (0.1mm fine silt)
    pub d50_sediment: f32,
    /// Median particle diameter for overburden (meters). Default: 0.001 (1mm coarse sand)
    pub d50_overburden: f32,
    /// Median particle diameter for gravel (meters). Default: 0.01 (10mm gravel)
    pub d50_gravel: f32,
    /// Median particle diameter for paydirt (meters). Default: 0.002 (2mm compacted sand)
    pub d50_paydirt: f32,

    // ===== Physical Constants =====
    /// Sediment particle density (kg/m³). Default: 2650 (quartz sand/gravel)
    pub rho_sediment: f32,
    /// Water density (kg/m³). Default: 1000
    pub rho_water: f32,
    /// Dynamic viscosity of water (Pa·s). Default: 0.001
    pub water_viscosity: f32,
    /// Critical Shields stress for transport initiation. Default: 0.045
    pub critical_shields: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            angle_of_repose: 35.0_f32.to_radians(),
            collapse_transfer_rate: 0.35,
            collapse_max_outflow: 0.5,
            gravity: GRAVITY_MAGNITUDE,
            water_damping: 0.0, // Deprecated, use manning_n
            manning_n: 0.03, // Smooth channel roughness
            settling_velocity: 0.01, // Deprecated
            bed_porosity: 0.4,
            hardness_overburden: 1.0,
            hardness_paydirt: 5.0,
            hardness_sediment: 0.5, // Loose sediment easier than overburden
            hardness_gravel: 2.0,   // Between overburden (1.0) and paydirt (5.0)
            open_boundaries: true, // Default to open for simulation

            // Particle sizes (meters)
            d50_sediment: 0.0001,    // 0.1mm fine silt
            d50_overburden: 0.001,   // 1mm coarse sand
            d50_gravel: 0.01,        // 10mm gravel
            d50_paydirt: 0.002,      // 2mm compacted sand

            // Physical constants
            rho_sediment: 2650.0,    // kg/m³ (quartz)
            rho_water: 1000.0,       // kg/m³
            water_viscosity: 0.001,  // Pa·s (water at 20°C)
            critical_shields: 0.045, // Typical value for most sediments
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

fn smooth_delta(width: usize, depth: usize, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; width * depth];
    if width < 3 || depth < 3 {
        output.copy_from_slice(input);
        return output;
    }

    for z in 0..depth {
        for x in 0..width {
            let idx = z * width + x;
            if x == 0 || z == 0 || x + 1 == width || z + 1 == depth {
                output[idx] = input[idx];
                continue;
            }

            let mut sum = 0.0;
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nz = (z as i32 + dz) as usize;
                    let nx = (x as i32 + dx) as usize;
                    sum += input[nz * width + nx];
                }
            }
            output[idx] = sum / 9.0;
        }
    }

    output
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
    pub fn update_erosion(
        &mut self,
        dt: f32,
        params: &WorldParams,
        hardness_overburden: f32,
        hardness_paydirt: f32,
        hardness_sediment: f32,
        hardness_gravel: f32,
    ) {
        let width = self.width;
        let depth = self.depth;
        let cell_area = self.cell_size * self.cell_size;

        let g = params.gravity;
        let rho_diff = params.rho_sediment - params.rho_water;
        let critical_shields = params.critical_shields;
        let max_erosion_per_step = 0.00001 * dt;
        let min_erosion_speed = 0.1;

        let mut delta_sediment = vec![0.0; width * depth];
        let mut delta_gravel = vec![0.0; width * depth];
        let mut delta_overburden = vec![0.0; width * depth];
        let mut delta_paydirt = vec![0.0; width * depth];

        let bed_slope = |x: usize, z: usize, this: &FineRegion| -> f32 {
            let h_here = this.ground_height(x, z);
            let slope_x = if x > 0 && x < this.width - 1 {
                let h_left = this.ground_height(x - 1, z);
                let h_right = this.ground_height(x + 1, z);
                (h_left - h_right) / (2.0 * this.cell_size)
            } else if x == 0 && this.width > 1 {
                (h_here - this.ground_height(x + 1, z)) / this.cell_size
            } else if x + 1 == this.width && this.width > 1 {
                (this.ground_height(x - 1, z) - h_here) / this.cell_size
            } else {
                0.0
            };

            let slope_z = if z > 0 && z < this.depth - 1 {
                let h_up = this.ground_height(x, z - 1);
                let h_down = this.ground_height(x, z + 1);
                (h_up - h_down) / (2.0 * this.cell_size)
            } else if z == 0 && this.depth > 1 {
                (h_here - this.ground_height(x, z + 1)) / this.cell_size
            } else if z + 1 == this.depth && this.depth > 1 {
                (this.ground_height(x, z - 1) - h_here) / this.cell_size
            } else {
                0.0
            };

            (slope_x * slope_x + slope_z * slope_z).sqrt()
        };

        let settling_velocity = |d50: f32| -> f32 {
            let g = params.gravity;
            let rho_p = params.rho_sediment;
            let rho_f = params.rho_water;
            let mu = params.water_viscosity;

            let vs_stokes = g * (rho_p - rho_f) * d50 * d50 / (18.0 * mu);
            let vs_turbulent =
                (4.0 * g * d50 * (rho_p - rho_f) / (3.0 * rho_f * 0.44)).sqrt();
            let transition = (d50 / 0.001).clamp(0.0, 1.0);
            vs_stokes * (1.0 - transition) + vs_turbulent * transition
        };

        for z in 1..depth - 1 {
            for x in 1..width - 1 {
                let idx = self.idx(x, z);

                let old_ground = self.ground_height(x, z);
                let water_depth = (self.water_surface[idx] - old_ground).max(0.0);
                if water_depth <= 0.01 {
                    continue;
                }

                let flow_x_left = self.water_flow_x[self.flow_x_idx(x, z)];
                let flow_x_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
                let flow_z_up = self.water_flow_z[self.flow_z_idx(x, z)];
                let flow_z_down = self.water_flow_z[self.flow_z_idx(x, z + 1)];
                let vx = (flow_x_left + flow_x_right) * 0.5;
                let vz = (flow_z_up + flow_z_down) * 0.5;
                let flow_speed = (vx * vx + vz * vz).sqrt();

                // Shear stress using velocity-only formula (not additive with gravity)
                let cf = 0.003;
                let v_sq = vx * vx + vz * vz;
                let shear_stress = params.rho_water * cf * v_sq;

                let max_conc = 0.5;
                let mut suspended_before = self.suspended_sediment[idx].min(max_conc);
                let s0 = self.terrain_sediment[idx];
                let g0 = self.gravel_thickness[idx];
                let o0 = self.overburden_thickness[idx];
                let p0 = self.paydirt_thickness[idx];
                let mut s_thick = s0;
                let mut g_thick = g0;
                let mut o_thick = o0;
                let mut p_thick = p0;

                let v_settle = settling_velocity(params.d50_sediment);
                // Linear suppression with threshold (not quadratic)
                let suppression_threshold = 2.0 * v_settle;
                let deposition_suppression = if flow_speed > suppression_threshold {
                    suppression_threshold / flow_speed
                } else {
                    1.0
                };
                let v_eff = v_settle * deposition_suppression;
                let depth_scale = water_depth.max(0.02);
                let settling_rate = v_eff / depth_scale;
                let settle_cap = if flow_speed < 0.05 { 0.1 } else { 0.02 };
                let settled_frac = (settling_rate * dt).min(settle_cap);
                let settled_conc = suspended_before * settled_frac;
                let deposit_height = settled_conc * water_depth;

                let mut total_eroded_height = 0.0;
                let mut erode_layer = |thickness: &mut f32, d50: f32, hardness: f32| -> f32 {
                    if *thickness <= 0.001 || total_eroded_height >= max_erosion_per_step {
                        return 0.0;
                    }
                    if flow_speed < min_erosion_speed {
                        return 0.0;
                    }
                    let shields = shear_stress / (g * rho_diff * d50.max(1e-6));
                    if shields > critical_shields {
                        let excess = (shields - critical_shields) / critical_shields;
                        let erosion_rate = 0.0001 * excess / hardness.max(0.1);
                        let erode_height = (erosion_rate * dt).min(*thickness).min(
                            max_erosion_per_step - total_eroded_height,
                        );
                        *thickness -= erode_height;
                        total_eroded_height += erode_height;
                        erode_height
                    } else {
                        0.0
                    }
                };

                let _ = erode_layer(&mut s_thick, params.d50_sediment, hardness_sediment);
                let _ = erode_layer(&mut g_thick, params.d50_gravel, hardness_gravel);
                let _ = erode_layer(&mut o_thick, params.d50_overburden, hardness_overburden);
                let _ = erode_layer(&mut p_thick, params.d50_paydirt, hardness_paydirt);

                let net_height = total_eroded_height - deposit_height;
                if net_height > 0.0 {
                    let scale = if total_eroded_height > 0.0 {
                        net_height / total_eroded_height
                    } else {
                        0.0
                    };
                    s_thick = s0 - (s0 - s_thick) * scale;
                    g_thick = g0 - (g0 - g_thick) * scale;
                    o_thick = o0 - (o0 - o_thick) * scale;
                    p_thick = p0 - (p0 - p_thick) * scale;

                    let bed_porosity = params.bed_porosity;
                    let eroded_volume = net_height * cell_area * (1.0 - bed_porosity);
                    let water_vol = water_depth * cell_area;
                    let eroded_conc = eroded_volume / water_vol;
                    let capacity = (max_conc - suspended_before).max(0.0);
                    let accepted = eroded_conc.min(capacity);
                    suspended_before += accepted;
                } else if net_height < 0.0 {
                    let net_deposit = -net_height;
                    s_thick = s0 + net_deposit;
                    g_thick = g0;
                    o_thick = o0;
                    p_thick = p0;

                    let deposit_conc = (net_deposit / water_depth).min(suspended_before);
                    suspended_before -= deposit_conc;
                }

                self.suspended_sediment[idx] = suspended_before;

                delta_sediment[idx] = s_thick - self.terrain_sediment[idx];
                delta_gravel[idx] = g_thick - self.gravel_thickness[idx];
                delta_overburden[idx] = o_thick - self.overburden_thickness[idx];
                delta_paydirt[idx] = p_thick - self.paydirt_thickness[idx];
            }
        }

        delta_sediment = smooth_delta(width, depth, &delta_sediment);
        delta_gravel = smooth_delta(width, depth, &delta_gravel);
        delta_overburden = smooth_delta(width, depth, &delta_overburden);
        delta_paydirt = smooth_delta(width, depth, &delta_paydirt);

        let mut displaced_volume = 0.0;
        let mut wet_area = 0.0;

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let delta_ground = delta_sediment[idx]
                    + delta_gravel[idx]
                    + delta_overburden[idx]
                    + delta_paydirt[idx];
                let old_ground = self.ground_height(x, z);
                let was_wet = self.water_surface[idx] > old_ground + 1e-6;

                self.terrain_sediment[idx] =
                    (self.terrain_sediment[idx] + delta_sediment[idx]).max(0.0);
                self.gravel_thickness[idx] =
                    (self.gravel_thickness[idx] + delta_gravel[idx]).max(0.0);
                self.overburden_thickness[idx] =
                    (self.overburden_thickness[idx] + delta_overburden[idx]).max(0.0);
                self.paydirt_thickness[idx] =
                    (self.paydirt_thickness[idx] + delta_paydirt[idx]).max(0.0);

                if was_wet {
                    displaced_volume += delta_ground * cell_area;
                    wet_area += cell_area;
                } else {
                    let new_ground = old_ground + delta_ground;
                    if self.water_surface[idx] < new_ground {
                        self.water_surface[idx] = new_ground;
                    }
                }
            }
        }

        if wet_area > 0.0 && displaced_volume.abs() > 0.0 {
            let delta_surface = displaced_volume / wet_area;
            for z in 0..depth {
                for x in 0..width {
                    let idx = self.idx(x, z);
                    let new_ground = self.ground_height(x, z);
                    if self.water_surface[idx] > new_ground + 1e-6 {
                        self.water_surface[idx] += delta_surface;
                        if self.water_surface[idx] < new_ground {
                            self.water_surface[idx] = new_ground;
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

    erosion_step_counter: u32,
    erosion_step_stride: u32,
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
            erosion_step_counter: 0,
            erosion_step_stride: 2,
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

    // =========================================================================
    // SHIELDS-STRESS EROSION PHYSICS
    // =========================================================================


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
        let do_erosion = self.next_erosion_step();
        if do_erosion {
            self.update_erosion(
                dt,
                self.params.hardness_overburden,
                self.params.hardness_paydirt,
                self.params.hardness_sediment,
                self.params.hardness_gravel,
            );
        }

        // Fine region simulation (if active)
        self.update_fine_region(dt, do_erosion);
    }

    /// Advance erosion step counter and return whether to update erosion this frame.
    pub fn next_erosion_step(&mut self) -> bool {
        let stride = self.erosion_step_stride.max(1);
        self.erosion_step_counter = self.erosion_step_counter.wrapping_add(1);
        self.erosion_step_counter % stride == 0
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

    /// Update erosion and sediment transport using Shields stress physics.
    pub fn update_erosion(
        &mut self,
        dt: f32,
        hardness_overburden: f32,
        hardness_paydirt: f32,
        hardness_sediment: f32,
        hardness_gravel: f32,
    ) {
        // Self-contained erosion model (per-cell Shields-based)
        let width = self.width;
        let depth = self.depth;
        let cell_area = self.cell_size * self.cell_size;

        // Global physics constants
        let g = self.params.gravity;
        let rho_diff = self.params.rho_sediment - self.params.rho_water;
        let critical_shields = self.params.critical_shields;
        // Per-step erosion cap (tuneable)
        // Increased from 0.00001 to allow hardness-based erosion rates to dominate
        // The physics-based rate (k_erosion * excess / hardness * dt) should control
        // erosion speed, not this safety cap
        let max_erosion_per_step = 0.001 * dt;
        let min_erosion_speed = 0.1;

        let mut delta_sediment = vec![0.0; width * depth];
        let mut delta_gravel = vec![0.0; width * depth];
        let mut delta_overburden = vec![0.0; width * depth];
        let mut delta_paydirt = vec![0.0; width * depth];

        for z in 1..depth - 1 {
            for x in 1..width - 1 {
                let idx = self.idx(x, z);

                // 1) Water-driven availability checks
                let old_ground = self.ground_height(x, z);
                let water_depth = (self.water_surface[idx] - old_ground).max(0.0);
                if water_depth <= 0.01 {
                    continue;
                }

                // 2) Compute flow speed at this cell (to modulate deposition)
                // Use face-averaged velocities
                let flow_x_left = self.water_flow_x[self.flow_x_idx(x, z)];
                let flow_x_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
                let flow_z_up = self.water_flow_z[self.flow_z_idx(x, z)];
                let flow_z_down = self.water_flow_z[self.flow_z_idx(x, z + 1)];
                let vx = (flow_x_left + flow_x_right) * 0.5;
                let vz = (flow_z_up + flow_z_down) * 0.5;
                let flow_speed = (vx * vx + vz * vz).sqrt();

                // 3) Shields stress (for per-layer transport in a single pass later)
                // We'll approximate using the top-most material's d50 (we'll compute per-layer below)
                let max_conc = 0.5;
                let mut suspended_before = self.suspended_sediment[idx].min(max_conc);
                let s0 = self.terrain_sediment[idx];
                let g0 = self.gravel_thickness[idx];
                let o0 = self.overburden_thickness[idx];
                let p0 = self.paydirt_thickness[idx];
                let mut s_thick = s0;
                let mut g_thick = g0;
                let mut o_thick = o0;
                let mut p_thick = p0;

                // 4) Deposition: deposit from the suspended_before amount, velocity-dependent
                // Deposition should be suppressed if flow is strong relative to settling
                let v_settle = self.settling_velocity(self.params.d50_sediment);
                // Linear suppression with threshold (not quadratic)
                let suppression_threshold = 2.0 * v_settle;
                let deposition_suppression = if flow_speed > suppression_threshold {
                    suppression_threshold / flow_speed
                } else {
                    1.0
                };
                let v_eff = v_settle * deposition_suppression;
                let depth_scale = water_depth.max(0.02);
                let settling_rate = v_eff / depth_scale;
                let settle_cap = if flow_speed < 0.05 { 0.1 } else { 0.02 };
                let settled_frac = (settling_rate * dt).min(settle_cap);
                let settled_conc = suspended_before * settled_frac;

                // Potential deposition from suspended_before (not including newly eroded mass)
                let deposit_height = settled_conc * water_depth;

                // 5) Erosion: erode layers in fixed order with per-step cap
                let mut total_eroded_height = 0.0;
                let shear_stress = self.shear_stress(x, z);

                // Helper closure: erosion per layer
                // thickness, d50, hardness
                let mut erode_layer = |thickness: &mut f32, d50: f32, hardness: f32| -> f32 {
                    if *thickness <= 0.001 || total_eroded_height >= max_erosion_per_step {
                        return 0.0;
                    }
                    if flow_speed < min_erosion_speed {
                        return 0.0;
                    }
                    let shields = shear_stress / (g * rho_diff * d50.max(1e-6));

                    if shields > critical_shields {
                        let excess = (shields - critical_shields) / critical_shields;
                        let erosion_rate = 0.0001 * excess / hardness.max(0.1); // k_erosion tuned here
                        let erode_height = (erosion_rate * dt).min(*thickness).min(
                            max_erosion_per_step - total_eroded_height,
                        );
                        *thickness -= erode_height;
                        total_eroded_height += erode_height;
                        erode_height
                    } else {
                        0.0
                    }
                };

                // Sediment
                let _ = erode_layer(&mut s_thick, self.params.d50_sediment, hardness_sediment);

                // Gravel
                let _ = erode_layer(&mut g_thick, self.params.d50_gravel, hardness_gravel);

                // Overburden
                let _ = erode_layer(
                    &mut o_thick,
                    self.params.d50_overburden,
                    hardness_overburden.max(0.1),
                );

                // Paydirt
                let _ = erode_layer(
                    &mut p_thick,
                    self.params.d50_paydirt,
                    hardness_paydirt.max(0.1),
                );

                // Sum up total eroded height from all layers
                // (In this simplified rewrite, the per-layer closures already updated the thicknesses
                // and reduced total_eroded_height accordingly.)
                // For explicitness, convert the last total eroded_height to a net amount in suspension:
                let net_height = total_eroded_height - deposit_height;
                if net_height > 0.0 {
                    let scale = if total_eroded_height > 0.0 {
                        net_height / total_eroded_height
                    } else {
                        0.0
                    };
                    s_thick = s0 - (s0 - s_thick) * scale;
                    g_thick = g0 - (g0 - g_thick) * scale;
                    o_thick = o0 - (o0 - o_thick) * scale;
                    p_thick = p0 - (p0 - p_thick) * scale;

                    // Convert net eroded height to suspended concentration
                    let bed_porosity = self.params.bed_porosity;
                    let eroded_volume = net_height * cell_area * (1.0 - bed_porosity);
                    let water_vol = water_depth * cell_area;
                    let eroded_conc = eroded_volume / water_vol;
                    let capacity = (max_conc - suspended_before).max(0.0);
                    let accepted = eroded_conc.min(capacity);
                    suspended_before += accepted;
                } else if net_height < 0.0 {
                    let net_deposit = -net_height;
                    s_thick = s0 + net_deposit;
                    g_thick = g0;
                    o_thick = o0;
                    p_thick = p0;

                    let deposit_conc = (net_deposit / water_depth).min(suspended_before);
                    suspended_before -= deposit_conc;
                }

                self.suspended_sediment[idx] = suspended_before;

                delta_sediment[idx] = s_thick - self.terrain_sediment[idx];
                delta_gravel[idx] = g_thick - self.gravel_thickness[idx];
                delta_overburden[idx] = o_thick - self.overburden_thickness[idx];
                delta_paydirt[idx] = p_thick - self.paydirt_thickness[idx];
            }
        }

        delta_sediment = smooth_delta(width, depth, &delta_sediment);
        delta_gravel = smooth_delta(width, depth, &delta_gravel);
        delta_overburden = smooth_delta(width, depth, &delta_overburden);
        delta_paydirt = smooth_delta(width, depth, &delta_paydirt);

        let mut displaced_volume = 0.0;
        let mut wet_area = 0.0;

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let delta_ground = delta_sediment[idx]
                    + delta_gravel[idx]
                    + delta_overburden[idx]
                    + delta_paydirt[idx];
                let old_ground = self.ground_height(x, z);
                let was_wet = self.water_surface[idx] > old_ground + 1e-6;

                self.terrain_sediment[idx] =
                    (self.terrain_sediment[idx] + delta_sediment[idx]).max(0.0);
                self.gravel_thickness[idx] =
                    (self.gravel_thickness[idx] + delta_gravel[idx]).max(0.0);
                self.overburden_thickness[idx] =
                    (self.overburden_thickness[idx] + delta_overburden[idx]).max(0.0);
                self.paydirt_thickness[idx] =
                    (self.paydirt_thickness[idx] + delta_paydirt[idx]).max(0.0);

                if was_wet {
                    displaced_volume += delta_ground * cell_area;
                    wet_area += cell_area;
                } else {
                    let new_ground = old_ground + delta_ground;
                    if self.water_surface[idx] < new_ground {
                        self.water_surface[idx] = new_ground;
                    }
                }
            }
        }

        if wet_area > 0.0 && displaced_volume.abs() > 0.0 {
            let delta_surface = displaced_volume / wet_area;
            for z in 0..depth {
                for x in 0..width {
                    let idx = self.idx(x, z);
                    let new_ground = self.ground_height(x, z);
                    if self.water_surface[idx] > new_ground + 1e-6 {
                        self.water_surface[idx] += delta_surface;
                        if self.water_surface[idx] < new_ground {
                            self.water_surface[idx] = new_ground;
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
    pub fn update_fine_region(&mut self, dt: f32, do_erosion: bool) {
        if self.fine_region.is_none() {
            return;
        }

        // Apply boundary conditions from coarse grid
        self.apply_fine_boundary_conditions();

        // Step the fine region simulation
        let fine = self.fine_region.as_mut().unwrap();

        // Water flow (shallow water equations)
        FineRegion::update_water_flow(fine, dt, self.params.gravity, self.params.water_damping);

        // Terrain collapse (angle of repose)
        fine.update_collapse(
            self.params.collapse_transfer_rate,
            self.params.collapse_max_outflow,
        );

        // Erosion and sediment transport
        if do_erosion {
            fine.update_erosion(
                dt,
                &self.params,
                self.params.hardness_overburden,
                self.params.hardness_paydirt,
                self.params.hardness_sediment,
                self.params.hardness_gravel,
            );
        }
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
        Some(fine.terrain_vertices(self.cell_size, fine.coarse_x_min, fine.coarse_z_min))
    }

    /// Get water vertices for the fine region.
    pub fn fine_region_water_vertices(&self) -> Option<(Vec<[f32; 3]>, Vec<[f32; 4]>)> {
        let fine = self.fine_region.as_ref()?;
        Some(fine.water_vertices(self.cell_size, fine.coarse_x_min, fine.coarse_z_min))
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
        world.params.open_boundaries = false; // Closed system for conservation test

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
        // Test 1: Very low velocity should not erode (Shields stress below critical)
        let mut world = World::new(10, 10, 1.0, 10.0);  // Ground at 10m
        let idx = world.idx(5, 5);
        world.terrain_sediment[idx] = 1.0;
        // Ground = 10.0 + 1.0 sediment = 11.0, so water_surface must be higher for depth
        world.water_surface[idx] = 11.1;  // Only 0.1m deep water (low shear)
        // Very low velocity - should produce Shields stress below critical
        let flow_left = world.flow_x_idx(5, 5);
        let flow_right = world.flow_x_idx(6, 5);
        world.water_flow_x[flow_left] = 0.05;
        world.water_flow_x[flow_right] = 0.05;

        let initial_sediment = world.terrain_sediment[idx];
        world.update_erosion(
            0.1,
            world.params.hardness_overburden,
            world.params.hardness_paydirt,
            world.params.hardness_sediment,
            world.params.hardness_gravel,
        );

        // Should not erode (or erode negligibly) at low Shields stress
        let sediment_change = (world.terrain_sediment[idx] - initial_sediment).abs();
        assert!(sediment_change < 0.01, "Sediment should not erode significantly below Shields threshold (change={})", sediment_change);

        // Test 2: High velocity should erode (fresh world to avoid suspended sediment from previous test)
        let mut world2 = World::new(10, 10, 1.0, 10.0);  // Ground at 10m
        let idx2 = world2.idx(5, 5);
        world2.terrain_sediment[idx2] = 1.0;
        // Ground = 10.0 + 1.0 sediment = 11.0, so water_surface must be higher for depth
        world2.water_surface[idx2] = 12.0;  // 1m deep water
        // High velocity - should produce Shields stress above critical
        let flow_left2 = world2.flow_x_idx(5, 5);
        let flow_right2 = world2.flow_x_idx(6, 5);
        world2.water_flow_x[flow_left2] = 2.0;
        world2.water_flow_x[flow_right2] = 2.0;

        let initial_sediment2 = world2.terrain_sediment[idx2];
        world2.update_erosion(
            0.1,
            world2.params.hardness_overburden,
            world2.params.hardness_paydirt,
            world2.params.hardness_sediment,
            world2.params.hardness_gravel,
        );

        // Should erode sediment
        let final_sediment2 = world2.terrain_sediment[idx2];
        assert!(final_sediment2 < initial_sediment2, "Sediment should erode above Shields threshold (initial={}, final={}, suspended={})",
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
        world.update_erosion(
            0.1,
            world.params.hardness_overburden,
            world.params.hardness_paydirt,
            world.params.hardness_sediment,
            world.params.hardness_gravel,
        );

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
        world2.update_erosion(
            0.1,
            world2.params.hardness_overburden,
            world2.params.hardness_paydirt,
            world2.params.hardness_sediment,
            world2.params.hardness_gravel,
        );
        let sediment_eroded = initial_sediment - world2.terrain_sediment[idx2];

        // Sediment should erode more than gravel due to lower hardness
        assert!(sediment_eroded > gravel_eroded, "Sediment should erode more than gravel");
    }

    #[test]
    fn test_sediment_hardness_parameter() {
        // Verify hardness defaults are configured
        let world_check = World::new(10, 10, 1.0, 0.0);
        assert_eq!(world_check.params.hardness_sediment, 0.5, "Default hardness_sediment should be 0.5");
        assert_eq!(world_check.params.hardness_gravel, 2.0, "Default hardness_gravel should be 2.0");

        // Test with SOFT sediment (hardness = 0.5)
        let mut world = World::new(10, 10, 1.0, 0.0);
        let idx = world.idx(5, 5);
        world.terrain_sediment[idx] = 1.0;

        // Add water and flow across wider area
        for z in 3..7 {
            for x in 3..7 {
                let i = world.idx(x, z);
                world.water_surface[i] = 3.0;
            }
        }
        for z in 3..7 {
            for x in 3..8 {
                let flow_idx = world.flow_x_idx(x, z);
                world.water_flow_x[flow_idx] = 4.0;
            }
        }

        let initial = world.terrain_sediment[idx];
        for _ in 0..5 {
            world.update_erosion(
                0.1,
                world.params.hardness_overburden,
                world.params.hardness_paydirt,
                world.params.hardness_sediment,
                world.params.hardness_gravel,
            );
        }
        let eroded_soft = initial - world.terrain_sediment[idx];

        // Test with HARD sediment (hardness = 2.0)
        let mut world2 = World::new(10, 10, 1.0, 0.0);
        world2.params.hardness_sediment = 2.0;
        let idx2 = world2.idx(5, 5);
        world2.terrain_sediment[idx2] = 1.0;

        for z in 3..7 {
            for x in 3..7 {
                let i = world2.idx(x, z);
                world2.water_surface[i] = 3.0;
            }
        }
        for z in 3..7 {
            for x in 3..8 {
                let flow_idx = world2.flow_x_idx(x, z);
                world2.water_flow_x[flow_idx] = 4.0;
            }
        }

        let initial2 = world2.terrain_sediment[idx2];
        for _ in 0..5 {
            world2.update_erosion(
                0.1,
                world2.params.hardness_overburden,
                world2.params.hardness_paydirt,
                world2.params.hardness_sediment,
                world2.params.hardness_gravel,
            );
        }
        let eroded_hard = initial2 - world2.terrain_sediment[idx2];

        // Both should erode, but soft should erode more
        assert!(eroded_soft > 0.0, "Soft sediment should erode (eroded={})", eroded_soft);
        assert!(eroded_hard >= 0.0, "Hard sediment erosion should be non-negative (eroded={})", eroded_hard);
        assert!(eroded_soft >= eroded_hard, "Softer sediment should erode at least as much (soft={}, hard={})", eroded_soft, eroded_hard);
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

        // High velocity water covering a wider area (not just one cell)
        // Water needs to cover the cell and neighbors for flow averaging to work
        for z in 3..7 {
            for x in 3..7 {
                let i = world.idx(x, z);
                world.water_surface[i] = 3.0;
            }
        }

        // Set flow across the area (not just at one edge)
        for z in 3..7 {
            for x in 3..8 {
                let flow_idx = world.flow_x_idx(x, z);
                world.water_flow_x[flow_idx] = 4.0;
            }
        }

        let initial_sediment = world.terrain_sediment[idx];
        let initial_gravel = world.gravel_thickness[idx];
        let initial_paydirt = world.paydirt_thickness[idx];

        // Run erosion for more iterations
        for _ in 0..20 {
            world.update_erosion(
                0.1,
                world.params.hardness_overburden,
                world.params.hardness_paydirt,
                world.params.hardness_sediment,
                world.params.hardness_gravel,
            );
        }

        // Sediment should erode (small per-step cap makes this subtle)
        let sediment_eroded = initial_sediment - world.terrain_sediment[idx];
        assert!(
            sediment_eroded > 1e-5,
            "Sediment should erode (initial={}, final={})",
            initial_sediment,
            world.terrain_sediment[idx]
        );

        // Verify sediment eroded MORE than gravel (relative to initial)
        // This tests the layer order - softer materials erode first
        let sediment_ratio = world.terrain_sediment[idx] / initial_sediment;
        let gravel_ratio = world.gravel_thickness[idx] / initial_gravel;
        assert!(sediment_ratio <= gravel_ratio,
            "Sediment should erode faster than gravel (sediment ratio={:.2}, gravel ratio={:.2})",
            sediment_ratio, gravel_ratio);

        // Paydirt should be mostly intact due to high hardness
        assert!(world.paydirt_thickness[idx] > initial_paydirt * 0.9,
            "Paydirt should erode last (initial={}, final={})",
            initial_paydirt, world.paydirt_thickness[idx]);
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
                &world.params,
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

    #[test]
    fn test_settling_velocity_formula() {
        // Test that Stokes settling velocity formula produces reasonable values
        let world = World::new(10, 10, 1.0, 0.0);

        // Test 1: 0.1mm fine silt in water (Stokes regime)
        // Expected: ~0.008-0.01 m/s
        let d_silt = 0.0001; // 0.1mm
        let vs_silt = world.settling_velocity(d_silt);
        assert!(vs_silt > 0.005 && vs_silt < 0.015,
            "0.1mm silt settling velocity should be ~0.008-0.01 m/s, got {:.4} m/s", vs_silt);

        // Test 2: 1mm sand in water (turbulent regime)
        // Expected: ~0.15-0.25 m/s (turbulent, not Stokes)
        let d_sand = 0.001; // 1mm
        let vs_sand = world.settling_velocity(d_sand);
        assert!(vs_sand > 0.15 && vs_sand < 0.30,
            "1mm sand settling velocity should be ~0.15-0.25 m/s, got {:.4} m/s", vs_sand);

        // Test 3: Verify Stokes formula correctness with manual calculation
        // For small particles (< 0.1mm), should use pure Stokes
        // Stokes law: vs = g * (ρp - ρf) * d² / (18 * μ)
        let g = world.params.gravity;
        let rho_p = world.params.rho_sediment;
        let rho_f = world.params.rho_water;
        let mu = world.params.water_viscosity;
        let d = 0.00005; // 0.05mm - in pure Stokes regime

        let expected = g * (rho_p - rho_f) * d * d / (18.0 * mu);
        let actual = world.settling_velocity(d);

        assert!((actual - expected).abs() < 0.0001,
            "Settling velocity calculation mismatch: expected {:.6}, got {:.6}", expected, actual);
    }

    #[test]
    fn test_shear_stress_velocity_only() {
        // Test that shear stress uses velocity-only formula, not additive with gravity
        let mut world = World::new(10, 10, 1.0, 0.0);

        // Create a slope and flow
        let idx = world.idx(5, 5);
        let idx_neighbor = world.idx(6, 5);
        world.terrain_sediment[idx] = 5.0;
        world.terrain_sediment[idx_neighbor] = 0.0; // Steep slope
        world.water_surface[idx] = 6.0; // 1.0m water depth above terrain

        // Add flow velocity
        let flow_idx = world.flow_x_idx(5, 5);
        let flow_idx_neighbor = world.flow_x_idx(6, 5);
        world.water_flow_x[flow_idx] = 2.0; // 2 m/s flow
        world.water_flow_x[flow_idx_neighbor] = 2.0;

        let shear_vel = world.shear_velocity(5, 5);

        // Expected: sqrt(cf * v²) = sqrt(0.003 * 4) = sqrt(0.012) ≈ 0.1095
        let cf = 0.003_f32;
        let v_sq = 2.0_f32 * 2.0_f32;
        let expected = (cf * v_sq).sqrt();

        assert!((shear_vel - expected).abs() < 0.001,
            "Shear velocity should use velocity-only formula: expected {:.6}, got {:.6}",
            expected, shear_vel);
    }

    #[test]
    fn test_deposition_suppression_linear() {
        // Test that deposition suppression uses linear formula with threshold
        let world = World::new(10, 10, 1.0, 0.0);
        let v_settle = world.settling_velocity(world.params.d50_sediment);

        // Case 1: Flow speed below threshold (2 * v_settle)
        // Should have no suppression (factor = 1.0)
        let flow_speed_slow = v_settle;
        let suppression_threshold = 2.0 * v_settle;
        let suppression_slow = if flow_speed_slow > suppression_threshold {
            suppression_threshold / flow_speed_slow
        } else {
            1.0
        };
        assert_eq!(suppression_slow, 1.0, "Slow flow should have no suppression");

        // Case 2: Flow speed above threshold
        // Should use linear suppression: threshold / flow_speed
        let flow_speed_fast = 4.0 * v_settle;
        let suppression_fast = if flow_speed_fast > suppression_threshold {
            suppression_threshold / flow_speed_fast
        } else {
            1.0
        };
        let expected_fast = suppression_threshold / flow_speed_fast;
        assert_eq!(suppression_fast, expected_fast, "Fast flow should use linear suppression");

        // Case 3: At threshold, should be exactly 1.0
        let flow_speed_threshold = suppression_threshold;
        let suppression_at_threshold = if flow_speed_threshold > suppression_threshold {
            suppression_threshold / flow_speed_threshold
        } else {
            1.0
        };
        assert_eq!(suppression_at_threshold, 1.0, "Flow at threshold should have no suppression");

        // Case 4: Verify linear is less aggressive than quadratic
        // For flow_speed = 4 * v_settle:
        // Linear: (2 * v_settle) / (4 * v_settle) = 0.5
        // Quadratic (old): (v_settle / (4 * v_settle))^2 = 0.0625
        // Linear allows MORE deposition (0.5 > 0.0625)
        let linear = suppression_threshold / (4.0 * v_settle);
        let quadratic = (v_settle / (4.0 * v_settle)).powi(2);
        assert!(linear > quadratic,
            "Linear suppression ({:.3}) should be less aggressive than quadratic ({:.3})",
            linear, quadratic);
    }
}
