//! Water Heightfield Mesh Renderer
//!
//! Renders water as a heightfield mesh with:
//! - Marching squares for organic boundary edges
//! - Physics-based wave displacement from velocity field
//! - Velocity-based foam coloring (fast water = pale/foamy)
//! - Temporal smoothing for stable surfaces
//! - Presence-based edge fading for smooth transitions
//!
//! # Usage
//!
//! ```rust,ignore
//! // Create renderer with grid dimensions
//! let mut renderer = WaterHeightfieldRenderer::new(
//!     grid_width,
//!     grid_depth,
//!     cell_size,
//!     WaterRenderConfig::default(),
//! );
//!
//! // Each frame, build the mesh from particles
//! let vertex_count = renderer.build_mesh(
//!     &water_particles,  // Iterator of (position, velocity)
//!     simulation_time,
//!     |x| floor_height_at_x,  // Floor height function
//! );
//!
//! // Get vertices for GPU upload
//! let vertices = renderer.vertices();
//! ```

use bytemuck::{Pod, Zeroable};

/// Vertex format for water surface mesh
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct WaterVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Configuration for water heightfield rendering
#[derive(Clone, Debug)]
pub struct WaterRenderConfig {
    /// Subdivision level per grid cell (2 = 2x2 = 4 sub-cells)
    pub subdivisions: usize,
    /// Maximum water height (particles above this are ignored)
    pub max_height: f32,

    // Wave parameters
    /// Base wave amplitude (always present)
    pub base_wave_amplitude: f32,
    /// Additional amplitude per m/s of flow velocity
    pub velocity_wave_scale: f32,
    /// High-frequency chop amplitude per m/s
    pub chop_scale: f32,
    /// Turbulence/splash amplitude scale
    pub turbulence_scale: f32,
    /// Main wave frequency
    pub wave_freq: f32,
    /// High-frequency chop frequency
    pub chop_freq: f32,
    /// Wave animation speed multiplier
    pub wave_speed_mult: f32,

    // Smoothing parameters
    /// Temporal blend factor (0.3 = 30% new, 70% old)
    pub temporal_blend: f32,
    /// Presence rise rate per frame (edge fade-in speed)
    pub presence_rise_rate: f32,
    /// Presence fall rate per frame (edge fade-out speed)
    pub presence_fall_rate: f32,
    /// Presence threshold for water detection
    pub presence_threshold: f32,

    // Color parameters
    /// Shallow water color [R, G, B, A]
    pub shallow_color: [f32; 4],
    /// Deep water color [R, G, B, A]
    pub deep_color: [f32; 4],
    /// Foam color for fast-moving water [R, G, B, A]
    pub foam_color: [f32; 4],
    /// Speed at which foam is fully applied (m/s)
    pub foam_full_speed: f32,
    /// Water depth for full deep color (in cell units)
    pub depth_color_cells: f32,
}

impl Default for WaterRenderConfig {
    fn default() -> Self {
        Self {
            subdivisions: 2,
            max_height: f32::MAX,

            // Wave parameters - "rushing water" feel
            base_wave_amplitude: 0.001,
            velocity_wave_scale: 0.003,
            chop_scale: 0.002,
            turbulence_scale: 0.002,
            wave_freq: 5.0,
            chop_freq: 25.0,
            wave_speed_mult: 8.0,

            // Smoothing
            temporal_blend: 0.3,
            presence_rise_rate: 0.15,
            presence_fall_rate: 0.08,
            presence_threshold: 0.1,

            // Colors
            shallow_color: [0.15, 0.45, 0.85, 0.55],
            deep_color: [0.06, 0.25, 0.60, 0.70],
            foam_color: [0.80, 0.88, 0.95, 0.65],
            foam_full_speed: 1.5,
            depth_color_cells: 3.0,
        }
    }
}

/// Water heightfield mesh renderer
pub struct WaterHeightfieldRenderer {
    width: usize,
    depth: usize,
    cell_size: f32,
    config: WaterRenderConfig,

    // Per-cell data
    heightfield: Vec<f32>,
    heightfield_smoothed: Vec<f32>,
    presence: Vec<f32>,
    vel_x: Vec<f32>,
    vel_z: Vec<f32>,
    vel_y: Vec<f32>,

    // Output
    vertices: Vec<WaterVertex>,
}

impl WaterHeightfieldRenderer {
    /// Create a new water heightfield renderer
    pub fn new(width: usize, depth: usize, cell_size: f32, config: WaterRenderConfig) -> Self {
        let cell_count = width * depth;
        let max_vertices = width * depth * 6 * (config.subdivisions * config.subdivisions * 2);

        Self {
            width,
            depth,
            cell_size,
            config,
            heightfield: vec![f32::NEG_INFINITY; cell_count],
            heightfield_smoothed: vec![f32::NEG_INFINITY; cell_count],
            presence: vec![0.0; cell_count],
            vel_x: vec![0.0; cell_count],
            vel_z: vec![0.0; cell_count],
            vel_y: vec![0.0; cell_count],
            vertices: Vec::with_capacity(max_vertices),
        }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: WaterRenderConfig) {
        self.config = config;
    }

    /// Get the output vertices
    pub fn vertices(&self) -> &[WaterVertex] {
        &self.vertices
    }

    /// Build the water mesh from particle data
    ///
    /// # Arguments
    /// * `particles` - Iterator yielding (position, velocity) tuples for water particles
    /// * `time` - Current simulation time for wave animation
    /// * `floor_height_fn` - Function returning floor height at a given X position
    ///
    /// # Returns
    /// Number of vertices generated
    pub fn build_mesh<I, F>(
        &mut self,
        particles: I,
        time: f32,
        floor_height_fn: F,
    ) -> usize
    where
        I: Iterator<Item = ([f32; 3], [f32; 3])>,
        F: Fn(f32) -> f32,
    {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let config = &self.config;

        // Reset heightfield
        self.heightfield.fill(f32::NEG_INFINITY);
        self.vertices.clear();

        // Velocity accumulation
        let mut vel_sum_x = vec![0.0f32; width * depth];
        let mut vel_sum_z = vec![0.0f32; width * depth];
        let mut vel_sum_y = vec![0.0f32; width * depth];
        let mut vel_count = vec![0u32; width * depth];

        // First pass: compute heightfield and velocity per cell
        for (pos, vel) in particles {
            let i = (pos[0] / cell_size).floor() as i32;
            let k = (pos[2] / cell_size).floor() as i32;
            if i >= 0 && i < width as i32 && k >= 0 && k < depth as i32 {
                let idx = k as usize * width + i as usize;
                let y = pos[1];

                // Clamp to max height to prevent outlier spikes
                if y > self.heightfield[idx] && y < config.max_height {
                    self.heightfield[idx] = y;
                }
                if y < config.max_height {
                    vel_sum_x[idx] += vel[0];
                    vel_sum_z[idx] += vel[2];
                    vel_sum_y[idx] += vel[1];
                    vel_count[idx] += 1;
                }
            }
        }

        // Compute raw average velocity per cell
        let mut vel_x_raw = vec![0.0f32; width * depth];
        let mut vel_z_raw = vec![0.0f32; width * depth];
        let mut vel_y_raw = vec![0.0f32; width * depth];
        for idx in 0..(width * depth) {
            if vel_count[idx] > 0 {
                let n = vel_count[idx] as f32;
                vel_x_raw[idx] = vel_sum_x[idx] / n;
                vel_z_raw[idx] = vel_sum_z[idx] / n;
                vel_y_raw[idx] = vel_sum_y[idx] / n;
            }
        }

        // Smooth velocity field (3x3 average)
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                if vel_count[idx] == 0 {
                    continue;
                }
                let mut sum_x = 0.0;
                let mut sum_z = 0.0;
                let mut sum_y = 0.0;
                let mut count = 0;
                for dk in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = i as i32 + di;
                        let nk = k as i32 + dk;
                        if ni >= 0 && ni < width as i32 && nk >= 0 && nk < depth as i32 {
                            let nidx = nk as usize * width + ni as usize;
                            if vel_count[nidx] > 0 {
                                sum_x += vel_x_raw[nidx];
                                sum_z += vel_z_raw[nidx];
                                sum_y += vel_y_raw[nidx];
                                count += 1;
                            }
                        }
                    }
                }
                if count > 0 {
                    self.vel_x[idx] = sum_x / count as f32;
                    self.vel_z[idx] = sum_z / count as f32;
                    self.vel_y[idx] = sum_y / count as f32;
                }
            }
        }

        // Smooth heightfield (3x3 spatial average)
        let mut smoothed = vec![f32::NEG_INFINITY; width * depth];
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let center = self.heightfield[idx];
                if !center.is_finite() {
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0;
                for dk in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = i as i32 + di;
                        let nk = k as i32 + dk;
                        if ni >= 0 && ni < width as i32 && nk >= 0 && nk < depth as i32 {
                            let nidx = nk as usize * width + ni as usize;
                            let h = self.heightfield[nidx];
                            if h.is_finite() {
                                sum += h;
                                count += 1;
                            }
                        }
                    }
                }
                if count > 0 {
                    smoothed[idx] = sum / count as f32;
                }
            }
        }

        // Temporal smoothing and presence update
        for idx in 0..(width * depth) {
            let new_h = smoothed[idx];
            let old_h = self.heightfield_smoothed[idx];

            if new_h.is_finite() && old_h.is_finite() {
                smoothed[idx] = old_h * (1.0 - config.temporal_blend) + new_h * config.temporal_blend;
            }
            self.heightfield_smoothed[idx] = smoothed[idx];

            // Update presence
            let has_water_now = new_h.is_finite();
            let old_presence = self.presence[idx];
            let target = if has_water_now { 1.0 } else { 0.0 };
            let new_presence = if target > old_presence {
                (old_presence + config.presence_rise_rate).min(1.0)
            } else {
                (old_presence - config.presence_fall_rate).max(0.0)
            };
            self.presence[idx] = new_presence;
        }

        // Build mesh using marching squares
        self.build_marching_squares_mesh(&smoothed, &floor_height_fn, time);

        self.vertices.len()
    }

    fn build_marching_squares_mesh<F>(&mut self, smoothed: &[f32], floor_height_fn: &F, time: f32)
    where
        F: Fn(f32) -> f32,
    {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let config = &self.config;

        // Helper closures
        let has_water = |i: i32, k: i32| -> bool {
            if i < 0 || i >= width as i32 || k < 0 || k >= depth as i32 {
                return false;
            }
            self.presence[k as usize * width + i as usize] > config.presence_threshold
        };

        let get_presence = |i: i32, k: i32| -> f32 {
            if i < 0 || i >= width as i32 || k < 0 || k >= depth as i32 {
                return 0.0;
            }
            self.presence[k as usize * width + i as usize]
        };

        let get_velocity_at = |x: f32, z: f32| -> (f32, f32, f32) {
            let fi = x / cell_size;
            let fk = z / cell_size;
            let i = fi.floor() as i32;
            let k = fk.floor() as i32;

            if i < 0 || i >= width as i32 - 1 || k < 0 || k >= depth as i32 - 1 {
                let ci = i.clamp(0, width as i32 - 1) as usize;
                let ck = k.clamp(0, depth as i32 - 1) as usize;
                let idx = ck * width + ci;
                return (self.vel_x[idx], self.vel_z[idx], self.vel_y[idx]);
            }

            let fx = fi - i as f32;
            let fz = fk - k as f32;
            let i = i as usize;
            let k = k as usize;

            let idx00 = k * width + i;
            let idx10 = k * width + i + 1;
            let idx01 = (k + 1) * width + i;
            let idx11 = (k + 1) * width + i + 1;

            let vx = self.vel_x[idx00] * (1.0 - fx) * (1.0 - fz)
                   + self.vel_x[idx10] * fx * (1.0 - fz)
                   + self.vel_x[idx01] * (1.0 - fx) * fz
                   + self.vel_x[idx11] * fx * fz;
            let vz = self.vel_z[idx00] * (1.0 - fx) * (1.0 - fz)
                   + self.vel_z[idx10] * fx * (1.0 - fz)
                   + self.vel_z[idx01] * (1.0 - fx) * fz
                   + self.vel_z[idx11] * fx * fz;
            let vy = self.vel_y[idx00] * (1.0 - fx) * (1.0 - fz)
                   + self.vel_y[idx10] * fx * (1.0 - fz)
                   + self.vel_y[idx01] * (1.0 - fx) * fz
                   + self.vel_y[idx11] * fx * fz;

            (vx, vz, vy)
        };

        let wave_offset = |x: f32, z: f32, vx: f32, vz: f32, vy: f32| -> f32 {
            let speed = (vx * vx + vz * vz).sqrt();
            let vert_speed = vy.abs();

            let (dir_x, dir_z) = if speed > 0.05 {
                (vx / speed, vz / speed)
            } else {
                (1.0, 0.0)
            };

            let flow_dist = dir_x * x + dir_z * z;
            let cross_dist = -dir_z * x + dir_x * z;

            let main_amp = config.base_wave_amplitude + speed * config.velocity_wave_scale;
            let main_phase = flow_dist * config.wave_freq - time * (2.0 + speed * config.wave_speed_mult);
            let main_wave = main_phase.sin() * main_amp;

            let chop_amp = speed * config.chop_scale;
            let chop1 = (flow_dist * config.chop_freq - time * speed * 15.0).sin() * chop_amp;
            let chop2 = (cross_dist * config.chop_freq * 0.8 + time * 6.0).sin() * chop_amp * 0.5;
            let chop3 = ((flow_dist + cross_dist) * config.chop_freq * 0.6 - time * 10.0).sin() * chop_amp * 0.3;

            let cross_amp = main_amp * 0.4;
            let cross_wave = (cross_dist * config.wave_freq * 0.7 + time * 2.5).sin() * cross_amp;

            let splash = if vert_speed > 0.15 {
                (x * 30.0 + z * 35.0 + time * 12.0).sin() * vert_speed * config.turbulence_scale
            } else {
                0.0
            };

            main_wave + cross_wave + chop1 + chop2 + chop3 + splash
        };

        let get_height_at = |x: f32, z: f32, base_h: f32| -> f32 {
            let (vx, vz, vy) = get_velocity_at(x, z);
            base_h + wave_offset(x, z, vx, vz, vy)
        };

        let get_corner_height = |ci: usize, ck: usize| -> Option<f32> {
            let mut sum = 0.0;
            let mut count = 0;
            for dk in 0..=1 {
                for di in 0..=1 {
                    if ci >= di && ck >= dk {
                        let cell_i = ci - di;
                        let cell_k = ck - dk;
                        if cell_i < width && cell_k < depth {
                            let idx = cell_k * width + cell_i;
                            let h = smoothed[idx];
                            if h.is_finite() {
                                sum += h;
                                count += 1;
                            }
                        }
                    }
                }
            }
            if count > 0 { Some(sum / count as f32) } else { None }
        };

        let calc_color = |i: usize, k: usize, center_h: f32, speed: f32| -> [f32; 4] {
            let floor_height = floor_height_fn(i as f32 * cell_size);
            let water_depth = (center_h - floor_height).max(0.0);
            let depth_factor = (water_depth / (config.depth_color_cells * cell_size)).min(1.0);

            let cell_presence = get_presence(i as i32, k as i32);

            let base_r = config.shallow_color[0] * (1.0 - depth_factor) + config.deep_color[0] * depth_factor;
            let base_g = config.shallow_color[1] * (1.0 - depth_factor) + config.deep_color[1] * depth_factor;
            let base_b = config.shallow_color[2] * (1.0 - depth_factor) + config.deep_color[2] * depth_factor;
            let base_a = config.shallow_color[3] * (1.0 - depth_factor) + config.deep_color[3] * depth_factor;

            let foam = (speed / config.foam_full_speed).min(1.0);
            let alpha = (base_a * (1.0 - foam * 0.5) + config.foam_color[3] * foam * 0.5) * cell_presence;

            [
                base_r * (1.0 - foam) + config.foam_color[0] * foam,
                base_g * (1.0 - foam) + config.foam_color[1] * foam,
                base_b * (1.0 - foam) + config.foam_color[2] * foam,
                alpha,
            ]
        };

        // Main marching squares loop
        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let center_h = smoothed[idx];

                let c0 = has_water(i as i32, k as i32);
                let c1 = has_water(i as i32 + 1, k as i32);
                let c2 = has_water(i as i32, k as i32 + 1);
                let c3 = has_water(i as i32 + 1, k as i32 + 1);

                let case = (c0 as u8) | ((c1 as u8) << 1) | ((c2 as u8) << 2) | ((c3 as u8) << 3);

                if case == 0 {
                    continue;
                }

                let h00 = get_corner_height(i, k).unwrap_or(center_h);
                let h10 = get_corner_height(i + 1, k).unwrap_or(center_h);
                let h01 = get_corner_height(i, k + 1).unwrap_or(center_h);
                let h11 = get_corner_height(i + 1, k + 1).unwrap_or(center_h);
                let h_center = (h00 + h10 + h01 + h11) / 4.0;

                let x0 = i as f32 * cell_size;
                let x1 = (i + 1) as f32 * cell_size;
                let xm = (x0 + x1) * 0.5;
                let z0 = k as f32 * cell_size;
                let z1 = (k + 1) as f32 * cell_size;
                let zm = (z0 + z1) * 0.5;

                let y00 = get_height_at(x0, z0, h00);
                let y10 = get_height_at(x1, z0, h10);
                let y01 = get_height_at(x0, z1, h01);
                let y11 = get_height_at(x1, z1, h11);
                let ym0 = get_height_at(xm, z0, (h00 + h10) * 0.5);
                let ym1 = get_height_at(xm, z1, (h01 + h11) * 0.5);
                let y0m = get_height_at(x0, zm, (h00 + h01) * 0.5);
                let y1m = get_height_at(x1, zm, (h10 + h11) * 0.5);
                let ymm = get_height_at(xm, zm, h_center);

                let cell_speed = (self.vel_x[idx] * self.vel_x[idx] + self.vel_z[idx] * self.vel_z[idx]).sqrt();
                let color = calc_color(i, k, center_h, cell_speed);

                let v00 = WaterVertex { position: [x0, y00, z0], color };
                let v10 = WaterVertex { position: [x1, y10, z0], color };
                let v01 = WaterVertex { position: [x0, y01, z1], color };
                let v11 = WaterVertex { position: [x1, y11, z1], color };
                let vm0 = WaterVertex { position: [xm, ym0, z0], color };
                let vm1 = WaterVertex { position: [xm, ym1, z1], color };
                let v0m = WaterVertex { position: [x0, y0m, zm], color };
                let v1m = WaterVertex { position: [x1, y1m, zm], color };
                let vmm = WaterVertex { position: [xm, ymm, zm], color };

                // Generate triangles based on marching squares case
                match case {
                    1 => self.vertices.extend_from_slice(&[v00, vm0, v0m]),
                    2 => self.vertices.extend_from_slice(&[vm0, v10, v1m]),
                    4 => self.vertices.extend_from_slice(&[v0m, vm1, v01]),
                    8 => self.vertices.extend_from_slice(&[v1m, v11, vm1]),
                    3 => self.vertices.extend_from_slice(&[v00, v10, v1m, v00, v1m, v0m]),
                    5 => self.vertices.extend_from_slice(&[v00, vm0, vm1, v00, vm1, v01]),
                    10 => self.vertices.extend_from_slice(&[vm0, v10, v11, vm0, v11, vm1]),
                    12 => self.vertices.extend_from_slice(&[v0m, v1m, v11, v0m, v11, v01]),
                    6 => self.vertices.extend_from_slice(&[vm0, v10, v1m, v1m, vmm, vm0, v0m, vm1, v01, v0m, vmm, vm1]),
                    9 => self.vertices.extend_from_slice(&[v00, vm0, v0m, vm0, vmm, v0m, v1m, v11, vm1, vmm, v1m, vm1]),
                    7 => self.vertices.extend_from_slice(&[v00, v10, v1m, v00, v1m, vmm, v00, vmm, vm1, v00, vm1, v01]),
                    11 => self.vertices.extend_from_slice(&[v00, v10, v11, v00, v11, vm1, v00, vm1, vmm, v00, vmm, v0m]),
                    13 => self.vertices.extend_from_slice(&[v00, vm0, vmm, v00, vmm, v1m, v00, v1m, v11, v00, v11, v01]),
                    14 => self.vertices.extend_from_slice(&[vm0, v10, v11, vm0, v11, v01, vm0, v01, v0m, vm0, v0m, vmm]),
                    15 => {
                        // Full cell - use subdivision
                        let n = config.subdivisions;
                        let sub_size = cell_size / n as f32;

                        for sk in 0..n {
                            for si in 0..n {
                                let u0 = si as f32 / n as f32;
                                let u1 = (si + 1) as f32 / n as f32;
                                let v0 = sk as f32 / n as f32;
                                let v1 = (sk + 1) as f32 / n as f32;

                                let sx0 = x0 + si as f32 * sub_size;
                                let sx1 = x0 + (si + 1) as f32 * sub_size;
                                let sz0 = z0 + sk as f32 * sub_size;
                                let sz1 = z0 + (sk + 1) as f32 * sub_size;
                                let sxm = (sx0 + sx1) * 0.5;
                                let szm = (sz0 + sz1) * 0.5;

                                let bh00 = h00 * (1.0 - u0) * (1.0 - v0) + h10 * u0 * (1.0 - v0)
                                         + h01 * (1.0 - u0) * v0 + h11 * u0 * v0;
                                let bh10 = h00 * (1.0 - u1) * (1.0 - v0) + h10 * u1 * (1.0 - v0)
                                         + h01 * (1.0 - u1) * v0 + h11 * u1 * v0;
                                let bh01 = h00 * (1.0 - u0) * (1.0 - v1) + h10 * u0 * (1.0 - v1)
                                         + h01 * (1.0 - u0) * v1 + h11 * u0 * v1;
                                let bh11 = h00 * (1.0 - u1) * (1.0 - v1) + h10 * u1 * (1.0 - v1)
                                         + h01 * (1.0 - u1) * v1 + h11 * u1 * v1;
                                let bhm = h00 * (1.0 - (u0 + u1) * 0.5) * (1.0 - (v0 + v1) * 0.5)
                                        + h10 * (u0 + u1) * 0.5 * (1.0 - (v0 + v1) * 0.5)
                                        + h01 * (1.0 - (u0 + u1) * 0.5) * (v0 + v1) * 0.5
                                        + h11 * (u0 + u1) * 0.5 * (v0 + v1) * 0.5;

                                let sy00 = get_height_at(sx0, sz0, bh00);
                                let sy10 = get_height_at(sx1, sz0, bh10);
                                let sy01 = get_height_at(sx0, sz1, bh01);
                                let sy11 = get_height_at(sx1, sz1, bh11);

                                let (sub_vx, sub_vz, _) = get_velocity_at(sxm, szm);
                                let sub_speed = (sub_vx * sub_vx + sub_vz * sub_vz).sqrt();
                                let sub_color = calc_color(i, k, bhm, sub_speed);

                                self.vertices.extend_from_slice(&[
                                    WaterVertex { position: [sx0, sy00, sz0], color: sub_color },
                                    WaterVertex { position: [sx1, sy10, sz0], color: sub_color },
                                    WaterVertex { position: [sx1, sy11, sz1], color: sub_color },
                                    WaterVertex { position: [sx0, sy00, sz0], color: sub_color },
                                    WaterVertex { position: [sx1, sy11, sz1], color: sub_color },
                                    WaterVertex { position: [sx0, sy01, sz1], color: sub_color },
                                ]);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Add side walls at water boundaries
        self.build_side_walls(smoothed, floor_height_fn);
    }

    fn build_side_walls<F>(&mut self, smoothed: &[f32], floor_height_fn: &F)
    where
        F: Fn(f32) -> f32,
    {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let config = &self.config;

        let has_water = |i: i32, k: i32| -> bool {
            if i < 0 || i >= width as i32 || k < 0 || k >= depth as i32 {
                return false;
            }
            self.presence[k as usize * width + i as usize] > config.presence_threshold
        };

        let get_presence = |i: i32, k: i32| -> f32 {
            if i < 0 || i >= width as i32 || k < 0 || k >= depth as i32 {
                return 0.0;
            }
            self.presence[k as usize * width + i as usize]
        };

        let get_corner_height = |ci: usize, ck: usize| -> Option<f32> {
            let mut sum = 0.0;
            let mut count = 0;
            for dk in 0..=1 {
                for di in 0..=1 {
                    if ci >= di && ck >= dk {
                        let cell_i = ci - di;
                        let cell_k = ck - dk;
                        if cell_i < width && cell_k < depth {
                            let idx = cell_k * width + cell_i;
                            let h = smoothed[idx];
                            if h.is_finite() {
                                sum += h;
                                count += 1;
                            }
                        }
                    }
                }
            }
            if count > 0 { Some(sum / count as f32) } else { None }
        };

        for k in 0..depth {
            for i in 0..width {
                let idx = k * width + i;
                let center_h = smoothed[idx];

                if !has_water(i as i32, k as i32) {
                    continue;
                }

                let h00 = get_corner_height(i, k).unwrap_or(center_h);
                let h10 = get_corner_height(i + 1, k).unwrap_or(center_h);
                let h01 = get_corner_height(i, k + 1).unwrap_or(center_h);
                let h11 = get_corner_height(i + 1, k + 1).unwrap_or(center_h);

                let x0 = i as f32 * cell_size;
                let x1 = (i + 1) as f32 * cell_size;
                let z0 = k as f32 * cell_size;
                let z1 = (k + 1) as f32 * cell_size;

                let cell_speed = (self.vel_x[idx] * self.vel_x[idx] + self.vel_z[idx] * self.vel_z[idx]).sqrt();
                let cell_presence = get_presence(i as i32, k as i32);

                let floor_height = floor_height_fn(x0);
                let water_depth = (center_h - floor_height).max(0.0);
                let depth_factor = (water_depth / (config.depth_color_cells * cell_size)).min(1.0);

                let base_r = config.shallow_color[0] * (1.0 - depth_factor) + config.deep_color[0] * depth_factor;
                let base_g = config.shallow_color[1] * (1.0 - depth_factor) + config.deep_color[1] * depth_factor;
                let base_b = config.shallow_color[2] * (1.0 - depth_factor) + config.deep_color[2] * depth_factor;
                let base_a = config.shallow_color[3] * (1.0 - depth_factor) + config.deep_color[3] * depth_factor;

                let foam = (cell_speed / config.foam_full_speed).min(1.0);
                let alpha = (base_a * (1.0 - foam * 0.5) + config.foam_color[3] * foam * 0.5) * cell_presence;

                let side_color = [
                    (base_r * (1.0 - foam) + config.foam_color[0] * foam) * 0.8,
                    (base_g * (1.0 - foam) + config.foam_color[1] * foam) * 0.8,
                    (base_b * (1.0 - foam) + config.foam_color[2] * foam) * 0.9,
                    alpha,
                ];

                let floor_y0 = floor_height_fn(x0);
                let floor_y1 = floor_height_fn(x1);

                // Front edge
                if k == 0 || !smoothed[(k - 1) * width + i].is_finite() {
                    self.vertices.extend_from_slice(&[
                        WaterVertex { position: [x0, floor_y0, z0], color: side_color },
                        WaterVertex { position: [x1, floor_y1, z0], color: side_color },
                        WaterVertex { position: [x1, h10, z0], color: side_color },
                        WaterVertex { position: [x0, floor_y0, z0], color: side_color },
                        WaterVertex { position: [x1, h10, z0], color: side_color },
                        WaterVertex { position: [x0, h00, z0], color: side_color },
                    ]);
                }

                // Back edge
                if k == depth - 1 || !smoothed[(k + 1) * width + i].is_finite() {
                    self.vertices.extend_from_slice(&[
                        WaterVertex { position: [x0, floor_y0, z1], color: side_color },
                        WaterVertex { position: [x0, h01, z1], color: side_color },
                        WaterVertex { position: [x1, h11, z1], color: side_color },
                        WaterVertex { position: [x0, floor_y0, z1], color: side_color },
                        WaterVertex { position: [x1, h11, z1], color: side_color },
                        WaterVertex { position: [x1, floor_y1, z1], color: side_color },
                    ]);
                }

                // Left edge
                if i == 0 || !smoothed[k * width + i - 1].is_finite() {
                    self.vertices.extend_from_slice(&[
                        WaterVertex { position: [x0, floor_y0, z0], color: side_color },
                        WaterVertex { position: [x0, h00, z0], color: side_color },
                        WaterVertex { position: [x0, h01, z1], color: side_color },
                        WaterVertex { position: [x0, floor_y0, z0], color: side_color },
                        WaterVertex { position: [x0, h01, z1], color: side_color },
                        WaterVertex { position: [x0, floor_y0, z1], color: side_color },
                    ]);
                }

                // Right edge
                if i == width - 1 || !smoothed[k * width + i + 1].is_finite() {
                    self.vertices.extend_from_slice(&[
                        WaterVertex { position: [x1, floor_y1, z0], color: side_color },
                        WaterVertex { position: [x1, floor_y1, z1], color: side_color },
                        WaterVertex { position: [x1, h11, z1], color: side_color },
                        WaterVertex { position: [x1, floor_y1, z0], color: side_color },
                        WaterVertex { position: [x1, h11, z1], color: side_color },
                        WaterVertex { position: [x1, h10, z0], color: side_color },
                    ]);
                }
            }
        }
    }

    /// Reset all state (for simulation restart)
    pub fn reset(&mut self) {
        self.heightfield.fill(f32::NEG_INFINITY);
        self.heightfield_smoothed.fill(f32::NEG_INFINITY);
        self.presence.fill(0.0);
        self.vel_x.fill(0.0);
        self.vel_z.fill(0.0);
        self.vel_y.fill(0.0);
        self.vertices.clear();
    }
}
