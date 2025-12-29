//! Two-way pressure coupling.
//!
//! Mixture-density pressure gradients and porosity drag.

#![allow(dead_code)]

use super::FlipSimulation;
use crate::grid::CellType;

impl FlipSimulation {
    /// Apply pressure gradient with mixture density for two-way coupling
    ///
    /// Higher sand fraction -> higher mixture density -> smaller acceleration
    /// This is what causes sand-laden flow to move slower than clear water
    pub(crate) fn apply_pressure_gradient_two_way_impl(&mut self, _dt: f32) {
        let scale = 1.0 / self.grid.cell_size;

        // Density constants
        const WATER_DENSITY: f32 = 1.0;
        const SAND_DENSITY: f32 = 2.65;

        // Update U velocities (horizontal)
        for j in 0..self.grid.height {
            for i in 1..self.grid.width {
                let idx_left = self.grid.cell_index(i - 1, j);
                let idx_right = self.grid.cell_index(i, j);

                let left_type = self.grid.cell_type[idx_left];
                let right_type = self.grid.cell_type[idx_right];

                let u_idx = self.grid.u_index(i, j);
                if left_type == CellType::Solid || right_type == CellType::Solid {
                    self.grid.u[u_idx] = 0.0;
                } else if left_type == CellType::Fluid || right_type == CellType::Fluid {
                    // Compute sand fraction at this face
                    let sand_vol = self.sand_volume_u[u_idx];
                    let water_vol = self.water_volume_u[u_idx];
                    let total_vol = sand_vol + water_vol;

                    // Mixture density: rho_mix = rho_water * (1 - phi) + rho_sand * phi
                    // Capped at 1.5 so water can still push through sand accumulations
                    let rho_mix = if total_vol > 0.0 {
                        let sand_frac = sand_vol / total_vol;
                        let raw_rho = WATER_DENSITY * (1.0 - sand_frac) + SAND_DENSITY * sand_frac;
                        raw_rho.min(1.5) // Cap at 1.5x water density
                    } else {
                        WATER_DENSITY // Default to water if no particles
                    };

                    let grad =
                        (self.grid.pressure[idx_right] - self.grid.pressure[idx_left]) * scale;
                    self.grid.u[u_idx] -= grad / rho_mix;
                }
            }
        }

        // Update V velocities (vertical)
        for j in 1..self.grid.height {
            for i in 0..self.grid.width {
                let idx_bottom = self.grid.cell_index(i, j - 1);
                let idx_top = self.grid.cell_index(i, j);

                let bottom_type = self.grid.cell_type[idx_bottom];
                let top_type = self.grid.cell_type[idx_top];

                let v_idx = self.grid.v_index(i, j);
                if bottom_type == CellType::Solid || top_type == CellType::Solid {
                    self.grid.v[v_idx] = 0.0;
                } else if bottom_type == CellType::Fluid || top_type == CellType::Fluid {
                    // Compute sand fraction at this face
                    let sand_vol = self.sand_volume_v[v_idx];
                    let water_vol = self.water_volume_v[v_idx];
                    let total_vol = sand_vol + water_vol;

                    // Mixture density (capped at 1.5)
                    let rho_mix = if total_vol > 0.0 {
                        let sand_frac = sand_vol / total_vol;
                        let raw_rho = WATER_DENSITY * (1.0 - sand_frac) + SAND_DENSITY * sand_frac;
                        raw_rho.min(1.5)
                    } else {
                        WATER_DENSITY
                    };

                    let grad =
                        (self.grid.pressure[idx_top] - self.grid.pressure[idx_bottom]) * scale;
                    self.grid.v[v_idx] -= grad / rho_mix;
                }
            }
        }
    }

    /// Apply porosity-based drag to grid velocities
    ///
    /// In dense particle regions, water experiences resistance (Darcy flow).
    /// This replaces rigid cell-based deposition with continuous drag:
    /// - Low particle density -> water flows freely
    /// - High particle density -> water slows/stops (pile acts like porous wall)
    ///
    /// The drag is applied exponentially: v *= exp(-drag_rate * sand_fraction * dt)
    /// At high sand fraction (~0.6), velocity decays rapidly toward zero.
    pub(crate) fn apply_porosity_drag_impl(&mut self, dt: f32) {
        // Drag coefficient: controls how strongly particles resist flow
        // Higher = more resistance, piles act more like solid walls
        // Typical range: 50-200 for visible pile effect
        const DRAG_COEFFICIENT: f32 = 100.0;

        // Threshold: below this sand fraction, no drag (sparse particles don't block)
        const DRAG_THRESHOLD: f32 = 0.2;

        // Maximum sand fraction (random close packing limit)
        const MAX_SAND_FRACTION: f32 = 0.64;

        // Apply drag to U velocities (horizontal)
        for j in 0..self.grid.height {
            for i in 1..self.grid.width {
                let u_idx = self.grid.u_index(i, j);
                let sand_vol = self.sand_volume_u[u_idx];
                let total_vol = sand_vol + self.water_volume_u[u_idx];

                if total_vol > 0.0 {
                    // Sand fraction by volume
                    let sand_frac = (sand_vol / total_vol).min(MAX_SAND_FRACTION);

                    // Only apply drag above threshold
                    if sand_frac > DRAG_THRESHOLD {
                        // Normalized fraction above threshold (0 at threshold, 1 at max)
                        let excess =
                            (sand_frac - DRAG_THRESHOLD) / (MAX_SAND_FRACTION - DRAG_THRESHOLD);

                        // Exponential decay: stronger drag at higher concentration
                        // Using squared excess for nonlinear response (gentle at low, strong at high)
                        let drag_rate = DRAG_COEFFICIENT * excess * excess;
                        let damping = (-drag_rate * dt).exp();

                        self.grid.u[u_idx] *= damping;
                    }
                }
            }
        }

        // Apply drag to V velocities (vertical)
        for j in 1..self.grid.height {
            for i in 0..self.grid.width {
                let v_idx = self.grid.v_index(i, j);
                let sand_vol = self.sand_volume_v[v_idx];
                let total_vol = sand_vol + self.water_volume_v[v_idx];

                if total_vol > 0.0 {
                    let sand_frac = (sand_vol / total_vol).min(MAX_SAND_FRACTION);

                    if sand_frac > DRAG_THRESHOLD {
                        let excess =
                            (sand_frac - DRAG_THRESHOLD) / (MAX_SAND_FRACTION - DRAG_THRESHOLD);
                        let drag_rate = DRAG_COEFFICIENT * excess * excess;
                        let damping = (-drag_rate * dt).exp();

                        self.grid.v[v_idx] *= damping;
                    }
                }
            }
        }
    }
}
