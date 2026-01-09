//! Vorticity computation and confinement.
//!
//! Preserves rotational flow structures.

use super::{CellType, Grid};

impl Grid {
    /// Compute vorticity (curl) field: omega = dv/dx - du/dy
    /// Stores result in self.vorticity for later use in confinement and diagnostics
    pub fn compute_vorticity(&mut self) {
        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = self.cell_index(i, j);

                if self.cell_type[idx] != CellType::Fluid {
                    self.vorticity[idx] = 0.0;
                    continue;
                }

                // Curl = dv/dx - du/dy using central differences
                let du_dy = (self.u[self.u_index(i, j + 1)] - self.u[self.u_index(i, j.saturating_sub(1))]) * 0.5;
                let dv_dx = (self.v[self.v_index(i + 1, j)] - self.v[self.v_index(i.saturating_sub(1), j)]) * 0.5;

                self.vorticity[idx] = dv_dx - du_dy;
            }
        }
    }

    /// Compute enstrophy: epsilon = (1/2) * integral(|omega|^2) dV
    /// Enstrophy measures total vorticity intensity in the fluid
    /// Higher enstrophy = more rotational motion
    pub fn compute_enstrophy(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        let mut enstrophy = 0.0f32;

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] == CellType::Fluid {
                    enstrophy += 0.5 * self.vorticity[idx] * self.vorticity[idx] * cell_area;
                }
            }
        }

        enstrophy
    }

    /// Compute total absolute vorticity: integral(|omega|) dV
    /// Useful for debugging and tracking vortex strength
    pub fn total_absolute_vorticity(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        let mut total = 0.0f32;

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] == CellType::Fluid {
                    total += self.vorticity[idx].abs() * cell_area;
                }
            }
        }

        total
    }

    /// Get maximum absolute vorticity in the field
    pub fn max_vorticity(&self) -> f32 {
        self.vorticity.iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
    }

    /// Apply vorticity confinement to maintain swirling motion
    /// Based on PavelDoGreat/WebGL-Fluid-Simulation
    ///
    /// Attenuates confinement near:
    /// - Solid surfaces (using SDF distance)
    /// - Dynamic pile surfaces (using pile_height)
    /// This prevents artificial turbulence at smooth boundaries.
    pub fn apply_vorticity_confinement(&mut self, dt: f32, strength: f32) {
        // Call with no pile (backward compatible)
        self.apply_vorticity_confinement_with_piles(dt, strength, &[]);
    }

    /// Apply vorticity confinement with pile-aware attenuation
    pub fn apply_vorticity_confinement_with_piles(&mut self, dt: f32, strength: f32, pile_height: &[f32]) {
        // First pass: compute curl (vorticity) at each cell and store it
        self.compute_vorticity();

        // Find the topmost fluid cell in each column (free surface)
        let mut surface_j: Vec<Option<usize>> = vec![None; self.width];
        for i in 0..self.width {
            for j in 0..self.height {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] == CellType::Fluid {
                    surface_j[i] = Some(j);
                    break;
                }
            }
        }

        // Use stored vorticity for confinement
        let curl = &self.vorticity;
        let dx = self.cell_size;

        // Cells within this depth of surface are skipped (prevents fake turbulence band)
        const SURFACE_SKIP_DEPTH: usize = 1;

        // Second pass: apply vorticity force
        for j in 2..self.height - 2 {
            for i in 2..self.width - 2 {
                let idx = self.cell_index(i, j);

                if self.cell_type[idx] != CellType::Fluid {
                    continue;
                }

                // === FREE SURFACE CHECK ===
                // Air is compressible - vorticity confinement only makes sense in
                // incompressible regions. Skip any fluid cell adjacent to air.
                let has_air_neighbor =
                    self.cell_type[self.cell_index(i, j - 1)] == CellType::Air ||
                    self.cell_type[self.cell_index(i, j + 1)] == CellType::Air ||
                    self.cell_type[self.cell_index(i - 1, j)] == CellType::Air ||
                    self.cell_type[self.cell_index(i + 1, j)] == CellType::Air;

                if has_air_neighbor {
                    continue; // Free surface - no confinement
                }

                // === TOP N CELLS SKIP ===
                // Skip vorticity confinement in top 2-3 cells per column
                // These cells create "fake turbulence bands" at the surface
                if let Some(surf) = surface_j[i] {
                    if j < surf + SURFACE_SKIP_DEPTH {
                        continue;
                    }
                }

                // === ATTENUATION NEAR SOLID/PILE SURFACES ===
                // Attenuate (don't skip) near solid walls and piles
                let cell_y = (j as f32 + 0.5) * dx;

                // Distance to pile surface
                let pile_atten = if i < pile_height.len() && pile_height[i] < f32::INFINITY {
                    let pile_y = pile_height[i];
                    let dist_to_pile = cell_y - pile_y; // positive = above pile
                    if dist_to_pile < 0.0 {
                        0.0 // Below pile - no confinement
                    } else {
                        (dist_to_pile / (3.0 * dx)).clamp(0.0, 1.0)
                    }
                } else {
                    1.0 // No pile in this column
                };

                if pile_atten < 0.01 {
                    continue;
                }

                // Gradient of curl magnitude
                let curl_l = curl[self.cell_index(i - 1, j)].abs();
                let curl_r = curl[self.cell_index(i + 1, j)].abs();
                let curl_b = curl[self.cell_index(i, j - 1)].abs();
                let curl_t = curl[self.cell_index(i, j + 1)].abs();

                let grad_x = (curl_r - curl_l) * 0.5;
                let grad_y = (curl_t - curl_b) * 0.5;

                let len = (grad_x * grad_x + grad_y * grad_y).sqrt() + 1e-5;
                let nx = grad_x / len;
                let ny = grad_y / len;

                // Force perpendicular to gradient, proportional to curl
                // F = epsilon * h * (N x omega) - scaled by grid spacing for grid independence
                // Reference: Fedkiw et al. 2001 "Visual Simulation of Smoke"
                let c = curl[idx];
                let h = self.cell_size;
                let fx = ny * c * strength * h * pile_atten;
                let fy = -nx * c * strength * h * pile_atten;

                // Apply to velocity
                let u_idx = self.u_index(i, j);
                let v_idx = self.v_index(i, j);
                self.u[u_idx] += fx * dt;
                self.v[v_idx] += fy * dt;
            }
        }
    }
}
