//! Pressure projection and multigrid solver.
//!
//! Enforces incompressibility via pressure solve.

#![allow(dead_code)]  // Methods will be used after Phase 3 migration

use super::{CellType, Grid};

// ============================================================================
// PHASE 3 MIGRATION: Uncomment this impl block after deleting originals from mod.rs
// Methods to delete from mod.rs:
//   - compute_divergence
//   - total_divergence
//   - pressure_stats
//   - solve_pressure
//   - update_pressure_cell
//   - compute_max_residual
//   - apply_pressure_gradient
//   - mg_sync_level_zero
//   - mg_copy_pressure_back
//   - mg_restrict
//   - mg_prolongate
//   - mg_compute_residual
//   - mg_smooth
//   - mg_clear_pressure
//   - mg_v_cycle
//   - solve_pressure_multigrid
// ============================================================================

// COMMENTED OUT - Enable in Phase 3 when originals are deleted from mod.rs
/*
impl Grid {
    /// Compute divergence of velocity field
    pub fn compute_divergence(&mut self) {
        let scale = 1.0 / self.cell_size;

        for j in 0..self.height {
            for i in 0..self.width {
                let idx = self.cell_index(i, j);

                if self.cell_type[idx] != CellType::Fluid {
                    self.divergence[idx] = 0.0;
                    continue;
                }

                let u_right = self.u[self.u_index(i + 1, j)];
                let u_left = self.u[self.u_index(i, j)];
                let v_top = self.v[self.v_index(i, j + 1)];
                let v_bottom = self.v[self.v_index(i, j)];

                self.divergence[idx] = (u_right - u_left + v_top - v_bottom) * scale;
            }
        }
    }

    /// Sum of absolute divergence across all fluid cells
    pub fn total_divergence(&self) -> f32 {
        self.divergence.iter().map(|d| d.abs()).sum()
    }

    /// Pressure statistics (min, max, avg) for debugging
    pub fn pressure_stats(&self) -> (f32, f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;
        let mut count = 0;
        for (idx, &p) in self.pressure.iter().enumerate() {
            if self.cell_type[idx] == CellType::Fluid {
                min = min.min(p);
                max = max.max(p);
                sum += p;
                count += 1;
            }
        }
        if count == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (min, max, sum / count as f32)
        }
    }

    /// Solve pressure using Red-Black Gauss-Seidel iteration
    /// 2x faster convergence than Jacobi - updates in-place using latest neighbor values
    /// This enforces incompressibility and creates vortices naturally
    ///
    /// Uses early termination when residual is below threshold, with a minimum
    /// of `iterations` and maximum of `iterations * 2` to ensure convergence
    /// for complex geometries while staying fast for simple cases.
    pub fn solve_pressure(&mut self, iterations: usize) {
        let h_sq = self.cell_size * self.cell_size;
        let convergence_threshold = 0.001; // Residual threshold for early exit
        let max_iterations = iterations * 2; // Allow more iterations if needed

        for iter in 0..max_iterations {
            // Red pass (i+j even) - can use updated values immediately
            for j in 1..self.height - 1 {
                for i in 1..self.width - 1 {
                    if (i + j) % 2 == 0 {
                        self.update_pressure_cell(i, j, h_sq);
                    }
                }
            }
            // Black pass (i+j odd) - uses updated red values
            for j in 1..self.height - 1 {
                for i in 1..self.width - 1 {
                    if (i + j) % 2 != 0 {
                        self.update_pressure_cell(i, j, h_sq);
                    }
                }
            }

            // Check convergence after minimum iterations
            if iter >= iterations - 1 {
                let max_residual = self.compute_max_residual(h_sq);
                if max_residual < convergence_threshold {
                    break;
                }
            }
        }
    }

    /// Update a single pressure cell (helper for Red-Black GS)
    #[inline]
    fn update_pressure_cell(&mut self, i: usize, j: usize, h_sq: f32) {
        let idx = self.cell_index(i, j);

        if self.cell_type[idx] != CellType::Fluid {
            self.pressure[idx] = 0.0;
            return;
        }

        let p = self.pressure[idx];

        // Neighbor pressures (solid boundaries use Neumann BC: dp/dn = 0)
        let p_left = if self.cell_type[self.cell_index(i - 1, j)] == CellType::Solid {
            p
        } else {
            self.pressure[self.cell_index(i - 1, j)]
        };
        let p_right = if self.cell_type[self.cell_index(i + 1, j)] == CellType::Solid {
            p
        } else {
            self.pressure[self.cell_index(i + 1, j)]
        };
        let p_bottom = if self.cell_type[self.cell_index(i, j - 1)] == CellType::Solid {
            p
        } else {
            self.pressure[self.cell_index(i, j - 1)]
        };
        let p_top = if self.cell_type[self.cell_index(i, j + 1)] == CellType::Solid {
            p
        } else {
            self.pressure[self.cell_index(i, j + 1)]
        };

        let div = self.divergence[idx];

        // Gauss-Seidel update for nabla^2 p = div
        // Discretized: (p_L + p_R + p_B + p_T - 4*p) / h^2 = div
        // Solving for p: p = (p_L + p_R + p_B + p_T - h^2*div) / 4
        self.pressure[idx] = (p_left + p_right + p_bottom + p_top - h_sq * div) * 0.25;
    }

    /// Compute maximum residual of pressure equation: |nabla^2 p - div|
    fn compute_max_residual(&self, h_sq: f32) -> f32 {
        let mut max_residual = 0.0f32;

        for j in 1..self.height - 1 {
            for i in 1..self.width - 1 {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] != CellType::Fluid {
                    continue;
                }

                let p = self.pressure[idx];
                let p_left = self.pressure[self.cell_index(i - 1, j)];
                let p_right = self.pressure[self.cell_index(i + 1, j)];
                let p_bottom = self.pressure[self.cell_index(i, j - 1)];
                let p_top = self.pressure[self.cell_index(i, j + 1)];

                // Laplacian: (p_L + p_R + p_B + p_T - 4*p) / h^2
                let laplacian = (p_left + p_right + p_bottom + p_top - 4.0 * p) / h_sq;
                let residual = (laplacian - self.divergence[idx]).abs();
                max_residual = max_residual.max(residual);
            }
        }

        max_residual
    }

    /// Subtract pressure gradient from velocity field
    /// Uses formulation: u -= nabla p_tilde / h where p_tilde is pseudo-pressure from nabla^2 p_tilde = div
    pub fn apply_pressure_gradient(&mut self, _dt: f32) {
        // CRITICAL FIX: Remove dt from scale. With dt, only 0.1% of divergence
        // was corrected per frame. Without dt, we get full correction.
        // let scale = 1.0 / self.cell_size;

        let scale = 1.0 / self.cell_size;

        // Update U velocities (horizontal)
        for j in 0..self.height {
            for i in 1..self.width {
                let idx_left = self.cell_index(i - 1, j);
                let idx_right = self.cell_index(i, j);

                // Skip if both cells are solid or air
                let left_type = self.cell_type[idx_left];
                let right_type = self.cell_type[idx_right];

                let u_idx = self.u_index(i, j);
                if left_type == CellType::Solid || right_type == CellType::Solid {
                    self.u[u_idx] = 0.0;
                } else if left_type == CellType::Fluid || right_type == CellType::Fluid {
                    let grad = (self.pressure[idx_right] - self.pressure[idx_left]) * scale;
                    self.u[u_idx] -= grad;
                }
            }
        }

        // Update V velocities (vertical)
        for j in 1..self.height {
            for i in 0..self.width {
                let idx_bottom = self.cell_index(i, j - 1);
                let idx_top = self.cell_index(i, j);

                let bottom_type = self.cell_type[idx_bottom];
                let top_type = self.cell_type[idx_top];

                let v_idx = self.v_index(i, j);
                if bottom_type == CellType::Solid || top_type == CellType::Solid {
                    self.v[v_idx] = 0.0;
                } else if bottom_type == CellType::Fluid || top_type == CellType::Fluid {
                    let grad = (self.pressure[idx_top] - self.pressure[idx_bottom]) * scale;
                    self.v[v_idx] -= grad;
                }
            }
        }
    }

    // ========================================================================
    // MULTIGRID PRESSURE SOLVER
    // ========================================================================

    /// Sync level 0 of multigrid with current grid state
    fn mg_sync_level_zero(&mut self) {
        let level = &mut self.mg_levels[0];

        // CRITICAL: The standard solver uses p = (neighbors - h^2*div) / 4
        // But the multigrid uses p = (neighbors - div) / 4 for simplicity.
        // To make them consistent, we pre-multiply divergence by h^2 here.
        let h_sq = self.cell_size * self.cell_size;
        for i in 0..self.divergence.len() {
            level.divergence[i] = self.divergence[i] * h_sq;
        }

        level.pressure.copy_from_slice(&self.pressure);
        for (i, &ct) in self.cell_type.iter().enumerate() {
            level.cell_type[i] = ct;
        }
    }

    /// Copy pressure back from level 0 to main grid
    fn mg_copy_pressure_back(&mut self) {
        self.pressure.copy_from_slice(&self.mg_levels[0].pressure);
    }

    /// Restrict residual from fine level to coarse level divergence
    /// Uses full-weighting stencil for smooth restriction
    fn mg_restrict(&mut self, fine_level: usize, coarse_level: usize) {
        let (fine_w, fine_h) = (self.mg_levels[fine_level].width, self.mg_levels[fine_level].height);
        let (coarse_w, coarse_h) = (self.mg_levels[coarse_level].width, self.mg_levels[coarse_level].height);

        // Clear coarse divergence and copy cell types
        for j in 0..coarse_h {
            for i in 0..coarse_w {
                let c_idx = j * coarse_w + i;

                // Map to fine grid (2:1 ratio)
                let fi = i * 2;
                let fj = j * 2;

                // Full-weighting: average 2x2 block of fine residuals
                let mut sum = 0.0;
                let mut count = 0;
                let mut any_fluid = false;

                for dj in 0..2 {
                    for di in 0..2 {
                        let fii = fi + di;
                        let fjj = fj + dj;
                        if fii < fine_w && fjj < fine_h {
                            let f_idx = fjj * fine_w + fii;
                            if self.mg_levels[fine_level].cell_type[f_idx] == CellType::Fluid {
                                sum += self.mg_levels[fine_level].residual[f_idx];
                                count += 1;
                                any_fluid = true;
                            }
                        }
                    }
                }

                // Coarse cell is fluid if any fine cell is fluid
                self.mg_levels[coarse_level].cell_type[c_idx] = if any_fluid {
                    CellType::Fluid
                } else {
                    // Check if any solid
                    let mut any_solid = false;
                    for dj in 0..2 {
                        for di in 0..2 {
                            let fii = fi + di;
                            let fjj = fj + dj;
                            if fii < fine_w && fjj < fine_h {
                                let f_idx = fjj * fine_w + fii;
                                if self.mg_levels[fine_level].cell_type[f_idx] == CellType::Solid {
                                    any_solid = true;
                                }
                            }
                        }
                    }
                    if any_solid { CellType::Solid } else { CellType::Air }
                };

                self.mg_levels[coarse_level].divergence[c_idx] = if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                };
            }
        }
    }

    /// Prolongate correction from coarse level to fine level
    /// Uses bilinear interpolation
    fn mg_prolongate(&mut self, coarse_level: usize, fine_level: usize) {
        let (fine_w, fine_h) = (self.mg_levels[fine_level].width, self.mg_levels[fine_level].height);
        let (coarse_w, coarse_h) = (self.mg_levels[coarse_level].width, self.mg_levels[coarse_level].height);

        for fj in 0..fine_h {
            for fi in 0..fine_w {
                let f_idx = fj * fine_w + fi;

                // Skip non-fluid cells
                if self.mg_levels[fine_level].cell_type[f_idx] != CellType::Fluid {
                    continue;
                }

                // Map to coarse grid coordinates (with fractional part)
                let cx = fi as f32 / 2.0;
                let cy = fj as f32 / 2.0;

                // Bilinear interpolation indices
                let ci0 = (cx.floor() as usize).min(coarse_w.saturating_sub(1));
                let cj0 = (cy.floor() as usize).min(coarse_h.saturating_sub(1));
                let ci1 = (ci0 + 1).min(coarse_w - 1);
                let cj1 = (cj0 + 1).min(coarse_h - 1);

                let tx = cx - ci0 as f32;
                let ty = cy - cj0 as f32;

                // Get coarse pressures
                let p00 = self.mg_levels[coarse_level].pressure[cj0 * coarse_w + ci0];
                let p10 = self.mg_levels[coarse_level].pressure[cj0 * coarse_w + ci1];
                let p01 = self.mg_levels[coarse_level].pressure[cj1 * coarse_w + ci0];
                let p11 = self.mg_levels[coarse_level].pressure[cj1 * coarse_w + ci1];

                // Bilinear interpolation
                let correction = (1.0 - tx) * (1.0 - ty) * p00
                               + tx * (1.0 - ty) * p10
                               + (1.0 - tx) * ty * p01
                               + tx * ty * p11;

                // ADD correction to fine pressure
                self.mg_levels[fine_level].pressure[f_idx] += correction;
            }
        }
    }

    /// Compute residual: r = divergence - Laplacian(pressure)
    /// Uses same stencil as original solver: Neumann mirroring for solid/boundary
    fn mg_compute_residual(&mut self, level: usize) {
        let w = self.mg_levels[level].width;
        let h = self.mg_levels[level].height;

        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;

                if self.mg_levels[level].cell_type[idx] != CellType::Fluid {
                    self.mg_levels[level].residual[idx] = 0.0;
                    continue;
                }

                let p_center = self.mg_levels[level].pressure[idx];

                // Get neighbor pressures with Neumann mirroring for solid/boundary
                // (same as original solve_pressure)
                let p_left = if i > 0 {
                    let n_idx = j * w + (i - 1);
                    if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                        p_center // Neumann: mirror
                    } else {
                        self.mg_levels[level].pressure[n_idx]
                    }
                } else {
                    p_center // Boundary: Neumann mirror
                };

                let p_right = if i + 1 < w {
                    let n_idx = j * w + (i + 1);
                    if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                        p_center
                    } else {
                        self.mg_levels[level].pressure[n_idx]
                    }
                } else {
                    p_center
                };

                let p_bottom = if j > 0 {
                    let n_idx = (j - 1) * w + i;
                    if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                        p_center
                    } else {
                        self.mg_levels[level].pressure[n_idx]
                    }
                } else {
                    p_center
                };

                let p_top = if j + 1 < h {
                    let n_idx = (j + 1) * w + i;
                    if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                        p_center
                    } else {
                        self.mg_levels[level].pressure[n_idx]
                    }
                } else {
                    p_center
                };

                // Residual = b - A*x = div - (neighbors - 4*p)
                // Note: divergence was pre-multiplied by h^2 in mg_sync_level_zero
                let neighbors_minus_4p = p_left + p_right + p_bottom + p_top - 4.0 * p_center;
                let div = self.mg_levels[level].divergence[idx];

                self.mg_levels[level].residual[idx] = div - neighbors_minus_4p;
            }
        }
    }

    /// Gauss-Seidel smoothing on a multigrid level
    /// Uses same stencil as original solver: always 4 neighbors with Neumann mirroring
    fn mg_smooth(&mut self, level: usize, iterations: usize) {
        let w = self.mg_levels[level].width;
        let h = self.mg_levels[level].height;

        // Divergence was pre-multiplied by h^2 in mg_sync_level_zero,
        // so we use the simplified form: p = (neighbors - div) / 4
        for _ in 0..iterations {
            // Red-black ordering
            for color in 0..2 {
                for j in 0..h {
                    for i in 0..w {
                        if (i + j) % 2 != color {
                            continue;
                        }

                        let idx = j * w + i;
                        if self.mg_levels[level].cell_type[idx] != CellType::Fluid {
                            self.mg_levels[level].pressure[idx] = 0.0;
                            continue;
                        }

                        let p_center = self.mg_levels[level].pressure[idx];

                        // Get neighbor pressures with Neumann mirroring for solid/boundary
                        let p_left = if i > 0 {
                            let n_idx = j * w + (i - 1);
                            if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                                p_center
                            } else {
                                self.mg_levels[level].pressure[n_idx]
                            }
                        } else {
                            p_center
                        };

                        let p_right = if i + 1 < w {
                            let n_idx = j * w + (i + 1);
                            if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                                p_center
                            } else {
                                self.mg_levels[level].pressure[n_idx]
                            }
                        } else {
                            p_center
                        };

                        let p_bottom = if j > 0 {
                            let n_idx = (j - 1) * w + i;
                            if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                                p_center
                            } else {
                                self.mg_levels[level].pressure[n_idx]
                            }
                        } else {
                            p_center
                        };

                        let p_top = if j + 1 < h {
                            let n_idx = (j + 1) * w + i;
                            if self.mg_levels[level].cell_type[n_idx] == CellType::Solid {
                                p_center
                            } else {
                                self.mg_levels[level].pressure[n_idx]
                            }
                        } else {
                            p_center
                        };

                        // GS update: p = (p_L + p_R + p_B + p_T - div) / 4
                        // Note: div was pre-multiplied by h^2 in mg_sync_level_zero
                        let div = self.mg_levels[level].divergence[idx];
                        self.mg_levels[level].pressure[idx] = (p_left + p_right + p_bottom + p_top - div) * 0.25;
                    }
                }
            }
        }
    }

    /// Clear pressure on a level (before coarse solve)
    fn mg_clear_pressure(&mut self, level: usize) {
        for p in &mut self.mg_levels[level].pressure {
            *p = 0.0;
        }
    }

    /// V-cycle multigrid iteration
    fn mg_v_cycle(&mut self, level: usize) {
        let max_level = self.mg_levels.len() - 1;
        let pre_smooth = 3;
        let post_smooth = 3;
        let coarse_solve = 20;

        // Pre-smoothing
        self.mg_smooth(level, pre_smooth);

        if level == max_level {
            // Coarsest level: solve more thoroughly
            self.mg_smooth(level, coarse_solve);
        } else {
            // Compute residual
            self.mg_compute_residual(level);

            // Restrict residual to coarser grid
            self.mg_restrict(level, level + 1);

            // Clear coarse pressure
            self.mg_clear_pressure(level + 1);

            // Recurse
            self.mg_v_cycle(level + 1);

            // Prolongate correction
            self.mg_prolongate(level + 1, level);

            // Post-smoothing
            self.mg_smooth(level, post_smooth);
        }
    }

    /// Solve pressure using multigrid V-cycles
    pub fn solve_pressure_multigrid(&mut self, num_cycles: usize) {
        if self.mg_levels.is_empty() {
            // Fallback to regular solver
            self.solve_pressure(15);
            return;
        }

        // Sync grid state to level 0
        self.mg_sync_level_zero();

        // Run V-cycles
        for _ in 0..num_cycles {
            self.mg_v_cycle(0);
        }

        // Copy result back
        self.mg_copy_pressure_back();
    }
}
*/
