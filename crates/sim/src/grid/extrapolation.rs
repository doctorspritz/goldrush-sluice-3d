//! Velocity extrapolation into air cells.
//!
//! Ensures valid velocities near the fluid-air interface.

#![allow(dead_code)]  // Methods will be used after Phase 3 migration

use super::{CellType, Grid};

// ============================================================================
// PHASE 3 MIGRATION: Uncomment this impl block after deleting originals from mod.rs
// Methods to delete from mod.rs:
//   - extrapolate_velocities
//   - mark_fluid_faces_known
//   - extrapolate_u_layer
//   - extrapolate_v_layer
//   - mark_fluid_faces_known_preallocated
//   - extrapolate_u_layer_preallocated
//   - extrapolate_v_layer_preallocated
// ============================================================================

// COMMENTED OUT - Enable in Phase 3 when originals are deleted from mod.rs
/*
impl Grid {
    // ============================================================================
    // VELOCITY EXTRAPOLATION (for FLIP momentum conservation)
    // ============================================================================

    /// Extrapolate velocities from fluid cells into non-fluid cells
    /// Uses layered wavefront: each layer copies from known neighbors
    ///
    /// This is critical for FLIP: particles near air boundaries need valid
    /// velocities to sample for the FLIP delta calculation. Without extrapolation,
    /// they sample zeros/undefined values causing phantom momentum loss.
    ///
    /// Optimization: Uses pre-allocated buffers to avoid ~1MB allocation per call.
    pub fn extrapolate_velocities(&mut self, max_layers: usize) {
        // Reset pre-allocated buffers (much faster than allocating)
        self.u_known.fill(false);
        self.v_known.fill(false);

        // Initialize: fluid cell faces are known
        self.mark_fluid_faces_known_preallocated();

        // Propagate layer by layer
        for _ in 0..max_layers {
            self.extrapolate_u_layer_preallocated();
            self.extrapolate_v_layer_preallocated();
        }
    }

    /// Mark U and V faces that have valid P2G velocities as "known"
    /// These are faces adjacent to at least one fluid cell.
    /// Both interior (fluid-fluid) and boundary (fluid-air) faces have valid values.
    fn mark_fluid_faces_known(&self, u_known: &mut [bool], v_known: &mut [bool]) {
        // Mark U faces: known if AT LEAST ONE adjacent cell is fluid
        // Interior faces (fluid-fluid) have P2G values
        // Boundary faces (fluid-air) have P2G values that pressure solve adjusts
        for j in 0..self.height {
            for i in 0..=self.width {
                let u_idx = self.u_index(i, j);

                // Check cells on either side
                let left_fluid = i > 0 && self.cell_type[self.cell_index(i - 1, j)] == CellType::Fluid;
                let right_fluid = i < self.width && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;

                // Mark if at least one cell is fluid (both interior and boundary faces)
                if left_fluid || right_fluid {
                    u_known[u_idx] = true;
                }
            }
        }

        // Mark V faces: known if AT LEAST ONE adjacent cell is fluid
        for j in 0..=self.height {
            for i in 0..self.width {
                let v_idx = self.v_index(i, j);

                let bottom_fluid = j > 0 && self.cell_type[self.cell_index(i, j - 1)] == CellType::Fluid;
                let top_fluid = j < self.height && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;

                if bottom_fluid || top_fluid {
                    v_known[v_idx] = true;
                }
            }
        }
    }

    /// Extrapolate U velocities one layer outward
    fn extrapolate_u_layer(&mut self, u_known: &mut [bool]) {
        // Two-buffer pattern: compute new values without reading them in same pass
        let mut new_known = vec![false; u_known.len()];
        let mut new_values = vec![0.0f32; self.u.len()];

        for j in 0..self.height {
            for i in 0..=self.width {
                let u_idx = self.u_index(i, j);

                if u_known[u_idx] {
                    continue; // Already known, skip
                }

                // Don't extrapolate into faces touching solid OR fluid cells
                // Fluid-adjacent faces already have valid values from P2G/pressure solve
                let left_solid = i == 0 || self.cell_type[self.cell_index(i - 1, j)] == CellType::Solid;
                let right_solid = i == self.width || self.cell_type[self.cell_index(i, j)] == CellType::Solid;
                let left_fluid = i > 0 && self.cell_type[self.cell_index(i - 1, j)] == CellType::Fluid;
                let right_fluid = i < self.width && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;
                if left_solid || right_solid || left_fluid || right_fluid {
                    continue; // Already has valid value
                }

                // Average of known cardinal neighbors (staggered U grid)
                let mut sum = 0.0;
                let mut count = 0;

                // Left neighbor (i-1, j)
                if i > 0 {
                    let idx = self.u_index(i - 1, j);
                    if u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }
                // Right neighbor (i+1, j)
                if i < self.width {
                    let idx = self.u_index(i + 1, j);
                    if u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }
                // Down neighbor (i, j-1)
                if j > 0 {
                    let idx = self.u_index(i, j - 1);
                    if u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }
                // Up neighbor (i, j+1)
                if j + 1 < self.height {
                    let idx = self.u_index(i, j + 1);
                    if u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    new_values[u_idx] = sum / count as f32;
                    new_known[u_idx] = true;
                }
                // If count == 0, leave unchanged (next layer may reach it)
            }
        }

        // Apply new values and merge known flags
        for i in 0..u_known.len() {
            if new_known[i] {
                self.u[i] = new_values[i];
                u_known[i] = true;
            }
        }
    }

    /// Extrapolate V velocities one layer outward
    fn extrapolate_v_layer(&mut self, v_known: &mut [bool]) {
        let mut new_known = vec![false; v_known.len()];
        let mut new_values = vec![0.0f32; self.v.len()];

        for j in 0..=self.height {
            for i in 0..self.width {
                let v_idx = self.v_index(i, j);

                if v_known[v_idx] {
                    continue;
                }

                // Don't extrapolate into faces touching solid OR fluid cells
                // Fluid-adjacent faces already have valid values from P2G/pressure solve
                let bottom_solid = j == 0 || self.cell_type[self.cell_index(i, j - 1)] == CellType::Solid;
                let top_solid = j == self.height || self.cell_type[self.cell_index(i, j)] == CellType::Solid;
                let bottom_fluid = j > 0 && self.cell_type[self.cell_index(i, j - 1)] == CellType::Fluid;
                let top_fluid = j < self.height && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;
                if bottom_solid || top_solid || bottom_fluid || top_fluid {
                    continue; // Already has valid value
                }

                let mut sum = 0.0;
                let mut count = 0;

                // Left neighbor (i-1, j)
                if i > 0 {
                    let idx = self.v_index(i - 1, j);
                    if v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }
                // Right neighbor (i+1, j)
                if i + 1 < self.width {
                    let idx = self.v_index(i + 1, j);
                    if v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }
                // Down neighbor (i, j-1)
                if j > 0 {
                    let idx = self.v_index(i, j - 1);
                    if v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }
                // Up neighbor (i, j+1)
                if j < self.height {
                    let idx = self.v_index(i, j + 1);
                    if v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    new_values[v_idx] = sum / count as f32;
                    new_known[v_idx] = true;
                }
            }
        }

        for i in 0..v_known.len() {
            if new_known[i] {
                self.v[i] = new_values[i];
                v_known[i] = true;
            }
        }
    }

    /// Mark fluid faces as known using pre-allocated buffer
    fn mark_fluid_faces_known_preallocated(&mut self) {
        // Mark U faces: known if AT LEAST ONE adjacent cell is fluid
        for j in 0..self.height {
            for i in 0..=self.width {
                let u_idx = self.u_index(i, j);
                let left_fluid = i > 0 && self.cell_type[self.cell_index(i - 1, j)] == CellType::Fluid;
                let right_fluid = i < self.width && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;
                if left_fluid || right_fluid {
                    self.u_known[u_idx] = true;
                }
            }
        }

        // Mark V faces
        for j in 0..=self.height {
            for i in 0..self.width {
                let v_idx = self.v_index(i, j);
                let bottom_fluid = j > 0 && self.cell_type[self.cell_index(i, j - 1)] == CellType::Fluid;
                let top_fluid = j < self.height && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;
                if bottom_fluid || top_fluid {
                    self.v_known[v_idx] = true;
                }
            }
        }
    }

    /// Extrapolate U velocities one layer using pre-allocated buffers
    fn extrapolate_u_layer_preallocated(&mut self) {
        // Clear temp buffers
        self.u_new_known.fill(false);

        for j in 0..self.height {
            for i in 0..=self.width {
                let u_idx = self.u_index(i, j);

                if self.u_known[u_idx] {
                    continue;
                }

                // Skip faces touching solid or fluid cells
                let left_solid = i == 0 || self.cell_type[self.cell_index(i - 1, j)] == CellType::Solid;
                let right_solid = i == self.width || self.cell_type[self.cell_index(i, j)] == CellType::Solid;
                let left_fluid = i > 0 && self.cell_type[self.cell_index(i - 1, j)] == CellType::Fluid;
                let right_fluid = i < self.width && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;
                if left_solid || right_solid || left_fluid || right_fluid {
                    continue;
                }

                let mut sum = 0.0;
                let mut count = 0;

                if i > 0 {
                    let idx = self.u_index(i - 1, j);
                    if self.u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }
                if i < self.width {
                    let idx = self.u_index(i + 1, j);
                    if self.u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }
                if j > 0 {
                    let idx = self.u_index(i, j - 1);
                    if self.u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }
                if j + 1 < self.height {
                    let idx = self.u_index(i, j + 1);
                    if self.u_known[idx] {
                        sum += self.u[idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    self.u_new_values[u_idx] = sum / count as f32;
                    self.u_new_known[u_idx] = true;
                }
            }
        }

        // Apply new values
        for i in 0..self.u_known.len() {
            if self.u_new_known[i] {
                self.u[i] = self.u_new_values[i];
                self.u_known[i] = true;
            }
        }
    }

    /// Extrapolate V velocities one layer using pre-allocated buffers
    fn extrapolate_v_layer_preallocated(&mut self) {
        self.v_new_known.fill(false);

        for j in 0..=self.height {
            for i in 0..self.width {
                let v_idx = self.v_index(i, j);

                if self.v_known[v_idx] {
                    continue;
                }

                let bottom_solid = j == 0 || self.cell_type[self.cell_index(i, j - 1)] == CellType::Solid;
                let top_solid = j == self.height || self.cell_type[self.cell_index(i, j)] == CellType::Solid;
                let bottom_fluid = j > 0 && self.cell_type[self.cell_index(i, j - 1)] == CellType::Fluid;
                let top_fluid = j < self.height && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;
                if bottom_solid || top_solid || bottom_fluid || top_fluid {
                    continue;
                }

                let mut sum = 0.0;
                let mut count = 0;

                if i > 0 {
                    let idx = self.v_index(i - 1, j);
                    if self.v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }
                if i + 1 < self.width {
                    let idx = self.v_index(i + 1, j);
                    if self.v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }
                if j > 0 {
                    let idx = self.v_index(i, j - 1);
                    if self.v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }
                if j < self.height {
                    let idx = self.v_index(i, j + 1);
                    if self.v_known[idx] {
                        sum += self.v[idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    self.v_new_values[v_idx] = sum / count as f32;
                    self.v_new_known[v_idx] = true;
                }
            }
        }

        for i in 0..self.v_known.len() {
            if self.v_new_known[i] {
                self.v[i] = self.v_new_values[i];
                self.v_known[i] = true;
            }
        }
    }
}
*/
