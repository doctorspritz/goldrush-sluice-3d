//! Pressure solver for 3D incompressible flow.
//!
//! Uses Jacobi iteration with a 6-neighbor stencil.

use crate::grid::{CellType, Grid3D};

/// Compute divergence of velocity field.
/// div(v) = du/dx + dv/dy + dw/dz
pub fn compute_divergence(grid: &mut Grid3D) {
    let scale = 1.0 / grid.cell_size;

    for k in 0..grid.depth {
        for j in 0..grid.height {
            for i in 0..grid.width {
                let idx = grid.cell_index(i, j, k);

                if grid.cell_type()[idx] != CellType::Fluid {
                    grid.divergence_mut()[idx] = 0.0;
                    continue;
                }

                // U faces: u[i+1,j,k] - u[i,j,k]
                let u_right = grid.u()[grid.u_index(i + 1, j, k)];
                let u_left = grid.u()[grid.u_index(i, j, k)];

                // V faces: v[i,j+1,k] - v[i,j,k]
                let v_top = grid.v()[grid.v_index(i, j + 1, k)];
                let v_bottom = grid.v()[grid.v_index(i, j, k)];

                // W faces: w[i,j,k+1] - w[i,j,k]
                let w_front = grid.w()[grid.w_index(i, j, k + 1)];
                let w_back = grid.w()[grid.w_index(i, j, k)];

                grid.divergence_mut()[idx] =
                    scale * ((u_right - u_left) + (v_top - v_bottom) + (w_front - w_back));
            }
        }
    }
}

/// Solve pressure Poisson equation using Red-Black Gauss-Seidel.
///
/// 2x faster convergence than Jacobi - updates in-place using latest neighbor values.
/// Uses Neumann boundary conditions at solid walls (dp/dn = 0).
pub fn solve_pressure_jacobi(grid: &mut Grid3D, iterations: usize) {
    let h_sq = grid.cell_size * grid.cell_size;

    for _ in 0..iterations {
        // Red pass: cells where (i+j+k) is even
        for k in 0..grid.depth {
            for j in 0..grid.height {
                for i in 0..grid.width {
                    if (i + j + k) % 2 == 0 {
                        update_pressure_cell(grid, i, j, k, h_sq);
                    }
                }
            }
        }

        // Black pass: cells where (i+j+k) is odd
        for k in 0..grid.depth {
            for j in 0..grid.height {
                for i in 0..grid.width {
                    if (i + j + k) % 2 != 0 {
                        update_pressure_cell(grid, i, j, k, h_sq);
                    }
                }
            }
        }
    }
}

/// Update a single pressure cell using Gauss-Seidel with Neumann BCs.
#[inline]
fn update_pressure_cell(grid: &mut Grid3D, i: usize, j: usize, k: usize, h_sq: f32) {
    let idx = grid.cell_index(i, j, k);

    if grid.cell_type()[idx] != CellType::Fluid {
        grid.pressure_mut()[idx] = 0.0;
        return;
    }

    let p = grid.pressure()[idx];

    // Neighbor pressures with Neumann BC at solids (dp/dn = 0 means use current pressure)
    let p_xm = if i > 0 {
        let nidx = grid.cell_index(i - 1, j, k);
        if grid.cell_type()[nidx] == CellType::Solid {
            p
        } else {
            grid.pressure()[nidx]
        }
    } else {
        p
    };

    let p_xp = if i + 1 < grid.width {
        let nidx = grid.cell_index(i + 1, j, k);
        if grid.cell_type()[nidx] == CellType::Solid {
            p
        } else {
            grid.pressure()[nidx]
        }
    } else {
        p
    };

    let p_ym = if j > 0 {
        let nidx = grid.cell_index(i, j - 1, k);
        if grid.cell_type()[nidx] == CellType::Solid {
            p
        } else {
            grid.pressure()[nidx]
        }
    } else {
        p
    };

    let p_yp = if j + 1 < grid.height {
        let nidx = grid.cell_index(i, j + 1, k);
        if grid.cell_type()[nidx] == CellType::Solid {
            p
        } else {
            grid.pressure()[nidx]
        }
    } else {
        p
    };

    let p_zm = if k > 0 {
        let nidx = grid.cell_index(i, j, k - 1);
        if grid.cell_type()[nidx] == CellType::Solid {
            p
        } else {
            grid.pressure()[nidx]
        }
    } else {
        p
    };

    let p_zp = if k + 1 < grid.depth {
        let nidx = grid.cell_index(i, j, k + 1);
        if grid.cell_type()[nidx] == CellType::Solid {
            p
        } else {
            grid.pressure()[nidx]
        }
    } else {
        p
    };

    // 6-neighbor Laplacian: (sum of neighbors - 6*p) / h^2 = div
    // Solve for p: p = (sum of neighbors - h^2 * div) / 6
    let div = grid.divergence()[idx];
    grid.pressure_mut()[idx] = (p_xm + p_xp + p_ym + p_yp + p_zm + p_zp - h_sq * div) / 6.0;
}

/// Apply pressure gradient to make velocity field divergence-free.
///
/// Uses pseudo-pressure formulation: v_new = v - ∇p / dx
/// The pseudo-pressure from ∇²p = div gives the exact velocity correction needed.
/// NO dt factor - that was the bug causing water to compress instead of fill!
pub fn apply_pressure_gradient(grid: &mut Grid3D) {
    let scale = 1.0 / grid.cell_size;

    // U velocities (between cells in X direction)
    for k in 0..grid.depth {
        for j in 0..grid.height {
            for i in 1..grid.width {
                let idx_left = grid.cell_index(i - 1, j, k);
                let idx_right = grid.cell_index(i, j, k);

                let left_type = grid.cell_type()[idx_left];
                let right_type = grid.cell_type()[idx_right];

                // Skip if both are solid
                if left_type == CellType::Solid && right_type == CellType::Solid {
                    continue;
                }

                // Skip if face is on solid boundary
                if left_type == CellType::Solid || right_type == CellType::Solid {
                    // Set velocity to zero on solid boundary
                    let u_idx = grid.u_index(i, j, k);
                    grid.u_mut()[u_idx] = 0.0;
                    continue;
                }

                // Apply gradient if at least one side is fluid
                if left_type == CellType::Fluid || right_type == CellType::Fluid {
                    let grad = (grid.pressure()[idx_right] - grid.pressure()[idx_left]) * scale;
                    let u_idx = grid.u_index(i, j, k);
                    grid.u_mut()[u_idx] -= grad;
                }
            }
        }
    }

    // V velocities (between cells in Y direction)
    for k in 0..grid.depth {
        for j in 1..grid.height {
            for i in 0..grid.width {
                let idx_bottom = grid.cell_index(i, j - 1, k);
                let idx_top = grid.cell_index(i, j, k);

                let bottom_type = grid.cell_type()[idx_bottom];
                let top_type = grid.cell_type()[idx_top];

                if bottom_type == CellType::Solid && top_type == CellType::Solid {
                    continue;
                }

                if bottom_type == CellType::Solid || top_type == CellType::Solid {
                    let v_idx = grid.v_index(i, j, k);
                    grid.v_mut()[v_idx] = 0.0;
                    continue;
                }

                if bottom_type == CellType::Fluid || top_type == CellType::Fluid {
                    let grad = (grid.pressure()[idx_top] - grid.pressure()[idx_bottom]) * scale;
                    let v_idx = grid.v_index(i, j, k);
                    grid.v_mut()[v_idx] -= grad;
                }
            }
        }
    }

    // W velocities (between cells in Z direction)
    for k in 1..grid.depth {
        for j in 0..grid.height {
            for i in 0..grid.width {
                let idx_back = grid.cell_index(i, j, k - 1);
                let idx_front = grid.cell_index(i, j, k);

                let back_type = grid.cell_type()[idx_back];
                let front_type = grid.cell_type()[idx_front];

                if back_type == CellType::Solid && front_type == CellType::Solid {
                    continue;
                }

                if back_type == CellType::Solid || front_type == CellType::Solid {
                    let w_idx = grid.w_index(i, j, k);
                    grid.w_mut()[w_idx] = 0.0;
                    continue;
                }

                if back_type == CellType::Fluid || front_type == CellType::Fluid {
                    let grad = (grid.pressure()[idx_front] - grid.pressure()[idx_back]) * scale;
                    let w_idx = grid.w_index(i, j, k);
                    grid.w_mut()[w_idx] -= grad;
                }
            }
        }
    }
}

/// Enforce boundary conditions: zero velocity at domain edges AND solid faces.
pub fn enforce_boundary_conditions(grid: &mut Grid3D) {
    let width = grid.width;
    let height = grid.height;
    let depth = grid.depth;

    // Domain edge U boundaries (i=0 and i=width)
    for k in 0..depth {
        for j in 0..height {
            let idx0 = grid.u_index(0, j, k);
            let idx1 = grid.u_index(width, j, k);
            grid.u_mut()[idx0] = 0.0;
            grid.u_mut()[idx1] = 0.0;
        }
    }

    // Domain edge V boundaries (j=0 and j=height)
    for k in 0..depth {
        for i in 0..width {
            let idx0 = grid.v_index(i, 0, k);
            let idx1 = grid.v_index(i, height, k);
            grid.v_mut()[idx0] = 0.0;
            grid.v_mut()[idx1] = 0.0;
        }
    }

    // Domain edge W boundaries (k=0 and k=depth)
    for j in 0..height {
        for i in 0..width {
            let idx0 = grid.w_index(i, j, 0);
            let idx1 = grid.w_index(i, j, depth);
            grid.w_mut()[idx0] = 0.0;
            grid.w_mut()[idx1] = 0.0;
        }
    }

    // Zero velocities at ALL solid cell faces (internal walls)
    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                let idx = grid.cell_index(i, j, k);
                if grid.cell_type()[idx] == CellType::Solid {
                    // Pre-compute indices to avoid borrow issues
                    let u_left = grid.u_index(i, j, k);
                    let u_right = grid.u_index(i + 1, j, k);
                    let v_bottom = grid.v_index(i, j, k);
                    let v_top = grid.v_index(i, j + 1, k);
                    let w_back = grid.w_index(i, j, k);
                    let w_front = grid.w_index(i, j, k + 1);

                    // Zero all 6 face velocities of this solid cell
                    grid.u_mut()[u_left] = 0.0;
                    grid.u_mut()[u_right] = 0.0;
                    grid.v_mut()[v_bottom] = 0.0;
                    grid.v_mut()[v_top] = 0.0;
                    grid.w_mut()[w_back] = 0.0;
                    grid.w_mut()[w_front] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_zero_velocity() {
        let mut grid = Grid3D::new(4, 4, 4, 1.0);
        // All velocities are zero, so divergence should be zero
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let idx = grid.cell_index(i, j, k);
                    grid.cell_type_mut()[idx] = CellType::Fluid;
                }
            }
        }

        compute_divergence(&mut grid);

        for &div in grid.divergence() {
            assert!((div).abs() < 1e-6);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let mut grid = Grid3D::new(4, 4, 4, 1.0);
        grid.u_mut().fill(1.0);
        grid.v_mut().fill(1.0);
        grid.w_mut().fill(1.0);

        enforce_boundary_conditions(&mut grid);

        // Check U boundaries
        for k in 0..4 {
            for j in 0..4 {
                assert_eq!(grid.u()[grid.u_index(0, j, k)], 0.0);
                assert_eq!(grid.u()[grid.u_index(4, j, k)], 0.0);
            }
        }

        // Check V boundaries
        for k in 0..4 {
            for i in 0..4 {
                assert_eq!(grid.v()[grid.v_index(i, 0, k)], 0.0);
                assert_eq!(grid.v()[grid.v_index(i, 4, k)], 0.0);
            }
        }

        // Check W boundaries
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(grid.w()[grid.w_index(i, j, 0)], 0.0);
                assert_eq!(grid.w()[grid.w_index(i, j, 4)], 0.0);
            }
        }
    }
}
