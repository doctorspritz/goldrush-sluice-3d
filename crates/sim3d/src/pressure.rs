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

                if grid.cell_type[idx] != CellType::Fluid {
                    grid.divergence[idx] = 0.0;
                    continue;
                }

                // U faces: u[i+1,j,k] - u[i,j,k]
                let u_right = grid.u[grid.u_index(i + 1, j, k)];
                let u_left = grid.u[grid.u_index(i, j, k)];

                // V faces: v[i,j+1,k] - v[i,j,k]
                let v_top = grid.v[grid.v_index(i, j + 1, k)];
                let v_bottom = grid.v[grid.v_index(i, j, k)];

                // W faces: w[i,j,k+1] - w[i,j,k]
                let w_front = grid.w[grid.w_index(i, j, k + 1)];
                let w_back = grid.w[grid.w_index(i, j, k)];

                grid.divergence[idx] =
                    scale * ((u_right - u_left) + (v_top - v_bottom) + (w_front - w_back));
            }
        }
    }
}

/// Solve pressure Poisson equation using Jacobi iteration.
///
/// Laplacian(p) = div(v)
///
/// 6-neighbor stencil:
/// p[i,j,k] = (sum of 6 neighbors - dx^2 * div) / count
pub fn solve_pressure_jacobi(grid: &mut Grid3D, iterations: usize) {
    let scale = grid.cell_size * grid.cell_size;

    // Need to double-buffer for true Jacobi (not Gauss-Seidel)
    let mut pressure_new = grid.pressure.clone();

    for _ in 0..iterations {
        for k in 0..grid.depth {
            for j in 0..grid.height {
                for i in 0..grid.width {
                    let idx = grid.cell_index(i, j, k);

                    if grid.cell_type[idx] != CellType::Fluid {
                        continue;
                    }

                    let mut sum = 0.0;
                    let mut count = 0;

                    // -X neighbor
                    if i > 0 {
                        let nidx = grid.cell_index(i - 1, j, k);
                        if grid.cell_type[nidx] != CellType::Solid {
                            sum += grid.pressure[nidx];
                            count += 1;
                        }
                    }

                    // +X neighbor
                    if i + 1 < grid.width {
                        let nidx = grid.cell_index(i + 1, j, k);
                        if grid.cell_type[nidx] != CellType::Solid {
                            sum += grid.pressure[nidx];
                            count += 1;
                        }
                    }

                    // -Y neighbor
                    if j > 0 {
                        let nidx = grid.cell_index(i, j - 1, k);
                        if grid.cell_type[nidx] != CellType::Solid {
                            sum += grid.pressure[nidx];
                            count += 1;
                        }
                    }

                    // +Y neighbor
                    if j + 1 < grid.height {
                        let nidx = grid.cell_index(i, j + 1, k);
                        if grid.cell_type[nidx] != CellType::Solid {
                            sum += grid.pressure[nidx];
                            count += 1;
                        }
                    }

                    // -Z neighbor
                    if k > 0 {
                        let nidx = grid.cell_index(i, j, k - 1);
                        if grid.cell_type[nidx] != CellType::Solid {
                            sum += grid.pressure[nidx];
                            count += 1;
                        }
                    }

                    // +Z neighbor
                    if k + 1 < grid.depth {
                        let nidx = grid.cell_index(i, j, k + 1);
                        if grid.cell_type[nidx] != CellType::Solid {
                            sum += grid.pressure[nidx];
                            count += 1;
                        }
                    }

                    if count > 0 {
                        pressure_new[idx] = (sum - scale * grid.divergence[idx]) / count as f32;
                    }
                }
            }
        }

        // Swap buffers
        std::mem::swap(&mut grid.pressure, &mut pressure_new);
    }
}

/// Apply pressure gradient to make velocity field divergence-free.
/// v_new = v_old - dt * grad(p) / rho
///
/// For unit density, this simplifies to: v -= grad(p) * dt
pub fn apply_pressure_gradient(grid: &mut Grid3D, dt: f32) {
    let scale = dt / grid.cell_size;

    // U velocities (between cells in X direction)
    for k in 0..grid.depth {
        for j in 0..grid.height {
            for i in 1..grid.width {
                let idx_left = grid.cell_index(i - 1, j, k);
                let idx_right = grid.cell_index(i, j, k);

                let left_type = grid.cell_type[idx_left];
                let right_type = grid.cell_type[idx_right];

                // Skip if both are solid
                if left_type == CellType::Solid && right_type == CellType::Solid {
                    continue;
                }

                // Skip if face is on solid boundary
                if left_type == CellType::Solid || right_type == CellType::Solid {
                    // Set velocity to zero on solid boundary
                    let u_idx = grid.u_index(i, j, k);
                    grid.u[u_idx] = 0.0;
                    continue;
                }

                // Apply gradient if at least one side is fluid
                if left_type == CellType::Fluid || right_type == CellType::Fluid {
                    let grad = (grid.pressure[idx_right] - grid.pressure[idx_left]) * scale;
                    let u_idx = grid.u_index(i, j, k);
                    grid.u[u_idx] -= grad;
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

                let bottom_type = grid.cell_type[idx_bottom];
                let top_type = grid.cell_type[idx_top];

                if bottom_type == CellType::Solid && top_type == CellType::Solid {
                    continue;
                }

                if bottom_type == CellType::Solid || top_type == CellType::Solid {
                    let v_idx = grid.v_index(i, j, k);
                    grid.v[v_idx] = 0.0;
                    continue;
                }

                if bottom_type == CellType::Fluid || top_type == CellType::Fluid {
                    let grad = (grid.pressure[idx_top] - grid.pressure[idx_bottom]) * scale;
                    let v_idx = grid.v_index(i, j, k);
                    grid.v[v_idx] -= grad;
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

                let back_type = grid.cell_type[idx_back];
                let front_type = grid.cell_type[idx_front];

                if back_type == CellType::Solid && front_type == CellType::Solid {
                    continue;
                }

                if back_type == CellType::Solid || front_type == CellType::Solid {
                    let w_idx = grid.w_index(i, j, k);
                    grid.w[w_idx] = 0.0;
                    continue;
                }

                if back_type == CellType::Fluid || front_type == CellType::Fluid {
                    let grad = (grid.pressure[idx_front] - grid.pressure[idx_back]) * scale;
                    let w_idx = grid.w_index(i, j, k);
                    grid.w[w_idx] -= grad;
                }
            }
        }
    }
}

/// Enforce boundary conditions: zero velocity at domain edges.
pub fn enforce_boundary_conditions(grid: &mut Grid3D) {
    let width = grid.width;
    let height = grid.height;
    let depth = grid.depth;

    // U boundaries (i=0 and i=width)
    for k in 0..depth {
        for j in 0..height {
            let idx0 = grid.u_index(0, j, k);
            let idx1 = grid.u_index(width, j, k);
            grid.u[idx0] = 0.0;
            grid.u[idx1] = 0.0;
        }
    }

    // V boundaries (j=0 and j=height)
    for k in 0..depth {
        for i in 0..width {
            let idx0 = grid.v_index(i, 0, k);
            let idx1 = grid.v_index(i, height, k);
            grid.v[idx0] = 0.0;
            grid.v[idx1] = 0.0;
        }
    }

    // W boundaries (k=0 and k=depth)
    for j in 0..height {
        for i in 0..width {
            let idx0 = grid.w_index(i, j, 0);
            let idx1 = grid.w_index(i, j, depth);
            grid.w[idx0] = 0.0;
            grid.w[idx1] = 0.0;
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
                    grid.cell_type[idx] = CellType::Fluid;
                }
            }
        }

        compute_divergence(&mut grid);

        for &div in &grid.divergence {
            assert!((div).abs() < 1e-6);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let mut grid = Grid3D::new(4, 4, 4, 1.0);
        grid.u.fill(1.0);
        grid.v.fill(1.0);
        grid.w.fill(1.0);

        enforce_boundary_conditions(&mut grid);

        // Check U boundaries
        for k in 0..4 {
            for j in 0..4 {
                assert_eq!(grid.u[grid.u_index(0, j, k)], 0.0);
                assert_eq!(grid.u[grid.u_index(4, j, k)], 0.0);
            }
        }

        // Check V boundaries
        for k in 0..4 {
            for i in 0..4 {
                assert_eq!(grid.v[grid.v_index(i, 0, k)], 0.0);
                assert_eq!(grid.v[grid.v_index(i, 4, k)], 0.0);
            }
        }

        // Check W boundaries
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(grid.w[grid.w_index(i, j, 0)], 0.0);
                assert_eq!(grid.w[grid.w_index(i, j, 4)], 0.0);
            }
        }
    }
}
