//! Velocity field operations.
//!
//! Sampling, gravity, viscosity, and boundary conditions.

use super::{quadratic_bspline_1d, CellType, Grid};
use glam::Vec2;

impl Grid {
    /// Sample velocity at a world position using bilinear interpolation
    pub fn sample_velocity(&self, pos: Vec2) -> Vec2 {
        let u = self.sample_u(pos);
        let v = self.sample_v(pos);
        Vec2::new(u, v)
    }

    /// Sample velocity using quadratic B-spline (same kernel as APIC/FLIP)
    /// This MUST be used for FLIP delta calculations to avoid kernel mismatch.
    pub fn sample_velocity_bspline(&self, pos: Vec2) -> Vec2 {
        let mut velocity = Vec2::ZERO;
        let mut u_weight_sum = 0.0f32;
        let mut v_weight_sum = 0.0f32;

        // ========== Sample U component (quadratic B-spline) ==========
        let u_pos = pos / self.cell_size - Vec2::new(0.0, 0.5);
        let base_i = u_pos.x.floor() as i32;
        let base_j = u_pos.y.floor() as i32;
        let fx = u_pos.x - base_i as f32;
        let fy = u_pos.y - base_j as f32;

        let u_wx = [
            quadratic_bspline_1d(fx + 1.0),
            quadratic_bspline_1d(fx),
            quadratic_bspline_1d(fx - 1.0),
        ];
        let u_wy = [
            quadratic_bspline_1d(fy + 1.0),
            quadratic_bspline_1d(fy),
            quadratic_bspline_1d(fy - 1.0),
        ];

        for dj in -1..=1i32 {
            let nj = base_j + dj;
            if nj < 0 || nj >= self.height as i32 {
                continue;
            }
            let wy = u_wy[(dj + 1) as usize];

            for di in -1..=1i32 {
                let ni = base_i + di;
                if ni < 0 || ni > self.width as i32 {
                    continue;
                }

                let w = u_wx[(di + 1) as usize] * wy;
                if w <= 0.0 {
                    continue;
                }

                let u_idx = self.u_index(ni as usize, nj as usize);
                velocity.x += w * self.u[u_idx];
                u_weight_sum += w;
            }
        }

        // ========== Sample V component (quadratic B-spline) ==========
        let v_pos = pos / self.cell_size - Vec2::new(0.5, 0.0);
        let base_i = v_pos.x.floor() as i32;
        let base_j = v_pos.y.floor() as i32;
        let fx = v_pos.x - base_i as f32;
        let fy = v_pos.y - base_j as f32;

        let v_wx = [
            quadratic_bspline_1d(fx + 1.0),
            quadratic_bspline_1d(fx),
            quadratic_bspline_1d(fx - 1.0),
        ];
        let v_wy = [
            quadratic_bspline_1d(fy + 1.0),
            quadratic_bspline_1d(fy),
            quadratic_bspline_1d(fy - 1.0),
        ];

        for dj in -1..=1i32 {
            let nj = base_j + dj;
            if nj < 0 || nj > self.height as i32 {
                continue;
            }
            let wy = v_wy[(dj + 1) as usize];

            for di in -1..=1i32 {
                let ni = base_i + di;
                if ni < 0 || ni >= self.width as i32 {
                    continue;
                }

                let w = v_wx[(di + 1) as usize] * wy;
                if w <= 0.0 {
                    continue;
                }

                let v_idx = self.v_index(ni as usize, nj as usize);
                velocity.y += w * self.v[v_idx];
                v_weight_sum += w;
            }
        }

        // Normalize
        if u_weight_sum > 0.0 {
            velocity.x /= u_weight_sum;
        }
        if v_weight_sum > 0.0 {
            velocity.y /= v_weight_sum;
        }

        velocity
    }

    /// Apply gravity - simple uniform downward
    /// Boundary conditions will zero velocity at solid faces
    /// Pressure solve will handle incompressibility
    pub fn apply_gravity(&mut self, dt: f32) {
        use crate::physics::GRAVITY;
        for v in &mut self.v {
            *v += GRAVITY * dt;
        }
    }

    /// Apply viscosity diffusion to velocity field
    /// This creates boundary layers near walls, enabling vortex shedding.
    ///
    /// Uses explicit Euler: dU/dt = nu * nabla^2 u
    /// Stability requires: nu * dt / h^2 < 0.25 (automatically clamped)
    ///
    /// Only affects velocities near walls where there are velocity gradients.
    /// Bulk flow with uniform velocity is unaffected (Laplacian approx 0).
    pub fn apply_viscosity(&mut self, dt: f32, viscosity: f32) {
        let h_sq = self.cell_size * self.cell_size;

        // Stability check: clamp effective viscosity to prevent blowup
        // For explicit Euler diffusion: nu * dt / h^2 < 0.25
        let max_viscosity = 0.2 * h_sq / dt; // Use 0.2 for safety margin
        let nu = viscosity.min(max_viscosity);
        let scale = nu * dt / h_sq;

        // Skip if viscosity is negligible
        if scale < 1e-6 {
            return;
        }

        // ========== Diffuse U velocities ==========
        // U is stored at left edges: (i, j+0.5) for i in 0..=width, j in 0..height
        // Copy to temp buffer, then write back (avoids allocation)
        self.u_temp.copy_from_slice(&self.u);

        for j in 0..self.height {
            for i in 1..self.width {
                let idx = self.u_index(i, j);

                // Check if adjacent cells are fluid (not solid/air)
                let cell_left = self.cell_index(i - 1, j);
                let cell_right = self.cell_index(i, j);

                // Only apply viscosity in fluid regions
                if self.cell_type[cell_left] != CellType::Fluid
                   && self.cell_type[cell_right] != CellType::Fluid {
                    continue;
                }

                let u = self.u_temp[idx];

                // Get neighbors with boundary handling
                let u_left = self.u_temp[self.u_index(i - 1, j)];
                let u_right = self.u_temp[self.u_index(i + 1, j)];
                let u_bottom = if j > 0 { self.u_temp[self.u_index(i, j - 1)] } else { u };
                let u_top = if j < self.height - 1 { self.u_temp[self.u_index(i, j + 1)] } else { u };

                // Laplacian with Neumann BC at domain edges
                let laplacian = u_left + u_right + u_bottom + u_top - 4.0 * u;
                self.u[idx] = u + scale * laplacian;
            }
        }

        // ========== Diffuse V velocities ==========
        // V is stored at bottom edges: (i+0.5, j) for i in 0..width, j in 0..=height
        self.v_temp.copy_from_slice(&self.v);

        for j in 1..self.height {
            for i in 0..self.width {
                let idx = self.v_index(i, j);

                // Check if adjacent cells are fluid
                let cell_bottom = self.cell_index(i, j - 1);
                let cell_top = self.cell_index(i, j);

                if self.cell_type[cell_bottom] != CellType::Fluid
                   && self.cell_type[cell_top] != CellType::Fluid {
                    continue;
                }

                let v = self.v_temp[idx];

                // Get neighbors
                let v_left = if i > 0 { self.v_temp[self.v_index(i - 1, j)] } else { v };
                let v_right = if i < self.width - 1 { self.v_temp[self.v_index(i + 1, j)] } else { v };
                let v_bottom = self.v_temp[self.v_index(i, j - 1)];
                let v_top = self.v_temp[self.v_index(i, j + 1)];

                let laplacian = v_left + v_right + v_bottom + v_top - 4.0 * v;
                self.v[idx] = v + scale * laplacian;
            }
        }
    }

    /// Zero out velocities at solid boundaries
    /// MUST be called before compute_divergence to prevent velocity into walls
    /// from corrupting the pressure solve
    pub fn enforce_boundary_conditions(&mut self) {
        // Zero u velocities at solid walls
        for j in 0..self.height {
            for i in 0..=self.width {
                let u_idx = self.u_index(i, j);
                // Check cells on either side of this u face
                let left_solid = i == 0 || self.is_solid(i - 1, j);
                let right_solid = i == self.width || self.is_solid(i, j);
                if left_solid || right_solid {
                    self.u[u_idx] = 0.0;
                }
            }
        }

        // Zero v velocities at solid walls
        for j in 0..=self.height {
            for i in 0..self.width {
                let v_idx = self.v_index(i, j);
                // Check cells above/below this v face
                let below_solid = j == 0 || self.is_solid(i, j - 1);
                let above_solid = j == self.height || self.is_solid(i, j);
                if below_solid || above_solid {
                    self.v[v_idx] = 0.0;
                }
            }
        }
    }
}
