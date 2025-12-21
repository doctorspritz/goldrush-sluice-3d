//! MAC (Marker-and-Cell) grid for APIC fluid simulation
//!
//! Uses staggered grid layout:
//! - u (horizontal velocity) stored on left edges of cells
//! - v (vertical velocity) stored on bottom edges of cells
//! - pressure stored at cell centers
//!
//! APIC requires quadratic B-spline kernels for momentum-conserving transfers.

use glam::Vec2;

// ============================================================================
// QUADRATIC B-SPLINE KERNEL FUNCTIONS (Required for APIC)
// ============================================================================

/// 1D Quadratic B-spline weight function
/// Support: [-1.5, 1.5] (3 cells wide)
///
/// W(r) = 3/4 - r² for |r| < 0.5
/// W(r) = 1/2 * (3/2 - |r|)² for 0.5 ≤ |r| < 1.5
/// W(r) = 0 for |r| ≥ 1.5
#[inline]
pub fn quadratic_bspline_1d(r: f32) -> f32 {
    let r_abs = r.abs();
    if r_abs < 0.5 {
        0.75 - r_abs * r_abs
    } else if r_abs < 1.5 {
        let t = 1.5 - r_abs;
        0.5 * t * t
    } else {
        0.0
    }
}

/// 2D Quadratic B-spline weight (tensor product of 1D kernels)
#[inline]
pub fn quadratic_bspline(delta: Vec2) -> f32 {
    quadratic_bspline_1d(delta.x) * quadratic_bspline_1d(delta.y)
}

/// APIC D matrix inverse (inertia-like tensor)
/// For quadratic B-splines on uniform grid: D = (1/4) * Δx² * I
/// So D_inv = (4 / Δx²) * I
/// This is constant for uniform grids and can be pre-computed.
#[inline]
pub fn apic_d_inverse(cell_size: f32) -> f32 {
    4.0 / (cell_size * cell_size)
}

/// Cell type for boundary conditions
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellType {
    /// Solid obstacle - blocks flow
    Solid,
    /// Contains fluid particles
    Fluid,
    /// Empty air
    Air,
}

/// Staggered MAC grid for pressure-velocity simulation
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cell_size: f32,

    /// Horizontal velocity (staggered on left edges)
    /// Size: (width+1) * height
    pub u: Vec<f32>,
    /// Vertical velocity (staggered on bottom edges)
    /// Size: width * (height+1)
    pub v: Vec<f32>,

    /// Pressure at cell centers
    pub pressure: Vec<f32>,
    /// Divergence at cell centers (computed during solve)
    pub divergence: Vec<f32>,

    /// Cell type for boundary handling
    pub cell_type: Vec<CellType>,

    /// Solid terrain (persistent, set from level geometry)
    pub solid: Vec<bool>,

    /// Signed distance field to nearest solid (negative inside, positive outside)
    /// Precomputed when terrain changes for O(1) collision queries
    pub sdf: Vec<f32>,

    /// Vorticity (curl) at cell centers: ω = ∂v/∂x - ∂u/∂y
    /// Used for vorticity confinement and diagnostics
    pub vorticity: Vec<f32>,

    /// Temporary buffer for viscosity diffusion (avoids allocation)
    u_temp: Vec<f32>,
    v_temp: Vec<f32>,
}

impl Grid {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let cell_count = width * height;
        let u_count = (width + 1) * height;
        let v_count = width * (height + 1);

        Self {
            width,
            height,
            cell_size,
            u: vec![0.0; u_count],
            v: vec![0.0; v_count],
            pressure: vec![0.0; cell_count],
            divergence: vec![0.0; cell_count],
            cell_type: vec![CellType::Air; cell_count],
            solid: vec![false; cell_count],
            sdf: vec![f32::MAX; cell_count],  // Will be computed after terrain setup
            vorticity: vec![0.0; cell_count],
            // Pre-allocated buffers for viscosity (avoids per-frame allocation)
            u_temp: vec![0.0; u_count],
            v_temp: vec![0.0; v_count],
        }
    }

    /// Compute signed distance field from solid cells using fast sweeping
    /// Call this after terrain changes (set_solid calls)
    ///
    /// Returns: negative inside solid, zero at surface, positive outside
    pub fn compute_sdf(&mut self) {
        let w = self.width;
        let h = self.height;
        let cell_size = self.cell_size;

        // First pass: compute unsigned distance to nearest solid
        // Initialize: solid cells = 0, others = large positive
        for idx in 0..self.sdf.len() {
            self.sdf[idx] = if self.solid[idx] { 0.0 } else { f32::MAX };
        }

        // Fast sweeping: 4 diagonal passes to propagate distances
        let offsets: [(i32, i32, f32); 4] = [
            (1, 0, cell_size),   // right
            (-1, 0, cell_size),  // left
            (0, 1, cell_size),   // down
            (0, -1, cell_size),  // up
        ];

        // Multiple sweep passes for convergence
        for _sweep in 0..4 {
            // Forward sweep (top-left to bottom-right)
            for j in 0..h {
                for i in 0..w {
                    let idx = j * w + i;
                    let mut min_dist = self.sdf[idx];
                    for &(di, dj, d) in &offsets {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        if ni >= 0 && ni < w as i32 && nj >= 0 && nj < h as i32 {
                            let nidx = nj as usize * w + ni as usize;
                            min_dist = min_dist.min(self.sdf[nidx].abs() + d);
                        }
                    }
                    self.sdf[idx] = min_dist;
                }
            }

            // Backward sweep (bottom-right to top-left)
            for j in (0..h).rev() {
                for i in (0..w).rev() {
                    let idx = j * w + i;
                    let mut min_dist = self.sdf[idx];
                    for &(di, dj, d) in &offsets {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        if ni >= 0 && ni < w as i32 && nj >= 0 && nj < h as i32 {
                            let nidx = nj as usize * w + ni as usize;
                            min_dist = min_dist.min(self.sdf[nidx].abs() + d);
                        }
                    }
                    self.sdf[idx] = min_dist;
                }
            }
        }

        // Second pass: make distances NEGATIVE inside solid cells
        // This creates a proper signed distance field
        for idx in 0..self.sdf.len() {
            if self.solid[idx] {
                // Inside solid: negate the distance
                // Distance should be negative, with magnitude = distance to nearest non-solid
                self.sdf[idx] = -self.sdf[idx].max(cell_size * 0.5);
            }
        }
    }

    /// Sample SDF at world position (bilinear interpolation)
    #[inline]
    pub fn sample_sdf(&self, pos: Vec2) -> f32 {
        let x = pos.x / self.cell_size - 0.5;
        let y = pos.y / self.cell_size - 0.5;

        let i0 = (x.floor() as i32).clamp(0, self.width as i32 - 2) as usize;
        let j0 = (y.floor() as i32).clamp(0, self.height as i32 - 2) as usize;
        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let tx = (x - i0 as f32).clamp(0.0, 1.0);
        let ty = (y - j0 as f32).clamp(0.0, 1.0);

        let d00 = self.sdf[j0 * self.width + i0];
        let d10 = self.sdf[j0 * self.width + i1];
        let d01 = self.sdf[j1 * self.width + i0];
        let d11 = self.sdf[j1 * self.width + i1];

        let d0 = d00 * (1.0 - tx) + d10 * tx;
        let d1 = d01 * (1.0 - tx) + d11 * tx;

        d0 * (1.0 - ty) + d1 * ty
    }

    /// Get SDF gradient (points away from solid) at position
    #[inline]
    pub fn sdf_gradient(&self, pos: Vec2) -> Vec2 {
        let eps = self.cell_size * 0.5;
        let dx = self.sample_sdf(pos + Vec2::new(eps, 0.0)) - self.sample_sdf(pos - Vec2::new(eps, 0.0));
        let dy = self.sample_sdf(pos + Vec2::new(0.0, eps)) - self.sample_sdf(pos - Vec2::new(0.0, eps));
        Vec2::new(dx, dy).normalize_or_zero()
    }

    /// Convert world position to grid cell indices
    pub fn pos_to_cell(&self, pos: Vec2) -> (usize, usize) {
        let i = (pos.x / self.cell_size).floor() as i32;
        let j = (pos.y / self.cell_size).floor() as i32;
        (
            i.clamp(0, self.width as i32 - 1) as usize,
            j.clamp(0, self.height as i32 - 1) as usize,
        )
    }

    /// Cell center index (for pressure, divergence, cell_type)
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.width + i
    }

    /// U velocity index (staggered on left edges)
    #[inline]
    pub fn u_index(&self, i: usize, j: usize) -> usize {
        j * (self.width + 1) + i
    }

    /// V velocity index (staggered on bottom edges)
    #[inline]
    pub fn v_index(&self, i: usize, j: usize) -> usize {
        j * self.width + i
    }

    /// Set cell as solid (terrain)
    pub fn set_solid(&mut self, i: usize, j: usize) {
        if i < self.width && j < self.height {
            let idx = self.cell_index(i, j);
            self.solid[idx] = true;
        }
    }

    /// Check if cell is solid terrain
    pub fn is_solid(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return true; // Out of bounds is solid
        }
        self.solid[self.cell_index(i, j)]
    }

    /// Check if cell contains fluid (for drift-flux coupling)
    pub fn is_fluid(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return false;
        }
        self.cell_type[self.cell_index(i, j)] == CellType::Fluid
    }

    /// Sample velocity at a world position using bilinear interpolation
    pub fn sample_velocity(&self, pos: Vec2) -> Vec2 {
        let u = self.sample_u(pos);
        let v = self.sample_v(pos);
        Vec2::new(u, v)
    }

    /// Sample U component (staggered - sample at left edges)
    fn sample_u(&self, pos: Vec2) -> f32 {
        // U is stored at (i, j+0.5) in cell-space
        let x = pos.x / self.cell_size;
        let y = pos.y / self.cell_size - 0.5;

        let i = x.floor() as i32;
        let j = y.floor() as i32;
        let fx = x - i as f32;
        let fy = y - j as f32;

        // Clamp indices
        let i0 = i.clamp(0, self.width as i32) as usize;
        let i1 = (i + 1).clamp(0, self.width as i32) as usize;
        let j0 = j.clamp(0, self.height as i32 - 1) as usize;
        let j1 = (j + 1).clamp(0, self.height as i32 - 1) as usize;

        // Bilinear interpolation
        let u00 = self.u[self.u_index(i0, j0)];
        let u10 = self.u[self.u_index(i1, j0)];
        let u01 = self.u[self.u_index(i0, j1)];
        let u11 = self.u[self.u_index(i1, j1)];

        let u0 = u00 * (1.0 - fx) + u10 * fx;
        let u1 = u01 * (1.0 - fx) + u11 * fx;
        u0 * (1.0 - fy) + u1 * fy
    }

    /// Sample V component (staggered - sample at bottom edges)
    fn sample_v(&self, pos: Vec2) -> f32 {
        // V is stored at (i+0.5, j) in cell-space
        let x = pos.x / self.cell_size - 0.5;
        let y = pos.y / self.cell_size;

        let i = x.floor() as i32;
        let j = y.floor() as i32;
        let fx = x - i as f32;
        let fy = y - j as f32;

        // Clamp indices
        let i0 = i.clamp(0, self.width as i32 - 1) as usize;
        let i1 = (i + 1).clamp(0, self.width as i32 - 1) as usize;
        let j0 = j.clamp(0, self.height as i32) as usize;
        let j1 = (j + 1).clamp(0, self.height as i32) as usize;

        // Bilinear interpolation
        let v00 = self.v[self.v_index(i0, j0)];
        let v10 = self.v[self.v_index(i1, j0)];
        let v01 = self.v[self.v_index(i0, j1)];
        let v11 = self.v[self.v_index(i1, j1)];

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;
        v0 * (1.0 - fy) + v1 * fy
    }

    /// Get bilinear interpolation weights for a position
    /// Returns (i, j, weights) where weights is [(di, dj, weight); 4]
    pub fn get_interp_weights(&self, pos: Vec2) -> (usize, usize, [(i32, i32, f32); 4]) {
        let x = pos.x / self.cell_size;
        let y = pos.y / self.cell_size;

        let i = x.floor() as i32;
        let j = y.floor() as i32;
        let fx = x - i as f32;
        let fy = y - j as f32;

        let i = i.clamp(0, self.width as i32 - 1) as usize;
        let j = j.clamp(0, self.height as i32 - 1) as usize;

        let weights = [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ];

        (i, j, weights)
    }

    /// Clear velocity field
    pub fn clear_velocities(&mut self) {
        self.u.fill(0.0);
        self.v.fill(0.0);
    }

    /// Clear pressure
    pub fn clear_pressure(&mut self) {
        self.pressure.fill(0.0);
        self.divergence.fill(0.0);
    }

    /// Apply gravity - simple uniform downward
    /// Boundary conditions will zero velocity at solid faces
    /// Pressure solve will handle incompressibility
    pub fn apply_gravity(&mut self, dt: f32) {
        // Strong gravity for rushing sluice water
        const GRAVITY: f32 = 400.0;
        for v in &mut self.v {
            *v += GRAVITY * dt;
        }
    }

    /// Apply viscosity diffusion to velocity field
    /// This creates boundary layers near walls, enabling vortex shedding.
    ///
    /// Uses explicit Euler: ∂u/∂t = ν∇²u
    /// Stability requires: ν * dt / h² < 0.25 (automatically clamped)
    ///
    /// Only affects velocities near walls where there are velocity gradients.
    /// Bulk flow with uniform velocity is unaffected (Laplacian ≈ 0).
    pub fn apply_viscosity(&mut self, dt: f32, viscosity: f32) {
        let h_sq = self.cell_size * self.cell_size;

        // Stability check: clamp effective viscosity to prevent blowup
        // For explicit Euler diffusion: ν * dt / h² < 0.25
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
                        self.update_pressure_cell(i, j);
                    }
                }
            }
            // Black pass (i+j odd) - uses updated red values
            for j in 1..self.height - 1 {
                for i in 1..self.width - 1 {
                    if (i + j) % 2 != 0 {
                        self.update_pressure_cell(i, j);
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

    /// Compute maximum residual of pressure equation: |∇²p - div|
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

                // Laplacian: (p_L + p_R + p_B + p_T - 4*p) / h²
                let laplacian = (p_left + p_right + p_bottom + p_top - 4.0 * p) / h_sq;
                let residual = (laplacian - self.divergence[idx]).abs();
                max_residual = max_residual.max(residual);
            }
        }

        max_residual
    }

    /// Update a single pressure cell (helper for Red-Black GS)
    #[inline]
    fn update_pressure_cell(&mut self, i: usize, j: usize) {
        let idx = self.cell_index(i, j);

        if self.cell_type[idx] != CellType::Fluid {
            self.pressure[idx] = 0.0;
            return;
        }

        // Neighbor pressures (solid boundaries treated as current pressure)
        let p_left = if self.cell_type[self.cell_index(i - 1, j)] == CellType::Solid {
            self.pressure[idx]
        } else {
            self.pressure[self.cell_index(i - 1, j)]
        };
        let p_right = if self.cell_type[self.cell_index(i + 1, j)] == CellType::Solid {
            self.pressure[idx]
        } else {
            self.pressure[self.cell_index(i + 1, j)]
        };
        let p_bottom = if self.cell_type[self.cell_index(i, j - 1)] == CellType::Solid {
            self.pressure[idx]
        } else {
            self.pressure[self.cell_index(i, j - 1)]
        };
        let p_top = if self.cell_type[self.cell_index(i, j + 1)] == CellType::Solid {
            self.pressure[idx]
        } else {
            self.pressure[self.cell_index(i, j + 1)]
        };

        let div = self.divergence[idx];

        // Gauss-Seidel update for ∇²p = div
        // Discretized: (p_L + p_R + p_B + p_T - 4*p) / h² = div
        // Solving for p: p = (p_L + p_R + p_B + p_T - h²*div) / 4
        let h_sq = self.cell_size * self.cell_size;
        self.pressure[idx] = (p_left + p_right + p_bottom + p_top - h_sq * div) * 0.25;
    }

    /// Subtract pressure gradient from velocity field
    /// Uses formulation: u -= ∇p̃ / h where p̃ is pseudo-pressure from ∇²p̃ = div
    pub fn apply_pressure_gradient(&mut self, _dt: f32) {
        // CRITICAL FIX: Remove dt from scale. With dt, only 0.1% of divergence
        // was corrected per frame. Without dt, we get full correction.
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

    /// Compute vorticity (curl) field: ω = ∂v/∂x - ∂u/∂y
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

    /// Compute enstrophy: ε = ½∫|ω|² dV
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

    /// Compute total absolute vorticity: ∫|ω| dV
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
    pub fn apply_vorticity_confinement(&mut self, dt: f32, strength: f32) {
        // First pass: compute curl (vorticity) at each cell and store it
        self.compute_vorticity();

        // Use stored vorticity for confinement
        let curl = &self.vorticity;

        // Second pass: apply vorticity force
        for j in 2..self.height - 2 {
            for i in 2..self.width - 2 {
                let idx = self.cell_index(i, j);

                if self.cell_type[idx] != CellType::Fluid {
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
                // F = ε × h × (N × ω) - scaled by grid spacing for grid independence
                // Reference: Fedkiw et al. 2001 "Visual Simulation of Smoke"
                let c = curl[idx];
                let h = self.cell_size;
                let fx = ny * c * strength * h;
                let fy = -nx * c * strength * h;

                // Apply to velocity
                let u_idx = self.u_index(i, j);
                let v_idx = self.v_index(i, j);
                self.u[u_idx] += fx * dt;
                self.v[v_idx] += fy * dt;
            }
        }
    }
}
