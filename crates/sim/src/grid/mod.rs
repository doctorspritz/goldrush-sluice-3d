//! MAC (Marker-and-Cell) grid for APIC fluid simulation
//!
//! Uses staggered grid layout:
//! - u (horizontal velocity) stored on left edges of cells
//! - v (vertical velocity) stored on bottom edges of cells
//! - pressure stored at cell centers
//!
//! APIC requires quadratic B-spline kernels for momentum-conserving transfers.

// Submodules (methods will be moved here incrementally)
mod cell_types;
mod extrapolation;
mod interp;
mod pressure;
mod sdf;
mod velocity;
mod vorticity;

// Re-exports for backwards compatibility
// NOTE: Add these re-exports after deleting originals from mod.rs (Phase 3):
// pub use interp::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d};

use glam::Vec2;

use crate::particle::ParticleMaterial;

// ============================================================================
// DEPOSITED CELL (Single-material per cell for correct entrainment)
// ============================================================================

/// Single-material deposited sediment cell.
/// Each cell contains ONE material type, enabling correct selective entrainment:
/// - Sand cells entrain at Shields 0.045 (easy)
/// - Magnetite cells entrain at Shields 0.07 (harder)
/// - Gold cells entrain at Shields 0.09 (hardest)
/// Natural stratification happens because heavier materials settle faster.
#[derive(Clone, Copy, Debug, Default)]
pub struct DepositedCell {
    /// The material type in this cell (None = not deposited)
    pub material: Option<ParticleMaterial>,
}

impl DepositedCell {
    /// Check if this cell has any deposited material
    #[inline]
    pub fn is_deposited(&self) -> bool {
        self.material.is_some()
    }

    /// Shields parameter for entrainment (exact value for this material)
    /// Higher values = harder to entrain
    pub fn effective_shields_critical(&self) -> f32 {
        match self.material {
            Some(mat) => mat.shields_critical(),
            None => 0.045, // Default to sand
        }
    }

    /// Density for physics/visualization (exact value for this material)
    pub fn effective_density(&self) -> f32 {
        match self.material {
            Some(mat) => mat.density(),
            None => 2.65, // Default to sand
        }
    }

    /// Color for rendering (exact color for this material)
    pub fn color(&self) -> [u8; 4] {
        match self.material {
            Some(mat) => mat.color(),
            None => [194, 178, 128, 255], // Default sand color
        }
    }

    /// Clear deposited material
    pub fn clear(&mut self) {
        self.material = None;
    }

    /// Get the material type (for entrainment spawning)
    pub fn get_material(&self) -> Option<ParticleMaterial> {
        self.material
    }
}

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

/// A single level in the multigrid hierarchy
#[derive(Clone)]
pub struct MultigridLevel {
    pub width: usize,
    pub height: usize,
    pub pressure: Vec<f32>,
    pub pressure_temp: Vec<f32>,  // For Jacobi iteration (read old, write new)
    pub divergence: Vec<f32>,  // Right-hand side (restricted from finer level)
    pub residual: Vec<f32>,
    pub cell_type: Vec<CellType>,
}

impl MultigridLevel {
    pub fn new(width: usize, height: usize) -> Self {
        let cell_count = width * height;
        Self {
            width,
            height,
            pressure: vec![0.0; cell_count],
            pressure_temp: vec![0.0; cell_count],
            divergence: vec![0.0; cell_count],
            residual: vec![0.0; cell_count],
            cell_type: vec![CellType::Air; cell_count],
        }
    }

    #[inline]
    pub fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.width + i
    }
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

    /// Deposited sediment composition per cell
    /// Tracks material fractions for mixed-material beds
    /// Separate from `solid` to enable distinct visualization
    pub deposited: Vec<DepositedCell>,

    /// Signed distance field to nearest solid (negative inside, positive outside)
    /// Precomputed when terrain changes for O(1) collision queries
    pub sdf: Vec<f32>,

    /// Vorticity (curl) at cell centers: ω = ∂v/∂x - ∂u/∂y
    /// Used for vorticity confinement and diagnostics
    pub vorticity: Vec<f32>,

    /// Bed heightfield (y-coordinate) for each x-column
    /// Used for optimized floor collision and rendering
    pub bed_height: Vec<f32>,

    /// Temporary buffer for viscosity diffusion (avoids allocation)
    u_temp: Vec<f32>,
    v_temp: Vec<f32>,

    /// Pre-allocated buffer for inner SDF (avoids 512KB allocation per frame)
    inner_sdf: Vec<f32>,

    /// Pre-allocated buffers for velocity extrapolation (avoids ~1MB allocation per frame)
    u_known: Vec<bool>,
    v_known: Vec<bool>,
    u_new_known: Vec<bool>,
    v_new_known: Vec<bool>,
    u_new_values: Vec<f32>,
    v_new_values: Vec<f32>,

    /// Multigrid hierarchy for fast pressure solve
    /// Level 0 is finest (same size as main grid), higher levels are coarser
    mg_levels: Vec<MultigridLevel>,
}

impl Grid {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let cell_count = width * height;
        let u_count = (width + 1) * height;
        let v_count = width * (height + 1);

        // Build multigrid hierarchy: halve dimensions until either < 16
        let mut mg_levels = Vec::new();
        let mut w = width;
        let mut h = height;
        while w >= 16 && h >= 16 {
            mg_levels.push(MultigridLevel::new(w, h));
            w /= 2;
            h /= 2;
        }
        // Add coarsest level even if small
        if mg_levels.is_empty() || (w >= 4 && h >= 4) {
            mg_levels.push(MultigridLevel::new(w, h));
        }

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
            deposited: vec![DepositedCell::default(); cell_count],
            sdf: vec![f32::MAX; cell_count],  // Will be computed after terrain setup
            vorticity: vec![0.0; cell_count],
            bed_height: vec![0.0; width],
            // Pre-allocated buffers for viscosity (avoids per-frame allocation)
            u_temp: vec![0.0; u_count],
            v_temp: vec![0.0; v_count],
            // Pre-allocated buffer for inner SDF computation
            inner_sdf: vec![0.0; cell_count],
            // Pre-allocated buffers for velocity extrapolation
            u_known: vec![false; u_count],
            v_known: vec![false; v_count],
            u_new_known: vec![false; u_count],
            v_new_known: vec![false; v_count],
            u_new_values: vec![0.0; u_count],
            v_new_values: vec![0.0; v_count],
            mg_levels,
        }
    }

    /// Compute signed distance field from solid cells using fast sweeping
    /// Call this after terrain changes (set_solid calls)
    ///
    /// Returns: negative inside solid, zero at surface, positive outside
    ///
    /// Optimization: Uses pre-allocated inner_sdf buffer and single sweep pass.
    pub fn compute_sdf(&mut self) {
        let w = self.width;
        let h = self.height;
        let cell_size = self.cell_size;
        let len = self.sdf.len();

        // Use pre-allocated buffer for inner SDF
        let inner_sdf = &mut self.inner_sdf;

        // Initialize both distance fields
        for idx in 0..len {
            if self.cell_type[idx] == CellType::Solid {
                // Solid: Outer dist = 0, Inner dist = MAX
                self.sdf[idx] = 0.0;
                inner_sdf[idx] = f32::MAX;
            } else {
                // Air/Fluid: Outer dist = MAX, Inner dist = 0
                self.sdf[idx] = f32::MAX;
                inner_sdf[idx] = 0.0;
            }
        }

        // Fast sweeping: single forward+backward pass
        // Sufficient for sluice geometry (simple convex-ish shapes)

        // Forward sweep (Top-Left -> Bottom-Right)
        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;
                let mut min_outer = self.sdf[idx];
                let mut min_inner = inner_sdf[idx];

                if i > 0 {
                    let nidx = idx - 1;
                    min_outer = min_outer.min(self.sdf[nidx] + cell_size);
                    min_inner = min_inner.min(inner_sdf[nidx] + cell_size);
                }
                if j > 0 {
                    let nidx = idx - w;
                    min_outer = min_outer.min(self.sdf[nidx] + cell_size);
                    min_inner = min_inner.min(inner_sdf[nidx] + cell_size);
                }

                self.sdf[idx] = min_outer;
                inner_sdf[idx] = min_inner;
            }
        }

        // Backward sweep (Bottom-Right -> Top-Left)
        for j in (0..h).rev() {
            for i in (0..w).rev() {
                let idx = j * w + i;
                let mut min_outer = self.sdf[idx];
                let mut min_inner = inner_sdf[idx];

                if i < w - 1 {
                    let nidx = idx + 1;
                    min_outer = min_outer.min(self.sdf[nidx] + cell_size);
                    min_inner = min_inner.min(inner_sdf[nidx] + cell_size);
                }
                if j < h - 1 {
                    let nidx = idx + w;
                    min_outer = min_outer.min(self.sdf[nidx] + cell_size);
                    min_inner = min_inner.min(inner_sdf[nidx] + cell_size);
                }

                self.sdf[idx] = min_outer;
                inner_sdf[idx] = min_inner;
            }
        }

        // Combine: SDF = Outer - Inner
        for idx in 0..len {
            self.sdf[idx] = self.sdf[idx] - inner_sdf[idx];
        }
    }

    /// Compute bed heightfield from solid terrain
    /// For each column, find the topmost solid cell and store its top edge y-coordinate
    /// Call this after terrain setup (after compute_sdf)
    pub fn compute_bed_heights(&mut self) {
        for i in 0..self.width {
            // Scan from top to bottom to find first solid cell
            let mut bed_y = 0.0f32;
            for j in (0..self.height).rev() {
                if self.is_solid(i, j) {
                    // Found solid - bed surface is top of this cell
                    bed_y = (j as f32 + 1.0) * self.cell_size;
                    break;
                }
            }
            self.bed_height[i] = bed_y;
        }
    }

    /// Sample bed height at a world x-position using linear interpolation
    #[inline]
    pub fn sample_bed_height(&self, x: f32) -> f32 {
        let fx = x / self.cell_size - 0.5;
        let i0 = (fx.floor() as i32).clamp(0, self.width as i32 - 2) as usize;
        let i1 = i0 + 1;
        let t = (fx - i0 as f32).clamp(0.0, 1.0);
        self.bed_height[i0] * (1.0 - t) + self.bed_height[i1] * t
    }

    /// Compute normalized height above bed (0 = at bed, 1 = at water surface)
    /// surface_height is the estimated water surface level
    #[inline]
    pub fn normalized_height_above_bed(&self, pos: Vec2, surface_height: f32) -> f32 {
        let bed = self.sample_bed_height(pos.x);
        let height_above_bed = (pos.y - bed).max(0.0);
        let water_depth = (surface_height - bed).max(0.01); // avoid div by zero
        (height_above_bed / water_depth).clamp(0.0, 1.0)
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
    /// Sets both the solid[] array (permanent terrain) and cell_type (current state)
    pub fn set_solid(&mut self, i: usize, j: usize) {
        if i < self.width && j < self.height {
            let idx = self.cell_index(i, j);
            self.solid[idx] = true;
            self.cell_type[idx] = CellType::Solid;
        }
    }

    /// Check if cell is solid terrain
    pub fn is_solid(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return true; // Out of bounds is solid
        }
        self.solid[self.cell_index(i, j)]
    }

    /// Mark cell as deposited sediment with a specific material (also marks as solid)
    pub fn set_deposited_with_material(&mut self, i: usize, j: usize, material: ParticleMaterial) {
        if i < self.width && j < self.height {
            let idx = self.cell_index(i, j);
            self.solid[idx] = true;
            self.deposited[idx] = DepositedCell {
                material: Some(material),
            };
        }
    }

    /// Mark cell as deposited sediment (legacy: assumes pure sand)
    pub fn set_deposited(&mut self, i: usize, j: usize) {
        self.set_deposited_with_material(i, j, ParticleMaterial::Sand);
    }

    /// Check if cell is deposited sediment
    pub fn is_deposited(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return false;
        }
        self.deposited[self.cell_index(i, j)].is_deposited()
    }

    /// Get deposited cell (returns reference for reading material type)
    pub fn get_deposited(&self, i: usize, j: usize) -> Option<&DepositedCell> {
        if i >= self.width || j >= self.height {
            return None;
        }
        let idx = self.cell_index(i, j);
        if self.deposited[idx].is_deposited() {
            Some(&self.deposited[idx])
        } else {
            None
        }
    }

    /// Clear deposited status and solid flag for a cell
    /// Used during entrainment when flow velocity exceeds threshold
    pub fn clear_deposited(&mut self, i: usize, j: usize) {
        if i < self.width && j < self.height {
            let idx = self.cell_index(i, j);
            self.solid[idx] = false;
            self.deposited[idx].clear();
        }
    }

    /// Check if cell contains fluid (for drift-flux coupling)
    pub fn is_fluid(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return false;
        }
        self.cell_type[self.cell_index(i, j)] == CellType::Fluid
    }

    // NOTE: sample_velocity and sample_velocity_bspline moved to velocity.rs

    /// Sample vorticity at position using bilinear interpolation
    /// Vorticity is stored at cell centers (same as pressure, cell_type)
    pub fn sample_vorticity(&self, pos: Vec2) -> f32 {
        // Cell center coordinates: center of cell (i,j) is at ((i+0.5)*h, (j+0.5)*h)
        // So to get cell-space coords for interpolation: x/h - 0.5
        let x = pos.x / self.cell_size - 0.5;
        let y = pos.y / self.cell_size - 0.5;

        let i0 = (x.floor() as i32).clamp(0, self.width as i32 - 2) as usize;
        let j0 = (y.floor() as i32).clamp(0, self.height as i32 - 2) as usize;
        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let tx = (x - i0 as f32).clamp(0.0, 1.0);
        let ty = (y - j0 as f32).clamp(0.0, 1.0);

        // Bilinear interpolation of vorticity
        let v00 = self.vorticity[j0 * self.width + i0];
        let v10 = self.vorticity[j0 * self.width + i1];
        let v01 = self.vorticity[j1 * self.width + i0];
        let v11 = self.vorticity[j1 * self.width + i1];

        let v0 = v00 * (1.0 - tx) + v10 * tx;
        let v1 = v01 * (1.0 - tx) + v11 * tx;

        v0 * (1.0 - ty) + v1 * ty
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

    // NOTE: apply_gravity, apply_viscosity, enforce_boundary_conditions moved to velocity.rs

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
            // Include right boundary (width-1) for open outflow
            for j in 1..self.height - 1 {
                for i in 1..self.width {
                    if (i + j) % 2 == 0 {
                        self.update_pressure_cell(i, j, h_sq);
                    }
                }
            }
            // Black pass (i+j odd) - uses updated red values
            for j in 1..self.height - 1 {
                for i in 1..self.width {
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
    /// Uses proper Neumann BC: denominator = number of non-solid neighbors
    /// Right boundary (i == width-1) uses Dirichlet BC (p=0) for open outflow
    #[inline]
    fn update_pressure_cell(&mut self, i: usize, j: usize, h_sq: f32) {
        let idx = self.cell_index(i, j);

        if self.cell_type[idx] != CellType::Fluid {
            self.pressure[idx] = 0.0;
            return;
        }

        // Count active (non-solid) neighbors and sum their pressures
        // Neumann BC (dp/dn = 0): exclude solid neighbors from both sum AND count
        // Right boundary uses Dirichlet BC (p=0) for open outflow
        let mut neighbor_sum = 0.0f32;
        let mut neighbor_count = 0.0f32;

        // Left neighbor
        if i > 0 {
            let left_idx = self.cell_index(i - 1, j);
            if self.cell_type[left_idx] != CellType::Solid {
                neighbor_sum += self.pressure[left_idx];
                neighbor_count += 1.0;
            }
        }

        // Right neighbor - open boundary at i == width-1
        // Treat as p=0 (Dirichlet BC for free outflow)
        if i + 1 < self.width {
            let right_idx = self.cell_index(i + 1, j);
            if self.cell_type[right_idx] != CellType::Solid {
                neighbor_sum += self.pressure[right_idx];
                neighbor_count += 1.0;
            }
        } else {
            // Open outflow: right boundary neighbor has p=0 (Dirichlet BC)
            // This allows flow to exit freely without pressure buildup
            neighbor_sum += 0.0; // p = 0 at outlet
            neighbor_count += 1.0;
        }

        // Bottom neighbor
        if j > 0 {
            let bottom_idx = self.cell_index(i, j - 1);
            if self.cell_type[bottom_idx] != CellType::Solid {
                neighbor_sum += self.pressure[bottom_idx];
                neighbor_count += 1.0;
            }
        }

        // Top neighbor
        if j + 1 < self.height {
            let top_idx = self.cell_index(i, j + 1);
            if self.cell_type[top_idx] != CellType::Solid {
                neighbor_sum += self.pressure[top_idx];
                neighbor_count += 1.0;
            }
        }

        let div = self.divergence[idx];

        // Gauss-Seidel update for ∇²p = div with proper Neumann BC
        // Only divide by the number of active neighbors
        if neighbor_count > 0.0 {
            self.pressure[idx] = (neighbor_sum - h_sq * div) / neighbor_count;
        } else {
            self.pressure[idx] = 0.0;
        }
    }

    /// Compute maximum residual of pressure equation: |∇²p - div|
    /// Includes right boundary with Dirichlet BC (p=0) for open outflow
    fn compute_max_residual(&self, h_sq: f32) -> f32 {
        let mut max_residual = 0.0f32;

        for j in 1..self.height - 1 {
            for i in 1..self.width {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] != CellType::Fluid {
                    continue;
                }

                let p = self.pressure[idx];
                let p_left = if i > 0 { self.pressure[self.cell_index(i - 1, j)] } else { 0.0 };
                let p_right = if i + 1 < self.width { self.pressure[self.cell_index(i + 1, j)] } else { 0.0 }; // Open boundary
                let p_bottom = if j > 0 { self.pressure[self.cell_index(i, j - 1)] } else { 0.0 };
                let p_top = if j + 1 < self.height { self.pressure[self.cell_index(i, j + 1)] } else { 0.0 };

                // Laplacian: (p_L + p_R + p_B + p_T - 4*p) / h²
                let laplacian = (p_left + p_right + p_bottom + p_top - 4.0 * p) / h_sq;
                let residual = (laplacian - self.divergence[idx]).abs();
                max_residual = max_residual.max(residual);
            }
        }

        max_residual
    }

    /// Subtract pressure gradient from velocity field
    /// Uses formulation: u -= ∇p̃ / h where p̃ is pseudo-pressure from ∇²p̃ = div
    /// Subtract pressure gradient from velocity field
    /// Uses formulation: u -= ∇p̃ / h where p̃ is pseudo-pressure from ∇²p̃ = div
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

    /// Damp vertical velocity at the free surface to eliminate "fizz"
    ///
    /// The top fluid cells in each column experience gravity-induced vertical jitter
    /// that creates a chaotic appearance. This function applies depth-weighted damping
    /// to v.y only, preserving horizontal transport while calming the surface.
    ///
    /// Called AFTER pressure projection so we don't fight incompressibility.
    pub fn damp_surface_vertical(&mut self) {
        // First pass: find the topmost fluid cell in each column (surface_j)
        let mut surface_j: Vec<Option<usize>> = vec![None; self.width];

        for i in 0..self.width {
            for j in 0..self.height {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] == CellType::Fluid {
                    // First fluid cell from top (smallest j = highest because +Y is down)
                    surface_j[i] = Some(j);
                    break;
                }
            }
        }

        // Second pass: damp v.y based on depth from surface
        // Depth 0 (at surface) = strongest damping
        // Depth 3+ = no damping
        const SURFACE_DEPTH: f32 = 3.0;

        for j in 1..self.height {
            for i in 0..self.width {
                let idx = self.cell_index(i, j);
                if self.cell_type[idx] != CellType::Fluid {
                    continue;
                }

                if let Some(surf) = surface_j[i] {
                    // depth = 0 at surface, increases downward
                    let depth = (j as i32 - surf as i32) as f32;

                    if depth < SURFACE_DEPTH {
                        // t = 0 at surface, 1 at SURFACE_DEPTH cells down
                        let t = (depth / SURFACE_DEPTH).clamp(0.0, 1.0);

                        // Damp vertical velocity only
                        // v is stored at bottom edge of cells, so v_index(i, j) is between j-1 and j
                        let v_idx = self.v_index(i, j);
                        self.v[v_idx] *= t;
                    }
                }
            }
        }
    }

    // NOTE: Vorticity methods moved to vorticity.rs:
    // - compute_vorticity
    // - compute_enstrophy
    // - total_absolute_vorticity
    // - max_vorticity
    // - apply_vorticity_confinement
    // - apply_vorticity_confinement_with_piles

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

    /// Total momentum vector of the grid (sum of velocity * cell_size for fluid-adjacent faces)
    /// Only counts faces that touch at least one fluid cell.
    /// Used for conservation tests - extrapolated velocities don't count.
    pub fn total_momentum(&self) -> Vec2 {
        let mut momentum = Vec2::ZERO;

        // Sum U momentum (horizontal) - only for faces adjacent to fluid
        for j in 0..self.height {
            for i in 0..=self.width {
                // Check if either adjacent cell is fluid
                let left_fluid = i > 0 && self.cell_type[self.cell_index(i - 1, j)] == CellType::Fluid;
                let right_fluid = i < self.width && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;

                if left_fluid || right_fluid {
                    let u_idx = self.u_index(i, j);
                    momentum.x += self.u[u_idx] * self.cell_size;
                }
            }
        }

        // Sum V momentum (vertical) - only for faces adjacent to fluid
        for j in 0..=self.height {
            for i in 0..self.width {
                let bottom_fluid = j > 0 && self.cell_type[self.cell_index(i, j - 1)] == CellType::Fluid;
                let top_fluid = j < self.height && self.cell_type[self.cell_index(i, j)] == CellType::Fluid;

                if bottom_fluid || top_fluid {
                    let v_idx = self.v_index(i, j);
                    momentum.y += self.v[v_idx] * self.cell_size;
                }
            }
        }

        momentum
    }

    // ========================================================================
    // MULTIGRID PRESSURE SOLVER
    // ========================================================================

    /// Sync level 0 of multigrid with current grid state
    fn mg_sync_level_zero(&mut self) {
        let level = &mut self.mg_levels[0];

        // CRITICAL: The standard solver uses p = (neighbors - h²*div) / 4
        // But the multigrid uses p = (neighbors - div) / 4 for simplicity.
        // To make them consistent, we pre-multiply divergence by h² here.
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
    /// Uses proper Neumann BC: only count active (non-solid) neighbors
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

                // Count active (non-solid) neighbors and sum their pressures
                // Neumann BC: exclude solid neighbors from the Laplacian stencil
                let mut neighbor_sum = 0.0f32;
                let mut neighbor_count = 0.0f32;

                // Left
                if i > 0 {
                    let n_idx = j * w + (i - 1);
                    if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                        neighbor_sum += self.mg_levels[level].pressure[n_idx];
                        neighbor_count += 1.0;
                    }
                }

                // Right
                if i + 1 < w {
                    let n_idx = j * w + (i + 1);
                    if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                        neighbor_sum += self.mg_levels[level].pressure[n_idx];
                        neighbor_count += 1.0;
                    }
                }

                // Bottom
                if j > 0 {
                    let n_idx = (j - 1) * w + i;
                    if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                        neighbor_sum += self.mg_levels[level].pressure[n_idx];
                        neighbor_count += 1.0;
                    }
                }

                // Top
                if j + 1 < h {
                    let n_idx = (j + 1) * w + i;
                    if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                        neighbor_sum += self.mg_levels[level].pressure[n_idx];
                        neighbor_count += 1.0;
                    }
                }

                // Residual = b - A*x = div - (neighbors - n*p)
                // Note: divergence was pre-multiplied by h² in mg_sync_level_zero
                let div = self.mg_levels[level].divergence[idx];

                if neighbor_count > 0.0 {
                    let laplacian = neighbor_sum - neighbor_count * p_center;
                    self.mg_levels[level].residual[idx] = div - laplacian;
                } else {
                    self.mg_levels[level].residual[idx] = 0.0;
                }
            }
        }
    }

    /// Gauss-Seidel smoothing on a multigrid level
    /// Uses proper Neumann BC: denominator = number of non-solid neighbors
    fn mg_smooth(&mut self, level: usize, iterations: usize) {
        let w = self.mg_levels[level].width;
        let h = self.mg_levels[level].height;

        // Divergence was pre-multiplied by h² in mg_sync_level_zero
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

                        // Count active (non-solid, non-boundary) neighbors
                        let mut neighbor_sum = 0.0f32;
                        let mut neighbor_count = 0.0f32;

                        // Left neighbor
                        if i > 0 {
                            let n_idx = j * w + (i - 1);
                            if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                                neighbor_sum += self.mg_levels[level].pressure[n_idx];
                                neighbor_count += 1.0;
                            }
                            // Solid neighbor: Neumann BC dp/dn=0 means no contribution to Laplacian
                        }
                        // If at domain boundary, no contribution (implicit Neumann)

                        // Right neighbor
                        if i + 1 < w {
                            let n_idx = j * w + (i + 1);
                            if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                                neighbor_sum += self.mg_levels[level].pressure[n_idx];
                                neighbor_count += 1.0;
                            }
                        }

                        // Bottom neighbor
                        if j > 0 {
                            let n_idx = (j - 1) * w + i;
                            if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                                neighbor_sum += self.mg_levels[level].pressure[n_idx];
                                neighbor_count += 1.0;
                            }
                        }

                        // Top neighbor
                        if j + 1 < h {
                            let n_idx = (j + 1) * w + i;
                            if self.mg_levels[level].cell_type[n_idx] != CellType::Solid {
                                neighbor_sum += self.mg_levels[level].pressure[n_idx];
                                neighbor_count += 1.0;
                            }
                        }

                        // GS update with proper denominator
                        // For interior cell: p = (p_L + p_R + p_B + p_T - h²*div) / 4
                        // For boundary cell with n neighbors: p = (sum_neighbors - h²*div) / n
                        let div = self.mg_levels[level].divergence[idx];
                        if neighbor_count > 0.0 {
                            self.mg_levels[level].pressure[idx] = (neighbor_sum - div) / neighbor_count;
                        } else {
                            self.mg_levels[level].pressure[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// Jacobi smoothing on a multigrid level (parallelizable with rayon)
    ///
    /// Unlike Gauss-Seidel, Jacobi reads only from old values, allowing full parallelization.
    /// Trade-off: needs ~2x iterations for same convergence, but uses all cores.
    /// Uses pre-allocated pressure_temp buffer to avoid allocation per iteration.
    fn mg_smooth_jacobi(&mut self, level: usize, iterations: usize) {
        use rayon::prelude::*;

        let level_data = &mut self.mg_levels[level];
        let w = level_data.width;
        let h = level_data.height;

        for _ in 0..iterations {
            // Copy current pressure to temp (source for this iteration)
            level_data.pressure_temp.copy_from_slice(&level_data.pressure);

            // Get references for parallel iteration
            let src = &level_data.pressure_temp;
            let cell_type = &level_data.cell_type;
            let divergence = &level_data.divergence;

            // Parallel Jacobi iteration: read from temp, write to pressure
            level_data.pressure.par_iter_mut().enumerate().for_each(|(idx, dst_p)| {
                if cell_type[idx] != CellType::Fluid {
                    *dst_p = 0.0;
                    return;
                }

                let i = idx % w;
                let j = idx / w;
                let p_center = src[idx];

                // Get neighbor pressures with Neumann mirroring
                let p_left = if i > 0 {
                    let n_idx = j * w + (i - 1);
                    if cell_type[n_idx] == CellType::Solid { p_center } else { src[n_idx] }
                } else {
                    p_center
                };

                let p_right = if i + 1 < w {
                    let n_idx = j * w + (i + 1);
                    if cell_type[n_idx] == CellType::Solid { p_center } else { src[n_idx] }
                } else {
                    p_center
                };

                let p_bottom = if j > 0 {
                    let n_idx = (j - 1) * w + i;
                    if cell_type[n_idx] == CellType::Solid { p_center } else { src[n_idx] }
                } else {
                    p_center
                };

                let p_top = if j + 1 < h {
                    let n_idx = (j + 1) * w + i;
                    if cell_type[n_idx] == CellType::Solid { p_center } else { src[n_idx] }
                } else {
                    p_center
                };

                // Jacobi update: p_new = (neighbors - div) / 4
                let div = divergence[idx];
                *dst_p = (p_left + p_right + p_bottom + p_top - div) * 0.25;
            });
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
        let pre_smooth = 10;
        let post_smooth = 10;
        let coarse_solve = 50;

        // Pre-smoothing (Gauss-Seidel - faster for small grids)
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
