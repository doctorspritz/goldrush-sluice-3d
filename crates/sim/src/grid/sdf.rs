//! Signed Distance Field computation and sampling.
//!
//! SDF represents distance to solid boundaries (negative inside solid).
//!
//! NOTE: These are COPIES of methods from mod.rs for the refactor extraction.
//! The originals in mod.rs should be deleted in Phase 3 by the Lead agent.

use super::{CellType, Grid};
use glam::Vec2;

// ============================================================================
// SDF METHODS (impl Grid)
// These use _impl suffix to avoid conflicts with the originals in mod.rs
// ============================================================================

impl Grid {
    /// Compute signed distance field from solid cells using fast sweeping
    /// Call this after terrain changes (set_solid calls)
    ///
    /// Returns: negative inside solid, zero at surface, positive outside
    ///
    /// Optimization: Uses pre-allocated inner_sdf buffer and single sweep pass.
    ///
    /// Copy of method from mod.rs - uses _impl suffix to avoid conflict
    pub fn compute_sdf_impl(&mut self) {
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

    /// Sample SDF at world position (bilinear interpolation)
    ///
    /// Copy of method from mod.rs - uses _impl suffix to avoid conflict
    #[inline]
    pub fn sample_sdf_impl(&self, pos: Vec2) -> f32 {
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
    ///
    /// Copy of method from mod.rs - uses _impl suffix to avoid conflict
    #[inline]
    pub fn sdf_gradient_impl(&self, pos: Vec2) -> Vec2 {
        let eps = self.cell_size * 0.5;
        let dx = self.sample_sdf(pos + Vec2::new(eps, 0.0)) - self.sample_sdf(pos - Vec2::new(eps, 0.0));
        let dy = self.sample_sdf(pos + Vec2::new(0.0, eps)) - self.sample_sdf(pos - Vec2::new(0.0, eps));
        Vec2::new(dx, dy).normalize_or_zero()
    }

    // ========================================================================
    // BED HEIGHT METHODS
    // ========================================================================

    /// Compute bed heightfield from solid terrain
    /// For each column, find the topmost solid cell and store its top edge y-coordinate
    /// Call this after terrain setup (after compute_sdf)
    ///
    /// Copy of method from mod.rs - uses _impl suffix to avoid conflict
    pub fn compute_bed_heights_impl(&mut self) {
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
    ///
    /// Copy of method from mod.rs - uses _impl suffix to avoid conflict
    #[inline]
    pub fn sample_bed_height_impl(&self, x: f32) -> f32 {
        let fx = x / self.cell_size - 0.5;
        let i0 = (fx.floor() as i32).clamp(0, self.width as i32 - 2) as usize;
        let i1 = i0 + 1;
        let t = (fx - i0 as f32).clamp(0.0, 1.0);
        self.bed_height[i0] * (1.0 - t) + self.bed_height[i1] * t
    }

    /// Compute normalized height above bed (0 = at bed, 1 = at water surface)
    /// surface_height is the estimated water surface level
    ///
    /// Copy of method from mod.rs - uses _impl suffix to avoid conflict
    #[inline]
    pub fn normalized_height_above_bed_impl(&self, pos: Vec2, surface_height: f32) -> f32 {
        let bed = self.sample_bed_height(pos.x);
        let height_above_bed = (pos.y - bed).max(0.0);
        let water_depth = (surface_height - bed).max(0.01); // avoid div by zero
        (height_above_bed / water_depth).clamp(0.0, 1.0)
    }
}
