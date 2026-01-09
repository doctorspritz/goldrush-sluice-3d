//! Interpolation helpers for APIC/FLIP transfer.
//!
//! Contains B-spline kernels and APIC matrix computation.

#![allow(dead_code, unused_imports)]  // Code will be used after Phase 3 migration

use glam::Vec2;

// ============================================================================
// PHASE 3 MIGRATION: Uncomment these functions after deleting originals from mod.rs
// Also add to mod.rs: pub use interp::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d};
// Functions to delete from mod.rs:
//   - quadratic_bspline_1d
//   - quadratic_bspline
//   - apic_d_inverse
// ============================================================================

// COMMENTED OUT - Enable in Phase 3 when originals are deleted from mod.rs
/*
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
*/
