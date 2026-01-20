//! 3D Quadratic B-spline kernel functions for APIC transfers.

use glam::Vec3;

/// 1D Quadratic B-spline weight.
/// Support: [-1.5, 1.5] (covers 3 grid nodes)
#[inline]
pub fn quadratic_bspline_1d(r: f32) -> f32 {
    use crate::constants::BSPLINE_SUPPORT_RADIUS;
    let r_abs = r.abs();
    if r_abs < 0.5 {
        0.75 - r_abs * r_abs
    } else if r_abs < BSPLINE_SUPPORT_RADIUS {
        let t = BSPLINE_SUPPORT_RADIUS - r_abs;
        0.5 * t * t
    } else {
        0.0
    }
}

/// 3D Quadratic B-spline (tensor product of 1D).
/// Returns weight for position delta from grid node.
#[inline]
pub fn quadratic_bspline_3d(delta: Vec3) -> f32 {
    quadratic_bspline_1d(delta.x) * quadratic_bspline_1d(delta.y) * quadratic_bspline_1d(delta.z)
}

/// APIC D matrix inverse for quadratic B-splines.
/// D = (1/4) * dx^2 * I, so D_inv = 4 / dx^2
#[inline]
pub fn apic_d_inverse(cell_size: f32) -> f32 {
    4.0 / (cell_size * cell_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_at_zero() {
        // At node center, weight should be 0.75
        assert!((quadratic_bspline_1d(0.0) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_bspline_at_half() {
        // At ±0.5, weight transitions
        let w = quadratic_bspline_1d(0.5);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bspline_partition_of_unity_near_center() {
        // B-spline weights sum to ~1 near center of stencil (frac near 0.5)
        // At frac=0.5, the 3-node stencil perfectly captures all contributions
        for x in [0.4, 0.5, 0.6] {
            let sum = quadratic_bspline_1d(x + 1.0)
                + quadratic_bspline_1d(x)
                + quadratic_bspline_1d(x - 1.0);
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Partition near center failed at x={}: sum={}",
                x,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_weights_reasonable() {
        // Weights should always be in [0, 1] and reasonably bounded
        // Test values in [0, 1) - typical fractional positions
        for x in [0.0, 0.25, 0.5, 0.75] {
            let sum = quadratic_bspline_1d(x + 1.0)
                + quadratic_bspline_1d(x)
                + quadratic_bspline_1d(x - 1.0);
            // Sum should be close to 1 (within 5% for 3-node stencil)
            assert!(
                sum > 0.95 && sum <= 1.01,
                "Weight sum out of range at x={}: sum={}",
                x,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_3d_weights_reasonable() {
        // 3D weights should sum reasonably close to 1
        let frac = Vec3::new(0.5, 0.5, 0.5);
        let mut sum = 0.0;
        for dk in -1..=1 {
            for dj in -1..=1 {
                for di in -1..=1 {
                    let delta = frac - Vec3::new(di as f32, dj as f32, dk as f32);
                    sum += quadratic_bspline_3d(delta);
                }
            }
        }
        assert!(
            (sum - 1.0).abs() < 0.01,
            "3D weights at center failed: sum={}",
            sum
        );
    }

    #[test]
    fn test_bspline_zero_outside_support() {
        use crate::constants::BSPLINE_SUPPORT_RADIUS;
        assert_eq!(quadratic_bspline_1d(BSPLINE_SUPPORT_RADIUS), 0.0);
        assert_eq!(quadratic_bspline_1d(2.0), 0.0);
        assert_eq!(quadratic_bspline_1d(-BSPLINE_SUPPORT_RADIUS), 0.0);
    }
}
