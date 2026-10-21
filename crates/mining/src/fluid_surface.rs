//! Render-only anisotropic kernels from local particle covariance. Moments are
//! accumulated in occupied bins, so reconstruction does not scan every pair.
use bevy::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Moments {
    count: f32,
    sum: Vec3,
    square: Mat3,
}
impl Default for Moments {
    fn default() -> Self {
        Self {
            count: 0.,
            sum: Vec3::ZERO,
            square: Mat3::ZERO,
        }
    }
}
fn outer(p: Vec3) -> Mat3 {
    Mat3::from_cols(p * p.x, p * p.y, p * p.z)
}
#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    pub axes: [Vec3; 3],
    pub center: Vec3,
    pub normal: Vec3,
    pub planarity: f32,
    /// Connected-neighbour support for rendering bulk water versus small spray.
    pub bulk: f32,
}
pub(crate) fn kernels(points: &[Vec3], cell: f32) -> Vec<Kernel> {
    let width = cell * 2.;
    let key = |p: Vec3| (p / width).floor().as_ivec3().to_array();
    let center =
        |k: [i32; 3]| (Vec3::new(k[0] as f32, k[1] as f32, k[2] as f32) + Vec3::splat(0.5)) * width;
    let mut bins = HashMap::<[i32; 3], Moments>::new();
    for &p in points {
        let k = key(p);
        let q = p - center(k);
        let m = bins.entry(k).or_default();
        m.count += 1.;
        m.sum += q;
        m.square += outer(q);
    }
    let mut cache = HashMap::<[i32; 3], Kernel>::new();
    points
        .iter()
        .map(|&p| {
            let k = key(p);
            let kernel = *cache.entry(k).or_insert_with(|| {
                let origin = center(k);
                let mut total = Moments::default();
                for z in -1..=1 {
                    for y in -1..=1 {
                        for x in -1..=1 {
                            let neighbour = [k[0] + x, k[1] + y, k[2] + z];
                            if let Some(m) = bins.get(&neighbour) {
                                let offset = center(neighbour) - origin;
                                total.count += m.count;
                                total.sum += m.sum + offset * m.count;
                                total.square += m.square
                                    + outer(offset) * m.count
                                    + Mat3::from_cols(
                                        m.sum * offset.x,
                                        m.sum * offset.y,
                                        m.sum * offset.z,
                                    )
                                    + Mat3::from_cols(
                                        offset * m.sum.x,
                                        offset * m.sum.y,
                                        offset * m.sum.z,
                                    );
                            }
                        }
                    }
                }
                if total.count < 12. {
                    let support = ((total.count - 1.) / 11.).clamp(0., 1.);
                    return Kernel {
                        axes: [Vec3::X, Vec3::Y, Vec3::Z],
                        center: p,
                        normal: Vec3::Y,
                        planarity: 0.,
                        bulk: support * support * (3. - 2. * support),
                    };
                }
                let mean = total.sum / total.count;
                let covariance = total.square / total.count - outer(mean);
                let (basis, variance) = eigen(covariance);
                let values = variance.to_array();
                let mut sorted = [0usize, 1, 2];
                sorted.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
                let mut normal = basis.col(sorted[0]);
                if normal.y < 0. {
                    normal = -normal;
                }
                let planarity =
                    (1. - values[sorted[0]].max(0.) / values[sorted[1]].max(1e-10)).clamp(0., 1.);
                let floor = variance.max_element().max(1e-10) * 0.16;
                let positive = variance.max(Vec3::splat(floor));
                let scale = Vec3::new(positive.x.sqrt(), positive.y.sqrt(), positive.z.sqrt());
                let geometric = (scale.x * scale.y * scale.z).cbrt().max(1e-10);
                let ratio = (scale / geometric).clamp(Vec3::splat(0.45), Vec3::splat(1.8));
                Kernel {
                    axes: [
                        basis.x_axis * ratio.x,
                        basis.y_axis * ratio.y,
                        basis.z_axis * ratio.z,
                    ],
                    center: origin + mean,
                    normal,
                    planarity,
                    bulk: 1.,
                }
            });
            Kernel {
                axes: kernel.axes,
                normal: kernel.normal,
                planarity: kernel.planarity,
                bulk: kernel.bulk,
                center: p + (kernel.center - p).clamp_length_max(cell * 0.2) * 0.35,
            }
        })
        .collect()
}
/// Fixed small symmetric Jacobi solve. Eigenvectors remain orthonormal even for
/// nearly isotropic water; no velocity-directed stretching or solver changes.
fn eigen(m: Mat3) -> (Mat3, Vec3) {
    let mut a = m.to_cols_array_2d();
    let mut v = Mat3::IDENTITY.to_cols_array_2d();
    for _ in 0..5 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[q][p];
            if apq.abs() < 1e-12 {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2. * apq);
            let t = if tau >= 0. {
                1. / (tau + (1. + tau * tau).sqrt())
            } else {
                -1. / (-tau + (1. + tau * tau).sqrt())
            };
            let c = 1. / (1. + t * t).sqrt();
            let s = t * c;
            for row in &mut a {
                let ap = row[p];
                let aq = row[q];
                row[p] = c * ap - s * aq;
                row[q] = s * ap + c * aq;
            }
            for row in &mut v {
                let ap = row[p];
                let aq = row[q];
                row[p] = c * ap - s * aq;
                row[q] = s * ap + c * aq;
            }
            // Above arrays are columns: rotate rows, then columns explicitly.
            let (before, after) = a.split_at_mut(q);
            for (ap, aq) in before[p].iter_mut().zip(&mut after[0]) {
                let old_p = *ap;
                let old_q = *aq;
                *ap = c * old_p - s * old_q;
                *aq = s * old_p + c * old_q;
            }
        }
    }
    // v's row rotations above produce the transpose of the accumulated basis.
    (
        Mat3::from_cols_array_2d(&v).transpose(),
        Vec3::new(a[0][0], a[1][1], a[2][2]),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotated_covariance_reconstruction_retains_orthogonal_axes() {
        let rotation = Mat3::from_rotation_z(0.63) * Mat3::from_rotation_y(0.27);
        let m = rotation * Mat3::from_diagonal(Vec3::new(0.02, 0.001, 0.01)) * rotation.transpose();
        let (v, d) = eigen(m);
        let reconstructed = v * Mat3::from_diagonal(d) * v.transpose();
        assert!(
            (reconstructed - m)
                .to_cols_array()
                .iter()
                .all(|n| n.abs() < 1e-6),
            "{reconstructed:?} != {m:?}"
        );
        assert!((v.determinant().abs() - 1.).abs() < 1e-5);
    }
    #[test]
    fn thin_sheet_gets_flattened_kernels_without_mutating_particles() {
        let points: Vec<_> = (-8..=8)
            .flat_map(|x| (-8..=8).map(move |z| Vec3::new(x as f32 * 0.005, 0., z as f32 * 0.005)))
            .collect();
        let before = points.clone();
        let data = kernels(&points, 0.01);
        assert_eq!(points, before);
        assert_eq!(data.len(), points.len());
        let k = data[data.len() / 2];
        let vertical = k.axes.iter().map(|v| v.y * v.y).sum::<f32>().sqrt();
        let horizontal = k.axes.iter().map(|v| v.x * v.x).sum::<f32>().sqrt();
        assert!(k.normal.dot(Vec3::Y) > 0.99 && k.planarity > 0.99);
        assert!(vertical < horizontal * 0.6, "{vertical} / {horizontal}");
        assert!(data
            .iter()
            .flat_map(|k| k.axes)
            .all(|v| v.is_finite() && v.length() <= 1.81));
    }
    #[test]
    fn detached_water_has_no_bulk_support_while_a_connected_sheet_does() {
        let isolated = kernels(&[Vec3::ZERO, Vec3::new(0.2, 0.1, 0.)], 0.01);
        assert!(isolated.iter().all(|k| k.bulk == 0.));
        let sheet: Vec<_> = (0..25)
            .map(|i| Vec3::new((i % 5) as f32 * 0.003, 0., (i / 5) as f32 * 0.003))
            .collect();
        assert!(kernels(&sheet, 0.01).iter().all(|k| k.bulk == 1.));
    }
}
