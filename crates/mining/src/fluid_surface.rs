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
    pub channel_center: Vec3,
    pub normal: Vec3,
    pub planarity: f32,
    /// Connected-neighbour support for rendering bulk water versus small spray.
    pub bulk: f32,
}
impl Kernel {
    /// Channel sheets use thinner support along their measured minor axis.
    /// Tangent expansion preserves kernel volume; spray and the outlet retain
    /// the original axes. This changes reconstruction, never particle motion.
    pub(crate) fn channel_axes(self) -> [Vec3; 3] {
        let mut axes = self.axes;
        let minor = (0..3)
            .min_by(|&a, &b| {
                axes[a]
                    .length_squared()
                    .total_cmp(&axes[b].length_squared())
            })
            .unwrap();
        let thickness = 1. - 0.5 * self.planarity * self.bulk;
        for (i, axis) in axes.iter_mut().enumerate() {
            *axis *= if i == minor {
                thickness
            } else {
                thickness.sqrt().recip()
            };
        }
        axes
    }
}
pub(crate) fn kernels(points: &[Vec3], cell: f32, sample_volume_m3: f64) -> Vec<Kernel> {
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
    let mut centroid_cache = HashMap::<[i32; 3], [(f32, Vec3); 8]>::new();
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
                let mean = total.sum / total.count;
                let covariance = total.square / total.count - outer(mean);
                let (basis, variance) = eigen(covariance);
                // A dozen neighbours across a 6 cm region can still be sparse
                // spray. Estimate local occupancy from represented volume and
                // covariance, rather than mistaking neighbour count for density.
                // A half-cell variance floor also supports thin water sheets.
                // This selects reconstruction support only, never solver density.
                let spread = variance.max(Vec3::splat(cell * cell * 0.25));
                let footprint = std::f64::consts::TAU.powf(1.5)
                    * ((spread.x as f64) * spread.y as f64 * spread.z as f64).sqrt();
                let density = (total.count as f64 * sample_volume_m3 / footprint) as f32;
                let occupancy = ((density - 0.05) / 0.45).clamp(0., 1.);
                let neighbours = ((total.count - 1.) / 11.).clamp(0., 1.);
                let support = occupancy
                    * occupancy
                    * (3. - 2. * occupancy)
                    * neighbours
                    * neighbours
                    * (3. - 2. * neighbours);
                if total.count < 12. {
                    return Kernel {
                        axes: [Vec3::X, Vec3::Y, Vec3::Z],
                        center: p,
                        channel_center: p,
                        normal: Vec3::Y,
                        planarity: 0.,
                        bulk: support,
                    };
                }
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
                    channel_center: origin + mean,
                    normal,
                    planarity,
                    bulk: support,
                }
            });
            // Continuous centroid relaxation joins the particle envelope at
            // crests. Interpolate occupied-bin moments, not bin-selected
            // centroids: a moving sample must not jump at an optical-cell edge.
            // This is a bounded render displacement, with no solver writeback.
            let grid = p / width - Vec3::splat(0.5);
            let base = grid.floor().as_ivec3();
            let f = grid - base.as_vec3();
            let base_key = base.to_array();
            let origin = center(base_key);
            let corners = centroid_cache.entry(base_key).or_insert_with(|| {
                std::array::from_fn(|n| {
                    let k = [
                        base_key[0] + (n & 1) as i32,
                        base_key[1] + ((n >> 1) & 1) as i32,
                        base_key[2] + ((n >> 2) & 1) as i32,
                    ];
                    bins.get(&k)
                        .map(|m| (m.count, m.sum + (center(k) - origin) * m.count))
                        .unwrap_or_default()
                })
            });
            let mut mass = 0.;
            let mut displacement = Vec3::ZERO;
            for (n, &(count, sum)) in corners.iter().enumerate() {
                let w = Vec3::new(
                    if n & 1 == 0 { 1. - f.x } else { f.x },
                    if n & 2 == 0 { 1. - f.y } else { f.y },
                    if n & 4 == 0 { 1. - f.z } else { f.z },
                )
                .element_product();
                mass += count * w;
                displacement += (sum + (origin - p) * count) * w;
            }
            let occupancy = ((mass as f64 * sample_volume_m3 / (width as f64).powi(3) - 0.05)
                / 0.45)
                .clamp(0., 1.) as f32;
            let support = occupancy * occupancy * (3. - 2. * occupancy);
            let channel_center =
                p + (displacement / mass.max(1e-8)).clamp_length_max(cell * 0.75) * (0.8 * support);
            Kernel {
                axes: kernel.axes,
                normal: kernel.normal,
                planarity: kernel.planarity,
                bulk: kernel.bulk,
                center: p + (kernel.center - p).clamp_length_max(cell * 0.2) * 0.35,
                channel_center,
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
        let data = kernels(&points, 0.01, 1.25e-7);
        assert_eq!(points, before);
        assert_eq!(data.len(), points.len());
        let k = data[data.len() / 2];
        let axes = k.channel_axes();
        let vertical = axes.iter().map(|v| v.y * v.y).sum::<f32>().sqrt();
        let horizontal = axes.iter().map(|v| v.x * v.x).sum::<f32>().sqrt();
        assert!(k.normal.dot(Vec3::Y) > 0.99 && k.planarity > 0.99);
        assert!(vertical < horizontal * 0.35, "{vertical} / {horizontal}");
        let volume = |axes: [Vec3; 3]| axes.iter().map(|a| a.length()).product::<f32>();
        assert!((volume(axes) - volume(k.axes)).abs() < 1e-5);
        let spray = Kernel { bulk: 0., ..k };
        assert_eq!(spray.channel_axes(), spray.axes);
        assert!(data
            .iter()
            .flat_map(|k| k.axes)
            .all(|v| v.is_finite() && v.length() <= 1.81));
    }
    #[test]
    fn sparse_crest_clouds_do_not_inherit_bulk_water_silhouettes() {
        let sparse: Vec<_> = (0..27)
            .map(|i| {
                Vec3::new(
                    (i % 3) as f32 * 0.019,
                    ((i / 3) % 3) as f32 * 0.019,
                    (i / 9) as f32 * 0.019,
                )
            })
            .collect();
        let compact: Vec<_> = sparse.iter().map(|p| *p * 0.16).collect();
        let spray = kernels(&sparse, 0.01, 1.25e-7);
        let bulk = kernels(&compact, 0.01, 1.25e-7);
        assert!(spray.iter().all(|k| k.bulk < 0.1));
        assert!(bulk.iter().all(|k| k.bulk == 1.));
        // Coincident subdivision changes sampling, not represented occupancy.
        let split: Vec<_> = compact.iter().flat_map(|p| [*p, *p]).collect();
        let divided = kernels(&split, 0.01, 6.25e-8);
        assert!(divided.iter().all(|k| k.bulk == 1.));
    }
    #[test]
    fn detached_water_has_no_bulk_support_while_a_connected_sheet_does() {
        let isolated = kernels(&[Vec3::ZERO, Vec3::new(0.2, 0.1, 0.)], 0.01, 1.25e-7);
        assert!(isolated.iter().all(|k| k.bulk == 0.));
        let sheet: Vec<_> = (0..25)
            .map(|i| Vec3::new((i % 5) as f32 * 0.003, 0., (i / 5) as f32 * 0.003))
            .collect();
        assert!(kernels(&sheet, 0.01, 1.25e-7).iter().all(|k| k.bulk == 1.));
    }
    #[test]
    fn channel_relaxation_reduces_sampling_noise_without_moving_water_or_spray() {
        let points: Vec<_> = (-12..=12)
            .flat_map(|x| {
                (-12..=12).map(move |z| {
                    Vec3::new(
                        x as f32 * 0.003,
                        0.15 + x as f32 * 0.0003 + if (x + z) % 2 == 0 { 0.003 } else { -0.003 },
                        z as f32 * 0.003,
                    )
                })
            })
            .collect();
        let saved = points.clone();
        let data = kernels(&points, 0.01, 1.25e-7);
        let mut before = 0.;
        let mut after = 0.;
        for (&p, k) in points.iter().zip(data) {
            assert!((k.channel_center - p).length() <= 0.006001);
            if p.x.abs() < 0.025 && p.z.abs() < 0.025 {
                before += (p.y - (0.15 + 0.1 * p.x)).powi(2);
                after += (k.channel_center.y - (0.15 + 0.1 * k.channel_center.x)).powi(2);
            }
        }
        assert!(
            after < before * 0.5,
            "sampling residual {after} vs {before}"
        );
        assert_eq!(points, saved);
        let p = Vec3::new(-0.2, 0.31, -0.02);
        assert_eq!(kernels(&[p], 0.01, 1.25e-7)[0].channel_center, p);
    }
    #[test]
    fn centroid_query_is_continuous_across_negative_and_positive_lattice_phases() {
        let fixed: Vec<_> = (-15..=15)
            .flat_map(|x| {
                (-6..=6).map(move |z| Vec3::new(x as f32 * 0.002, 0.15, z as f32 * 0.002))
            })
            .collect();
        for x in [-0.01, 0.01] {
            let query = |offset: f32| {
                let mut p = fixed.clone();
                p.push(Vec3::new(x + offset, 0.153, 0.));
                kernels(&p, 0.01, 1.25e-7).last().unwrap().channel_center
            };
            assert!((query(1e-6) - query(-1e-6)).length() < 1e-5);
        }
    }
}
