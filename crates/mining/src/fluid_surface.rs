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
pub(crate) fn kernels(
    points: &[Vec3],
    cell: f32,
    sample_volume_m3: f64,
    channel: &[bool],
) -> Vec<Kernel> {
    assert_eq!(points.len(), channel.len());
    let mut field = ChannelField::new(points, cell);
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
        .enumerate()
        .map(|(index, &p)| {
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
            // Warm the legacy PCA cache in its original particle order: its
            // sparse-bin representative must stay identical for outlet drops.
            if channel[index] {
                return field.sample(p, cell, sample_volume_m3);
            }
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
            let local_density = (mass as f64 * sample_volume_m3 / (width as f64).powi(3)) as f32;
            let occupancy = ((local_density - 0.05) / 0.45).clamp(0., 1.);
            // The broad covariance neighbourhood can include a full sheet below
            // detached spray. Require some water in the continuous local stencil
            // before lending that sample a bulk silhouette. These permissive
            // render thresholds retain thin sheets; they are not solver density.
            // Using represented volume keeps coincident subdivision invariant.
            let local_support = ((local_density - 0.025) / 0.02).clamp(0., 1.);
            let local_support = local_support * local_support * (3. - 2. * local_support);
            let support = occupancy * occupancy * (3. - 2. * occupancy);
            let channel_center =
                p + (displacement / mass.max(1e-8)).clamp_length_max(cell * 0.75) * (0.8 * support);
            Kernel {
                axes: kernel.axes,
                normal: kernel.normal,
                planarity: kernel.planarity,
                bulk: kernel.bulk * local_support,
                center: p + (kernel.center - p).clamp_length_max(cell * 0.2) * 0.35,
                channel_center,
            }
        })
        .collect()
}
// Deposit and query continuous moments instead of selecting one bin's PCA.
// A hard bin lookup changes the ellipsoid as a particle crosses a cell face,
// even when its physical trajectory and represented volume are continuous.
struct ChannelField {
    nodes: HashMap<[i32; 3], Moments>,
    neighbourhoods: HashMap<[i32; 3], Moments>,
    stencils: HashMap<[i32; 3], [(Moments, Moments); 8]>,
    width: f32,
}
impl Moments {
    fn translated(self, offset: Vec3) -> Self {
        Self {
            count: self.count,
            sum: self.sum + offset * self.count,
            square: self.square
                + outer(offset) * self.count
                + Mat3::from_cols(
                    self.sum * offset.x,
                    self.sum * offset.y,
                    self.sum * offset.z,
                )
                + Mat3::from_cols(
                    offset * self.sum.x,
                    offset * self.sum.y,
                    offset * self.sum.z,
                ),
        }
    }
    fn accumulate(&mut self, other: Self, weight: f32) {
        self.count += other.count * weight;
        self.sum += other.sum * weight;
        self.square += other.square * weight;
    }
}
fn stencil(p: Vec3, width: f32) -> [([i32; 3], f32); 8] {
    let grid = p / width - Vec3::splat(0.5);
    let base = grid.floor().as_ivec3();
    let f = grid - base.as_vec3();
    std::array::from_fn(|n| {
        let d = IVec3::new((n & 1) as i32, ((n >> 1) & 1) as i32, ((n >> 2) & 1) as i32);
        let weight = Vec3::new(
            if d.x == 0 { 1. - f.x } else { f.x },
            if d.y == 0 { 1. - f.y } else { f.y },
            if d.z == 0 { 1. - f.z } else { f.z },
        )
        .element_product();
        ((base + d).to_array(), weight)
    })
}
impl ChannelField {
    fn center(&self, k: [i32; 3]) -> Vec3 {
        (Vec3::from_array(k.map(|i| i as f32)) + Vec3::splat(0.5)) * self.width
    }
    fn new(points: &[Vec3], cell: f32) -> Self {
        let mut out = Self {
            nodes: HashMap::new(),
            neighbourhoods: HashMap::new(),
            stencils: HashMap::new(),
            width: cell * 2.,
        };
        for &p in points {
            for (k, weight) in stencil(p, out.width) {
                if weight <= 0. {
                    continue;
                }
                let q = p - out.center(k);
                out.nodes.entry(k).or_default().accumulate(
                    Moments {
                        count: 1.,
                        sum: q,
                        square: outer(q),
                    },
                    weight,
                );
            }
        }
        out
    }
    fn sample(&mut self, p: Vec3, cell: f32, volume: f64) -> Kernel {
        let grid = p / self.width - Vec3::splat(0.5);
        let base = grid.floor().as_ivec3().to_array();
        let origin = self.center(base);
        let corners = if let Some(m) = self.stencils.get(&base) {
            *m
        } else {
            let corners = std::array::from_fn(|n| {
                let k = [
                    base[0] + (n & 1) as i32,
                    base[1] + ((n >> 1) & 1) as i32,
                    base[2] + ((n >> 2) & 1) as i32,
                ];
                let center = self.center(k);
                let local = self.nodes.get(&k).copied().unwrap_or_default();
                let total = if let Some(m) = self.neighbourhoods.get(&k) {
                    *m
                } else {
                    let mut m = Moments::default();
                    for z in -1..=1 {
                        for y in -1..=1 {
                            for x in -1..=1 {
                                let key = [k[0] + x, k[1] + y, k[2] + z];
                                if let Some(n) = self.nodes.get(&key) {
                                    m.accumulate(n.translated(self.center(key) - center), 1.);
                                }
                            }
                        }
                    }
                    self.neighbourhoods.insert(k, m);
                    m
                };
                (
                    local.translated(center - origin),
                    total.translated(center - origin),
                )
            });
            self.stencils.insert(base, corners);
            corners
        };
        let mut local = Moments::default();
        let mut total = Moments::default();
        for ((_, weight), (a, b)) in stencil(p, self.width).into_iter().zip(corners) {
            local.accumulate(a, weight);
            total.accumulate(b, weight);
        }
        let local = local.translated(origin - p);
        let total = total.translated(origin - p);
        let mean = total.sum / total.count.max(1e-8);
        let covariance = total.square / total.count.max(1e-8) - outer(mean);
        let (basis, variance) = eigen(covariance);
        let spread = variance.max(Vec3::splat(cell * cell * 0.25));
        let footprint = std::f64::consts::TAU.powf(1.5)
            * ((spread.x as f64) * spread.y as f64 * spread.z as f64).sqrt();
        let density = (total.count as f64 * volume / footprint) as f32;
        let smooth = |t: f32| {
            let t = t.clamp(0., 1.);
            t * t * (3. - 2. * t)
        };
        let occupancy = smooth((density - 0.05) / 0.45);
        let neighbours = smooth((total.count - 1.) / 11.);
        let local_density = (local.count as f64 * volume / (self.width as f64).powi(3)) as f32;
        let bulk = occupancy * neighbours * smooth((local_density - 0.025) / 0.02);
        let values = variance.to_array();
        let mut sorted = [0usize, 1, 2];
        sorted.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let mut normal = basis.col(sorted[0]);
        if normal.y < 0. {
            normal = -normal;
        }
        let planarity =
            (1. - values[sorted[0]].max(0.) / values[sorted[1]].max(1e-10)).clamp(0., 1.);
        // Continuous deposition broadens the density stencil. A front spray
        // point must still lie near that measured layer before borrowing its
        // bulk silhouette. Use the actual moment spread, not a velocity cap.
        let sigma = values[sorted[0]].max(0.).sqrt();
        let proximity = smooth(
            ((2.5 * sigma).max(cell * 0.75) - normal.dot(mean).abs()) / sigma.max(cell * 0.25),
        );
        // An isotropic volume has no distinguished layer normal. Suppress the
        // distance gate there; otherwise tiny eigenvector changes can falsely
        // classify a compact, fully filled cloud as detached spray.
        let layer = smooth((planarity - 0.1) / 0.4);
        let bulk = bulk * (1. - layer * (1. - proximity));
        let floor = variance.max_element().max(1e-10) * 0.16;
        let positive = variance.max(Vec3::splat(floor));
        let scale = Vec3::new(positive.x.sqrt(), positive.y.sqrt(), positive.z.sqrt());
        let geometric = scale.element_product().cbrt().max(1e-10);
        let ratio = (scale / geometric).clamp(Vec3::splat(0.45), Vec3::splat(1.8));
        // Sparse water keeps spherical support. Blend continuously near twelve
        // neighbours so crossing that former hard threshold cannot pop a drop.
        let anisotropy = smooth((total.count - 4.) / 8.);
        let ratio = Vec3::ONE.lerp(ratio, anisotropy);
        let mut axes = [
            basis.x_axis * ratio.x,
            basis.y_axis * ratio.y,
            basis.z_axis * ratio.z,
        ];
        let mut surface_planarity = planarity * anisotropy;
        // A thick crest has three resolved dimensions, so PCA alone calls it
        // isotropic and draws rounded support right at its free boundary.
        // The continuous moment deficit points out of the liquid there. Use
        // that measured boundary normal to flatten connected crest support;
        // an interior volume, side wall or detached drop supplies no such cue.
        let deficit = -mean;
        let boundary_normal = deficit.normalize_or_zero();
        let boundary = smooth((deficit.length() / cell - 0.2) / 0.65)
            * smooth((boundary_normal.y - 0.25) / 0.5)
            * (1. - surface_planarity)
            * bulk;
        if boundary > 0. {
            let n = normal.lerp(boundary_normal, boundary).normalize_or_zero();
            let tx = (Vec3::X - n * n.x).normalize_or_zero();
            let tz = n.cross(tx);
            let tangent_x = tx.dot(covariance * tx).max(cell * cell * 0.25);
            let tangent_z = tz.dot(covariance * tz).max(cell * cell * 0.25);
            let minor = n
                .dot(covariance * n)
                .max(1e-10)
                .min(tangent_x.max(tangent_z) * 0.16);
            let target = outer(tx) * tangent_x + outer(tz) * tangent_z + outer(n) * minor;
            let (v, d) = eigen(covariance * (1. - boundary) + target * boundary);
            let positive = d.max(Vec3::splat(d.max_element().max(1e-10) * 0.16));
            let s = Vec3::new(positive.x.sqrt(), positive.y.sqrt(), positive.z.sqrt());
            let r = (s / s.element_product().cbrt().max(1e-10))
                .clamp(Vec3::splat(0.45), Vec3::splat(1.8));
            axes = [v.x_axis * r.x, v.y_axis * r.y, v.z_axis * r.z];
            normal = n;
            surface_planarity += (1. - surface_planarity) * boundary;
        }
        Kernel {
            axes,
            center: p,
            channel_center: p
                + (local.sum / local.count.max(1e-8)).clamp_length_max(cell * 0.75)
                    * (0.8 * smooth((local_density - 0.05) / 0.45)),
            normal,
            planarity: surface_planarity,
            bulk,
        }
    }
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
    fn thick_water_boundary_has_thin_support_without_flattening_its_interior() {
        let points: Vec<_> = (-10..=10)
            .flat_map(|x| {
                (-24..=0).flat_map(move |y| {
                    (-10..=10).map(move |z| Vec3::new(x as f32, y as f32, z as f32) * 0.005)
                })
            })
            .collect();
        let before = points.clone();
        let k = kernels(&points, 0.01, 1.25e-7);
        let top = points.iter().position(|p| *p == Vec3::ZERO).unwrap();
        let middle = points
            .iter()
            .position(|p| (*p - Vec3::new(0., -0.06, 0.)).length() < 1e-6)
            .unwrap();
        let support = |k: Kernel| {
            let a = k.channel_axes();
            (
                a.iter().map(|p| p.y * p.y).sum::<f32>().sqrt(),
                a.iter().map(|p| p.x * p.x).sum::<f32>().sqrt(),
            )
        };
        let (vertical, horizontal) = support(k[top]);
        assert!(k[top].normal.dot(Vec3::Y) > 0.99);
        assert!(
            vertical / horizontal < 0.34,
            "free boundary support: {vertical}/{horizontal}"
        );
        let (vertical, horizontal) = support(k[middle]);
        assert!(
            (vertical / horizontal - 1.).abs() < 0.02,
            "interior acquired a false surface: {vertical}/{horizontal}"
        );
        assert_eq!(points, before);
    }
    fn kernels(points: &[Vec3], cell: f32, volume: f64) -> Vec<Kernel> {
        super::kernels(points, cell, volume, &vec![true; points.len()])
    }
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
        assert!(
            divided.iter().all(|k| k.bulk == 1.),
            "subdivided minimum bulk {}",
            divided.iter().map(|k| k.bulk).fold(1_f32, f32::min)
        );
    }
    #[test]
    fn spray_above_a_dense_sheet_does_not_borrow_its_bulk_silhouette() {
        let mut points: Vec<_> = (0..12)
            .flat_map(|x| {
                (0..4).flat_map(move |y| {
                    (0..12).map(move |z| {
                        Vec3::new(
                            x as f32 * 0.005 + 0.0025,
                            y as f32 * 0.005 + 0.0025,
                            z as f32 * 0.005 + 0.0025,
                        )
                    })
                })
            })
            .collect();
        points.push(Vec3::new(0.03, 0.03001, 0.03));
        let before = points.clone();
        let data = kernels(&points, 0.01, 1.25e-7);
        assert!(
            data.last().unwrap().bulk < 0.02,
            "detached spray borrowed bulk: {}",
            data.last().unwrap().bulk
        );
        assert_eq!(points, before);
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
    #[test]
    fn curved_channel_shape_is_continuous_across_reconstruction_cells() {
        let shape = |k: Kernel| {
            k.channel_axes()
                .into_iter()
                .map(outer)
                .fold(Mat3::ZERO, |a, b| a + b)
        };
        let sheet: Vec<_> = (-18..=18)
            .flat_map(|x| {
                (-10..=10).map(move |z| {
                    let x = x as f32 * 0.003;
                    let z = z as f32 * 0.003;
                    Vec3::new(x, 0.15 + 0.12 * x + 7. * x * x + 2. * z * z, z)
                })
            })
            .collect();
        let mut worst = 0_f32;
        for x in [-0.04, -0.02, 0., 0.02, 0.04] {
            let at = |dx: f32| {
                let mut p = sheet.clone();
                p.push(Vec3::new(x + dx, 0.15 + 0.12 * x + 7. * x * x, 0.));
                *kernels(&p, 0.01, 1.25e-7).last().unwrap()
            };
            let a = at(-1e-7);
            let b = at(1e-7);
            let error = (shape(a) - shape(b))
                .to_cols_array()
                .into_iter()
                .map(f32::abs)
                .fold(0_f32, f32::max);
            worst = worst.max(error);
            println!(
                "boundary {x}: shape jump {error}, support {} -> {}",
                a.bulk, b.bulk
            );
        }
        assert!(worst < 0.001, "reconstruction-cell shape jump {worst}");
    }
}
