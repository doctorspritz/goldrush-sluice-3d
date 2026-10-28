//! Render-only anisotropic kernels from local particle covariance. Moments are
//! accumulated in occupied bins, so reconstruction does not scan every pair.
use bevy::math::{Mat3A, Vec3A};
use bevy::prelude::*;
// Integer spatial keys are internal render data. Use the engine cache hasher;
// all floating-point accumulation remains in explicit particle/stencil order.
use bevy::platform::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct Moments {
    count: f32,
    sum: Vec3,
    square: Mat3A,
}
impl Default for Moments {
    fn default() -> Self {
        Self {
            count: 0.,
            sum: Vec3::ZERO,
            square: Mat3A::ZERO,
        }
    }
}
// SIMD storage preserves each scalar operation and explicit accumulation order.
// Convert back before covariance/shape math; padding is never an optical moment.
fn moment_outer(p: Vec3) -> Mat3A {
    let v = Vec3A::from(p);
    Mat3A::from_cols(v * p.x, v * p.y, v * p.z)
}
fn moment_columns(a: Vec3, b: Vec3) -> Mat3A {
    let a = Vec3A::from(a);
    Mat3A::from_cols(a * b.x, a * b.y, a * b.z)
}
fn outer(p: Vec3) -> Mat3 {
    Mat3::from_cols(p * p.x, p * p.y, p * p.z)
}
// Retain the complete measured tangent ellipse at a free boundary. Keeping
// only two tangent diagonal entries erases their covariance and turns a crest
// rotated away from those arbitrary axes into round support.
fn boundary_covariance(covariance: Mat3, normal: Vec3, tangent_floor: f32) -> Mat3 {
    let projection = Mat3::IDENTITY - outer(normal);
    let tangent = projection * covariance * projection;
    // The two nonzero eigenvalues of P C P give a basis-independent floor.
    // Lift only an unresolved minor tangent: the resolved major footprint must
    // not widen to compensate for missing variance in another direction.
    let trace = tangent.x_axis.x + tangent.y_axis.y + tangent.z_axis.z;
    let square = tangent
        .to_cols_array()
        .into_iter()
        .map(|v| v * v)
        .sum::<f32>();
    let spread = (2. * square - trace * trace).max(0.).sqrt();
    let minimum = 0.5 * (trace - spread);
    let maximum = 0.5 * (trace + spread);
    let tangent = if maximum <= tangent_floor {
        projection * tangent_floor
    } else if minimum < tangent_floor {
        // Here maximum > floor > minimum, so spread is positive and this
        // weight is bounded. Both tangent eigenvectors remain unchanged.
        let weight = ((tangent_floor - minimum) / spread).clamp(0., 1.);
        tangent * (1. - weight) + projection * (maximum * weight)
    } else {
        tangent
    };
    let minor = normal
        .dot(covariance * normal)
        .max(1e-10)
        .min(maximum.max(tangent_floor) * 0.16);
    tangent + outer(normal) * minor
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
    /// Samples in the covariance neighbourhood; sparse spray and dilute
    /// connected sheets must not be classified from occupancy alone.
    pub neighbour_count: f32,
}
impl Kernel {
    /// Channel sheets use thinner support along their measured surface normal.
    /// Keep the measured tangential footprint: redistributing normal support
    /// into its tangents inflates crest shoulders and muddy optical patches.
    /// The caller normalizes by the actual determinant, retaining represented
    /// water volume. Spray and the outlet retain their original support.
    pub(crate) fn channel_axes(self) -> [Vec3; 3] {
        let thickness = 1. - 0.5 * self.planarity * self.bulk;
        if self.bulk == 0. || thickness == 1. {
            return self.axes;
        }
        let normal = self.normal.normalize_or_zero();
        if normal == Vec3::ZERO {
            return self.axes;
        }
        // Exact coordinate-aligned sheets need no eigensolve: S acts on one
        // axis only and every other support axis has zero normal component.
        for coordinate in 0..3 {
            if normal[coordinate].abs() != 1.
                || normal[(coordinate + 1) % 3] != 0.
                || normal[(coordinate + 2) % 3] != 0.
            {
                continue;
            }
            for index in 0..3 {
                if self.axes[index][coordinate] != 0.
                    && self.axes[index][(coordinate + 1) % 3] == 0.
                    && self.axes[index][(coordinate + 2) % 3] == 0.
                    && (0..3).all(|other| other == index || self.axes[other][coordinate] == 0.)
                {
                    let mut axes = self.axes;
                    axes[index] *= thickness;
                    return axes;
                }
            }
        }
        // A normal and a short tangent can share an eigenvalue after flooring.
        // Choosing the shortest axis then depends on an arbitrary eigenbasis.
        // Compress the measured NORMAL in the support tensor instead. Every
        // tangent t retains t^T C t, including through eigenvalue degeneracy.
        // Let A contain the original support columns and B = S A. Instead of
        // building and diagonalizing the rotated world tensor B B^T, solve its
        // right Gram matrix G = B^T B. If V diagonalizes G, B V has orthogonal
        // columns and exactly the same support tensor: B V V^T B^T = S C S.
        // This also preserves the actual rounded input axes; it does not assume
        // their Gram matrix is exactly diagonal. Near a measured principal
        // normal G is nearly diagonal regardless of the world's orientation.
        let compressed = Mat3::from_cols(
            self.axes[0] - normal * ((1. - thickness) * normal.dot(self.axes[0])),
            self.axes[1] - normal * ((1. - thickness) * normal.dot(self.axes[1])),
            self.axes[2] - normal * ((1. - thickness) * normal.dot(self.axes[2])),
        );
        let x = compressed.x_axis;
        let y = compressed.y_axis;
        let z = compressed.z_axis;
        let xy = x.dot(y);
        let xz = x.dot(z);
        let yz = y.dot(z);
        let gram = Mat3::from_cols(
            Vec3::new(x.dot(x), xy, xz),
            Vec3::new(xy, y.dot(y), yz),
            Vec3::new(xz, yz, z.dot(z)),
        );
        let (right, _) = eigen(gram);
        [
            compressed * right.x_axis,
            compressed * right.y_axis,
            compressed * right.z_axis,
        ]
    }
    #[cfg(test)]
    fn channel_axes_world_reference(self) -> [Vec3; 3] {
        let thickness = 1. - 0.5 * self.planarity * self.bulk;
        if self.bulk == 0. || thickness == 1. {
            return self.axes;
        }
        let normal = self.normal.normalize_or_zero();
        if normal == Vec3::ZERO {
            return self.axes;
        }
        // Exact coordinate-aligned sheets need no eigensolve: S acts on one
        // axis only and every other support axis has zero normal component.
        for coordinate in 0..3 {
            if normal[coordinate].abs() != 1.
                || normal[(coordinate + 1) % 3] != 0.
                || normal[(coordinate + 2) % 3] != 0.
            {
                continue;
            }
            for index in 0..3 {
                if self.axes[index][coordinate] != 0.
                    && self.axes[index][(coordinate + 1) % 3] == 0.
                    && self.axes[index][(coordinate + 2) % 3] == 0.
                    && (0..3).all(|other| other == index || self.axes[other][coordinate] == 0.)
                {
                    let mut axes = self.axes;
                    axes[index] *= thickness;
                    return axes;
                }
            }
        }
        // A normal and a short tangent can share an eigenvalue after flooring.
        // Choosing the shortest axis then depends on an arbitrary eigenbasis.
        // Compress the measured NORMAL in the support tensor instead. Every
        // tangent t retains t^T C t, including through eigenvalue degeneracy.
        let covariance = self
            .axes
            .into_iter()
            .map(outer)
            .fold(Mat3::ZERO, |a, b| a + b);
        let compression = Mat3::IDENTITY - outer(normal) * (1. - thickness);
        let (basis, variance) = eigen(compression * covariance * compression);
        let positive = variance.max(Vec3::ZERO);
        let lengths = Vec3::new(positive.x.sqrt(), positive.y.sqrt(), positive.z.sqrt());
        [
            basis.x_axis * lengths.x,
            basis.y_axis * lengths.y,
            basis.z_axis * lengths.z,
        ]
    }
}
pub(crate) fn kernels(
    points: &[Vec3],
    cell: f32,
    sample_volume_m3: f64,
    channel: &[bool],
) -> Vec<Kernel> {
    assert_eq!(points.len(), channel.len());
    #[cfg(test)]
    let deposit_started = std::time::Instant::now();
    let mut field = ChannelField::new(points, cell);
    #[cfg(test)]
    profile::record_stage("channel_moment_deposit", deposit_started.elapsed());
    #[cfg(test)]
    let bins_started = std::time::Instant::now();
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
        m.square += moment_outer(q);
    }
    // A channel-only bin never consumes the legacy PCA. Identify bins needed
    // by outlet samples before the particle-order traversal. Shared bins still
    // warm on their FIRST particle, retaining the sparse outlet representative.
    let legacy_bins: HashSet<_> = points
        .iter()
        .zip(channel)
        .filter(|(_, in_channel)| !**in_channel)
        .map(|(&p, _)| key(p))
        .collect();
    let mut cache = HashMap::<[i32; 3], Kernel>::new();
    let mut centroid_cache = HashMap::<[i32; 3], [(f32, Vec3); 8]>::new();
    #[cfg(test)]
    profile::record_stage("legacy_outlet_bins", bins_started.elapsed());
    #[cfg(test)]
    let kernels_started = std::time::Instant::now();
    let kernels = points
        .iter()
        .enumerate()
        .map(|(index, &p)| {
            let k = key(p);
            if channel[index] && !legacy_bins.contains(&k) {
                return field.sample(p, cell, sample_volume_m3);
            }
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
                                    + moment_outer(offset) * m.count
                                    + moment_columns(m.sum, offset)
                                    + moment_columns(offset, m.sum);
                            }
                        }
                    }
                }
                let mean = total.sum / total.count;
                let covariance = Mat3::from(total.square) / total.count - outer(mean);
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
                        neighbour_count: total.count,
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
                    neighbour_count: total.count,
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
                neighbour_count: kernel.neighbour_count,
                center: p + (kernel.center - p).clamp_length_max(cell * 0.2) * 0.35,
                channel_center,
            }
        })
        .collect();
    #[cfg(test)]
    profile::record_stage("kernel_field_evaluation", kernels_started.elapsed());
    kernels
}
// Deposit and query continuous moments instead of selecting one bin's PCA.
// A hard bin lookup changes the ellipsoid as a particle crosses a cell face,
// even when its physical trajectory and represented volume are continuous.
struct ChannelField {
    nodes: HashMap<[i32; 3], Moments>,
    neighbourhoods: HashMap<[i32; 3], Moments>,
    stencils: HashMap<[i32; 3], [(Moments, Moments); 27]>,
    width: f32,
}
impl Moments {
    fn translated(self, offset: Vec3) -> Self {
        Self {
            count: self.count,
            sum: self.sum + offset * self.count,
            square: self.square
                + moment_outer(offset) * self.count
                + moment_columns(self.sum, offset)
                + moment_columns(offset, self.sum),
        }
    }
    fn accumulate(&mut self, other: Self, weight: f32) {
        self.count += other.count * weight;
        self.sum += other.sum * weight;
        self.square += other.square * weight;
    }
}
// Positive quadratic B-splines retain partition of unity and first moments.
// Their variance is independent of the lattice phase. Tent weights changed
// from zero to h²/4 as a parcel crossed one cell, so the measured shape and
// the water/extinction support breathed even on a smooth physical trajectory.
fn lattice_weights(p: Vec3, width: f32) -> (IVec3, [[f32; 3]; 3]) {
    let grid = p / width - Vec3::splat(0.5);
    let base = (grid - Vec3::splat(0.5)).floor().as_ivec3();
    let t = grid - base.as_vec3() - Vec3::ONE;
    let weights = t.to_array().map(|t| {
        [
            0.5 * (0.5 - t).powi(2),
            0.75 - t * t,
            0.5 * (0.5 + t).powi(2),
        ]
    });
    (base, weights)
}
fn stencil(p: Vec3, width: f32) -> [([i32; 3], f32); 27] {
    let (base, weights) = lattice_weights(p, width);
    std::array::from_fn(|n| {
        let d = IVec3::new((n % 3) as i32, ((n / 3) % 3) as i32, (n / 9) as i32);
        (
            (base + d).to_array(),
            weights[0][d.x as usize] * weights[1][d.y as usize] * weights[2][d.z as usize],
        )
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
            // Match the phase-averaged variance of the former tent stencil:
            // h_quad²/4 = h_tent²/6. Smoothness must not broaden the water.
            width: cell * (8_f32 / 3.).sqrt(),
        };
        // A lattice cell always addresses the same 27 nodes. Cache their
        // dense indices/centres once rather than hashing every deposit. Values
        // still accumulate in the original particle and stencil order, with
        // identical arithmetic; no coalescing, resampling or worker is added.
        let mut indices = HashMap::<[i32; 3], usize>::new();
        let mut values = Vec::<Moments>::new();
        let mut blocks = HashMap::<[i32; 3], [(usize, Vec3); 27]>::new();
        for &p in points {
            let weights = stencil(p, out.width);
            let entries = blocks.entry(weights[0].0).or_insert_with(|| {
                weights.map(|(k, _)| {
                    let index = *indices.entry(k).or_insert_with(|| {
                        let index = values.len();
                        values.push(Moments::default());
                        index
                    });
                    (index, out.center(k))
                })
            });
            for ((_, weight), &(index, center)) in weights.into_iter().zip(entries.iter()) {
                if weight <= 0. {
                    continue;
                }
                let q = p - center;
                values[index].accumulate(
                    Moments {
                        count: 1.,
                        sum: q,
                        square: moment_outer(q),
                    },
                    weight,
                );
            }
        }
        // A zero-weight boundary node was absent in the scalar deposition.
        // Omit unoccupied entries so later neighbourhood sums remain identical,
        // including their addition order and exact-zero behaviour.
        out.nodes = indices
            .into_iter()
            .filter_map(|(k, index)| (values[index].count > 0.).then_some((k, values[index])))
            .collect();
        out
    }
    fn sample(&mut self, p: Vec3, cell: f32, volume: f64) -> Kernel {
        #[cfg(test)]
        let mut point_timer = profile::begin_point();
        let (base, weights) = lattice_weights(p, self.width);
        let base = base.to_array();
        let origin = self.center(base);
        let width = self.width;
        let center_of =
            |k: [i32; 3]| (Vec3::from_array(k.map(|i| i as f32)) + Vec3::splat(0.5)) * width;
        let nodes = &self.nodes;
        let neighbourhoods = &mut self.neighbourhoods;
        // The common query path performs one lookup and borrows its block.
        let corners = self.stencils.entry(base).or_insert_with(|| {
            std::array::from_fn(|n| {
                let k = [
                    base[0] + (n % 3) as i32,
                    base[1] + ((n / 3) % 3) as i32,
                    base[2] + (n / 9) as i32,
                ];
                let center = center_of(k);
                let local = nodes.get(&k).copied().unwrap_or_default();
                let total = if let Some(m) = neighbourhoods.get(&k) {
                    *m
                } else {
                    let mut m = Moments::default();
                    for z in -1..=1 {
                        for y in -1..=1 {
                            for x in -1..=1 {
                                let key = [k[0] + x, k[1] + y, k[2] + z];
                                if let Some(n) = nodes.get(&key) {
                                    m.accumulate(n.translated(center_of(key) - center), 1.);
                                }
                            }
                        }
                    }
                    neighbourhoods.insert(k, m);
                    m
                };
                (
                    local.translated(center - origin),
                    total.translated(center - origin),
                )
            })
        });
        let mut local = Moments::default();
        let mut total = Moments::default();
        // A query needs weights, not another 27-key scatter stencil. Keep
        // the same x/y/z multiplication and accumulation order as deposition.
        for (n, (a, b)) in corners.iter().enumerate() {
            let weight = weights[0][n % 3] * weights[1][(n / 3) % 3] * weights[2][n / 9];
            local.accumulate(*a, weight);
            total.accumulate(*b, weight);
        }
        let local = local.translated(origin - p);
        let total = total.translated(origin - p);
        let mean = total.sum / total.count.max(1e-8);
        let covariance = Mat3::from(total.square) / total.count.max(1e-8) - outer(mean);
        #[cfg(test)]
        profile::point_stage("point_moment_interpolation", &mut point_timer);
        let (basis, variance) = eigen(covariance);
        #[cfg(test)]
        profile::point_stage("point_primary_eigen", &mut point_timer);
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
        #[cfg(test)]
        profile::point_stage("point_density_and_shape", &mut point_timer);
        if boundary > 0. {
            let n = normal.lerp(boundary_normal, boundary).normalize_or_zero();
            let target = boundary_covariance(covariance, n, cell * cell * 0.25);
            let (v, d) = eigen(covariance * (1. - boundary) + target * boundary);
            let positive = d.max(Vec3::splat(d.max_element().max(1e-10) * 0.16));
            let s = Vec3::new(positive.x.sqrt(), positive.y.sqrt(), positive.z.sqrt());
            let r = (s / s.element_product().cbrt().max(1e-10))
                .clamp(Vec3::splat(0.45), Vec3::splat(1.8));
            axes = [v.x_axis * r.x, v.y_axis * r.y, v.z_axis * r.z];
            normal = n;
            surface_planarity += (1. - surface_planarity) * boundary;
        }
        #[cfg(test)]
        profile::point_stage("point_boundary_refinement", &mut point_timer);
        Kernel {
            axes,
            center: p,
            channel_center: p
                + (local.sum / local.count.max(1e-8)).clamp_length_max(cell * 0.75)
                    * (0.8 * smooth((local_density - 0.05) / 0.45)),
            normal,
            planarity: surface_planarity,
            bulk,
            neighbour_count: total.count,
        }
    }
}
/// Fixed small symmetric Jacobi solve. Eigenvectors remain orthonormal even for
/// nearly isotropic water; no velocity-directed stretching or solver changes.
#[inline(always)]
fn jacobi_rotation<const P: usize, const Q: usize>(a: &mut [[f32; 3]; 3], v: &mut [[f32; 3]; 3]) {
    let apq = a[Q][P];
    if apq.abs() < 1e-12 {
        return;
    }
    let tau = (a[Q][Q] - a[P][P]) / (2. * apq);
    let t = if tau >= 0. {
        1. / (tau + (1. + tau * tau).sqrt())
    } else {
        -1. / (-tau + (1. + tau * tau).sqrt())
    };
    let c = 1. / (1. + t * t).sqrt();
    let s = t * c;
    for row in a.iter_mut() {
        let ap = row[P];
        let aq = row[Q];
        row[P] = c * ap - s * aq;
        row[Q] = s * ap + c * aq;
    }
    for row in v.iter_mut() {
        let ap = row[P];
        let aq = row[Q];
        row[P] = c * ap - s * aq;
        row[Q] = s * ap + c * aq;
    }
    let (before, after) = a.split_at_mut(Q);
    for (ap, aq) in before[P].iter_mut().zip(&mut after[0]) {
        let old_p = *ap;
        let old_q = *aq;
        *ap = c * old_p - s * old_q;
        *aq = s * old_p + c * old_q;
    }
}
fn eigen(m: Mat3) -> (Mat3, Vec3) {
    let mut a = m.to_cols_array_2d();
    let mut v = Mat3::IDENTITY.to_cols_array_2d();
    for _ in 0..5 {
        // The same ordered rotations and floating-point expressions as the
        // reference. Fixed planes let LLVM remove dynamic indexing/branches;
        // no sweep, tolerance, tensor or water model is approximated.
        jacobi_rotation::<0, 1>(&mut a, &mut v);
        jacobi_rotation::<0, 2>(&mut a, &mut v);
        jacobi_rotation::<1, 2>(&mut a, &mut v);
    }
    (
        Mat3::from_cols_array_2d(&v).transpose(),
        Vec3::new(a[0][0], a[1][1], a[2][2]),
    )
}
#[cfg(test)]
fn eigen_reference(m: Mat3) -> (Mat3, Vec3) {
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
    fn simd_moment_translation_and_accumulation_preserve_scalar_reference_bits() {
        let mut state = 0x38c51f29u32;
        let mut random = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / 8388608. - 1.
        };
        for _ in 0..2048 {
            let p = Vec3::new(random(), random(), random());
            let q = Vec3::new(random(), random(), random());
            let offset = Vec3::new(random(), random(), random()) * 0.03;
            let count = random().abs() * 20.;
            let weight = random().abs();
            let square = outer(q);
            let m = Moments {
                count,
                sum: p,
                square: Mat3A::from(square),
            };
            let actual = m.translated(offset);
            let scalar = square
                + outer(offset) * count
                + Mat3::from_cols(p * offset.x, p * offset.y, p * offset.z)
                + Mat3::from_cols(offset * p.x, offset * p.y, offset * p.z);
            assert_eq!(
                actual.square.to_cols_array().map(f32::to_bits),
                scalar.to_cols_array().map(f32::to_bits)
            );
            let mut summed = Moments {
                count: 1.,
                sum: q,
                square: Mat3A::from(outer(p)),
            };
            summed.accumulate(actual, weight);
            assert_eq!(summed.count.to_bits(), (1. + count * weight).to_bits());
            assert_eq!(
                summed.sum.to_array().map(f32::to_bits),
                (q + (p + offset * count) * weight)
                    .to_array()
                    .map(f32::to_bits)
            );
            assert_eq!(
                summed.square.to_cols_array().map(f32::to_bits),
                (outer(p) + scalar * weight)
                    .to_cols_array()
                    .map(f32::to_bits)
            );
        }
    }

    #[test]
    fn specialized_jacobi_preserves_reference_bits_across_scale_and_degeneracy() {
        let mut state = 0x7319b42du32;
        let mut random = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / 8388608. - 1.
        };
        for scale in [0., 1e-16, 1e-10, 1e-4, 1., 1e6] {
            for i in 0..512 {
                let x = Vec3::new(random(), random(), random());
                let y = Vec3::new(random(), random(), random());
                let mut m = (outer(x) + outer(y)) * scale;
                if i % 4 == 0 {
                    m = Mat3::from_diagonal(Vec3::splat(scale));
                }
                if i % 4 == 1 {
                    m.z_axis.x += scale * 1e-7;
                }
                if i % 4 == 2 {
                    m.x_axis.x -= scale;
                }
                for input in [m, m.transpose()] {
                    let (v, d) = eigen(input);
                    let (rv, rd) = eigen_reference(input);
                    assert_eq!(
                        v.to_cols_array().map(f32::to_bits),
                        rv.to_cols_array().map(f32::to_bits),
                        "basis scale={scale} case={i}"
                    );
                    assert_eq!(
                        d.to_array().map(f32::to_bits),
                        rd.to_array().map(f32::to_bits),
                        "variance scale={scale} case={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn cached_deposition_retains_scalar_moments_and_kernel_bytes_at_cell_edges() {
        let cell = 0.02;
        let width = cell * (8_f32 / 3.).sqrt();
        let points: Vec<_> = (-3..=3)
            .flat_map(|x| {
                (-2..=2).flat_map(move |z| {
                    [0., 0.125, 0.5, 0.875, 1.].map(move |phase| {
                        Vec3::new(
                            (x as f32 + phase) * width,
                            (0.1 * (x * z) as f32 + phase) * width,
                            (z as f32 + phase) * width,
                        )
                    })
                })
            })
            .collect();
        let mut cached = ChannelField::new(&points, cell);
        let mut scalar = ChannelField {
            nodes: HashMap::new(),
            neighbourhoods: HashMap::new(),
            stencils: HashMap::new(),
            width,
        };
        // Independent original scalar scatter, with no address/block cache.
        for &p in &points {
            for (k, weight) in stencil(p, width) {
                if weight > 0. {
                    let q = p - scalar.center(k);
                    scalar.nodes.entry(k).or_default().accumulate(
                        Moments {
                            count: 1.,
                            sum: q,
                            square: moment_outer(q),
                        },
                        weight,
                    );
                }
            }
        }
        let bits = |m: Moments| {
            std::iter::once(m.count)
                .chain(m.sum.to_array())
                .chain(m.square.to_cols_array())
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(cached.nodes.len(), scalar.nodes.len());
        for (key, &reference) in &scalar.nodes {
            assert_eq!(bits(cached.nodes[key]), bits(reference));
        }
        let kernel_bits = |k: Kernel| {
            k.axes
                .into_iter()
                .flat_map(|v| v.to_array())
                .chain(k.center.to_array())
                .chain(k.channel_center.to_array())
                .chain(k.normal.to_array())
                .chain([k.planarity, k.bulk])
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        for &p in &points {
            assert_eq!(
                kernel_bits(cached.sample(p, cell, 1e-7)),
                kernel_bits(scalar.sample(p, cell, 1e-7))
            );
        }
    }
    #[test]
    fn shared_outlet_bin_retains_its_first_channel_representative() {
        let cell = 0.02;
        let points = [Vec3::splat(0.005), Vec3::splat(0.015)];
        let output = super::kernels(&points, cell, 1e-8, &[true, false]);
        // Sparse legacy support is seeded by the first particle in the bin,
        // even when that particle itself uses the continuous channel field.
        let expected = points[1] + (points[0] - points[1]).clamp_length_max(cell * 0.2) * 0.35;
        assert_eq!(
            output[1].center.to_array().map(f32::to_bits),
            expected.to_array().map(f32::to_bits)
        );
        assert_eq!(output[1].axes, [Vec3::X, Vec3::Y, Vec3::Z]);
    }
    #[test]
    fn channel_stencil_retains_material_moments_with_phase_independent_support() {
        let width = 0.02;
        let mut worst = 0_f32;
        for phase in 0..101 {
            let p = Vec3::new(-0.031 + phase as f32 * 0.0002, 0.017, -0.023);
            let mut mass = 0.;
            let mut center = Vec3::ZERO;
            let mut covariance = Mat3::ZERO;
            for (k, w) in stencil(p, width) {
                assert!(w >= 0. && w.is_finite());
                let point = (Vec3::from_array(k.map(|v| v as f32)) + Vec3::splat(0.5)) * width;
                mass += w;
                center += point * w;
                covariance += outer(point - p) * w;
            }
            assert!((mass - 1.).abs() < 3e-7);
            assert!((center - p).length() < 1e-8);
            let error = (covariance - Mat3::IDENTITY * (width * width * 0.25))
                .to_cols_array()
                .into_iter()
                .map(f32::abs)
                .fold(0_f32, f32::max);
            worst = worst.max(error);
            assert!(
                error < 1e-10,
                "lattice phase changed the footprint: {error}"
            );
        }
        // Independent former tent variance: a node has no support variance;
        // a midpoint has h²/4. This phase variation changed parcel shape and
        // therefore the density/extinction even before the temporal bridge.
        let old_node_variance = 0.;
        let old_midpoint_variance = width * width / 4.;
        assert!(old_midpoint_variance - old_node_variance > 0.00009);
        println!("channel stencil: old variance swing={old_midpoint_variance}m², final max error={worst}m²");
    }
    #[test]
    fn channel_stencil_has_continuous_slopes_across_all_cell_planes() {
        // A deliberately non-affine positive nodal field. First-moment checks
        // alone would miss a slope kink because both schemes reproduce planes.
        let at = |p: Vec3| -> f64 {
            stencil(p, 1.)
                .into_iter()
                .map(|(k, w)| {
                    let x = f64::from(k[0]);
                    let y = f64::from(k[1]);
                    let z = f64::from(k[2]);
                    f64::from(w) * (1. + x * x + 2. * y * y + 3. * z * z + x * y)
                })
                .sum()
        };
        let mut worst = 0_f64;
        for axis in 0..3 {
            for plane in [-2., -1., 0., 1., 2.] {
                let mut p = Vec3::new(0.17, -0.23, 0.31);
                p[axis] = plane;
                let step = Vec3::AXES[axis] * 0.001;
                let left = (at(p) - at(p - step)) / 0.001;
                let right = (at(p + step) - at(p)) / 0.001;
                let jump = (right - left).abs();
                worst = worst.max(jump);
                assert!(
                    jump < 0.015,
                    "channel moment slope kink: axis={axis}, plane={plane}, jump={jump}"
                );
            }
        }
        println!("channel stencil maximum one-sided slope difference={worst}");
    }
    #[test]
    fn boundary_covariance_retains_oblique_crest_cross_terms_under_rotation() {
        let measured = Mat3::from_diagonal(Vec3::new(0.0009, 0.0004, 0.0001));
        // Independent expected tangent variances, with only the normal support
        // bounded to 16 percent of the longest measured tangent variance.
        let expected = Mat3::from_diagonal(Vec3::new(0.0009, 0.000144, 0.0001));
        for angle in [0., 0.23, std::f32::consts::FRAC_PI_4, 1.1] {
            for tilt in [0., 0.6, std::f32::consts::FRAC_PI_2] {
                let rotation = Mat3::from_rotation_z(tilt) * Mat3::from_rotation_y(angle);
                let covariance = rotation * measured * rotation.transpose();
                let target = boundary_covariance(covariance, rotation * Vec3::Y, 0.000025);
                let reference = rotation * expected * rotation.transpose();
                let error = (target - reference)
                    .to_cols_array()
                    .into_iter()
                    .map(f32::abs)
                    .fold(0_f32, f32::max);
                assert!(
                    error < 1e-8,
                    "boundary ellipse depends on orientation: {angle}, {tilt}, {error}"
                );
            }
        }
        let rotation = Mat3::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let covariance = rotation * measured * rotation.transpose();
        // Actual former diagonal-only construction: at 45 degrees it loses a
        // 0.0004 m² tangent covariance and gives this 3:1 ridge a round ellipse.
        let old = outer(Vec3::X) * covariance.x_axis.x
            + outer(Vec3::Z) * covariance.z_axis.z
            + outer(Vec3::Y) * (covariance.x_axis.x.max(covariance.z_axis.z) * 0.16);
        let actual = boundary_covariance(covariance, Vec3::Y, 0.000025);
        assert!((actual.x_axis.z - covariance.x_axis.z).abs() < 1e-8);
        assert!((old.x_axis.z - covariance.x_axis.z).abs() > 0.00039);
        assert!((covariance.x_axis.x - covariance.z_axis.z).abs() < 1e-8);
        assert!(actual.determinant() > 0. && actual.is_finite());
    }
    #[test]
    fn final_channel_support_thins_normal_through_degenerate_eigenframes() {
        let support = |axes: [Vec3; 3]| axes.into_iter().map(outer).fold(Mat3::ZERO, |a, b| a + b);
        let expected = Mat3::from_diagonal(Vec3::new(3.24, 0.1296, 0.5184));
        let volume = 1.8 * 0.72 * 0.72;
        for angle in [0., 0.23, std::f32::consts::FRAC_PI_4, 1.1] {
            for tilt in [0., 0.6, std::f32::consts::FRAC_PI_2] {
                let rotation = Mat3::from_rotation_z(tilt) * Mat3::from_rotation_y(angle);
                for frame_angle in [0., 0.2, std::f32::consts::FRAC_PI_4, 0.9] {
                    // These are different eigenframes of exactly the same
                    // support: normal Y and tangent Z share a floored length.
                    let (sin, cos) = frame_angle.sin_cos();
                    let axes = [
                        rotation * Vec3::X * 1.8,
                        rotation * (Vec3::Y * cos + Vec3::Z * sin) * 0.72,
                        rotation * (-Vec3::Y * sin + Vec3::Z * cos) * 0.72,
                    ];
                    let normal = rotation * Vec3::Y;
                    let kernel = Kernel {
                        axes,
                        center: Vec3::ZERO,
                        channel_center: Vec3::ZERO,
                        normal,
                        planarity: 1.,
                        bulk: 1.,
                        neighbour_count: 12.,
                    };
                    let final_axes = kernel.channel_axes();
                    let actual = support(final_axes);
                    let reference = rotation * expected * rotation.transpose();
                    let error = (actual - reference)
                        .to_cols_array()
                        .into_iter()
                        .map(f32::abs)
                        .fold(0_f32, f32::max);
                    assert!(
                        error < 2e-5,
                        "final support depends on eigenframe: angle={angle}, tilt={tilt}, frame={frame_angle}, error={error}"
                    );
                    assert!((normal.dot(actual * normal) - 0.1296).abs() < 2e-5);
                    let represented = final_axes.iter().map(|a| a.length()).product::<f32>();
                    assert!((represented - volume * 0.5).abs() < 2e-5);
                    assert!(final_axes
                        .iter()
                        .all(|a| a.is_finite() && a.length() <= 1.80001));
                    assert_eq!(Kernel { bulk: 0., ..kernel }.channel_axes(), axes);
                    assert_eq!(
                        Kernel {
                            planarity: 0.,
                            ..kernel
                        }
                        .channel_axes(),
                        axes
                    );
                }
            }
        }
        // Actual previous shortest-axis rule on a 45-degree degenerate basis
        // compresses an arbitrary Y/Z mixture rather than the measured Y.
        let (sin, cos) = std::f32::consts::FRAC_PI_4.sin_cos();
        let mut old = [
            Vec3::X * 1.8,
            (Vec3::Y * cos + Vec3::Z * sin) * 0.72,
            (-Vec3::Y * sin + Vec3::Z * cos) * 0.72,
        ];
        let shortest = (0..3)
            .min_by(|&a, &b| old[a].length_squared().total_cmp(&old[b].length_squared()))
            .unwrap();
        old[shortest] *= 0.5;
        assert!(
            (Vec3::Y.dot(support(old) * Vec3::Y) - 0.1296).abs() > 0.19,
            "old rule did not reproduce wrong-direction crest thinning"
        );
    }
    #[test]
    fn boundary_tangent_floor_stays_positive_and_continuous() {
        let floor = 0.000025;
        let at = |variance| {
            boundary_covariance(
                Mat3::from_diagonal(Vec3::new(0.0005, 0.0001, variance)),
                Vec3::Y,
                floor,
            )
        };
        for variance in [0., floor * 0.5, floor * 0.99999, floor * 1.00001] {
            let target = at(variance);
            assert!(target.is_finite() && target.determinant() > 0.);
            assert!((target.x_axis.x - 0.0005).abs() < 1e-9);
            assert!(target.z_axis.z >= floor - 1e-9);
            assert!(target.y_axis.y > 0. && target.y_axis.y <= target.x_axis.x * 0.16 + 1e-9);
        }
        let jump = (at(floor + 1e-9) - at(floor - 1e-9))
            .to_cols_array()
            .into_iter()
            .map(f32::abs)
            .fold(0_f32, f32::max);
        assert!(
            jump < 3e-9,
            "tangent support jumps at its variance floor: {jump}"
        );
    }
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
        // The volume-normalized poly6, rather than expanded tangents, owns
        // represented water volume. Thinning cannot broaden the wave shoulder.
        assert!(volume(axes) < volume(k.axes));
        for (old, new) in k.axes.iter().zip(axes) {
            assert!(new.length() <= old.length() + 1e-6);
        }
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
        let data = kernels(&sheet, 0.01, 1.25e-7);
        assert!(
            data.iter().all(|k| k.bulk == 1.),
            "dense sheet minimum support {}",
            data.iter().map(|k| k.bulk).fold(1_f32, f32::min)
        );
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

#[cfg(test)]
#[path = "fluid_surface_profile.rs"]
pub(super) mod profile;
