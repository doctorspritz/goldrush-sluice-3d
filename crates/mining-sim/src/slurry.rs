//! Read-only reconstruction of the weighted slurry, not another physics solver.
//! Conservatively project water volume, suspended fine volume and grain cross
//! section to a small 3D field. Optics never count display points as material.
use crate::{GrainKind, Sluice, Vec3};

pub const CELL_M: f32 = 0.03;
#[derive(Clone, Copy, Default)]
struct Cell {
    water: f64,
    fines: f64,
    area: f64,
}
impl Cell {
    fn add(&mut self, other: Self, weight: f64) {
        self.water += other.water * weight;
        self.fines += other.fines * weight;
        self.area += other.area * weight;
    }
}
pub struct SlurryField {
    dims: [usize; 3],
    cells: Vec<Cell>,
}
impl SlurryField {
    /// Legacy field retained for the accepted outlet's existing optical sample.
    /// Moving sand alone does not establish contact with the liquid carrier.
    pub fn new(s: &Sluice) -> Self {
        Self::build(s, None)
    }
    /// Channel optics include only moving sand in the supplied current water
    /// neighbourhood. The physics adapter supplies the same coarse presence
    /// criterion used for drag; this is not an exact submergence calculation.
    /// The mask follows the complete, unfiltered grain array in snapshot order.
    pub fn new_with_suspension(s: &Sluice, wet: &[bool]) -> Self {
        assert_eq!(wet.len(), s.grains.len(), "complete grain wetness mask");
        Self::build(s, Some(wet))
    }
    fn build(s: &Sluice, wet: Option<&[bool]>) -> Self {
        let extent = Vec3::new(
            s.profile.length(),
            (s.floor_cells() + 1) as f32 * s.profile.cell() + 0.20,
            s.profile.width() + 2. * s.profile.cell(),
        );
        let dims = extent.to_array().map(|v| (v / CELL_M).ceil() as usize + 1);
        let mut field = Self {
            dims,
            cells: vec![Cell::default(); dims.iter().product()],
        };
        for p in &s.water {
            // Project both phases with the same conservative stencil. A
            // histogram for water and linear weights for fines shift their
            // ratio as a uniformly mixed suspension crosses an optical cell.
            field.deposit(
                p.position,
                Cell {
                    water: s.water_particle_m3(),
                    ..Default::default()
                },
            );
        }
        for (index, g) in s.grains.iter().enumerate() {
            if g.kind != GrainKind::Sand || g.resting || wet.is_some_and(|mask| !mask[index]) {
                continue;
            }
            let volume = g.solid_volume();
            // N*pi*d²/4 = 3*solid_volume/(2*d). Unresolved optical cross section
            // preserves the represented population even when the display is capped.
            field.deposit(
                g.position,
                Cell {
                    fines: volume,
                    area: 1.5 * volume / g.diameter_m as f64,
                    ..Default::default()
                },
            );
        }
        // Separable binomial reconstruction of optical moments. Reconstruct
        // the horizontal footprint twice: weighted fine parcels are samples of
        // a suspension, rather than opaque centimetre-sized dirt clods. Keep
        // the vertical footprint narrow so bed/bulk stratification survives.
        // This is a presentation filter, not physical sediment diffusion.
        // Clamped destinations retain every projected water/fine/area integral.
        for axis in [0, 1, 2, 0, 2] {
            field.filter_axis(axis);
        }
        field
    }
    /// Weighted fine parcels undersample the visible suspension. Reconstruct
    /// their optical moments farther along the sluice's downstream coordinate,
    /// rather than drawing each parcel as a round mud patch. This positive,
    /// conservative presentation filter changes neither particle motion nor any
    /// material ledger. Keep the established outlet sample before applying it.
    pub fn reconstruct_channel(&mut self) {
        // Including the two projection passes, this gives an approximately
        // 12 cm streamwise standard deviation. A 6 cm cross-channel footprint
        // joins undersampled fine parcels into suspension instead of separate
        // round brown islands. Both remain smaller than the channel width.
        // The weighted fine parcels undersample a continuous suspension: their
        // optical footprint should form a streak, not a round dirt clod. All
        // three additive moments use the same positive conservative stencil.
        // Vertical stratification and the separately sampled outlet stay intact.
        for _ in 0..30 {
            self.filter_axis(0);
        }
        for _ in 0..6 {
            self.filter_axis(2);
        }
    }
    fn filter_axis(&mut self, axis: usize) {
        let stride = [1, self.dims[0], self.dims[0] * self.dims[1]][axis];
        let length = self.dims[axis];
        let block = stride * length;
        let mut next = vec![Cell::default(); self.cells.len()];
        // Gather in the original scatter's contribution order: lower cell,
        // centre, upper cell, including duplicated clamped boundary taps.
        // Each destination is written once rather than updated three times.
        // Contiguous blocks also avoid rebuilding XYZ indices for every tap.
        for (source, target) in self
            .cells
            .chunks_exact(block)
            .zip(next.chunks_exact_mut(block))
        {
            for coordinate in 0..length {
                let lower = coordinate.saturating_sub(1) * stride;
                let center = coordinate * stride;
                let upper = (coordinate + 1).min(length - 1) * stride;
                for lane in 0..stride {
                    let value = &mut target[center + lane];
                    value.add(source[lower + lane], 0.25);
                    value.add(source[center + lane], 0.5);
                    value.add(source[upper + lane], 0.25);
                }
            }
        }
        self.cells = next;
    }
    fn index(&self, p: [usize; 3]) -> usize {
        (p[2] * self.dims[1] + p[1]) * self.dims[0] + p[0]
    }
    fn stencil(&self, p: Vec3) -> [(usize, f64); 8] {
        let q = p / CELL_M - Vec3::splat(0.5);
        let base = q.floor();
        let f = q - base;
        std::array::from_fn(|n| {
            let delta = [n & 1, (n >> 1) & 1, (n >> 2) & 1];
            let b = base.to_array();
            let f = f.to_array();
            let cell = std::array::from_fn(|a| {
                (b[a] as i32 + delta[a] as i32).clamp(0, self.dims[a] as i32 - 1) as usize
            });
            let weight = (0..3)
                .map(|a| {
                    if delta[a] == 0 {
                        1. - f[a] as f64
                    } else {
                        f[a] as f64
                    }
                })
                .product();
            (self.index(cell), weight)
        })
    }
    fn deposit(&mut self, p: Vec3, value: Cell) {
        for (i, w) in self.stencil(p) {
            self.cells[i].add(value, w);
        }
    }
    /// (solid volume / water volume, optical extinction in inverse metres).
    /// Q_ext=2 is an optical approximation for grains much larger than light
    /// wavelengths, not an NTU calibration or a dense-slurry constitutive law.
    pub fn sample(&self, p: Vec3) -> (f32, f32) {
        self.sample_reconstructed(p, false)
    }
    pub fn sample_channel(&self, p: Vec3) -> (f32, f32) {
        self.sample_reconstructed(p, true)
    }
    fn sample_reconstructed(&self, p: Vec3, channel: bool) -> (f32, f32) {
        let mut cell = Cell::default();
        if !channel {
            for (i, w) in self.stencil(p) {
                cell.add(self.cells[i], w);
            }
        } else {
            // Reconstruct the additive moments with one C2 B-spline footprint.
            // Linear interpolation leaves a kink at each optical-cell plane; after
            // Beer-Lambert shading those kinks read as separate muddy patches.
            // Divide only after gathering, so a homogeneous mixture stays uniform.
            let q = p / CELL_M - Vec3::splat(0.5);
            let base = q.floor().as_ivec3();
            let f = (q - base.as_vec3()).to_array();
            let weights = f.map(cubic_weights);
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let offsets = [x, y, z];
                        let index = std::array::from_fn(|a| {
                            (base[a] + offsets[a] as i32 - 1).clamp(0, self.dims[a] as i32 - 1)
                                as usize
                        });
                        let w = weights[0][x] * weights[1][y] * weights[2][z];
                        cell.add(self.cells[self.index(index)], w);
                    }
                }
            }
        }
        if cell.water < 1e-12 {
            return (0., 0.);
        }
        (
            (cell.fines / cell.water).max(0.) as f32,
            (2. * cell.area / cell.water).clamp(0., 100_000.) as f32,
        )
    }
    pub fn projected_volumes(&self) -> (f64, f64) {
        self.cells
            .iter()
            .fold((0., 0.), |(w, f), c| (w + c.water, f + c.fines))
    }
    /// Current outlet concentration, used only for the existing airborne-water
    /// optical presentation. It is not a transported solid-mass ledger.
    pub fn outlet_extinction(&self, s: &Sluice) -> f32 {
        let mut beta = 0.;
        let mut count = 0;
        for p in &s.water {
            if p.position.x > s.profile.length() - 0.09 {
                beta += self.sample(p.position).1;
                count += 1;
            }
        }
        beta / count.max(1) as f32
    }
}

// Positive cardinal cubic weights sum to one, including clamped boundaries.
fn cubic_weights(t: f32) -> [f64; 4] {
    let t = t as f64;
    [
        (1. - t).powi(3) / 6.,
        (3. * t.powi(3) - 6. * t * t + 4.) / 6.,
        (-3. * t.powi(3) + 3. * t * t + 3. * t + 1.) / 6.,
        t.powi(3) / 6.,
    ]
}

pub fn diagnostics(s: &Sluice) -> serde_json::Value {
    let field = SlurryField::new(s);
    let (water, fines) = field.projected_volumes();
    let fine_count = s
        .grains
        .iter()
        .filter(|g| g.kind == GrainKind::Sand && g.diameter_m < 0.0002)
        .count();
    serde_json::json!({"feed_grading":s.feed_grading,"solver_parcels":s.grains.len(),
        "represented_grains":s.grains.iter().map(|g|g.represented_grains()).sum::<f64>(),
        "fine_parcels_below_200um":fine_count,
        "projected_water_m3":water,"projected_suspended_fines_m3":fines,
        "projection_water_error_m3":water-s.water.len() as f64*s.water_particle_m3(),
        "projection_fines_error_m3":fines-s.grains.iter().filter(|g|g.kind==GrainKind::Sand&&!g.resting).map(|g|g.solid_volume()).sum::<f64>(),
        "outlet_extinction_m1":field.outlet_extinction(s),"optical_grid_m":CELL_M,
        "optics_calibrated":false,"independently_resolved_grain_collisions":false})
}

#[cfg(test)]
mod reconstruction_tests {
    use super::*;

    #[test]
    fn gathered_filter_matches_ordered_scatter_at_every_axis_and_boundary() {
        // Keep the original destination scatter as an independent reference.
        // Singleton axes exercise duplicated boundary taps; the patterned
        // moments span many magnitudes so a changed addition order is visible.
        for dims in [[1, 1, 1], [2, 3, 5], [9, 5, 9], [101, 1, 1], [4, 1, 7]] {
            let mut field = SlurryField {
                dims,
                cells: (0..dims.iter().product())
                    .map(|i| {
                        let v = ((i * 73 + 17) % 113) as f64 * 10f64.powi(i as i32 % 16 - 12);
                        Cell {
                            water: v,
                            fines: v * 0.0173,
                            area: v * 372.81,
                        }
                    })
                    .collect(),
            };
            let mut reference = field.cells.clone();
            for axis in [0, 1, 2, 0, 2]
                .into_iter()
                .chain(std::iter::repeat_n(0, 30))
                .chain(std::iter::repeat_n(2, 6))
            {
                let mut next = vec![Cell::default(); reference.len()];
                for z in 0..dims[2] {
                    for y in 0..dims[1] {
                        for x in 0..dims[0] {
                            let p = [x, y, z];
                            let value = reference[field.index(p)];
                            for (offset, weight) in [(-1, 0.25), (0, 0.5), (1, 0.25)] {
                                let mut q = p;
                                q[axis] = (p[axis] as i32 + offset).clamp(0, dims[axis] as i32 - 1)
                                    as usize;
                                next[field.index(q)].add(value, weight);
                            }
                        }
                    }
                }
                reference = next;
                field.filter_axis(axis);
                for (index, (a, b)) in field.cells.iter().zip(&reference).enumerate() {
                    assert_eq!(
                        [a.water.to_bits(), a.fines.to_bits(), a.area.to_bits()],
                        [b.water.to_bits(), b.fines.to_bits(), b.area.to_bits()],
                        "dims={dims:?} axis={axis} cell={index}"
                    );
                }
            }
        }
    }

    #[test]
    fn fine_parcel_footprint_is_streamwise_and_keeps_a_broad_plume() {
        // Interior impulse measures the real footprint independently of colour.
        let mut field = SlurryField {
            dims: [101, 1, 1],
            cells: vec![Cell::default(); 101],
        };
        field.cells[50] = Cell {
            water: 1.,
            fines: 0.02,
            area: 300.,
        };
        field.reconstruct_channel();
        let total: f64 = field.cells.iter().map(|c| c.water).sum();
        let variance: f64 = field
            .cells
            .iter()
            .enumerate()
            .map(|(x, c)| c.water * ((x as f64 - 50.) * CELL_M as f64).powi(2))
            .sum::<f64>()
            / total;
        // Each conservative [1/4,1/2,1/4] pass has variance dx^2/2.
        assert!((variance - 30. * (CELL_M as f64).powi(2) / 2.).abs() < 1e-12);
        assert!((total - 1.).abs() < 1e-13);
        for cell in &field.cells {
            assert!(cell.water >= 0. && cell.fines >= 0. && cell.area >= 0.);
            if cell.water > 0. {
                assert!((cell.fines / cell.water - 0.02).abs() < 1e-12);
                assert!((cell.area / cell.water - 300.).abs() < 1e-9);
            }
        }
        // A lengthwise concentration gradient must survive reconstruction:
        // removing parcel noise must not turn real plumes into uniform colour.
        for (x, cell) in field.cells.iter_mut().enumerate() {
            *cell = Cell {
                water: 1.,
                fines: x as f64 * 0.001,
                area: x as f64,
            };
        }
        field.reconstruct_channel();
        for x in 32..69 {
            assert!((field.cells[x].fines / field.cells[x].water - x as f64 * 0.001).abs() < 1e-12);
        }
    }

    #[test]
    fn channel_filter_preserves_positive_moments_and_vertical_stratification() {
        for x in [0, 4, 8] {
            let mut field = SlurryField {
                dims: [9, 5, 9],
                cells: vec![Cell::default(); 405],
            };
            let index = field.index([x, 2, 4]);
            field.cells[index] = Cell {
                water: 0.001,
                fines: 0.00002,
                area: 0.3,
            };
            field.reconstruct_channel();
            let mut total = Cell::default();
            let mut nonzero = 0;
            for z in 0..9 {
                for y in 0..5 {
                    for q in 0..9 {
                        let cell = field.cells[field.index([q, y, z])];
                        assert!(cell.water >= 0. && cell.fines >= 0. && cell.area >= 0.);
                        if cell.water > 0. {
                            nonzero += 1;
                            assert_eq!(y, 2, "bed/bulk stratification was changed");
                            assert!((cell.fines / cell.water - 0.02).abs() < 1e-14);
                            assert!((cell.area / cell.water - 300.).abs() < 1e-10);
                        }
                        total.add(cell, 1.);
                    }
                }
            }
            assert!(nonzero >= 3, "parcel remains a point-like optical patch");
            assert!((total.water - 0.001).abs() < 1e-17);
            assert!((total.fines - 0.00002).abs() < 1e-18);
            assert!((total.area - 0.3).abs() < 1e-14);
        }
    }

    #[test]
    fn cross_channel_fine_footprint_keeps_a_real_concentration_gradient() {
        let mut field = SlurryField {
            dims: [1, 1, 101],
            cells: vec![Cell::default(); 101],
        };
        field.cells[50] = Cell {
            water: 1.,
            fines: 0.02,
            area: 300.,
        };
        field.reconstruct_channel();
        let total: f64 = field.cells.iter().map(|c| c.water).sum();
        let variance: f64 = field
            .cells
            .iter()
            .enumerate()
            .map(|(z, c)| c.water * ((z as f64 - 50.) * CELL_M as f64).powi(2))
            .sum::<f64>()
            / total;
        assert!((total - 1.).abs() < 1e-13);
        assert!((variance - 6. * (CELL_M as f64).powi(2) / 2.).abs() < 1e-12);
        for (z, cell) in field.cells.iter_mut().enumerate() {
            *cell = Cell {
                water: 1.,
                fines: z as f64 * 0.001,
                area: z as f64,
            };
        }
        field.reconstruct_channel();
        for z in 12..89 {
            assert!((field.cells[z].fines / field.cells[z].water - z as f64 * 0.001).abs() < 1e-12);
            assert!((field.cells[z].area / field.cells[z].water - z as f64).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod continuous_optics_tests {
    use super::*;
    #[test]
    fn nonuniform_optics_have_continuous_slopes_across_every_cell_axis() {
        let mut f = SlurryField {
            dims: [9; 3],
            cells: vec![
                Cell {
                    water: 1.,
                    fines: 0.,
                    area: 0.
                };
                729
            ],
        };
        let i = f.index([4; 3]);
        f.cells[i].area = 1.;
        let before = f.projected_volumes();
        for axis in 0..3 {
            for knot in [3, 4, 5] {
                let mut p = Vec3::splat(4.5 * CELL_M);
                p[axis] = (knot as f32 + 0.5) * CELL_M;
                let mut d = Vec3::ZERO;
                d[axis] = CELL_M * 0.002;
                let value = |q| f.sample_channel(q).1 as f64;
                let (a, b, c) = (value(p - d), value(p), value(p + d));
                let left = (b - a) / d[axis] as f64;
                let right = (c - b) / d[axis] as f64;
                assert!(
                    (left - right).abs() < 0.13,
                    "axis {axis}, knot {knot}: {left} vs {right}"
                );
                assert!(a >= 0. && b >= 0. && c >= 0.);
            }
        }
        assert_eq!(
            before,
            f.projected_volumes(),
            "sampling changed represented inventory"
        );
        for t in [0., 0.1, 0.5, 0.9, 1.] {
            let w = cubic_weights(t);
            assert!(w.iter().all(|v| *v >= 0.));
            assert!((w.iter().sum::<f64>() - 1.).abs() < 1e-14);
        }
    }
}
