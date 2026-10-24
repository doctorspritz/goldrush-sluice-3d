//! Scratch broad-phase masks for small grains. Original coarse membership
//! and the caller's increasing pair cursor remain authoritative.
use crate::clump::{Clump3D, ClumpTemplate3D};
use glam::Vec3;

const HALO: i32 = 2;
const MAX_MASK_WORDS: usize = 1_048_576;
#[derive(Default)]
pub(crate) struct FineContactGrid {
    enabled: bool,
    width: f32,
    max_radius: f32,
    radii: Vec<f32>,
    small: Vec<u64>,
    cells: Vec<[i32; 3]>,
    min: [i32; 3],
    dims: [usize; 3],
    offsets: [usize; 3],
    members: Vec<u64>,
    query_cell: Option<[i32; 3]>,
    query_starts: [usize; 3],
}
impl FineContactGrid {
    pub(crate) fn rebuild(
        &mut self,
        bodies: &[Clump3D],
        templates: &[ClumpTemplate3D],
        diameter: f32,
    ) {
        self.enabled = false;
        self.query_cell = None;
        self.max_radius = 0.;
        self.small.resize(bodies.len().div_ceil(64), 0);
        self.small.fill(0);
        self.radii.clear();
        self.radii.extend(
            bodies
                .iter()
                .map(|b| templates[b.template_idx].bounding_radius)
                .filter(|r| r.is_finite() && *r >= 0.),
        );
        if self.radii.is_empty() {
            return;
        }
        let middle = self.radii.len() / 2;
        let median = *self.radii.select_nth_unstable_by(middle, f32::total_cmp).1;
        let cutoff = (diameter * 0.125).min(median * 4.);
        let mut count = 0;
        for (i, body) in bodies.iter().enumerate() {
            let radius = templates[body.template_idx].bounding_radius;
            if radius >= 0. && radius <= cutoff {
                self.small[i / 64] |= 1u64 << (i % 64);
                count += 1;
                self.max_radius = self.max_radius.max(radius);
            }
        }
        self.width = self.max_radius * 2.;
        if count * 2 < bodies.len() || self.width <= 0. || !self.width.is_finite() {
            return;
        }
        self.cells.resize(bodies.len(), [0; 3]);
        let mut low = [i32::MAX; 3];
        let mut high = [i32::MIN; 3];
        for (i, body) in bodies.iter().enumerate() {
            if !self.is_small(i) {
                continue;
            }
            let Some(key) = self.key(body.position) else {
                return;
            };
            self.cells[i] = key;
            for a in 0..3 {
                low[a] = low[a].min(key[a]);
                high[a] = high[a].max(key[a]);
            }
        }
        // Leave room for positional corrections. Exceeding this scratch domain
        // falls back to the original candidates; it never constrains a body.
        let padding = (diameter * 2. / self.width).ceil();
        if !padding.is_finite() || !(0. ..=1_000_000.).contains(&padding) {
            return;
        }
        let padding = padding as i64 + HALO as i64 + 1;
        for a in 0..3 {
            let lo = low[a] as i64 - padding;
            let hi = high[a] as i64 + padding;
            if lo < i32::MIN as i64 || hi > i32::MAX as i64 {
                return;
            }
            self.min[a] = lo as i32;
            self.dims[a] = (hi - lo + 1) as usize;
        }
        self.offsets = [0, self.dims[0], self.dims[0] + self.dims[1]];
        let words = self
            .dims
            .iter()
            .sum::<usize>()
            .saturating_mul(self.small.len());
        if words > MAX_MASK_WORDS {
            return;
        }
        self.members.resize(words, 0);
        self.members.fill(0);
        for i in 0..bodies.len() {
            if self.is_small(i) {
                self.write_membership(i, self.cells[i], true);
            }
        }
        self.enabled = true;
    }
    #[inline]
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }
    #[inline]
    pub(crate) fn is_small(&self, index: usize) -> bool {
        self.small[index / 64] & (1u64 << (index % 64)) != 0
    }
    fn key(&self, position: Vec3) -> Option<[i32; 3]> {
        let q = (position / self.width).floor();
        if !q.is_finite() || q.min_element() < -2_000_000_000. || q.max_element() > 2_000_000_000. {
            return None;
        }
        Some(q.as_ivec3().to_array())
    }
    fn slot(&self, axis: usize, cell: i32) -> usize {
        (self.offsets[axis] + (cell as i64 - self.min[axis] as i64) as usize) * self.small.len()
    }
    fn inside(&self, key: [i32; 3]) -> bool {
        (0..3).all(|a| {
            let relative = key[a] as i64 - self.min[a] as i64;
            relative >= HALO as i64 && relative + (HALO as i64) < self.dims[a] as i64
        })
    }
    fn write_membership(&mut self, index: usize, key: [i32; 3], add: bool) {
        let bit = 1u64 << (index % 64);
        for (axis, cell) in key.into_iter().enumerate() {
            for delta in -HALO..=HALO {
                let slot = self.slot(axis, cell + delta) + index / 64;
                if add {
                    self.members[slot] |= bit;
                } else {
                    self.members[slot] &= !bit;
                }
            }
        }
    }
    #[inline]
    pub(crate) fn moved(&mut self, index: usize, position: Vec3) {
        if !self.enabled || !self.is_small(index) {
            return;
        }
        let Some(key) = self.key(position).filter(|&k| self.inside(k)) else {
            self.enabled = false;
            return;
        };
        let old = self.cells[index];
        if key == old {
            return;
        }
        self.write_membership(index, old, false);
        self.write_membership(index, key, true);
        self.cells[index] = key;
    }
    fn bounds(&self, position: Vec3, radius: f32) -> Option<([i32; 3], [i32; 3])> {
        let reach =
            (radius + self.max_radius) * (1. + 16. * f32::EPSILON) + f32::MIN_POSITIVE.sqrt();
        let lo = Vec3::from_array(
            (position - Vec3::splat(reach))
                .to_array()
                .map(f32::next_down),
        );
        let hi = Vec3::from_array((position + Vec3::splat(reach)).to_array().map(f32::next_up));
        Some((self.key(lo)?, self.key(hi)?))
    }
    fn covers(cell: [i32; 3], lo: [i32; 3], hi: [i32; 3]) -> bool {
        (0..3).all(|a| {
            lo[a] as i64 >= cell[a] as i64 - HALO as i64
                && hi[a] as i64 <= cell[a] as i64 + HALO as i64
        })
    }
    pub(crate) fn query(&mut self, position: Vec3, radius: f32) {
        self.query_cell = None;
        if !self.enabled {
            return;
        }
        let Some((lo, hi)) = self.bounds(position, radius) else {
            self.enabled = false;
            return;
        };
        let Some(cell) = self
            .key(position)
            .filter(|&k| self.inside(k) && Self::covers(k, lo, hi))
        else {
            self.enabled = false;
            return;
        };
        self.query_starts = std::array::from_fn(|a| self.slot(a, cell[a]));
        self.query_cell = Some(cell);
    }
    /// Only processed targets move within one source traversal. If the old
    /// mask still covers the new interval, no unprocessed target can be missed.
    pub(crate) fn refresh(&mut self, position: Vec3, radius: f32) -> bool {
        if !self.enabled {
            return true;
        }
        let Some((lo, hi)) = self.bounds(position, radius) else {
            self.enabled = false;
            return true;
        };
        if self
            .query_cell
            .is_some_and(|cell| Self::covers(cell, lo, hi))
        {
            return false;
        }
        self.query(position, radius);
        true
    }
    #[inline]
    pub(crate) fn candidates(&self, word: usize, original: u64) -> u64 {
        if !self.enabled {
            return original;
        }
        let [x, y, z] = self.query_starts;
        original
            & (!self.small[word]
                | (self.members[x + word] & self.members[y + word] & self.members[z + word]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clump::{ClumpShape3D, ClusterSimulation3D, IrregularStyle3D};
    fn fixture() -> ClusterSimulation3D {
        let mut sim = ClusterSimulation3D::new(Vec3::splat(-1.), Vec3::ONE);
        for radius in [0.001, 0.01] {
            sim.add_template(ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 1,
                    seed: 0,
                    style: IrregularStyle3D::Round,
                },
                radius,
                1.,
            ));
        }
        for i in 0..192 {
            sim.spawn(
                usize::from(i >= 160),
                Vec3::new(
                    (i * 37 % 17) as f32 * 0.0015 - 0.012,
                    (i * 13 % 7) as f32 * 0.001 - 0.003,
                    (i % 3) as f32 * 0.0015 - 0.002,
                ),
                Vec3::ZERO,
            );
        }
        sim
    }
    fn coverage(grid: &mut FineContactGrid, sim: &ClusterSimulation3D) {
        for (i, body) in sim.clumps.iter().enumerate().take(160).step_by(7) {
            let radius = sim.templates[body.template_idx].bounding_radius;
            grid.query(body.position, radius);
            assert!(grid.enabled());
            for (j, target) in sim.clumps.iter().enumerate() {
                let original = 0x55aa_55aa_55aa_55aau64;
                let bit = 1u64 << (j % 64);
                let result = grid.candidates(j / 64, original);
                assert_eq!(result & !original, 0);
                let r = radius + sim.templates[target.template_idx].bounding_radius;
                if original & bit != 0
                    && (!grid.is_small(j)
                        || (target.position - body.position).length_squared() <= r * r)
                {
                    assert_ne!(result & bit, 0, "missed pair {i}/{j}");
                }
            }
        }
        for i in 0..160 {
            let key = grid.key(sim.clumps[i].position).unwrap();
            assert_eq!(key, grid.cells[i]);
            for (axis, cell) in key.into_iter().enumerate() {
                for delta in -HALO..=HALO {
                    assert_ne!(
                        grid.members[grid.slot(axis, cell + delta) + i / 64] & (1u64 << (i % 64)),
                        0
                    );
                }
            }
        }
        assert_eq!(
            grid.members
                .iter()
                .map(|v| v.count_ones() as usize)
                .sum::<usize>(),
            160 * 3 * (2 * HALO as usize + 1)
        );
    }
    #[test]
    fn moved_nodes_remain_conservative_without_adding_coarse_members() {
        let mut sim = fixture();
        let mut grid = FineContactGrid::default();
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(grid.enabled());
        coverage(&mut grid, &sim);
        for i in (0..160).rev() {
            sim.clumps[i].position += Vec3::new(0.006, -0.004, 0.005);
            grid.moved(i, sim.clumps[i].position);
        }
        coverage(&mut grid, &sim);
        // Radius/template edits are rebuilt instead of inheriting stale membership.
        sim.templates[0] = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 0,
                style: IrregularStyle3D::Round,
            },
            0.005,
            1.,
        );
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(!grid.enabled());
        assert_eq!(grid.candidates(0, 0x12345), 0x12345);
    }
    #[test]
    fn rare_larger_parcels_do_not_set_the_fine_gold_spacing() {
        let mut sim = ClusterSimulation3D::new(Vec3::splat(-1.), Vec3::ONE);
        for radius in [0.0001, 0.0018, 0.01] {
            sim.add_template(ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 1,
                    seed: 0,
                    style: IrregularStyle3D::Round,
                },
                radius,
                1.,
            ));
        }
        for i in 0..192 {
            sim.spawn(
                if i < 180 {
                    0
                } else if i < 190 {
                    1
                } else {
                    2
                },
                Vec3::new(i as f32 * 0.0003, 0., 0.),
                Vec3::ZERO,
            );
        }
        let mut grid = FineContactGrid::default();
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(grid.enabled());
        assert!((grid.width - 0.0002).abs() < 1e-9);
        grid.query(Vec3::ZERO, 0.0001);
        assert!(!grid.refresh(Vec3::ZERO, 0.0001));
        assert!(grid.refresh(Vec3::new(0.01, 0., 0.), 0.0001));
        assert_ne!(
            grid.candidates(2, u64::MAX) & (1u64 << (185 % 64)),
            0,
            "larger target was incorrectly filtered"
        );
    }

    #[test]
    fn unrepresentable_queries_fall_back_to_every_original_candidate() {
        let mut sim = fixture();
        let mut grid = FineContactGrid::default();
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        grid.query(Vec3::splat(f32::INFINITY), 0.001);
        assert!(!grid.enabled());
        assert_eq!(grid.candidates(1, 0x12345), 0x12345);
        sim.clumps[3].position = Vec3::splat(f32::NAN);
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(!grid.enabled());
        assert_eq!(grid.candidates(1, 0x12345), 0x12345);
    }

    #[test]
    fn excessive_domains_and_out_of_domain_moves_keep_all_original_candidates() {
        let mut sim = fixture();
        sim.clumps[0].position.x = 10000.;
        let mut grid = FineContactGrid::default();
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(!grid.enabled());
        assert_eq!(grid.candidates(0, 0x12345), 0x12345);
        sim.clumps[0].position = Vec3::ZERO;
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(grid.enabled());
        grid.moved(1, Vec3::splat(100.));
        assert!(!grid.enabled());
        assert_eq!(grid.candidates(1, 0x56789), 0x56789);
    }
    #[test]
    fn subnormal_spacing_falls_back_before_padding_or_allocation_overflows() {
        let mut sim = fixture();
        sim.templates[0].bounding_radius = f32::from_bits(1);
        for body in &mut sim.clumps {
            body.position = Vec3::ZERO;
        }
        let mut grid = FineContactGrid::default();
        grid.rebuild(&sim.clumps, &sim.templates, 0.02);
        assert!(!grid.enabled());
        assert!(grid.members.is_empty());
        assert_eq!(grid.candidates(0, 0x12345), 0x12345);
    }
}
