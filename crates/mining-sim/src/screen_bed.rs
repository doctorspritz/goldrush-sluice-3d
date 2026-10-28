//! Derived finite stock field. Coefficients own no inventory; their integral is
//! the supported parcels' reference handling volume (including gold once).
use super::{BulkDeck, Screen};
use crate::Material;
use glam::DVec2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BulkProjection {
    /// Saved model 2: original nearest-cell loading and pressure law.
    #[default]
    CellBinsV1,
    /// Saved model 3: conservative smooth deposition/gather and paired pressure.
    SmoothCellsV2,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct BulkBedCell {
    pub material: Material,
    /// Mean normal depth above the sloped deck, not another material owner.
    pub depth_m: f64,
    pub blinded: f64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct BulkBedSample {
    pub material: Material,
    pub depth_m: f64,
    /// Derivatives with respect to projected downstream x and across z.
    pub gradient: DVec2,
    pub blinded: f64,
}
#[derive(Clone, Debug)]
pub struct BulkBedField {
    nx: usize,
    nz: usize,
    length: f64,
    width: f64,
    cosine: f64,
    wire_radius: f64,
    projection: BulkProjection,
    area: f64,
    cells: Vec<BulkBedCell>,
}
impl BulkBedField {
    pub(super) fn new(screen: &Screen, config: &BulkDeck) -> Self {
        let nx = config.cells_downstream as usize;
        let nz = config.cells_across as usize;
        let cosine = screen.config.slope_deg.to_radians().cos();
        // Preserve the model-2 arithmetic order exactly.
        let area = screen.config.length_m / nx as f64 / cosine * screen.config.width_m / nz as f64;
        Self {
            nx,
            nz,
            length: screen.config.length_m,
            width: screen.config.width_m,
            cosine,
            wire_radius: screen.config.wire_radius_m,
            projection: config.projection,
            area,
            cells: vec![BulkBedCell::default(); nx * nz],
        }
    }
    pub(super) fn cell_area(&self) -> f64 {
        self.area
    }
    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.nz)
    }
    pub fn cells(&self) -> &[BulkBedCell] {
        &self.cells
    }
    pub fn length_m(&self) -> f64 {
        self.length
    }
    pub fn width_m(&self) -> f64 {
        self.width
    }
    pub fn slope_cosine(&self) -> f64 {
        self.cosine
    }
    pub fn wire_radius_m(&self) -> f64 {
        self.wire_radius
    }
    pub fn integrated_volume_m3(&self) -> f64 {
        self.cells.iter().map(|c| c.depth_m * self.area).sum()
    }
    pub fn material(&self) -> Material {
        let mut m = Material::default();
        for c in &self.cells {
            m.add(c.material);
        }
        m
    }
    pub(super) fn index(&self, x: f64, z: f64) -> usize {
        let ix = (x / self.length * self.nx as f64)
            .floor()
            .clamp(0., (self.nx - 1) as f64) as usize;
        let iz = ((z / self.width + 0.5) * self.nz as f64)
            .floor()
            .clamp(0., (self.nz - 1) as f64) as usize;
        ix * self.nz + iz
    }
    fn stencil(&self, x: f64, z: f64) -> [(usize, f64, DVec2); 4] {
        let sx = axis(x / self.length, self.nx, self.length);
        let sz = axis(z / self.width + 0.5, self.nz, self.width);
        std::array::from_fn(|i| {
            let (ix, wx, dx) = sx[i / 2];
            let (iz, wz, dz) = sz[i % 2];
            (ix * self.nz + iz, wx * wz, DVec2::new(dx * wz, wx * dz))
        })
    }
    pub(super) fn deposit(&mut self, x: f64, z: f64, m: Material, adhesive_coverage: f64) {
        if self.projection == BulkProjection::CellBinsV1 {
            let i = self.index(x, z);
            let c = &mut self.cells[i];
            c.material.add(m);
            c.depth_m += m.handling_volume_m3() / self.area;
            c.blinded += adhesive_coverage;
        } else {
            for (i, w, _) in self.stencil(x, z) {
                let c = &mut self.cells[i];
                let share = m.scaled(w);
                c.material.add(share);
                c.depth_m += share.handling_volume_m3() / self.area;
                c.blinded += adhesive_coverage * w;
            }
        }
    }
    pub(super) fn finish(&mut self) {
        for c in &mut self.cells {
            c.blinded = c.blinded.clamp(0., 1.);
        }
    }
    /// Positive C1 interpolation. Edge basis functions are extended constantly
    /// over the outer half-cell, so every basis integrates to one cell area.
    pub fn sample(&self, x: f64, z: f64) -> BulkBedSample {
        let mut out = BulkBedSample::default();
        for (i, w, dw) in self.stencil(x, z) {
            let c = self.cells[i];
            out.material.add(c.material.scaled(w));
            out.depth_m += c.depth_m * w;
            out.gradient += dw * c.depth_m;
            out.blinded += c.blinded * w;
        }
        out
    }
    /// Central pressure differences use zero-pressure exterior cells. D is
    /// skew-symmetric; matching deposit/gather weights give zero self force.
    /// The symmetric mass/volume term also makes variable-density pair forces
    /// equal/opposite. This does not claim exact angular momentum or energy.
    pub fn pressure_slopes(&self, x: f64, z: f64, parcel: Material) -> DVec2 {
        let specific_volume = parcel.handling_volume_m3() / parcel.wet_mass_kg().max(1e-30);
        let height = |ix: isize, iz: isize| -> f64 {
            if ix < 0 || iz < 0 || ix >= self.nx as isize || iz >= self.nz as isize {
                return 0.;
            }
            let c = self.cells[ix as usize * self.nz + iz as usize];
            0.5 * (c.depth_m + specific_volume * c.material.wet_mass_kg() / self.area)
        };
        let mut gradient = DVec2::ZERO;
        for (i, w, _) in self.stencil(x, z) {
            let ix = (i / self.nz) as isize;
            let iz = (i % self.nz) as isize;
            gradient.x +=
                w * (height(ix + 1, iz) - height(ix - 1, iz)) / (2. * self.length / self.nx as f64);
            gradient.y +=
                w * (height(ix, iz + 1) - height(ix, iz - 1)) / (2. * self.width / self.nz as f64);
        }
        gradient
    }
    pub fn interpolated(&self, next: &Self, alpha: f64) -> Option<Self> {
        if self.nx != next.nx
            || self.nz != next.nz
            || self.length != next.length
            || self.width != next.width
            || self.cosine != next.cosine
            || self.wire_radius != next.wire_radius
            || self.projection != next.projection
            || !alpha.is_finite()
        {
            return None;
        }
        let a = alpha.clamp(0., 1.);
        let mut out = self.clone();
        for ((c, p), q) in out.cells.iter_mut().zip(&self.cells).zip(&next.cells) {
            c.material = p.material.scaled(1. - a);
            c.material.add(q.material.scaled(a));
            c.depth_m = p.depth_m * (1. - a) + q.depth_m * a;
            c.blinded = p.blinded * (1. - a) + q.blinded * a;
        }
        Some(out)
    }
    /// Includes both boundaries and every cell centre. Symmetric subdivisions
    /// preserve the integral of the smooth basis under trapezoidal quadrature.
    pub fn mesh_axes(&self, subdivisions: usize) -> (Vec<f64>, Vec<f64>) {
        let n = subdivisions.clamp(1, 8);
        (
            knots(self.nx, self.length, n),
            knots(self.nz, self.width, n)
                .into_iter()
                .map(|z| z - self.width * 0.5)
                .collect(),
        )
    }
}
fn axis(unit: f64, count: usize, length: f64) -> [(usize, f64, f64); 2] {
    let p = unit.clamp(0., 1.) * count as f64 - 0.5;
    let lower = p.floor();
    let t = p - lower;
    let i = (lower as isize).clamp(0, count as isize - 1) as usize;
    let j = (lower as isize + 1).clamp(0, count as isize - 1) as usize;
    if i == j {
        return [(i, 1., 0.), (j, 0., 0.)];
    }
    let smooth = t * t * (3. - 2. * t);
    let derivative = 6. * t * (1. - t) * count as f64 / length;
    [(i, 1. - smooth, -derivative), (j, smooth, derivative)]
}
fn knots(count: usize, length: f64, subdivisions: usize) -> Vec<f64> {
    let mut base = vec![0.];
    base.extend((0..count).map(|i| (i as f64 + 0.5) * length / count as f64));
    base.push(length);
    let mut out = Vec::with_capacity((count + 1) * subdivisions + 1);
    for pair in base.windows(2) {
        for k in 0..subdivisions {
            out.push(pair[0] + (pair[1] - pair[0]) * k as f64 / subdivisions as f64);
        }
    }
    out.push(length);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;
    fn field(nx: u16, nz: u16) -> BulkBedField {
        let s = Screen::new(Vec3::ZERO, 0.);
        BulkBedField::new(
            &s,
            &BulkDeck {
                cells_downstream: nx,
                cells_across: nz,
                ..Default::default()
            },
        )
    }
    fn close(a: Material, b: Material) {
        for (x, y) in [
            (a.soil_m3, b.soil_m3),
            (a.sand_m3, b.sand_m3),
            (a.gravel_m3, b.gravel_m3),
            (a.gold_g, b.gold_g),
            (a.pore_water_m3, b.pore_water_m3),
        ] {
            assert!((x - y).abs() < 1e-12, "{x} != {y}");
        }
    }
    #[test]
    fn positive_projection_keeps_every_material_and_the_slope_correct_volume() {
        for (nx, nz) in [(1, 1), (2, 3), (16, 4), (31, 7), (64, 16)] {
            let mut f = field(nx, nz);
            let mut total = Material::default();
            for i in 0..101 {
                let m = Material {
                    soil_m3: 1e-6,
                    sand_m3: 2e-6,
                    gravel_m3: 3e-6,
                    gold_g: 0.001,
                    pore_water_m3: 1e-7,
                };
                let x = f.length * (i as f64 / 100.);
                let z = f.width * ((i * 37 % 101) as f64 / 100. - 0.5);
                f.deposit(x, z, m, 0.05);
                total.add(m);
            }
            f.finish();
            close(f.material(), total);
            assert!((f.integrated_volume_m3() - total.handling_volume_m3()).abs() < 1e-15);
            assert!(f
                .cells()
                .iter()
                .all(|c| c.depth_m >= 0. && (0.0..=1.).contains(&c.blinded)));
        }
    }
    #[test]
    fn reconstructed_surface_integrates_to_source_stock_with_independent_quadrature() {
        for (nx, nz) in [(1, 1), (3, 2), (16, 4)] {
            let mut f = field(nx, nz);
            let m = Material {
                gravel_m3: 0.0001,
                gold_g: 0.01,
                ..Default::default()
            };
            for (x, z) in [(0., -0.19), (0.134, 0.07), (0.8, 0.19)] {
                f.deposit(x, z, m, 0.);
            }
            let n = 511;
            let k = 239;
            let dx = f.length / n as f64;
            let dz = f.width / k as f64;
            let mut integral = 0.;
            for i in 0..n {
                for j in 0..k {
                    integral += f
                        .sample((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dz - f.width * 0.5)
                        .depth_m
                        * dx
                        * dz
                        / f.cosine;
                }
            }
            let target = 3. * m.handling_volume_m3();
            assert!(
                (integral / target - 1.).abs() < 3e-5,
                "resolution{nx}/{nz} volume {integral} target {target}"
            );
            for subdiv in [1, 2, 4, 8] {
                let (x, z) = f.mesh_axes(subdiv);
                let mut volume = 0.;
                for a in x.windows(2) {
                    for b in z.windows(2) {
                        let corners = f.sample(a[0], b[0]).depth_m
                            + f.sample(a[1], b[0]).depth_m
                            + f.sample(a[0], b[1]).depth_m
                            + f.sample(a[1], b[1]).depth_m;
                        let centre = f.sample((a[0] + a[1]) * 0.5, (b[0] + b[1]) * 0.5).depth_m;
                        volume +=
                            (a[1] - a[0]) * (b[1] - b[0]) / f.cosine * (corners / 6. + centre / 3.);
                    }
                }
                assert!(
                    (volume - target).abs() < 1e-15,
                    "fan volume {volume} != {target}"
                );
            }
        }
    }
    #[test]
    fn pressure_has_no_self_force_even_at_edges_and_with_gold_density() {
        for (nx, nz) in [(1, 1), (2, 3), (16, 4)] {
            for i in 0..201 {
                let mut f = field(nx, nz);
                let m = if i % 2 == 0 {
                    Material {
                        gold_g: 2.,
                        ..Default::default()
                    }
                } else {
                    Material {
                        soil_m3: 0.0002,
                        pore_water_m3: 1e-5,
                        ..Default::default()
                    }
                };
                let x = i as f64 / 200. * f.length;
                let z = ((i * 23 % 201) as f64 / 200. - 0.5) * f.width;
                f.deposit(x, z, m, 0.);
                let force = f.pressure_slopes(x, z, m) * m.wet_mass_kg();
                assert!(
                    force.length() < 1e-13,
                    "self force{force:?} position{x}/{z}"
                );
            }
        }
    }
    #[test]
    fn pair_pressure_is_equal_and_opposite_for_different_densities_and_water() {
        let mut f = field(16, 4);
        let a = Material {
            soil_m3: 0.0003,
            pore_water_m3: 1e-5,
            ..Default::default()
        };
        let b = Material {
            gold_g: 2.,
            ..Default::default()
        };
        let p = (0.235, -0.05);
        let q = (0.272, -0.02);
        f.deposit(p.0, p.1, a, 0.);
        f.deposit(q.0, q.1, b, 0.);
        let fa = f.pressure_slopes(p.0, p.1, a) * a.wet_mass_kg();
        let fb = f.pressure_slopes(q.0, q.1, b) * b.wet_mass_kg();
        assert!(fa.length() > 1e-8);
        assert!(
            (fa + fb).length() < 1e-13,
            "net internal force {:?}",
            fa + fb
        );
    }
    #[test]
    fn pressure_and_surface_are_continuous_across_the_old_cell_boundaries() {
        let mut f = field(16, 4);
        let m = Material {
            sand_m3: 0.0001,
            ..Default::default()
        };
        f.deposit(0.14, -0.03, m, 0.4);
        for i in 1..16 {
            let x = i as f64 * f.length / 16.;
            let a = f.sample(x - 1e-10, -0.03);
            let b = f.sample(x + 1e-10, -0.03);
            assert!((a.depth_m - b.depth_m).abs() < 1e-9);
            assert!((a.gradient - b.gradient).length() < 1e-7);
            assert!(
                (f.pressure_slopes(x - 1e-10, -0.03, m) - f.pressure_slopes(x + 1e-10, -0.03, m))
                    .length()
                    < 1e-7
            );
        }
    }
    #[test]
    fn field_interpolation_keeps_volume_and_material_without_mutating_either_owner() {
        let mut a = field(16, 4);
        let mut b = field(16, 4);
        a.deposit(
            0.1,
            -0.12,
            Material {
                soil_m3: 0.0001,
                ..Default::default()
            },
            0.3,
        );
        b.deposit(
            0.6,
            0.12,
            Material {
                gravel_m3: 0.0004,
                gold_g: 0.1,
                ..Default::default()
            },
            0.1,
        );
        for alpha in [0., 0.1, 0.5, 1.] {
            let mixed = a.interpolated(&b, alpha).unwrap();
            let mut m = a.material().scaled(1. - alpha);
            m.add(b.material().scaled(alpha));
            close(mixed.material(), m);
            assert!((mixed.integrated_volume_m3() - m.handling_volume_m3()).abs() < 1e-15);
        }
        assert!(a.interpolated(&field(8, 4), 0.5).is_none());
        assert!(a.interpolated(&b, f64::NAN).is_none());
        assert!((a.material().soil_m3 - 0.0001).abs() < 1e-15);
    }
    #[test]
    fn missing_projection_preserves_saved_model_two_instead_of_selecting_the_new_law() {
        let old: BulkDeck = serde_json::from_str(
            r#"{"cells_downstream":16,"cells_across":4,"soil_yield_pa":25.0}"#,
        )
        .unwrap();
        assert_eq!(old.projection, BulkProjection::CellBinsV1);
        let mut s = Screen::new(Vec3::ZERO, 0.);
        s.set_bulk_deck(Some(old)).unwrap();
        assert_eq!(s.processing_model_version(), 2);
        s.set_bulk_deck(Some(BulkDeck::default())).unwrap();
        assert_eq!(s.processing_model_version(), 3);
    }
}
