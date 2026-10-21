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
    pub fn new(s: &Sluice) -> Self {
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
            // Water already densely samples every wet cell. A conservative
            // histogram avoids eight scatter operations per water particle;
            // the binomial reconstruction and final sample remain continuous.
            let q = (p.position / CELL_M).floor().to_array();
            let cell = std::array::from_fn(|a| (q[a] as i32).clamp(0, dims[a] as i32 - 1) as usize);
            let i = field.index(cell);
            field.cells[i].water += s.profile.particle_m3();
        }
        for g in &s.grains {
            if g.kind != GrainKind::Sand || g.resting {
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
        // Separable binomial reconstruction. Clamping destinations (rather
        // than dropping edge taps) keeps the projected integrals unchanged.
        for axis in 0..3 {
            let mut next = vec![Cell::default(); field.cells.len()];
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let p = [x, y, z];
                        let value = field.cells[field.index(p)];
                        for (offset, weight) in [(-1, 0.25), (0, 0.5), (1, 0.25)] {
                            let mut q = p;
                            q[axis] =
                                (p[axis] as i32 + offset).clamp(0, dims[axis] as i32 - 1) as usize;
                            next[field.index(q)].add(value, weight);
                        }
                    }
                }
            }
            field.cells = next;
        }
        field
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
        let mut cell = Cell::default();
        for (i, w) in self.stencil(p) {
            cell.add(self.cells[i], w);
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
        "projection_water_error_m3":water-s.water.len() as f64*s.profile.particle_m3(),
        "projection_fines_error_m3":fines-s.grains.iter().filter(|g|g.kind==GrainKind::Sand&&!g.resting).map(|g|g.solid_volume()).sum::<f64>(),
        "outlet_extinction_m1":field.outlet_extinction(s),"optical_grid_m":CELL_M,
        "optics_calibrated":false,"independently_resolved_grain_collisions":false})
}
