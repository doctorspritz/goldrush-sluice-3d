//! Local heightfield earthwork. Cuts follow the swept bucket floor, rather
//! than removing a depth per second around an idle tool. The 25 cm columns
//! cannot represent overhangs or soil wedges; local soil refinement remains
//! necessary before this is a continuum digging model.
use crate::{
    machine::ArmPose,
    site::{CELL, WIDTH},
    Material, Site, Vec3,
};

pub const BUCKET_WIDTH_M: f32 = 0.82;

#[derive(Clone, Copy, Debug)]
pub struct SoilStrength {
    pub density_kg_m3: f32,
    pub cohesion_pa: f32,
    pub friction_deg: f32,
}
impl SoilStrength {
    /// A level-ground, vertical-blade passive-pressure approximation, not the
    /// general Reece equation or a calibrated bucket/soil constitutive law.
    pub fn resistance_n(self, depth_m: f32, width_m: f32) -> f32 {
        let depth = depth_m.max(0.);
        let root_k = (std::f32::consts::FRAC_PI_4 + self.friction_deg.to_radians() * 0.5).tan();
        width_m
            * (0.5 * self.density_kg_m3 * 9.81 * depth * depth * root_k * root_k
                + 2. * self.cohesion_pa * depth * root_k)
    }
    fn mixture(m: Material) -> Self {
        let v = m.volume().max(1e-12);
        // Generic loose alluvium / cohesive overburden. These values are
        // explicit assumptions pending measured ground and saturation data.
        Self {
            density_kg_m3: (m.mass_kg() / v) as f32,
            cohesion_pa: ((m.soil_m3 * 12_000. + m.gravel_m3 * 1_000.) / v) as f32,
            friction_deg: ((m.soil_m3 * 28. + m.sand_m3 * 30. + m.gravel_m3 * 36.) / v) as f32,
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct CutPlan {
    cells: Vec<(usize, f32)>,
    pub material: Material,
    pub soil_force_n: f32,
    pub rock_blocked: bool,
}

fn floor_at(heel: Vec3, lip: Vec3, right: Vec3, p: Vec3) -> Option<f32> {
    let along = Vec3::new(lip.x - heel.x, 0., lip.z - heel.z);
    let length2 = along.length_squared();
    if length2 < 1e-8 {
        return None;
    }
    let offset = p - heel;
    let t = Vec3::new(offset.x, 0., offset.z).dot(along) / length2;
    if !(0. ..=1.).contains(&t) || offset.dot(right).abs() > BUCKET_WIDTH_M * 0.5 {
        return None;
    }
    Some(heel.y + (lip.y - heel.y) * t)
}

fn floor_corners(pose: ArmPose) -> [Vec3; 4] {
    let r = Vec3::new(pose.forward.z, 0., -pose.forward.x) * (BUCKET_WIDTH_M * 0.5);
    [pose.heel - r, pose.heel + r, pose.lip + r, pose.lip - r]
}
fn swept_faces(from: ArmPose, to: ArmPose) -> [[Vec3; 3]; 12] {
    let a = floor_corners(from);
    let b = floor_corners(to);
    let mut faces = [[Vec3::ZERO; 3]; 12];
    faces[0] = [a[0], a[1], a[2]];
    faces[1] = [a[0], a[2], a[3]];
    faces[2] = [b[0], b[1], b[2]];
    faces[3] = [b[0], b[2], b[3]];
    for i in 0..4 {
        let j = (i + 1) % 4;
        faces[4 + i * 2] = [a[i], a[j], b[j]];
        faces[5 + i * 2] = [a[i], b[j], b[i]];
    }
    faces
}
fn vertical_hit(p: Vec3, t: [Vec3; 3]) -> Option<f32> {
    let [a, b, c] = t;
    let ab = (b - a).as_dvec3();
    let ac = (c - a).as_dvec3();
    let q = (p - a).as_dvec3();
    let det = ab.x * ac.z - ab.z * ac.x;
    if det.abs() < 1e-12 {
        return None;
    }
    let u = (q.x * ac.z - q.z * ac.x) / det;
    let v = (ab.x * q.z - ab.z * q.x) / det;
    if u < -1e-6 || v < -1e-6 || u + v > 1. + 1e-6 {
        return None;
    }
    Some((a.y as f64 + u * ab.y + v * ac.y) as f32)
}

impl Site {
    pub fn bedrock_height(&self, p: Vec3) -> f32 {
        let q = (p / CELL).clamp(Vec3::ZERO, Vec3::splat((WIDTH - 1) as f32));
        let (x, z) = (q.x.floor() as usize, q.z.floor() as usize);
        let (nx, nz) = ((x + 1).min(WIDTH - 1), (z + 1).min(WIDTH - 1));
        let value = |x, z| self.world.bedrock_elevation[self.world.idx(x, z)];
        (value(x, z) * (1. - q.x.fract()) + value(nx, z) * q.x.fract()) * (1. - q.z.fract())
            + (value(x, nz) * (1. - q.x.fract()) + value(nx, nz) * q.x.fract()) * q.z.fract()
    }
    /// Check actual floor/tooth points as well as terrain columns, so a tooth
    /// between lattice points cannot pass through rock. Existing penetrations
    /// may retreat, which also lets older saves leave an invalid contact.
    pub(crate) fn bucket_blocked(&self, from: ArmPose, to: ArmPose, cutting: bool) -> bool {
        let r0 = Vec3::new(from.forward.z, 0., -from.forward.x);
        let r1 = Vec3::new(to.forward.z, 0., -to.forward.x);
        let travel = from.lip.distance(to.lip).max(from.heel.distance(to.heel));
        let steps = (travel / (CELL * 0.25)).ceil().clamp(1., 64.) as usize;
        for along in [0., 0.5, 1.] {
            for side in [-0.5, 0., 0.5] {
                let a = from.heel.lerp(from.lip, along) + r0 * (BUCKET_WIDTH_M * side);
                let b = to.heel.lerp(to.lip, along) + r1 * (BUCKET_WIDTH_M * side);
                let surface = |p: Vec3| {
                    if cutting {
                        self.bedrock_height(p)
                    } else {
                        self.height(p.x, p.z)
                    }
                };
                let initial = surface(a) - a.y;
                for step in 1..=steps {
                    let p = a.lerp(b, step as f32 / steps as f32);
                    let penetration = surface(p) - p.y;
                    if penetration > initial.max(0.0001) {
                        return true;
                    }
                }
            }
        }
        false
    }
    /// Preview a finite sweep without changing terrain. Rejected machine
    /// strokes therefore cannot mine material before their motion is accepted.
    pub(crate) fn bucket_cut(&self, from: ArmPose, to: ArmPose) -> CutPlan {
        let travel = from.lip.distance(to.lip).max(from.heel.distance(to.heel));
        if travel < 1e-6 {
            return CutPlan::default();
        }
        let r0 = Vec3::new(from.forward.z, 0., -from.forward.x);
        let r1 = Vec3::new(to.forward.z, 0., -to.forward.x);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for (pose, right) in [(from, r0), (to, r1)] {
            for p in [pose.heel, pose.lip] {
                for side in [-0.5, 0.5] {
                    let q = p + right * BUCKET_WIDTH_M * side;
                    min = min.min(q);
                    max = max.max(q);
                }
            }
        }
        let bounds = |v: f32| ((v / CELL).floor() as i32).clamp(1, WIDTH as i32 - 2) as usize;
        let faces = swept_faces(from, to);
        let mut out = CutPlan::default();
        let mut peak = 0f32;
        for z in bounds(min.z)..=(bounds(max.z) + 1).min(WIDTH - 2) {
            for x in bounds(min.x)..=(bounds(max.x) + 1).min(WIDTH - 2) {
                let i = self.world.idx(x, z);
                let ground = self.world.ground_height(x, z);
                let p = Vec3::new(x as f32 * CELL, 0., z as f32 * CELL);
                let mut target = ground;
                // Intersect the entire swept floor, including the side faces.
                // Sampling only the two endpoint floors misses the last cut
                // as a tooth leaves a terrain column, making yield depend on dt.
                for face in faces {
                    if let Some(h) = vertical_hit(p, face) {
                        target = target.min(h);
                    }
                }
                if let Some(h) = floor_at(to.heel, to.lip, r1, p) {
                    if h < self.world.bedrock_elevation[i] - 1e-5 {
                        let previous = floor_at(from.heel, from.lip, r0, p).unwrap_or(ground);
                        out.rock_blocked |= h < previous - 1e-5;
                    }
                }
                target = target.max(self.world.bedrock_elevation[i]);
                let depth = (ground - target).max(0.);
                if depth > 1e-7 {
                    out.cells.push((i, depth));
                    out.material.add(self.column_material(i, depth));
                    peak = peak.max(depth);
                }
            }
        }
        if !out.material.is_empty() {
            // Resistance depends on burial relative to uncut ground around
            // the tool, not just the tiny layer removed in this timestep.
            for point in [to.heel, to.lip] {
                for offset in [-CELL, CELL] {
                    let p = point + to.forward * offset;
                    if Site::bounds(p, 0.5) {
                        peak = peak.max(self.height(p.x, p.z) - point.y);
                    }
                }
            }
            out.soil_force_n =
                SoilStrength::mixture(out.material).resistance_n(peak, BUCKET_WIDTH_M);
        }
        out
    }
    pub(crate) fn apply_bucket_cut(&mut self, plan: CutPlan) -> Material {
        self.excavate_columns(plan.cells)
    }
    /// Shared stratigraphy for hand tools and machine cuts. Deposited gravel
    /// remains excavatable and paydirt's trace gold follows its removed volume.
    fn column_material(&self, i: usize, depth: f32) -> Material {
        let area = (CELL as f64).powi(2);
        let mut remaining = depth.max(0.) as f64 * area;
        let mut take = |available: f64| {
            let amount = remaining.min(available);
            remaining -= amount;
            amount
        };
        let sand = take(self.bed_class_volume_m3(i, 0));
        let soil = take(self.bed_class_volume_m3(i, 1));
        let gravel = take(self.bed_class_volume_m3(i, 2));
        let pay_available = self.bed_class_volume_m3(i, 3);
        let paydirt = take(pay_available);
        let mut material = Material {
            soil_m3: soil,
            sand_m3: sand,
            gravel_m3: gravel + paydirt,
            gold_g: if pay_available > 0. {
                self.gold_g[i] * paydirt / pay_available
            } else {
                0.
            },
            ..Default::default()
        };
        material.pore_water_m3 = material.pore_volume_m3() * self.column_saturation(i);
        material
    }
    pub(crate) fn excavate_columns(&mut self, cuts: Vec<(usize, f32)>) -> Material {
        let mut batch = Material::default();
        for (i, depth) in cuts {
            let mut material = self.column_material(i, depth);
            material.pore_water_m3 = self.take_pore_water(i, material.pore_water_m3);
            let mut left = depth.max(0.) as f64 * (CELL as f64).powi(2);
            let water = self.world.water_depth(i % WIDTH, i / WIDTH);
            for class in 0..4 {
                let old = self.bed_class_volume_m3(i, class);
                let amount = left.min(old);
                self.set_bed_volume_m3(i, class, old - amount);
                left -= amount;
            }
            self.gold_g[i] = (self.gold_g[i] - material.gold_g).max(0.);
            self.world.water_surface[i] = self.world.ground_height(i % WIDTH, i / WIDTH) + water;
            batch.add(material);
        }
        if !batch.is_empty() {
            self.revision += 1;
            self.excavated.add(batch);
        }
        batch
    }
}
