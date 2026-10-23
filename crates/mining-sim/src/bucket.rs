//! Versioned hollow bucket geometry shared by steel contacts and presentation.
//! Dimensions are a declared generic 0.28 m3 design, not manufacturer calibration.
use crate::payload::{outer, MassMoments};
use glam::{DMat3, DVec2, DVec3, Vec3};
use rapier3d::prelude as r;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Model {
    #[default]
    LegacyFloor,
    ShellV1,
}
impl Model {
    pub fn is_legacy(&self) -> bool {
        *self == Self::LegacyFloor
    }
}
pub const STEEL_DENSITY_KG_M3: f64 = 7850.;
pub const STEEL_MASS_KG: f64 = 250.;
pub const WIDTH_M: f64 = 0.82;
pub const TOOTH_TIP: Vec3 = Vec3::new(0., -0.245, 1.065);
#[derive(Debug)]
pub struct Part {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub tooth: bool,
}
pub struct Geometry {
    pub parts: Vec<Part>,
    pub steel: MassMoments,
    pub thickness_m: f64,
    pub capacity_m3: f64,
    pub inner_half_width_m: f64,
    profile: Vec<DVec2>,
    shape: r::SharedShape,
}
fn area(p: &[DVec2]) -> f64 {
    (0..p.len())
        .map(|i| {
            let a = p[i];
            let b = p[(i + 1) % p.len()];
            a.x * b.y - a.y * b.x
        })
        .sum::<f64>()
        .abs()
        * 0.5
}
fn profile(height: f64) -> Vec<DVec2> {
    let mut p = vec![DVec2::new(height, 0.)];
    for i in 0..=6 {
        let a = i as f64 * std::f64::consts::FRAC_PI_2 / 6.;
        p.push(DVec2::new(-0.04 - 0.16 * a.sin(), 0.16 - 0.16 * a.cos()));
    }
    p.push(DVec2::new(-0.20, 0.88));
    p
}
fn prism(p: &[DVec2], lo: f64, hi: f64, tooth: bool) -> Part {
    let mut p = p.to_vec();
    let signed = (0..p.len())
        .map(|i| p[i].x * p[(i + 1) % p.len()].y - p[(i + 1) % p.len()].x * p[i].y)
        .sum::<f64>();
    if signed > 0. {
        p.reverse();
    }
    let n = p.len();
    let vertices = [lo, hi]
        .into_iter()
        .flat_map(|x| {
            p.iter()
                .map(move |q| Vec3::new(x as f32, q.x as f32, q.y as f32))
        })
        .collect();
    let mut triangles = Vec::new();
    for i in 1..n - 1 {
        triangles.push([0, i as u32, (i + 1) as u32]);
        triangles.push([n as u32, (n + i + 1) as u32, (n + i) as u32]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        triangles.push([i as u32, (n + i) as u32, (n + j) as u32]);
        triangles.push([i as u32, (n + j) as u32, j as u32]);
    }
    Part {
        vertices,
        triangles,
        tooth,
    }
}
/// Signed closed-triangle volume moments; f32 vertices own collision and mesh.
fn moments(p: &Part, density: f64) -> MassMoments {
    let (mut volume, mut first, mut second) = (0., DVec3::ZERO, DMat3::ZERO);
    for f in &p.triangles {
        let v = f.map(|i| p.vertices[i as usize].as_dvec3());
        let dv = v[0].dot(v[1].cross(v[2])) / 6.;
        let sum = v.into_iter().sum::<DVec3>();
        volume += dv;
        first += sum * (dv / 4.);
        second += (v.into_iter().map(outer).sum::<DMat3>() + outer(sum)) * (dv / 20.);
    }
    assert!(volume > 0.);
    let center = first / volume;
    MassMoments {
        mass_kg: volume * density,
        center,
        second: (second - outer(center) * volume) * density,
    }
}
fn parts(t: f64) -> (Vec<Part>, Vec<DVec2>, f64) {
    let half = WIDTH_M * 0.5 - t;
    let h = (crate::machine::BUCKET_CAPACITY / (half * 2.) - area(&profile(0.)))
        / (area(&profile(1.)) - area(&profile(0.)));
    let inner = profile(h);
    let n = inner.len();
    let normals: Vec<_> = inner
        .windows(2)
        .map(|p| (p[1] - p[0]).perp().normalize())
        .collect();
    let outer: Vec<_> = (0..n)
        .map(|i| {
            inner[i]
                + if i == 0 {
                    normals[0] * t
                } else if i == n - 1 {
                    normals[n - 2] * t
                } else {
                    (normals[i - 1] + normals[i]) * (t / (1. + normals[i - 1].dot(normals[i])))
                }
        })
        .collect();
    let mut parts = Vec::new();
    for i in 0..n - 1 {
        parts.push(prism(
            &[inner[i], inner[i + 1], outer[i + 1], outer[i]],
            -half,
            half,
            false,
        ));
    }
    // Carry each cheek to the actual mouth edge. Closing the offset shell
    // endpoints alone cuts a narrow wedge out of the declared cavity's side.
    let cheek: Vec<_> = std::iter::once(inner[0])
        .chain(outer.iter().copied())
        .chain(std::iter::once(inner[n - 1]))
        .collect();
    parts.push(prism(&cheek, -WIDTH_M * 0.5, -half, false));
    parts.push(prism(&cheek, half, WIDTH_M * 0.5, false));
    for x in [-0.30, -0.15, 0., 0.15, 0.30] {
        parts.push(prism(
            &[
                // Flush entry: a raised tooth heel blocks soil at the mouth.
                DVec2::new(-0.20, 0.88),
                DVec2::new(-0.20 - t, 0.88),
                DVec2::new(TOOTH_TIP.y as f64, TOOTH_TIP.z as f64),
            ],
            x - 0.05,
            x + 0.05,
            true,
        ));
    }
    (parts, inner, half)
}
pub fn geometry() -> &'static Geometry {
    static GEOMETRY: OnceLock<Geometry> = OnceLock::new();
    GEOMETRY.get_or_init(|| {
        let (mut lo, mut hi) = (0.004, 0.04);
        for _ in 0..48 {
            let t = (lo + hi) * 0.5;
            let mass = parts(t)
                .0
                .iter()
                .map(|p| moments(p, STEEL_DENSITY_KG_M3).mass_kg)
                .sum::<f64>();
            if mass < STEEL_MASS_KG {
                lo = t;
            } else {
                hi = t;
            }
        }
        let t = (lo + hi) * 0.5;
        let (parts, profile, half) = parts(t);
        let steel = parts
            .iter()
            .map(|p| moments(p, STEEL_DENSITY_KG_M3))
            .reduce(MassMoments::combine)
            .unwrap();
        let shape = r::SharedShape::compound(
            parts
                .iter()
                .map(|p| {
                    (
                        r::Pose::IDENTITY,
                        r::SharedShape::convex_hull(
                            &p.vertices
                                .iter()
                                .map(|v| r::Vector::new(v.x, v.y, v.z))
                                .collect::<Vec<_>>(),
                        )
                        .expect("convex steel part"),
                    )
                })
                .collect(),
        );
        let profile = profile
            .into_iter()
            .map(|p| p.as_vec2().as_dvec2())
            .collect::<Vec<_>>();
        let half = half as f32 as f64;
        Geometry {
            capacity_m3: area(&profile) * half * 2.,
            parts,
            steel,
            thickness_m: t,
            inner_half_width_m: half,
            profile,
            shape,
        }
    })
}
impl Geometry {
    pub fn shape(&self) -> r::SharedShape {
        self.shape.clone()
    }
    pub fn contains(&self, point: DVec3) -> bool {
        if point.x.abs() > self.inner_half_width_m {
            return false;
        }
        let q = DVec2::new(point.y, point.z);
        (0..self.profile.len()).all(|i| {
            let a = self.profile[i];
            let d = self.profile[(i + 1) % self.profile.len()] - a;
            d.perp().dot(q - a) <= 1e-9
        })
    }
    /// Necessary spatial-moment bounds for the explicit custody API. This is
    /// not an automatic contact/retention criterion for independent soil.
    pub fn accepts_moments(&self, m: MassMoments) -> bool {
        if !m.valid() || m.mass_kg <= 0. || !self.contains(m.center) {
            return false;
        }
        let vertices: Vec<_> = self
            .profile
            .iter()
            .flat_map(|p| {
                [-self.inner_half_width_m, self.inner_half_width_m].map(|x| DVec3::new(x, p.x, p.y))
            })
            .collect();
        std::iter::once(DVec3::X)
            .chain((0..self.profile.len()).map(|i| {
                let d = (self.profile[(i + 1) % self.profile.len()] - self.profile[i])
                    .perp()
                    .normalize();
                DVec3::new(0., d.x, d.y)
            }))
            .all(|n| {
                let lo = vertices
                    .iter()
                    .map(|p| p.dot(n))
                    .fold(f64::INFINITY, f64::min);
                let hi = vertices
                    .iter()
                    .map(|p| p.dot(n))
                    .fold(f64::NEG_INFINITY, f64::max);
                let c = m.center.dot(n);
                n.dot(m.second * n) / m.mass_kg <= (hi - c) * (c - lo) + 1e-10
            })
    }
    pub fn fill_moments(&self, volume: f64, mass: f64) -> MassMoments {
        if mass == 0. {
            return MassMoments {
                mass_kg: 0.,
                center: DVec3::ZERO,
                second: DMat3::ZERO,
            };
        }
        let floor = self
            .profile
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let (mut lo, mut hi) = (floor, self.profile[0].x);
        let target = volume.min(self.capacity_m3);
        let clipped = |level: f64| {
            let mut p = Vec::new();
            for i in 0..self.profile.len() {
                let a = self.profile[i];
                let b = self.profile[(i + 1) % self.profile.len()];
                if a.x <= level {
                    p.push(a);
                }
                if (a.x <= level) != (b.x <= level) {
                    p.push(a + (b - a) * ((level - a.x) / (b.x - a.x)));
                }
            }
            p
        };
        for _ in 0..48 {
            let mid = (lo + hi) * 0.5;
            if area(&clipped(mid)) * self.inner_half_width_m * 2. < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let p = clipped(hi);
        let origin = DVec2::new(floor, 0.16);
        let mut a = 0.;
        let mut first = DVec2::ZERO;
        let mut yy = 0.;
        let mut zz = 0.;
        let mut yz = 0.;
        for i in 0..p.len() {
            let u = p[i] - origin;
            let v = p[(i + 1) % p.len()] - origin;
            let c = u.x * v.y - u.y * v.x;
            a += c;
            first += (u + v) * c;
            yy += (u.x * u.x + u.x * v.x + v.x * v.x) * c;
            zz += (u.y * u.y + u.y * v.y + v.y * v.y) * c;
            yz += (2. * u.x * u.y + u.x * v.y + v.x * u.y + 2. * v.x * v.y) * c;
        }
        if a.abs() < 1e-28 {
            return MassMoments::cuboid(
                mass,
                DVec3::new(0., -0.20, 0.52),
                DVec3::new(self.inner_half_width_m, 0., 0.36),
                glam::DQuat::IDENTITY,
            );
        }
        let c = first / (3. * a);
        let cyy = yy / (6. * a) - c.x * c.x;
        let czz = zz / (6. * a) - c.y * c.y;
        let cyz = yz / (12. * a) - c.x * c.y;
        MassMoments {
            mass_kg: mass,
            center: DVec3::new(0., c.x + origin.x, c.y + origin.y),
            second: DMat3::from_cols(
                DVec3::new(self.inner_half_width_m.powi(2) / 3., 0., 0.),
                DVec3::new(0., cyy, cyz),
                DVec3::new(0., cyz, czz),
            ) * mass,
        }
    }
}

/// Diagnostic COM classification only. These clods remain independent bodies;
/// this observation never transfers custody or changes contact/inertia.
pub fn estimated_soil_contents(
    m: &crate::Excavator,
    soil: &crate::soil::ActiveSoil,
) -> crate::Material {
    if m.bucket_model() != Model::ShellV1 {
        return crate::Material::default();
    }
    let q = m.bucket_rotation().as_dquat();
    let wrist = m.arm().wrist.as_dvec3();
    let mut contents = crate::Material::default();
    for c in &soil.clods {
        if geometry().contains(q.inverse() * (c.center.as_dvec3() - wrist)) {
            contents.add(c.material());
        }
    }
    contents
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{soil::Clod, Material};
    use std::collections::BTreeMap;
    #[test]
    fn closed_steel_parts_match_independent_extrusion_integrals_and_real_void() {
        let g = geometry();
        assert_eq!(g.parts.len(), 15);
        assert!((g.steel.mass_kg - 250.).abs() < 0.001);
        assert!((g.capacity_m3 - 0.28).abs() < 1e-7);
        assert!(g
            .parts
            .iter()
            .filter(|p| p.tooth)
            .flat_map(|p| &p.vertices)
            .all(|p| p.y <= -0.20));
        let mut expected: Option<MassMoments> = None;
        for p in &g.parts {
            let mut edges = BTreeMap::new();
            for f in &p.triangles {
                for [a, b] in [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]] {
                    *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
            assert!(edges.values().all(|n| *n == 2), "open steel component");
            let n = p.vertices.len() / 2;
            let q = &p.vertices[..n];
            let lo = q[0].x as f64;
            let hi = p.vertices[n].x as f64;
            let (mut a, mut cy, mut cz, mut yy, mut zz, mut yz) = (0., 0., 0., 0., 0., 0.);
            for i in 0..n {
                let u = q[i].as_dvec3();
                let v = q[(i + 1) % n].as_dvec3();
                let c = u.y * v.z - u.z * v.y;
                a += c;
                cy += (u.y + v.y) * c;
                cz += (u.z + v.z) * c;
                yy += (u.y * u.y + u.y * v.y + v.y * v.y) * c;
                zz += (u.z * u.z + u.z * v.z + v.z * v.z) * c;
                yz += (2. * u.y * u.z + u.y * v.z + v.y * u.z + 2. * v.y * v.z) * c;
            }
            let c = DVec3::new((lo + hi) * 0.5, cy / (3. * a), cz / (3. * a));
            let mass = a.abs() * 0.5 * (hi - lo) * STEEL_DENSITY_KG_M3;
            let m = MassMoments {
                mass_kg: mass,
                center: c,
                second: DMat3::from_cols(
                    DVec3::new((hi - lo).powi(2) / 12., 0., 0.),
                    DVec3::new(0., yy / (6. * a) - c.y * c.y, yz / (12. * a) - c.y * c.z),
                    DVec3::new(0., yz / (12. * a) - c.y * c.z, zz / (6. * a) - c.z * c.z),
                ) * mass,
            };
            let actual = moments(p, STEEL_DENSITY_KG_M3);
            assert!((actual.mass_kg - m.mass_kg).abs() < 1e-9);
            assert!((actual.center - m.center).length() < 1e-10);
            assert!((actual.second - m.second)
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-9));
            expected = Some(expected.map_or(m, |old| old.combine(m)));
        }
        let expected = expected.unwrap();
        assert!((g.steel.second - expected.second)
            .to_cols_array()
            .iter()
            .all(|v| v.abs() < 1e-8));
        assert!(g.steel.valid());
        let p = r::Vector::new(0., 0., 0.4);
        assert!(g.contains(DVec3::new(0., 0., 0.4)));
        assert!(
            !g.shape.contains_point(&r::Pose::IDENTITY, p),
            "solid convex hull filled the cavity"
        );
        println!(
            "bucket thickness={}mm steel={}kg capacity={}m3 COM={:?} inertia={:?}",
            g.thickness_m * 1000.,
            g.steel.mass_kg,
            g.capacity_m3,
            g.steel.center,
            g.steel.inertia()
        );
    }
    #[test]
    fn partial_fill_moments_remain_finite_at_tiny_and_full_volumes() {
        let g = geometry();
        for v in [1e-18, 1e-12, 0.001, 0.1, 0.28] {
            let m = g.fill_moments(v, v * 1800.);
            assert!(m.valid(), "{v}: {m:?}");
            assert!((m.mass_kg - v * 1800.).abs() < 1e-12);
            assert!(g.contains(m.center));
        }
    }
    fn fixture(inverted: bool, sideways: bool, hz: usize, shell: bool) -> (usize, Vec<DVec3>) {
        let q = if inverted {
            glam::Quat::from_rotation_x(std::f32::consts::PI)
        } else if sideways {
            glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)
        } else {
            glam::Quat::IDENTITY
        };
        let mut world = r::PhysicsWorld::new();
        world.gravity = r::Vector::new(0., -9.81, 0.);
        world.integration_parameters.dt = 1. / hz as f32;
        world.integration_parameters.num_solver_iterations = 4;
        world.integration_parameters.num_internal_pgs_iterations = 16;
        world.integration_parameters.normalized_allowed_linear_error = 0.0001;
        world.integration_parameters.maximum_dissipation_friction = true;
        let shape = if shell {
            geometry().shape()
        } else {
            r::SharedShape::cuboid(0.41, 0.02, 0.54)
        };
        world.insert(
            r::RigidBodyBuilder::fixed(),
            r::ColliderBuilder::new(shape)
                .position(crate::tracks::pose(
                    if shell {
                        Vec3::ZERO
                    } else {
                        q * Vec3::new(0., -0.22, 0.44)
                    },
                    q,
                ))
                .friction(0.65),
        );
        let mut clods = Vec::new();
        let mut handles = Vec::new();
        for x in [-0.15, 0.15] {
            for z in [0.3, 0.6] {
                for y in [-0.14, -0.06] {
                    let mut c = Clod::prism(
                        Material {
                            soil_m3: 0.08 * 0.08 * 0.06 * 0.5,
                            ..Default::default()
                        },
                        [
                            DVec3::new(-0.04, 0., -0.04),
                            DVec3::new(0.04, 0., -0.04),
                            DVec3::new(-0.04, 0., 0.04),
                            DVec3::new(-0.04, 0.06, -0.04),
                            DVec3::new(0.04, 0.06, -0.04),
                            DVec3::new(-0.04, 0.06, 0.04),
                        ],
                    )
                    .unwrap();
                    assert!(
                        c.vertices().iter().all(|v| geometry()
                            .contains((*v + c.center + Vec3::new(x, y, z)).as_dvec3())),
                        "fixture clod straddles the open mouth before the test"
                    );
                    c.center = q * (c.center + Vec3::new(x, y, z));
                    c.rotation = q;
                    handles.push(c.insert(&mut world).unwrap());
                    clods.push(c);
                }
            }
        }
        for _ in 0..hz {
            world.step();
        }
        let mut positions = Vec::new();
        let mut retained = 0;
        for (mut c, h) in clods.into_iter().zip(handles) {
            c.read(&world.bodies[h]);
            assert!(c.valid());
            let p = (q.inverse() * c.center).as_dvec3();
            if sideways && shell && !geometry().contains(p) {
                println!("45-degree escaped: {p:?} velocity={:?}", c.velocity);
            }
            retained += usize::from(geometry().contains(p));
            positions.push(p);
        }
        (retained, positions)
    }
    #[test]
    fn actual_soil_is_retained_by_shell_and_cheeks_and_spills_through_open_mouth() {
        for hz in [240, 480] {
            let up = fixture(false, false, hz, true);
            let side = fixture(false, true, hz, true);
            let open = fixture(true, false, hz, true);
            let floor = fixture(false, true, hz, false);
            println!(
                "bucket fixture hz={hz}: upright={} tilted45={} inverted={} floor-only-tilted45={}",
                up.0, side.0, open.0, floor.0
            );
            assert_eq!(up.0, 8);
            assert_eq!(side.0, 8);
            assert_eq!(open.0, 0);
            assert_eq!(floor.0, 0);
        }
    }
}
