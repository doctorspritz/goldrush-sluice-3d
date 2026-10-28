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
    MountedShellV2,
}
impl Model {
    pub fn is_legacy(&self) -> bool {
        *self == Self::LegacyFloor
    }
    pub fn is_shell(self) -> bool {
        !self.is_legacy()
    }
    pub fn geometry(self) -> &'static Geometry {
        geometry_for(self)
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
    pub model: Model,
    pub tooth_tip: Vec3,
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

// Sloped sector bowl below and in front of an exterior wrist pin. The
// pin/ear masses occupy only the exterior, with no overlapping steel volumes.
fn mounted_parts(t: f64) -> (Vec<Part>, Vec<DVec2>, f64) {
    let half = WIDTH_M * 0.5 - t;
    let unit_area = (0..8).map(|_| (1.1_f64 / 8.).sin() * 0.5).sum::<f64>();
    let radius = (crate::machine::BUCKET_CAPACITY / (2. * half * unit_area)).sqrt();
    let mut inner = vec![DVec2::new(-0.16, 0.12)];
    for i in 0..=8 {
        let a = i as f64 * 1.1 / 8.;
        inner.push(DVec2::new(
            -0.16 - radius * a.cos(),
            0.12 + radius * a.sin(),
        ));
    }
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
    let cheek: Vec<_> = std::iter::once(inner[0])
        .chain(outer.iter().copied())
        .chain(std::iter::once(inner[n - 1]))
        .collect();
    parts.push(prism(&cheek, -WIDTH_M * 0.5, -half, false));
    parts.push(prism(&cheek, half, WIDTH_M * 0.5, false));
    // Continue the actual inner-floor tangent. The old downward tooth vector
    // pointed almost vertically at the working rake, creating a raised entry
    // obstruction and driving thin clods into the fixed substrate.
    let tip = inner[n - 1] + (inner[n - 1] - inner[n - 2]).normalize() * 0.09;
    for x in [-0.30, -0.15, 0., 0.15, 0.30] {
        parts.push(prism(
            &[inner[n - 1], outer[n - 1], tip],
            x - 0.05,
            x + 0.05,
            true,
        ));
    }
    let back = outer[0].y;
    let ear = [
        DVec2::new(-0.35, -0.06),
        DVec2::new(0.07, -0.06),
        DVec2::new(0.07, back),
        DVec2::new(-0.35, back),
    ];
    parts.push(prism(&ear, -0.205, -0.18, false));
    parts.push(prism(&ear, 0.18, 0.205, false));
    let pin: Vec<_> = (0..16)
        .map(|i| {
            let a = i as f64 * std::f64::consts::TAU / 16.;
            DVec2::new(a.cos() * 0.035, a.sin() * 0.035)
        })
        .collect();
    parts.push(prism(&pin, -0.18, 0.18, false));
    (parts, inner, half)
}
pub fn geometry() -> &'static Geometry {
    geometry_for(Model::MountedShellV2)
}
fn geometry_for(model: Model) -> &'static Geometry {
    static V1: OnceLock<Geometry> = OnceLock::new();
    static V2: OnceLock<Geometry> = OnceLock::new();
    let slot = match model {
        Model::ShellV1 => &V1,
        Model::MountedShellV2 => &V2,
        Model::LegacyFloor => panic!("legacy floor has no shell"),
    };
    let make_parts = |t| match model {
        Model::ShellV1 => parts(t),
        Model::MountedShellV2 => mounted_parts(t),
        Model::LegacyFloor => unreachable!(),
    };
    slot.get_or_init(|| {
        let (mut lo, mut hi) = (0.004, 0.04);
        for _ in 0..48 {
            let t = (lo + hi) * 0.5;
            let mass = make_parts(t)
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
        let (parts, profile, half) = make_parts(t);
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
        let tooth_tip = if model == Model::ShellV1 {
            TOOTH_TIP
        } else {
            // Read the tip from the registered tooth vertices. Controls and
            // activation must follow the actual steel after a profile change.
            let tangent = (profile[profile.len() - 1] - profile[profile.len() - 2]).normalize();
            let tip = parts
                .iter()
                .filter(|p| p.tooth)
                .flat_map(|p| &p.vertices)
                .max_by(|a, b| {
                    tangent
                        .dot(DVec2::new(a.y as f64, a.z as f64))
                        .total_cmp(&tangent.dot(DVec2::new(b.y as f64, b.z as f64)))
                })
                .unwrap();
            Vec3::new(0., tip.y, tip.z)
        };
        Geometry {
            model,
            tooth_tip,
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
    /// Physical cutting direction along the final inner-floor segment, from
    /// tooth root toward the registered tip. This is a geometry query only.
    pub fn cutting_direction(&self) -> Vec3 {
        let edge = self.profile[self.profile.len() - 1] - self.profile[self.profile.len() - 2];
        Vec3::new(0., edge.x as f32, edge.y as f32).normalize()
    }
    pub fn mouth_normal(&self) -> Vec3 {
        let edge = self.profile[self.profile.len() - 1] - self.profile[0];
        Vec3::new(0., edge.y as f32, -edge.x as f32).normalize()
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
        let origin = DVec2::new(
            floor,
            if self.model == Model::ShellV1 {
                0.16
            } else {
                0.12
            },
        );
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
            if self.model == Model::MountedShellV2 {
                return MassMoments::cuboid(
                    mass,
                    DVec3::new(0., floor, 0.12),
                    DVec3::new(self.inner_half_width_m, 0., 0.),
                    glam::DQuat::IDENTITY,
                );
            }
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
    if !m.bucket_model().is_shell() {
        return crate::Material::default();
    }
    let q = m.bucket_rotation().as_dquat();
    let wrist = m.arm().wrist.as_dvec3();
    let mut contents = crate::Material::default();
    for c in &soil.clods {
        if m.bucket_model()
            .geometry()
            .contains(q.inverse() * (c.center.as_dvec3() - wrist))
        {
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
        assert_eq!(g.parts.len(), 19);
        assert!((g.steel.mass_kg - 250.).abs() < 0.001);
        assert!((g.capacity_m3 - 0.28).abs() < 1e-7);
        assert!(g
            .parts
            .iter()
            .filter(|p| p.tooth)
            .flat_map(|p| &p.vertices)
            .all(|p| p.y <= g.profile[0].x as f32 + 1e-6));
        let mut expected: Option<MassMoments> = None;
        for p in &g.parts {
            let mut edges = BTreeMap::new();
            for f in &p.triangles {
                let v = f.map(|i| p.vertices[i as usize]);
                assert!(
                    (v[1] - v[0]).cross(v[2] - v[0]).length_squared() > 0.,
                    "degenerate visual face"
                );
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
        let p = r::Vector::new(0., -0.45, 0.4);
        assert!(g.contains(DVec3::new(0., -0.45, 0.4)));
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
    fn fixture(
        inverted: bool,
        sideways: bool,
        hz: usize,
        shell: bool,
        model: Model,
    ) -> (usize, Vec<DVec3>) {
        let g = model.geometry();
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
        if model == Model::MountedShellV2 {
            crate::soil_contacts::configure(&mut world);
        }
        let shape = if shell {
            g.shape()
        } else {
            r::SharedShape::compound(
                g.parts
                    .iter()
                    .take(g.profile.len() - 1)
                    .map(|p| {
                        (
                            r::Pose::IDENTITY,
                            r::SharedShape::convex_hull(
                                &p.vertices
                                    .iter()
                                    .map(|v| r::Vector::new(v.x, v.y, v.z))
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap(),
                        )
                    })
                    .collect(),
            )
        };
        world.insert(
            r::RigidBodyBuilder::fixed(),
            r::ColliderBuilder::new(shape)
                .position(crate::tracks::pose(Vec3::ZERO, q))
                .friction(0.65),
        );
        let mut clods = Vec::new();
        let mut handles = Vec::new();
        for x in [-0.15, 0.15] {
            for z in if model == Model::ShellV1 {
                [0.3, 0.6]
            } else {
                [0.32, 0.48]
            } {
                for y in if model == Model::ShellV1 {
                    [-0.14, -0.06]
                } else {
                    [-0.55, -0.47]
                } {
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
                        c.vertices()
                            .iter()
                            .all(|v| g.contains((*v + c.center + Vec3::new(x, y, z)).as_dvec3())),
                        "fixture clod straddles the open mouth before the test"
                    );
                    c.center = q * (c.center + Vec3::new(x, y, z));
                    c.rotation = q;
                    let handle = if model == Model::MountedShellV2 {
                        c.insert_precise(&mut world)
                    } else {
                        c.insert(&mut world)
                    }
                    .unwrap();
                    handles.push(handle);
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
            if sideways && shell && !g.contains(p) {
                println!("45-degree escaped: {p:?} velocity={:?}", c.velocity);
            }
            retained += usize::from(g.contains(p));
            positions.push(p);
        }
        (retained, positions)
    }
    #[test]
    fn actual_soil_is_retained_by_shell_and_cheeks_and_spills_through_open_mouth() {
        for model in [Model::ShellV1, Model::MountedShellV2] {
            for hz in [240, 480] {
                let up = fixture(false, false, hz, true, model);
                let side = fixture(false, true, hz, true, model);
                let open = fixture(true, false, hz, true, model);
                let floor = fixture(false, true, hz, false, model);
                println!(
                "bucket fixture model={model:?} hz={hz}: upright={} tilted45={} inverted={} floor-only-tilted45={}",
                up.0, side.0, open.0, floor.0
            );
                assert_eq!(up.0, 8);
                assert_eq!(side.0, 8);
                assert_eq!(open.0, 0);
                assert_eq!(floor.0, 0);
            }
        }
    }
    #[test]
    fn working_teeth_advance_into_the_bank_without_a_vertical_entry_step() {
        let g = geometry();
        let front = g.profile[g.profile.len() - 1];
        let mut machine = crate::Excavator::new(&crate::Site::new());
        machine.curl = machine.boom + machine.stick + std::f32::consts::FRAC_PI_2 - 1.1;
        machine.enable_articulation().unwrap();
        assert!((machine.arm().angle - 1.1).abs() < 1e-6);
        let q = machine.bucket_rotation();
        assert!(
            g.parts.iter().filter(|p| p.tooth).all(|p| p
                .vertices
                .iter()
                .any(|v| v.y == g.tooth_tip.y && v.z == g.tooth_tip.z)),
            "control tip differs from registered teeth"
        );
        let delta = q * (g.tooth_tip - Vec3::new(0., front.x as f32, front.y as f32));
        assert!(
            delta.z < -0.08 && delta.y.abs() < 0.015,
            "tooth is an entry obstruction at the working rake: {delta}"
        );
        let old = q * Vec3::new(0., -0.035, 0.08);
        assert!(
            old.y < -0.08 && old.z.abs() < 0.01,
            "old-tooth control did not reproduce the vertical step"
        );
    }
    #[test]
    fn exterior_mount_keeps_the_actual_stick_capsule_clear_of_the_entire_cavity() {
        let cavity = |g: &Geometry| {
            r::SharedShape::convex_hull(
                &g.profile
                    .iter()
                    .flat_map(|p| {
                        [-g.inner_half_width_m, g.inner_half_width_m]
                            .map(|x| r::Vector::new(x as f32, p.x as f32, p.y as f32))
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let new = cavity(geometry());
        let old = cavity(Model::ShellV1.geometry());
        let mut minimum = f32::INFINITY;
        let mut old_min = f32::INFINITY;
        for radius in [0.13, 0.143] {
            for i in 0..=315 {
                let curl = -1.70 + i as f32 * 0.01;
                let direction = r::Vector::new(0., curl.cos(), curl.sin());
                let stick = r::SharedShape::capsule(
                    r::Vector::ZERO,
                    direction * crate::machine::STICK_LENGTH,
                    radius,
                );
                let d = rapier3d::parry::query::distance(
                    &r::Pose::IDENTITY,
                    &*new,
                    &r::Pose::IDENTITY,
                    &*stick,
                )
                .unwrap()
                .distance;
                let previous = rapier3d::parry::query::distance(
                    &r::Pose::IDENTITY,
                    &*old,
                    &r::Pose::IDENTITY,
                    &*stick,
                )
                .unwrap()
                .distance;
                if radius == 0.143 {
                    // Check the actual rectangular rendered section as well
                    // as the physical capsule, in its registered hinge plane.
                    let half_width = crate::machine::STICK_VISUAL_WIDTH_M * 0.5;
                    let half_depth = half_width * crate::machine::LINK_VISUAL_DEPTH_RATIO;
                    let vertices: Vec<_> = [-half_width, half_width]
                        .into_iter()
                        .flat_map(|x| {
                            [0., crate::machine::STICK_LENGTH]
                                .into_iter()
                                .flat_map(move |y| {
                                    [-half_depth, half_depth].map(|z| {
                                        r::Vector::new(
                                            x,
                                            direction.y * y - direction.z * z,
                                            direction.z * y + direction.y * z,
                                        )
                                    })
                                })
                        })
                        .collect();
                    let beam = r::SharedShape::convex_hull(&vertices).unwrap();
                    let gap = rapier3d::parry::query::distance(
                        &r::Pose::IDENTITY,
                        &*new,
                        &r::Pose::IDENTITY,
                        &*beam,
                    )
                    .unwrap()
                    .distance;
                    assert!(
                        gap > 0.025,
                        "actual beam section enters cavity: curl={curl} gap={gap}"
                    );
                }
                minimum = minimum.min(d);
                old_min = old_min.min(previous);
                assert!(d > 0.025, "curl={curl} cavity/stick gap={d}");
            }
        }
        println!(
            "stick/cavity minimum gap including visible member={minimum}m; V1 control={old_min}m"
        );
        assert!(
            old_min < 0.001,
            "negative control did not reproduce intrusion"
        );
    }
}

#[cfg(test)]
mod mouth_entry_tests {
    use super::*;
    use crate::{soil::Clod, Material};
    #[test]
    fn independent_clod_enters_the_actual_open_mouth_but_not_a_closed_hull() {
        let g = geometry();
        let first = g.profile[0];
        let last = g.profile[g.profile.len() - 1];
        let midpoint = Vec3::new(
            0.,
            ((first.x + last.x) * 0.5) as f32,
            ((first.y + last.y) * 0.5) as f32,
        );
        let n = g.mouth_normal();
        for side in [0.02_f32, 0.08] {
            for angle in [0., 1.1, 1.3] {
                for closed in [false, true] {
                    let q = glam::Quat::from_rotation_x(-angle)
                        * glam::Quat::from_rotation_y(std::f32::consts::PI);
                    let mut world = r::PhysicsWorld::new();
                    crate::soil_contacts::configure(&mut world);
                    world.gravity = r::Vector::ZERO;
                    world.integration_parameters.dt = 1. / 240.;
                    world.integration_parameters.num_solver_iterations = 4;
                    world.integration_parameters.num_internal_pgs_iterations = 16;
                    world.integration_parameters.normalized_allowed_linear_error = 0.0001;
                    let shape = if closed {
                        r::SharedShape::convex_hull(
                            &g.parts
                                .iter()
                                .flat_map(|p| &p.vertices)
                                .map(|p| r::Vector::new(p.x, p.y, p.z))
                                .collect::<Vec<_>>(),
                        )
                        .unwrap()
                    } else {
                        g.shape()
                    };
                    world.insert(
                        r::RigidBodyBuilder::fixed(),
                        r::ColliderBuilder::new(shape.clone())
                            .position(crate::tracks::pose(Vec3::ZERO, q))
                            .friction(0.65),
                    );
                    let length = side as f64;
                    let mut c = Clod::prism(
                        Material {
                            sand_m3: length.powi(3) * 0.5,
                            ..Default::default()
                        },
                        [
                            DVec3::ZERO,
                            DVec3::new(length, 0., 0.),
                            DVec3::new(0., 0., length),
                            DVec3::new(0., length, 0.),
                            DVec3::new(length, length, 0.),
                            DVec3::new(0., length, length),
                        ],
                    )
                    .unwrap();
                    c.center = q * (midpoint + n * 0.20);
                    c.rotation = q;
                    c.velocity = q * (-n * 0.4);
                    let h = c.insert_precise(&mut world).unwrap();
                    let clod_shape = r::SharedShape::convex_hull(
                        &c.vertices()
                            .iter()
                            .map(|v| r::Vector::new(v.x, v.y, v.z))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let initial_contact = rapier3d::parry::query::contact(
                        &crate::tracks::pose(Vec3::ZERO, q),
                        &*shape,
                        &crate::tracks::pose(c.center, c.rotation),
                        &*clod_shape,
                        0.02,
                    )
                    .unwrap();
                    assert!(
                        initial_contact.is_none_or(|p| p.dist > 0.),
                        "entry fixture starts inside steel"
                    );

                    let before = c.totals(Vec3::ZERO).kinetic_j;
                    for _ in 0..240 {
                        world.step();
                    }
                    c.read(&world.bodies[h]);
                    let local = q.inverse() * c.center;
                    println!(
                        "mouth side={side} angle={angle} closed={closed}: local={local:?} K={}",
                        c.totals(Vec3::ZERO).kinetic_j
                    );
                    assert_eq!(
                        g.contains(local.as_dvec3()),
                        !closed,
                        "clod did not pass through the physical open mouth"
                    );
                    assert!(
                        c.totals(Vec3::ZERO).kinetic_j <= before + 1e-6,
                        "fixed unpowered shell created energy"
                    );
                }
            }
        }
    }
}
