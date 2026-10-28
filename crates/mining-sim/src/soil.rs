//! Finite moving soil in the machine's contact world. This first constitutive
//! case is frictional/drained: no cohesion, pore pressure or compaction law.
use crate::{
    payload::MassMoments,
    site::{CELL, WIDTH},
    tracks::{pose, rv, sq, sv},
    Material, Site, Vec3,
};
use glam::{DMat3, DVec3, Quat};
use rapier3d::prelude as r;
use serde::{Deserialize, Serialize};

pub const MAX_CLODS: usize = 2048;
const TETS: [[usize; 4]; 3] = [[0, 1, 2, 5], [0, 1, 4, 5], [0, 3, 4, 5]];
pub const FACES: [[usize; 3]; 8] = [
    [0, 2, 1],
    [3, 4, 5],
    [0, 1, 4],
    [0, 4, 3],
    [1, 2, 5],
    [1, 5, 4],
    [2, 0, 3],
    [2, 3, 5],
];
fn outer(p: DVec3) -> DMat3 {
    DMat3::from_cols(p * p.x, p * p.y, p * p.z)
}
/// Exact uniform-density tetrahedral moments of a convex vertical prism.
/// Three independent simplex integrals, with a nearby origin for conditioning.
pub fn prism_moments(vertices: [DVec3; 6], mass: f64) -> Result<(f64, MassMoments), String> {
    if !mass.is_finite() || mass <= 0. || vertices.iter().any(|v| !v.is_finite()) {
        return Err("Invalid soil prism".into());
    }
    for i in 0..3 {
        if ((vertices[i].x - vertices[i + 3].x).abs() + (vertices[i].z - vertices[i + 3].z).abs())
            > 1e-10
            || vertices[i + 3].y < vertices[i].y
        {
            return Err("Soil geometry must be a vertical convex prism".into());
        }
    }
    let origin = vertices[0];
    let p = vertices.map(|v| v - origin);
    let mut volume = 0.;
    let mut first = DVec3::ZERO;
    let mut second = DMat3::ZERO;
    for ids in TETS {
        let v = ids.map(|i| p[i]);
        let dv = (v[1] - v[0]).dot((v[2] - v[0]).cross(v[3] - v[0])).abs() / 6.;
        let sum = v.into_iter().sum::<DVec3>();
        volume += dv;
        first += sum * (dv / 4.);
        second += (v.into_iter().map(outer).sum::<DMat3>() + outer(sum)) * (dv / 20.);
    }
    if !volume.is_finite() || volume <= 1e-16 {
        return Err("Unresolved soil prism volume".into());
    }
    let center = first / volume;
    Ok((
        volume,
        MassMoments {
            mass_kg: mass,
            center: origin + center,
            second: (second / volume - outer(center)) * mass,
        },
    ))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Clod {
    material: Material,
    vertices: [Vec3; 6],
    moments: MassMoments,
    pub center: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    #[serde(skip)]
    shape: Option<r::SharedShape>,
    /// Derived from immutable local geometry/material; pose and velocity do not
    /// affect it. Recomputed deterministically after deserialization.
    #[serde(skip)]
    precise_properties: Option<r::MassProperties>,
}
impl Clod {
    pub fn prism(material: Material, vertices: [DVec3; 6]) -> Result<Self, String> {
        if !material.valid() || material.volume() <= 0. {
            return Err("Invalid soil material".into());
        }
        let (_, world_moments) = prism_moments(vertices, material.wet_mass_kg())?;
        let local = vertices.map(|p| (p - world_moments.center).as_vec3());
        // The f32 collider geometry owns registered inertia and the mesh;
        // exact f64 finite stocks own the mass and material custody.
        let (_, mut moments) = prism_moments(local.map(|v| v.as_dvec3()), material.wet_mass_kg())?;
        moments.center = DVec3::ZERO;
        let body = Self {
            material,
            vertices: local,
            moments,
            center: world_moments.center.as_vec3(),
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            shape: None,
            precise_properties: None,
        };
        if !body.valid() {
            return Err("Invalid soil shape or mass moments".into());
        }
        Ok(body)
    }
    pub fn material(&self) -> Material {
        self.material
    }
    pub fn vertices(&self) -> &[Vec3; 6] {
        &self.vertices
    }
    pub fn moments(&self) -> MassMoments {
        self.moments
    }
    pub fn geometry_volume_m3(&self) -> f64 {
        prism_moments(self.vertices.map(|v| v.as_dvec3()), 1.).map_or(0., |x| x.0)
    }
    pub fn valid(&self) -> bool {
        self.material.valid()
            && self.material.volume() > 0.
            && self.material.wet_mass_kg() == self.moments.mass_kg
            && self.moments.valid()
            && self.moments.center == DVec3::ZERO
            && self.center.is_finite()
            && self.center.length() < 1000.
            && self.rotation.is_finite()
            && (self.rotation.length_squared() - 1.).abs() < 0.0001
            && self.velocity.is_finite()
            && self.velocity.length() < 100.
            && self.angular_velocity.is_finite()
            // Bound physical point speed, not an arbitrary angular rate.
            // A centimetre clod can legitimately spin at 200 rad/s while its
            // surface moves only a few m/s. Large or explosive motion still
            // has the same 100 m/s bound as translation.
            && self.vertices.iter().all(|p| {
                (self.velocity + self.angular_velocity.cross(self.rotation * *p)).length() < 100.
            })
            && self
                .vertices
                .iter()
                .all(|p| p.is_finite() && p.length() < 2.)
            && prism_moments(
                self.vertices.map(|v| v.as_dvec3()),
                self.material.wet_mass_kg(),
            )
            .is_ok_and(|(_, m)| m.second == self.moments.second)
    }
    pub fn friction(&self) -> f32 {
        let m = self.material;
        ((m.sand_m3 * 30. + m.soil_m3 * 28. + m.gravel_m3 * 36.) / m.volume())
            .to_radians()
            .tan() as f32
    }
    pub(crate) fn insert(
        &mut self,
        world: &mut r::PhysicsWorld,
    ) -> Result<r::RigidBodyHandle, String> {
        self.insert_with_registration(world, false)
    }
    pub(crate) fn insert_precise(
        &mut self,
        world: &mut r::PhysicsWorld,
    ) -> Result<r::RigidBodyHandle, String> {
        self.insert_with_registration(world, true)
    }
    fn insert_with_registration(
        &mut self,
        world: &mut r::PhysicsWorld,
        precise: bool,
    ) -> Result<r::RigidBodyHandle, String> {
        if self.shape.is_none() {
            self.shape = Some(
                r::SharedShape::convex_hull(&self.vertices.map(rv))
                    .ok_or("Invalid soil convex hull")?,
            );
        }
        let properties = if precise {
            let moments = self.moments;
            *self
                .precise_properties
                .get_or_insert_with(|| precise_mass_properties(moments))
        } else {
            r::MassProperties::with_inertia_matrix(
                r::Vector::ZERO,
                self.moments.mass_kg as f32,
                r::Matrix::from_cols_array(
                    &self.moments.inertia().to_cols_array().map(|v| v as f32),
                ),
            )
        };
        let h = world.insert_body(
            r::RigidBodyBuilder::dynamic()
                .pose(pose(self.center, self.rotation))
                .linvel(rv(self.velocity))
                .angvel(rv(self.angular_velocity))
                .can_sleep(false)
                // The default full-step rotation cap silently discards spin
                // energy above 45 degrees. New small clods are bounded by
                // physical point speed and need their true angular motion.
                .allow_fast_rotation(precise)
                .ccd_enabled(true)
                .additional_mass_properties(properties),
        );
        world.insert_collider(
            r::ColliderBuilder::new(self.shape.as_ref().unwrap().clone())
                .density(0.)
                .friction(self.friction())
                .friction_combine_rule(r::CoefficientCombineRule::Min)
                .restitution(0.)
                .user_data(20),
            Some(h),
        );
        world.bodies[h].recompute_mass_properties_from_colliders(&world.colliders);
        Ok(h)
    }
    pub(crate) fn read(&mut self, body: &r::RigidBody) {
        self.center = sv(body.translation());
        self.rotation = sq(*body.rotation());
        self.velocity = sv(body.linvel());
        self.angular_velocity = sv(body.angvel());
    }
    pub fn totals(&self, gravity: Vec3) -> Totals {
        let mass = self.moments.mass_kg;
        let q = glam::DMat3::from_quat(self.rotation.as_dquat());
        let i = q * self.moments.inertia() * q.transpose();
        let w = self.angular_velocity.as_dvec3();
        let v = self.velocity.as_dvec3();
        let p = v * mass;
        let spin = i * w;
        Totals {
            mass_kg: mass,
            first_kg_m: self.center.as_dvec3() * mass,
            momentum_ns: p,
            angular_nms: self.center.as_dvec3().cross(p) + spin,
            kinetic_j: 0.5 * (v.dot(p) + w.dot(spin)),
            gravitational_j: -mass * gravity.as_dvec3().dot(self.center.as_dvec3()),
        }
    }
}
// Scale-independent symmetric Jacobi diagonalization. Starting from the
// identity keeps already principal axes (including repeated eigenspaces) stable.
// The pinned general eigensolver uses an absolute cross-product cutoff that is
// inappropriate for small SI tensors. Exact geometry and mass remain unchanged.
fn precise_mass_properties(moments: MassMoments) -> r::MassProperties {
    let inertia = moments.inertia();
    let scale = inertia
        .to_cols_array()
        .into_iter()
        .map(f64::abs)
        .fold(0., f64::max);
    let mut a = (inertia / scale).to_cols_array_2d();
    let mut v = DMat3::IDENTITY.to_cols_array_2d();
    for _ in 0..12 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[q][p];
            if apq.abs() < 1e-14 {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2. * apq);
            let t = if tau >= 0. {
                1. / (tau + (1. + tau * tau).sqrt())
            } else {
                -1. / (-tau + (1. + tau * tau).sqrt())
            };
            let c = 1. / (1. + t * t).sqrt();
            let sn = t * c;
            for row in &mut a {
                let ap = row[p];
                let aq = row[q];
                row[p] = c * ap - sn * aq;
                row[q] = sn * ap + c * aq;
            }
            for row in &mut v {
                let ap = row[p];
                let aq = row[q];
                row[p] = c * ap - sn * aq;
                row[q] = sn * ap + c * aq;
            }
            let (before, after) = a.split_at_mut(q);
            for (ap, aq) in before[p].iter_mut().zip(&mut after[0]) {
                let old_p = *ap;
                let old_q = *aq;
                *ap = c * old_p - sn * old_q;
                *aq = sn * old_p + c * old_q;
            }
        }
    }
    let basis = DMat3::from_cols_array_2d(&v).transpose();
    let frame = r::Rotation::from_mat3(&r::Matrix::from_cols_array(
        &basis.to_cols_array().map(|x| x as f32),
    ))
    .normalize();
    r::MassProperties::with_principal_inertia_frame(
        r::Vector::ZERO,
        moments.mass_kg as f32,
        r::Vector::new(
            (a[0][0] * scale) as f32,
            (a[1][1] * scale) as f32,
            (a[2][2] * scale) as f32,
        ),
        frame,
    )
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Totals {
    pub mass_kg: f64,
    pub first_kg_m: DVec3,
    pub momentum_ns: DVec3,
    pub angular_nms: DVec3,
    pub kinetic_j: f64,
    pub gravitational_j: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContactFeedback {
    pub steps: u64,
    pub bucket_contacts: usize,
    pub bucket_normal_impulse_ns: DVec3,
    pub machine_momentum_change_ns: DVec3,
    pub soil_momentum_change_ns: DVec3,
    pub coupled_passivity_residual_j: f64,
    /// Only true without fixed contacts, imposed external loads or driven belts.
    pub closed_free_system: bool,
    pub free_linear_residual_ns: DVec3,
    pub free_angular_residual_nms: DVec3,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveSoil {
    pub clods: Vec<Clod>,
    pub source: Material,
    pub geometry_volume_error_m3: f64,
    pub feedback: ContactFeedback,
    #[serde(skip)]
    ground: Option<(u64, r::SharedShape)>,
}
impl ActiveSoil {
    pub fn from_clods(clods: Vec<Clod>) -> Result<Self, String> {
        let mut source = Material::default();
        for c in &clods {
            source.add(c.material);
        }
        let result = Self {
            geometry_volume_error_m3: clods.iter().map(Clod::geometry_volume_m3).sum::<f64>()
                - source.volume(),
            clods,
            source,
            feedback: Default::default(),
            ground: None,
        };
        if !result.valid() {
            return Err("Invalid active soil".into());
        }
        Ok(result)
    }
    pub fn material(&self) -> Material {
        let mut m = Material::default();
        for c in &self.clods {
            m.add(c.material);
        }
        m
    }
    pub fn valid(&self) -> bool {
        let m = self.material();
        let a = [m.soil_m3, m.sand_m3, m.gravel_m3, m.gold_g, m.pore_water_m3];
        let b = [
            self.source.soil_m3,
            self.source.sand_m3,
            self.source.gravel_m3,
            self.source.gold_g,
            self.source.pore_water_m3,
        ];
        !self.clods.is_empty()
            && self.clods.len() <= MAX_CLODS
            && self.source.valid()
            && self.clods.iter().all(Clod::valid)
            && self.geometry_volume_error_m3.is_finite()
            && self.feedback.steps < 1_000_000_000_000
            && a.into_iter()
                .zip(b)
                .all(|(a, b)| (a - b).abs() <= 1e-12 * (1. + b))
            && self.feedback.bucket_normal_impulse_ns.is_finite()
            && self.feedback.machine_momentum_change_ns.is_finite()
            && self.feedback.soil_momentum_change_ns.is_finite()
            && self.feedback.coupled_passivity_residual_j.is_finite()
            && self.feedback.free_linear_residual_ns.is_finite()
            && self.feedback.free_angular_residual_nms.is_finite()
    }
    pub fn totals(&self, gravity: Vec3) -> Totals {
        let mut t = Totals::default();
        for c in &self.clods {
            let a = c.totals(gravity);
            t.mass_kg += a.mass_kg;
            t.first_kg_m += a.first_kg_m;
            t.momentum_ns += a.momentum_ns;
            t.angular_nms += a.angular_nms;
            t.kinetic_j += a.kinetic_j;
            t.gravitational_j += a.gravitational_j;
        }
        t
    }
    pub(crate) fn ground(&mut self, site: &Site) -> r::SharedShape {
        if self
            .ground
            .as_ref()
            .is_none_or(|(revision, _)| *revision != site.revision)
        {
            let heights = (0..WIDTH)
                .flat_map(|x| (0..WIDTH).map(move |z| site.world.ground_height(x, z)))
                .collect();
            let shape = r::SharedShape::heightfield_with_flags(
                r::Array2::new(WIDTH, WIDTH, heights),
                r::Vector::new((WIDTH - 1) as f32 * CELL, 1., (WIDTH - 1) as f32 * CELL),
                r::HeightFieldFlags::FIX_INTERNAL_EDGES,
            );
            self.ground = Some((site.revision, shape));
        }
        self.ground.as_ref().unwrap().1.clone()
    }
    /// Wet contact is deliberately refused until moving-solid displacement and
    /// effective stress are coupled to the existing conservative flow solver.
    pub fn dry_contact(&self, site: &Site) -> bool {
        self.clods.iter().all(|c| {
            let p = c.vertices.map(|v| c.center + c.rotation * v);
            let min = p.into_iter().fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let max = p
                .into_iter()
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            let lo = (min / CELL).floor().as_ivec3();
            let hi = (max / CELL).ceil().as_ivec3();
            (lo.z.max(0)..=hi.z.min(WIDTH as i32 - 1)).all(|z| {
                (lo.x.max(0)..=hi.x.min(WIDTH as i32 - 1)).all(|x| {
                    let i = site.world.idx(x as usize, z as usize);
                    site.world.water_depth(x as usize, z as usize) <= 1e-6
                        || site.world.water_surface[i] < min.y - 0.001
                })
            })
        })
    }
}

/// Vertical height of the actual two-triangle collision cell. The smoother
/// bilinear site query can differ by centimetres across an activated pit and
/// must not be used as a signed contact-penetration oracle.
pub fn substrate_height(site: &Site, x: f32, z: f32) -> f32 {
    let gx = (x / CELL).clamp(0., (WIDTH - 1) as f32);
    let gz = (z / CELL).clamp(0., (WIDTH - 1) as f32);
    let ix = (gx.floor() as usize).min(WIDTH - 2);
    let iz = (gz.floor() as usize).min(WIDTH - 2);
    let fx = gx - ix as f32;
    let fz = gz - iz as f32;
    let a = site.world.ground_height(ix, iz);
    let b = site.world.ground_height(ix, iz + 1);
    let c = site.world.ground_height(ix + 1, iz);
    let d = site.world.ground_height(ix + 1, iz + 1);
    if fx + fz <= 1. {
        a + (c - a) * fx + (b - a) * fz
    } else {
        d + (b - d) * (1. - fx) + (c - d) * (1. - fz)
    }
}

/// Activate the complete support of selected terrain nodes. Matching Parry's
/// triangle diagonal makes the moving prisms fill exactly the removed surface
/// layer, including boundary wedges. The substrate remains a real collider.
pub fn activate(
    site: &mut Site,
    nodes: &[(usize, f32)],
    subdivisions: usize,
) -> Result<ActiveSoil, String> {
    if nodes.is_empty() || nodes.len() > 64 || !(1..=4).contains(&subdivisions) {
        return Err("Bounded soil activation required".into());
    }
    let mut unique = std::collections::HashSet::new();
    if nodes.iter().any(|&(i, d)| {
        i >= WIDTH * WIDTH
            || i % WIDTH < 1
            || i % WIDTH >= WIDTH - 1
            || i / WIDTH < 1
            || i / WIDTH >= WIDTH - 1
            || !unique.insert(i)
            || !d.is_finite()
            || d <= 0.
            || d > 0.5
    }) {
        return Err("Invalid activation nodes/depth".into());
    }
    let mut candidate = site.clone();
    let mut removed = vec![Material::default(); WIDTH * WIDTH];
    let old = (0..WIDTH * WIDTH)
        .map(|i| site.world.ground_height(i % WIDTH, i / WIDTH) as f64)
        .collect::<Vec<_>>();
    for &(i, d) in nodes {
        removed[i] = candidate.excavate_columns(vec![(i, d)]);
    }
    let new = (0..WIDTH * WIDTH)
        .map(|i| candidate.world.ground_height(i % WIDTH, i / WIDTH) as f64)
        .collect::<Vec<_>>();
    let mut clods = Vec::new();
    let n = subdivisions;
    for z in 0..WIDTH - 1 {
        for x in 0..WIDTH - 1 {
            let a = z * WIDTH + x;
            let b = (z + 1) * WIDTH + x;
            let c = z * WIDTH + x + 1;
            let d = (z + 1) * WIDTH + x + 1;
            for ids in [[a, b, c], [b, d, c]] {
                if ids.iter().all(|&i| removed[i].volume() == 0.) {
                    continue;
                }
                let point = |uv: [usize; 2], layer: f64| {
                    let w = [
                        1. - (uv[0] + uv[1]) as f64 / n as f64,
                        uv[0] as f64 / n as f64,
                        uv[1] as f64 / n as f64,
                    ];
                    let p = std::array::from_fn::<_, 3, _>(|k| {
                        DVec3::new(
                            (ids[k] % WIDTH) as f64 * CELL as f64,
                            new[ids[k]] + (old[ids[k]] - new[ids[k]]) * layer,
                            (ids[k] / WIDTH) as f64 * CELL as f64,
                        )
                    });
                    p[0] * w[0] + p[1] * w[1] + p[2] * w[2]
                };
                let layers = ((ids.iter().map(|&i| old[i] - new[i]).fold(0., f64::max)
                    / (CELL as f64 / n as f64))
                    .ceil() as usize)
                    .max(1);
                for u in 0..n {
                    for v in 0..n - u {
                        let mut triangles = vec![[[u, v], [u + 1, v], [u, v + 1]]];
                        if u + v + 1 < n {
                            triangles.push([[u + 1, v], [u + 1, v + 1], [u, v + 1]]);
                        }
                        for tri in triangles {
                            for layer in 0..layers {
                                let mut material = Material::default();
                                for uv in tri {
                                    let w = [
                                        1. - (uv[0] + uv[1]) as f64 / n as f64,
                                        uv[0] as f64 / n as f64,
                                        uv[1] as f64 / n as f64,
                                    ];
                                    for k in 0..3 {
                                        material.add(
                                            removed[ids[k]]
                                                .scaled(w[k] / (6. * (n * n * layers) as f64)),
                                        );
                                    }
                                }
                                if material.volume() == 0. {
                                    continue;
                                }
                                let vertices = std::array::from_fn(|i| {
                                    point(tri[i % 3], (layer + i / 3) as f64 / layers as f64)
                                });
                                clods.push(Clod::prism(material, vertices)?);
                                if clods.len() > MAX_CLODS {
                                    return Err(
                                        "Active soil capacity exceeded without changing terrain"
                                            .into(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let soil = ActiveSoil::from_clods(clods)?;
    if !soil.dry_contact(&candidate) {
        return Err("Moving wet-soil displacement is not coupled yet".into());
    }
    if soil.geometry_volume_error_m3.abs() > 1e-4 * soil.source.volume() {
        return Err("Terrain projection does not resolve this activation depth".into());
    }
    *site = candidate;
    Ok(soil)
}

#[cfg(test)]
mod substrate_query_tests {
    use super::*;
    #[test]
    fn contact_height_matches_actual_heightfield_rays_on_both_triangles() {
        let mut site = Site::new();
        let mut soil = activate(&mut site, &[(80 * WIDTH + 76, 0.18)], 2).unwrap();
        let shape = soil.ground(&site);
        let center = (WIDTH - 1) as f32 * CELL * 0.5;
        let mut bilinear_error = 0f32;
        for x in 75..78 {
            for z in 79..82 {
                for fx in [0.2, 0.6, 0.9] {
                    for fz in [0.2, 0.6, 0.9] {
                        let px = (x as f32 + fx) * CELL;
                        let pz = (z as f32 + fz) * CELL;
                        let ray = r::Ray::new(
                            r::Vector::new(px - center, 10., pz - center),
                            -r::Vector::Y,
                        );
                        let hit = shape.cast_local_ray(&ray, 20., true).unwrap();
                        let actual = 10. - hit;
                        let query = substrate_height(&site, px, pz);
                        assert!(
                            (query - actual).abs() < 0.00001,
                            "triangle contact height disagrees: {query} vs {actual}"
                        );
                        bilinear_error = bilinear_error.max((site.height(px, pz) - actual).abs());
                    }
                }
            }
        }
        assert!(
            bilinear_error > 0.005,
            "smooth-height negative control did not reproduce the pit mismatch"
        );
    }

    // Independent double-precision SAT over the actual registered hull points.
    // All triples include each face normal, and all edge pairs include each
    // edge cross. Extra chord axes cannot separate truly intersecting hulls.
    fn hull_separation(a: &[DVec3], b: &[DVec3]) -> f64 {
        let mut best = f64::NEG_INFINITY;
        let mut axis = |n: DVec3| {
            if n.length_squared() < 1e-24 {
                return;
            }
            let n = n.normalize();
            let lo = |p: &[DVec3]| p.iter().map(|p| p.dot(n)).fold(f64::INFINITY, f64::min);
            let hi = |p: &[DVec3]| p.iter().map(|p| p.dot(n)).fold(f64::NEG_INFINITY, f64::max);
            best = best.max(lo(b) - hi(a)).max(lo(a) - hi(b));
        };
        for points in [a, b] {
            for i in 0..points.len() {
                for j in i + 1..points.len() {
                    for k in j + 1..points.len() {
                        axis((points[j] - points[i]).cross(points[k] - points[i]));
                    }
                }
            }
        }
        for i in 0..a.len() {
            for j in i + 1..a.len() {
                for k in 0..b.len() {
                    for l in k + 1..b.len() {
                        axis((a[j] - a[i]).cross(b[l] - b[k]));
                    }
                }
            }
        }
        best
    }
    fn registered_points(c: &r::Collider) -> Vec<DVec3> {
        let p = c.position();
        let q = p.rotation.to_array();
        let q = glam::DQuat::from_xyzw(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
        let t = DVec3::from_array(p.translation.to_array().map(f64::from));
        c.shape()
            .as_convex_polyhedron()
            .unwrap()
            .points()
            .iter()
            .map(|v| q * DVec3::from_array(v.to_array().map(f64::from)) + t)
            .collect()
    }
    #[test]
    fn independent_hull_oracle_detects_real_overlap_at_the_same_tolerance() {
        let q = glam::DQuat::from_rotation_y(0.37) * glam::DQuat::from_rotation_z(0.7);
        let origin = DVec3::new(18., 0.9, 22.);
        let box_points: Vec<_> = [-0.05, 0.05]
            .into_iter()
            .flat_map(|x| {
                [-0.05, 0.05]
                    .into_iter()
                    .flat_map(move |y| [-0.05, 0.05].map(move |z| DVec3::new(x, y, z)))
            })
            .collect();
        let a: Vec<_> = box_points.iter().map(|p| origin + q * *p).collect();
        for gap in [-0.0001, -0.00002, 0., 0.00002, 0.0001] {
            let b: Vec<_> = box_points
                .iter()
                .map(|p| origin + q * (*p + DVec3::X * (0.1 + gap)))
                .collect();
            let measured = hull_separation(&a, &b);
            assert!(
                (measured - gap).abs() < 1e-11,
                "independent known box gap={gap}, measured={measured}"
            );
            assert_eq!(
                measured < -0.00001,
                gap < -0.00001,
                "actual overlap escaped the unchanged10um gate"
            );
        }
    }
    #[test]
    fn activated_bank_has_no_initial_contact_overlap() {
        let mut s = crate::Session::new();
        s.prepare_soil_trial().unwrap();
        let soil = s.active_soil.as_mut().unwrap();
        let ground = soil.ground(&s.site);
        let mut world = r::PhysicsWorld::new();
        crate::soil_contacts::configure(&mut world);
        let center = (WIDTH - 1) as f32 * CELL * 0.5;
        world.insert(
            r::RigidBodyBuilder::fixed(),
            r::ColliderBuilder::new(ground)
                .translation(r::Vector::new(center, 0., center))
                .user_data(10),
        );
        let mut initial = Vec::new();
        // Every initial prism is contained in one terrain triangle. Its linear
        // bottom is above that same triangle iff all its vertices are above it.
        let mut ground_gap = f64::INFINITY;
        for clod in &mut soil.clods {
            for v in clod.vertices() {
                let p = clod.center.as_dvec3() + clod.rotation.as_dquat() * v.as_dvec3();
                ground_gap =
                    ground_gap.min(p.y - substrate_height(&s.site, p.x as f32, p.z as f32) as f64);
            }
            let h = clod.insert(&mut world).unwrap();
            initial.push((h, clod.center));
        }
        assert!(
            ground_gap > -0.00001,
            "initial prism intersects its substrate: {ground_gap}"
        );
        world.integration_parameters.dt = 0.;
        world.step();
        for (h, p) in initial {
            assert_eq!(
                world.bodies[h].translation().to_array(),
                p.to_array(),
                "zero-time query moved soil"
            );
        }
        let mut total = 0;
        let mut reported_worst = 0f32;
        let mut geometry_worst = f64::INFINITY;
        let mut hull_pairs = 0;
        for pair in world.narrow_phase.contact_pairs() {
            let a = &world.colliders[pair.collider1];
            let b = &world.colliders[pair.collider2];
            if a.shape().as_convex_polyhedron().is_some()
                && b.shape().as_convex_polyhedron().is_some()
            {
                let gap = hull_separation(&registered_points(a), &registered_points(b));
                geometry_worst = geometry_worst.min(gap);
                hull_pairs += 1;
                assert!(gap > -0.00001, "registered initial hulls overlap: {gap}");
            }
            for manifold in pair.solver_manifolds() {
                for point in &manifold.points {
                    assert!(point.dist.is_finite());
                    total += 1;
                    reported_worst = reported_worst.min(point.dist);
                }
            }
        }
        assert!(
            total > 0 && hull_pairs > 0,
            "no initial contact/hull pairs were checked"
        );
        // Keep the clipped solver estimate visible: it is not an independent
        // geometry oracle. This bank reproduces a -17um estimate at zero true
        // hull overlap. Real overlaps still fail the same10um geometry bound.
        println!("initial bank: {hull_pairs} hull pairs, registered gap={geometry_worst}m, substrate gap={ground_gap}m, clipped-contact estimate={reported_worst}m");
    }
}

#[cfg(test)]
mod inertia_registration_tests {
    use super::*;
    fn prism(scale: f64, tapered: bool) -> Clod {
        let turn = glam::DQuat::from_rotation_y(0.37);
        let top = if tapered { [0.8, 1.2, 1.] } else { [1.; 3] };
        let bottom = [
            DVec3::ZERO,
            DVec3::new(scale, 0., 0.),
            DVec3::new(0., 0., scale),
        ];
        let vertices = std::array::from_fn(|i| {
            let j = i % 3;
            turn * (bottom[j]
                + if i < 3 {
                    DVec3::ZERO
                } else {
                    DVec3::Y * (scale * top[j])
                })
        });
        let volume = prism_moments(vertices, 1.).unwrap().0;
        Clod::prism(
            Material {
                sand_m3: volume,
                ..Default::default()
            },
            vertices,
        )
        .unwrap()
    }
    fn relative_error(c: &mut Clod, precise: bool) -> f64 {
        let mut world = r::PhysicsWorld::new();
        let h = if precise {
            c.insert_precise(&mut world)
        } else {
            c.insert(&mut world)
        }
        .unwrap();
        let properties = &world.bodies[h].mass_properties().local_mprops;
        let actual = DMat3::from_cols_array(
            &properties
                .reconstruct_inertia_matrix()
                .to_cols_array()
                .map(|x| x as f64),
        );
        let expected = c.moments().inertia();
        assert!(
            (properties.mass() as f64 - c.material().wet_mass_kg()).abs()
                < c.material().wet_mass_kg() * 1e-6
        );
        let norm = expected
            .to_cols_array()
            .into_iter()
            .map(f64::abs)
            .fold(0., f64::max);
        (actual - expected)
            .to_cols_array()
            .into_iter()
            .map(f64::abs)
            .fold(0., f64::max)
            / norm
    }
    #[test]
    fn small_clods_register_full_geometry_inertia_across_scales_and_repeated_moments() {
        let mut maximum = 0f64;
        for scale in [0.005, 0.02, 0.125, 0.5] {
            for tapered in [false, true] {
                let error = relative_error(&mut prism(scale, tapered), true);
                maximum = maximum.max(error);
                assert!(
                    error < 1e-5,
                    "scale={scale} tapered={tapered}: inertia error={error}"
                );
            }
        }
        let old = relative_error(&mut prism(0.02, false), false);
        println!("clod inertia relative error: old={old}, precise maximum={maximum}");
        assert!(
            old > 0.05,
            "old small-tensor negative control did not reproduce: {old}"
        );
    }
}
