//! Dimensional normal response of a reference material patch against a plane.
//!
//! This is an elastic/viscous interface model for calibration, not a complete
//! soil constitutive model. Patch area is a material reference area. The plane
//! is unbounded. Finite opposing footprints, shear, plasticity, cohesion, pore
//! pressure and live soil coupling are outside this module. The dynamics child
//! couples this interface in a two-body calibration solver. Nothing here
//! selects a Session policy, material default, save format or contact frequency.
use glam::{DQuat, DVec3};

/// Coupled body/patch calibration dynamics; not a Session or retained-soil policy.
pub mod dynamics;

const MAX_POLYGON_VERTICES: usize = 256;
const MAX_TRIANGLES: usize = 65_536;

/// A validated dimensional layer: pressure = modulus * compression / thickness,
/// with viscosity * normal compression rate / thickness before unilateral capping.
#[derive(Clone, Copy, Debug)]
pub struct NormalLayer {
    stiffness_density: f64,
    viscosity_density: f64,
    thickness: f64,
}
impl NormalLayer {
    /// Construct a specified layer. Modulus is a constrained normal modulus in
    /// Pa; thickness is metres; viscosity is Pa s. No soil values are inferred.
    pub fn new(modulus_pa: f64, thickness_m: f64, viscosity_pa_s: f64) -> Result<Self, String> {
        if !modulus_pa.is_finite()
            || modulus_pa <= 0.
            || !thickness_m.is_finite()
            || thickness_m <= 0.
            || !viscosity_pa_s.is_finite()
            || viscosity_pa_s < 0.
        {
            return Err("Invalid dimensional normal layer".into());
        }
        let result = Self {
            stiffness_density: modulus_pa / thickness_m,
            viscosity_density: viscosity_pa_s / thickness_m,
            thickness: thickness_m,
        };
        if !result.stiffness_density.is_finite()
            || result.stiffness_density <= 0.
            || !result.viscosity_density.is_finite()
            || (viscosity_pa_s > 0. && result.viscosity_density == 0.)
        {
            return Err("Normal layer coefficients overflow".into());
        }
        Ok(result)
    }
    /// Pressure/displacement slope, N/m³; a uniformly compressed patch has
    /// force/displacement stiffness equal to this slope times its area.
    pub fn stiffness_density_n_m3(self) -> f64 {
        self.stiffness_density
    }
    /// Pressure/velocity slope, N s/m³, independent of body mass and sampling.
    pub fn viscosity_density_ns_m3(self) -> f64 {
        self.viscosity_density
    }

    /// Evaluate actual current relative poses and velocities. Elastic energy is
    /// integrated over geometric compression; non-tensile pressure is integrated
    /// over its own active subset. No point-count or effective-mass factor occurs.
    pub fn evaluate(
        self,
        patch: &SurfacePatch,
        a: RigidMotion,
        b: RigidMotion,
        plane: Plane,
    ) -> Result<Response, String> {
        a.validate()?;
        b.validate()?;
        let normal = b.rotation * plane.normal;
        let delta = a.center_of_mass_m - b.center_of_mass_m;
        let reference = delta - b.rotation * plane.point;
        let mut geometric = Integral::default();
        let mut loaded = Integral::default();
        let mut maximum_pressure = 0f64;
        let mut maximum_compression = 0f64;
        for triangle in &patch.triangles {
            let mut original = Polygon::default();
            for &point in triangle {
                let arm = a.rotation * point;
                let gap = normal.dot(reference + arm);
                let velocity = normal.dot(
                    a.linear_velocity_m_s - b.linear_velocity_m_s
                        + a.angular_velocity_rad_s.cross(arm)
                        - b.angular_velocity_rad_s.cross(delta + arm),
                );
                if !gap.is_finite() || !velocity.is_finite() {
                    return Err("Nonfinite patch kinematics".into());
                }
                if !(self.stiffness_density * gap + self.viscosity_density * velocity).is_finite() {
                    return Err("Normal pressure overflow".into());
                }
                maximum_compression = maximum_compression.max(-gap);
                original.push(Sample {
                    point,
                    gap,
                    velocity,
                });
            }
            let contact = original.clip(|s| s.gap);
            geometric.add(contact.integrate(self));
            // Zero traction everywhere has no loaded area. Boundary zeroes on
            // an otherwise positive field do not change its integral.
            let pressure = if contact
                .samples()
                .iter()
                .any(|s| self.stiffness_density * s.gap + self.viscosity_density * s.velocity < 0.)
            {
                contact
                    .clip(|s| self.stiffness_density * s.gap + self.viscosity_density * s.velocity)
            } else {
                Polygon::default()
            };
            for s in pressure.samples() {
                maximum_pressure = maximum_pressure
                    .max(-self.stiffness_density * s.gap - self.viscosity_density * s.velocity);
            }
            loaded.add(pressure.integrate(self));
        }
        let force = normal * loaded.pressure;
        let moment = a.rotation * loaded.point_pressure;
        let torque_a = moment.cross(normal);
        let torque_b = -(delta * loaded.pressure + moment).cross(normal);
        let body_power = force.dot(a.linear_velocity_m_s) + torque_a.dot(a.angular_velocity_rad_s)
            - force.dot(b.linear_velocity_m_s)
            + torque_b.dot(b.angular_velocity_rad_s);
        let response = Response {
            geometric_area_m2: geometric.area,
            loaded_area_m2: loaded.area,
            elastic_energy_j: 0.5 * self.stiffness_density * geometric.gap_squared,
            elastic_energy_rate_w: self.stiffness_density * geometric.gap_velocity,
            force_on_a_n: force,
            force_on_b_n: -force,
            torque_on_a_nm: torque_a,
            torque_on_b_nm: torque_b,
            body_power_w: body_power,
            integrated_contact_power_w: loaded.pressure_velocity,
            // Include opening regions where tensile dashpot traction was capped.
            // Their stored energy still decreases even though applied pressure is zero.
            dissipation_w: self.viscosity_density * loaded.velocity_squared
                - self.stiffness_density * (geometric.gap_velocity - loaded.gap_velocity),
            maximum_pressure_pa: maximum_pressure,
            maximum_compressive_strain: maximum_compression / self.thickness,
        };
        if !response.finite() {
            return Err("Normal patch response overflow".into());
        }
        Ok(response)
    }
}

/// Body pose and twist, with patch coordinates relative to its center of mass.
/// Rotations must be unit quaternions; evaluation rejects invalid inputs.
#[derive(Clone, Copy, Debug)]
pub struct RigidMotion {
    pub center_of_mass_m: DVec3,
    pub rotation: DQuat,
    pub linear_velocity_m_s: DVec3,
    pub angular_velocity_rad_s: DVec3,
}
impl Default for RigidMotion {
    fn default() -> Self {
        Self {
            center_of_mass_m: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            linear_velocity_m_s: DVec3::ZERO,
            angular_velocity_rad_s: DVec3::ZERO,
        }
    }
}
impl RigidMotion {
    fn validate(self) -> Result<(), String> {
        if !self.center_of_mass_m.is_finite()
            || !self.rotation.is_finite()
            || (self.rotation.length_squared() - 1.).abs() > 1e-10
            || !self.linear_velocity_m_s.is_finite()
            || !self.angular_velocity_rad_s.is_finite()
        {
            return Err("Invalid patch body pose or velocity".into());
        }
        Ok(())
    }
}
/// Unbounded plane, attached to body B in its center-of-mass coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    point: DVec3,
    normal: DVec3,
}
impl Plane {
    /// The outward direction is normalized as part of defining the plane.
    pub fn new(point_in_b_m: DVec3, normal_in_b: DVec3) -> Result<Self, String> {
        let length = normal_in_b.length();
        if !point_in_b_m.is_finite() || !length.is_finite() || length <= 0. {
            return Err("Invalid response plane".into());
        }
        Ok(Self {
            point: point_in_b_m,
            normal: normal_in_b / length,
        })
    }
}
/// A convex, planar reference polygon, partitioned without duplicate material.
/// Triangles and area are private so refinement cannot change the domain.
#[derive(Clone, Debug)]
pub struct SurfacePatch {
    triangles: Vec<[DVec3; 3]>,
    area: f64,
}
impl SurfacePatch {
    /// Validate an ordered convex polygon (3–256 vertices). No hull, projection
    /// or vertex reduction silently substitutes for the supplied material area.
    pub fn new(vertices: &[DVec3]) -> Result<Self, String> {
        if !(3..=MAX_POLYGON_VERTICES).contains(&vertices.len())
            || vertices.iter().any(|p| !p.is_finite())
        {
            return Err("Invalid material patch vertices".into());
        }
        // Repeated winding can satisfy every convex half-plane test while
        // counting one material domain twice. Never admit repeated vertices.
        if vertices
            .iter()
            .enumerate()
            .any(|(i, p)| vertices[..i].contains(p))
        {
            return Err("Duplicate material patch vertices".into());
        }
        let origin = vertices[0];
        let scale = vertices
            .iter()
            .map(|p| (*p - origin).length())
            .fold(0f64, f64::max);
        let mut area_vector = DVec3::ZERO;
        for i in 1..vertices.len() - 1 {
            area_vector += (vertices[i] - origin).cross(vertices[i + 1] - origin);
        }
        let length = area_vector.length();
        if !scale.is_finite() || scale <= 0. || !length.is_finite() || length <= 0. {
            return Err("Degenerate material patch".into());
        }
        let normal = area_vector / length;
        if vertices
            .iter()
            .any(|p| (*p - origin).dot(normal).abs() > 1e-10 * scale)
        {
            return Err("Nonplanar material patch".into());
        }
        for i in 0..vertices.len() {
            let a = vertices[i];
            let edge = vertices[(i + 1) % vertices.len()] - a;
            if edge.length_squared() == 0.
                || vertices
                    .iter()
                    .any(|p| edge.cross(*p - a).dot(normal) < -1e-12 * scale * scale)
            {
                return Err("Nonconvex, overlapping or duplicate patch boundary".into());
            }
        }
        let mut triangles = Vec::with_capacity(vertices.len() - 2);
        let mut area = 0.;
        for i in 1..vertices.len() - 1 {
            let triangle = [origin, vertices[i], vertices[i + 1]];
            let a = triangle_area(triangle);
            if a > 0. {
                triangles.push(triangle);
                area += a;
            }
        }
        if !area.is_finite() || area <= 0. {
            return Err("Invalid patch area".into());
        }
        Ok(Self { triangles, area })
    }
    pub fn area_m2(&self) -> f64 {
        self.area
    }
    pub fn triangles(&self) -> usize {
        self.triangles.len()
    }
    /// Split each existing material triangle into four exact subdomains per
    /// level. Reject requests above 65,536 triangles rather than coarsening.
    pub fn refined(&self, levels: u32) -> Result<Self, String> {
        let multiplier = 4usize
            .checked_pow(levels)
            .ok_or("Patch refinement overflow")?;
        let count = self
            .triangles
            .len()
            .checked_mul(multiplier)
            .ok_or("Patch refinement overflow")?;
        if count > MAX_TRIANGLES {
            return Err("Patch refinement exceeds explicit storage bound".into());
        }
        let mut triangles = self.triangles.clone();
        for _ in 0..levels {
            let mut next = Vec::with_capacity(triangles.len() * 4);
            for [a, b, c] in triangles {
                let ab = (a + b) * 0.5;
                let bc = (b + c) * 0.5;
                let ca = (c + a) * 0.5;
                next.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
            }
            triangles = next;
        }
        Ok(Self {
            triangles,
            area: self.area,
        })
    }
}
/// Resultant wrenches, energy and power of the defined normal interface.
/// Signed numerical residuals are retained; no energy or dissipation is clipped.
#[derive(Default, Clone, Copy, Debug)]
pub struct Response {
    pub geometric_area_m2: f64,
    pub loaded_area_m2: f64,
    pub elastic_energy_j: f64,
    pub elastic_energy_rate_w: f64,
    pub force_on_a_n: DVec3,
    pub force_on_b_n: DVec3,
    pub torque_on_a_nm: DVec3,
    pub torque_on_b_nm: DVec3,
    pub body_power_w: f64,
    pub integrated_contact_power_w: f64,
    pub dissipation_w: f64,
    pub maximum_pressure_pa: f64,
    pub maximum_compressive_strain: f64,
}
impl Response {
    fn finite(self) -> bool {
        [
            self.geometric_area_m2,
            self.loaded_area_m2,
            self.elastic_energy_j,
            self.elastic_energy_rate_w,
            self.body_power_w,
            self.integrated_contact_power_w,
            self.dissipation_w,
            self.maximum_pressure_pa,
            self.maximum_compressive_strain,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.force_on_a_n.is_finite()
            && self.force_on_b_n.is_finite()
            && self.torque_on_a_nm.is_finite()
            && self.torque_on_b_nm.is_finite()
    }
}
#[derive(Default, Clone, Copy)]
struct Sample {
    point: DVec3,
    gap: f64,
    velocity: f64,
}
impl Sample {
    fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            point: self.point * (1. - t) + other.point * t,
            gap: self.gap * (1. - t) + other.gap * t,
            velocity: self.velocity * (1. - t) + other.velocity * t,
        }
    }
}
// A triangle clipped by two half-planes has at most five vertices. Six slots
// include a defensive bound; evaluation allocates no per-triangle heap storage.
#[derive(Default)]
struct Polygon {
    samples: [Sample; 6],
    len: usize,
}
impl Polygon {
    fn push(&mut self, sample: Sample) {
        assert!(self.len < 6);
        self.samples[self.len] = sample;
        self.len += 1;
    }
    fn samples(&self) -> &[Sample] {
        &self.samples[..self.len]
    }
    fn clip(&self, signed: impl Fn(Sample) -> f64) -> Self {
        let mut result = Self::default();
        if self.len == 0 {
            return result;
        }
        let mut previous = self.samples[self.len - 1];
        let mut pa = signed(previous);
        for &current in self.samples() {
            let ca = signed(current);
            if (pa <= 0.) != (ca <= 0.) {
                let scale = pa.abs().max(ca.abs());
                result.push(previous.lerp(current, (pa / scale) / (pa / scale - ca / scale)));
            }
            if ca <= 0. {
                result.push(current);
            }
            previous = current;
            pa = ca;
        }
        result
    }
    fn integrate(&self, layer: NormalLayer) -> Integral {
        let mut result = Integral::default();
        for i in 1..self.len.saturating_sub(1) {
            let s = [self.samples[0], self.samples[i], self.samples[i + 1]];
            let points = s.map(|s| s.point);
            let area = triangle_area(points);
            let g = s.map(|s| s.gap);
            let v = s.map(|s| s.velocity);
            let q =
                s.map(|s| -layer.stiffness_density * s.gap - layer.viscosity_density * s.velocity);
            let product = |a: [f64; 3], b: [f64; 3]| {
                area / 12.
                    * (a.iter().sum::<f64>() * b.iter().sum::<f64>()
                        + a.into_iter().zip(b).map(|(a, b)| a * b).sum::<f64>())
            };
            let sum = points.into_iter().sum::<DVec3>();
            let sum_q = q.iter().sum::<f64>();
            result.area += area;
            result.gap_squared += product(g, g);
            result.gap_velocity += product(g, v);
            result.velocity_squared += product(v, v);
            result.pressure += area * sum_q / 3.;
            result.point_pressure += (sum * sum_q
                + points.into_iter().zip(q).map(|(p, q)| p * q).sum::<DVec3>())
                * (area / 12.);
            result.pressure_velocity += product(q, v);
        }
        result
    }
}
#[derive(Default)]
struct Integral {
    area: f64,
    gap_squared: f64,
    gap_velocity: f64,
    velocity_squared: f64,
    pressure: f64,
    point_pressure: DVec3,
    pressure_velocity: f64,
}
impl Integral {
    fn add(&mut self, b: Self) {
        self.area += b.area;
        self.gap_squared += b.gap_squared;
        self.gap_velocity += b.gap_velocity;
        self.velocity_squared += b.velocity_squared;
        self.pressure += b.pressure;
        self.point_pressure += b.point_pressure;
        self.pressure_velocity += b.pressure_velocity;
    }
}
fn triangle_area(p: [DVec3; 3]) -> f64 {
    0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
}

#[cfg(test)]
mod tests;
