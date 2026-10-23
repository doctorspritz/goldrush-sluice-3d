//! Spatial material moments and finite rigid parcels. This is the momentum
//! custody foundation for active soil, not a soil contact/capture criterion.
use crate::{session::FallingLoad, Material};
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

pub(crate) fn outer(v: DVec3) -> DMat3 {
    DMat3::from_cols(v * v.x, v * v.y, v * v.z)
}
fn trace(a: DMat3) -> f64 {
    a.x_axis.x + a.y_axis.y + a.z_axis.z
}

/// Central second mass moment C = integral(r r^T dm), in a declared frame.
/// Inertia is tr(C) Identity - C; storing C makes the inertia triangle
/// inequalities explicit, as well as positive mass and covariance.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct MassMoments {
    pub mass_kg: f64,
    pub center: DVec3,
    pub second: DMat3,
}
impl MassMoments {
    pub fn cuboid(mass_kg: f64, center: DVec3, half: DVec3, rotation: DQuat) -> Self {
        let r = DMat3::from_quat(rotation);
        Self {
            mass_kg,
            center,
            second: r * DMat3::from_diagonal(half * half * (mass_kg / 3.)) * r.transpose(),
        }
    }
    pub fn inertia(self) -> DMat3 {
        DMat3::IDENTITY * trace(self.second) - self.second
    }
    /// Equivalent uniform ellipsoid for read-only parcel visualization. These
    /// axes reproduce C; they are not a collision surface or packing model.
    pub fn visual_ellipsoid(self) -> (DQuat, DVec3) {
        let mut a = self.second;
        let mut basis = DMat3::IDENTITY;
        for _ in 0..24 {
            let values = a.to_cols_array_2d();
            let mut pair = (0, 1);
            let mut largest = 0.;
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                if values[i][j].abs() > largest {
                    largest = values[i][j].abs();
                    pair = (i, j);
                }
            }
            if largest <= 1e-13 * trace(a).abs().max(1e-30) {
                break;
            }
            let (i, j) = pair;
            let angle = 0.5 * (2. * values[i][j]).atan2(values[j][j] - values[i][i]);
            let (sn, cs) = angle.sin_cos();
            let mut r = DMat3::IDENTITY.to_cols_array_2d();
            r[i][i] = cs;
            r[j][j] = cs;
            r[i][j] = -sn;
            r[j][i] = sn;
            let rot = DMat3::from_cols_array_2d(&r);
            a = rot.transpose() * a * rot;
            basis *= rot;
        }
        let d = DVec3::new(a.x_axis.x, a.y_axis.y, a.z_axis.z).max(DVec3::ZERO);
        (
            DQuat::from_mat3(&basis).normalize(),
            DVec3::new(d.x.sqrt(), d.y.sqrt(), d.z.sqrt()) * (5. / self.mass_kg).sqrt(),
        )
    }

    pub fn combine(self, rhs: Self) -> Self {
        let mass = self.mass_kg + rhs.mass_kg;
        if mass <= 0. {
            return Self {
                mass_kg: 0.,
                center: DVec3::ZERO,
                second: DMat3::ZERO,
            };
        }
        let center = (self.center * self.mass_kg + rhs.center * rhs.mass_kg) / mass;
        Self {
            mass_kg: mass,
            center,
            second: self.second
                + rhs.second
                + outer(self.center - center) * self.mass_kg
                + outer(rhs.center - center) * rhs.mass_kg,
        }
    }
    pub fn scaled(self, fraction: f64) -> Self {
        Self {
            mass_kg: self.mass_kg * fraction,
            center: self.center,
            second: self.second * fraction,
        }
    }
    pub fn transformed(self, translation: DVec3, rotation: DQuat) -> Self {
        let r = DMat3::from_quat(rotation);
        Self {
            mass_kg: self.mass_kg,
            center: translation + rotation * self.center,
            second: r * self.second * r.transpose(),
        }
    }
    pub fn valid(self) -> bool {
        if !self.mass_kg.is_finite()
            || !(0. ..=10_000.).contains(&self.mass_kg)
            || !self.center.is_finite()
            || self.center.length() > 1000.
            || !self.second.is_finite()
        {
            return false;
        }
        let scale = self
            .second
            .to_cols_array()
            .into_iter()
            .map(f64::abs)
            .fold(0., f64::max);
        if scale > 1e8 {
            return false;
        }
        if scale == 0. {
            return true;
        }
        // Normalize first: an absolute epsilon admits nonphysical negative
        // covariance when the parcel is smaller than a visible clump.
        if self.mass_kg == 0. {
            return false;
        }
        let a = (self.second / scale).to_cols_array_2d();
        let eps = 1e-11;
        for i in 0..3 {
            if a[i][i] < -eps {
                return false;
            }
            for j in 0..3 {
                if (a[i][j] - a[j][i]).abs() > eps {
                    return false;
                }
                if a[i][i] * a[j][j] - a[i][j] * a[j][i] < -eps {
                    return false;
                }
            }
        }
        (self.second / scale).determinant() >= -eps
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RigidState {
    /// Canonical double-precision COM pose/velocity. FallingLoad's legacy
    /// f32 positions are checked render/terrain-query projections of these.
    pub center: DVec3,
    pub velocity: DVec3,
    pub rotation: DQuat,
    pub spin_nms: DVec3,
    /// Body-frame moments about COM; center must be zero here.
    pub moments: MassMoments,
}
impl RigidState {
    pub fn world_inertia(&self) -> DMat3 {
        let r = DMat3::from_quat(self.rotation);
        r * self.moments.inertia() * r.transpose()
    }
    pub fn angular_velocity(&self) -> DVec3 {
        self.world_inertia().inverse() * self.spin_nms
    }
    pub fn kinetic_j(&self) -> f64 {
        0.5 * self.moments.mass_kg * self.velocity.length_squared()
            + 0.5 * self.spin_nms.dot(self.angular_velocity())
    }
    pub fn linear_momentum(&self) -> DVec3 {
        self.velocity * self.moments.mass_kg
    }
    pub fn angular_momentum(&self) -> DVec3 {
        self.center.cross(self.linear_momentum()) + self.spin_nms
    }
    pub fn valid_for(&self, material: Material) -> bool {
        self.moments.valid()
            && self.moments.mass_kg > 0.
            && self.moments.center == DVec3::ZERO
            && (self.moments.mass_kg - material.wet_mass_kg()).abs()
                <= 1e-9 * self.moments.mass_kg.max(1.)
            && self.center.is_finite()
            && self.center.length() < 1000.
            && self.velocity.is_finite()
            && self.velocity.length() < 100.
            && self.rotation.is_finite()
            && (self.rotation.length_squared() - 1.).abs() < 1e-9
            && self.spin_nms.is_finite()
            && self.moments.inertia().determinant() > 0.
            && self.angular_velocity().is_finite()
            && self.angular_velocity().length() < 100.
    }
    /// Uniform gravity gives exact COM ballistic motion. World spin angular
    /// momentum is constant; a midpoint rotation update integrates Euler's
    /// torque-free motion. Bounded angular increments retain physical time.
    pub fn advance(&mut self, gravity: DVec3, dt: f64) -> Result<(), String> {
        if !dt.is_finite() || !(0. ..=0.05).contains(&dt) || !gravity.is_finite() {
            return Err("Invalid rigid parcel timestep/gravity".into());
        }
        if !self.center.is_finite()
            || !self.velocity.is_finite()
            || !self.rotation.is_finite()
            || !self.spin_nms.is_finite()
            || !self.moments.valid()
            || self.moments.inertia().determinant() <= 0.
            || (self.rotation.length_squared() - 1.).abs() > 1e-9
        {
            return Err("Invalid rigid parcel state".into());
        }
        if dt == 0. {
            return Ok(());
        }
        let n = (self.angular_velocity().length() * dt / 0.05)
            .ceil()
            .max(1.) as usize;
        if n > 512 {
            return Err("Rigid parcel angular step exceeds supported range".into());
        }
        let mut candidate = self.clone();
        candidate.center += self.velocity * dt + gravity * (0.5 * dt * dt);
        candidate.velocity += gravity * dt;
        let h = dt / n as f64;
        for _ in 0..n {
            let rotation = candidate.rotation;
            let mut omega = candidate.angular_velocity();
            for _ in 0..4 {
                candidate.rotation =
                    (DQuat::from_scaled_axis(omega * (0.5 * h)) * rotation).normalize();
                omega = candidate.angular_velocity();
            }
            candidate.rotation = (DQuat::from_scaled_axis(omega * h) * rotation).normalize();
        }
        if !candidate.center.is_finite()
            || !candidate.velocity.is_finite()
            || !candidate.rotation.is_finite()
            || !candidate.angular_velocity().is_finite()
            || candidate.angular_velocity().length() >= 100.
        {
            return Err("Non-finite rigid parcel result".into());
        }
        *self = candidate;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RigidParcel {
    pub material: Material,
    pub state: RigidState,
}
impl RigidParcel {
    pub fn cuboid(
        material: Material,
        center: DVec3,
        half: DVec3,
        rotation: DQuat,
        velocity: DVec3,
        omega: DVec3,
    ) -> Result<Self, String> {
        if !material.valid()
            || material.wet_mass_kg() <= 0.
            || !half.is_finite()
            || half.min_element() <= 0.
            || half.max_element() > 10.
        {
            return Err("Invalid rigid material or shape".into());
        }
        let moments =
            MassMoments::cuboid(material.wet_mass_kg(), DVec3::ZERO, half, DQuat::IDENTITY);
        let r = DMat3::from_quat(rotation);
        let spin = r * moments.inertia() * r.transpose() * omega;
        let out = Self {
            material,
            state: RigidState {
                center,
                velocity,
                rotation,
                spin_nms: spin,
                moments,
            },
        };
        if !out.valid() {
            return Err("Invalid rigid material state".into());
        }
        Ok(out)
    }
    pub fn valid(&self) -> bool {
        self.material.valid()
            && self.material.wet_mass_kg() > 0.
            && self.state.valid_for(self.material)
    }
    pub(crate) fn fraction(&self, f: f64) -> Self {
        let mut out = self.clone();
        out.material = self.material.scaled(f);
        out.state.moments = out.state.moments.scaled(f);
        out.state.spin_nms *= f;
        out
    }
    pub fn into_falling(self) -> FallingLoad {
        FallingLoad {
            position: self.state.center.as_vec3(),
            velocity: self.state.velocity.as_vec3(),
            material: self.material,
            rigid: Some(self.state),
        }
    }
}

impl FallingLoad {
    pub fn valid_rigid_state(&self) -> bool {
        self.rigid.as_ref().is_none_or(|r| {
            r.valid_for(self.material)
                && r.center.as_vec3() == self.position
                && r.velocity.as_vec3() == self.velocity
        })
    }
    pub fn rigid_parcel(&self) -> Option<RigidParcel> {
        self.rigid.as_ref().map(|s| RigidParcel {
            material: self.material,
            state: s.clone(),
        })
    }
    pub(crate) fn advance_rigid(&mut self, dt: f32) -> Result<bool, String> {
        if !self.valid_rigid_state() {
            return Err("Invalid rigid material/projection before motion".into());
        }
        let Some(r) = &mut self.rigid else {
            return Ok(false);
        };
        r.advance(DVec3::new(0., -9.81, 0.), dt as f64)?;
        self.position = r.center.as_vec3();
        self.velocity = r.velocity.as_vec3();
        Ok(true)
    }
}

/// Covectors/mass matrices have ten generalized coordinates: free base 6,
/// then four hinges. An unpivoted SPD Cholesky rejects invalid physical mass;
/// no diagonal regularization or velocity rescaling conceals a bad exchange.
pub(crate) fn solve_spd(a: &[[f64; 10]; 10], b: &[f64; 10]) -> Result<[f64; 10], String> {
    let mut l = [[0.; 10]; 10];
    for i in 0..10 {
        for j in 0..=i {
            let mut x = a[i][j];
            for (left, right) in l[i][..j].iter().zip(&l[j][..j]) {
                x -= *left * *right;
            }
            if i == j {
                if !x.is_finite() || x <= 1e-12 {
                    return Err("Payload mass matrix is not positive definite".into());
                }
                l[i][j] = x.sqrt();
            } else {
                l[i][j] = x / l[j][j];
            }
        }
    }
    let mut y = [0.; 10];
    for i in 0..10 {
        let mut x = b[i];
        for k in 0..i {
            x -= l[i][k] * y[k];
        }
        y[i] = x / l[i][i];
    }
    let mut out = [0.; 10];
    for i in (0..10).rev() {
        let mut x = y[i];
        for k in i + 1..10 {
            x -= l[k][i] * out[k];
        }
        out[i] = x / l[i][i];
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("Non-finite payload velocity".into());
    }
    Ok(out)
}
pub(crate) fn multiply(a: &[[f64; 10]; 10], v: &[f64; 10]) -> [f64; 10] {
    std::array::from_fn(|i| (0..10).map(|j| a[i][j] * v[j]).sum())
}
pub(crate) fn quadratic(a: &[[f64; 10]; 10], v: &[f64; 10]) -> f64 {
    let p = multiply(a, v);
    0.5 * (0..10).map(|i| v[i] * p[i]).sum::<f64>()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExchangeReport {
    pub capture: bool,
    pub transferred_kg: f64,
    pub kinetic_before_j: f64,
    pub kinetic_after_j: f64,
    /// Independently calculated projection loss, not an energy residual.
    pub inelastic_heat_j: f64,
    pub kinetic_residual_j: f64,
    pub linear_residual_ns: DVec3,
    pub angular_residual_nms: DVec3,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExchangeLedger {
    pub events: u64,
    pub captured_kg: f64,
    pub released_kg: f64,
    pub inelastic_heat_j: f64,
    pub last: ExchangeReport,
}
impl ExchangeLedger {
    pub fn valid(&self) -> bool {
        self.events < 1_000_000_000_000
            && [
                self.captured_kg,
                self.released_kg,
                self.inelastic_heat_j,
                self.last.transferred_kg,
                self.last.kinetic_before_j,
                self.last.kinetic_after_j,
                self.last.inelastic_heat_j,
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.)
            && self.last.kinetic_residual_j.is_finite()
            && self.last.linear_residual_ns.is_finite()
            && self.last.angular_residual_nms.is_finite()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BoundaryReceipt {
    pub parcels: u64,
    pub impulse_ns: DVec3,
    pub angular_impulse_nms: DVec3,
    pub received_kinetic_j: f64,
}
impl BoundaryReceipt {
    pub fn receive(&mut self, parcel: &RigidParcel) {
        self.parcels += 1;
        self.impulse_ns += parcel.state.linear_momentum();
        self.angular_impulse_nms += parcel.state.angular_momentum();
        self.received_kinetic_j += parcel.state.kinetic_j();
    }
    pub fn valid(&self) -> bool {
        self.parcels < 1_000_000_000_000
            && self.impulse_ns.is_finite()
            && self.angular_impulse_nms.is_finite()
            && self.received_kinetic_j.is_finite()
            && self.received_kinetic_j >= 0.
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MaterialMotion {
    pub terrain: BoundaryReceipt,
    pub hopper: BoundaryReceipt,
}
impl MaterialMotion {
    pub fn valid(&self) -> bool {
        self.terrain.valid() && self.hopper.valid()
    }
}
