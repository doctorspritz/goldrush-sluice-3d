//! Implicit two-body normal-material calibration dynamics.
//!
//! Principal-axis inertia, translation, rotation, gravity and applied wrenches
//! are coupled. Cayley rotations and endpoint-gap secant pressure pair discrete
//! work with the same material potential. No velocity or energy correction is
//! applied after the solve. This is not the live soil, joint/friction solver,
//! continuous collision detection or a calibrated soil policy.
use super::{NormalLayer, Plane, RigidMotion, SurfacePatch};
use glam::{DQuat, DVec3};
mod quadrature;

#[derive(Clone, Copy, Debug)]
struct Mass {
    kg: f64,
    inertia: DVec3,
    length: f64,
}
/// Body coordinates (patch and plane included) are principal-inertia axes at CoM.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    motion: RigidMotion,
    mass: Option<Mass>,
}
impl Body {
    /// Admit a dynamic body with specified positive principal moments, kg m².
    pub fn dynamic(
        motion: RigidMotion,
        mass_kg: f64,
        principal_inertia_kg_m2: DVec3,
    ) -> Result<Self, String> {
        motion.validate()?;
        let i = principal_inertia_kg_m2;
        if !mass_kg.is_finite()
            || mass_kg <= 0.
            || !i.is_finite()
            || i.min_element() <= 0.
            || 2. * i.max_element() > i.element_sum() * (1. + 1e-12)
        {
            return Err("Invalid physical body mass or principal inertia".into());
        }
        let length = (i.max_element() / mass_kg).sqrt();
        if !length.is_finite() || length <= 0. {
            return Err("Unrepresentable body inertia scale".into());
        }
        Ok(Self {
            motion,
            mass: Some(Mass {
                kg: mass_kg,
                inertia: i,
                length,
            }),
        })
    }
    /// A stationary body receives reported support reactions but no kinetic work.
    pub fn fixed(motion: RigidMotion) -> Result<Self, String> {
        motion.validate()?;
        if motion.linear_velocity_m_s != DVec3::ZERO || motion.angular_velocity_rad_s != DVec3::ZERO
        {
            return Err("Prescribed moving boundary work is not admitted".into());
        }
        Ok(Self { motion, mass: None })
    }
    pub fn motion(self) -> RigidMotion {
        self.motion
    }
    pub fn mass_kg(self) -> Option<f64> {
        self.mass.map(|m| m.kg)
    }
    pub fn kinetic_energy_j(self) -> f64 {
        self.mass.map_or(0., |m| {
            let omega = self.motion.rotation.conjugate() * self.motion.angular_velocity_rad_s;
            0.5 * m.kg * self.motion.linear_velocity_m_s.length_squared()
                + 0.5 * (m.inertia * omega).dot(omega)
        })
    }
    pub fn linear_momentum_ns(self) -> DVec3 {
        self.mass
            .map_or(DVec3::ZERO, |m| m.kg * self.motion.linear_velocity_m_s)
    }
    pub fn spin_momentum_nms(self) -> DVec3 {
        self.mass.map_or(DVec3::ZERO, |m| {
            self.motion.rotation
                * (m.inertia
                    * (self.motion.rotation.conjugate() * self.motion.angular_velocity_rad_s))
        })
    }
    pub fn angular_momentum_nms(self) -> DVec3 {
        self.spin_momentum_nms()
            + self
                .motion
                .center_of_mass_m
                .cross(self.linear_momentum_ns())
    }
}
/// Constant world loads over this step. Gravity has a matching potential.
#[derive(Default, Clone, Copy, Debug)]
pub struct Loads {
    pub gravity_m_s2: DVec3,
    pub forces_n: [DVec3; 2],
    pub torques_nm: [DVec3; 2],
}
/// Numerical admission/convergence settings, not material properties.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub maximum_iterations: usize,
    pub velocity_residual_tolerance: f64,
    pub quadrature_absolute_force_tolerance_n: f64,
    pub quadrature_relative_tolerance: f64,
    pub maximum_quadrature_depth: u32,
    pub maximum_rotation_rad: f64,
    /// Absolute numerical closure tolerance; never a velocity/energy correction.
    pub maximum_abs_work_residual_j: f64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            maximum_iterations: 24,
            velocity_residual_tolerance: 1e-10,
            quadrature_absolute_force_tolerance_n: 1e-8,
            quadrature_relative_tolerance: 1e-9,
            maximum_quadrature_depth: 8,
            maximum_rotation_rad: 0.25,
            maximum_abs_work_residual_j: 1e-8,
        }
    }
}
impl Settings {
    fn validate(self) -> Result<(), String> {
        if !(1..=64).contains(&self.maximum_iterations)
            || self.maximum_quadrature_depth > 10
            || [
                self.velocity_residual_tolerance,
                self.quadrature_absolute_force_tolerance_n,
                self.quadrature_relative_tolerance,
                self.maximum_rotation_rad,
                self.maximum_abs_work_residual_j,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.)
            || self.maximum_rotation_rad > 1.
        {
            return Err("Invalid patch solve settings".into());
        }
        Ok(())
    }
}
/// Accepted dynamic step, including raw physical work and numerical residuals.
#[derive(Clone, Debug)]
pub struct Step {
    pub bodies: [Body; 2],
    pub iterations: usize,
    pub velocity_residual: f64,
    pub mean_contact_forces_n: [DVec3; 2],
    pub mean_contact_torques_nm: [DVec3; 2],
    pub mechanical_before_j: f64,
    pub mechanical_after_j: f64,
    pub external_work_j: f64,
    pub damping_loss_j: f64,
    pub energy_residual_j: f64,
    pub material_work_residual_j: f64,
    pub contact_work_residual_j: f64,
    /// Global coarse/fine estimate for scalar pressure force and scaled moment.
    /// This is an integration error estimate, not a rigorous physical bound.
    pub estimated_force_moment_error_n: f64,
    pub linear_momentum_residual_ns: DVec3,
    pub angular_momentum_residual_nms: DVec3,
    /// Samples in the accepted endpoint force quadrature, not total Newton cost.
    pub accepted_quadrature_samples: usize,
    pub accepted_quadrature_depth: u32,
}
struct Context<'a> {
    layer: NormalLayer,
    patch: &'a SurfacePatch,
    plane: Plane,
    before: [Body; 2],
    loads: Loads,
    h: f64,
    settings: Settings,
    n: usize,
}
struct Trial {
    bodies: [Body; 2],
    residual: [f64; 12],
    norm: f64,
    contact: quadrature::Contact,
}
fn triple(v: &[f64], i: usize) -> DVec3 {
    DVec3::new(v[i], v[i + 1], v[i + 2])
}
fn put(v: &mut [f64], i: usize, x: DVec3) {
    v[i..i + 3].copy_from_slice(&x.to_array());
}
impl Context<'_> {
    fn decode(&self, z: &[f64; 12]) -> Result<[Body; 2], String> {
        let mut after = self.before;
        let mut index = 0;
        for (slot, before) in self.before.iter().enumerate() {
            if let Some(m) = before.mass {
                let v = triple(z, index);
                let w = triple(z, index + 3) / m.length;
                let r = w * self.h;
                let angle = 2. * (0.5 * r.length()).atan();
                if !v.is_finite()
                    || !w.is_finite()
                    || !angle.is_finite()
                    || angle > self.settings.maximum_rotation_rad
                {
                    return Err("Patch rotation needs a smaller timestep".into());
                }
                let cayley = DQuat::from_xyzw(r.x * 0.5, r.y * 0.5, r.z * 0.5, 1.).normalize();
                let rotation = before.motion.rotation * cayley;
                let old_l = m.inertia
                    * (before.motion.rotation.conjugate() * before.motion.angular_velocity_rad_s);
                let new_l = 2. * m.inertia * w - old_l;
                after[slot].motion = RigidMotion {
                    center_of_mass_m: before.motion.center_of_mass_m + self.h * v,
                    rotation,
                    linear_velocity_m_s: 2. * v - before.motion.linear_velocity_m_s,
                    angular_velocity_rad_s: rotation * (new_l / m.inertia),
                };
                after[slot].motion.validate()?;
                index += 6;
            }
        }
        Ok(after)
    }
    fn trial(&self, z: &[f64; 12]) -> Result<Trial, String> {
        let after = self.decode(z)?;
        let contact = quadrature::integrate(
            self.layer,
            self.patch,
            self.plane,
            self.before.map(|b| b.motion),
            after.map(|b| b.motion),
            self.h,
            self.settings,
        )?;
        let mut residual = [0.; 12];
        let mut index = 0;
        for (i, &b) in after.iter().enumerate() {
            if let Some(m) = self.before[i].mass {
                let a = self.before[i];
                let dv = b.motion.linear_velocity_m_s
                    - a.motion.linear_velocity_m_s
                    - self.h
                        * (self.loads.gravity_m_s2
                            + (self.loads.forces_n[i] + contact.forces[i]) / m.kg);
                let dh = b.spin_momentum_nms()
                    - a.spin_momentum_nms()
                    - self.h * (self.loads.torques_nm[i] + contact.torques[i]);
                let dw = (a.motion.rotation.conjugate() * dh) / m.inertia * m.length;
                put(&mut residual, index, dv);
                put(&mut residual, index + 3, dw);
                index += 6;
            }
        }
        let norm = residual[..self.n]
            .iter()
            .map(|v| v.abs())
            .fold(0f64, f64::max);
        if !norm.is_finite() {
            return Err("Nonfinite patch dynamics residual".into());
        }
        Ok(Trial {
            bodies: after,
            residual,
            norm,
            contact,
        })
    }
    fn finish(&self, trial: Trial, iterations: usize) -> Result<Step, String> {
        let motion_zero = |b: Body| RigidMotion {
            linear_velocity_m_s: DVec3::ZERO,
            angular_velocity_rad_s: DVec3::ZERO,
            ..b.motion
        };
        let u0 = self
            .layer
            .evaluate(
                self.patch,
                motion_zero(self.before[0]),
                motion_zero(self.before[1]),
                self.plane,
            )?
            .elastic_energy_j;
        let u1 = self
            .layer
            .evaluate(
                self.patch,
                motion_zero(trial.bodies[0]),
                motion_zero(trial.bodies[1]),
                self.plane,
            )?
            .elastic_energy_j;
        let energy = |b: Body| {
            b.kinetic_energy_j()
                - b.mass.map_or(0., |m| {
                    m.kg * self.loads.gravity_m_s2.dot(b.motion.center_of_mass_m)
                })
        };
        let before = u0 + self.before.into_iter().map(energy).sum::<f64>();
        let after = u1 + trial.bodies.into_iter().map(energy).sum::<f64>();
        let mut work = 0.;
        let mut contact_work = 0.;
        let mut linear = DVec3::ZERO;
        let mut angular = DVec3::ZERO;
        for i in 0..2 {
            let a = self.before[i];
            let b = trial.bodies[i];
            let mid = (a.motion.center_of_mass_m + b.motion.center_of_mass_m) * 0.5;
            if let Some(m) = a.mass {
                let dx = b.motion.center_of_mass_m - a.motion.center_of_mass_m;
                // r is recovered from the accepted midpoint angular momenta.
                let l0 = a.motion.rotation.conjugate() * a.spin_momentum_nms();
                let l1 = b.motion.rotation.conjugate() * b.spin_momentum_nms();
                let rotation_increment =
                    a.motion.rotation * ((l0 + l1) / (2. * m.inertia) * self.h);
                work += self.loads.forces_n[i].dot(dx)
                    + self.loads.torques_nm[i].dot(rotation_increment);
                contact_work += trial.contact.forces[i].dot(dx)
                    + trial.contact.torques[i].dot(rotation_increment);
                linear += b.linear_momentum_ns()
                    - a.linear_momentum_ns()
                    - self.h * (m.kg * self.loads.gravity_m_s2 + self.loads.forces_n[i]);
                angular += b.angular_momentum_nms()
                    - a.angular_momentum_nms()
                    - self.h
                        * (mid.cross(m.kg * self.loads.gravity_m_s2 + self.loads.forces_n[i])
                            + self.loads.torques_nm[i]);
            } else {
                linear += self.h * trial.contact.forces[i];
                angular += self.h * (mid.cross(trial.contact.forces[i]) + trial.contact.torques[i]);
            }
        }
        let material = (u1 - u0) + trial.contact.elastic_work;
        let result = Step {
            bodies: trial.bodies,
            iterations,
            velocity_residual: trial.norm,
            mean_contact_forces_n: trial.contact.forces,
            mean_contact_torques_nm: trial.contact.torques,
            mechanical_before_j: before,
            mechanical_after_j: after,
            external_work_j: work,
            damping_loss_j: trial.contact.loss,
            energy_residual_j: after - before - work + trial.contact.loss,
            material_work_residual_j: material,
            contact_work_residual_j: contact_work - trial.contact.normal_work,
            estimated_force_moment_error_n: trial.contact.estimated_force_moment_error_n,
            linear_momentum_residual_ns: linear,
            angular_momentum_residual_nms: angular,
            accepted_quadrature_samples: trial.contact.samples,
            accepted_quadrature_depth: trial.contact.depth,
        };
        if ![
            before,
            after,
            work,
            result.damping_loss_j,
            result.energy_residual_j,
            result.contact_work_residual_j,
            result.estimated_force_moment_error_n,
            material,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err("Nonfinite patch work audit".into());
        }
        let tolerance = self.settings.maximum_abs_work_residual_j;
        if result.energy_residual_j.abs() > tolerance
            || result.material_work_residual_j.abs() > tolerance
            || result.contact_work_residual_j.abs() > tolerance
            || result.damping_loss_j < -tolerance
        {
            return Err(format!("Patch work audit did not close: energy={}J material={}J contact={}J loss={}J tolerance={tolerance}J",
                result.energy_residual_j,result.material_work_residual_j,result.contact_work_residual_j,result.damping_loss_j));
        }
        Ok(result)
    }
}
/// Solve both bodies together. Inputs are immutable values; failed convergence,
/// quadrature or timestep admission returns no partially advanced state.
pub fn step_pair(
    layer: NormalLayer,
    patch: &SurfacePatch,
    plane: Plane,
    bodies: [Body; 2],
    loads: Loads,
    dt_s: f64,
    settings: Settings,
) -> Result<Step, String> {
    settings.validate()?;
    if !dt_s.is_finite()
        || dt_s <= 0.
        || !loads.gravity_m_s2.is_finite()
        || loads
            .forces_n
            .iter()
            .chain(loads.torques_nm.iter())
            .any(|v| !v.is_finite())
    {
        return Err("Invalid patch timestep or loads".into());
    }
    let mut n = 0;
    let mut z = [0.; 12];
    for (i, body) in bodies.iter().enumerate() {
        body.motion.validate()?;
        if let Some(m) = body.mass {
            put(
                &mut z,
                n,
                body.motion.linear_velocity_m_s
                    + 0.5 * dt_s * (loads.gravity_m_s2 + loads.forces_n[i] / m.kg),
            );
            put(
                &mut z,
                n + 3,
                (body.motion.rotation.conjugate() * body.motion.angular_velocity_rad_s) * m.length,
            );
            n += 6;
        } else if loads.forces_n[i] != DVec3::ZERO || loads.torques_nm[i] != DVec3::ZERO {
            return Err("Applied loads on a fixed calibration body are not admitted".into());
        }
    }
    if n == 0 {
        return Err("Patch dynamics needs a dynamic body".into());
    }
    let context = Context {
        layer,
        patch,
        plane,
        before: bodies,
        loads,
        h: dt_s,
        settings,
        n,
    };
    let mut current = context.trial(&z)?;
    for iteration in 0..settings.maximum_iterations {
        if current.norm <= settings.velocity_residual_tolerance {
            return context.finish(current, iteration);
        }
        let mut jacobian = [[0.; 12]; 12];
        for column in 0..n {
            let epsilon = 1e-6 * z[column].abs().max(1.);
            let mut plus = z;
            let mut minus = z;
            plus[column] += epsilon;
            minus[column] -= epsilon;
            let a = context.trial(&plus)?;
            let b = context.trial(&minus)?;
            for (row, j) in jacobian.iter_mut().enumerate().take(n) {
                j[column] = (a.residual[row] - b.residual[row]) / (2. * epsilon);
            }
        }
        let rhs = current.residual.map(|r| -r);
        let delta = linear_solve(jacobian, rhs, n)?;
        let mut accepted = None;
        let mut scale = 1.;
        for _ in 0..18 {
            let mut candidate = z;
            for i in 0..n {
                candidate[i] += scale * delta[i];
            }
            if let Ok(trial) = context.trial(&candidate) {
                if trial.norm < current.norm {
                    accepted = Some((candidate, trial));
                    break;
                }
            }
            scale *= 0.5;
        }
        let (next, trial) = accepted.ok_or_else(|| {
            format!(
                "Patch Newton line search failed at residual {}",
                current.norm
            )
        })?;
        z = next;
        current = trial;
    }
    if current.norm <= settings.velocity_residual_tolerance {
        context.finish(current, settings.maximum_iterations)
    } else {
        Err(format!(
            "Patch Newton solve did not converge: {}",
            current.norm
        ))
    }
}
fn linear_solve(mut a: [[f64; 12]; 12], mut b: [f64; 12], n: usize) -> Result<[f64; 12], String> {
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))
            .unwrap();
        a.swap(column, pivot);
        b.swap(column, pivot);
        if !a[column][column].is_finite() || a[column][column].abs() < 1e-14 {
            return Err("Singular patch Newton matrix".into());
        }
        let pivot_row = a[column];
        for row in column + 1..n {
            let factor = a[row][column] / pivot_row[column];
            for (value, pivot) in a[row][column..n].iter_mut().zip(&pivot_row[column..n]) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.; 12];
    for row in (0..n).rev() {
        let value = b[row] - (row + 1..n).map(|j| a[row][j] * x[j]).sum::<f64>();
        x[row] = value / a[row][row];
        if !x[row].is_finite() {
            return Err("Nonfinite patch Newton direction".into());
        }
    }
    Ok(x)
}
#[cfg(test)]
mod tests;
