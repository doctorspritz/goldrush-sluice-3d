//! Impulse contacts for the externally integrated FLIP/DEM path.
//! Coulomb impulses use contact-point velocity and rotational effective mass.
//! They do not apply gravity or a timestep-dependent tangential damping.
use crate::clump::{Clump3D, ClumpTemplate3D};
use glam::{Mat3, Vec3};

#[derive(Clone, Copy)]
pub(crate) struct ContactMaterial {
    pub friction: f32,
    pub restitution: f32,
    pub rolling: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct ContactPoint {
    pub normal: Vec3,
    pub arm_a: Vec3,
    pub arm_b: Vec3,
}

fn inverse_mass(m: f32, inertia: Mat3, arm: Vec3, direction: Vec3) -> f32 {
    m + direction.dot((inertia * arm.cross(direction)).cross(arm))
}
fn apply(body: &mut Clump3D, mass: f32, inertia: Mat3, arm: Vec3, impulse: Vec3) {
    body.velocity += impulse * mass;
    body.angular_velocity += inertia * arm.cross(impulse);
}

pub(crate) fn resolve_static(
    body: &mut Clump3D,
    template: &ClumpTemplate3D,
    normal: Vec3,
    arm: Vec3,
    material: ContactMaterial,
) {
    let mass = template.inv_mass();
    let inertia = body.world_inertia_inv(template);
    let speed = body.particle_world_velocity(arm).dot(normal);
    let kn = inverse_mass(mass, inertia, arm, normal);
    if speed >= 0.0 || kn <= 0.0 {
        return;
    }
    let jn = -(1.0 + material.restitution.clamp(0.0, 1.0)) * speed / kn;
    apply(body, mass, inertia, arm, normal * jn);
    let velocity = body.particle_world_velocity(arm);
    let tangent = velocity - normal * velocity.dot(normal);
    let speed = tangent.length();
    if speed > 1e-12 {
        let direction = tangent / speed;
        let kt = inverse_mass(mass, inertia, arm, direction);
        let jt = (speed / kt.max(1e-30)).min(material.friction.max(0.0) * jn);
        apply(body, mass, inertia, arm, -direction * jt);
    }
    // Rolling resistance is a torque bounded by the normal contact impulse.
    let roll = body.angular_velocity - normal * body.angular_velocity.dot(normal);
    let speed = roll.length();
    if speed > 1e-12 && material.rolling > 0.0 {
        let axis = roll / speed;
        let limit = material.rolling * jn * template.particle_radius;
        let impulse = (speed / axis.dot(inertia * axis).max(1e-30)).min(limit);
        body.angular_velocity -= inertia * axis * impulse;
    }
}

pub(crate) fn resolve_pair(
    a: &mut Clump3D,
    ta: &ClumpTemplate3D,
    b: &mut Clump3D,
    tb: &ClumpTemplate3D,
    point: ContactPoint,
    material: ContactMaterial,
) {
    let ma = ta.inv_mass();
    let mb = tb.inv_mass();
    let ia = a.world_inertia_inv(ta);
    let ib = b.world_inertia_inv(tb);
    let relative = b.particle_world_velocity(point.arm_b) - a.particle_world_velocity(point.arm_a);
    let closing = relative.dot(point.normal);
    let kn = inverse_mass(ma, ia, point.arm_a, point.normal)
        + inverse_mass(mb, ib, point.arm_b, point.normal);
    if closing >= 0.0 || kn <= 0.0 {
        return;
    }
    let jn = -(1.0 + material.restitution.clamp(0.0, 1.0)) * closing / kn;
    let impulse = point.normal * jn;
    apply(a, ma, ia, point.arm_a, -impulse);
    apply(b, mb, ib, point.arm_b, impulse);
    let relative = b.particle_world_velocity(point.arm_b) - a.particle_world_velocity(point.arm_a);
    let tangent = relative - point.normal * relative.dot(point.normal);
    let speed = tangent.length();
    if speed > 1e-12 {
        let direction = tangent / speed;
        let kt = inverse_mass(ma, ia, point.arm_a, direction)
            + inverse_mass(mb, ib, point.arm_b, direction);
        let jt = (speed / kt.max(1e-30)).min(material.friction.max(0.0) * jn);
        let impulse = -direction * jt;
        apply(a, ma, ia, point.arm_a, -impulse);
        apply(b, mb, ib, point.arm_b, impulse);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clump::{ClumpShape3D, IrregularStyle3D};
    fn sphere(mass: f32) -> ClumpTemplate3D {
        ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 1,
                seed: 0,
                style: IrregularStyle3D::Round,
            },
            0.01,
            mass,
        )
    }
    fn material(mu: f32) -> ContactMaterial {
        ContactMaterial {
            friction: mu,
            restitution: 0.,
            rolling: 0.,
        }
    }
    fn energy(b: &Clump3D, t: &ClumpTemplate3D) -> f32 {
        0.5 * t.mass * b.velocity.length_squared()
            + 0.5
                * b.angular_velocity
                    .dot(b.world_inertia_inv(t).inverse() * b.angular_velocity)
    }
    #[test]
    fn sliding_impulse_is_coulomb_limited_and_generates_rotation() {
        let t = sphere(2.);
        let mut b = Clump3D::new(Vec3::ZERO, Vec3::new(10., -1., 0.), 0);
        let before = energy(&b, &t);
        resolve_static(&mut b, &t, Vec3::Y, -Vec3::Y * 0.01, material(0.2));
        assert!((b.velocity.x - 9.8).abs() < 1e-6);
        assert!(b.velocity.y.abs() < 1e-6 && b.angular_velocity.z < 0.);
        assert!(energy(&b, &t) < before);
    }
    #[test]
    fn sticking_contact_reaches_no_slip_without_stopping_centre_of_mass() {
        let t = sphere(1.);
        let mut b = Clump3D::new(Vec3::ZERO, Vec3::new(0.1, -1., 0.), 0);
        resolve_static(&mut b, &t, Vec3::Y, -Vec3::Y * 0.01, material(0.6));
        assert!(b.particle_world_velocity(-Vec3::Y * 0.01).length() < 1e-6);
        assert!(
            b.velocity.x > 0. && b.angular_velocity.z < 0.,
            "sphere must be able to roll"
        );
    }
    #[test]
    fn separating_contact_adds_no_friction_or_energy() {
        let t = sphere(1.);
        let mut b = Clump3D::new(Vec3::ZERO, Vec3::new(1., 1., 0.), 0);
        let before = b.velocity;
        resolve_static(&mut b, &t, Vec3::Y, -Vec3::Y * 0.01, material(1.));
        assert_eq!(b.velocity, before);
        assert_eq!(b.angular_velocity, Vec3::ZERO);
    }
    #[test]
    fn oblique_pair_conserves_linear_and_angular_momentum_and_dissipates_energy() {
        let ta = sphere(1.);
        let tb = sphere(3.);
        let mut a = Clump3D::new(-Vec3::X * 0.01, Vec3::new(1., 2., 0.), 0);
        let mut b = Clump3D::new(Vec3::X * 0.01, Vec3::new(-1., -1., 0.), 1);
        let momentum = |a: &Clump3D, b: &Clump3D| a.velocity * ta.mass + b.velocity * tb.mass;
        let angular = |a: &Clump3D, b: &Clump3D| {
            a.position.cross(a.velocity * ta.mass)
                + b.position.cross(b.velocity * tb.mass)
                + a.world_inertia_inv(&ta).inverse() * a.angular_velocity
                + b.world_inertia_inv(&tb).inverse() * b.angular_velocity
        };
        let p = momentum(&a, &b);
        let l = angular(&a, &b);
        let e = energy(&a, &ta) + energy(&b, &tb);
        resolve_pair(
            &mut a,
            &ta,
            &mut b,
            &tb,
            ContactPoint {
                normal: Vec3::X,
                arm_a: Vec3::X * 0.01,
                arm_b: -Vec3::X * 0.01,
            },
            material(0.4),
        );
        assert!((momentum(&a, &b) - p).length() < 1e-6);
        assert!((angular(&a, &b) - l).length() < 1e-7);
        assert!(energy(&a, &ta) + energy(&b, &tb) < e);
        assert!(a.angular_velocity.z.abs() > 1. && b.angular_velocity.z.abs() > 1.);
    }
}
