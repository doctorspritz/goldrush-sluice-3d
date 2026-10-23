//! Instantaneous inelastic material custody. Active soil must establish the
//! contact/retention event; these functions do not create a digging force.
use super::*;
use crate::payload::{
    multiply, quadratic, solve_spd, ExchangeLedger, ExchangeReport, MassMoments, RigidParcel,
    RigidState,
};

type Matrix = [[f64; 10]; 10];
type Velocity = [f64; 10];
fn angular_inertia(b: &r::RigidBody) -> DMat3 {
    let i = b.mass_properties().effective_angular_inertia();
    DMat3::from_cols(
        sv(i * r::Vector::X).as_dvec3(),
        sv(i * r::Vector::Y).as_dvec3(),
        sv(i * r::Vector::Z).as_dvec3(),
    )
}
fn jacobian(model: &Model, part: usize, point: DVec3) -> ([DVec3; 10], [DVec3; 10]) {
    let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
    let j = mb.body_jacobian(part);
    let b = &model.world.bodies[model.bodies[part]];
    let offset = point - sv(b.center_of_mass()).as_dvec3();
    let angular =
        std::array::from_fn(|i| DVec3::new(j[(3, i)] as f64, j[(4, i)] as f64, j[(5, i)] as f64));
    let linear = std::array::from_fn(|i| {
        DVec3::new(j[(0, i)] as f64, j[(1, i)] as f64, j[(2, i)] as f64) + angular[i].cross(offset)
    });
    (linear, angular)
}
fn mass_matrix(model: &Model) -> Matrix {
    let mut out = [[0.; 10]; 10];
    for (part, h) in model.bodies.iter().enumerate() {
        let b = &model.world.bodies[*h];
        let inertia = angular_inertia(b);
        let (linear, angular) = jacobian(model, part, sv(b.center_of_mass()).as_dvec3());
        for i in 0..10 {
            for j in 0..10 {
                out[i][j] += b.mass() as f64 * linear[i].dot(linear[j])
                    + angular[i].dot(inertia * angular[j]);
            }
        }
    }
    out
}
fn momentum(model: &Model) -> (DVec3, DVec3) {
    let (mut p, mut l) = (DVec3::ZERO, DVec3::ZERO);
    for h in model.bodies {
        let b = &model.world.bodies[h];
        let bp = sv(b.linvel()).as_dvec3() * b.mass() as f64;
        p += bp;
        l += sv(b.center_of_mass()).as_dvec3().cross(bp)
            + angular_inertia(b) * sv(b.angvel()).as_dvec3();
    }
    (p, l)
}
fn make_model(m: &mut Excavator) -> Result<Model, String> {
    let state = m
        .articulation
        .as_ref()
        .ok_or("Articulated material exchange needs a dynamic mechanism")?
        .clone();
    if !parameters_valid(m) || !state.valid_for(m) {
        return Err("Invalid machine before material exchange".into());
    }
    Model::new(m, &Environment::free(Vec3::ZERO), &state)
}
pub fn payload_frame(m: &Excavator) -> (DVec3, DQuat) {
    let q = m.bucket_rotation();
    (
        m.arm().wrist.as_dvec3(),
        DQuat::from_xyzw(q.x as f64, q.y as f64, q.z as f64, q.w as f64).normalize(),
    )
}
fn existing_parcel(m: &Excavator, model: &Model, fraction: f64) -> Result<RigidParcel, String> {
    if m.bucket.wet_mass_kg() <= 0. {
        return Err("No finite payload to release".into());
    }
    let moments = carried_moments(m).scaled(fraction);
    let (wrist, q) = payload_frame(m);
    let center = wrist + q * moments.center;
    let b = &model.world.bodies[model.bodies[4]];
    let omega = sv(b.angvel()).as_dvec3();
    let r = DMat3::from_quat(q);
    let state = RigidState {
        center,
        velocity: sv(b.linvel()).as_dvec3()
            + omega.cross(center - sv(b.center_of_mass()).as_dvec3()),
        rotation: q,
        spin_nms: r * moments.inertia() * r.transpose() * omega,
        moments: MassMoments {
            center: DVec3::ZERO,
            ..moments
        },
    };
    let parcel = RigidParcel {
        material: m.bucket.scaled(fraction),
        state,
    };
    if !parcel.valid() {
        return Err("Payload cannot form a finite rigid parcel".into());
    }
    Ok(parcel)
}
/// Read the actual carried distribution and its point velocity/spin without
/// changing custody. Useful for independent controls and future contact tests.
pub fn sample_payload(m: &Excavator) -> Result<RigidParcel, String> {
    let mut copy = m.clone();
    let model = make_model(&mut copy)?;
    existing_parcel(m, &model, 1.)
}
fn finish(
    original: &Excavator,
    candidate: &mut Excavator,
    old: &Model,
    mut new: Model,
    parcel: &RigidParcel,
    capture: bool,
) -> Result<ExchangeReport, String> {
    let before_mass = mass_matrix(old);
    let after_mass = mass_matrix(&new);
    let velocity: Velocity = original
        .articulation
        .as_ref()
        .unwrap()
        .velocity
        .map(|x| x as f64);
    let kinetic = quadratic(&before_mass, &velocity);
    let (linear, angular) = jacobian(if capture { &new } else { old }, 4, parcel.state.center);
    let pp = parcel.state.linear_momentum();
    let spin = parcel.state.spin_nms;
    let mut target = multiply(&before_mass, &velocity);
    for i in 0..10 {
        target[i] += if capture { 1. } else { -1. } * (linear[i].dot(pp) + angular[i].dot(spin));
    }
    let solved = solve_spd(&after_mass, &target)?;
    let rounded = solved.map(|x| x as f32);
    if rounded.iter().any(|x| !x.is_finite() || x.abs() >= 100.) {
        return Err("Payload exchange exceeds supported mechanism speed".into());
    }
    let final_velocity = rounded.map(|x| x as f64);
    new.set_velocity(&rounded);
    let mut heat = 0.;
    if capture {
        let delta = std::array::from_fn(|i| final_velocity[i] - velocity[i]);
        let projected_v = (0..10)
            .map(|i| linear[i] * final_velocity[i])
            .sum::<DVec3>();
        let projected_w = (0..10)
            .map(|i| angular[i] * final_velocity[i])
            .sum::<DVec3>();
        heat = quadratic(&before_mass, &delta)
            + 0.5
                * parcel.material.wet_mass_kg()
                * (parcel.state.velocity - projected_v).length_squared()
            + 0.5
                * (parcel.state.angular_velocity() - projected_w).dot(
                    parcel.state.world_inertia() * (parcel.state.angular_velocity() - projected_w),
                );
    }
    let before = kinetic
        + if capture {
            parcel.state.kinetic_j()
        } else {
            0.
        };
    let after = quadratic(&after_mass, &final_velocity)
        + if capture {
            0.
        } else {
            parcel.state.kinetic_j()
        };
    let (old_p, old_l) = momentum(old);
    let (new_p, new_l) = momentum(&new);
    let residual = after + heat - before;
    let report = ExchangeReport {
        capture,
        transferred_kg: parcel.material.wet_mass_kg(),
        kinetic_before_j: before,
        kinetic_after_j: after,
        inelastic_heat_j: heat,
        kinetic_residual_j: residual,
        linear_residual_ns: new_p - old_p + if capture { -pp } else { pp },
        angular_residual_nms: new_l - old_l
            + if capture {
                -parcel.state.angular_momentum()
            } else {
                parcel.state.angular_momentum()
            },
    };
    // Explicit f32 registration/velocity rounding allowance. Residuals remain
    // reported; never rescale velocity to manufacture conservation/passivity.
    if !heat.is_finite() || heat < -1e-9 || residual.abs() > 2e-5 * (1. + before + after) {
        return Err(format!(
            "Material exchange kinetic projection failed: residual={residual}, heat={heat}"
        ));
    }
    let feedback = new.feedback(
        candidate.articulation.as_ref().unwrap(),
        &candidate.hydraulics,
    );
    let state = candidate.articulation.as_mut().unwrap();
    state.velocity = rounded;
    state.feedback = feedback;
    let ledger = state.exchange.get_or_insert_with(ExchangeLedger::default);
    ledger.events += 1;
    ledger.inelastic_heat_j += heat;
    if capture {
        ledger.captured_kg += report.transferred_kg;
    } else {
        ledger.released_kg += report.transferred_kg;
    }
    ledger.last = report.clone();
    candidate.tracks.velocity = state.feedback.linear_momentum_ns / state.feedback.mass_kg as f32;
    candidate.tracks.angular_velocity = Vec3::from_slice(&rounded[3..6]);
    candidate.tracks.feedback.mass_kg = state.feedback.mass_kg as f32;
    candidate.tracks.feedback.kinetic_energy_j = state.feedback.kinetic_j as f32;
    candidate.tracks.feedback.center_of_mass = state.feedback.center_of_mass;
    if !candidate.valid() {
        return Err("Invalid mechanism after material exchange".into());
    }
    Ok(report)
}
/// Capture a declared fraction of an already existing finite parcel. Its COM
/// must be inside the declared bucket envelope. Active soil/contact/retention
/// must establish that this inelastic event is appropriate before calling.
/// On failure neither the source parcel nor the mechanism is changed.
pub fn capture_payload(
    m: &mut Excavator,
    source: &mut Option<RigidParcel>,
    fraction: f64,
) -> Result<ExchangeReport, String> {
    if !fraction.is_finite() || !(0. ..=1.).contains(&fraction) || fraction == 0. {
        return Err("Invalid capture fraction".into());
    }
    let src = source.as_ref().ok_or("No finite source parcel")?;
    if !src.valid() {
        return Err("Invalid source parcel".into());
    }
    let parcel = src.fraction(fraction);
    if m.bucket.volume() + parcel.material.volume() > crate::machine::BUCKET_CAPACITY + 1e-12 {
        return Err("Finite bucket capacity exceeded".into());
    }
    let (wrist, q) = payload_frame(m);
    let local = q.inverse() * (parcel.state.center - wrist);
    if m.articulation.as_ref().unwrap().bucket_model.is_shell() {
        let local_moments = parcel
            .state
            .moments
            .transformed(local, q.inverse() * parcel.state.rotation);
        if !m.bucket_model().geometry().accepts_moments(local_moments) {
            return Err("Source spatial moments cannot fit the hollow bucket".into());
        }
    } else {
        let floor = steel_moments(crate::bucket::Model::LegacyFloor);
        let fq = DQuat::from_rotation_arc(DVec3::Z, DVec3::new(0., 0.0075, 1.075).normalize());
        let p = fq.inverse() * (local - floor.center);
        if p.x.abs() > 0.41 || !(0.0249..=0.65).contains(&p.y) || p.z.abs() > 0.5376 {
            return Err("Source parcel is outside the bucket capture envelope".into());
        }
        let local_moments = parcel
            .state
            .moments
            .transformed(local, q.inverse() * parcel.state.rotation);
        let r = DMat3::from_quat(fq);
        let c = r.transpose() * local_moments.second * r / local_moments.mass_kg;
        let variance = DVec3::new(c.x_axis.x, c.y_axis.y, c.z_axis.z);
        let low = DVec3::new(-0.41, 0.0249, -0.5376);
        let high = DVec3::new(0.41, 0.65, 0.5376);
        if (variance - (high - p) * (p - low)).max_element() > 1e-10 {
            return Err("Source spatial moments cannot fit the bucket envelope".into());
        }
    }
    let mut original = m.clone();
    let old = make_model(&mut original)?;
    let incoming = parcel
        .state
        .moments
        .transformed(local, q.inverse() * parcel.state.rotation);
    let moments = carried_moments(m).combine(incoming);
    let mut candidate = m.clone();
    candidate.bucket.add(parcel.material);
    let state = candidate.articulation.as_mut().unwrap();
    state.payload_kg = candidate.bucket.wet_mass_kg() as f32;
    state.payload_moments = Some(moments);
    state.exchange.get_or_insert_with(ExchangeLedger::default);
    let new = make_model(&mut candidate)?;
    let report = finish(m, &mut candidate, &old, new, &parcel, true)?;
    let remainder = if fraction == 1. {
        None
    } else {
        Some(src.fraction(1. - fraction))
    };
    *m = candidate;
    *source = remainder;
    Ok(report)
}
/// Uniformly split the carried spatial distribution. The parcel inherits its
/// actual COM velocity and spin. No forward kick, teleport to a chute, new
/// pump work or gravity step is introduced by this custody event.
pub fn release_payload(m: &mut Excavator, fraction: f64) -> Result<RigidParcel, String> {
    if !fraction.is_finite() || !(0. ..=1.).contains(&fraction) || fraction == 0. {
        return Err("Invalid release fraction".into());
    }
    let mut original = m.clone();
    let old = make_model(&mut original)?;
    let parcel = existing_parcel(m, &old, fraction)?;
    let remaining = carried_moments(m).scaled(1. - fraction);
    let mut candidate = m.clone();
    candidate.bucket = candidate.bucket.scaled(1. - fraction);
    let state = candidate.articulation.as_mut().unwrap();
    state.payload_kg = candidate.bucket.wet_mass_kg() as f32;
    state.payload_moments = (fraction < 1.).then_some(remaining);
    state.exchange.get_or_insert_with(ExchangeLedger::default);
    let new = make_model(&mut candidate)?;
    finish(m, &mut candidate, &old, new, &parcel, false)?;
    *m = candidate;
    Ok(parcel)
}
