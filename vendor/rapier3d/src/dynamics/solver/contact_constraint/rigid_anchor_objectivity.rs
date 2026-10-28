//! Objectivity controls for recycled Coulomb contact anchors.
//!
//! The source manifold keeps body-local contact witnesses from its last full
//! narrow-phase update. These controls apply common rigid motion to that pair,
//! then exercise the actual builder generation and biased/unbiased RHS updates.
//! A common rotation cannot create physical contact slip. Genuine persistent
//! tangential displacement must remain in the correction target.

use super::*;
use crate::dynamics::{MassProperties, RigidBodyBuilder};
use crate::geometry::{ColliderSet, SolverContact};
use crate::math::{Rotation, Vector};
use parry::shape::PackedFeatureId;

const RADIUS: Real = 0.025;
const SUBSTEP: Real = 1. / 1920.;
const VELOCITY_TOLERANCE: Real = 3.0e-4;

struct Fixture {
    builder: ContactWithCoulombFrictionBuilder,
    constraint: ContactWithCoulombFriction<SimdReal>,
    solver: SolverBodies,
    params: IntegrationParameters,
    actual_delta: Vector,
}

fn fixture(angle: Real, common_translation: Vector, persistent_slip: Vector) -> Fixture {
    fixture_with_surface_velocity(angle, common_translation, persistent_slip, Vector::ZERO)
}
fn fixture_with_surface_velocity(
    angle: Real,
    common_translation: Vector,
    persistent_slip: Vector,
    surface_velocity: Vector,
) -> Fixture {
    let rotation = Rotation::from_rotation_z(angle);
    let old_com = [Vector::Y * RADIUS, -Vector::Y * RADIUS];
    let anchors = [-old_com[0], -old_com[1]];
    let mut bodies = RigidBodySet::new();
    let mut solver = SolverBodies::default();
    solver.resize(2);
    for side in 0..2 {
        let slip = if side == 0 {
            persistent_slip
        } else {
            Vector::ZERO
        };
        let mut body = RigidBodyBuilder::dynamic()
            .translation(rotation * old_com[side] + common_translation + slip)
            .rotation(Vector::Z * angle)
            .additional_mass_properties(MassProperties::new(Vector::ZERO, 1., Vector::ONE))
            .build();
        body.recompute_mass_properties_from_colliders(&ColliderSet::new());
        let handle = bodies.insert(body);
        // Insertion resets internal solver references.
        bodies[handle].ids.active_set_id = side as u32;
        solver.copy_from(side, &bodies[handle]);
    }

    let mut manifold = ContactManifold::new();
    manifold.data.normal = Vector::Y;
    manifold.data.friction = 0.7;
    manifold.data.restitution = 0.;
    manifold.data.relative_dominance = 0;
    manifold.data.solver_body_ids = [0, 1];
    let mut point = parry::query::TrackedContact::<crate::geometry::ContactData>::new(
        anchors[0],
        anchors[1],
        PackedFeatureId::face(0),
        PackedFeatureId::face(0),
        0.,
    );
    // Exactly the world-space arms frozen at the original coincident witness.
    point.data.solver_dp1 = anchors[0];
    point.data.solver_dp2 = anchors[1];
    point.data.impulse = 1.;
    point.data.warmstart_impulse = 1.;
    point.data.warmstart_tangent_world = Vector::new(0.1, 0., -0.05);
    manifold.points.push(point);
    manifold.data.solver_contacts.push(SolverContact {
        anchor1: anchors[0],
        anchor2: anchors[1],
        dist: 0.,
        tangent_velocity: surface_velocity,
        contact_id: [0],
        padding: [0.],
    });
    let actual_points = core::array::from_fn::<_, 2, _>(|side| {
        solver.poses[side].rotation * anchors[side] + solver.poses[side].translation
    });
    let actual_delta = actual_points[0] - actual_points[1];
    assert!(
        (actual_delta - persistent_slip).length() < 1.0e-7,
        "independent material-witness construction changed slip: {actual_delta:?}"
    );
    // The common rotation preserves relative pose; it is inside the actual
    // recycling policy's additional absolute-rotation bound, cos(theta) > .98.
    assert!(angle.cos() > 0.98);

    let mut builder = unsafe { core::mem::zeroed() };
    let mut constraint = unsafe { core::mem::zeroed() };
    let ids = core::array::from_fn(|lane| {
        if lane == 0 {
            ContactRef {
                edge: 0,
                manifold: 0,
            }
        } else {
            ContactRef::PADDING
        }
    });
    ContactWithCoulombFrictionBuilder::generate(
        ids,
        [&manifold; SIMD_WIDTH],
        &bodies,
        &solver,
        &mut builder,
        &mut constraint,
    );
    let params = IntegrationParameters {
        dt: SUBSTEP,
        ..IntegrationParameters::default()
    };
    Fixture {
        builder,
        constraint,
        solver,
        params,
        actual_delta,
    }
}

fn scalar(vector: <SimdReal as ScalarType>::Vector) -> Vector {
    Vector::new(
        vector.x.extract(0),
        vector.y.extract(0),
        vector.z.extract(0),
    )
}

fn tangent_rhs_world(constraint: &ContactWithCoulombFriction<SimdReal>) -> Vector {
    let normal = scalar(constraint.dir1);
    let tangent = scalar(constraint.tangent1);
    let bitangent = normal.cross(tangent);
    tangent * constraint.tangent_part[0].rhs[0].extract(0)
        + bitangent * constraint.tangent_part[0].rhs[1].extract(0)
}

fn expected_rhs(fixture: &Fixture) -> Vector {
    let normal = scalar(fixture.constraint.dir1);
    (fixture.actual_delta - normal * fixture.actual_delta.dot(normal)) / fixture.params.dt
}

fn update_biased(fixture: &mut Fixture) {
    fixture.builder.update(
        &fixture.params,
        0.,
        &fixture.solver,
        &MultibodyJointSet::new(),
        &mut fixture.constraint,
    );
}

#[test]
#[ignore = "known legacy anchor-objectivity defect; explicit negative control, not a passing physics gate"]
fn common_rigid_rotation_does_not_invent_tangential_anchor_velocity() {
    for angle in [0., 0.025, 0.1, -0.1] {
        for translation in [Vector::ZERO, Vector::new(0.125, -0.25, 0.5)] {
            let mut f = fixture(angle, translation, Vector::ZERO);
            update_biased(&mut f);
            let actual = tangent_rhs_world(&f.constraint);
            let expected = expected_rhs(&f);
            assert!(
                (actual - expected).length() <= VELOCITY_TOLERANCE,
                "common rigid angle={angle}, translation={translation:?} invented contact \
                 velocity: actual={actual:?}, material-witness target={expected:?}, \
                 discrepancy={} m/s",
                (actual - expected).length()
            );
            assert!(f.constraint.normal_part[0].rhs.extract(0).abs() <= VELOCITY_TOLERANCE);
        }
    }
}

#[test]
fn actual_persistent_tangential_slip_is_preserved() {
    let mut f = fixture(0., Vector::ZERO, Vector::new(0.001, 0., -0.0007));
    update_biased(&mut f);
    let actual = tangent_rhs_world(&f.constraint);
    let expected = expected_rhs(&f);
    assert!(expected.length() > 2.);
    assert!(
        (actual - expected).length() <= VELOCITY_TOLERANCE,
        "persistent material slip must not be erased: {actual:?} versus {expected:?}"
    );
}

#[test]
#[ignore = "known legacy anchor-objectivity defect; explicit negative control, not a passing physics gate"]
fn common_rotation_preserves_real_tangential_slip_without_adding_false_slip() {
    for angle in [0.025, 0.1, -0.1] {
        let mut f = fixture(angle, Vector::ZERO, Vector::new(0.001, 0., -0.0007));
        update_biased(&mut f);
        let actual = tangent_rhs_world(&f.constraint);
        let expected = expected_rhs(&f);
        assert!(expected.length() > 2.);
        assert!(
            (actual - expected).length() <= VELOCITY_TOLERANCE,
            "angle={angle}: true persistent slip target={expected:?}, \
             builder target={actual:?}"
        );
    }
}

#[test]
fn unbiased_update_removes_position_target_and_preserves_source_impulses() {
    let mut f = fixture(0.1, Vector::ZERO, Vector::new(0.001, 0., -0.0007));
    update_biased(&mut f);
    let normal_impulse = f.constraint.normal_part[0].impulse;
    let tangent_impulse = f.constraint.tangent_part[0].impulse;
    f.builder
        .update_rhs_wo_bias(&f.params, 0., &f.solver, &mut f.constraint);
    assert_eq!(tangent_rhs_world(&f.constraint), Vector::ZERO);
    assert_eq!(f.constraint.normal_part[0].impulse, normal_impulse);
    assert_eq!(f.constraint.tangent_part[0].impulse, tangent_impulse);
}

#[cfg(feature = "contact-work-audit")]
fn move_common(f: &mut Fixture, rotation: Rotation, translation: Vector, added_slip: Vector) {
    for (side, pose) in f.solver.poses.iter_mut().enumerate() {
        pose.translation = rotation * pose.translation
            + translation
            + if side == 0 { added_slip } else { Vector::ZERO };
        pose.rotation = rotation * pose.rotation;
    }
    f.actual_delta = rotation * f.actual_delta + added_slip;
}
#[cfg(feature = "contact-work-audit")]
fn legacy_tracking_delta(f: &Fixture) -> Vector {
    let info = &f.builder.infos[0];
    f.solver.poses[0].rotation * scalar(info.local_p1) + f.solver.poses[0].translation
        - (f.solver.poses[1].rotation * scalar(info.local_p2) + f.solver.poses[1].translation)
}
#[test]
#[cfg(feature = "contact-work-audit")]
fn material_tangent_trial_is_objective_at_generation_without_erasing_real_slip() {
    for angle in [0., 0.025, 0.1, -0.1] {
        for translation in [Vector::ZERO, Vector::new(0.125, -0.25, 0.5)] {
            for slip in [
                Vector::ZERO,
                Vector::new(0.001, 0., -0.0007),
                Vector::new(-0.0005, 0., 0.0002),
            ] {
                let mut f = fixture(angle, translation, slip);
                let mut legacy = fixture(angle, translation, slip);
                f.params.diagnostic_material_tangent_anchors = true;
                update_biased(&mut f);
                update_biased(&mut legacy);
                let actual = tangent_rhs_world(&f.constraint);
                assert!(
                    (actual - expected_rhs(&f)).length() <= VELOCITY_TOLERANCE,
                    "angle={angle}, translation={translation:?}, slip={slip:?}, actual={actual:?}, expected={:?}",
                    expected_rhs(&f)
                );
                for lane in 0..SIMD_WIDTH {
                    assert_eq!(
                        f.constraint.normal_part[0].rhs.extract(lane).to_bits(),
                        legacy.constraint.normal_part[0].rhs.extract(lane).to_bits()
                    );
                    assert_eq!(
                        f.constraint.normal_part[0].impulse.extract(lane).to_bits(),
                        legacy.constraint.normal_part[0]
                            .impulse
                            .extract(lane)
                            .to_bits()
                    );
                }
                if slip != Vector::ZERO {
                    assert!(expected_rhs(&f).length() > 0.5);
                }
            }
        }
    }
}
#[test]
#[cfg(feature = "contact-work-audit")]
fn material_tangent_trial_remains_objective_during_substeps_and_retains_new_slip() {
    for slip in [Vector::ZERO, Vector::new(0.001, 0., -0.0007)] {
        let mut f = fixture(0.025, Vector::ZERO, slip);
        f.params.diagnostic_material_tangent_anchors = true;
        for rotation in [
            Rotation::from_rotation_x(0.06),
            Rotation::from_rotation_y(-0.09),
            Rotation::from_rotation_z(0.04),
        ] {
            move_common(
                &mut f,
                rotation,
                Vector::new(0.02, -0.01, 0.03),
                Vector::ZERO,
            );
            update_biased(&mut f);
            assert!(
                (tangent_rhs_world(&f.constraint) - expected_rhs(&f)).length()
                    <= VELOCITY_TOLERANCE
            );
        }
        move_common(
            &mut f,
            Rotation::IDENTITY,
            Vector::ZERO,
            Vector::new(0.0005, 0., -0.0003),
        );
        update_biased(&mut f);
        assert!(
            (tangent_rhs_world(&f.constraint) - expected_rhs(&f)).length() <= VELOCITY_TOLERANCE
        );
        assert!(
            expected_rhs(&f).length() > 0.5,
            "new real relative slip was erased"
        );
        let normal = f.constraint.normal_part[0].impulse;
        let tangent = f.constraint.tangent_part[0].impulse;
        f.builder
            .update_rhs_wo_bias(&f.params, 0., &f.solver, &mut f.constraint);
        assert_eq!(tangent_rhs_world(&f.constraint), Vector::ZERO);
        assert_eq!(f.constraint.normal_part[0].impulse, normal);
        assert_eq!(f.constraint.tangent_part[0].impulse, tangent);
    }
}
#[test]
#[cfg(feature = "contact-work-audit")]
fn fixed_world_rebase_is_a_negative_control_for_substep_objectivity() {
    let mut f = fixture(0.1, Vector::ZERO, Vector::ZERO);
    let rebase = f.actual_delta - legacy_tracking_delta(&f);
    assert!((legacy_tracking_delta(&f) + rebase).length() / SUBSTEP < VELOCITY_TOLERANCE);
    move_common(
        &mut f,
        Rotation::from_rotation_z(0.05),
        Vector::new(0.01, 0.02, -0.01),
        Vector::ZERO,
    );
    let remaining = legacy_tracking_delta(&f) + rebase;
    let normal = scalar(f.constraint.dir1);
    let false_target = (remaining - normal * remaining.dot(normal)) / SUBSTEP;
    assert!(
        false_target.length() > 0.01,
        "negative control must expose fixed-offset drift: {false_target:?}"
    );
    f.params.diagnostic_material_tangent_anchors = true;
    update_biased(&mut f);
    assert!((tangent_rhs_world(&f.constraint) - expected_rhs(&f)).length() <= VELOCITY_TOLERANCE);
}
#[test]
#[cfg(feature = "contact-work-audit")]
fn material_tangent_trial_preserves_moving_surface_velocity_and_displacement() {
    let velocity = Vector::new(0.3, 0., -0.2);
    let slip = Vector::new(0.001, 0., -0.0007);
    let mut f = fixture_with_surface_velocity(0.1, Vector::ZERO, slip, velocity);
    f.params.diagnostic_material_tangent_anchors = true;
    let elapsed = 2. * SUBSTEP;
    f.builder.update(
        &f.params,
        elapsed,
        &f.solver,
        &MultibodyJointSet::new(),
        &mut f.constraint,
    );
    let normal = scalar(f.constraint.dir1);
    let target = velocity + (f.actual_delta + velocity * elapsed) / SUBSTEP;
    let target = target - normal * target.dot(normal);
    assert!((tangent_rhs_world(&f.constraint) - target).length() <= VELOCITY_TOLERANCE);
    f.builder
        .update_rhs_wo_bias(&f.params, elapsed, &f.solver, &mut f.constraint);
    assert!((tangent_rhs_world(&f.constraint) - velocity).length() < VELOCITY_TOLERANCE);
}

#[test]
#[cfg(feature = "contact-work-audit")]
fn material_anchor_trial_exposes_normal_gap_rotation_with_frozen_basis() {
    // A normal witness gap is not tangential material slip. After a common
    // rigid motion it stays normal to the transported contact plane. The trial
    // still uses the legacy frozen basis/Jacobians, so it is not a complete
    // objective contact model even though coincident-witness controls pass.
    let mut f = fixture(0., Vector::ZERO, Vector::Y * 0.001);
    f.params.diagnostic_material_tangent_anchors = true;
    update_biased(&mut f);
    assert!(tangent_rhs_world(&f.constraint).length() < VELOCITY_TOLERANCE);
    let old_normal = scalar(f.constraint.dir1);
    let rotation = Rotation::from_rotation_z(0.1);
    move_common(&mut f, rotation, Vector::ZERO, Vector::ZERO);
    update_biased(&mut f);
    let transported_normal = rotation * old_normal;
    let true_tangent =
        (f.actual_delta - transported_normal * f.actual_delta.dot(transported_normal)) / SUBSTEP;
    let actual = tangent_rhs_world(&f.constraint);
    assert!(true_tangent.length() < VELOCITY_TOLERANCE);
    assert!(
        actual.length() > 0.15,
        "counterexample disappeared; reassess the frozen-basis limitation: {actual:?}"
    );
    std::println!(
        "normal-gap common-rotation counterexample: material_trial_target={actual:?}, transported-plane_target={true_tangent:?}, false_speed={}m/s",
        actual.length()
    );
}
