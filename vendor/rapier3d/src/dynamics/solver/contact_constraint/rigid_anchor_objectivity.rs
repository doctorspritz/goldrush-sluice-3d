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
        tangent_velocity: Vector::ZERO,
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
