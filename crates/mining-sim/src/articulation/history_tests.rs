//! Saved numerical state before persistent contact integration is admitted.
use super::*;
use crate::contact_history::{ContactHistory, Dispatcher};

fn bank() -> (Model, Articulation) {
    let mut session = crate::Session::new();
    session.prepare_soil_trial().unwrap();
    let mut soil = session.active_soil.take().unwrap();
    assert_eq!(soil.clods.len(), 1040);
    let state = session.machine.articulation.as_ref().unwrap().clone();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let ground = soil.ground(&session.site);
    let mut model = Model::with_ground(&mut session.machine, &env, &state, Some(ground)).unwrap();
    model
        .world
        .integration_parameters
        .maximum_dissipation_friction = true;
    crate::soil_contacts::configure(&mut model.world);
    for clod in &mut soil.clods {
        clod.insert_precise(&mut model.world).unwrap();
    }
    (model, state)
}
fn advance(model: &mut Model, state: &Articulation, tick: usize) {
    let (mb, _) = model
        .world
        .multibody_joints
        .get_mut(model.joints[0])
        .unwrap();
    for i in 0..4 {
        let offset = if i == 3 {
            (tick as f32 * 0.015).sin() * 0.1
        } else {
            0.
        };
        mb.link_mut(i + 1).unwrap().joint.data.set_motor_position(
            r::JointAxis::AngX,
            state.reference[i] + offset,
            state.stiffness_nm_rad[i],
            state.damping_nms_rad[i],
        );
    }
    model.world.step_with_events(
        &Belts {
            forward: *model.world.bodies[model.bodies[0]].rotation() * r::Vector::Z,
            speed: [0.; 2],
        },
        &(),
    );
    let (mb, _) = model
        .world
        .multibody_joints
        .get_mut(model.joints[0])
        .unwrap();
    let velocity: [f32; 10] = mb.generalized_velocity().as_slice().try_into().unwrap();
    mb.forward_kinematics(&model.world.bodies, false);
    mb.update_rigid_bodies(&mut model.world.bodies, true);
    model.set_velocity(&velocity);
}
fn copy_model(model: &Model, world: r::PhysicsWorld) -> Model {
    Model {
        world,
        bodies: model.bodies,
        joints: model.joints,
        soil_handles: model.soil_handles.clone(),
        gravity: model.gravity,
    }
}
fn body_state(body: &r::RigidBody) -> Vec<f32> {
    sv(body.translation())
        .to_array()
        .into_iter()
        .chain(sq(*body.rotation()).to_array())
        .chain(sv(body.linvel()).to_array())
        .chain(sv(body.angvel()).to_array())
        .collect()
}
fn equal_physics(a: &Model, b: &Model, tick: usize, label: &str) {
    assert_eq!(a.world.bodies.len(), b.world.bodies.len());
    for (handle, body) in a.world.bodies.iter() {
        let left = body_state(body);
        let right = body_state(&b.world.bodies[handle]);
        for (axis, (x, y)) in left.iter().zip(&right).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label} tick={tick} body={handle:?} field={axis}: {x} vs {y}"
            );
        }
    }
    for pair in a.world.narrow_phase.contact_pairs() {
        let other = b
            .world
            .narrow_phase
            .contact_pair(pair.collider1, pair.collider2)
            .unwrap();
        for (kind, (left, right)) in [
            (pair.manifolds(), other.manifolds()),
            (pair.solver_manifolds(), other.solver_manifolds()),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                left.len(),
                right.len(),
                "{label} tick={tick} manifold count"
            );
            for (i, (x, y)) in left.iter().zip(right).enumerate() {
                assert!(bincode::serialize(x).unwrap()==bincode::serialize(y).unwrap(),
                    "{label} tick={tick} pair={:?}/{:?} kind={kind} manifold={i} cached contact state",pair.collider1,pair.collider2);
            }
        }
    }
    let (am, _) = a.world.multibody_joints.get(a.joints[0]).unwrap();
    let (bm, _) = b.world.multibody_joints.get(b.joints[0]).unwrap();
    assert_eq!(
        am.generalized_velocity(),
        bm.generalized_velocity(),
        "{label} tick={tick} generalized velocity"
    );
}
#[test]
fn bank_world_clone_and_binary_restore_preserve_contact_continuation() {
    for warmup in [0, 24] {
        check_bank_boundary(warmup);
    }
}
fn check_bank_boundary(warmup: usize) {
    let (mut reference, state) = bank();
    for tick in 0..warmup {
        advance(&mut reference, &state, tick);
    }
    assert!(
        warmup == 0
            || reference
                .world
                .narrow_phase
                .contact_pairs()
                .any(|p| p.has_any_active_contact())
    );
    let boundary =
        ContactHistory::from_world(std::mem::take(&mut reference.world), Dispatcher::RigidSoil);
    let bytes = boundary
        .encode()
        .expect("complete physical world must serialize");
    println!("bank world binary bytes: {}", bytes.len());
    let restored_boundary = ContactHistory::decode(&bytes).expect("complete world must restore");
    assert_eq!(restored_boundary.dispatcher(), Dispatcher::RigidSoil);
    let mut restored = copy_model(&reference, restored_boundary.into_world());
    let mut cloned = copy_model(&reference, boundary.clone().into_world());
    reference.world = boundary.into_world();
    equal_physics(&reference, &restored, warmup, "restored boundary");
    equal_physics(&reference, &cloned, warmup, "cloned boundary");
    for tick in warmup..warmup + 32 {
        advance(&mut reference, &state, tick);
        advance(&mut restored, &state, tick);
        advance(&mut cloned, &state, tick);
        equal_physics(&reference, &restored, tick, "restored continuation");
        equal_physics(&reference, &cloned, tick, "cloned continuation");
        if tick % 8 == 7 {
            let warm = reference
                .world
                .narrow_phase
                .contact_pairs()
                .flat_map(|p| p.solver_manifolds())
                .flat_map(|m| &m.points)
                .filter(|p| p.data.warmstart_impulse > 0.)
                .count();
            let tangent = reference
                .world
                .narrow_phase
                .contact_pairs()
                .flat_map(|p| p.solver_manifolds())
                .flat_map(|m| &m.points)
                .filter(|p| p.data.warmstart_tangent_impulse.norm() > 0.)
                .count();
            println!("nonzero cached normal/tangent contacts: {warm}/{tangent}");
            assert!(
                warm > 0 && tangent > 0,
                "continuation needs real nonzero contact history"
            );
            let boundary = ContactHistory::from_world(
                std::mem::take(&mut restored.world),
                Dispatcher::RigidSoil,
            );
            restored.world = ContactHistory::decode(&boundary.encode().unwrap())
                .unwrap()
                .into_world();
        }
    }
}

#[test]
fn contact_boundary_preserves_saved_softness_and_rejects_bad_envelopes() {
    let mut world = r::PhysicsWorld::new();
    crate::soil_contacts::configure(&mut world);
    world.integration_parameters.contact_softness = r::SpringCoefficients::new(43210., 0.7);
    let params = bincode::serialize(&world.integration_parameters).unwrap();
    let state = ContactHistory::from_world(world, Dispatcher::RigidSoil);
    let bytes = state.encode().unwrap();
    let restored = ContactHistory::decode(&bytes).unwrap();
    assert_eq!(
        bincode::serialize(&restored.world().integration_parameters).unwrap(),
        params,
        "restoring a dispatcher must not retune physical contact parameters"
    );
    for index in [0, 8, 10] {
        let mut corrupt = bytes.clone();
        corrupt[index] = 255;
        assert!(ContactHistory::decode(&corrupt).is_err());
    }
    assert!(ContactHistory::decode(&bytes[..bytes.len() - 1]).is_err());
    assert!(ContactHistory::decode(&bytes[..10]).is_err());
    let oversized = vec![0u8; crate::contact_history::MAX_WORLD_BYTES + 12];
    assert!(ContactHistory::decode(&oversized).is_err());
    let mut extra = bytes;
    extra.push(0);
    assert!(
        ContactHistory::decode(&extra).is_err(),
        "trailing state accepted"
    );
}

#[test]
fn saved_free_fall_continues_exactly_through_a_fast_ccd_collision() {
    let mut world = r::PhysicsWorld::new();
    world.integration_parameters.dt = 1. / 240.;
    world.integration_parameters.num_solver_iterations = 4;
    world.integration_parameters.num_internal_pgs_iterations = 16;
    world.insert(
        r::RigidBodyBuilder::fixed(),
        r::ColliderBuilder::cuboid(2., 0.005, 2.),
    );
    let (body, _) = world.insert(
        r::RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .ccd_enabled(true)
            .translation(r::Vector::Y * 0.7)
            .linvel(r::Vector::new(3., -20., 0.))
            .angvel(r::Vector::new(20., 5., -3.)),
        r::ColliderBuilder::ball(0.015).mass(1.),
    );
    for _ in 0..3 {
        world.step();
    }
    let vy = world.bodies[body].linvel().y;
    assert!(
        (vy - (-20. - 9.81 * 3. / 240.)).abs() < 0.0001,
        "freefall momentum drift {vy}"
    );
    let mut reference = ContactHistory::from_world(world, Dispatcher::Default);
    let mut restored = ContactHistory::decode(&reference.encode().unwrap()).unwrap();
    let mut cloned = reference.clone();
    let mut contacted = false;
    for tick in 0..32 {
        reference.world_mut().step();
        restored.world_mut().step();
        cloned.world_mut().step();
        let expected = body_state(&reference.world().bodies[body]);
        for state in [&restored, &cloned] {
            let actual = body_state(&state.world().bodies[body]);
            for (axis, (x, y)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "CCD restore tick={tick} field={axis}"
                );
            }
        }
        contacted |= reference
            .world()
            .narrow_phase
            .contact_pairs()
            .any(|p| p.has_any_active_contact());
        assert!(
            reference.world().bodies[body].translation().y > 0.005,
            "tunnelled through thin plate"
        );
        if tick % 7 == 6 {
            restored = ContactHistory::decode(&restored.encode().unwrap()).unwrap();
        }
    }
    assert!(contacted, "fast body never reached a real plate contact");
}
