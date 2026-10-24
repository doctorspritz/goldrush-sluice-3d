use super::*;
use crate::soil::{ActiveSoil, Clod};
use crate::Material;
fn bank() -> crate::Session {
    let mut s = crate::Session::new();
    s.prepare_soil_trial().unwrap();
    s
}
fn same(a: &RetainedSoil, b: &RetainedSoil, label: &str) {
    assert!(
        serde_json::to_string(a.machine()).unwrap() == serde_json::to_string(b.machine()).unwrap(),
        "{label}: machine/controller changed"
    );
    assert!(
        serde_json::to_string(a.soil()).unwrap() == serde_json::to_string(b.soil()).unwrap(),
        "{label}: canonical soil changed"
    );
    let a = a.model.as_ref().unwrap();
    let b = b.model.as_ref().unwrap();
    for (h, body) in a.world.bodies.iter() {
        let other = &b.world.bodies[h];
        for (x, y) in body
            .translation()
            .to_array()
            .into_iter()
            .chain(body.rotation().to_array())
            .chain(body.linvel().to_array())
            .chain(body.angvel().to_array())
            .zip(
                other
                    .translation()
                    .to_array()
                    .into_iter()
                    .chain(other.rotation().to_array())
                    .chain(other.linvel().to_array())
                    .chain(other.angvel().to_array()),
            )
        {
            assert_eq!(x.to_bits(), y.to_bits(), "{label}: physical body {h:?}");
        }
    }
}
#[test]
fn complete_retained_bank_owner_clones_and_restores_through_real_contact() {
    let mut s = bank();
    let obstacles = s.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&s.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut a = RetainedSoil::new(s.machine.clone(), s.active_soil.take().unwrap(), &env).unwrap();
    let mut b = RetainedSoil::decode(&a.encode().unwrap(), &env).unwrap();
    let mut c = a.clone();
    let initial = serde_json::to_string(&a.soil().material()).unwrap();
    for i in 0..64 {
        let input = MachineInput {
            curl: 0.1 * (i as f32 * 0.025).sin(),
            ..Default::default()
        };
        a.step(input, 1. / 240., &env).unwrap();
        b.step(input, 1. / 240., &env).unwrap();
        c.step(input, 1. / 240., &env).unwrap();
        if i % 8 == 7 {
            same(&a, &b, "restored");
            same(&a, &c, "cloned");
        }
        if i == 31 {
            let bytes = a.encode().unwrap();
            println!("retained full bank boundary bytes={}", bytes.len());
            b = RetainedSoil::decode(&bytes, &env).unwrap();
        }
    }
    assert_eq!(a.soil().clods.len(), 1040);
    assert_eq!(
        serde_json::to_string(&a.soil().material()).unwrap(),
        initial
    );
    let impulses = a
        .model
        .as_ref()
        .unwrap()
        .world
        .narrow_phase
        .contact_pairs()
        .flat_map(|p| p.solver_manifolds())
        .flat_map(|m| &m.points)
        .filter(|p| p.data.warmstart_impulse > 0. && p.data.warmstart_tangent_impulse.norm() > 0.)
        .count();
    assert!(impulses > 0);
    println!("real retained normal/friction histories={impulses}");
}
#[test]
fn refused_inputs_geometry_and_corrupt_owners_cannot_reset_contact_history() {
    let mut s = bank();
    let obstacles = s.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&s.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut a = RetainedSoil::new(s.machine.clone(), s.active_soil.take().unwrap(), &env).unwrap();
    a.step(MachineInput::default(), 1. / 240., &env).unwrap();
    let before = a.encode().unwrap();
    for (input, dt) in [
        (
            MachineInput {
                curl: f32::NAN,
                ..Default::default()
            },
            1. / 240.,
        ),
        (MachineInput::default(), 0.),
    ] {
        assert!(a.step(input, dt, &env).is_err());
        assert!(before == a.encode().unwrap());
        assert!(!a.failed());
    }
    let obstacle = [Obstacle {
        center: Vec3::new(20., 2., 20.),
        half_extents: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }];
    let changed = Environment {
        obstacles: &obstacle,
        ..env
    };
    assert!(a
        .step(MachineInput::default(), 1. / 240., &changed)
        .is_err());
    assert!(before == a.encode().unwrap());
    assert!(RetainedSoil::decode(&before, &changed).is_err());
    for corruption in 0..3 {
        let mut saved: Saved = codec().deserialize(&before[HEADER..]).unwrap();
        match corruption {
            0 => saved.bodies.swap(0, 4),
            1 => {
                let mut machine: Excavator = serde_json::from_str(&saved.machine_json).unwrap();
                machine.position.x += 1.;
                saved.machine_json = serde_json::to_string(&machine).unwrap();
            }
            _ => {
                let mut world = ContactHistory::decode(&saved.numerical_world).unwrap();
                world
                    .world_mut()
                    .integration_parameters
                    .num_solver_iterations = 1;
                saved.numerical_world = world.encode().unwrap();
            }
        }
        let mut bytes = before[..HEADER].to_vec();
        bytes.extend(codec().serialize(&saved).unwrap());
        assert!(
            RetainedSoil::decode(&bytes, &env).is_err(),
            "corrupt owner/configuration {corruption} admitted"
        );
    }
    let mut trailing = before.clone();
    trailing.push(0);
    assert!(RetainedSoil::decode(&trailing, &env).is_err());
    assert!(RetainedSoil::decode(&before[..before.len() - 1], &env).is_err());
}
fn free_machine(swing: f32) -> Excavator {
    let mut m = Excavator::new(&Site::new());
    m.position = Vec3::new(16., 10., 16.);
    m.boom = 1.1;
    m.stick = -1.5;
    m.curl = -0.2;
    m.swing = swing;
    m.engine_on = false;
    m.enable_articulation().unwrap();
    m
}
fn prism() -> [DVec3; 6] {
    [
        DVec3::ZERO,
        DVec3::new(0.2, 0., 0.),
        DVec3::new(0., 0., 0.2),
        DVec3::new(0., 0.2, 0.),
        DVec3::new(0.2, 0.2, 0.),
        DVec3::new(0., 0.2, 0.2),
    ]
}
fn clod() -> Clod {
    Clod::prism(
        Material {
            sand_m3: 0.004,
            ..Default::default()
        },
        prism(),
    )
    .unwrap()
}
fn totals(m: &Excavator, s: &ActiveSoil) -> (DVec3, DVec3, f64) {
    let f = &m.articulation.as_ref().unwrap().feedback;
    let t = s.totals(Vec3::ZERO);
    (
        f.linear_momentum_ns.as_dvec3() + t.momentum_ns,
        f.angular_momentum_nms.as_dvec3() + t.angular_nms,
        f.kinetic_j + f.elastic_j + t.kinetic_j,
    )
}
#[test]
fn retained_soil_impact_closes_momentum_and_dissipates_at_two_timesteps() {
    for hz in [240, 480] {
        let mut m = free_machine(0.);
        m.tracks.mechanical_hz = hz;
        let q = m.bucket_rotation();
        let floor = m.arm().wrist + q * Vec3::new(0., -0.65, 0.4);
        let mut clod = clod();
        clod.center = floor + q * (clod.center + Vec3::new(-0.1, 0.14, -0.1));
        clod.rotation = q;
        clod.velocity = q * (-Vec3::Y);
        let soil = ActiveSoil::from_clods(vec![clod]).unwrap();
        let before = totals(&m, &soil);
        let env = Environment::free(Vec3::ZERO);
        let mut a = RetainedSoil::new(m, soil, &env).unwrap();
        let mut contacts = 0;
        let mut residual = 0f64;
        for _ in 0..hz / 2 {
            a.step(MachineInput::default(), 1. / hz as f32, &env)
                .unwrap();
            assert!(a.soil().feedback.closed_free_system);
            contacts += a.soil().feedback.bucket_contacts;
            residual = residual.max(a.soil().feedback.coupled_passivity_residual_j);
        }
        let after = totals(a.machine(), a.soil());
        println!(
            "retained impact {hz}Hz: contacts={contacts},dP={},dL={},dE={},max_residual={residual}",
            (after.0 - before.0).length(),
            (after.1 - before.1).length(),
            after.2 - before.2
        );
        assert!(contacts > 0);
        assert!((after.0 - before.0).length() < 0.003);
        assert!((after.1 - before.1).length() < 0.03);
        assert!(after.2 <= before.2 + 0.01);
        assert!(residual < 0.01);
    }
}
#[test]
fn retained_swing_chart_crosses_pi_without_losing_saved_contact_state_or_momentum() {
    let mut m = free_machine(3.13);
    m.engine_on = true;
    let mut clod = clod();
    clod.center = Vec3::new(25., 12., 25.);
    let soil = ActiveSoil::from_clods(vec![clod]).unwrap();
    let before = totals(&m, &soil);
    let initial_com = (m
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .center_of_mass
        .as_dvec3()
        * m.articulation.as_ref().unwrap().feedback.mass_kg
        + soil.totals(Vec3::ZERO).first_kg_m)
        / (m.articulation.as_ref().unwrap().feedback.mass_kg + soil.totals(Vec3::ZERO).mass_kg);
    let env = Environment::free(Vec3::ZERO);
    let mut a = RetainedSoil::new(m, soil, &env).unwrap();
    let mut previous = 3.13;
    let mut wraps = 0;
    let mut restored = None;
    for tick in 0..120 {
        let input = MachineInput {
            swing: 1.,
            ..Default::default()
        };
        a.step(input, 1. / 240., &env).unwrap();
        if let Some(b) = restored.as_mut() {
            RetainedSoil::step(b, input, 1. / 240., &env).unwrap();
            if tick % 16 == 15 {
                same(&a, b, "swing restoration");
            }
        }
        let coordinate = a.machine().articulation.as_ref().unwrap().coordinates[0];
        if previous > 3. && coordinate < -3. {
            wraps += 1;
            restored = Some(RetainedSoil::decode(&a.encode().unwrap(), &env).unwrap());
        }
        previous = coordinate;
    }
    let after = totals(a.machine(), a.soil());
    println!(
        "retained swing chart: wraps={wraps},dP={},dL={}",
        (after.0 - before.0).length(),
        (after.1 - before.1).length()
    );
    assert!(wraps > 0);
    // Match the existing half-second powered-actuator budget, distinct from
    // the stricter unpowered, small-clod impact control above. Origin angular
    // momentum includes COM cross linear drift; use the intrinsic quantity.
    let feedback = &a.machine().articulation.as_ref().unwrap().feedback;
    let st = a.soil().totals(Vec3::ZERO);
    let final_com = (feedback.center_of_mass.as_dvec3() * feedback.mass_kg + st.first_kg_m)
        / (feedback.mass_kg + st.mass_kg);
    let intrinsic = after.1 - final_com.cross(after.0);
    println!(
        "powered wrap: centroid drift={}m, intrinsic momentum={}",
        (final_com - initial_com).length(),
        intrinsic.length()
    );
    assert!((final_com - initial_com).length() < 0.002);
    assert!((after.0 - before.0).length() < 1.);
    assert!(intrinsic.length() < 5.);
}

// A bounded diagnostic, deliberately separate from physics acceptance tests.
// Probe every passive source before attributing the full bank's residual.
#[test]
#[ignore = "explicit passive actuator diagnosis; no accuracy pass is implied"]
fn passive_actuator_source_probe() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let mut rows = Vec::new();
    let output = std::env::var_os("SLUICE_PASSIVITY_PROBE").expect("diagnostic output path");
    let cases = if std::env::var_os("SLUICE_PASSIVITY_BANK_ONLY").is_some() {
        vec!["bank"]
    } else {
        vec!["free", "fall", "ground"]
    };
    for case in cases {
        for retained in [false, true] {
            for elastic in [false, true] {
                for hz in [240, 480] {
                    let mut machine = session.machine.clone();
                    machine.engine_on = false;
                    machine.tracks.mechanical_hz = hz;
                    if !elastic {
                        let a = machine.articulation.as_mut().unwrap();
                        a.stiffness_nm_rad.fill(0.);
                        a.damping_nms_rad.fill(0.);
                    }
                    let env = Environment {
                        gravity: if case == "free" {
                            Vec3::ZERO
                        } else {
                            -Vec3::Y * 9.81
                        },
                        site: if matches!(case, "ground" | "bank") {
                            Some(&session.site)
                        } else {
                            None
                        },
                        obstacles: if matches!(case, "ground" | "bank") {
                            &obstacles
                        } else {
                            &[]
                        },
                        loads: &[],
                    };
                    let mut soil = if case == "bank" {
                        session.active_soil.clone().unwrap()
                    } else {
                        // The owner requires finite nonempty soil. This clod
                        // remains far above and away from the machine/terrain.
                        let mut distant = clod();
                        distant.center = Vec3::new(2., 40., 2.);
                        ActiveSoil::from_clods(vec![distant]).unwrap()
                    };
                    let mut owner = retained
                        .then(|| RetainedSoil::new(machine.clone(), soil.clone(), &env).unwrap());
                    let mut maximum = f64::NEG_INFINITY;
                    let mut sum = 0.;
                    let mut worst = serde_json::Value::Null;
                    let mut failure = None;
                    let mut completed_steps = 0;
                    for step in 0..2 * hz {
                        let result = if let Some(o) = &mut owner {
                            o.step(MachineInput::default(), 1. / hz as f32, &env)
                        } else {
                            step_soil(
                                &mut machine,
                                MachineInput::default(),
                                1. / hz as f32,
                                &env,
                                &mut soil,
                            )
                        };
                        if let Err(error) = result {
                            failure = Some(error);
                            break;
                        }
                        completed_steps = step + 1;
                        let (m, s) = owner
                            .as_ref()
                            .map(|o| (o.machine(), o.soil()))
                            .unwrap_or((&machine, &soil));
                        if case != "bank" {
                            assert_eq!(
                                s.feedback.bucket_contacts, 0,
                                "distant control touched tool"
                            );
                        }
                        let residual = s.feedback.coupled_passivity_residual_j;
                        assert!(residual.is_finite());
                        sum += residual;
                        if residual > maximum {
                            maximum = residual;
                            worst = serde_json::json!({"step":step,"position":m.position,"articulation":m.articulation,"soil_feedback":s.feedback});
                        }
                    }
                    let row = serde_json::json!({"case":case,"retained":retained,"elastic":elastic,"hz":hz,"completed":failure.is_none(),"completed_steps":completed_steps,"failure":failure,"maximum_positive_residual_j":maximum.max(0.),"sum_residual_j":sum,"worst":worst});
                    println!("PASSIVE PROBE case={case} retained={retained} elastic={elastic} hz={hz} steps={completed_steps}/{} failure={failure:?} max={maximum} sum={sum}",2*hz);
                    rows.push(row);
                    std::fs::write(&output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
                }
            }
        }
    }
}

#[test]
fn zero_force_retained_hinges_keep_free_motion_and_saved_continuation() {
    let mut m = free_machine(0.);
    m.tracks.mechanical_hz = 960;
    let a = m.articulation.as_mut().unwrap();
    a.stiffness_nm_rad.fill(0.);
    a.damping_nms_rad.fill(0.);
    a.velocity[9] = 0.3;
    let mut distant = clod();
    distant.center = Vec3::new(2., 40., 2.);
    let soil = ActiveSoil::from_clods(vec![distant]).unwrap();
    let env = Environment::free(Vec3::ZERO);
    let mut owner = RetainedSoil::new(m.clone(), soil.clone(), &env).unwrap();
    let mut restored = RetainedSoil::decode(&owner.encode().unwrap(), &env).unwrap();
    let mut cold_machine = m;
    let mut cold_soil = soil;
    for _ in 0..8 {
        owner
            .step(MachineInput::default(), 1. / 960., &env)
            .unwrap();
        restored
            .step(MachineInput::default(), 1. / 960., &env)
            .unwrap();
        step_soil(
            &mut cold_machine,
            MachineInput::default(),
            1. / 960.,
            &env,
            &mut cold_soil,
        )
        .unwrap();
        same(&owner, &restored, "free actuator restore");
        for machine in [owner.machine(), &cold_machine] {
            let a = machine.articulation.as_ref().unwrap();
            assert!(
                a.velocity[9] > 0.29,
                "zero-force hinge was braked: {}",
                a.velocity[9]
            );
            assert_eq!(a.pump_energy_j, 0.);
            assert_eq!(a.feedback.elastic_j, 0.);
        }
        let model = owner.model.as_ref().unwrap();
        let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
        assert!(mb.links().skip(1).all(|link| !link
            .joint
            .data
            .motor_axes
            .contains(r::JointAxesMask::ANG_X)));
    }
}

// A portable diagnostic witness for the remaining contact/dynamics failure.
// A successful test means restore/replay and energy observation agree. It is
// explicitly not a passivity acceptance gate; the JSON records that gate.
#[test]
#[ignore = "explicit full-bank energy witness; known passivity failure"]
fn zero_actuator_bank_energy_witness() {
    let mut session = bank();
    session.machine.engine_on = false;
    session.machine.tracks.mechanical_hz = 480;
    let a = session.machine.articulation.as_mut().unwrap();
    a.stiffness_nm_rad.fill(0.);
    a.damping_nms_rad.fill(0.);
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut owner = RetainedSoil::new(
        session.machine.clone(),
        session.active_soil.take().unwrap(),
        &env,
    )
    .unwrap();
    let output = std::path::PathBuf::from(
        std::env::var_os("SLUICE_ENERGY_WITNESS").expect("fresh witness directory"),
    );
    std::fs::create_dir(&output).unwrap();
    let energy = |owner: &RetainedSoil| {
        owner
            .model
            .as_ref()
            .unwrap()
            .world
            .bodies
            .iter()
            .filter(|(_, b)| b.is_dynamic())
            .map(|(_, b)| {
                let m = b.mass() as f64;
                let v = sv(b.linvel()).as_dvec3();
                let omega = sv(b.angvel()).as_dvec3();
                let inertia = b.mass_properties().effective_angular_inertia();
                let columns = [r::Vector::X, r::Vector::Y, r::Vector::Z]
                    .map(|axis| sv(inertia * axis).as_dvec3());
                let tensor = DMat3::from_cols(columns[0], columns[1], columns[2]);
                0.5 * m * v.length_squared() + 0.5 * omega.dot(tensor * omega)
                    - m * env
                        .gravity
                        .as_dvec3()
                        .dot(sv(b.center_of_mass()).as_dvec3())
            })
            .sum::<f64>()
    };
    for step in 0..300 {
        if step == 299 {
            let before = owner.encode().unwrap();
            let initial = energy(&owner);
            std::fs::write(output.join("before.bin"), &before).unwrap();
            let mut restored = RetainedSoil::decode(&before, &env).unwrap();
            owner
                .step(MachineInput::default(), 1. / 480., &env)
                .unwrap();
            restored
                .step(MachineInput::default(), 1. / 480., &env)
                .unwrap();
            same(&owner, &restored, "zero actuator energy witness");
            let delta = energy(&owner) - initial;
            let reported = owner.soil().feedback.coupled_passivity_residual_j;
            let model = owner.model.as_ref().unwrap();
            let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
            assert!(mb.links().skip(1).all(|link| !link
                .joint
                .data
                .motor_axes
                .contains(r::JointAxesMask::ANG_X)));
            let actuator = owner.machine().articulation.as_ref().unwrap();
            assert_eq!(actuator.pump_energy_j, 0.);
            assert_eq!(actuator.feedback.elastic_j, 0.);
            assert!(
                (delta - reported).abs() < 0.05,
                "feedback disagrees with independent body energy: {delta} versus {reported}"
            );
            std::fs::write(output.join("after.bin"), owner.encode().unwrap()).unwrap();
            let report = serde_json::json!({"hz":480,"steps_before":299,"clods":owner.soil().clods.len(),
                "dynamic_bodies":model.world.bodies.iter().filter(|(_,b)|b.is_dynamic()).count(),
                "independent_energy_delta_j":delta,"reported_residual_j":reported,
                "actuator_motor_rows":0,"pump_work_j":actuator.pump_energy_j,"elastic_energy_j":actuator.feedback.elastic_j,
                "restored_step_exact":true,"passivity_gate_j":0.01,"passivity_gate_passed":delta<=0.01});
            println!("ENERGY WITNESS {report}");
            std::fs::write(
                output.join("report.json"),
                serde_json::to_vec_pretty(&report).unwrap(),
            )
            .unwrap();
        } else {
            owner
                .step(MachineInput::default(), 1. / 480., &env)
                .unwrap();
        }
    }
}

// Replay the preserved failing boundary without paying for another warm-up.
// This observes the error; it must never be reported as a passivity gate.
#[test]
#[ignore = "explicit saved-bank energy-source diagnostic"]
fn saved_bank_energy_source_breakdown() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let bytes = std::fs::read(
        std::env::var_os("SLUICE_ENERGY_BOUNDARY").expect("saved before.bin boundary"),
    )
    .unwrap();
    let output = std::path::PathBuf::from(
        std::env::var_os("SLUICE_ENERGY_BREAKDOWN").expect("fresh output JSON"),
    );
    assert!(!output.exists());
    let before = RetainedSoil::decode(&bytes, &env).unwrap();
    let mut after = RetainedSoil::decode(&bytes, &env).unwrap();
    let mut engine = RetainedSoil::decode(&bytes, &env).unwrap();
    let reference = RetainedSoil::decode(
        &std::fs::read(
            std::env::var_os("SLUICE_ENERGY_REFERENCE").expect("saved after.bin boundary"),
        )
        .unwrap(),
        &env,
    )
    .unwrap();
    let components = |body: &r::RigidBody| {
        let mass = body.mass() as f64;
        let velocity = sv(body.linvel()).as_dvec3();
        let omega = sv(body.angvel()).as_dvec3();
        let inertia = body.mass_properties().effective_angular_inertia();
        let columns =
            [r::Vector::X, r::Vector::Y, r::Vector::Z].map(|axis| sv(inertia * axis).as_dvec3());
        let tensor = DMat3::from_cols(columns[0], columns[1], columns[2]);
        [
            0.5 * mass * velocity.length_squared(),
            0.5 * omega.dot(tensor * omega),
            -mass
                * env
                    .gravity
                    .as_dvec3()
                    .dot(sv(body.center_of_mass()).as_dvec3()),
        ]
    };
    let snapshot = |owner: &RetainedSoil| {
        owner
            .model
            .as_ref()
            .unwrap()
            .world
            .bodies
            .iter()
            .filter(|(_, body)| body.is_dynamic())
            .map(|(handle, body)| (handle, components(body)))
            .collect::<std::collections::HashMap<_, _>>()
    };
    let initial = snapshot(&before);
    after
        .step(MachineInput::default(), 1. / 480., &env)
        .unwrap();
    same(&after, &reference, "preserved bank energy boundary");
    let model = engine.model.as_mut().unwrap();
    let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
    assert!(mb.links().skip(1).all(|link| !link
        .joint
        .data
        .motor_axes
        .contains(r::JointAxesMask::ANG_X)));
    model.world.integration_parameters.dt = 1. / 480.;
    model.world.step_with_events(
        &Belts {
            forward: *model.world.bodies[model.bodies[0]].rotation() * r::Vector::Z,
            speed: [0.; 2],
        },
        &(),
    );
    let raw = snapshot(&engine);
    let model = engine.model.as_mut().unwrap();
    let (mb, _) = model
        .world
        .multibody_joints
        .get_mut(model.joints[0])
        .unwrap();
    let velocity: [f32; 10] = mb.generalized_velocity().as_slice().try_into().unwrap();
    mb.forward_kinematics(&model.world.bodies, false);
    mb.update_rigid_bodies(&mut model.world.bodies, true);
    model.set_velocity(&velocity);
    // The phase-only reproduction must reach the exact authoritative body
    // state, not a superficially similar alternative force/controller path.
    let expected = &after.model.as_ref().unwrap().world.bodies;
    for (handle, body) in model.world.bodies.iter() {
        let other = &expected[handle];
        assert_eq!(
            body.position(),
            other.position(),
            "phase replay pose {handle:?}"
        );
        assert_eq!(
            body.linvel(),
            other.linvel(),
            "phase replay velocity {handle:?}"
        );
        assert_eq!(
            body.angvel(),
            other.angvel(),
            "phase replay spin {handle:?}"
        );
    }
    let final_state = snapshot(&after);
    let model = after.model.as_ref().unwrap();
    let label = |handle: Option<r::RigidBodyHandle>| {
        handle
            .map(|h| {
                if let Some(i) = model.bodies.iter().position(|x| *x == h) {
                    format!("machine-{i}")
                } else if let Some(i) = model.soil_handles.iter().position(|x| *x == h) {
                    format!("clod-{i}")
                } else {
                    format!("body-{h:?}")
                }
            })
            .unwrap_or("static".into())
    };
    let mut rows = Vec::new();
    let mut totals = [[0f64; 3]; 3];
    for (handle, value) in &initial {
        let end = final_state[handle];
        let mid = raw[handle];
        for (axis, &component) in value.iter().enumerate() {
            totals[0][axis] += component;
            totals[1][axis] += mid[axis];
            totals[2][axis] += end[axis];
        }
        let delta: [f64; 3] = std::array::from_fn(|i| end[i] - value[i]);
        let body = &model.world.bodies[*handle];
        rows.push(serde_json::json!({"body":label(Some(*handle)),"mass_kg":body.mass(),"before_j":value,"engine_j":mid,"after_j":end,"delta_j":delta,"delta_total_j":delta.iter().sum::<f64>(),"after_position":body.translation().to_array(),"after_velocity":body.linvel().to_array(),"after_spin":body.angvel().to_array()}));
    }
    rows.sort_by(|a, b| {
        b["delta_total_j"]
            .as_f64()
            .unwrap()
            .abs()
            .total_cmp(&a["delta_total_j"].as_f64().unwrap().abs())
    });
    let mut contacts = Vec::new();
    for pair in model.world.narrow_phase.contact_pairs() {
        let a = &model.world.colliders[pair.collider1];
        let b = &model.world.colliders[pair.collider2];
        for manifold in pair.solver_manifolds() {
            for point in &manifold.points {
                if point.data.impulse > 0. {
                    contacts.push(serde_json::json!({"a":label(a.parent()),"b":label(b.parent()),"impulse_ns":point.data.impulse,"distance_m":point.dist,"normal":manifold.data.normal.to_array()}));
                }
            }
        }
    }
    contacts.sort_by(|a, b| {
        b["impulse_ns"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["impulse_ns"].as_f64().unwrap())
    });
    // These are one-step attribution controls, not production parameter
    // changes. A refused case remains a refusal; no angle is clamped/admitted.
    let mut variants = Vec::new();
    let initial_energy: f64 = initial.values().flatten().sum();
    for (name, stabilization, tgs, steps, no_bias, no_history) in [
        ("baseline", 1, 4, 1, false, false),
        ("relax-2", 2, 4, 1, false, false),
        ("relax-4", 4, 4, 1, false, false),
        ("relax-8", 8, 4, 1, false, false),
        ("relax-16", 16, 4, 1, false, false),
        ("relax-64", 64, 4, 1, false, false),
        ("friction-bias", 1, 4, 1, false, false),
        ("warmstart-half", 1, 4, 1, false, false),
        ("no-bias", 1, 4, 1, true, false),
        ("no-history", 1, 4, 1, false, true),
        ("contact-refresh-4", 1, 1, 4, false, false),
        ("contact-refresh-4-relax-4", 4, 1, 4, false, false),
        ("tgs-8", 1, 8, 1, false, false),
        ("tgs-16", 1, 16, 1, false, false),
    ] {
        let mut variant = RetainedSoil::decode(&bytes, &env).unwrap();
        let parameters = &mut variant.model.as_mut().unwrap().world.integration_parameters;
        parameters.num_internal_stabilization_iterations = stabilization;
        parameters.num_solver_iterations = tgs;
        if no_bias {
            parameters.normalized_max_corrective_velocity = 0.;
        }
        if no_history {
            parameters.warmstart_coefficient = 0.;
        }
        if name == "friction-bias" {
            parameters.friction_in_bias_pass = true;
        }
        if name == "warmstart-half" {
            parameters.warmstart_coefficient *= 0.5;
        }
        let warmstart_coefficient = parameters.warmstart_coefficient;
        let friction_in_bias_pass = parameters.friction_in_bias_pass;
        let started = std::time::Instant::now();
        let mut failure = None;
        for _ in 0..steps {
            if let Err(error) =
                variant.step(MachineInput::default(), 1. / (480 * steps) as f32, &env)
            {
                failure = Some(error);
                break;
            }
        }
        let delta = if failure.is_none() {
            Some(snapshot(&variant).values().flatten().sum::<f64>() - initial_energy)
        } else {
            None
        };
        variants.push(serde_json::json!({"name":name,"stabilization":stabilization,"tgs":tgs,"outer_steps":steps,"no_bias":no_bias,"no_history":no_history,"warmstart_coefficient":warmstart_coefficient,"friction_in_bias_pass":friction_in_bias_pass,"accepted":failure.is_none(),"failure":failure,"delta_energy_j":delta,"step_wall_ms":started.elapsed().as_secs_f64()*1000.}));
    }
    let report = serde_json::json!({"energy_components":["linear_kinetic","angular_kinetic","gravity"],"phase_totals_j":totals,"reported_residual_j":after.soil().feedback.coupled_passivity_residual_j,"body_rows":rows,"largest_contacts":contacts.iter().take(32).collect::<Vec<_>>(),"contact_points":contacts.len(),"phase_replay_body_state_exact":true,"saved_reference_after_state_exact":true,"attribution_variants":variants,"variants_are_not_timestep_convergence_tests":true,"accuracy_gate_passed":false});
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!(
        "ENERGY SOURCE totals={totals:?}, controls={}",
        report["attribution_variants"]
    );
}
