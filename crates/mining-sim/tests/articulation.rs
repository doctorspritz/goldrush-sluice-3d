use glam::Quat;
use mining_sim::{
    articulation::{self, Environment, Load},
    Excavator, MachineInput, Site, Vec3,
};
fn plane() -> Site {
    let mut s = Site::new();
    for i in 0..s.gold_g.len() {
        s.world.bedrock_elevation[i] = 0.;
        s.world.overburden_thickness[i] = 0.;
        s.world.paydirt_thickness[i] = 0.;
        s.world.gravel_thickness[i] = 0.;
        s.world.terrain_sediment[i] = 0.;
        s.world.water_surface[i] = 0.;
    }
    s
}
fn machine(y: f32) -> Excavator {
    machine_at(Vec3::new(16., y, 16.))
}
fn machine_at(position: Vec3) -> Excavator {
    let site = plane();
    let mut m = Excavator::new(&site);
    m.position = position;
    m.boom = 1.1;
    m.stick = -1.5;
    m.curl = -0.2;
    m.enable_articulation().unwrap();
    m
}
fn run(m: &mut Excavator, input: MachineInput, seconds: f32, hz: u16, env: &Environment<'_>) {
    m.tracks.mechanical_hz = hz;
    for _ in 0..(seconds * hz as f32).round() as usize {
        articulation::step(m, input, 1. / hz as f32, env).unwrap();
        assert!(m.articulation.as_ref().unwrap().valid());
    }
}
#[test]
fn five_body_mass_and_forward_geometry_match_the_declared_machine() {
    for load in [0., 0.2] {
        let site = plane();
        let mut m = Excavator::new(&site);
        m.position = Vec3::new(16., 10., 16.);
        m.heading = 0.7;
        m.swing = 0.4;
        m.boom = 0.8;
        m.stick = -1.3;
        m.curl = -0.2;
        m.tracks.tilt = Quat::from_rotation_x(0.2) * Quat::from_rotation_z(-0.15);
        m.bucket.gravel_m3 = load;
        m.bucket.pore_water_m3 = load * 0.2;
        m.enable_articulation().unwrap();
        let arm = m.arm();
        let up = m.up();
        let uq = m.upper_rotation();
        let bq = m.bucket_rotation();
        let moments = mining_sim::bucket::geometry().steel.combine(
            mining_sim::bucket::geometry().fill_moments(m.bucket.volume(), m.bucket.wet_mass_kg()),
        );
        let bucket = arm.wrist + bq * moments.center.as_vec3();
        let payload = m.bucket.wet_mass_kg() as f32;
        let mass = 8690. + payload;
        let com = ((m.position + up * 0.325) * 2600.
            + (m.position + up * 0.70) * 700.
            + (m.position + uq * Vec3::new(0., 1.10, -0.1)) * 3000.
            + (arm.shoulder + arm.elbow) * 650.
            + (arm.elbow + arm.wrist) * 420.
            + bucket * (250. + payload))
            / mass;
        m.enable_articulation().unwrap();
        let f = &m.articulation.as_ref().unwrap().feedback;
        eprintln!(
            "geometry load={load}: mass={} COM {:?} expected {:?}",
            f.mass_kg, f.center_of_mass, com
        );
        assert!((f.mass_kg - mass as f64).abs() < 0.01);
        assert!(f.center_of_mass.distance(com) < 0.00001);
        articulation::step(
            &mut m,
            MachineInput::default(),
            1. / 240.,
            &Environment::free(Vec3::ZERO),
        )
        .unwrap();
        assert!(m.arm().heel.distance(arm.heel) < 0.00001);
        assert!(m.arm().lip.distance(arm.lip) < 0.00001);
    }
}
#[test]
fn unforced_floating_chain_falls_ballistically_without_airborne_traction() {
    let mut m = machine(10.);
    m.engine_on = false;
    let com = m.articulation.as_ref().unwrap().feedback.center_of_mass;
    run(
        &mut m,
        MachineInput {
            drive: 1.,
            steer: 1.,
            ..Default::default()
        },
        0.2,
        240,
        &Environment::free(Vec3::new(0., -9.81, 0.)),
    );
    let f = &m.articulation.as_ref().unwrap().feedback;
    eprintln!("fall: {f:?}");
    assert!((f.center_of_mass.y - (com.y - 0.5 * 9.81 * 0.2f32.powi(2))).abs() < 0.003);
    assert!((f.linear_momentum_ns.y / (f.mass_kg as f32) + 9.81 * 0.2).abs() < 0.0003);
    assert!(Vec3::new(f.linear_momentum_ns.x, 0., f.linear_momentum_ns.z).length() < 0.1);
    assert_eq!(f.ground_normal_n, 0.);
}
#[test]
fn internal_actuation_recoils_the_base_and_preserves_free_system_momentum() {
    let mut m = machine(10.);
    let start = m.articulation.as_ref().unwrap().feedback.center_of_mass;
    let root = m.position;
    run(
        &mut m,
        MachineInput {
            boom: -1.,
            swing: 0.3,
            ..Default::default()
        },
        0.5,
        240,
        &Environment::free(Vec3::ZERO),
    );
    let f = &m.articulation.as_ref().unwrap().feedback;
    eprintln!(
        "recoil: deltaCOM={:?} root={:?} P={:?} L={:?} q={:?}",
        f.center_of_mass - start,
        m.position - root,
        f.linear_momentum_ns,
        f.angular_momentum_about_com_nms,
        m.articulation.as_ref().unwrap().coordinates
    );
    assert!(m.position.distance(root) > 0.005, "no base recoil");
    assert!(f.center_of_mass.distance(start) < 0.002);
    assert!(f.linear_momentum_ns.length() < 1.);
    // Translation error at a remote world origin must not be confused with
    // intrinsic angular error: L_origin = L_COM + COM × P.
    assert!(f.angular_momentum_about_com_nms.length() < 5.);
}
#[test]
fn finite_pump_work_and_passive_elastic_actuators_do_not_invent_energy() {
    let mut m = machine(10.);
    m.hydraulics.tool_power_w = 2500.;
    let dt = 1. / 240.;
    for _ in 0..120 {
        articulation::step(
            &mut m,
            MachineInput {
                boom: -1.,
                stick: 0.5,
                swing: 0.2,
                curl: 0.7,
                ..Default::default()
            },
            dt,
            &Environment::free(Vec3::ZERO),
        )
        .unwrap();
        let s = m.articulation.as_ref().unwrap();
        assert!(s.feedback.pump_work_j <= 2500. * dt as f64 + 1e-6);
        assert!(
            s.feedback.kinetic_j + s.feedback.elastic_j + s.relief_heat_j <= s.pump_energy_j + 0.5
        );
    }
    let s = m.articulation.as_ref().unwrap();
    let initial = s.feedback.kinetic_j + s.feedback.elastic_j + s.relief_heat_j;
    let pump = s.pump_energy_j;
    m.engine_on = false;
    run(
        &mut m,
        MachineInput::default(),
        0.5,
        240,
        &Environment::free(Vec3::ZERO),
    );
    let s = m.articulation.as_ref().unwrap();
    let final_energy = s.feedback.kinetic_j + s.feedback.elastic_j + s.relief_heat_j;
    eprintln!(
        "passivity: before={initial} after={final_energy} pump={pump} residual={}",
        s.feedback.passivity_residual_j
    );
    assert_eq!(s.pump_energy_j, pump);
    assert!(final_energy <= initial + 0.5);
}
#[test]
fn applied_bucket_force_has_the_complete_system_impulse_and_moment() {
    let mut m = machine(10.);
    let p = m.arm().lip;
    let load = Load {
        part: 4,
        force_n: Vec3::X * 1000.,
        point: p,
        torque_nm: Vec3::ZERO,
    };
    let dt = 1. / 240.;
    articulation::step(
        &mut m,
        MachineInput::default(),
        dt,
        &Environment {
            gravity: Vec3::ZERO,
            site: None,
            obstacles: &[],
            loads: &[load],
        },
    )
    .unwrap();
    let f = &m.articulation.as_ref().unwrap().feedback;
    let impulse = load.force_n * dt;
    let moment = p.cross(impulse);
    eprintln!(
        "external: P={:?} expected={impulse:?}; L={:?} expected={moment:?}",
        f.linear_momentum_ns, f.angular_momentum_nms
    );
    assert!(f.linear_momentum_ns.distance(impulse) < 0.01);
    assert!(f.angular_momentum_nms.distance(moment) < 0.1);
}
#[test]
fn supported_joint_chain_holds_payload_and_drives_through_real_track_contacts() {
    let site = plane();
    let env = Environment {
        gravity: Vec3::new(0., -9.81, 0.),
        site: Some(&site),
        obstacles: &[],
        loads: &[],
    };
    let mut m = machine(0.);
    run(&mut m, MachineInput::default(), 2., 240, &env);
    let f = &m.articulation.as_ref().unwrap().feedback;
    eprintln!(
        "equilibrium: body={:?} boom={} feedback={f:?}",
        m.position, m.boom
    );
    assert!((f.ground_normal_n / (f.mass_kg as f32 * 9.81) - 1.).abs() < 0.01);
    assert!(m.up().y > 0.999);
    assert!(m.position.y.abs() < 0.003);
    run(
        &mut m,
        MachineInput {
            drive: 1.,
            ..Default::default()
        },
        2.,
        240,
        &env,
    );
    assert!(m.position.z > 17.5);
    assert!((1.2..1.6).contains(&m.tracks.velocity.z));
    assert!(
        !m.articulation
            .as_ref()
            .unwrap()
            .feedback
            .passivity_inputs_accounted,
        "ideal belt drive work was incorrectly treated as metered"
    );
}
#[test]
fn constant_payload_core_explicitly_refuses_free_excavation_and_dump_admission() {
    let site = plane();
    let mut m = machine(0.);
    let before = m.bucket;
    articulation::step(
        &mut m,
        MachineInput {
            cut: true,
            dump: true,
            boom: -1.,
            ..Default::default()
        },
        1. / 240.,
        &Environment {
            gravity: Vec3::new(0., -9.81, 0.),
            site: Some(&site),
            obstacles: &[],
            loads: &[],
        },
    )
    .unwrap();
    assert_eq!(m.bucket, before);
    assert!(
        m.articulation
            .as_ref()
            .unwrap()
            .feedback
            .unsupported_transfer
    );
    m.bucket.gravel_m3 += 0.01;
    let pos = m.position;
    let art = serde_json::to_string(&m.articulation).unwrap();
    assert!(articulation::step(
        &mut m,
        MachineInput::default(),
        1. / 240.,
        &Environment::free(Vec3::ZERO)
    )
    .is_err());
    assert_eq!(m.position, pos);
    assert_eq!(serde_json::to_string(&m.articulation).unwrap(), art);
}
#[test]
fn mechanical_refinement_reduces_floating_centroid_drift() {
    let mut errors = Vec::new();
    for hz in [120, 240, 480] {
        let mut m = machine_at(Vec3::ZERO);
        let start = m.articulation.as_ref().unwrap().feedback.center_of_mass;
        run(
            &mut m,
            MachineInput {
                boom: -1.,
                swing: 0.3,
                ..Default::default()
            },
            0.5,
            hz,
            &Environment::free(Vec3::ZERO),
        );
        let f = &m.articulation.as_ref().unwrap().feedback;
        errors.push(f.center_of_mass.distance(start));
    }
    eprintln!("COM refinement: {errors:?}");
    assert!(errors[1] < errors[0] * 0.75);
    assert!(errors[2] < errors[1] * 0.75);
    assert!(errors[2] < 0.001);
}

#[test]
fn ballistic_refinement_has_an_independent_closed_form_reference() {
    let mut errors = Vec::new();
    for hz in [120, 240, 480] {
        let mut m = machine(10.);
        let start = m.articulation.as_ref().unwrap().feedback.center_of_mass;
        run(
            &mut m,
            MachineInput::default(),
            0.2,
            hz,
            &Environment::free(Vec3::new(0., -9.81, 0.)),
        );
        let end = m.articulation.as_ref().unwrap().feedback.center_of_mass;
        errors.push((end.y - (start.y - 0.5 * 9.81 * 0.2f32.powi(2))).abs());
    }
    eprintln!("ballistic refinement: {errors:?}");
    assert!(errors[1] < errors[0] * 0.6 && errors[2] < errors[1] * 0.6);
    assert!(errors[2] < 0.0007);
}

#[test]
fn reduced_coordinate_motor_torque_cap_matches_an_independent_cube_inertia() {
    use rapier3d::prelude as r;
    let dt = 1. / 240.;
    // A 2 kg, 1 m cube has I=2*(1^2+1^2)/12=1/3 kg m^2.
    // No gravity, no offset anchor and no other contact can supply torque.
    for axis in [r::Vector::X, r::Vector::Y, r::Vector::Z] {
        for cap in [0.5, 30., 3000.] {
            let mut world = r::PhysicsWorld::new();
            world.gravity = r::Vector::ZERO;
            world.integration_parameters.dt = dt;
            world.integration_parameters.num_solver_iterations = 4;
            world.integration_parameters.num_internal_pgs_iterations = 16;
            let ground = world.insert_body(r::RigidBodyBuilder::fixed());
            let (body, _) = world.insert(
                r::RigidBodyBuilder::dynamic().can_sleep(false),
                r::ColliderBuilder::cuboid(0.5, 0.5, 0.5).mass(2.),
            );
            world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
            let handle = world
                .insert_multibody_joint(
                    ground,
                    body,
                    r::RevoluteJointBuilder::new(axis)
                        .motor_model(r::MotorModel::ForceBased)
                        .motor_position(1., 1e6, 0.)
                        .motor_max_force(cap),
                )
                .unwrap();
            let (mb, _) = world.multibody_joints.get_mut(handle).unwrap();
            mb.forward_kinematics(&world.bodies, true);
            mb.damping_mut().fill(0.);
            mb.update_rigid_bodies(&mut world.bodies, true);
            world.step();
            let (mb, _) = world.multibody_joints.get(handle).unwrap();
            let velocity = mb.generalized_velocity()[0];
            let expected = cap * dt / (1. / 3.);
            eprintln!("torque axis={axis:?} cap={cap}: rate={velocity}, expected={expected}");
            assert!((velocity / expected - 1.).abs() < 0.0001);
        }
    }
}
#[test]
fn joystick_commands_cannot_prescribe_motion_when_the_pump_has_no_power() {
    let mut m = machine(10.);
    m.hydraulics.tool_power_w = 0.;
    let initial = [m.swing, m.boom, m.stick, m.curl];
    let root = m.position;
    run(
        &mut m,
        MachineInput {
            boom: -1.,
            stick: 1.,
            curl: 1.,
            swing: 1.,
            ..Default::default()
        },
        0.5,
        240,
        &Environment::free(Vec3::ZERO),
    );
    let s = m.articulation.as_ref().unwrap();
    assert_eq!(s.pump_energy_j, 0.);
    assert_eq!([m.swing, m.boom, m.stick, m.curl], initial);
    assert_eq!(m.position, root);
    assert_eq!(s.feedback.kinetic_j, 0.);
}
#[test]
fn dynamic_joint_limits_bound_real_angles_without_pose_clamping() {
    let mut m = machine(10.);
    for _ in 0..720 {
        articulation::step(
            &mut m,
            MachineInput {
                boom: 1.,
                stick: 1.,
                curl: 1.,
                swing: 0.3,
                ..Default::default()
            },
            1. / 240.,
            &Environment::free(Vec3::ZERO),
        )
        .unwrap();
        assert!((-0.241..=1.201).contains(&m.boom), "boom={}", m.boom);
        assert!((-2.151..=-0.149).contains(&m.stick), "stick={}", m.stick);
        assert!((-1.701..=1.451).contains(&m.curl), "curl={}", m.curl);
    }
}
#[test]
fn enabling_the_articulated_model_preserves_the_moving_legacy_com_velocity() {
    let site = plane();
    let mut m = Excavator::new(&site);
    m.position = Vec3::new(16., 10., 16.);
    m.tracks.velocity = Vec3::new(0.3, -0.2, 0.1);
    m.tracks.angular_velocity = Vec3::new(0.1, 0.2, 0.05);
    let velocity = m.tracks.velocity;
    m.enable_articulation().unwrap();
    let f = &m.articulation.as_ref().unwrap().feedback;
    assert!((f.linear_momentum_ns / f.mass_kg as f32).distance(velocity) < 0.00001);
    assert_eq!(
        &m.articulation.as_ref().unwrap().velocity[3..6],
        &[0.1, 0.2, 0.05]
    );
}

#[test]
fn complete_dynamic_and_elastic_state_survives_session_save_and_exact_continuation() {
    use mining_sim::{session::Controls, Action, Session};
    let mut s = Session::new();
    s.machine.enable_articulation().unwrap();
    s.on_action(Action::EnterExit).unwrap();
    let input = Controls {
        machine: MachineInput {
            boom: 0.25,
            swing: 0.2,
            ..Default::default()
        },
        ..Default::default()
    };
    for _ in 0..20 {
        s.step(input);
        assert!(s.machine.valid(), "{:?}", s.machine);
    }
    let saved = s.save_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(value["version"], 10);
    let mut restored = Session::load_json(&saved).unwrap();
    for _ in 0..30 {
        s.step(input);
        restored.step(input);
    }
    assert_eq!(
        serde_json::to_string(&s.machine).unwrap(),
        serde_json::to_string(&restored.machine).unwrap()
    );
    for field in ["coordinates", "reference", "velocity"] {
        let mut invalid = value.clone();
        invalid["machine"]["articulation"][field][0] = 10_000.0.into();
        assert!(
            Session::load_json(&invalid.to_string()).is_err(),
            "accepted invalid {field}"
        );
    }
    // A sine-only comparison also accepts a half-turn. Shift the reference
    // with the coordinate so pressure-strain validation cannot hide the bug.
    let mut opposite = value.clone();
    for field in ["coordinates", "reference"] {
        let old = opposite["machine"]["articulation"][field][0]
            .as_f64()
            .unwrap();
        opposite["machine"]["articulation"][field][0] = (old + std::f32::consts::PI as f64).into();
    }
    assert!(
        Session::load_json(&opposite.to_string()).is_err(),
        "accepted half-turn mechanism/render mismatch"
    );
    let mut old = value.clone();
    old["version"] = 6.into();
    assert!(Session::load_json(&old.to_string()).is_err());
    let mut absent = value;
    absent["machine"]
        .as_object_mut()
        .unwrap()
        .remove("articulation");
    assert!(Session::load_json(&absent.to_string()).is_err());
    let ordinary = Session::new().save_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&ordinary).unwrap()["version"],
        6
    );
}
#[test]
fn validation_runtime_refuses_cut_and_dump_without_changing_material_custody() {
    use mining_sim::{session::Controls, Action, Session};
    let mut s = Session::new();
    let batch = s.site.dig(Vec3::new(19., 0., 22.), 0.4, 0.3, 0.08);
    s.machine.bucket.add(batch);
    s.machine.enable_articulation().unwrap();
    s.on_action(Action::EnterExit).unwrap();
    let material = s.accounted_material();
    let bucket = s.machine.bucket;
    let gold = s.accounted_gold();
    let water = s.stored_water_m3();
    for _ in 0..30 {
        s.step(Controls {
            machine: MachineInput {
                cut: true,
                dump: true,
                boom: -0.1,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    assert_eq!(s.machine.bucket, bucket);
    assert_eq!(s.accounted_material(), material);
    assert_eq!(s.accounted_gold(), gold);
    assert_eq!(s.stored_water_m3(), water);
    assert!(s.falling.is_empty());
    assert!(matches!(
        s.machine.digging.reason,
        mining_sim::machine::DigLimit::MechanismValidation
    ));
    assert!(
        s.machine
            .articulation
            .as_ref()
            .unwrap()
            .feedback
            .unsupported_transfer
    );
}

#[test]
fn invalid_mechanical_parameters_fail_before_any_dynamic_state_changes() {
    for parameter in ["power", "friction", "pose"] {
        let mut m = machine(10.);
        match parameter {
            "power" => m.hydraulics.tool_power_w = f32::NAN,
            "friction" => m.tracks.friction = -0.1,
            _ => m.position.x = f32::INFINITY,
        }
        let before = serde_json::to_string(&m.articulation).unwrap();
        assert!(articulation::step(
            &mut m,
            MachineInput::default(),
            1. / 240.,
            &Environment::free(Vec3::ZERO)
        )
        .is_err());
        assert_eq!(serde_json::to_string(&m.articulation).unwrap(), before);
    }
}
#[test]
fn swing_coordinate_recharting_preserves_bounded_elastic_state_across_revolutions() {
    let mut m = machine(10.);
    let mut crossings = 0;
    let mut previous = 0.;
    for _ in 0..2400 {
        articulation::step(
            &mut m,
            MachineInput {
                swing: 1.,
                ..Default::default()
            },
            1. / 240.,
            &Environment::free(Vec3::ZERO),
        )
        .unwrap();
        let s = m.articulation.as_ref().unwrap();
        let q = s.coordinates[0];
        if q < -3. && previous > 3. {
            crossings += 1;
        }
        previous = q;
        assert!(s.reference.iter().all(|x| x.abs() < 4.));
        assert!(s.feedback.pump_work_j <= 30_000. / 240. + 0.00001);
    }
    eprintln!(
        "swing chart crossings={crossings}, q={:?}",
        m.articulation.as_ref().unwrap().coordinates
    );
    assert!(crossings > 0);
}

#[test]
fn explicit_external_wrench_work_is_included_in_the_passivity_ledger() {
    let mut m = machine_at(Vec3::ZERO);
    let point = m.arm().lip;
    let load = Load {
        part: 4,
        force_n: Vec3::X * 1000.,
        point,
        torque_nm: Vec3::ZERO,
    };
    articulation::step(
        &mut m,
        MachineInput::default(),
        1. / 240.,
        &Environment {
            gravity: Vec3::ZERO,
            site: None,
            obstacles: &[],
            loads: &[load],
        },
    )
    .unwrap();
    let f = &m.articulation.as_ref().unwrap().feedback;
    eprintln!(
        "wrench work={} mechanical={} residual={}",
        f.external_work_j,
        f.kinetic_j + f.elastic_j,
        f.passivity_residual_j
    );
    assert!(f.external_work_j > 0.);
    assert!(f.passivity_inputs_accounted);
    assert!(f.kinetic_j + f.elastic_j <= f.external_work_j + 0.0001);
}

#[test]
fn shell_model_is_explicit_and_cannot_be_loaded_under_a_legacy_format() {
    let mut s = mining_sim::Session::new();
    s.machine.enable_articulation().unwrap();
    let value: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    assert_eq!(value["version"], 10);
    assert_eq!(value["machine"]["articulation"]["bucket_model"], "ShellV1");
    for version in [6, 7, 8, 9] {
        let mut bad = value.clone();
        bad["version"] = version.into();
        assert!(mining_sim::Session::load_json(&bad.to_string()).is_err());
    }
    let mut bad = value.clone();
    bad["machine"]["articulation"]["bucket_model"] = "LegacyFloor".into();
    assert!(mining_sim::Session::load_json(&bad.to_string()).is_err());
    bad["machine"]["articulation"]
        .as_object_mut()
        .unwrap()
        .remove("bucket_model");
    assert!(mining_sim::Session::load_json(&bad.to_string()).is_err());
    let saved = s.save_json().unwrap();
    assert_eq!(
        mining_sim::Session::load_json(&saved)
            .unwrap()
            .save_json()
            .unwrap(),
        saved
    );
}
