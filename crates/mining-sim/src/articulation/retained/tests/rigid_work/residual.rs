//! Explicit one-step, full-inventory coupled convergence comparison.
use super::*;

#[test]
#[ignore = "explicit complete saved-bank fixed-point convergence audit; no gameplay adoption"]
fn saved_bank_coupled_residual() {
    let registered = match std::env::var("SLUICE_RESIDUAL_MODE").unwrap().as_str() {
        "baseline" => false,
        "registered" => true,
        other => panic!("undeclared residual mode: {other}"),
    };
    let pgs: usize = std::env::var("SLUICE_RESIDUAL_PGS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([16, 32, 64, 128].contains(&pgs));
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let boundary = std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap();
    let reference = std::fs::read(std::env::var_os("SLUICE_ENERGY_REFERENCE").unwrap()).unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("SLUICE_RIGID_WORK").unwrap());
    std::fs::create_dir(&output).unwrap();
    let prepare = || {
        let mut owner = RetainedSoil::decode(&boundary, &env).unwrap();
        let changes = install_trial(
            &mut owner,
            if registered {
                WorkTrial::SatFeatures
            } else {
                WorkTrial::SavedBaseline
            },
        );
        let params = &mut owner.model.as_mut().unwrap().world.integration_parameters;
        assert_eq!(params.num_internal_pgs_iterations, 16);
        let before = serde_json::to_value(*params).unwrap();
        params.num_internal_pgs_iterations = pgs;
        let mut expected = before.clone();
        expected["num_internal_pgs_iterations"] = json!(pgs);
        assert_eq!(serde_json::to_value(*params).unwrap(), expected);
        (owner, changes, before)
    };
    let (mut owner, changes, original_params) = prepare();
    let inventory = passive_inventory(&owner);
    let material = serde_json::to_string(&owner.soil().material()).unwrap();
    let initial = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let colliders = owner.model.as_ref().unwrap().world.colliders.clone();
    let (result, report) = rapier3d::contact_work_audit::residual::record(
        &[
            [
                r::RigidBodyHandle::from_raw_parts(653, 0),
                r::RigidBodyHandle::from_raw_parts(654, 0),
            ],
            [
                r::RigidBodyHandle::from_raw_parts(655, 0),
                r::RigidBodyHandle::from_raw_parts(656, 0),
            ],
            [
                r::RigidBodyHandle::from_raw_parts(657, 0),
                r::RigidBodyHandle::from_raw_parts(658, 0),
            ],
        ],
        || owner.step(MachineInput::default(), 1. / 480., &env),
    );
    result.unwrap();
    let (mut control, _, _) = prepare();
    control
        .step(MachineInput::default(), 1. / 480., &env)
        .unwrap();
    assert_eq!(
        owner.encode().unwrap(),
        control.encode().unwrap(),
        "scratch probes changed numerical continuation"
    );
    if !registered && pgs == 16 {
        assert_eq!(
            owner.encode().unwrap(),
            reference,
            "baseline after-boundary differs"
        );
    }
    assert_eq!(
        material,
        serde_json::to_string(&owner.soil().material()).unwrap()
    );
    assert_eq!(inventory, passive_inventory(&owner));
    let after_colliders = &owner.model.as_ref().unwrap().world.colliders;
    assert_eq!(colliders.len(), after_colliders.len());
    for (handle, a) in colliders.iter() {
        let b = &after_colliders[handle];
        assert_eq!(a.parent(), b.parent());
        assert_eq!(a.position_wrt_parent(), b.position_wrt_parent());
        assert!(std::ptr::eq(a.shape(), b.shape()));
        if a.parent().is_none() {
            assert_eq!(a.position(), b.position());
        }
    }
    assert_eq!(report.dropped_frames, 0);
    assert_eq!(report.frames.len(), 4);
    for frame in &report.frames {
        assert_eq!(frame.groups, 1);
        assert_eq!(frame.pgs_iterations, pgs);
        assert!(frame.biased_friction && !frame.twist_not_probed);
        assert_eq!(frame.soft_slots_not_probed, 0);
        assert!(frame.rigid.rows > 0 && frame.generic_contacts.rows > 0);
        assert!(frame.generic_joints.rows > 0, "multibody limits omitted");
        assert!(!frame.watched.is_empty());
        assert!(frame
            .watched
            .iter()
            .all(|p| p.tracking_gap_after_m.is_some()));
    }
    // Re-query the complete registered convex shapes at actual captured global
    // CoM poses. This independently tests overlap; stale witness gaps alone do not.
    let world = &owner.model.as_ref().unwrap().world;
    let oracle = Oracle::new(world);
    let mut geometry = Vec::new();
    for (frame_id, frame) in report.frames.iter().enumerate() {
        for point in &frame.watched {
            let handles = point
                .colliders
                .map(|(i, g)| r::ColliderHandle::from_raw_parts(i, g));
            let collider_poses = |after: bool| {
                std::array::from_fn(|side| {
                    let collider = &world.colliders[handles[side]];
                    let body = &world.bodies[collider.parent().unwrap()];
                    let local_com =
                        DVec3::from_array(body.local_center_of_mass().to_array().map(f64::from));
                    let com = if after {
                        point.com_after_m.unwrap()[side]
                    } else {
                        point.com_before_m[side]
                    };
                    let rotation = if after {
                        point.rotation_after_xyzw.unwrap()[side]
                    } else {
                        point.rotation_before_xyzw[side]
                    };
                    let rotation = DQuat::from_array(rotation);
                    let local = collider.position_wrt_parent().unwrap();
                    let local_translation =
                        DVec3::from_array(local.translation.to_array().map(f64::from));
                    let translation =
                        DVec3::from_array(com) + rotation * (local_translation - local_com);
                    CapturedPose {
                        translation: translation.to_array(),
                        rotation_xyzw: (rotation
                            * DQuat::from_array(local.rotation.to_array().map(f64::from)))
                        .to_array(),
                    }
                })
            };
            let n = DVec3::from_array(point.force_direction1);
            let pre = oracle.pair_primitive_audit(handles, collider_poses(false), n);
            let post = oracle.pair_primitive_audit(handles, collider_poses(true), n);
            assert_eq!(pre["applicable"], true);
            assert_eq!(post["applicable"], true);
            let gap_rate = (point.tracking_gap_after_m.unwrap() - point.tracking_gap_before_m)
                / f64::from(frame.dt_s);
            geometry.push(json!({"frame":frame_id,"contact_id":point.contact_id,"bodies":point.bodies,
                "tracking_gap_rate_m_s":gap_rate,"pre_integration_normal_velocity_m_s":point.normal_velocity_m_s,
                "gap_rate_minus_frozen_jacobian_m_s":gap_rate-point.normal_velocity_m_s,
                "independent_geometry_before":pre,"independent_geometry_after":post}));
        }
    }
    let final_energy = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let delta = final_energy - initial;
    let result = json!({"registered_geometry":registered,"pgs_iterations":pgs,"declared_geometry_changes":changes,
        "original_integration_parameters":original_params,
        "actual_integration_parameters":owner.model.as_ref().unwrap().world.integration_parameters,
        "inventory":inventory,"physical_energy_before_j":initial,"physical_energy_after_j":final_energy,
        "physical_energy_gain_j":delta,"unchanged_physical_gate_j":PASSIVITY_J,"physical_gate_passed":delta<=PASSIVITY_J,
        "observer_after_state_byte_exact":true,"baseline_reference_byte_exact":!registered && pgs==16,
        "full_material_and_attachment_preservation":true,"global_residual":report,"watched_current_registered_geometry":geometry,"gameplay_adopted":false,
        "scope":"four global pre-integration barriers; isolated copies of actual rigid/generic contact and scalar/SIMD/generic joint operators. No soft/twist model, energy proxy or fresh trajectory. Independent current registered convex overlap only on watched pair."});
    write_json(&output.join("summary.json"), &result);
    println!("RESIDUAL: registered={registered} pgs={pgs} energy_gain={delta}J physical_pass={}; isolated collector leaves after-state byte-exact",delta<=PASSIVITY_J);
}
