//! Manufactured normal-law trial and explicit full-bank non-adoption controls.
use super::*;

#[test]
fn unqualified_normal_compliance_cannot_be_saved_and_runtime_clones_keep_it() {
    let mut session = bank();
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
    let baseline =
        serde_json::to_value(owner.model.as_ref().unwrap().world.integration_parameters).unwrap();
    owner
        .model
        .as_mut()
        .unwrap()
        .world
        .integration_parameters
        .diagnostic_preserve_normal_compliance = true;
    let selected = owner.model.as_ref().unwrap().world.integration_parameters;
    assert_eq!(
        serde_json::to_value(selected).unwrap(),
        baseline,
        "audit flag leaked into saved schema"
    );
    let decoded: r::IntegrationParameters = serde_json::from_value(baseline).unwrap();
    assert!(!decoded.diagnostic_preserve_normal_compliance);
    let cloned = owner.clone();
    assert!(
        cloned
            .model
            .as_ref()
            .unwrap()
            .world
            .integration_parameters
            .diagnostic_preserve_normal_compliance
    );
    assert!(owner.encode().unwrap_err().contains("normal compliance"));
    let history = ContactHistory::from_world(
        clone_world(&owner.model.as_ref().unwrap().world),
        owner.dispatcher,
    );
    assert!(
        history
            .world()
            .integration_parameters
            .diagnostic_preserve_normal_compliance
    );
    assert!(history.encode().unwrap_err().contains("normal compliance"));
    owner
        .model
        .as_mut()
        .unwrap()
        .world
        .integration_parameters
        .diagnostic_preserve_normal_compliance = false;
    let restored = RetainedSoil::decode(&owner.encode().unwrap(), &env).unwrap();
    assert!(
        !restored
            .model
            .as_ref()
            .unwrap()
            .world
            .integration_parameters
            .diagnostic_preserve_normal_compliance
    );
    same(&owner, &restored, "normal trial refusal control");
}

#[test]
#[ignore = "explicit full saved-bank normal-compliance experiment; no calibrated material or runtime adoption"]
fn saved_bank_normal_compliance_trial() {
    let mode = std::env::var("SLUICE_NORMAL_MODE").unwrap();
    let physical = mode == "preserve";
    assert!(physical || mode == "baseline" || mode == "registered");
    let frequency: f32 = std::env::var("SLUICE_NORMAL_FREQUENCY")
        .unwrap()
        .parse()
        .unwrap();
    assert!([30., 120., 1.0e6].contains(&frequency));
    if !physical {
        assert_eq!(frequency, 1.0e6);
    }
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
    let prepare = || {
        let mut owner = RetainedSoil::decode(&boundary, &env).unwrap();
        let trial = if mode == "baseline" {
            WorkTrial::SavedBaseline
        } else {
            WorkTrial::SatFeatures
        };
        let changes = install_trial(&mut owner, trial);
        let p = &mut owner.model.as_mut().unwrap().world.integration_parameters;
        let before = *p;
        if physical {
            p.diagnostic_preserve_normal_compliance = true;
            p.contact_softness = r::SpringCoefficients::new(frequency, 1.);
            p.static_contact_softness = p.contact_softness;
            let mut restored = *p;
            restored.diagnostic_preserve_normal_compliance = false;
            restored.contact_softness = before.contact_softness;
            restored.static_contact_softness = before.static_contact_softness;
            assert_eq!(
                restored, before,
                "normal law trial changed another parameter"
            );
        }
        assert_eq!(p.num_internal_pgs_iterations, 16);
        assert_eq!(p.num_solver_iterations, 4);
        assert_eq!(p.num_internal_stabilization_iterations, 1);
        assert!(p.friction_in_bias_pass);
        (owner, changes)
    };
    let (mut owner, changes) = prepare();
    let inventory = passive_inventory(&owner);
    let initial = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let colliders = owner.model.as_ref().unwrap().world.colliders.clone();
    let (outcome, report) = rapier3d::contact_work_audit::residual::record(
        &[[
            r::RigidBodyHandle::from_raw_parts(653, 0),
            r::RigidBodyHandle::from_raw_parts(654, 0),
        ]],
        || owner.step(MachineInput::default(), 1. / 480., &env),
    );
    outcome.unwrap();
    assert_eq!(inventory, passive_inventory(&owner));
    assert_eq!(report.dropped_frames, 0);
    assert_eq!(report.frames.len(), 4);
    let (mut control, _) = prepare();
    control
        .step(MachineInput::default(), 1. / 480., &env)
        .unwrap();
    same(&owner, &control, "normal compliance observed/control");
    // In-memory equality only: unqualified trials may not pass the save wrapper.
    assert_eq!(
        bincode::serialize(&owner.model.as_ref().unwrap().world).unwrap(),
        bincode::serialize(&control.model.as_ref().unwrap().world).unwrap()
    );
    for (handle, a) in colliders.iter() {
        let b = &owner.model.as_ref().unwrap().world.colliders[handle];
        assert_eq!(a.parent(), b.parent());
        assert_eq!(a.position_wrt_parent(), b.position_wrt_parent());
        assert!(std::ptr::eq(a.shape(), b.shape()));
    }
    if physical {
        assert!(owner.encode().unwrap_err().contains("normal compliance"));
        assert!(ContactHistory::from_world(
            clone_world(&owner.model.as_ref().unwrap().world),
            owner.dispatcher
        )
        .encode()
        .unwrap_err()
        .contains("normal compliance"));
    } else if mode == "baseline" {
        assert_eq!(owner.encode().unwrap(), reference);
    }
    let after = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let delta = after - initial;
    let output = std::path::PathBuf::from(std::env::var_os("SLUICE_RIGID_WORK").unwrap());
    std::fs::create_dir(&output).unwrap();
    write_json(
        &output.join("summary.json"),
        &json!({"mode":mode,"frequency_hz":frequency,"preserve_normal_response":physical,"geometry_changes":changes,"inventory":inventory,
        "integration_parameters":owner.model.as_ref().unwrap().world.integration_parameters,"observer_after_state_byte_exact_in_memory":true,
        "baseline_reference_byte_exact":mode=="baseline","save_refused":physical,"physical_mechanical_energy_gain_j":delta,
        "unchanged_mechanical_gate_j":PASSIVITY_J,"mechanical_gate_passed":delta<=PASSIVITY_J,"normal_spring_energy_credit_j":0.,
        "residual":report,"calibrated_soil":false,"gameplay_adopted":false,
        "limits":"A numerical per-effective-mass frequency sweep, not a material stiffness or closed global strain ledger. One saved outer step, no fresh/loaded bank, no FPS claim."}),
    );
    println!("NORMAL_BANK mode={mode} frequency={frequency}Hz physical_mechanical_gain={delta}J; unqualified strain credit=0");
}

#[test]
#[ignore = "explicit fresh 300-step complete-bank normal-law trial; no gameplay adoption"]
fn fresh_bank_normal_compliance_trial() {
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
    install_trial(&mut owner, WorkTrial::SatFeatures);
    let p = &mut owner.model.as_mut().unwrap().world.integration_parameters;
    p.diagnostic_preserve_normal_compliance = true;
    p.contact_softness = r::SpringCoefficients::new(120., 1.);
    p.static_contact_softness = p.contact_softness;
    assert_eq!(p.num_solver_iterations, 4);
    assert_eq!(p.num_internal_pgs_iterations, 16);
    assert_eq!(p.num_internal_stabilization_iterations, 1);
    let params = *p;
    let inventory = passive_inventory(&owner);
    let initial = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let mut previous = initial;
    let material = serde_json::to_string(&owner.soil().material()).unwrap();
    let mut peak = f64::NEG_INFINITY;
    let mut violations = 0;
    let mut top = Vec::new();
    for step in 0..300 {
        let (result, report) = rapier3d::contact_work_audit::residual::record(&[], || {
            owner.step(MachineInput::default(), 1. / 480., &env)
        });
        result.unwrap();
        assert_eq!(report.frames.len(), 4);
        assert_eq!(report.dropped_frames, 0);
        assert_eq!(inventory, passive_inventory(&owner));
        let energy = super::super::contact_geometry::independent_energy(&owner, env.gravity);
        let delta = energy - previous;
        peak = peak.max(delta);
        violations += usize::from(delta > PASSIVITY_J);
        let max_residual = report
            .frames
            .iter()
            .map(|f| f.rigid.maximum_normal_response_m_s)
            .fold(0f64, f64::max);
        let row = json!({"step":step,"mechanical_energy_change_j":delta,"maximum_rigid_normal_response_m_s":max_residual});
        top.push(row);
        top.sort_unstable_by(|a, b| {
            number(b, "mechanical_energy_change_j")
                .total_cmp(&number(a, "mechanical_energy_change_j"))
        });
        top.truncate(TOP);
        previous = energy;
    }
    assert_eq!(
        params,
        owner.model.as_ref().unwrap().world.integration_parameters
    );
    assert_eq!(
        material,
        serde_json::to_string(&owner.soil().material()).unwrap()
    );
    assert!(owner.encode().unwrap_err().contains("normal compliance"));
    let output = std::path::PathBuf::from(std::env::var_os("SLUICE_RIGID_WORK").unwrap());
    std::fs::create_dir(&output).unwrap();
    write_json(
        &output.join("summary.json"),
        &json!({"steps":300,"physical_time_s":300./480.,"inventory":inventory,"frequency_hz":120.,
        "integration_parameters":params,"peak_step_mechanical_energy_gain_j":peak,"mechanical_gate_j":PASSIVITY_J,"violations":violations,"net_mechanical_energy_change_j":previous-initial,
        "all_mechanical_steps_passed":violations==0,"normal_spring_energy_credit_j":0.,"top":top,"save_refused":true,"calibrated_soil":false,"gameplay_adopted":false,
        "scope":"Fresh complete passive bank, bounded four-frame observer per step; no raw trace retained. No loaded refinement, material/strain calibration or native performance claim."}),
    );
    println!("NORMAL_FRESH_300 peak={peak}J violations={violations} net={}J stock intact, no strain credit",previous-initial);
}
