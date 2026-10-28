//! Equal physical integration cadence, independently refreshed contact frames.
//! Reports close one outer step at a time; trajectories never enable row capture.
use super::*;

const MACRO_DT: f32 = 1. / 480.;
const MICRO_PASSIVITY_J: f64 = PASSIVITY_J / 4.;

#[derive(Clone, Copy)]
struct Cadence {
    calls: usize,
    material: bool,
}
impl Cadence {
    fn selected() -> Self {
        match std::env::var("SLUICE_CONTACT_CADENCE").unwrap().as_str() {
            "outer_480_legacy" => Self {
                calls: 1,
                material: false,
            },
            "outer_1920_legacy" => Self {
                calls: 4,
                material: false,
            },
            "outer_480_material" => Self {
                calls: 1,
                material: true,
            },
            "outer_1920_material" => Self {
                calls: 4,
                material: true,
            },
            other => panic!("undeclared contact cadence: {other}"),
        }
    }
    fn dt(self) -> f32 {
        MACRO_DT / self.calls as f32
    }
    fn trial(self) -> WorkTrial {
        if self.material {
            WorkTrial::SatMaterialTangents
        } else {
            WorkTrial::SatFeatures
        }
    }
    fn install(self, owner: &mut RetainedSoil) -> Value {
        assert!(matches!(self.calls, 1 | 4));
        let mut changes = install_trial(owner, self.trial());
        let p = &mut owner.model.as_mut().unwrap().world.integration_parameters;
        let before = serde_json::to_value(*p).unwrap();
        assert_eq!(p.num_solver_iterations, 4);
        assert_eq!(p.num_internal_pgs_iterations, 16);
        p.dt = self.dt();
        p.num_solver_iterations = 4 / self.calls;
        let after = serde_json::to_value(*p).unwrap();
        let mut expected = before.clone();
        expected["dt"] = json!(self.dt());
        expected["num_solver_iterations"] = json!(4 / self.calls);
        assert_eq!(
            after, expected,
            "cadence changed another integration parameter"
        );
        assert_eq!(self.dt() / p.num_solver_iterations as f32, MACRO_DT / 4.);
        assert_eq!(self.dt() * self.calls as f32, MACRO_DT);
        assert!(p.friction_in_bias_pass);
        assert_eq!(p.normalized_contact_recycle_distance, 0.);
        changes["cadence"] = json!({"world_steps_per_macro":self.calls,
            "macro_dt_s":MACRO_DT,"outer_dt_s":self.dt(),
            "temporal_substeps_per_world_step":p.num_solver_iterations,
            "physical_integration_dt_s":MACRO_DT/4.,"pgs_passes_per_substep":16,
            "parameters_before_cadence_change":before,"parameters_after_cadence_change":after,
            "other_integration_parameters_exact":true,
            "attribution":"full world refresh also changes query/cache, restitution and mass/Jacobian scheduling; not an isolated frozen-normal ablation"});
        changes
    }
}

fn assert_attachments(before: &r::ColliderSet, owner: &RetainedSoil) {
    let after = &owner.model.as_ref().unwrap().world.colliders;
    assert_eq!(before.len(), after.len());
    for (handle, a) in before.iter() {
        let b = &after[handle];
        assert_eq!(a.parent(), b.parent());
        assert_eq!(a.position_wrt_parent(), b.position_wrt_parent());
        assert!(std::ptr::eq(a.shape(), b.shape()));
        if a.parent().is_none() {
            assert_eq!(a.position(), b.position());
        }
    }
}

fn pose_summary(owner: &RetainedSoil) -> Value {
    let mut lo = r::Vector::splat(f32::INFINITY);
    let mut hi = r::Vector::splat(f32::NEG_INFINITY);
    let mut linear: f32 = 0.;
    let mut angular: f32 = 0.;
    let mut dynamic = 0;
    for (_, body) in owner.model.as_ref().unwrap().world.bodies.iter() {
        if !body.is_dynamic() {
            continue;
        }
        let position = body.position();
        assert!(position.translation.is_finite() && position.rotation.is_finite());
        assert!(body.linvel().is_finite() && body.angvel().is_finite());
        lo = lo.min(position.translation);
        hi = hi.max(position.translation);
        linear = linear.max(body.linvel().length());
        angular = angular.max(body.angvel().length());
        dynamic += 1;
    }
    assert_eq!(dynamic, 1045);
    json!({"all_dynamic_poses_and_velocities_finite":true,"dynamic_bodies":dynamic,
        "position_min_m":lo.to_array(),"position_max_m":hi.to_array(),
        "maximum_linear_speed_m_s":linear,"maximum_angular_speed_rad_s":angular})
}

#[test]
#[ignore = "explicit full-bank saved/300-macro contact-refresh comparison; no runtime adoption"]
fn registered_contact_frame_cadence() {
    let cadence = Cadence::selected();
    let saved = std::env::var_os("SLUICE_CADENCE_SAVED").is_some();
    let mut session = bank();
    session.machine.engine_on = false;
    session.machine.tracks.mechanical_hz = 480;
    let actuator = session.machine.articulation.as_mut().unwrap();
    actuator.stiffness_nm_rad.fill(0.);
    actuator.damping_nms_rad.fill(0.);
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut owner = if saved {
        let bytes = std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap();
        RetainedSoil::decode(&bytes, &env).unwrap()
    } else {
        RetainedSoil::new(
            session.machine.clone(),
            session.active_soil.take().unwrap(),
            &env,
        )
        .unwrap()
    };
    let output = std::path::PathBuf::from(std::env::var_os("SLUICE_RIGID_WORK").unwrap());
    std::fs::create_dir(&output).unwrap();
    let declared_changes = cadence.install(&mut owner);
    let initial_persistence = candidate_persistence_refusal(&owner, &env);
    let inventory = passive_inventory(&owner);
    let material = serde_json::to_string(&owner.soil().material()).unwrap();
    let parameters = owner.model.as_ref().unwrap().world.integration_parameters;
    let initial_energy = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let mut previous = initial_energy;
    let mut peak_macro = f64::NEG_INFINITY;
    let mut peak_micro = f64::NEG_INFINITY;
    let mut macro_violations = 0;
    let mut micro_violations = 0;
    let mut max_feedback_error: f64 = 0.;
    let mut max_macro_feedback_error: f64 = 0.;
    let mut rows = Vec::with_capacity(300);
    let mut reports = Vec::with_capacity(4);
    let mut refusal = None;
    let requested = if saved { 1 } else { 300 };
    let started = std::time::Instant::now();
    'macro_steps: for macro_step in 1..=requested {
        let macro_before = previous;
        let mut feedback = 0.;
        let mut micros = Vec::with_capacity(cadence.calls);
        for micro_step in 1..=cadence.calls {
            let result = if saved {
                let colliders = owner.model.as_ref().unwrap().world.colliders.clone();
                let (result, report) =
                    rapier3d::contact_work_audit::record_with_colliders(&colliders, || {
                        owner.step(MachineInput::default(), cadence.dt(), &env)
                    });
                if result.is_ok() {
                    assert_attachments(&colliders, &owner);
                    let mut summary = summarize_work_report(&owner, &report, env.gravity, previous);
                    let limit = if cadence.calls == 4 {
                        MICRO_PASSIVITY_J
                    } else {
                        PASSIVITY_J
                    };
                    summary["physical_passivity_gate_j"] = json!(limit);
                    summary["physical_passivity_gate_passed"] =
                        json!(number(&summary, "independent_energy_delta_j") <= limit);
                    summary["world_step"] = json!(micro_step);
                    write_json(&output.join(format!("work-{micro_step}.json")), &summary);
                    assert_work_summary(&summary);
                    reports.push(json!({"world_step":micro_step,"rigid_blocks":report.rigid_len(),
                        "generic_blocks":report.len(),"observer_success":summary["observer_success"],
                        "phase_minus_body_j":summary["phase_minus_body_j"]}));
                    // The full raw row report drops here before the next step.
                }
                result
            } else {
                owner.step(MachineInput::default(), cadence.dt(), &env)
            };
            if let Err(error) = result {
                refusal = Some(json!({"macro_step":macro_step,"world_step":micro_step,
                    "error":error,"owner_poisoned":owner.failed()}));
                break 'macro_steps;
            }
            assert_eq!(
                owner.model.as_ref().unwrap().world.integration_parameters,
                parameters
            );
            assert_eq!(passive_inventory(&owner), inventory);
            assert_eq!(
                serde_json::to_string(&owner.soil().material()).unwrap(),
                material
            );
            let energy = super::super::contact_geometry::independent_energy(&owner, env.gravity);
            assert!(energy.is_finite());
            let gain = energy - previous;
            let reported = owner.soil().feedback.coupled_passivity_residual_j;
            let error = (reported - gain).abs();
            assert!(error.is_finite());
            feedback += reported;
            max_feedback_error = max_feedback_error.max(error);
            if cadence.calls == 4 {
                peak_micro = peak_micro.max(gain);
                micro_violations += usize::from(gain > MICRO_PASSIVITY_J);
            }
            micros.push(
                json!({"world_step":micro_step,"independent_energy_j":energy,
                "delta_energy_j":gain,"reported_residual_j":reported,"feedback_error_j":error,
                "pose":pose_summary(&owner)}),
            );
            previous = energy;
        }
        let gain = previous - macro_before;
        let feedback_error = (feedback - gain).abs();
        peak_macro = peak_macro.max(gain);
        macro_violations += usize::from(gain > PASSIVITY_J);
        max_macro_feedback_error = max_macro_feedback_error.max(feedback_error);
        rows.push(
            json!({"macro_step":macro_step,"independent_energy_j":previous,
            "delta_energy_j":gain,"summed_feedback_j":feedback,"feedback_error_j":feedback_error,
            "inventory":passive_inventory(&owner),"world_steps":micros}),
        );
        if macro_step % 50 == 0 {
            println!("CADENCE calls={} material={} macro={macro_step}/{requested} peak_j={peak_macro:.9} violations={macro_violations} elapsed_s={:.3}",cadence.calls,cadence.material,started.elapsed().as_secs_f64());
            write_json(
                &output.join(format!("progress-{macro_step:05}.json")),
                &json!({"complete":false,"checkpoint_only":true,"accepted_macro_steps":macro_step,
                    "requested_macro_steps":requested,"elapsed_wall_s":started.elapsed().as_secs_f64(),
                    "maximum_macro_gain_j":peak_macro,"macro_energy_violations":macro_violations,
                    "maximum_micro_gain_j":(cadence.calls==4).then_some(peak_micro),
                    "micro_energy_violations":(cadence.calls==4).then_some(micro_violations),
                    "maximum_macro_feedback_error_j":max_macro_feedback_error,
                    "inventory":passive_inventory(&owner),"trajectory":rows}),
            );
            io::stdout().flush().unwrap();
        }
    }
    let completed = rows.len() == requested && refusal.is_none();
    let final_persistence = (!owner.failed()).then(|| candidate_persistence_refusal(&owner, &env));
    let summary = json!({"trial":cadence.trial().label(),"world_steps_per_macro":cadence.calls,
        "saved_failure_history_imported":saved,"fresh_start":!saved,"requested_macro_steps":requested,
        "accepted_macro_steps":rows.len(),"completed":completed,"refusal":refusal,
        "declared_changes":declared_changes,"parameters_verified_every_accepted_world_step":true,
        "all_inventory_and_material_fields_exact_every_accepted_world_step":true,"initial_inventory":inventory,
        "initial_independent_energy_j":initial_energy,"last_accepted_independent_energy_j":previous,
        "net_energy_gain_j":previous-initial_energy,"maximum_macro_gain_j":peak_macro,
        "macro_gate_j":PASSIVITY_J,"macro_energy_violations":macro_violations,
        "all_macro_gates_passed":completed && macro_violations==0,
        "micro_gate_observed":cadence.calls==4,"micro_gate_j":MICRO_PASSIVITY_J,
        "maximum_micro_gain_j":(cadence.calls==4).then_some(peak_micro),
        "micro_energy_violations":(cadence.calls==4).then_some(micro_violations),
        "all_micro_gates_passed":(cadence.calls==4).then_some(completed && micro_violations==0),
        "maximum_feedback_error_j":max_feedback_error,"maximum_macro_feedback_error_j":max_macro_feedback_error,
        "existing_feedback_agreement_limit_j":0.05,"work_row_observer_enabled":saved,
        "per_world_step_work_reports":reports,"initial_persistence_refusal":initial_persistence,
        "final_persistence_refusal":final_persistence,"candidate_boundary_files_written":false,
        "no_session_adoption":true,"elapsed_wall_s":started.elapsed().as_secs_f64(),
        "sat_diagnostics":crate::soil_contacts::sat_diagnostics_for_test(),"trajectory":rows});
    write_json(&output.join("cadence-summary.json"), &summary);
    println!("CADENCE completed={completed} calls={} material={} peak_macro_j={peak_macro:.9} macro_violations={macro_violations} peak_micro_j={:?} micro_violations={micro_violations} feedback_error_j={max_macro_feedback_error:.9}",cadence.calls,cadence.material,(cadence.calls==4).then_some(peak_micro));
    assert!(
        completed,
        "actual refusal interrupted the comparison; see bounded summary"
    );
    assert!(
        max_feedback_error < 0.05 && max_macro_feedback_error < 0.05,
        "independent energy and summed feedback disagree"
    );
}
