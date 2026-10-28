//! Read-only fixed-manifold normal-regularization hypothesis. This does not add
//! contact potential to the physical passivity gate or certify a soil modulus.
use super::*;

#[derive(Clone, Copy, Debug)]
struct Spring {
    mass: f64,
    stiffness: f64,
    damping: f64,
}
fn spring(mass: f64, h: f64, factor: f64, gain: f64) -> Option<Spring> {
    if mass <= 0. || h <= 0. || factor <= 0. || factor >= 1. {
        return None;
    }
    let gamma = (factor.recip() - 1.) / mass;
    let stiffness = gain / (h * gamma);
    let damping = (h.recip() - gain) / gamma;
    assert!(stiffness.is_finite() && stiffness > 0.);
    // Preserve signed roundoff near zero damping; do not silently clamp the
    // reconstructed law. Full-bank model admission below requires d >= 0.
    assert!(damping.is_finite());
    Some(Spring {
        mass,
        stiffness,
        damping,
    })
}
fn strain(k: f64, gap: f64) -> f64 {
    0.5 * k * gap.min(0.).powi(2)
}
// Unilateral contact cannot pull during separation. Cap the dashpot at the
// available spring traction and use the convex unilateral spring remainder;
// a full d*v^2 loss falsely reports energy creation when the row clamps to zero.
fn losses(law: Spring, before: f64, after: f64, h: f64) -> (f64, f64, f64) {
    let delta = after - before;
    let velocity = delta / h;
    let du = strain(law.stiffness, after) - strain(law.stiffness, before);
    let damping = if velocity > 0. {
        h * (law.damping * velocity).min(-law.stiffness * after.min(0.)) * velocity
    } else {
        h * law.damping * velocity * velocity
    };
    let remainder = if after >= 0. {
        strain(law.stiffness, before)
    } else if before < 0. {
        0.5 * law.stiffness * delta * delta
    } else {
        0.5 * law.stiffness * after * after - law.stiffness * after * before
    };
    (du, damping, remainder)
}
fn normal_velocity(row: &Value, point: &Value, velocity: &str) -> f64 {
    let n = vector(&point["force_direction1"]);
    row["sides"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(side, body)| {
            let v: [f64; 6] = serde_json::from_value(body[velocity].clone()).unwrap();
            let linear = DVec3::from_slice(&v[..3]);
            let angular = DVec3::from_slice(&v[3..]);
            let force = if side == 0 { n } else { -n };
            force.dot(linear) + vector(&point["normal_torque_directions"][side]).dot(angular)
        })
        .sum()
}
fn material_gap(point: &Value) -> f64 {
    (vector(&point["world_anchors_m"][0]) - vector(&point["world_anchors_m"][1]))
        .dot(vector(&point["force_direction1"]))
}
#[derive(Debug)]
struct Interval {
    key: [u32; 4],
    clustered: bool,
    h: f64,
    mass: f64,
    law: Option<Spring>,
    normal: Value,
    start_gap: f64,
    start_material_gap: f64,
    end_gap: Option<f64>,
    end_material_gap: Option<f64>,
    work_before_integration: f64,
    work_relaxation: f64,
    tangent_work: f64,
    prescribed_work: f64,
    last_bias_local_residual: Option<f64>,
    last_bias_velocity: Option<f64>,
    biased_calls: usize,
    bias_capped: bool,
    restitution_seed: bool,
}
fn fold(
    report: &rapier3d::contact_work_audit::WorkReport,
    params: r::IntegrationParameters,
) -> Value {
    let mut intervals = BTreeMap::<[u32; 4], Interval>::new();
    let mut blocks = 0;
    let mut snapshots = 0;
    let mut block_normal_work = 0.;
    let mut max_coefficient_relative_error = 0f64;
    let mut coefficient_precision_witness = Value::Null;
    for record in report.iter_rigid_contacts() {
        // Stream one serialized record; never clone or write the entire trace.
        let row = serde_json::to_value(record).unwrap();
        blocks += 1;
        block_normal_work += number(&row, "normal_midpoint_work_j");
        let h = number(&row, "inverse_substep_dt_s").recip();
        let time = number(&row, "solved_time_s") / h;
        assert!(
            (time - time.round()).abs() < 0.00001,
            "unexpected phase time {time}"
        );
        let relax = row["stage"] == "solve" && row["unbiased"] == true;
        let step = time.round() as i32 - i32::from(relax);
        assert!(step >= 0);
        let manifold: [u32; 2] = serde_json::from_value(row["manifold"].clone()).unwrap();
        for point in row["points"].as_array().unwrap() {
            snapshots += 1;
            let key = [
                step as u32,
                manifold[0],
                manifold[1],
                point["contact_id"].as_u64().unwrap() as u32,
            ];
            // The historical observer key is misnamed: np.r is m_eff, in kg.
            // The actual scalar-row response/energy controls verify this unit.
            let mass = number(point, "normal_inverse_effective_mass");
            let factor = number(point, "normal_cfm_factor");
            let gain = number(&row, "normal_bias_gain_s_inv");
            let gap = number(point, "tracking_distance_m");
            let material = material_gap(point);
            let entry = intervals.entry(key).or_insert_with(|| {
                assert!(!relax, "missing pre-integration interval start");
                let law = spring(mass, h, factor, gain);
                if let Some(law) = law {
                    assert!(
                        law.damping >= 0.,
                        "bank row has no dissipative spring mapping"
                    );
                    let ids: Vec<_> = row["sides"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|s| s["solver_body_id"].as_u64().unwrap())
                        .collect();
                    let settings = if ids.contains(&u64::from(u32::MAX)) {
                        params.static_contact_softness
                    } else {
                        params.contact_softness
                    };
                    let omega = std::f64::consts::TAU * f64::from(settings.natural_frequency);
                    let expected_k = mass * omega * omega;
                    let expected_d = 2. * f64::from(settings.damping_ratio) * mass * omega;
                    let error = (law.stiffness / expected_k - 1.)
                        .abs()
                        .max((law.damping - expected_d).abs() / expected_d.abs().max(1.));
                    // Near the rigid limit, rounding a factor close to one is
                    // ill-conditioned when inverted to infer a physical spring.
                    // Verify the actual f32 pipeline, then report that difference;
                    // do not silently replace the applied law by nominal k/d.
                    let step = h as f32;
                    let dynamic = params.contact_softness.cfm_factor(step);
                    let fixed = params.static_contact_softness.cfm_factor(step);
                    let expected_factor = if ids.contains(&u64::from(u32::MAX)) {dynamic+(fixed-dynamic)} else {dynamic};
                    assert_eq!(factor as f32,expected_factor,"actual CFM factor disagrees with its configured f32 computation");
                    assert_eq!(gain as f32,settings.erp_inv_dt(step),"recorded spring gain disagrees with configured f32 computation");
                    if error>max_coefficient_relative_error {
                        max_coefficient_relative_error=error;
                        coefficient_precision_witness=json!({"key":key,"h_s":h,"actual_cfm_factor":factor,"bias_gain_s_inv":gain,
                            "normal_effective_mass_kg":mass,"equivalent_stiffness_n_m":law.stiffness,"equivalent_damping_ns_m":law.damping,
                            "nominal_stiffness_n_m":expected_k,"nominal_damping_ns_m":expected_d,"settings":settings,
                            "float_steps_below_one":1f32.to_bits()-(factor as f32).to_bits()});
                    }
                }
                Interval {
                    key,
                    clustered: row["clustered"].as_bool().unwrap(),
                    h,
                    mass,
                    law,
                    normal: point["force_direction1"].clone(),
                    start_gap: gap,
                    start_material_gap: material,
                    end_gap: None,
                    end_material_gap: None,
                    work_before_integration: 0.,
                    work_relaxation: 0.,
                    tangent_work: 0.,
                    prescribed_work: 0.,
                    last_bias_local_residual: None,
                    last_bias_velocity: None,
                    biased_calls: 0,
                    bias_capped: -gain*gap>number(&row,"max_corrective_velocity_m_s"),
                    restitution_seed: number(point,"restitution_seed_m_s")!=0.,
                }
            });
            assert_eq!(
                entry.mass, mass,
                "fixed-manifold mass changed inside a world step"
            );
            assert_eq!(
                entry.normal, point["force_direction1"],
                "normal changed without a frame-transport ledger"
            );
            let work = number(point, "normal_midpoint_work_j");
            if relax {
                entry.work_relaxation += work;
                entry.end_gap = Some(gap);
                entry.end_material_gap = Some(material);
            } else {
                entry.work_before_integration += work;
                if row["stage"] == "solve" {
                    entry.biased_calls += 1;
                    let velocity = normal_velocity(&row, point, "after_velocity");
                    entry.last_bias_velocity = Some(velocity);
                    let impulse = number(point, "normal_impulse_after_ns");
                    let gamma = if factor > 0. && mass > 0. {
                        (factor.recip() - 1.) / mass
                    } else {
                        0.
                    };
                    entry.last_bias_local_residual =
                        Some(velocity + number(point, "normal_rhs_m_s") + gamma * impulse);
                }
            }
            entry.tangent_work += number(point, "tangent_midpoint_work_j");
            entry.prescribed_work += number(point, "prescribed_midpoint_work_j");
        }
    }
    let mut all = Vec::new();
    let mut total_work = 0.;
    let mut soft = 0;
    let mut rigid = 0;
    let mut sums = [0f64; 8];
    let mut grouped = BTreeMap::<bool, (usize, f64, f64)>::new();
    for entry in intervals.values() {
        total_work += entry.work_before_integration + entry.work_relaxation;
        let end = entry
            .end_gap
            .expect("every fixed interval must have a post-integration observation");
        let material_end = entry.end_material_gap.unwrap();
        let Some(law) = entry.law else {
            rigid += 1;
            continue;
        };
        soft += 1;
        let delta = end - entry.start_gap;
        let (du, damping, numerical) = losses(law, entry.start_gap, end, entry.h);
        let material_du =
            strain(law.stiffness, material_end) - strain(law.stiffness, entry.start_material_gap);
        let before = entry.work_before_integration + du + damping + numerical;
        let after = before + entry.work_relaxation;
        let group = grouped.entry(entry.clustered).or_default();
        group.0 += 1;
        group.1 += after;
        group.2 += after.max(0.);
        for (target, value) in sums.iter_mut().zip([
            entry.work_before_integration,
            entry.work_relaxation,
            du,
            damping,
            numerical,
            before,
            after,
            material_du,
        ]) {
            *target += value;
        }
        all.push(json!({"key_substep_edge_manifold_point":entry.key,"clustered":entry.clustered,
            "normal_effective_mass_kg":law.mass,"equivalent_stiffness_n_m":law.stiffness,"equivalent_damping_ns_m":law.damping,
            "tracking_gap_before_m":entry.start_gap,"tracking_gap_after_m":end,
            "material_witness_gap_before_m":entry.start_material_gap,"material_witness_gap_after_m":material_end,
            "tracking_minus_material_gap_increment_m":delta-(material_end-entry.start_material_gap),
            "normal_work_before_integration_j":entry.work_before_integration,"normal_relaxation_work_j":entry.work_relaxation,
            "tangent_work_j":entry.tangent_work,"prescribed_work_j":entry.prescribed_work,
            "regularization_strain_change_j":du,"material_witness_strain_proxy_change_j":material_du,
            "geometric_unilateral_dashpot_proxy_j":damping,"spring_discretization_proxy_j":numerical,
            "normal_balance_before_relaxation_j":before,"normal_balance_with_relaxation_j":after,
            "last_bias_block_local_residual_m_s":entry.last_bias_local_residual,
            "last_bias_block_normal_velocity_m_s":entry.last_bias_velocity,"geometric_average_normal_velocity_m_s":delta/entry.h,
            "last_local_velocity_minus_gap_rate_m_s":entry.last_bias_velocity.map(|v|v-delta/entry.h),"biased_calls":entry.biased_calls,
            "bias_capped":entry.bias_capped,"nonzero_restitution_seed":entry.restitution_seed}));
    }
    assert!(
        (total_work - block_normal_work).abs() < AGREEMENT_J,
        "normal-work partition failed"
    );
    let positive: f64 = all
        .iter()
        .map(|v| number(v, "normal_balance_with_relaxation_j").max(0.))
        .sum();
    let violating = all
        .iter()
        .filter(|v| number(v, "normal_balance_with_relaxation_j") > AGREEMENT_J)
        .count();
    all.sort_by(|a, b| {
        number(b, "normal_balance_with_relaxation_j")
            .total_cmp(&number(a, "normal_balance_with_relaxation_j"))
    });
    let max = all
        .first()
        .map_or(0., |v| number(v, "normal_balance_with_relaxation_j"));
    all.truncate(TOP);
    json!({"scope":"fixed-manifold normal-regularization hypothesis only; not a global potential, soil calibration, or replacement physical gate",
        "rigid_blocks":blocks,"point_snapshots":snapshots,"paired_intervals":intervals.len(),"soft_intervals":soft,"rigid_only_intervals":rigid,
        "generic_blocks_outside_model":report.len(),"max_equivalent_vs_nominal_relative_difference":max_coefficient_relative_error,"coefficient_precision_witness":coefficient_precision_witness,
        "bias_capped_intervals":intervals.values().filter(|e|e.bias_capped).count(),"nonzero_restitution_intervals":intervals.values().filter(|e|e.restitution_seed).count(),
        "proxy_applicability":"uncapped unilateral spring hypothesis; capped intervals and restitution are reported, not silently qualified",
        "proxy_version":"unilateral dashpot traction cap and convex spring remainder",
        "all_rigid_normal_work_j":total_work,"normal_work_partition_error_j":total_work-block_normal_work,
        "soft_normal_work_before_integration_j":sums[0],"soft_normal_relaxation_work_j":sums[1],"strain_change_j":sums[2],
        "geometric_unilateral_dashpot_proxy_j":sums[3],"spring_discretization_proxy_j":sums[4],"balance_before_relaxation_j":sums[5],
        "balance_with_relaxation_j":sums[6],"material_witness_strain_proxy_change_j":sums[7],
        "positive_interval_balance_sum_j":positive,"maximum_interval_balance_j":max,"positive_intervals_above_observer_tolerance":violating,
        "grouped_by_cluster_flag":grouped.iter().map(|(clustered,(count,balance,positive))|json!({"clustered":clustered,"intervals":count,"balance_j":balance,"positive_balance_sum_j":positive})).collect::<Vec<_>>(),
        "top_positive_intervals":all,"no_proxy_terms_added_to_mechanical_energy_gate":true})
}

// Retain only four complete blocks for the worst interval. A second streaming
// read keeps this reproduction bounded instead of retaining another raw trace.
fn witness(report: &rapier3d::contact_work_audit::WorkReport, wanted: [u32; 4]) -> Value {
    let mut selected = BTreeMap::<&str, Value>::new();
    for record in report.iter_rigid_contacts() {
        let row = serde_json::to_value(record).unwrap();
        let manifold: [u32; 2] = serde_json::from_value(row["manifold"].clone()).unwrap();
        if manifold != [wanted[1], wanted[2]] {
            continue;
        }
        let relax = row["stage"] == "solve" && row["unbiased"] == true;
        let index = (number(&row, "solved_time_s") * number(&row, "inverse_substep_dt_s")).round()
            as i32
            - i32::from(relax);
        if index != wanted[0] as i32
            || !row["points"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["contact_id"].as_u64() == Some(u64::from(wanted[3])))
        {
            continue;
        }
        let phase = if relax {
            "first_relaxation"
        } else if row["stage"] == "warmstart" {
            "warmstart"
        } else if row["stage"] == "projection" {
            "projection"
        } else {
            "last_biased_solve"
        };
        if phase == "last_biased_solve" || !selected.contains_key(phase) {
            selected.insert(phase, row);
        }
    }
    assert!(
        selected.contains_key("last_biased_solve") && selected.contains_key("first_relaxation")
    );
    json!({"key_substep_edge_manifold_point":wanted,"scope":"four retained blocks for reproduction; no full raw trace written","blocks":selected})
}

#[test]
fn opening_unilateral_contact_does_not_invent_dashpot_loss() {
    let h = 1. / 1920.;
    let omega = std::f64::consts::TAU * 30.;
    let law = Spring {
        mass: 2.5,
        stiffness: 2.5 * omega * omega,
        damping: 20. * 2.5 * omega,
    };
    let before = -0.001;
    let after = before + h;
    // Independent constitutive force is clamped to zero for this fast opening
    // state, so kinetic work is zero. Stored spring loss supplies all damping.
    assert!(-law.stiffness * after - law.damping < 0.);
    let (du, damping, numerical) = losses(law, before, after, h);
    assert!((du + damping + numerical).abs() < 1e-12);
    assert!(
        du + h * law.damping + 0.5 * law.stiffness * h * h > 1.,
        "negative control missed the unclipped-loss defect"
    );
    let (du, damping, numerical) = losses(law, before, 0.001, h);
    assert!(
        damping == 0. && (du + numerical).abs() < 1e-12,
        "opening a contact must not retain tensile damping"
    );
}

#[test]
fn equivalent_softness_recovers_mass_scaled_spring_without_inverting_mass() {
    for mass in [0.1, 2.5, 250.] {
        for frequency in [15f32, 30., 60.] {
            for damping in [0f32, 1., 10.] {
                let h = 1f32 / 1920.;
                let s = rapier3d::dynamics::SpringCoefficients::new(frequency, damping);
                let actual = spring(
                    mass,
                    f64::from(h),
                    f64::from(s.cfm_factor(h)),
                    f64::from(s.erp_inv_dt(h)),
                )
                .unwrap();
                let omega = std::f64::consts::TAU * f64::from(frequency);
                assert!((actual.stiffness / (mass * omega * omega) - 1.).abs() < 0.00001);
                assert!(
                    (actual.damping - 2. * f64::from(damping) * mass * omega).abs()
                        / (mass * omega)
                        < 0.00001
                );
            }
        }
    }
    assert!(
        spring(1., 1. / 1920., 1., 1.).is_none(),
        "rigid rows have no finite spring mapping"
    );
}

#[test]
#[ignore = "explicit complete saved-bank contact-strain diagnostic; physical limits unchanged"]
fn saved_bank_contact_strain_ledger() {
    let registered = match std::env::var("SLUICE_STRAIN_MODE").unwrap().as_str() {
        "baseline" => false,
        "registered" => true,
        other => panic!("undeclared strain mode: {other}"),
    };
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
    let mut owner = RetainedSoil::decode(&boundary, &env).unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("SLUICE_RIGID_WORK").unwrap());
    std::fs::create_dir(&output).unwrap();
    let changes = install_trial(
        &mut owner,
        if registered {
            WorkTrial::SatFeatures
        } else {
            WorkTrial::SavedBaseline
        },
    );
    let before = passive_inventory(&owner);
    let material = serde_json::to_string(&owner.soil().material()).unwrap();
    let params = owner.model.as_ref().unwrap().world.integration_parameters;
    let initial = super::super::contact_geometry::independent_energy(&owner, env.gravity);
    let colliders = owner.model.as_ref().unwrap().world.colliders.clone();
    let (result, report) = rapier3d::contact_work_audit::record_with_colliders(&colliders, || {
        owner.step(MachineInput::default(), 1. / 480., &env)
    });
    result.unwrap();
    if !registered {
        assert_eq!(
            owner.encode().unwrap(),
            reference,
            "observer changed saved numerical continuation"
        );
    }
    assert_eq!(
        material,
        serde_json::to_string(&owner.soil().material()).unwrap()
    );
    assert_eq!(before, passive_inventory(&owner));
    assert_eq!(
        serde_json::to_value(params).unwrap(),
        serde_json::to_value(owner.model.as_ref().unwrap().world.integration_parameters).unwrap()
    );
    let mechanical = summarize_work_report(&owner, &report, env.gravity, initial);
    assert_work_summary(&mechanical);
    let strain = fold(&report, params);
    let worst: [u32; 4] = serde_json::from_value(
        strain["top_positive_intervals"][0]["key_substep_edge_manifold_point"].clone(),
    )
    .unwrap();
    let selected = witness(&report, worst);
    let final_state = json!({"registered_geometry":registered,"changes":changes,"inventory":before,
        "physical_gate_j":PASSIVITY_J,"physical_gate_passed":number(&mechanical,"independent_energy_delta_j")<=PASSIVITY_J,
        "mechanical":mechanical,"normal_strain_diagnostic":strain,"selected_interval_witness":selected,"gameplay_adopted":false});
    write_json(&output.join("strain-summary.json"), &final_state);
    println!("STRAIN: saved complete bank mode={} rigid={} paired={} soft={} mechanical_gain={}J proxy_balance={}J; raw trace not written",if registered {"registered"} else {"baseline"},report.rigid_len(),strain["paired_intervals"],strain["soft_intervals"],mechanical["independent_energy_delta_j"],strain["balance_with_relaxation_j"]);
}
