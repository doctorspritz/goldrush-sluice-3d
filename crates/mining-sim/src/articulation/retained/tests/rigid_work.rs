//! Explicit saved-bank attribution. Observer success is not physical acceptance.
use super::rigid_geometry::{CapturedPose, Oracle};
use super::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

const AGREEMENT_J: f64 = 0.0001;
const PASSIVITY_J: f64 = 0.01;
const TOP: usize = 16;
const SUMMARY_LIMIT_BYTES: usize = 16 * 1024 * 1024;

fn number(row: &Value, key: &str) -> f64 {
    let value = row[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing finite {key}"));
    assert!(value.is_finite(), "non-finite {key}");
    value
}
fn vector(value: &Value) -> DVec3 {
    DVec3::from_array(serde_json::from_value(value.clone()).unwrap())
}
// The observer retains every row for exact aggregation, but its full trace is
// deliberately never serialized. Even the bounded evidence summary has a hard
// byte limit and removes its own partial output on any write/serialization error.
struct BoundedWriter {
    inner: BufWriter<File>,
    written: usize,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.written {
            return Err(io::Error::other(format!(
                "rigid-work summary exceeds {} byte limit",
                self.limit
            )));
        }
        let written = self.inner.write(bytes)?;
        self.written += written;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
fn write_json_with_limit(path: &Path, value: &impl Serialize, limit: usize) -> io::Result<()> {
    // A failed create_new has acquired no ownership: never remove that path.
    let file = File::create_new(path)?;
    let mut writer = BoundedWriter {
        inner: BufWriter::new(file),
        written: 0,
        limit,
    };
    let result = serde_json::to_writer(&mut writer, value)
        .map_err(io::Error::other)
        .and_then(|()| writer.flush());
    drop(writer);
    if let Err(error) = result {
        if let Err(cleanup) = std::fs::remove_file(path) {
            return Err(io::Error::other(format!(
                "bounded summary write failed: {error}; partial-file removal failed: {cleanup}"
            )));
        }
        return Err(error);
    }
    Ok(())
}
fn write_json(path: &Path, value: &impl Serialize) {
    write_json_with_limit(path, value, SUMMARY_LIMIT_BYTES).unwrap();
}

#[test]
fn bounded_summary_cleans_failed_owned_outputs_and_preserves_existing_files() {
    struct SerializationFailure;
    impl Serialize for SerializationFailure {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut sequence = serializer.serialize_seq(Some(2))?;
            sequence.serialize_element("partial prefix")?;
            Err(serde::ser::Error::custom(
                "intentional serialization failure",
            ))
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "sluice-rigid-work-io-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    let existing = directory.join("unrelated.txt");
    let sentinel = b"existing unrelated file";
    std::fs::write(&existing, sentinel).unwrap();
    let overflow = directory.join("overflow.json");
    let error = write_json_with_limit(&overflow, &"x".repeat(256), 64).unwrap_err();
    assert!(error.to_string().contains("64 byte limit"));
    assert!(
        !overflow.exists(),
        "budget overflow left an owned partial file"
    );
    let failed = directory.join("serialization-failed.json");
    let error = write_json_with_limit(&failed, &SerializationFailure, 64).unwrap_err();
    assert!(error
        .to_string()
        .contains("intentional serialization failure"));
    assert!(
        !failed.exists(),
        "serialization failure left an owned partial file"
    );
    let error = write_json_with_limit(&existing, &json!({"replace":true}), 64).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read(&existing).unwrap(), sentinel);
    let accepted = directory.join("accepted.json");
    write_json_with_limit(&accepted, &json!({"ok":true}), 64).unwrap();
    assert_eq!(std::fs::read(&accepted).unwrap(), br#"{"ok":true}"#);
    assert!(std::fs::metadata(&accepted).unwrap().len() <= 64);
    std::fs::remove_file(accepted).unwrap();
    std::fs::remove_file(existing).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
fn retain(top: &mut Vec<Value>, row: &Value, key: &str) {
    if number(row, key) <= 0. {
        return;
    }
    if top.len() == TOP && number(top.last().unwrap(), key) >= number(row, key) {
        return;
    }
    top.push(row.clone());
    top.sort_by(|a, b| number(b, key).total_cmp(&number(a, key)));
    top.truncate(TOP);
}
#[derive(Default, Serialize)]
struct Work {
    blocks: usize,
    physical_kinetic_change_j: f64,
    normal_midpoint_work_j: f64,
    tangent_midpoint_work_j: f64,
    prescribed_midpoint_work_j: f64,
    max_abs_physical_minus_impulse_work_j: f64,
    max_impulse_velocity_residual: f64,
}
impl Work {
    fn add(&mut self, row: &Value) {
        self.blocks += 1;
        self.physical_kinetic_change_j += number(row, "physical_kinetic_change_j");
        self.normal_midpoint_work_j += number(row, "normal_midpoint_work_j");
        self.tangent_midpoint_work_j += number(row, "tangent_midpoint_work_j");
        self.prescribed_midpoint_work_j += row["prescribed_midpoint_work_j"].as_f64().unwrap_or(0.);
        self.max_abs_physical_minus_impulse_work_j = self
            .max_abs_physical_minus_impulse_work_j
            .max(number(row, "physical_minus_impulse_work_j").abs());
        self.max_impulse_velocity_residual = self
            .max_impulse_velocity_residual
            .max(number(row, "max_impulse_velocity_residual"));
    }
}
#[derive(Default, Serialize)]
struct Bucket {
    rigid: Work,
    generic: Work,
    phase_kinetic_change_j: f64,
    phase_windows: usize,
}
// A substep's warmstart+projection share one phase window. Repeated solve
// passes have no pass ordinal in contact rows, so close their exact sum at the
// same float32 time and bias policy, rather than inventing a per-pass join.
fn bucket_key(row: &Value, time_key: &str, scope: &str) -> String {
    format!("{:08x}/{scope}", (number(row, time_key) as f32).to_bits())
}
fn row_scope(row: &Value) -> &'static str {
    match row["stage"].as_str().unwrap() {
        "warmstart" | "projection" => "warmstart_and_projection",
        "solve" if row["unbiased"] == true => "unbiased_contacts",
        "solve" => "biased_contacts",
        stage => panic!("unaccounted contact stage {stage}"),
    }
}
fn passive_inventory(owner: &RetainedSoil) -> Value {
    assert!(!owner.machine().engine_on);
    assert_eq!(owner.machine().tracks.mechanical_hz, 480);
    let actuator = owner.machine().articulation.as_ref().unwrap();
    assert!(actuator.stiffness_nm_rad.iter().all(|v| *v == 0.));
    assert!(actuator.damping_nms_rad.iter().all(|v| *v == 0.));
    assert_eq!(actuator.pump_energy_j, 0.);
    assert_eq!(actuator.feedback.pump_work_j, 0.);
    assert_eq!(actuator.feedback.elastic_j, 0.);
    let model = owner.model.as_ref().unwrap();
    let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
    let motor_rows: u32 = mb
        .links()
        .skip(1)
        .map(|link| link.joint.data.motor_axes.bits().count_ones())
        .sum();
    assert_eq!(motor_rows, 0);
    let dynamic: Vec<_> = model
        .world
        .bodies
        .iter()
        .filter(|(_, b)| b.is_dynamic())
        .collect();
    assert_eq!(dynamic.len(), 1045);
    assert_eq!(owner.soil().clods.len(), 1040);
    assert_eq!(model.soil_handles.len(), 1040);
    json!({"dynamic_bodies":dynamic.len(),"clods":owner.soil().clods.len(),
        "physical_mass_kg":dynamic.iter().map(|(_,b)|f64::from(b.mass())).sum::<f64>(),
        "motor_rows":motor_rows,"engine_on":false,"mechanical_hz":480,
        "pump_energy_j":actuator.pump_energy_j,"pump_work_j":actuator.feedback.pump_work_j,
        "elastic_energy_j":actuator.feedback.elastic_j,"stiffness_and_damping_zero":true})
}
fn inspect_blocks(top: &mut [Value], oracle: &mut Oracle<'_>, inspect_geometry: bool) {
    for row in top {
        let raw: [[u32; 2]; 2] = serde_json::from_value(row["collider_handles"].clone()).unwrap();
        let handles = raw.map(|[i, generation]| r::ColliderHandle::from_raw_parts(i, generation));
        let poses: [CapturedPose; 2] =
            serde_json::from_value(row["collider_world_poses"].clone()).unwrap();
        let mut checks = Vec::new();
        for point in row["points"].as_array().unwrap() {
            let material = [
                vector(&point["world_anchors_m"][0]),
                vector(&point["world_anchors_m"][1]),
            ];
            let frozen = std::array::from_fn(|i| {
                let arm = vector(&point["frozen_world_arms_m"][i]);
                if row["sides"][i]["solver_body_id"].as_u64().unwrap() == u64::from(u32::MAX) {
                    arm // The real solver uses identity/zero COM for this sentinel.
                } else {
                    arm + vector(&row["sides"][i]["physical_center_of_mass_pose"]["translation"])
                }
            });
            let normal = vector(&point["force_direction1"]);
            assert!(normal.is_finite() && normal.length_squared() > 0.);
            let unit = normal.normalize();
            let tangent = |d: DVec3| (d - unit * unit.dot(d)).to_array();
            let tracked = vector(&point["tracking_world_anchors_m"][0])
                - vector(&point["tracking_world_anchors_m"][1]);
            let material_delta = material[0] - material[1];
            let target_difference = tracked - material_delta;
            let conveyor = vector(&point["tangent_tracking_offset_m"]);
            checks.push(json!({"contact_id":point["contact_id"],
                "geometry_evaluated":inspect_geometry,
                "material_witnesses":inspect_geometry.then(||oracle.inspect(handles,poses,material,normal)),
                "frozen_application_points":inspect_geometry.then(||oracle.inspect(handles,poses,frozen,normal)),
                "frozen_application_world_points_m":frozen.map(|p| p.to_array()),
                "frozen_minus_material_world_m":std::array::from_fn::<_,2,_>(|i|(frozen[i]-material[i]).to_array()),
                "tracking_minus_material_world_m":target_difference.to_array(),
                "tracking_minus_material_tangent_m":tangent(target_difference),
                "tracking_minus_material_minus_conveyor_tangent_m":tangent(target_difference-conveyor),
                "tracking_target_note":"tracking delta minus material delta; raw target includes conveyor offset; differences include within-step frozen-anchor approximation",
                "frozen_point_rule":"physical current COM plus frozen world arm for active solver slots; absolute arm for u32::MAX identity sentinel",
                "full_contact_equation_certified":false}));
        }
        row.as_object_mut()
            .unwrap()
            .insert("geometry_checks".into(), Value::Array(checks));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkTrial {
    SavedBaseline,
    FreshManifolds,
    MaterialTangents,
    FreshMaterialTangents,
}
impl WorkTrial {
    fn label(self) -> &'static str {
        match self {
            Self::SavedBaseline => "saved_policy_baseline",
            Self::FreshManifolds => "test_only_full_feature_refresh",
            Self::MaterialTangents => "test_only_material_tangent_anchors",
            Self::FreshMaterialTangents => "test_only_fresh_features_and_material_tangent_anchors",
        }
    }
}
fn install_fresh_manifolds(owner: &mut RetainedSoil) -> Value {
    assert_eq!(owner.dispatcher, Dispatcher::RigidSoilVerified);
    let world = &mut owner.model.as_mut().unwrap().world;
    assert!(world.integration_parameters.friction_in_bias_pass);
    let before = serde_json::to_value(world.integration_parameters).unwrap();
    crate::soil_contacts::restore_fresh_dispatcher_for_test(world);
    world
        .integration_parameters
        .normalized_contact_recycle_distance = 0.;
    let after = serde_json::to_value(world.integration_parameters).unwrap();
    let mut expected = before.clone();
    expected["normalized_contact_recycle_distance"] = json!(0.);
    assert_eq!(
        after, expected,
        "fresh attribution changed an undeclared numerical parameter"
    );
    json!({"test_only_dispatcher":"verified repair with persistent full feature refresh",
        "stored_dispatcher_enum_changed":false,"production_save_policy_added":false,
        "only_integration_parameter_change":{"name":"normalized_contact_recycle_distance",
            "before":before["normalized_contact_recycle_distance"],"after":0.},
        "all_other_integration_parameters_exact":true,"full_biased_friction_preserved":true})
}
fn install_trial(owner: &mut RetainedSoil, trial: WorkTrial) -> Value {
    let fresh = matches!(
        trial,
        WorkTrial::FreshManifolds | WorkTrial::FreshMaterialTangents
    );
    let material = matches!(
        trial,
        WorkTrial::MaterialTangents | WorkTrial::FreshMaterialTangents
    );
    let mut changes = if fresh {
        install_fresh_manifolds(owner)
    } else {
        json!({"dispatcher_changed":false,"integration_parameters_changed":false})
    };
    if material {
        let world = &mut owner.model.as_mut().unwrap().world;
        let before = world.integration_parameters;
        assert!(!before.diagnostic_material_tangent_anchors);
        world
            .integration_parameters
            .diagnostic_material_tangent_anchors = true;
        let mut restored = world.integration_parameters;
        restored.diagnostic_material_tangent_anchors = false;
        assert_eq!(
            restored, before,
            "material-anchor trial changed another parameter"
        );
        let clone = clone_world(world);
        assert!(
            clone
                .integration_parameters
                .diagnostic_material_tangent_anchors,
            "runtime clone dropped diagnostic selection"
        );
        changes["audit_only_material_tangent_anchors"] = json!(true);
        changes["material_anchor_flag_is_serde_skipped"] = json!(true);
        changes["save_refusal_required"] = json!(true);
        changes["normal_geometry_jacobians_and_generic_twist_unchanged"] = json!(true);
    }
    changes
}
fn candidate_persistence_refusal(owner: &RetainedSoil, env: &Environment<'_>) -> Value {
    if owner
        .model
        .as_ref()
        .unwrap()
        .world
        .integration_parameters
        .diagnostic_material_tangent_anchors
    {
        let error = owner.encode().unwrap_err();
        assert!(
            error.contains("material tangent anchors cannot be saved"),
            "unexpected retained encode refusal: {error}"
        );
        let world = clone_world(&owner.model.as_ref().unwrap().world);
        assert!(
            world
                .integration_parameters
                .diagnostic_material_tangent_anchors
        );
        let boundary = ContactHistory::from_world(world, owner.dispatcher);
        let direct_error = boundary.encode().unwrap_err();
        assert!(direct_error.contains("material tangent anchors cannot be saved"));
        return json!({"retained_encode_refused":true,"contact_history_encode_refused":true,
            "runtime_clone_keeps_selection":true,"refusal":error,"direct_refusal":direct_error,
            "ram_encode_attempted":true,"candidate_boundary_file_written":false,
            "candidate_persistence_supported":false,"saved_continuation_claimed":false});
    }
    // Encode only in RAM to exercise the existing admission boundary. Encoding
    // reinstalls the stored dispatcher in a clone; it is not candidate storage.
    let bytes = owner
        .encode()
        .expect("candidate RAM probe should reach retained parameter validation");
    let error = match RetainedSoil::decode(&bytes, env) {
        Ok(_) => panic!("test-only fresh numerical parameters were silently admitted"),
        Err(error) => error,
    };
    assert!(
        error.contains("parameters do not match"),
        "unexpected admission failure: {error}"
    );
    json!({"ram_encode_attempted":true,"ram_decode_refused":true,"refusal":error,
        "candidate_boundary_file_written":false,"candidate_persistence_supported":false,
        "saved_continuation_claimed":false,"stored_dispatcher_does_not_encode_test_only_refresh":true})
}

#[test]
#[ignore = "explicit saved full-bank rigid work and registered-geometry observer"]
fn saved_checked_bank_rigid_work() {
    saved_bank_work_trial(WorkTrial::SavedBaseline);
}
#[test]
#[ignore = "test-only full feature refresh attribution from the saved complete bank"]
fn saved_bank_fresh_manifold_work() {
    saved_bank_work_trial(WorkTrial::FreshManifolds);
}
#[test]
#[ignore = "audit-only material tangent anchors on the exact saved complete bank"]
fn saved_bank_material_tangent_work() {
    saved_bank_work_trial(WorkTrial::MaterialTangents);
}
#[test]
#[ignore = "audit-only material anchors plus full feature regeneration on the exact saved bank"]
fn saved_bank_fresh_material_tangent_work() {
    saved_bank_work_trial(WorkTrial::FreshMaterialTangents);
}
fn saved_bank_work_trial(trial: WorkTrial) {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let boundary = std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap();
    let reference_bytes =
        std::fs::read(std::env::var_os("SLUICE_ENERGY_REFERENCE").unwrap()).unwrap();
    let reference = RetainedSoil::decode(&reference_bytes, &env).unwrap();
    let mut owner = RetainedSoil::decode(&boundary, &env).unwrap();
    let output = std::path::PathBuf::from(
        std::env::var_os("SLUICE_RIGID_WORK").expect("fresh output directory"),
    );
    std::fs::create_dir(&output).unwrap();
    let declared_changes = install_trial(&mut owner, trial);
    let initial_inventory = passive_inventory(&owner);
    let material_before = serde_json::to_string(&owner.soil().material()).unwrap();
    let parameters_before =
        serde_json::to_string(&owner.model.as_ref().unwrap().world.integration_parameters).unwrap();
    let colliders = owner.model.as_ref().unwrap().world.colliders.clone();
    let initial_energy = super::contact_geometry::independent_energy(&owner, env.gravity);
    let (result, report) = rapier3d::contact_work_audit::record_with_colliders(&colliders, || {
        owner.step(MachineInput::default(), 1. / 480., &env)
    });
    println!("RIGID WORK captured rigid={} generic={} phases={} step_accepted={}; full raw trace omitted, every row will be folded", report.rigid_len(), report.len(), report.iter_phases().count(), result.is_ok());
    io::stdout().flush().unwrap();
    result.unwrap();
    let baseline = trial == WorkTrial::SavedBaseline;
    if baseline {
        same(&owner, &reference, "rigid observer unchanged saved physics");
        assert!(
            owner.encode().unwrap() == reference_bytes,
            "observer changed complete saved numerical boundary"
        );
        assert!(
            reference.encode().unwrap() == reference_bytes,
            "reference does not roundtrip exactly"
        );
    }
    let candidate_persistence = (!baseline).then(|| candidate_persistence_refusal(&owner, &env));
    assert_eq!(
        parameters_before,
        serde_json::to_string(&owner.model.as_ref().unwrap().world.integration_parameters).unwrap()
    );
    assert_eq!(
        material_before,
        serde_json::to_string(&owner.soil().material()).unwrap()
    );
    let final_inventory = passive_inventory(&owner);
    assert_eq!(
        initial_inventory["physical_mass_kg"],
        final_inventory["physical_mass_kg"]
    );
    let world = &owner.model.as_ref().unwrap().world;
    assert_eq!(colliders.len(), world.colliders.len());
    for (handle, before) in colliders.iter() {
        let after = &world.colliders[handle];
        assert_eq!(before.parent(), after.parent());
        assert_eq!(before.position_wrt_parent(), after.position_wrt_parent());
        assert!(
            std::ptr::eq(before.shape(), after.shape()),
            "registered shared shape changed during observation"
        );
        if before.parent().is_none() {
            assert_eq!(before.position(), after.position());
        }
    }
    assert_eq!(report.islands().len(), 1);
    assert_eq!(report.islands()[0].len(), 1045);
    assert!(report.rigid_len() > 0 && !report.is_empty());
    assert!(
        !report.twist_selected_unobserved(),
        "unobserved twist policy"
    );

    let phases: Vec<_> = report
        .iter_phases()
        .map(|p| serde_json::to_value(p).unwrap())
        .collect();
    assert!(phases
        .iter()
        .all(|p| p["dynamic_bodies"] == 1045 && p["island"] == 0));
    let potential = |p: &Value| {
        -(0..3)
            .map(|i| f64::from(env.gravity[i]) * p["mass_position_kg_m"][i].as_f64().unwrap())
            .sum::<f64>()
    };
    let energy = |p: &Value| number(p, "kinetic_j") + potential(p);
    let mut buckets = BTreeMap::<String, Bucket>::new();
    let mut stages = BTreeMap::<String, [f64; 2]>::new();
    for window in phases.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        let (stage, scope) = match b["label"].as_str().unwrap() {
            "after_forces" => ("force_and_gyroscopic_increments", None),
            "after_warm" => {
                assert_eq!(a["label"], "after_forces");
                ("warmstart_and_projection", Some("warmstart_and_projection"))
            }
            "after_joints" => {
                assert_eq!(a["label"], "before_pass");
                ("joints_and_soft", None)
            }
            "after_contacts" => {
                assert_eq!(a["label"], "after_joints");
                let scope = if b["unbiased"] == true {
                    "unbiased_contacts"
                } else {
                    "biased_contacts"
                };
                (scope, Some(scope))
            }
            "after_integrate" => ("position_integration", None),
            _ => ("between_stage_snapshots", None),
        };
        let delta = number(b, "kinetic_j") - number(a, "kinetic_j");
        let sum = stages.entry(stage.into()).or_default();
        sum[0] += delta;
        sum[1] += potential(b) - potential(a);
        if let Some(scope) = scope {
            assert_eq!(a["time_s"], b["time_s"]);
            let bucket = buckets.entry(bucket_key(b, "time_s", scope)).or_default();
            bucket.phase_kinetic_change_j += delta;
            bucket.phase_windows += 1;
        }
    }
    let mut totals = BTreeMap::<String, Work>::new();
    let mut biased = Vec::new();
    let mut warm = Vec::new();
    let mut tangent_positive = Vec::new();
    let mut rigid_points = 0;
    for observed in report.iter_rigid_contacts() {
        let row = serde_json::to_value(observed).unwrap();
        assert_eq!(row["record_kind"], "rigid_coulomb");
        assert!(row["collider_world_poses"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| !p.is_null()));
        for point in row["points"].as_array().unwrap() {
            assert_eq!(
                number(point, "restitution_seed_m_s"),
                0.,
                "unobserved restitution must be absent"
            );
            rigid_points += 1;
        }
        let scope = row_scope(&row);
        buckets
            .entry(bucket_key(&row, "solved_time_s", scope))
            .or_default()
            .rigid
            .add(&row);
        totals
            .entry(format!(
                "rigid/{}/{}",
                row["stage"].as_str().unwrap(),
                row["unbiased"]
            ))
            .or_default()
            .add(&row);
        if row["stage"] == "solve" && row["unbiased"] == false {
            retain(&mut biased, &row, "physical_kinetic_change_j");
            retain(&mut tangent_positive, &row, "tangent_midpoint_work_j");
        } else if row["stage"] == "warmstart" {
            retain(&mut warm, &row, "physical_kinetic_change_j");
        }
    }
    for observed in report.iter_generic_contacts() {
        let row = serde_json::to_value(observed).unwrap();
        buckets
            .entry(bucket_key(&row, "solved_time_s", row_scope(&row)))
            .or_default()
            .generic
            .add(&row);
        totals
            .entry(format!(
                "generic/{}/{}",
                row["stage"].as_str().unwrap(),
                row["unbiased"]
            ))
            .or_default()
            .add(&row);
    }
    let closures: Vec<_> = buckets.iter().map(|(key,b)| {
        let observed=b.rigid.physical_kinetic_change_j+b.generic.physical_kinetic_change_j;
        let error=observed-b.phase_kinetic_change_j;
        json!({"time_bits_and_scope":key,"phase_windows":b.phase_windows,"rigid":b.rigid,"generic":b.generic,
            "phase_kinetic_change_j":b.phase_kinetic_change_j,"observed_contact_kinetic_change_j":observed,
            "closure_error_j":error,"agreement_limit_j":AGREEMENT_J,"closed":b.phase_windows>0 && error.abs()<AGREEMENT_J})
    }).collect();
    let contact_buckets_closed = closures.iter().all(|c| c["closed"] == true);
    let final_energy = super::contact_geometry::independent_energy(&owner, env.gravity);
    let body_delta = final_energy - initial_energy;
    let phase_delta = energy(phases.last().unwrap()) - energy(phases.first().unwrap());
    let reported = owner.soil().feedback.coupled_passivity_residual_j;
    let phase_body_agree = (phase_delta - body_delta).abs() < AGREEMENT_J;
    let feedback_agrees = (reported - body_delta).abs() < 0.05;
    let no_joint_or_between_work =
        ["joints_and_soft", "between_stage_snapshots"]
            .iter()
            .all(|key| {
                stages
                    .get(*key)
                    .is_none_or(|s| s[0].abs() < AGREEMENT_J && s[1].abs() < AGREEMENT_J)
            });
    let mut oracle = Oracle::new(world);
    inspect_blocks(&mut biased, &mut oracle, true);
    inspect_blocks(&mut warm, &mut oracle, false);
    inspect_blocks(&mut tangent_positive, &mut oracle, false);
    let mut summary = json!({"saved_after_state_exact":baseline,"complete_numerical_boundary_exact":baseline,
        "selected_body_machine_soil_exact":baseline,"solver_parameters_unchanged":baseline,
        "registered_shapes_and_collider_attachments_unchanged":true,"initial_inventory":initial_inventory,"final_inventory":final_inventory,
        "rigid_blocks":report.rigid_len(),"generic_blocks":report.len(),"rigid_point_observations":rigid_points,
        "phase_samples":phases.len(),"islands":report.islands(),"twist_selected_unobserved":report.twist_selected_unobserved(),
        "rigid_restitution_seeds_zero":true,"stage_contact_work":totals,"phase_stage_changes_kinetic_and_gravity_j":stages,
        "contact_bucket_closures":closures,"contact_physical_work_buckets_closed":contact_buckets_closed,
        "observer_agreement_limit_j":AGREEMENT_J,"initial_independent_energy_j":initial_energy,"final_independent_energy_j":final_energy,
        "independent_energy_delta_j":body_delta,"phase_energy_delta_j":phase_delta,"phase_minus_body_j":phase_delta-body_delta,
        "phase_body_agreement":phase_body_agree,"reported_passivity_residual_j":reported,"reported_feedback_agrees_within_existing_0_05_j":feedback_agrees,
        "joint_and_between_snapshot_changes_zero_within_observer_limit":no_joint_or_between_work,
        "observer_success":contact_buckets_closed && phase_body_agree && feedback_agrees && no_joint_or_between_work,
        "physical_passivity_gate_j":PASSIVITY_J,"physical_passivity_gate_passed":body_delta<=PASSIVITY_J,
        "top16_positive_biased_solve_blocks_with_geometry":biased,"top16_positive_warmstart_blocks":warm,
        "top16_positive_biased_tangent_work_blocks":tangent_positive,
        "top_geometry_is_subset":true,"all_clustered_contact_geometry_certified":false,
        "individual_impulse_work_identity_required":false,
        "partition_note":"normal/tangent work are fixed-block midpoint partitions, not sequential row work or a closed causal bias budget",
        "phases":phases});
    let evidence = summary.as_object_mut().unwrap();
    evidence.insert("trial".into(), json!(trial.label()));
    evidence.insert("saved_failure_history_imported".into(), json!(true));
    evidence.insert("initial_warmstart_history_preserved".into(), json!(true));
    evidence.insert(
        "saved_reference_equality_required_and_evaluated".into(),
        json!(baseline),
    );
    evidence.insert("declared_changes".into(), declared_changes);
    evidence.insert("parameters_unchanged_during_step".into(), json!(true));
    evidence.insert(
        "candidate_persistence_probe".into(),
        json!(candidate_persistence),
    );
    evidence.insert("candidate_boundary_files_written".into(), json!(false));
    evidence.insert("saved_candidate_continuation_claimed".into(), json!(false));
    evidence.insert("raw_report_omitted".into(), json!(true));
    evidence.insert("raw_report_omission_reason".into(), json!("unbounded trace is not written; all observed rows contribute to exact stage aggregates and closures"));
    evidence.insert("all_observed_rows_folded".into(), json!(true));
    evidence.insert(
        "summary_write_limit_bytes".into(),
        json!(SUMMARY_LIMIT_BYTES),
    );
    write_json(&output.join("summary.json"), &summary);
    println!("RIGID WORK rigid={} generic={} contact_buckets_closed={} energy_gain_j={body_delta:.9} passive_gate_passed={}", report.rigid_len(),report.len(),contact_buckets_closed,body_delta<=PASSIVITY_J);
    assert!(
        contact_buckets_closed,
        "rigid+generic contact stage closure failed; see summary.json"
    );
    assert!(
        phase_body_agree,
        "independent saved-body and phase energy disagree"
    );
    assert!(
        feedback_agrees,
        "independent energy and existing feedback disagree"
    );
    assert!(
        no_joint_or_between_work,
        "unaccounted joint/interstage work"
    );
}

#[test]
#[ignore = "bounded primitive SAT interpretation of an existing verified rigid-work summary; no replay"]
fn saved_rigid_work_primitive_pair_geometry() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let boundary_path =
        std::path::PathBuf::from(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap());
    let boundary = std::fs::read(&boundary_path).unwrap();
    let owner = RetainedSoil::decode(&boundary, &env).unwrap();
    assert!(
        owner.encode().unwrap() == boundary,
        "geometry source boundary does not roundtrip exactly"
    );
    let source_path = std::path::PathBuf::from(
        std::env::var_os("SLUICE_RIGID_WORK_SUMMARY").expect("existing verified summary.json"),
    );
    assert!(
        std::fs::metadata(&source_path).unwrap().len() <= SUMMARY_LIMIT_BYTES as u64,
        "input summary exceeds bounded interpretation budget"
    );
    let source: Value =
        serde_json::from_reader(std::io::BufReader::new(File::open(&source_path).unwrap()))
            .unwrap();
    for flag in [
        "observer_success",
        "saved_after_state_exact",
        "complete_numerical_boundary_exact",
        "registered_shapes_and_collider_attachments_unchanged",
        "all_observed_rows_folded",
    ] {
        assert_eq!(source[flag], true, "source summary lacks verified {flag}");
    }
    let inventory = passive_inventory(&owner);
    let energy = super::contact_geometry::independent_energy(&owner, env.gravity);
    assert!(
        (energy - number(&source, "initial_independent_energy_j")).abs() < AGREEMENT_J,
        "registered before world does not match source energy"
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("SLUICE_RIGID_GEOMETRY").expect("fresh geometry report directory"),
    );
    std::fs::create_dir(&output).unwrap();
    let oracle = Oracle::new(&owner.model.as_ref().unwrap().world);
    let mut pairs = BTreeMap::<String, Value>::new();
    let mut block_count = 0;
    let mut point_count = 0;
    for ranking in [
        "top16_positive_biased_solve_blocks_with_geometry",
        "top16_positive_warmstart_blocks",
    ] {
        let rows = source[ranking].as_array().unwrap();
        assert!(rows.len() <= TOP);
        for (rank, row) in rows.iter().enumerate() {
            block_count += 1;
            let raw: [[u32; 2]; 2] =
                serde_json::from_value(row["collider_handles"].clone()).unwrap();
            let handles =
                raw.map(|[i, generation]| r::ColliderHandle::from_raw_parts(i, generation));
            let poses: [CapturedPose; 2] =
                serde_json::from_value(row["collider_world_poses"].clone()).unwrap();
            for point in row["points"].as_array().unwrap() {
                point_count += 1;
                let normal = vector(&point["force_direction1"]);
                let unit = normal.normalize();
                let material = [
                    vector(&point["world_anchors_m"][0]),
                    vector(&point["world_anchors_m"][1]),
                ];
                let frozen: [DVec3; 2] = std::array::from_fn(|i| {
                    let arm = vector(&point["frozen_world_arms_m"][i]);
                    if row["sides"][i]["solver_body_id"].as_u64().unwrap() == u64::from(u32::MAX) {
                        arm
                    } else {
                        arm + vector(
                            &row["sides"][i]["physical_center_of_mass_pose"]["translation"],
                        )
                    }
                });
                let key = serde_json::to_string(&json!([
                    row["collider_handles"],
                    row["collider_world_poses"],
                    point["force_direction1"]
                ]))
                .unwrap();
                let pair=pairs.entry(key).or_insert_with(||json!({
                    "collider_handles":row["collider_handles"],"captured_collider_poses":row["collider_world_poses"],
                    "primitive_pair_audit":oracle.pair_primitive_audit(handles,poses,normal),"source_points":[]}));
                pair["source_points"].as_array_mut().unwrap().push(json!({
                    "ranking":ranking,"rank":rank+1,"manifold":row["manifold"],"stage":row["stage"],
                    "solved_time_s":row["solved_time_s"],"chunk":row["chunk"],"lane":row["lane"],
                    "contact_id":point["contact_id"],"clustered":row["clustered"],
                    "physical_kinetic_change_j":row["physical_kinetic_change_j"],
                    "normal_midpoint_work_j":point["normal_midpoint_work_j"],"tangent_midpoint_work_j":point["tangent_midpoint_work_j"],
                    "normal_rhs_m_s":point["normal_rhs_m_s"],"tracking_distance_m":point["tracking_distance_m"],
                    "material_world_anchors_m":point["world_anchors_m"],"material_normal_gap_m":(material[0]-material[1]).dot(unit),
                    "frozen_application_world_points_m":frozen.map(|p|p.to_array()),"frozen_application_normal_gap_m":(frozen[0]-frozen[1]).dot(unit)}));
            }
        }
    }
    let applicable = pairs
        .values()
        .filter(|p| p["primitive_pair_audit"]["applicable"] == true)
        .count();
    let summary = json!({"physical_replay_performed":false,"source_summary":source_path,"source_boundary":boundary_path,
        "source_inventory":inventory,"source_initial_independent_energy_j":energy,
        "source_observer_success":true,"source_physical_passivity_gate_passed":source["physical_passivity_gate_passed"],
        "ranked_blocks_interpreted":block_count,"source_points_interpreted":point_count,"unique_pair_pose_normal_cases":pairs.len(),
        "primitive_sat_applicable_cases":applicable,"pair_cases":pairs.into_values().collect::<Vec<_>>(),
        "deduplication":"actual collider pair plus exact captured current collider poses and raw constraint normal; changed substep poses are preserved",
        "scope":"single registered closed convex polyhedral leaves only; unions/terrain/curved shapes remain unproved",
        "different_mtd_alone_proves_invalid_contact":false,"all_contact_geometry_certified":false,
        "physical_passivity_gate_j":PASSIVITY_J,"physical_policy_changed":false});
    write_json(&output.join("report.json"), &summary);
    println!("RIGID GEOMETRY blocks={block_count} points={point_count} unique_cases={} applicable={applicable} replay=false",summary["unique_pair_pose_normal_cases"]);
}

#[test]
#[ignore = "explicit complete fresh 300-step passive bank; no work collector and no candidate save"]
fn complete_bank_fresh_manifold_trajectory() {
    complete_bank_trajectory_trial(WorkTrial::FreshManifolds);
}
#[test]
#[ignore = "audit-only material tangent anchors, full fresh 300-step bank, no work recorder or saves"]
fn complete_bank_material_tangent_trajectory() {
    complete_bank_trajectory_trial(WorkTrial::MaterialTangents);
}
#[test]
#[ignore = "audit-only material anchors and regenerated features, full 300-step bank, no work recorder"]
fn complete_bank_fresh_material_tangent_trajectory() {
    complete_bank_trajectory_trial(WorkTrial::FreshMaterialTangents);
}
fn complete_bank_trajectory_trial(trial: WorkTrial) {
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
    let mut owner = RetainedSoil::new(
        session.machine.clone(),
        session.active_soil.take().unwrap(),
        &env,
    )
    .unwrap();
    let declared_changes = install_trial(&mut owner, trial);
    let persistence_refusal = candidate_persistence_refusal(&owner, &env);
    let initial_inventory = passive_inventory(&owner);
    let material = serde_json::to_string(&owner.soil().material()).unwrap();
    let parameters =
        serde_json::to_string(&owner.model.as_ref().unwrap().world.integration_parameters).unwrap();
    let initial_energy = super::contact_geometry::independent_energy(&owner, env.gravity);
    let output = std::path::PathBuf::from(
        std::env::var_os("SLUICE_RIGID_WORK").expect("fresh trajectory output directory"),
    );
    std::fs::create_dir(&output).unwrap();
    let mut previous_energy = initial_energy;
    let mut maximum_gain = f64::NEG_INFINITY;
    let mut energy_violations = 0;
    let mut maximum_feedback_error: f64 = 0.;
    let mut trajectory = Vec::with_capacity(300);
    let mut refusal = None;
    let started = std::time::Instant::now();
    for step in 1..=300 {
        if let Err(error) = owner.step(MachineInput::default(), 1. / 480., &env) {
            refusal = Some(
                json!({"attempted_step":step,"actual_error":error,"owner_poisoned":owner.failed()}),
            );
            break;
        }
        assert_eq!(
            owner
                .model
                .as_ref()
                .unwrap()
                .world
                .integration_parameters
                .diagnostic_material_tangent_anchors,
            matches!(
                trial,
                WorkTrial::MaterialTangents | WorkTrial::FreshMaterialTangents
            ),
            "material tangent selection changed during trajectory"
        );
        let inventory = passive_inventory(&owner);
        assert_eq!(
            inventory["physical_mass_kg"],
            initial_inventory["physical_mass_kg"]
        );
        assert_eq!(
            material,
            serde_json::to_string(&owner.soil().material()).unwrap()
        );
        assert_eq!(
            parameters,
            serde_json::to_string(&owner.model.as_ref().unwrap().world.integration_parameters)
                .unwrap()
        );
        let energy = super::contact_geometry::independent_energy(&owner, env.gravity);
        assert!(energy.is_finite());
        let gain = energy - previous_energy;
        let reported = owner.soil().feedback.coupled_passivity_residual_j;
        let feedback_error = (reported - gain).abs();
        assert!(feedback_error.is_finite());
        maximum_gain = maximum_gain.max(gain);
        energy_violations += usize::from(gain > PASSIVITY_J);
        maximum_feedback_error = maximum_feedback_error.max(feedback_error);
        trajectory.push(json!({"step":step,"independent_energy_j":energy,"delta_energy_j":gain,
            "reported_residual_j":reported,"feedback_error_j":feedback_error,"feedback_agreement_limit_j":0.05,
            "feedback_agrees":feedback_error<0.05,"physical_passivity_gate_j":PASSIVITY_J,
            "physical_passivity_gate_passed":gain<=PASSIVITY_J,"inventory":inventory}));
        previous_energy = energy;
    }
    let completed = trajectory.len() == 300 && refusal.is_none();
    let final_persistence_refusal =
        (!owner.failed()).then(|| candidate_persistence_refusal(&owner, &env));
    let final_inventory = (!owner.failed()).then(|| passive_inventory(&owner));
    let summary = json!({"trial":format!("{}_fresh_300_step_bank",trial.label()),
        "fresh_start":true,"saved_failure_history_imported":false,"physical_replay_performed":true,
        "work_observer_enabled":false,"requested_steps":300,"accepted_steps":trajectory.len(),
        "trajectory_completed":completed,"actual_refusal":refusal,"initial_inventory":initial_inventory,"final_inventory":final_inventory,
        "declared_changes":declared_changes,"initial_persistence_refusal":persistence_refusal,"final_persistence_refusal":final_persistence_refusal,
        "initial_independent_energy_j":initial_energy,"last_accepted_independent_energy_j":previous_energy,
        "net_energy_gain_j":previous_energy-initial_energy,"maximum_step_energy_gain_j":(!trajectory.is_empty()).then_some(maximum_gain),
        "energy_violation_count":energy_violations,"physical_passivity_gate_j":PASSIVITY_J,
        "all_steps_physical_passivity_gate_passed":completed && energy_violations==0,
        "maximum_feedback_error_j":maximum_feedback_error,"existing_feedback_agreement_limit_j":0.05,
        "observer_diagnostic_success":completed && maximum_feedback_error<0.05,
        "candidate_boundary_files_written":false,"saved_candidate_continuation_claimed":false,
        "elapsed_wall_s":started.elapsed().as_secs_f64(),"trajectory":trajectory});
    write_json(&output.join("trajectory-summary.json"), &summary);
    println!("FRESH TRAJECTORY accepted={} completed={completed} net_gain_j={:.9} max_gain_j={maximum_gain:.9} violations={energy_violations} feedback_error_j={maximum_feedback_error:.9}",summary["accepted_steps"],previous_energy-initial_energy);
    assert!(
        completed,
        "actual refusal interrupted fresh trajectory; see bounded summary"
    );
    assert!(
        maximum_feedback_error < 0.05,
        "independent and existing feedback energy disagree"
    );
}

#[test]
fn material_tangent_trial_refuses_both_save_boundaries_and_keeps_runtime_clone() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut owner =
        RetainedSoil::new(session.machine.clone(), session.active_soil.unwrap(), &env).unwrap();
    let legacy = owner.encode().unwrap();
    assert!(
        !owner
            .model
            .as_ref()
            .unwrap()
            .world
            .integration_parameters
            .diagnostic_material_tangent_anchors
    );
    let declared = install_trial(&mut owner, WorkTrial::MaterialTangents);
    assert_eq!(declared["audit_only_material_tangent_anchors"], true);
    assert_eq!(
        candidate_persistence_refusal(&owner, &env)["retained_encode_refused"],
        true
    );
    let clone = owner.clone();
    assert!(
        clone
            .model
            .as_ref()
            .unwrap()
            .world
            .integration_parameters
            .diagnostic_material_tangent_anchors
    );
    assert!(clone
        .encode()
        .unwrap_err()
        .contains("material tangent anchors cannot be saved"));
    let restored = RetainedSoil::decode(&legacy, &env).unwrap();
    assert!(
        !restored
            .model
            .as_ref()
            .unwrap()
            .world
            .integration_parameters
            .diagnostic_material_tangent_anchors
    );
    assert_eq!(
        restored.encode().unwrap(),
        legacy,
        "legacy boundary bytes changed"
    );
}
