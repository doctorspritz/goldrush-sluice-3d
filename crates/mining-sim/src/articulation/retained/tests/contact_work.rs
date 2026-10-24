//! Explicit observer of the saved passive failure; no solver policy change.
use super::*;

#[test]
#[ignore = "explicit full-bank work observer, requires contact-work-audit"]
fn saved_checked_bank_contact_work() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let bytes = std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap();
    let reference = RetainedSoil::decode(
        &std::fs::read(std::env::var_os("SLUICE_ENERGY_REFERENCE").unwrap()).unwrap(),
        &env,
    )
    .unwrap();
    let mut owner = RetainedSoil::decode(&bytes, &env).unwrap();
    let initial_body_energy = super::contact_geometry::independent_energy(&owner, env.gravity);
    let output = std::path::PathBuf::from(std::env::var_os("SLUICE_CONTACT_WORK").unwrap());
    assert!(!output.exists());
    let (result, rows) = rapier3d::contact_work_audit::record(|| {
        owner.step(MachineInput::default(), 1. / 480., &env)
    });
    result.unwrap();
    same(&owner, &reference, "work observer unchanged saved physics");
    assert!(!rows.is_empty());
    assert!(
        owner.encode().unwrap() == reference.encode().unwrap(),
        "observer changed complete numerical boundary"
    );
    let final_body_energy = super::contact_geometry::independent_energy(&owner, env.gravity);
    let report = serde_json::to_value(&rows).unwrap();
    let phases = report["phases"].as_array().unwrap();
    assert_eq!(report["islands"].as_array().unwrap().len(), 1);
    assert!(phases.iter().all(|p| p["dynamic_bodies"] == 1045));
    let energy = |p: &serde_json::Value| {
        p["kinetic_j"].as_f64().unwrap()
            - (0..3)
                .map(|i| f64::from(env.gravity[i]) * p["mass_position_kg_m"][i].as_f64().unwrap())
                .sum::<f64>()
    };
    let first_phase_energy = energy(phases.first().unwrap());
    let last_phase_energy = energy(phases.last().unwrap());
    let phase_delta = last_phase_energy - first_phase_energy;
    let body_delta = final_body_energy - initial_body_energy;
    // Observer agreement is distinct from the unchanged 0.01J passivity gate.
    assert!((phase_delta - body_delta).abs() < 0.0001);
    let value = serde_json::json!({"saved_after_state_exact":true,"complete_numerical_boundary_exact":true,
        "solver_parameters_unchanged":true,"rows":report,"initial_body_energy_j":initial_body_energy,
        "final_body_energy_j":final_body_energy,"first_phase_energy_j":first_phase_energy,
        "last_phase_energy_j":last_phase_energy,"phase_delta_j":phase_delta,"body_delta_j":body_delta,
        "whole_world_impulse_work_budget_closed":false,"passivity_gate_j":0.01,"passivity_gate_passed":body_delta<=0.01});
    std::fs::write(output, serde_json::to_vec(&value).unwrap()).unwrap();
    println!(
        "CONTACT WORK rows={} exact_saved_continuation=true",
        rows.len()
    );
}
