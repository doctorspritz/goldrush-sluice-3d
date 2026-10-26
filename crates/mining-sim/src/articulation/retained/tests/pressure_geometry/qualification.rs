//! Independent pose differences and same-material partition controls. No cached
//! contact normal or impulse supplies the expected force/torque.
use super::*;
pub(super) fn check(
    name: &str,
    a: &PressureBody,
    b: Opposing<'_>,
    motions: [RigidMotion; 2],
) -> serde_json::Value {
    let evaluate = |m: [RigidMotion; 2]| pressure::evaluate(a, m[0], opponent(&b), m[1]).unwrap();
    let reference = evaluate(motions);
    assert_eq!(
        reference.directional_tie_volume_m3, 0.,
        "saved derivative needs a smooth direction"
    );
    let mut derivatives = Vec::new();
    let mut worst_force = 0_f64;
    let mut worst_torque = 0_f64;
    let mut worst_normalized = 0_f64;
    for body in 0..2 {
        for dof in 0..6 {
            let axis = [DVec3::X, DVec3::Y, DVec3::Z][dof % 3];
            let h = if dof < 3 { 1e-7 } else { 1e-6 };
            let mut plus = motions;
            let mut minus = motions;
            if dof < 3 {
                plus[body].center_of_mass_m += axis * h;
                minus[body].center_of_mass_m -= axis * h;
            } else {
                plus[body].rotation = DQuat::from_axis_angle(axis, h) * motions[body].rotation;
                minus[body].rotation = DQuat::from_axis_angle(axis, -h) * motions[body].rotation;
            }
            let derivative =
                (evaluate(plus).elastic_energy_j - evaluate(minus).elastic_energy_j) / (2. * h);
            let force = if dof < 3 {
                reference.forces_n[body]
            } else {
                reference.torques_nm[body]
            }
            .dot(axis);
            let absolute = (derivative + force).abs();
            let floor = if dof < 3 { 1. } else { 0.01 };
            let normalized = absolute / derivative.abs().max(force.abs()).max(floor);
            worst_normalized = worst_normalized.max(normalized);
            if dof < 3 {
                worst_force = worst_force.max(absolute);
            } else {
                worst_torque = worst_torque.max(absolute);
            }
            derivatives.push(serde_json::json!({"body":body,"dof":dof,"step":h,"energy_derivative":derivative,"negative_wrench":-force,"absolute_error":absolute,"normalized_error":normalized}));
        }
    }
    let mut partitions = Vec::new();
    let mut worst_partition_force = 0_f64;
    let mut worst_partition_torque = 0_f64;
    let mut worst_partition_energy = 0_f64;
    let mut worst_partition_volume = 0_f64;
    let levels = if matches!(b, Opposing::Compliant(_)) {
        vec![(1, 0), (0, 1), (1, 1)]
    } else {
        vec![(1, 0), (2, 0)]
    };
    for (left, right) in levels {
        let a = a.refined(left).unwrap();
        let r = match &b {
            Opposing::Compliant(b) => {
                let b = b.refined(right).unwrap();
                pressure::evaluate(&a, motions[0], Opposing::Compliant(&b), motions[1]).unwrap()
            }
            _ => pressure::evaluate(&a, motions[0], opponent(&b), motions[1]).unwrap(),
        };
        let force = (r.forces_n[0] - reference.forces_n[0])
            .length()
            .max((r.forces_n[1] - reference.forces_n[1]).length());
        let torque = (r.torques_nm[0] - reference.torques_nm[0])
            .length()
            .max((r.torques_nm[1] - reference.torques_nm[1]).length());
        let energy = (r.elastic_energy_j - reference.elastic_energy_j).abs();
        let volume = (r.overlap_volume_m3 - reference.overlap_volume_m3).abs();
        worst_partition_force = worst_partition_force.max(force);
        worst_partition_torque = worst_partition_torque.max(torque);
        worst_partition_energy = worst_partition_energy.max(energy);
        worst_partition_volume = worst_partition_volume.max(volume);
        partitions.push(serde_json::json!({"levels":[left,right],"force_error_n":force,"torque_error_nm":torque,"energy_error_j":energy,"overlap_error_m3":volume}));
    }
    let passed = worst_normalized < 1e-5
        && worst_partition_force < 1e-7
        && worst_partition_torque < 1e-9
        && worst_partition_energy < 1e-11
        && worst_partition_volume < 1e-15;
    serde_json::json!({"name":name,"passed":passed,"finite_difference_derivatives":derivatives,"max_force_derivative_error_n":worst_force,"max_torque_derivative_error_nm":worst_torque,"max_normalized_derivative_error":worst_normalized,"same_material_partition_controls":partitions,"reference_potential_j":reference.elastic_energy_j,"reference_overlap_m3":reference.overlap_volume_m3})
}

fn opponent<'a>(b: &Opposing<'a>) -> Opposing<'a> {
    match b {
        Opposing::Compliant(b) => Opposing::Compliant(b),
        Opposing::Rigid(b) => Opposing::Rigid(b),
        Opposing::RigidCompound(b) => Opposing::RigidCompound(b),
        Opposing::TerrainLayer(b) => Opposing::TerrainLayer(b),
    }
}
pub(super) type Candidates = BTreeMap<&'static str, Vec<(f64, [r::ColliderHandle; 2])>>;
pub(super) fn remember(
    candidates: &mut Candidates,
    name: &'static str,
    metric: f64,
    ids: [r::ColliderHandle; 2],
) {
    let list = candidates.entry(name).or_default();
    list.push((metric, ids));
    list.sort_by(|a, b| b.0.total_cmp(&a.0));
    list.truncate(4);
}
pub(super) fn selected(
    candidates: &Candidates,
    names: &[&'static str],
) -> Vec<(&'static str, [r::ColliderHandle; 2], usize)> {
    let mut used = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for &name in names {
        if let Some(list) = candidates.get(name) {
            for (rank, (_, ids)) in list.iter().enumerate() {
                let mut key = ids.map(|id| id.into_raw_parts());
                key.sort();
                if used.insert(key) {
                    out.push((name, *ids, rank));
                    break;
                }
            }
        }
    }
    out
}
