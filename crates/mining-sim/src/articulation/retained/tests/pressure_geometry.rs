//! Read-only finite pressure qualification on the actual saved bank. No force
//! is applied and no potential is credited to the existing solver's energy.
use super::*;
use crate::soil_response::{
    pressure::{self, Opposing, PressureBody, RegisteredSolid},
    RigidMotion,
};
use glam::{DQuat, DVec3};
use std::collections::{BTreeMap, HashMap};
mod qualification;
mod rigid_shell;
mod terrain_layer;
fn vector(v: r::Vector) -> DVec3 {
    DVec3::from_array(v.to_array().map(f64::from))
}
fn rotation(q: r::Rotation) -> DQuat {
    DQuat::from_array(q.to_array().map(f64::from))
}
fn construct(
    world: &r::PhysicsWorld,
    collider_handle: r::ColliderHandle,
) -> Result<(PressureBody, RigidMotion), String> {
    let collider = &world.colliders[collider_handle];
    let hull = collider
        .shape()
        .as_convex_polyhedron()
        .ok_or("Clod is not its registered convex hull")?;
    let solid = RegisteredSolid::from_hull(hull)?;
    let (solid, motion) = captured(world, collider_handle, solid)?;
    Ok((PressureBody::new(solid, 200_000.)?, motion))
}
fn captured(
    world: &r::PhysicsWorld,
    collider_handle: r::ColliderHandle,
    mut solid: RegisteredSolid,
) -> Result<(RegisteredSolid, RigidMotion), String> {
    let collider = &world.colliders[collider_handle];
    let body = &world.bodies[collider.parent().ok_or("Registered solid has no body")?];
    let pose = body.position();
    let q = rotation(pose.rotation);
    let translation = vector(pose.translation);
    let com = translation + q * vector(body.mass_properties().local_mprops.local_com);
    if let Some(local) = collider.position_wrt_parent() {
        solid = solid.captured_frame(vector(local.translation), rotation(local.rotation))?;
    }
    solid = solid.captured_frame(translation - com, q)?;
    Ok((
        solid,
        RigidMotion {
            center_of_mass_m: com,
            rotation: DQuat::IDENTITY,
            linear_velocity_m_s: vector(body.linvel()),
            angular_velocity_rad_s: vector(body.angvel()),
        },
    ))
}
fn descriptor(body: &PressureBody, motion: RigidMotion) -> serde_json::Value {
    serde_json::json!({"vertices":body.solid().vertices(),"faces":body.solid().faces(),"tets":body.solid().reference_tetrahedra(),"pressure_pa":body.pressure_scale_pa(),"com":motion.center_of_mass_m,"rotation":motion.rotation,"linear":motion.linear_velocity_m_s,"angular":motion.angular_velocity_rad_s})
}
#[test]
#[ignore = "explicit read-only finite-pressure geometry on complete saved bank; no gameplay adoption"]
fn saved_bank_finite_pressure_geometry() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let boundary = std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap();
    let owner = RetainedSoil::decode(&boundary, &env).unwrap();
    let before = owner.encode().unwrap();
    let model = owner.model.as_ref().unwrap();
    let world = &model.world;
    assert_eq!(model.soil_handles.len(), 1040);
    let mut meshes = HashMap::new();
    let mut vertices = 0;
    let mut faces = 0;
    let mut cells = 0;
    for handle in &model.soil_handles {
        for collider in world.bodies[*handle].colliders() {
            let body = construct(world, *collider).unwrap();
            vertices += body.0.solid().vertices().len();
            faces += body.0.solid().faces().len();
            cells += body.0.tetrahedra();
            meshes.insert(*collider, body);
        }
    }
    let mut total = 0;
    let mut admitted = 0;
    let mut unsupported = 0;
    let mut positive = 0;
    let mut zero = 0;
    let mut failures = Vec::new();
    let mut failure_count = 0;
    let mut potential = 0.;
    let mut volume = 0.;
    let mut worst_force = 0f64;
    let mut worst_moment = 0f64;
    let mut worst_power = 0f64;
    let mut ties = 0.;
    let mut negative_cached_at_zero_overlap = 0;
    let mut outside_shapes = BTreeMap::<String, usize>::new();
    let mut samples = qualification::Candidates::new();
    let mut steel = rigid_shell::Steel::new(world).unwrap();
    for pair in world.narrow_phase.contact_pairs() {
        total += 1;
        let (Some((a, am)), Some((b, bm))) =
            (meshes.get(&pair.collider1), meshes.get(&pair.collider2))
        else {
            if steel.observe([pair.collider1, pair.collider2], &meshes) {
                continue;
            }
            unsupported += 1;
            let kind = |id: r::ColliderHandle| {
                let collider = &world.colliders[id];
                format!(
                    "{}:{:?}",
                    if meshes.contains_key(&id) {
                        "clod"
                    } else {
                        "other"
                    },
                    collider.shape().shape_type()
                )
            };
            *outside_shapes
                .entry(format!(
                    "{} / {}",
                    kind(pair.collider1),
                    kind(pair.collider2)
                ))
                .or_default() += 1;
            continue;
        };
        admitted += 1;
        match pressure::evaluate(a, *am, Opposing::Compliant(b), *bm) {
            Ok(result) => {
                if result.overlap_volume_m3 > 0. {
                    positive += 1;
                    for (name, metric) in [
                        ("largest_potential", result.elastic_energy_j),
                        ("largest_force", result.forces_n[0].length()),
                        ("largest_torque", result.torques_nm[0].length()),
                        ("smallest_positive_overlap", -result.overlap_volume_m3),
                    ] {
                        qualification::remember(
                            &mut samples,
                            name,
                            metric,
                            [pair.collider1, pair.collider2],
                        );
                    }
                } else {
                    zero += 1;
                    if pair
                        .manifolds()
                        .iter()
                        .any(|m| m.points.iter().any(|p| p.dist < 0.))
                    {
                        negative_cached_at_zero_overlap += 1;
                    }
                }
                potential += result.elastic_energy_j;
                volume += result.overlap_volume_m3;
                ties += result.directional_tie_volume_m3;
                worst_force = worst_force.max(result.force_residual_n.length());
                worst_moment = worst_moment.max(result.moment_residual_nm.length());
                worst_power = worst_power.max(result.power_residual_w.abs());
            }
            Err(error) => {
                failure_count += 1;
                if failures.len() < 16 {
                    failures.push(serde_json::json!({"colliders":[pair.collider1.into_raw_parts(),pair.collider2.into_raw_parts()],"error":error,"a":descriptor(a,*am),"b":descriptor(b,*bm)}));
                }
            }
        }
    }
    let qualifications: Vec<_> = qualification::selected(
        &samples,
        &[
            "largest_potential",
            "largest_force",
            "largest_torque",
            "smallest_positive_overlap",
        ],
    )
    .into_iter()
    .map(|(name, ids, rank)| {
        let (a, am) = &meshes[&ids[0]];
        let (b, bm) = &meshes[&ids[1]];
        let mut report = qualification::check(name, a, Opposing::Compliant(b), [*am, *bm]);
        report["colliders"] = serde_json::json!(ids.map(|id| id.into_raw_parts()));
        report["selected_rank"] = serde_json::json!(rank);
        report
    })
    .collect();
    let steel_report = steel.report(&meshes);
    assert_eq!(total, admitted + unsupported + steel.admitted);
    assert_eq!(
        owner.encode().unwrap(),
        before,
        "pressure observation mutated the complete bank boundary"
    );
    let report = serde_json::json!({"observational_only":true,"gameplay_adopted":false,"applied_pressure_forces":false,"strain_credit_j":0.,"calibrated_material":false,"specified_pressure_scale_pa":200000,"geometry":{"clods":meshes.len(),"vertices":vertices,"faces":faces,"pressure_tets":cells,"captured_coefficients_normalized":false},"pairs":{"all":total,"admitted_clod_clod":admitted,"outside_qualified_scope":unsupported,"positive_finite_overlap":positive,"zero_finite_overlap":zero,"zero_overlap_with_negative_cached_contact":negative_cached_at_zero_overlap,"outside_shape_census":outside_shapes},"elastic_potential_j":potential,"overlap_volume_m3":volume,"directional_pressure_tie_volume_m3":ties,"max_force_pair_residual_n":worst_force,"max_moment_pair_residual_nm":worst_moment,"max_power_residual_w":worst_power,"input_encoding_roundtrip_exact":before==boundary,"saved_pose_and_partition_qualification":qualifications,"rigid_steel_union":steel_report,"failure_count":failure_count,"failures_first16":failures,"complete_owner_encoding_unchanged":true});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    assert!(bytes.len() < 1024 * 1024);
    std::fs::write(std::env::var_os("SLUICE_PRESSURE_REPORT").unwrap(), bytes).unwrap();
    println!("FINITE_SAVED clods={} admitted={admitted} positive={positive} zero={zero} unsupported={unsupported} failures={} max_force={worst_force}N max_moment={worst_moment}Nm max_power={worst_power}W; encoded owner unchanged",meshes.len(),failure_count);
    assert!(
        failures.is_empty(),
        "finite pressure evaluation failed; bounded reproducer report written"
    );
    assert!(
        qualifications.iter().all(|j| j["passed"] == true),
        "saved finite-pressure derivative/partition qualification failed; report written"
    );
    assert_eq!(qualifications.len(), 4);
    assert!(
        steel_report["passed"] == true,
        "registered steel qualification failed; report written"
    );
    assert!(admitted > 0 && positive > 0);
    assert_eq!(positive + zero, admitted);
    assert!(
        worst_force < 1e-5 && worst_moment < 1e-5 && worst_power < 1e-7,
        "finite pressure physical closure failed"
    );
}
