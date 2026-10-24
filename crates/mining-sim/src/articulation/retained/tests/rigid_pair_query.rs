//! Explicit bounded dispatcher queries on the registered saved clod pair.
//!
//! Install below articulation::retained::tests (like contact_geometry). This
//! module decodes the exact saved owner itself. It queries existing registered
//! shapes without reconstructing their hulls, and never steps or edits the owner.
//!
//! A persistent-dispatcher query is not a replay of NarrowPhase's earlier
//! OUTCOME_RECYCLED decision. Query modes below name exactly what is executed.
use super::*;
use rapier3d::{
    geometry::{ContactData, ContactManifoldData},
    parry::{
        query::{ContactManifold, DefaultQueryDispatcher, PersistentQueryDispatcher},
        shape::{PolygonalFeature, PolygonalFeatureMap},
    },
};
use serde_json::{json, Value};

type Pair2242Manifold = ContactManifold<ContactManifoldData, ContactData>;

fn pair2242_pose(value: &Value) -> r::Pose {
    fn exact<const N: usize>(value: &Value) -> [f32; N] {
        let values = value.as_array().expect("captured coefficients");
        assert_eq!(values.len(), N);
        core::array::from_fn(|i| {
            let original = values[i].as_f64().unwrap();
            let coefficient = original as f32;
            assert!(coefficient.is_finite());
            assert_eq!(
                f64::from(coefficient),
                original,
                "capture is not an exact f32 coefficient"
            );
            coefficient
        })
    }
    let translation = r::Vector::from_array(exact(&value["translation"]));
    let rotation = r::Rotation::from_array(exact(&value["rotation_xyzw"]));
    assert!((rotation.length_squared() - 1.).abs() < 0.0001);
    // Never normalize or otherwise improve the recorded engine geometry.
    r::Pose::from_parts(translation, rotation)
}

fn pair2242_pose_json(pose: &r::Pose) -> Value {
    json!({"translation":pose.translation.to_array().map(f64::from),
           "rotation_xyzw":pose.rotation.to_array().map(f64::from)})
}

fn pair2242_manifold_json(m: &Pair2242Manifold, poses: &[r::Pose; 2]) -> Value {
    let points: Vec<_> = m.points.iter().map(|point| {
        let world1 = poses[0] * point.local_p1;
        let world2 = poses[1] * point.local_p2;
        let relative_point2 = poses[0].inverse() * world2;
        json!({
            "local_points_m":[point.local_p1.to_array(),point.local_p2.to_array()],
            "world_points_m":[world1.to_array(),world2.to_array()],
            "features":[format!("{:?}",point.fid1.unpack()),format!("{:?}",point.fid2.unpack())],
            "stored_dist_m":point.dist,
            "independently_recomputed_local_normal_gap_m":(relative_point2-point.local_p1).dot(m.local_n1),
        })
    }).collect();
    let minimum = m.points.iter().map(|point| point.dist).reduce(f32::min);
    let maximum = m.points.iter().map(|point| point.dist).reduce(f32::max);
    json!({
        "local_normals":[m.local_n1.to_array(),m.local_n2.to_array()],
        "world_outward_normal_side1":(poses[0].rotation*m.local_n1).to_array(),
        "point_count":points.len(),"min_stored_gap_m":minimum,"max_stored_gap_m":maximum,
        "points":points,
    })
}

fn pair2242_support_feature_json(feature: &PolygonalFeature) -> Value {
    json!({"feature":format!("{:?}",feature.fid.unpack()),
        "vertices":feature.vertices[..feature.num_vertices].iter().map(|p|p.to_array()).collect::<Vec<_>>(),
        "vertex_features":feature.vids[..feature.num_vertices].iter().map(|p|format!("{:?}",p.unpack())).collect::<Vec<_>>()})
}

#[test]
#[ignore = "explicit bounded registered-pair dispatcher experiment; not full-bank acceptance"]
fn saved_pair2242_convex_queries() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let bytes = std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap();
    let before = RetainedSoil::decode(&bytes, &env).unwrap();
    assert_eq!(
        before.encode().unwrap(),
        bytes,
        "decode changed the original saved boundary"
    );
    assert_eq!(
        before.dispatcher,
        crate::contact_history::Dispatcher::RigidSoilVerified
    );
    let summary: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("SLUICE_RIGID_WORK_SUMMARY").unwrap()).unwrap(),
    )
    .unwrap();
    let selected = summary["top16_positive_biased_solve_blocks_with_geometry"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["manifold"] == json!([2242, 0]))
        .expect("observed pair2242 row");
    assert_eq!(selected["collider_handles"], json!([[898, 0], [900, 0]]));
    assert_eq!(selected["clustered"], false);
    let handles = [
        r::ColliderHandle::from_raw_parts(898, 0),
        r::ColliderHandle::from_raw_parts(900, 0),
    ];
    let world = &before.model.as_ref().unwrap().world;
    let a = &world.colliders[handles[0]];
    let b = &world.colliders[handles[1]];
    let ah = a
        .shape()
        .as_convex_polyhedron()
        .expect("actual registered convex A");
    let bh = b
        .shape()
        .as_convex_polyhedron()
        .expect("actual registered convex B");
    assert!(
        ah.points().len() <= 64 && bh.points().len() <= 64,
        "bounded fixed pair fixture"
    );
    let pair = world
        .narrow_phase
        .contact_pair(handles[0], handles[1])
        .expect("saved original contact pair");
    assert_eq!(
        [pair.collider1, pair.collider2],
        handles,
        "preserve actual pair ordering"
    );
    let rigid = pair.rigid().unwrap();
    assert!(
        rigid.solver_clusters.is_empty(),
        "primitive manifold provenance changed"
    );
    let original = rigid
        .manifolds
        .first()
        .expect("saved primitive manifold")
        .clone();
    let mut touch_world = crate::contact_history::clone_world(world);
    crate::soil_contacts::restore_dispatcher(&mut touch_world);
    let default = DefaultQueryDispatcher;
    let dispatchers: [(
        &str,
        &dyn PersistentQueryDispatcher<ContactManifoldData, ContactData>,
    ); 3] = [
        ("Default", &default),
        ("TOUCH_ONLY", touch_world.narrow_phase.query_dispatcher()),
        ("VERIFIED", world.narrow_phase.query_dispatcher()),
    ];
    let pose_sets = [
        (
            "captured_current_solver_collider_poses",
            [
                pair2242_pose(&selected["collider_world_poses"][0]),
                pair2242_pose(&selected["collider_world_poses"][1]),
            ],
        ),
        (
            "captured_engine_collider_poses_at_step_entry",
            [
                pair2242_pose(&selected["sides"][0]["captured_collider_world_pose"]),
                pair2242_pose(&selected["sides"][1]["captured_collider_world_pose"]),
            ],
        ),
        (
            "saved_before_engine_collider_poses",
            [*a.position(), *b.position()],
        ),
    ];
    let prediction = world.integration_parameters.prediction_distance();
    let mut rows = Vec::new();
    for (pose_source, poses) in &pose_sets {
        let relative = poses[0].inverse() * poses[1];
        let mut fast_path_probe = original.clone();
        // Exact constants from pinned Parry0.31.1 pfm/pfm path. This invokes its
        // own coherence predicate, not a hand-derived prediction of its result.
        let cached_feature_fast_path_accepts =
            fast_path_probe.try_update_contacts_eps(&relative, 0.999_847_7, 1.0e-6);
        for (name, dispatcher) in &dispatchers {
            for mode in [
                "fresh_empty",
                "saved_cached",
                "saved_cached_mark_shapes_deformed",
            ] {
                let mut m = if mode == "fresh_empty" {
                    Pair2242Manifold::new()
                } else {
                    original.clone()
                };
                if mode == "saved_cached_mark_shapes_deformed" {
                    m.mark_shapes_deformed();
                }
                dispatcher
                    .contact_manifold_convex_convex(
                        &relative,
                        a.shape(),
                        b.shape(),
                        None,
                        None,
                        prediction,
                        &mut m,
                    )
                    .unwrap();
                let mut fa = PolygonalFeature::default();
                let mut fb = PolygonalFeature::default();
                ah.local_support_feature(m.local_n1, &mut fa);
                bh.local_support_feature(m.local_n2, &mut fb);
                rows.push(json!({
                    "pose_source":pose_source,"world_poses":poses.iter().map(pair2242_pose_json).collect::<Vec<_>>(),
                    "relative_pose":pair2242_pose_json(&relative),
                    "dispatcher":name,"query_mode":mode,
                    "executed_api":"contact_manifold_convex_convex",
                    "replayed_narrow_phase_recycle_decision":false,
                    "original_cached_feature_fast_path_accepts_at_this_pose":cached_feature_fast_path_accepts,
                    "normal_selected_support_features":[pair2242_support_feature_json(&fa),pair2242_support_feature_json(&fb)],
                    "result":pair2242_manifold_json(&m,poses),
                }));
            }
        }
    }
    assert_eq!(rows.len(), 27);
    assert_eq!(
        before.encode().unwrap(),
        bytes,
        "read-only fixture queries changed the saved owner"
    );
    let report = json!({
        "collider_handles":[[898,0],[900,0]],"observed_manifold_reference":[2242,0],
        "shape_source":"exact registered SharedShape references; no convex-hull reconstruction",
        "registered_local_vertices":[ah.points().iter().map(|p|p.to_array()).collect::<Vec<_>>(),bh.points().iter().map(|p|p.to_array()).collect::<Vec<_>>()],
        "registered_shared_shapes":[serde_json::to_value(a.shared_shape()).unwrap(),serde_json::to_value(b.shared_shape()).unwrap()],
        "saved_original_manifold":serde_json::to_value(&original).unwrap(),
        "observed_solved_time_s":selected["solved_time_s"],
        "prediction_m":prediction,"queries":rows,
        "source_boundary_unchanged":true,
        "historical_query_time_or_recycle_path_reproduced":false,
        "independent_sat_result_join_required":true,
        "full_bank_passivity_or_candidate_admission_proved":false,
    });
    let path = std::path::PathBuf::from(std::env::var_os("SLUICE_PAIR2242_QUERY_REPORT").unwrap());
    assert!(!path.exists(), "fresh bounded fixture report required");
    let encoded = serde_json::to_vec_pretty(&report).unwrap();
    assert!(encoded.len() <= 1024 * 1024, "bounded pair fixture output");
    std::fs::write(&path, encoded).unwrap();
    println!(
        "PAIR2242 queries={} report={}",
        report["queries"].as_array().unwrap().len(),
        path.display()
    );
}
