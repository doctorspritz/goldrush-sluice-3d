//! Read-only geometry audit. No engine distance query supplies the oracle.
use super::*;
use rapier3d::parry::query::{DefaultQueryDispatcher, PersistentQueryDispatcher};
use rapier3d::parry::shape::Shape;

fn vector(v: r::Vector) -> DVec3 {
    DVec3::from_array(v.to_array().map(f64::from))
}
fn transform(pose: &r::Pose, point: DVec3) -> DVec3 {
    // Promote the actual f32 coefficients exactly, before arithmetic/JSON.
    // Normalizing or serializing as short f32 decimals changes micrometre gaps.
    let q = DQuat::from_array(pose.rotation.to_array().map(f64::from));
    q * point + vector(pose.translation)
}
fn vertices(shape: &dyn Shape, id: u32) -> (r::Pose, Vec<DVec3>, &'static str) {
    if let Some(compound) = shape.as_compound() {
        let (pose, child) = &compound.shapes()[id as usize];
        let (nested, points, kind) = vertices(child.as_ref(), 0);
        assert_eq!(
            nested,
            r::Pose::IDENTITY,
            "nested compound needs an explicit audit"
        );
        return (*pose, points, kind);
    }
    let points = if let Some(hull) = shape.as_convex_polyhedron() {
        (hull.points().to_vec(), "convex")
    } else if let Some(cuboid) = shape.as_cuboid() {
        (
            (0..8)
                .map(|i| {
                    r::Vector::new(
                        if i & 1 == 0 {
                            -cuboid.half_extents.x
                        } else {
                            cuboid.half_extents.x
                        },
                        if i & 2 == 0 {
                            -cuboid.half_extents.y
                        } else {
                            cuboid.half_extents.y
                        },
                        if i & 4 == 0 {
                            -cuboid.half_extents.z
                        } else {
                            cuboid.half_extents.z
                        },
                    )
                })
                .collect::<Vec<_>>(),
            "cuboid",
        )
    } else if let Some(field) = shape.as_heightfield() {
        (
            field
                .triangle_at_id(id)
                .expect("registered terrain triangle")
                .vertices()
                .to_vec(),
            "heightfield_triangle",
        )
    } else if let Some(mesh) = shape.as_trimesh() {
        (mesh.triangle(id).vertices().to_vec(), "mesh_triangle")
    } else if let Some(triangle) = shape.as_triangle() {
        (triangle.vertices().to_vec(), "triangle")
    } else {
        panic!("Unsupported registered shape: {:?}", shape.shape_type());
    };
    (
        r::Pose::IDENTITY,
        points.0.into_iter().map(vector).collect(),
        points.1,
    )
}
fn world_point(
    world: &r::PhysicsWorld,
    collider: &r::Collider,
    child: &r::Pose,
    p: DVec3,
    stored: bool,
) -> DVec3 {
    let p = transform(child, p);
    if !stored {
        if let Some(parent) = collider.parent() {
            return transform(
                world.bodies[parent].position(),
                transform(collider.position_wrt_parent().unwrap(), p),
            );
        }
    }
    transform(collider.position(), p)
}
fn projection_gap(a: &[DVec3], b: &[DVec3], axis: DVec3) -> f64 {
    let lo = |p: &[DVec3]| p.iter().map(|p| p.dot(axis)).fold(f64::INFINITY, f64::min);
    let hi = |p: &[DVec3]| {
        p.iter()
            .map(|p| p.dot(axis))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    (lo(b) - hi(a)).max(lo(a) - hi(b))
}
fn face_axes(points: &[DVec3], mut axis: impl FnMut(DVec3)) {
    // Every face is represented by a triple. Extra chord planes cannot
    // falsely separate genuinely intersecting convex hulls.
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            for k in j + 1..points.len() {
                axis((points[j] - points[i]).cross(points[k] - points[i]));
            }
        }
    }
}
fn separation(a: &[DVec3], b: &[DVec3]) -> (f64, DVec3) {
    let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
    let mut axis = |n: DVec3| {
        if n.length_squared() > 1e-24 {
            let n = n.normalize();
            let gap = projection_gap(a, b, n);
            if gap > best.0 {
                best = (gap, n);
            }
        }
    };
    face_axes(a, &mut axis);
    face_axes(b, &mut axis);
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            for k in 0..b.len() {
                for l in k + 1..b.len() {
                    axis((a[j] - a[i]).cross(b[l] - b[k]));
                }
            }
        }
    }
    best
}
fn outside(points: &[DVec3], point: DVec3) -> f64 {
    let mut result = 0f64;
    let mut axis = |n: DVec3| {
        if n.length_squared() > 1e-24 {
            let n = n.normalize();
            let hi = points
                .iter()
                .map(|p| p.dot(n))
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = points
                .iter()
                .map(|p| p.dot(n))
                .fold(f64::INFINITY, f64::min);
            result = result.max(point.dot(n) - hi).max(lo - point.dot(n));
        }
    };
    face_axes(points, &mut axis);
    if points.len() == 3 {
        let n = (points[1] - points[0]).cross(points[2] - points[0]);
        for i in 0..3 {
            axis((points[(i + 1) % 3] - points[i]).cross(n));
        }
    }
    result
}
#[test]
fn independent_registered_geometry_oracle_detects_true_overlap_and_false_witnesses() {
    let q = DQuat::from_rotation_y(0.37) * DQuat::from_rotation_z(0.7);
    let origin = DVec3::new(18., 0.9, 22.);
    let points: Vec<_> = (0..8)
        .map(|i| {
            DVec3::new(
                if i & 1 == 0 { -0.05 } else { 0.05 },
                if i & 2 == 0 { -0.05 } else { 0.05 },
                if i & 4 == 0 { -0.05 } else { 0.05 },
            )
        })
        .collect();
    let a: Vec<_> = points.iter().map(|p| origin + q * *p).collect();
    for gap in [-0.0001, -0.00002, 0., 0.00002, 0.0001] {
        let b: Vec<_> = points
            .iter()
            .map(|p| origin + q * (*p + DVec3::X * (0.1 + gap)))
            .collect();
        let (measured, _) = separation(&a, &b);
        assert!((measured - gap).abs() < 1e-11);
        assert_eq!(measured < -0.00001, gap < -0.00001);
    }
    assert!(outside(&a, origin + q * DVec3::new(0.05002, 0., 0.)) > 0.000019999);
    assert!(outside(&a, origin).abs() < 1e-12);
    let pose = r::Pose::from_translation(r::Vector::new(20.3, 0., 0.));
    assert_eq!(transform(&pose, DVec3::ZERO).x, f64::from(20.3f32));
    assert!((transform(&pose, DVec3::ZERO).x - 20.3f64).abs() > 0.0000001);
}

#[test]
#[ignore = "explicit read-only saved-bank geometry audit, not an accuracy pass"]
fn saved_bank_registered_geometry_audit() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let before_bytes =
        std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").expect("before owner")).unwrap();
    let after_bytes =
        std::fs::read(std::env::var_os("SLUICE_ENERGY_REFERENCE").expect("after owner")).unwrap();
    let before = RetainedSoil::decode(&before_bytes, &env).unwrap();
    let after = RetainedSoil::decode(&after_bytes, &env).unwrap();
    let bm = before.model.as_ref().unwrap();
    let am = after.model.as_ref().unwrap();
    let path = std::path::PathBuf::from(
        std::env::var_os("SLUICE_GEOMETRY_AUDIT").expect("fresh audit JSON"),
    );
    assert!(!path.exists());
    let label = |h: Option<r::RigidBodyHandle>| {
        h.map(|h| {
            if let Some(i) = am.bodies.iter().position(|x| *x == h) {
                format!("machine-{i}")
            } else if let Some(i) = am.soil_handles.iter().position(|x| *x == h) {
                format!("clod-{i}")
            } else {
                "static".into()
            }
        })
        .unwrap_or("static".into())
    };
    let compact = std::env::var_os("SLUICE_GEOMETRY_COMPACT").is_some();
    let mut rows = Vec::new();
    let mut point_count = 0;
    for pair in am.world.narrow_phase.contact_pairs() {
        let ac = &am.world.colliders[pair.collider1];
        let bc = &am.world.colliders[pair.collider2];
        let clustered = !pair.rigid().unwrap().solver_clusters.is_empty();
        for manifold in pair.solver_manifolds() {
            if !manifold.points.iter().any(|p| p.data.impulse > 0.) {
                continue;
            }
            let (ap, av, ak) = vertices(ac.shape(), manifold.subshape1);
            let (bp, bv, bk) = vertices(bc.shape(), manifold.subshape2);
            let aq = manifold
                .subshape_pos1()
                .copied()
                .unwrap_or(r::Pose::IDENTITY);
            let bq = manifold
                .subshape_pos2()
                .copied()
                .unwrap_or(r::Pose::IDENTITY);
            if !clustered {
                assert_eq!(ap, aq, "registered/query child frame differs");
                assert_eq!(bp, bq, "registered/query child frame differs");
            }
            let normal = vector(manifold.data.normal);
            let mut epochs = Vec::new();
            for (name, model, stored) in [
                ("before_body", bm, false),
                ("before_stored", bm, true),
                ("after_body", am, false),
                ("after_stored", am, true),
            ] {
                let ac = &model.world.colliders[pair.collider1];
                let bc = &model.world.colliders[pair.collider2];
                let a: Vec<_> = av
                    .iter()
                    .map(|p| world_point(&model.world, ac, &ap, *p, stored))
                    .collect();
                let b: Vec<_> = bv
                    .iter()
                    .map(|p| world_point(&model.world, bc, &bp, *p, stored))
                    .collect();
                let (gap, axis) = separation(&a, &b);
                let points: Vec<_> = manifold.points.iter().enumerate().map(|(i, p)| {
                    let p1 = world_point(&model.world, ac, &aq, vector(p.local_p1), stored);
                    let p2 = world_point(&model.world, bc, &bq, vector(p.local_p2), stored);
                    serde_json::json!({"id":i,"impulse_ns":f64::from(p.data.impulse),"cached_distance_m":f64::from(p.dist),"witness_normal_gap_m":(p2-p1).dot(normal),"outside_a_m":outside(&a,p1),"outside_b_m":outside(&b,p2),"world_a_m":p1.to_array(),"world_b_m":p2.to_array()})
                }).collect();
                let active: Vec<_> = points
                    .iter()
                    .filter(|p| p["impulse_ns"].as_f64().unwrap() > 0.)
                    .collect();
                let minimum = |key: &str| {
                    active
                        .iter()
                        .map(|p| p[key].as_f64().unwrap())
                        .reduce(f64::min)
                };
                let max_outside = active
                    .iter()
                    .map(|p| {
                        p["outside_a_m"]
                            .as_f64()
                            .unwrap()
                            .max(p["outside_b_m"].as_f64().unwrap())
                    })
                    .reduce(f64::max);
                let mut epoch = serde_json::json!({"name":name,"sat_gap_m":gap,"sat_axis":axis.to_array(),"active_points":active.len(),"cached_min_m":minimum("cached_distance_m"),"witness_normal_min_m":minimum("witness_normal_gap_m"),"maximum_active_witness_outside_m":max_outside});
                if !compact {
                    epoch["vertices_a_m"] =
                        serde_json::json!(a.iter().map(|p| p.to_array()).collect::<Vec<_>>());
                    epoch["vertices_b_m"] =
                        serde_json::json!(b.iter().map(|p| p.to_array()).collect::<Vec<_>>());
                    epoch["points"] = serde_json::json!(points);
                }
                epochs.push(epoch);
            }
            // Solver clusters combine several primitives in collider-local
            // space. Their first subshape id cannot certify the entire union.
            let mut fresh_queries = Vec::new();
            if !clustered
                && epochs[0]["sat_gap_m"].as_f64().unwrap() > 0.00001
                && manifold
                    .points
                    .iter()
                    .any(|p| p.data.impulse > 0. && p.dist < -0.00001)
            {
                let acb = &bm.world.colliders[pair.collider1];
                let bcb = &bm.world.colliders[pair.collider2];
                let a = acb.shape().as_convex_polyhedron().unwrap();
                let b = bcb.shape().as_convex_polyhedron().unwrap();
                let pos = (acb.position() * ap).inverse() * (bcb.position() * bp);
                for seeded in [false, true] {
                    let mut fresh: rapier3d::geometry::ContactManifold = Default::default();
                    if seeded {
                        fresh.local_n1 = manifold.local_n1;
                    }
                    DefaultQueryDispatcher
                        .contact_manifold_convex_convex(
                            &pos,
                            a,
                            b,
                            None,
                            None,
                            am.world.integration_parameters.prediction_distance(),
                            &mut fresh,
                        )
                        .unwrap();
                    let min = fresh
                        .points
                        .iter()
                        .map(|p| f64::from(p.dist))
                        .reduce(f64::min);
                    fresh_queries.push(serde_json::json!({"cached_normal_seed":seeded,"points":fresh.points.len(),"minimum_distance_m":min,"local_normal":vector(fresh.local_n1).to_array()}));
                }
            }
            let solver_points: Vec<_> = manifold.data.solver_contacts.iter().map(|p| {
                let (p1,p2)=manifold.data.solver_contact_world_points(p,&bm.world.bodies);
                serde_json::json!({"contact_id":p.contact_id[0] & !rapier3d::geometry::NEW_CONTACT_BIT,"cached_distance_m":f64::from(p.dist),"before_live_distance_m":(vector(p2)-vector(p1)).dot(normal),"world_a_m":vector(p1).to_array(),"world_b_m":vector(p2).to_array()})
            }).collect();
            point_count += manifold.points.len();
            rows.push(serde_json::json!({"a":label(ac.parent()),"b":label(bc.parent()),"shape_a":ak,"shape_b":bk,"subshape_a":manifold.subshape1,"subshape_b":manifold.subshape2,"normal":normal.to_array(),"clustered":clustered,"primitive_geometry_certified":!clustered,"fresh_queries":fresh_queries,"epochs":epochs,"solver_points":solver_points}));
        }
    }
    let result = serde_json::json!({"clods":after.soil().clods.len(),"dynamic_bodies":am.world.bodies.iter().filter(|(_,b)|b.is_dynamic()).count(),"active_manifolds":rows.len(),"manifold_points":point_count,"geometry_tolerance_m":0.00001,"observational_only":true,"compact_metrics":compact,"registered_coordinates_promoted_to_f64_before_transform":true,"epochs_do_not_assert_narrow_phase_query_time":true,"manifolds":rows});
    std::fs::write(&path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    assert_eq!(
        before_bytes,
        std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap()
    );
    assert_eq!(
        after_bytes,
        std::fs::read(std::env::var_os("SLUICE_ENERGY_REFERENCE").unwrap()).unwrap()
    );
    println!(
        "READ-ONLY GEOMETRY AUDIT manifolds={} points={point_count} path={}",
        result["active_manifolds"],
        path.display()
    );
}

fn independent_energy(owner: &RetainedSoil, gravity: Vec3) -> f64 {
    owner
        .model
        .as_ref()
        .unwrap()
        .world
        .bodies
        .iter()
        .filter(|(_, b)| b.is_dynamic())
        .map(|(_, b)| {
            let inertia = b.mass_properties().effective_angular_inertia();
            let columns =
                [r::Vector::X, r::Vector::Y, r::Vector::Z].map(|axis| vector(inertia * axis));
            let tensor = DMat3::from_cols(columns[0], columns[1], columns[2]);
            let velocity = vector(b.linvel());
            let omega = vector(b.angvel());
            0.5 * f64::from(b.mass()) * velocity.length_squared() + 0.5 * omega.dot(tensor * omega)
                - f64::from(b.mass()) * gravity.as_dvec3().dot(vector(b.center_of_mass()))
        })
        .sum()
}
#[test]
#[ignore = "explicit controlled contact-geometry trial, not whole-trajectory acceptance"]
fn saved_bank_contact_geometry_trial() {
    let session = bank();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let before = RetainedSoil::decode(
        &std::fs::read(std::env::var_os("SLUICE_ENERGY_BOUNDARY").unwrap()).unwrap(),
        &env,
    )
    .unwrap();
    let reference = RetainedSoil::decode(
        &std::fs::read(std::env::var_os("SLUICE_ENERGY_REFERENCE").unwrap()).unwrap(),
        &env,
    )
    .unwrap();
    let directory = std::path::PathBuf::from(
        std::env::var_os("SLUICE_GEOMETRY_TRIAL").expect("fresh trial directory"),
    );
    assert!(!directory.exists());
    std::fs::create_dir(&directory).unwrap();
    let bm = before.model.as_ref().unwrap();
    let am = reference.model.as_ref().unwrap();
    for pair in am.world.narrow_phase.contact_pairs() {
        let ac = &am.world.colliders[pair.collider1];
        let bc = &am.world.colliders[pair.collider2];
        if ac.parent() != Some(am.soil_handles[636]) || bc.parent() != Some(am.soil_handles[639]) {
            continue;
        }
        let m = &pair.rigid().unwrap().manifolds[0];
        let ap = bm.world.colliders[pair.collider1].position();
        let bp = bm.world.colliders[pair.collider2].position();
        let relative = ap.inverse() * *bp;
        let mut cached = m.clone();
        cached.data = Default::default();
        for p in &mut cached.points {
            p.data = Default::default();
        }
        let fixture = serde_json::json!({"a":ac.shape().as_convex_polyhedron().unwrap().points().iter().map(|p|vector(*p).to_array()).collect::<Vec<_>>(),"b":bc.shape().as_convex_polyhedron().unwrap().points().iter().map(|p|vector(*p).to_array()).collect::<Vec<_>>(),"translation":vector(relative.translation).to_array(),"rotation":relative.rotation.to_array().map(f64::from),"cached_manifold":cached,"independent_world_sat_gap_m":0.0007492162898863342});
        std::fs::write(
            directory.join("separated-clods.json"),
            serde_json::to_vec_pretty(&fixture).unwrap(),
        )
        .unwrap();
    }
    assert!(directory.join("separated-clods.json").exists());
    let initial = independent_energy(&before, env.gravity);
    let mut rows = Vec::new();
    for (name, verified) in [("legacy", false), ("verified", true)] {
        let mut owner = before.clone();
        if verified {
            owner.dispatcher = Dispatcher::RigidSoilVerified;
            crate::soil_contacts::restore_verified_dispatcher(
                &mut owner.model.as_mut().unwrap().world,
            );
        }
        let seed = owner.encode().unwrap();
        let started = std::time::Instant::now();
        let step = owner.step(MachineInput::default(), 1. / 480., &env);
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        match step {
            Ok(()) => {
                if !verified {
                    same(&owner, &reference, "unchanged legacy geometry policy");
                }
                let delta = independent_energy(&owner, env.gravity) - initial;
                if verified {
                    std::fs::write(directory.join("trial-before.bin"), seed).unwrap();
                    std::fs::write(directory.join("trial-after.bin"), owner.encode().unwrap())
                        .unwrap();
                }
                rows.push(serde_json::json!({"name":name,"accepted":true,"delta_energy_j":delta,"step_ms":elapsed,"clods":owner.soil().clods.len()}));
            }
            Err(error) => rows.push(
                serde_json::json!({"name":name,"accepted":false,"error":error,"step_ms":elapsed}),
            ),
        }
    }
    let report = serde_json::json!({"variants":rows,"same_input_boundary":true,"legacy_reference_state_exact":true,"diagnostic_policy_upgrade_is_explicit":true,"timestep_convergence_test":false,"whole_trajectory_acceptance":false});
    std::fs::write(
        directory.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("GEOMETRY TRIAL {}", report);
}
