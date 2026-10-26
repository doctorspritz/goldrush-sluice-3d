//! Actual registered ground triangles, bounded below by the site's bedrock
//! surface. This observes finite erodible support, not deep-bedrock contact,
//! physical material custody, terrain-edit work or a live contact policy.
use super::*;
use pressure::RegisteredTerrainLayer;
use rapier3d::parry::shape::HeightField;
use std::time::Instant;

fn captured_layer(
    world: &r::PhysicsWorld,
    id: r::ColliderHandle,
    mut layer: RegisteredTerrainLayer,
) -> Result<(RegisteredTerrainLayer, RigidMotion), String> {
    let collider = &world.colliders[id];
    let body = &world.bodies[collider
        .parent()
        .ok_or("Terrain layer needs a registered body")?];
    let pose = body.position();
    let q = rotation(pose.rotation);
    let translation = vector(pose.translation);
    let com = translation + q * vector(body.mass_properties().local_mprops.local_com);
    if let Some(local) = collider.position_wrt_parent() {
        layer = layer.captured_frame(vector(local.translation), rotation(local.rotation))?;
    }
    layer = layer.captured_frame(translation - com, q)?;
    Ok((
        layer,
        RigidMotion {
            center_of_mass_m: com,
            rotation: DQuat::IDENTITY,
            linear_velocity_m_s: vector(body.linvel()),
            angular_velocity_rad_s: vector(body.angvel()),
        },
    ))
}
fn fingerprint(field: &HeightField) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for triangle in field.triangles() {
        for point in triangle.vertices() {
            for v in point.to_array() {
                for b in f64::from(v).to_bits().to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    format!("{h:016x}")
}

#[test]
#[ignore = "explicit read-only whole-bank finite erodible-terrain qualification; no live adoption"]
fn saved_bank_finite_erodible_terrain_layer() {
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
    let mut meshes = HashMap::new();
    for handle in &model.soil_handles {
        for id in world.bodies[*handle].colliders() {
            meshes.insert(*id, construct(world, *id).unwrap());
        }
    }
    let mut layers = HashMap::new();
    let mut geometry = Vec::new();
    for (id, collider) in world.colliders.iter() {
        let Some(top) = collider.shape().as_heightfield() else {
            continue;
        };
        assert_eq!(
            (top.nrows(), top.ncols()),
            (crate::site::WIDTH - 1, crate::site::WIDTH - 1)
        );
        assert_eq!(top.scale().y, 1.);
        let heights = (0..crate::site::WIDTH)
            .flat_map(|x| {
                let site = &session.site;
                (0..crate::site::WIDTH)
                    .map(move |z| site.world.bedrock_elevation[site.world.idx(x, z)])
            })
            .collect();
        let bottom = HeightField::with_flags(
            r::Array2::new(crate::site::WIDTH, crate::site::WIDTH, heights),
            top.scale(),
            top.flags(),
        );
        let layer = RegisteredTerrainLayer::from_heightfields(top, &bottom).unwrap();
        let local_volume = layer.volume_m3();
        let (layer, motion) = captured_layer(world, id, layer).unwrap();
        geometry.push(serde_json::json!({"collider":id.into_raw_parts(),"triangles":layer.triangles(),"nonzero_tetrahedra":layer.tetrahedra(),"reference_envelope_m3":local_volume,"captured_envelope_m3":layer.volume_m3(),"top_coefficient_fnv1a64":fingerprint(top),"lower_coefficient_fnv1a64":fingerprint(&bottom),"lower_boundary":"actual Site bedrock elevation; matching registered grid/topology","material_inventory_derived_from_envelope":false}));
        layers.insert(id, (layer, motion));
    }
    assert_eq!(layers.len(), 1);
    assert_eq!(meshes.len(), 1040);
    let mut pairs = 0;
    let mut positive = 0;
    let mut zero = 0;
    let mut failures = Vec::new();
    let mut potential = 0.;
    let mut volume = 0.;
    let mut force = 0_f64;
    let mut moment = 0_f64;
    let mut power = 0_f64;
    let mut candidate_max = 0;
    let mut samples = qualification::Candidates::new();
    let start = Instant::now();
    for pair in world.narrow_phase.contact_pairs() {
        let ids = if meshes.contains_key(&pair.collider1) && layers.contains_key(&pair.collider2) {
            [pair.collider1, pair.collider2]
        } else if meshes.contains_key(&pair.collider2) && layers.contains_key(&pair.collider1) {
            [pair.collider2, pair.collider1]
        } else {
            continue;
        };
        let (a, am) = &meshes[&ids[0]];
        let (layer, bm) = &layers[&ids[1]];
        pairs += 1;
        match pressure::evaluate(a, *am, Opposing::TerrainLayer(layer), *bm) {
            Ok(r) => {
                if r.overlap_volume_m3 > 0. {
                    positive += 1;
                    qualification::remember(
                        &mut samples,
                        "largest_terrain_force",
                        r.forces_n[0].length(),
                        ids,
                    );
                } else {
                    zero += 1;
                }
                potential += r.elastic_energy_j;
                volume += r.overlap_volume_m3;
                force = force.max(r.force_residual_n.length());
                moment = moment.max(r.moment_residual_nm.length());
                power = power.max(r.power_residual_w.abs());
                candidate_max = candidate_max.max(r.union_candidate_tetrahedra);
            }
            Err(error) => {
                if failures.len() < 8 {
                    failures.push(serde_json::json!({"colliders":ids.map(|id|id.into_raw_parts()),"error":error,"clod":descriptor(a,*am)}));
                }
            }
        }
        if pairs % 128 == 0 {
            println!(
                "FINITE_TERRAIN_PROGRESS pairs={pairs} positive={positive} failures={} seconds={}",
                pairs - positive - zero,
                start.elapsed().as_secs_f64()
            );
        }
    }
    let wall_seconds = start.elapsed().as_secs_f64();
    let qualified: Vec<_> = qualification::selected(&samples, &["largest_terrain_force"])
        .into_iter()
        .map(|(name, ids, rank)| {
            let (a, am) = &meshes[&ids[0]];
            let (layer, bm) = &layers[&ids[1]];
            let mut report =
                qualification::check(name, a, Opposing::TerrainLayer(layer), [*am, *bm]);
            report["colliders"] = serde_json::json!(ids.map(|id| id.into_raw_parts()));
            report["selected_rank"] = serde_json::json!(rank);
            report
        })
        .collect();
    assert_eq!(
        owner.encode().unwrap(),
        before,
        "finite terrain observation changed the complete owner"
    );
    let report = serde_json::json!({"observational_only":true,"gameplay_adopted":false,"calibrated_material":false,"applied_pressure_forces":false,"strain_credit_j":0.,"bedrock_contact_included":false,"terrain_edit_work_qualified":false,"geometry":geometry,"clods":meshes.len(),"terrain_pairs":pairs,"positive_erodible_overlap":positive,"zero_erodible_overlap":zero,"failure_count":pairs-positive-zero,"failures_first8":failures,"elastic_potential_j":potential,"overlap_m3":volume,"maximum_force_residual_n":force,"maximum_moment_residual_nm":moment,"maximum_power_residual_w":power,"maximum_candidate_tetrahedra_per_pair":candidate_max,"cpu_wall_seconds":wall_seconds,"pose_and_partition_qualification":qualified,"complete_owner_encoding_unchanged":true,"input_encoding_roundtrip_exact":before==boundary});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    assert!(bytes.len() < 2 * 1024 * 1024);
    std::fs::write(
        std::env::var_os("SLUICE_TERRAIN_PRESSURE_REPORT").unwrap(),
        bytes,
    )
    .unwrap();
    println!("FINITE_TERRAIN pairs={pairs} positive={positive} zero={zero} seconds={wall_seconds} max_force={force}N max_moment={moment}Nm max_power={power}W");
    assert_eq!(pairs, 1040);
    assert!(failures.is_empty());
    assert_eq!(positive + zero, pairs);
    assert!(positive > 0);
    assert_eq!(qualified.len(), 1);
    assert!(qualified.iter().all(|q| q["passed"] == true));
    assert!(force < 1e-5 && moment < 1e-5 && power < 1e-7);
}
