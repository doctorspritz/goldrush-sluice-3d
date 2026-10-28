use super::*;
use glam::DQuat;
use rapier3d::prelude as r;
fn solid(half: DVec3) -> RegisteredSolid {
    let vertices: Vec<_> = (0..8)
        .map(|i| {
            r::Vector::new(
                if i & 1 == 0 {
                    -half.x as f32
                } else {
                    half.x as f32
                },
                if i & 2 == 0 {
                    -half.y as f32
                } else {
                    half.y as f32
                },
                if i & 4 == 0 {
                    -half.z as f32
                } else {
                    half.z as f32
                },
            )
        })
        .collect();
    let shape = r::SharedShape::convex_hull(&vertices).unwrap();
    RegisteredSolid::from_hull(shape.as_convex_polyhedron().unwrap()).unwrap()
}
fn pressure(half: DVec3, pa: f64) -> PressureBody {
    PressureBody::new(solid(half), pa).unwrap()
}
fn motions() -> [RigidMotion; 2] {
    [
        RigidMotion {
            center_of_mass_m: DVec3::new(0.12, 0.76, 0.08),
            rotation: DQuat::from_rotation_z(0.07) * DQuat::from_rotation_x(-0.11),
            ..Default::default()
        },
        RigidMotion {
            rotation: DQuat::from_rotation_y(0.1) * DQuat::from_rotation_x(0.02),
            ..Default::default()
        },
    ]
}
fn sample(a: &PressureBody, b: &PressureBody, m: [RigidMotion; 2], rigid: bool) -> Response {
    evaluate(
        a,
        m[0],
        if rigid {
            Opposing::Rigid(b.solid())
        } else {
            Opposing::Compliant(b)
        },
        m[1],
    )
    .unwrap()
}
fn balance(r: &Response) {
    assert!(r.elastic_energy_j >= 0. && r.overlap_volume_m3 >= 0.);
    assert!(
        r.force_residual_n.length() < 1e-6,
        "force residual {:?}",
        r.force_residual_n
    );
    assert!(
        r.moment_residual_nm.length() < 1e-6,
        "moment residual {:?}",
        r.moment_residual_nm
    );
    assert!(
        r.power_residual_w.abs() < 1e-8,
        "power residual {}W",
        r.power_residual_w
    );
    assert!((r.weak_pressure_volume_m3.iter().sum::<f64>() - r.overlap_volume_m3).abs() < 1e-12);
}
#[test]
fn finite_rigid_punch_matches_analytic_layer_force_moment_and_energy() {
    let a = pressure(DVec3::new(2., 1., 2.), 2e6);
    let b = solid(DVec3::new(0.2, 0.5, 0.3));
    let delta = 0.01;
    let m = RigidMotion {
        center_of_mass_m: DVec3::new(0.3, -1.5 + delta, 0.2),
        ..Default::default()
    };
    let response = evaluate(&a, RigidMotion::default(), Opposing::Rigid(&b), m).unwrap();
    balance(&response);
    // Match the actual registered f32 punch dimensions, promoted before use.
    let area = 4. * (0.2f32 as f64) * (0.3f32 as f64);
    let force = 2e6 * delta * area;
    assert!((response.forces_n[0] - DVec3::Y * force).length() < 1e-7);
    assert!((response.torques_nm[0] - DVec3::new(-0.2 * force, 0., 0.3 * force)).length() < 1e-7);
    assert!(response.torques_nm[1].length() < 1e-7);
    assert!((response.elastic_energy_j - 0.5 * 2e6 * area * delta * delta).abs() < 1e-9);
    assert!((response.overlap_volume_m3 - area * delta).abs() < 1e-14);
    println!(
        "FINITE_PUNCH force={}N potential={}J volume={}m3",
        response.forces_n[0].y, response.elastic_energy_j, response.overlap_volume_m3
    );
}
#[test]
fn a_separated_finite_face_does_not_inherit_an_infinite_plane_force() {
    let a = pressure(DVec3::new(2., 1., 2.), 2e6);
    let b = solid(DVec3::new(0.2, 0.5, 0.3));
    let m = RigidMotion {
        center_of_mass_m: DVec3::new(3., -1.49, 0.),
        ..Default::default()
    };
    let response = evaluate(&a, RigidMotion::default(), Opposing::Rigid(&b), m).unwrap();
    assert_eq!(response.elastic_energy_j, 0.);
    assert_eq!(response.forces_n, [DVec3::ZERO; 2]);
    let patch = super::super::SurfacePatch::new(&[
        DVec3::new(-2., -1., -2.),
        DVec3::new(2., -1., -2.),
        DVec3::new(2., -1., 2.),
        DVec3::new(-2., -1., 2.),
    ])
    .unwrap();
    let old = super::super::NormalLayer::new(2e6, 1., 0.)
        .unwrap()
        .evaluate(
            &patch,
            RigidMotion::default(),
            m,
            super::super::Plane::new(DVec3::Y * 0.5, DVec3::Y).unwrap(),
        )
        .unwrap();
    assert!(old.force_on_a_n.length() > 100_000.);
}
#[test]
fn finite_wrenches_are_the_negative_energy_derivative_in_all_world_pose_dofs() {
    let a = pressure(DVec3::new(0.6, 0.5, 0.7), 2e5);
    let b = pressure(DVec3::new(1., 0.4, 0.9), 3e5);
    let m = motions();
    let mut worst = 0f64;
    for rigid in [true, false] {
        let reference = sample(&a, &b, m, rigid);
        balance(&reference);
        for body in 0..2 {
            for dof in 0..6 {
                let axis = [DVec3::X, DVec3::Y, DVec3::Z][dof % 3];
                let h = 1e-6;
                let mut plus = m;
                let mut minus = m;
                if dof < 3 {
                    plus[body].center_of_mass_m += axis * h;
                    minus[body].center_of_mass_m -= axis * h;
                } else {
                    plus[body].rotation = DQuat::from_axis_angle(axis, h) * m[body].rotation;
                    minus[body].rotation = DQuat::from_axis_angle(axis, -h) * m[body].rotation;
                }
                let derivative = (sample(&a, &b, plus, rigid).elastic_energy_j
                    - sample(&a, &b, minus, rigid).elastic_energy_j)
                    / (2. * h);
                let force = if dof < 3 {
                    reference.forces_n[body]
                } else {
                    reference.torques_nm[body]
                }
                .dot(axis);
                let error = (derivative + force).abs() / force.abs().max(derivative.abs()).max(1.);
                worst = worst.max(error);
                assert!(error<1e-5,"rigid={rigid} body={body} dof={dof}: derivative={derivative} force={force} relative={error}");
            }
        }
    }
    println!("FINITE_GRADIENT 24 force/torque derivatives worst_relative_error={worst}");
}
#[test]
fn moving_finite_solids_close_instantaneous_power_against_independent_energy_motion() {
    let a = pressure(DVec3::new(0.6, 0.5, 0.7), 2e5);
    let b = pressure(DVec3::new(1., 0.4, 0.9), 3e5);
    let mut m = motions();
    m[0].linear_velocity_m_s = DVec3::new(0.1, -0.2, 0.3);
    m[0].angular_velocity_rad_s = DVec3::new(0.4, -0.3, 0.2);
    m[1].linear_velocity_m_s = DVec3::new(-0.2, 0.1, 0.01);
    m[1].angular_velocity_rad_s = DVec3::new(-0.1, 0.3, 0.1);
    for rigid in [true, false] {
        let r = sample(&a, &b, m, rigid);
        balance(&r);
        let h = 1e-6;
        let advance = |sign: f64| {
            m.map(|mut body| {
                body.center_of_mass_m += body.linear_velocity_m_s * (h * sign);
                body.rotation = DQuat::from_scaled_axis(body.angular_velocity_rad_s * (h * sign))
                    * body.rotation;
                body
            })
        };
        let derivative = (sample(&a, &b, advance(1.), rigid).elastic_energy_j
            - sample(&a, &b, advance(-1.), rigid).elastic_energy_j)
            / (2. * h);
        assert!(
            (derivative + r.body_power_w).abs() < 1e-4,
            "rigid={rigid} finite rate={derivative} power={}",
            r.body_power_w
        );
    }
}
#[test]
fn finite_pressure_field_is_unchanged_by_material_mesh_refinement() {
    let a = pressure(DVec3::new(0.6, 0.5, 0.7), 2e5);
    let b = pressure(DVec3::new(1., 0.4, 0.9), 3e5);
    let m = motions();
    let reference = sample(&a, &b, m, false);
    for (levels_a, levels_b) in [(1, 0), (0, 1), (1, 1), (2, 0)] {
        let r = sample(
            &a.refined(levels_a).unwrap(),
            &b.refined(levels_b).unwrap(),
            m,
            false,
        );
        balance(&r);
        assert!((r.elastic_energy_j - reference.elastic_energy_j).abs() < 1e-7);
        assert!((r.overlap_volume_m3 - reference.overlap_volume_m3).abs() < 1e-12);
        for body in 0..2 {
            assert!((r.forces_n[body] - reference.forces_n[body]).length() < 1e-5);
            assert!((r.torques_nm[body] - reference.torques_nm[body]).length() < 1e-5);
        }
    }
}
#[test]
fn invalid_parameters_and_unbounded_refinement_are_refused() {
    assert!(PressureBody::new(solid(DVec3::ONE), 0.).is_err());
    let a = pressure(DVec3::ONE, 1e5);
    assert!(a.refined(20).is_err());
    assert!(evaluate(
        &a,
        RigidMotion {
            rotation: DQuat::from_xyzw(0., 0., 0., 2.),
            ..Default::default()
        },
        Opposing::Rigid(a.solid()),
        RigidMotion::default()
    )
    .is_err());
}

#[test]
fn pressure_scale_and_body_exchange_preserve_the_finite_material_law() {
    let a = pressure(DVec3::new(0.6, 0.5, 0.7), 2e5);
    let b = pressure(DVec3::new(1., 0.4, 0.9), 3e5);
    let m = motions();
    let reference = sample(&a, &b, m, false);
    let swap = sample(&b, &a, [m[1], m[0]], false);
    balance(&swap);
    assert!((swap.elastic_energy_j - reference.elastic_energy_j).abs() < 1e-8);
    for i in 0..2 {
        assert!((swap.forces_n[i] - reference.forces_n[1 - i]).length() < 1e-6);
        assert!((swap.torques_nm[i] - reference.torques_nm[1 - i]).length() < 1e-6);
    }
    let scaled = sample(
        &PressureBody::new(a.solid.clone(), 6e5).unwrap(),
        &PressureBody::new(b.solid.clone(), 9e5).unwrap(),
        m,
        false,
    );
    assert!((scaled.elastic_energy_j - 3. * reference.elastic_energy_j).abs() < 1e-7);
    for i in 0..2 {
        assert!((scaled.forces_n[i] - 3. * reference.forces_n[i]).length() < 1e-5);
    }
}
#[test]
fn pressure_wrenches_are_objective_under_common_pose_and_motion() {
    let a = pressure(DVec3::new(0.6, 0.5, 0.7), 2e5);
    let b = pressure(DVec3::new(1., 0.4, 0.9), 3e5);
    let m = motions();
    let reference = sample(&a, &b, m, false);
    let q = DQuat::from_axis_angle(DVec3::new(0.2, -0.3, 0.5).normalize(), 0.8);
    let translation = DVec3::new(14., -12., 32.);
    let drift = DVec3::new(0.4, 0.3, 0.2);
    let spin = DVec3::new(-0.2, 0.3, 0.1);
    let transformed = m.map(|body| {
        let center = q * body.center_of_mass_m + translation;
        RigidMotion {
            center_of_mass_m: center,
            rotation: q * body.rotation,
            linear_velocity_m_s: q * body.linear_velocity_m_s + drift + spin.cross(center),
            angular_velocity_rad_s: q * body.angular_velocity_rad_s + spin,
        }
    });
    let result = sample(&a, &b, transformed, false);
    balance(&result);
    assert!((result.elastic_energy_j - reference.elastic_energy_j).abs() < 1e-7);
    for i in 0..2 {
        assert!((result.forces_n[i] - q * reference.forces_n[i]).length() < 1e-5);
        assert!((result.torques_nm[i] - q * reference.torques_nm[i]).length() < 1e-5);
    }
}
#[test]
fn thinning_finite_overlap_has_no_premature_force_or_volume_cutoff() {
    let a = pressure(DVec3::new(2., 1., 2.), 2e6);
    let b = solid(DVec3::new(0.2, 0.5, 0.3));
    let area = 4. * (0.2f32 as f64) * (0.3f32 as f64);
    for requested in [1e-3, 1e-5, 1e-7, 1e-9, 1e-10] {
        let center = -1.5 + requested;
        let represented = center + 1.5;
        let r = evaluate(
            &a,
            RigidMotion::default(),
            Opposing::Rigid(&b),
            RigidMotion {
                center_of_mass_m: DVec3::Y * center,
                ..Default::default()
            },
        )
        .unwrap();
        let relative = (r.forces_n[0].y / (2e6 * area * represented) - 1.).abs();
        assert!(
            r.overlap_volume_m3 > 0. && r.elastic_energy_j > 0. && relative < 0.00001,
            "depth={requested} force={}N relative={relative}",
            r.forces_n[0].y
        );
    }
}
#[test]
fn increasing_opposing_pressure_approaches_the_finite_rigid_limit() {
    let a = pressure(DVec3::new(2., 1., 2.), 2e6);
    let solid = solid(DVec3::new(0.2, 0.5, 0.3));
    let motion = RigidMotion {
        center_of_mass_m: DVec3::Y * (-1.49),
        ..Default::default()
    };
    let rigid = evaluate(&a, RigidMotion::default(), Opposing::Rigid(&solid), motion)
        .unwrap()
        .forces_n[0]
        .y;
    let mut errors = Vec::new();
    for scale in [1e6, 1e8, 1e10] {
        let body = PressureBody::new(solid.clone(), scale).unwrap();
        let r = evaluate(
            &a,
            RigidMotion::default(),
            Opposing::Compliant(&body),
            motion,
        )
        .unwrap();
        balance(&r);
        assert!(r.forces_n[0].y > 0. && r.forces_n[0].y < rigid);
        errors.push((rigid - r.forces_n[0].y) / rigid);
    }
    assert!(
        errors[1] < 0.03 * errors[0] && errors[2] < 0.03 * errors[1],
        "soft-to-rigid convergence {errors:?}"
    );
    println!("FINITE_RIGID_LIMIT relative_force_errors={errors:?}");
}
#[test]
fn every_prepared_clod_preserves_its_full_registered_boundary_and_custody() {
    let mut session = crate::Session::new();
    session.prepare_soil_trial().unwrap();
    let soil = session.active_soil.as_mut().unwrap();
    let before = serde_json::to_string(soil).unwrap();
    let mut world = r::PhysicsWorld::new();
    let mut vertices = 0;
    let mut faces = 0;
    let mut tets = 0;
    assert_eq!(soil.clods.len(), 1040);
    for clod in &mut soil.clods {
        let handle = clod.insert_precise(&mut world).unwrap();
        let collider = &world.colliders[world.bodies[handle].colliders()[0]];
        let hull = collider.shape().as_convex_polyhedron().unwrap();
        let geometry = RegisteredSolid::from_hull(hull).unwrap();
        assert_eq!(geometry.vertices().len(), hull.points().len());
        assert_eq!(geometry.faces().len(), hull.faces().len());
        for (p, source) in geometry.vertices().iter().zip(hull.points()) {
            assert_eq!(*p, DVec3::from_array(source.to_array().map(f64::from)));
        }
        let independent = geometry
            .faces()
            .iter()
            .map(|f| {
                (1..f.len() - 1)
                    .map(|i| {
                        geometry.points[f[0]]
                            .dot(geometry.points[f[i]].cross(geometry.points[f[i + 1]]))
                            / 6.
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        assert!((geometry.volume_m3() / independent - 1.).abs() < 1e-11);
        vertices += geometry.vertices().len();
        faces += geometry.faces().len();
        tets += geometry.tetrahedra();
        let body = PressureBody::new(geometry, 2e5).unwrap();
        assert_eq!(body.tetrahedra(), body.solid().tetrahedra());
    }
    assert_eq!(
        before,
        serde_json::to_string(soil).unwrap(),
        "pressure construction changed finite custody or motion"
    );
    println!("FINITE_REGISTERED bank_clods=1040 vertices={vertices} faces={faces} pressure_tets={tets}; custody unchanged");
}

#[test]
fn captured_frames_preserve_supplied_coefficients_without_quaternion_normalization() {
    let source = solid(DVec3::new(0.6, 0.5, 0.7));
    let rotation = DQuat::from_xyzw(0., 0., 0.1, 0.995);
    let translation = DVec3::new(0.3, -0.2, 0.1);
    let captured = source.captured_frame(translation, rotation).unwrap();
    assert_eq!(source.faces(), captured.faces());
    assert_eq!(
        source.reference_tetrahedra().len(),
        captured.reference_tetrahedra().len()
    );
    let mut normalized_difference = 0f64;
    for (a, b) in source.vertices().iter().zip(captured.vertices()) {
        assert_eq!(rotation * *a + translation, *b);
        normalized_difference =
            normalized_difference.max((rotation.normalize() * *a + translation - *b).length());
    }
    assert!(
        normalized_difference > 1e-7,
        "fixture must detect silent normalization"
    );
}
#[test]
fn coincident_pressure_regions_use_the_actual_one_sided_material_rate() {
    let a = pressure(DVec3::ONE, 1e5);
    let b = pressure(DVec3::ONE, 1e5);
    let am = RigidMotion {
        center_of_mass_m: DVec3::Y * 1.9,
        linear_velocity_m_s: DVec3::new(0.2, -0.1, 0.05),
        ..Default::default()
    };
    let bm = RigidMotion::default();
    let response = evaluate(&a, am, Opposing::Compliant(&b), bm).unwrap();
    balance(&response);
    assert!(
        response.directional_tie_volume_m3 > 0.,
        "fixture needs positive-volume affine pressure ties"
    );
    let mut errors = Vec::new();
    for h in [1e-4, 1e-5, 1e-6] {
        let shifted = RigidMotion {
            center_of_mass_m: am.center_of_mass_m + am.linear_velocity_m_s * h,
            ..am
        };
        let after = evaluate(&a, shifted, Opposing::Compliant(&b), bm).unwrap();
        errors.push(
            ((after.elastic_energy_j - response.elastic_energy_j) / h + response.body_power_w)
                .abs(),
        );
    }
    println!(
        "FINITE_TIE volume={}m3 one_sided_rate_errors={errors:?}W",
        response.directional_tie_volume_m3
    );
    assert!(
        errors[1] < 0.2 * errors[0] && errors[2] < 0.2 * errors[1],
        "pressure tie lost the directional potential derivative"
    );
}

#[test]
fn rigid_compound_integrates_overlapping_parts_once_and_retains_finite_gaps() {
    let a = pressure(DVec3::new(2., 1., 2.), 2e6);
    let am = RigidMotion::default();
    let bm = RigidMotion {
        center_of_mass_m: DVec3::new(0.3, -1.49, 0.2),
        ..Default::default()
    };
    let part = solid(DVec3::new(0.25, 0.5, 0.25));
    let parts = [
        part.captured_frame(DVec3::X * -0.125, DQuat::IDENTITY)
            .unwrap(),
        part.captured_frame(DVec3::X * 0.125, DQuat::IDENTITY)
            .unwrap(),
    ];
    let union = evaluate(&a, am, Opposing::RigidCompound(&parts), bm).unwrap();
    let whole = solid(DVec3::new(0.375, 0.5, 0.25));
    let expected = evaluate(&a, am, Opposing::Rigid(&whole), bm).unwrap();
    balance(&union);
    assert!((union.elastic_energy_j - expected.elastic_energy_j).abs() < 1e-9);
    for i in 0..2 {
        assert!((union.forces_n[i] - expected.forces_n[i]).length() < 1e-7);
        assert!((union.torques_nm[i] - expected.torques_nm[i]).length() < 1e-7);
    }
    assert!((union.overlap_volume_m3 - expected.overlap_volume_m3).abs() < 1e-12);
    let naive: f64 = parts
        .iter()
        .map(|b| {
            evaluate(&a, am, Opposing::Rigid(b), bm)
                .unwrap()
                .elastic_energy_j
        })
        .sum();
    assert!(
        (naive / union.elastic_energy_j - 4. / 3.).abs() < 1e-10,
        "overlap negative control lost its double count"
    );
    let duplicate = evaluate(&a, am, Opposing::RigidCompound(&[whole.clone(), whole]), bm).unwrap();
    assert!((duplicate.elastic_energy_j - expected.elastic_energy_j).abs() < 1e-9);
    println!(
        "FINITE_RIGID_UNION naive={naive}J union={}J reference={}J depth={} visits={}",
        union.elastic_energy_j,
        expected.elastic_energy_j,
        union.union_maximum_depth,
        union.union_intersection_visits
    );
}
#[test]
fn rigid_compound_leaves_a_real_hollow_cavity_open() {
    let a = pressure(DVec3::splat(0.05), 2e5);
    let part = solid(DVec3::new(0.1, 0.5, 0.25));
    let parts = [
        part.captured_frame(DVec3::X * -0.5, DQuat::IDENTITY)
            .unwrap(),
        part.captured_frame(DVec3::X * 0.5, DQuat::IDENTITY)
            .unwrap(),
    ];
    let am = RigidMotion {
        center_of_mass_m: DVec3::Y * 0.46,
        ..Default::default()
    };
    let r = evaluate(
        &a,
        am,
        Opposing::RigidCompound(&parts),
        RigidMotion::default(),
    )
    .unwrap();
    assert_eq!(r.elastic_energy_j, 0.);
    assert_eq!(r.overlap_volume_m3, 0.);
    assert_eq!(r.forces_n, [DVec3::ZERO; 2]);
    let filled = solid(DVec3::new(0.6, 0.5, 0.25));
    let wrong = evaluate(&a, am, Opposing::Rigid(&filled), RigidMotion::default()).unwrap();
    assert!(
        wrong.elastic_energy_j > 0. && wrong.forces_n[0].length() > 1.,
        "filled cavity negative control must repel empty air"
    );
    assert!(evaluate(&a, am, Opposing::RigidCompound(&[]), RigidMotion::default()).is_err());
}
#[test]
fn overlapping_rigid_union_wrenches_match_every_pose_energy_derivative() {
    let a = pressure(DVec3::new(0.6, 0.5, 0.7), 2e5);
    let part = solid(DVec3::new(0.6, 0.4, 0.9));
    let parts = [
        part.captured_frame(DVec3::X * -0.2, DQuat::IDENTITY)
            .unwrap(),
        part.captured_frame(DVec3::X * 0.2, DQuat::from_rotation_y(0.08))
            .unwrap(),
    ];
    let m = motions();
    let sample =
        |m: [RigidMotion; 2]| evaluate(&a, m[0], Opposing::RigidCompound(&parts), m[1]).unwrap();
    let reference = sample(m);
    balance(&reference);
    let mut worst = 0_f64;
    for body in 0..2 {
        for dof in 0..6 {
            let axis = [DVec3::X, DVec3::Y, DVec3::Z][dof % 3];
            let h = 1e-6;
            let mut plus = m;
            let mut minus = m;
            if dof < 3 {
                plus[body].center_of_mass_m += axis * h;
                minus[body].center_of_mass_m -= axis * h;
            } else {
                plus[body].rotation = DQuat::from_axis_angle(axis, h) * m[body].rotation;
                minus[body].rotation = DQuat::from_axis_angle(axis, -h) * m[body].rotation;
            }
            let derivative =
                (sample(plus).elastic_energy_j - sample(minus).elastic_energy_j) / (2. * h);
            let wrench = if dof < 3 {
                reference.forces_n[body]
            } else {
                reference.torques_nm[body]
            }
            .dot(axis);
            let error = (derivative + wrench).abs() / wrench.abs().max(derivative.abs()).max(1.);
            worst = worst.max(error);
            assert!(
                error < 1e-5,
                "union pose derivative body={body} dof={dof}: {derivative} vs {} error={error}",
                -wrench
            );
        }
    }
    println!("FINITE_RIGID_UNION_GRADIENT worst_relative_error={worst}");
}

#[test]
fn sliding_off_a_finite_edge_releases_the_actual_stored_energy_as_work() {
    let a = pressure(DVec3::new(2., 1., 2.), 2e6);
    let b = solid(DVec3::new(0.25, 0.5, 0.25));
    let depth = 1_f64 / 64.;
    let ky = 2e6;
    let kx = 1e6;
    let band = ky / kx * depth;
    let density = |s: f64| {
        let s = s.max(0.);
        if s < band {
            kx * s * depth - kx * kx * s * s / (2. * ky)
        } else {
            0.5 * ky * depth * depth
        }
    };
    let primitive = |s: f64| {
        let s = s.max(0.);
        if s < band {
            0.5 * kx * depth * s * s - kx * kx * s * s * s / (6. * ky)
        } else {
            kx * depth * band * band * 0.5 - kx * kx * band * band * band / (6. * ky)
                + 0.5 * ky * depth * depth * (s - band)
        }
    };
    let normal_primitive = |s: f64| {
        let s = s.max(0.);
        if s < band {
            0.5 * kx * s * s
        } else {
            0.5 * kx * band * band + ky * depth * (s - band)
        }
    };
    let steps = 512_usize;
    let h = 1. / steps as f64;
    let mut work = 0.;
    let mut rows = Vec::new();
    let mut worst_energy = 0_f64;
    let mut worst_force = 0_f64;
    for i in 0..=steps {
        let x = 1.5 + i as f64 * h;
        let bm = RigidMotion {
            center_of_mass_m: DVec3::new(x, -1.5 + depth, 0.),
            ..Default::default()
        };
        let r = evaluate(&a, RigidMotion::default(), Opposing::Rigid(&b), bm).unwrap();
        let (left, right) = (2. - (x - 0.25), 2. - (x + 0.25));
        let expected_u = 0.5 * (primitive(left) - primitive(right));
        let expected_fx = 0.5 * (density(left) - density(right));
        let expected_fy = 0.5 * (normal_primitive(left) - normal_primitive(right));
        worst_energy = worst_energy.max((r.elastic_energy_j - expected_u).abs());
        worst_force = worst_force
            .max((r.forces_n[1].x - expected_fx).abs())
            .max((r.forces_n[0].y - expected_fy).abs());
        let weight = if i == 0 || i == steps {
            1.
        } else if i.is_multiple_of(2) {
            2.
        } else {
            4.
        };
        work += weight * r.forces_n[1].x * h / 3.;
        rows.push(serde_json::json!({"punch_center_x_m":x,"energy_j":r.elastic_energy_j,"normal_force_n":r.forces_n[0].y,"edge_force_n":r.forces_n[1].x,"analytic_energy_j":expected_u,"analytic_normal_force_n":expected_fy,"analytic_edge_force_n":expected_fx}));
    }
    let initial = 0.5 * ky * depth * depth * 0.25;
    let error = (work - initial).abs();
    assert!(
        worst_energy < 1e-8 && worst_force < 1e-6,
        "finite edge differs from independent field integral: {worst_energy}J {worst_force}N"
    );
    assert!(
        error < 1e-6,
        "finite boundary discarded or invented stored work: initial={initial}J work={work}J"
    );
    let report = serde_json::json!({"source":"manufactured held-depth finite punch; prescribed translation, no motion integrator or game step","rows":rows,"initial_storage_j":initial,"integrated_edge_work_j":work,"edge_work_error_j":error,"maximum_energy_error_j":worst_energy,"maximum_force_error_n":worst_force,"simpson_intervals":steps});
    if let Some(path) = std::env::var_os("SLUICE_FINITE_EDGE_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    println!("FINITE_EDGE initial={initial}J integrated_work={work}J work_error={error}J analytic_energy_error={worst_energy}J analytic_force_error={worst_force}N");
}
