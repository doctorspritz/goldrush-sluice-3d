use super::*;
use rapier3d::prelude as r;
use std::collections::BTreeMap;

fn field(n: usize, heights: Vec<f32>) -> HeightField {
    HeightField::with_flags(
        r::Array2::new(n, n, heights),
        r::Vector::new(2., 1., 2.),
        r::HeightFieldFlags::FIX_INTERNAL_EDGES,
    )
}
fn layer(n: usize) -> RegisteredTerrainLayer {
    RegisteredTerrainLayer::from_heightfields(
        &field(n, vec![1.; n * n]),
        &field(n, vec![0.; n * n]),
    )
    .unwrap()
}
fn solid(half: DVec3) -> RegisteredSolid {
    let points: Vec<_> = (0..8)
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
    let shape = r::SharedShape::convex_hull(&points).unwrap();
    RegisteredSolid::from_hull(shape.as_convex_polyhedron().unwrap()).unwrap()
}
fn close(a: &Response, b: &Response) {
    assert!((a.overlap_volume_m3 - b.overlap_volume_m3).abs() < 1e-12);
    assert!((a.elastic_energy_j - b.elastic_energy_j).abs() < 1e-7);
    for i in 0..2 {
        assert!((a.forces_n[i] - b.forces_n[i]).length() < 1e-6);
        assert!((a.torques_nm[i] - b.torques_nm[i]).length() < 1e-6);
    }
    assert!(a.force_residual_n.length() < 1e-8);
    assert!(a.moment_residual_nm.length() < 1e-8);
    assert!(a.power_residual_w.abs() < 1e-8);
}

#[test]
fn volume_matches_projected_triangle_integrals_through_zero_and_thin_columns() {
    for (heights, lower) in [
        (vec![1., 2., 3., 4.], vec![0.; 4]),
        (vec![0., 1., 0., 1.], vec![0.; 4]),
        (vec![2_f32.powi(-30); 4], vec![0.; 4]),
        (vec![0.; 4], vec![0.; 4]),
        (vec![0.3, 0.6, 0.5, 0.9], vec![0.2, 0.4, 0.1, 0.5]),
    ] {
        let top = field(2, heights);
        let bottom = field(2, lower);
        let layer = RegisteredTerrainLayer::from_heightfields(&top, &bottom).unwrap();
        let mut expected = 0.;
        for (id, triangle) in top.triangles().enumerate() {
            let p = triangle
                .vertices()
                .map(|p| DVec3::from_array(p.to_array().map(f64::from)));
            let area = ((p[1].x - p[0].x) * (p[2].z - p[0].z)
                - (p[1].z - p[0].z) * (p[2].x - p[0].x))
                .abs()
                * 0.5;
            // The two 2x2 triangles are yielded in their ID order here.
            let base_triangle = bottom.triangle_at_id(id as u32).unwrap();
            let base = base_triangle.vertices();
            expected += area
                * ((p[0].y + p[1].y + p[2].y) - base.iter().map(|p| f64::from(p.y)).sum::<f64>())
                / 3.;
        }
        assert!((layer.volume_m3() - expected).abs() <= expected.abs() * 2e-15);
        assert_eq!(layer.triangles(), 2);
        assert_eq!(layer.tetrahedra() == 0, expected == 0.);
        assert!(layer
            .reference_cells()
            .all(|(id, t)| id < 2 && volume(t) > 0.));
    }
}

#[test]
fn common_side_diagonals_are_closed_before_and_after_captured_frame_mapping() {
    for transformed in [false, true] {
        let mut layer = layer(3);
        if transformed {
            layer = layer
                .captured_frame(
                    DVec3::new(19.2, 0.3, -4.7),
                    DQuat::from_rotation_y(0.13) * DQuat::from_rotation_z(-0.07),
                )
                .unwrap();
        }
        let mut faces = BTreeMap::<[[u64; 3]; 3], usize>::new();
        for (_, tet) in layer.reference_cells() {
            for ids in geometry::FACES {
                let mut key = ids.map(|i| tet[i].to_array().map(f64::to_bits));
                key.sort();
                *faces.entry(key).or_default() += 1;
            }
        }
        assert!(faces.values().all(|n| *n == 1 || *n == 2));
        assert_eq!(faces.values().filter(|n| **n == 1).count(), 32);
    }
}

#[test]
fn layer_seams_edges_and_top_bottom_contacts_match_a_complete_finite_box() {
    let a = PressureBody::new(solid(DVec3::splat(0.2)), 200000.).unwrap();
    let b = solid(DVec3::new(1., 0.5, 1.))
        .captured_frame(DVec3::Y * 0.5, DQuat::IDENTITY)
        .unwrap();
    for n in [2, 3, 5] {
        let layer = layer(n);
        for center in [
            DVec3::new(0., 1.1, 0.),
            DVec3::new(0.95, 1.1, 0.13),
            DVec3::new(1.3, 1.1, 0.),
            DVec3::new(0., -0.3, 0.),
            DVec3::new(0.13, 1.4, 0.),
        ] {
            let motion = RigidMotion {
                center_of_mass_m: center,
                ..Default::default()
            };
            let actual = evaluate(
                &a,
                motion,
                Opposing::TerrainLayer(&layer),
                RigidMotion::default(),
            )
            .unwrap();
            let expected =
                evaluate(&a, motion, Opposing::Rigid(&b), RigidMotion::default()).unwrap();
            close(&actual, &expected);
            assert_eq!(actual.union_input_tetrahedra, layer.tetrahedra());
        }
    }
}

#[test]
fn moving_rotated_layer_wrenches_match_independent_pose_energy_derivatives() {
    let a = PressureBody::new(solid(DVec3::splat(0.13)), 200000.).unwrap();
    let layer = layer(2);
    let q = DQuat::from_rotation_y(0.11) * DQuat::from_rotation_z(-0.04);
    let bm = RigidMotion {
        center_of_mass_m: DVec3::new(3., 2., -4.),
        rotation: q,
        linear_velocity_m_s: DVec3::new(0.1, -0.05, 0.),
        angular_velocity_rad_s: DVec3::new(0.02, 0.03, -0.01),
    };
    let am = RigidMotion {
        center_of_mass_m: bm.center_of_mass_m + q * DVec3::new(0.17, 1.08, 0.11),
        rotation: q * DQuat::from_rotation_x(0.03),
        linear_velocity_m_s: DVec3::new(-0.03, -0.1, 0.01),
        angular_velocity_rad_s: DVec3::new(-0.02, 0.01, 0.03),
    };
    let evaluate_at =
        |m: [RigidMotion; 2]| evaluate(&a, m[0], Opposing::TerrainLayer(&layer), m[1]).unwrap();
    let m = [am, bm];
    let response = evaluate_at(m);
    assert!(response.overlap_volume_m3 > 0.);
    assert!(response.power_residual_w.abs() < 1e-8);
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
            let derivative = (evaluate_at(plus).elastic_energy_j
                - evaluate_at(minus).elastic_energy_j)
                / (2. * h);
            let wrench = if dof < 3 {
                response.forces_n[body]
            } else {
                response.torques_nm[body]
            }
            .dot(axis);
            assert!(
                (derivative + wrench).abs() / wrench.abs().max(derivative.abs()).max(1.) < 1e-5,
                "body{body} dof{dof} derivative{derivative} wrench{wrench}"
            );
        }
    }
}

#[test]
fn geometry_edits_change_only_the_explicit_envelope_and_invalid_bounds_are_refused() {
    let bottom = field(2, vec![0.; 4]);
    let upper = field(2, vec![1.; 4]);
    let lower = field(2, vec![0.5; 4]);
    let first = RegisteredTerrainLayer::from_heightfields(&upper, &bottom).unwrap();
    let edited = RegisteredTerrainLayer::from_heightfields(&lower, &bottom).unwrap();
    assert!((first.volume_m3() - 4.).abs() < 1e-14);
    assert!((edited.volume_m3() - 2.).abs() < 1e-14);
    let probe = PressureBody::new(solid(DVec3::splat(0.2)), 200000.).unwrap();
    let pose = RigidMotion {
        center_of_mass_m: DVec3::new(0., 1.1, 0.),
        ..Default::default()
    };
    let before = evaluate(
        &probe,
        pose,
        Opposing::TerrainLayer(&first),
        RigidMotion::default(),
    )
    .unwrap();
    let after = evaluate(
        &probe,
        pose,
        Opposing::TerrainLayer(&edited),
        RigidMotion::default(),
    )
    .unwrap();
    assert!(before.elastic_energy_j > 0.);
    assert_eq!(after.elastic_energy_j, 0.);
    assert_eq!(after.forces_n, [DVec3::ZERO; 2]);
    assert!(RegisteredTerrainLayer::from_heightfields(&bottom, &upper).is_err());
    assert!(RegisteredTerrainLayer::from_heightfields(&upper, &field(3, vec![0.; 9])).is_err());
    let excessive = field(183, vec![1.; 183 * 183]);
    let excessive_base = field(183, vec![0.; 183 * 183]);
    assert!(RegisteredTerrainLayer::from_heightfields(&excessive, &excessive_base).is_err());
    let a = PressureBody::new(solid(DVec3::splat(0.2)), 200000.).unwrap();
    let cells = super::super::cells(
        &a,
        RigidMotion {
            center_of_mass_m: DVec3::Y,
            ..Default::default()
        },
        DVec3::ZERO,
    )
    .unwrap();
    assert!(first
        .cells(&cells, RigidMotion::default(), DVec3::ZERO, 0)
        .is_err());
    assert!(evaluate(
        &a,
        RigidMotion {
            center_of_mass_m: DVec3::splat(-f64::MAX),
            ..Default::default()
        },
        Opposing::TerrainLayer(&first),
        RigidMotion {
            center_of_mass_m: DVec3::splat(f64::MAX),
            ..Default::default()
        },
    )
    .is_err());
    assert!(first
        .captured_frame(DVec3::ZERO, DQuat::from_xyzw(0., 0., 0., 2.))
        .is_err());
}
