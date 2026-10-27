use super::*;
use glam::DQuat;

fn source() -> WorldCell {
    WorldCell::new(
        [DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z],
        Some(Affine {
            origin: DVec3::ZERO,
            value: 1.,
            gradient: DVec3::Y,
            triangle: None,
            triangle_scale: 0.,
        }),
    )
    .unwrap()
}
fn opposing(apex: DVec3) -> WorldCell {
    // Swapped X/Y vertices retain positive orientation after reflection.
    WorldCell::new(
        [apex, apex - DVec3::Y, apex - DVec3::X, apex - DVec3::Z],
        None,
    )
    .unwrap()
}
fn prepared(b: &WorldCell) -> Cell {
    Cell {
        vertices: b.tet,
        planes: geometry::FACES.map(|f| Halfspace::face(f.map(|i| b.tet[i]))),
        quick_planes: b.planes,
        low: point(b.minimum),
        high: point(b.maximum),
    }
}

#[test]
fn exact_face_rejection_proves_separation_inside_overlapping_axis_bounds() {
    let a = source();
    let b = opposing(DVec3::splat(0.8));
    assert!(
        a.intersects(&b),
        "axis rejection alone must not prove this case"
    );
    assert!(face_separated(
        &ExactPoly::tet(a.tet),
        &a.tet,
        &prepared(&b)
    ));
    // The original rational clip independently proves the same zero volume.
    let mut clipped = ExactPoly::tet(a.tet);
    for plane in &prepared(&b).planes {
        clipped = clipped.clip(plane).unwrap();
    }
    assert!(clipped.faces.is_empty());
}

#[test]
fn positive_slivers_and_exact_touch_are_never_discarded_by_the_hint() {
    let a = source();
    for depth in [0., 2_f64.powi(-40), 0.1] {
        let b = opposing(DVec3::new(0.5, 0.5, 1. - depth));
        assert!(a.intersects(&b));
        assert!(!face_separated(
            &ExactPoly::tet(a.tet),
            &a.tet,
            &prepared(&b)
        ));
        if depth > 0. {
            let mut misleading = prepared(&b);
            misleading.quick_planes.fill(Affine {
                origin: DVec3::ZERO,
                value: 1.,
                gradient: DVec3::ZERO,
                triangle: None,
                triangle_scale: 0.,
            });
            // Even a deliberately false floating separation hint cannot
            // discard positive overlap without an exact sign proof.
            assert!(!face_separated(&ExactPoly::tet(a.tet), &a.tet, &misleading));
        }
        let r = union_impl(
            std::slice::from_ref(&a),
            &[b],
            RigidMotion::default(),
            RigidMotion::default(),
            DVec3::ZERO,
            true,
        )
        .unwrap();
        if depth > 0. {
            assert!(r.overlap_volume_m3 > 0.);
        } else {
            assert_eq!(r.overlap_volume_m3, 0.);
        }
    }
}

#[test]
fn fast_face_hints_match_complete_exact_union_through_rotation_and_overlapping_parts() {
    let motions = [
        RigidMotion {
            linear_velocity_m_s: DVec3::new(0.1, -0.2, 0.3),
            angular_velocity_rad_s: DVec3::new(0.2, 0.3, -0.1),
            ..Default::default()
        },
        RigidMotion {
            angular_velocity_rad_s: DVec3::new(-0.3, 0.1, 0.2),
            ..Default::default()
        },
    ];
    for angle in [0., 0.13, -0.37] {
        let rotation = DQuat::from_rotation_z(angle) * DQuat::from_rotation_y(angle * 0.7);
        let a = source();
        let field = a.pressure.unwrap();
        let a = WorldCell::new(
            a.tet.map(|p| rotation * p),
            Some(Affine {
                gradient: rotation * field.gradient,
                ..field
            }),
        )
        .unwrap();
        let b = [
            opposing(DVec3::splat(0.8)),
            opposing(DVec3::new(0.5, 0.5, 0.9)),
            opposing(DVec3::new(0.52, 0.51, 0.91)),
        ];
        let b: Vec<_> = b
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let mut cell = WorldCell::new(b.tet.map(|p| rotation * p), None).unwrap();
                cell.rigid_component = i;
                cell
            })
            .collect();
        let filtered = union_impl(
            std::slice::from_ref(&a),
            &b,
            motions[0],
            motions[1],
            DVec3::ZERO,
            true,
        )
        .unwrap();
        let reference = union_impl(&[a], &b, motions[0], motions[1], DVec3::ZERO, false).unwrap();
        assert!(reference.overlap_volume_m3 > 0.);
        // Debug retains every response field, full precision and signed zero.
        assert_eq!(format!("{filtered:?}"), format!("{reference:?}"));
    }
}

#[test]
fn source_face_certifies_zero_when_opposing_face_and_axis_bounds_cannot() {
    let a = source();
    let b = WorldCell::new(
        [
            DVec3::new(0., -0.125, 1.4375),
            DVec3::new(-0.125, -0.1875, 1.4375),
            DVec3::new(0.5, 0.6875, 0.25),
            DVec3::new(0.0625, -0.25, 1.5),
        ],
        None,
    )
    .unwrap();
    let poly = ExactPoly::tet(a.tet);
    let prepared = prepared(&b);
    assert!(a.intersects(&b));
    assert!(
        !face_separated(&poly, &a.tet, &prepared),
        "existing opposing-face path already rejected fixture"
    );
    assert!(reverse_face_separated(&poly, &a, &prepared));
    let fast = union_impl(
        std::slice::from_ref(&a),
        std::slice::from_ref(&b),
        RigidMotion::default(),
        RigidMotion::default(),
        DVec3::ZERO,
        true,
    )
    .unwrap();
    let full = union_impl(
        std::slice::from_ref(&a),
        std::slice::from_ref(&b),
        RigidMotion::default(),
        RigidMotion::default(),
        DVec3::ZERO,
        false,
    )
    .unwrap();
    assert_eq!(full.overlap_volume_m3, 0.);
    assert_eq!(format!("{fast:?}"), format!("{full:?}"));
    // Exercise this new rejection alongside positive overlapping parts. The
    // latter still require recursive inclusion-exclusion of constructed polys.
    let mut parts = [
        b,
        opposing(DVec3::new(0.5, 0.5, 0.9)),
        opposing(DVec3::new(0.52, 0.51, 0.91)),
    ];
    for (i, part) in parts.iter_mut().enumerate() {
        part.rigid_component = i;
    }
    let am = RigidMotion {
        linear_velocity_m_s: DVec3::new(0.1, -0.2, 0.3),
        angular_velocity_rad_s: DVec3::new(0.2, 0.3, -0.1),
        ..Default::default()
    };
    let bm = RigidMotion {
        angular_velocity_rad_s: DVec3::new(-0.3, 0.1, 0.2),
        ..Default::default()
    };
    let fast = union_impl(std::slice::from_ref(&a), &parts, am, bm, DVec3::ZERO, true).unwrap();
    let full = union_impl(&[a], &parts, am, bm, DVec3::ZERO, false).unwrap();
    assert!(full.overlap_volume_m3 > 0. && full.union_maximum_depth > 0);
    assert_eq!(format!("{fast:?}"), format!("{full:?}"));
}

#[test]
fn exact_projection_refuses_zero_invalid_and_misleading_axes_for_positive_slivers() {
    let a = source();
    let poly = ExactPoly::tet(a.tet);
    for depth in [2_f64.powi(-40), 0.1] {
        let b = opposing(DVec3::new(0.5, 0.5, 1. - depth));
        let points = b.tet.map(point);
        for axis in [
            DVec3::ZERO,
            DVec3::splat(f64::NAN),
            DVec3::splat(f64::INFINITY),
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::ONE,
            DVec3::new(1., -2., 3.),
            DVec3::splat(f64::from_bits(1)),
            DVec3::splat(f64::MAX),
        ] {
            assert!(!axis_separated(&poly.points, &points, axis));
        }
        assert!(!reverse_face_separated(&poly, &a, &prepared(&b)));
    }
}
