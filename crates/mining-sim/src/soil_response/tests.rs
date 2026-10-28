use super::*;
fn patch() -> SurfacePatch {
    SurfacePatch::new(&[
        DVec3::new(-0.5, 0., -1.),
        DVec3::new(-0.5, 0., 1.),
        DVec3::new(0.5, 0., 1.),
        DVec3::new(0.5, 0., -1.),
    ])
    .unwrap()
}
fn plane() -> Plane {
    Plane::new(DVec3::ZERO, DVec3::Y).unwrap()
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.),
        "{a} != {b}"
    );
}
fn close_v(a: DVec3, b: DVec3, tolerance: f64) {
    assert!(
        (a - b).length() <= tolerance * a.length().max(b.length()).max(1.),
        "{a:?} != {b:?}"
    );
}
fn fixture() -> (RigidMotion, RigidMotion) {
    (
        RigidMotion {
            center_of_mass_m: DVec3::new(0.2, -0.01, 0.1),
            rotation: DQuat::from_axis_angle(DVec3::new(0.2, 0.3, 1.).normalize(), 0.08),
            linear_velocity_m_s: DVec3::new(0.1, -0.03, 0.2),
            angular_velocity_rad_s: DVec3::new(0.2, -0.1, 1.2),
        },
        RigidMotion {
            center_of_mass_m: DVec3::new(0.1, 0.005, -0.2),
            rotation: DQuat::from_rotation_x(-0.04),
            linear_velocity_m_s: DVec3::new(-0.03, 0.01, -0.04),
            angular_velocity_rad_s: DVec3::new(-0.3, 0.1, -0.2),
        },
    )
}
#[test]
fn uniform_pressure_has_dimensional_stiffness_energy_and_load_support() {
    let layer = NormalLayer::new(2e6, 0.2, 100.).unwrap();
    let p = patch();
    close(layer.stiffness_density_n_m3(), 1e7, 1e-14);
    close(layer.viscosity_density_ns_m3(), 500., 1e-14);
    let a = RigidMotion {
        center_of_mass_m: -DVec3::Y * 0.003,
        linear_velocity_m_s: -DVec3::Y * 0.04,
        ..Default::default()
    };
    let r = layer
        .evaluate(&p, a, RigidMotion::default(), plane())
        .unwrap();
    close(p.area_m2(), 2., 1e-14);
    close(r.geometric_area_m2, 2., 1e-14);
    close(r.loaded_area_m2, 2., 1e-14);
    close_v(r.force_on_a_n, DVec3::Y * 60040., 1e-13);
    close_v(r.torque_on_a_nm, DVec3::ZERO, 1e-11);
    close(r.elastic_energy_j, 90., 1e-13);
    close(r.elastic_energy_rate_w, 2400., 1e-13);
    close(r.body_power_w, -2401.6, 1e-13);
    close(r.dissipation_w, 1.6, 1e-11);
    for load in [100., 1000., 10_000.] {
        let depth = load / (layer.stiffness_density_n_m3() * p.area_m2());
        let a = RigidMotion {
            center_of_mass_m: -DVec3::Y * depth,
            ..Default::default()
        };
        let response = layer
            .evaluate(&p, a, RigidMotion::default(), plane())
            .unwrap();
        close(response.force_on_a_n.y, load, 1e-13);
        close(response.elastic_energy_j, 0.5 * load * depth, 1e-13);
    }
}
#[test]
fn partially_compressed_rotating_patch_matches_independent_integrals() {
    let theta = 0.35f64;
    let layer = NormalLayer::new(2e6, 0.2, 0.).unwrap();
    let a = RigidMotion {
        rotation: DQuat::from_rotation_z(theta),
        ..Default::default()
    };
    let r = layer
        .evaluate(&patch(), a, RigidMotion::default(), plane())
        .unwrap();
    let k = 1e7;
    let half_x = 0.5f64;
    let half_z = 1.;
    let sin = theta.sin();
    let cos = theta.cos();
    close(r.geometric_area_m2, 1., 1e-13);
    close(r.force_on_a_n.y, k * sin * half_z * half_x.powi(2), 1e-13);
    close(
        r.elastic_energy_j,
        k * sin * sin * half_z * half_x.powi(3) / 3.,
        1e-13,
    );
    close(
        r.torque_on_a_nm.z,
        -k * sin * cos * 2. * half_z * half_x.powi(3) / 3.,
        1e-13,
    );
    close(r.dissipation_w, 0., 1e-13);
}
#[test]
fn both_body_wrenches_are_the_negative_energy_gradient() {
    let p = patch();
    let layer = NormalLayer::new(2e6, 0.2, 0.).unwrap();
    let (a, b) = fixture();
    let boundary = Plane::new(DVec3::new(0., 0.015, 0.), DVec3::Y).unwrap();
    let r = layer.evaluate(&p, a, b, boundary).unwrap();
    let epsilon = 1e-7;
    let mut worst = 0f64;
    for body in 0..2 {
        for angular in [false, true] {
            for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                let perturbed = |sign: f64| {
                    let mut aa = a;
                    let mut bb = b;
                    let target = if body == 0 { &mut aa } else { &mut bb };
                    if angular {
                        target.rotation =
                            DQuat::from_scaled_axis(axis * epsilon * sign) * target.rotation;
                    } else {
                        target.center_of_mass_m += axis * epsilon * sign;
                    }
                    layer
                        .evaluate(&p, aa, bb, boundary)
                        .unwrap()
                        .elastic_energy_j
                };
                let derivative = (perturbed(1.) - perturbed(-1.)) / (2. * epsilon);
                let wrench = match (body, angular) {
                    (0, false) => r.force_on_a_n,
                    (0, true) => r.torque_on_a_nm,
                    (1, false) => r.force_on_b_n,
                    _ => r.torque_on_b_nm,
                }
                .dot(axis);
                let error =
                    (derivative + wrench).abs() / derivative.abs().max(wrench.abs()).max(1.);
                worst = worst.max(error);
                assert!(
                    error < 2e-6,
                    "energy gradient body={body} angular={angular} axis={axis:?}: {error}"
                );
            }
        }
    }
    println!(
        "PATCH_GRADIENT 12 current-pose force/torque derivatives; worst_relative_error={worst}"
    );
}
#[test]
fn non_tensile_unloading_and_mixed_damping_close_mechanical_power() {
    let p = patch();
    let layer = NormalLayer::new(100_000., 0.1, 1000.).unwrap();
    let a = RigidMotion {
        center_of_mass_m: -DVec3::Y * 0.01,
        linear_velocity_m_s: DVec3::Y * 5.,
        ..Default::default()
    };
    let r = layer
        .evaluate(&p, a, RigidMotion::default(), plane())
        .unwrap();
    close_v(r.force_on_a_n, DVec3::ZERO, 1e-14);
    close(r.loaded_area_m2, 0., 1e-14);
    close(r.elastic_energy_rate_w, -100_000., 1e-13);
    close(r.dissipation_w, 100_000., 1e-13);
    assert!(
        (r.dissipation_w - 500_000.).abs() > 100_000.,
        "naive uncapped dashpot loss passed"
    );
    let mut count = 0;
    let mut worst = 0f64;
    for viscosity in [0., 500., 100_000.] {
        for speed in [-2., -0.03, 0., 0.03, 2.] {
            let layer = NormalLayer::new(2e6, 0.2, viscosity).unwrap();
            let (mut a, b) = fixture();
            a.linear_velocity_m_s.y = speed;
            let r = layer.evaluate(&p, a, b, plane()).unwrap();
            let scale =
                (r.body_power_w.abs() + r.elastic_energy_rate_w.abs() + r.dissipation_w.abs())
                    .max(1.);
            let error = (r.body_power_w + r.elastic_energy_rate_w + r.dissipation_w).abs() / scale;
            worst = worst.max(error);
            assert!(error < 1e-11);
            assert!(r.dissipation_w >= -1e-11 * scale);
            close(r.body_power_w, r.integrated_contact_power_w, 1e-11);
            count += 1;
        }
    }
    let open = layer
        .evaluate(
            &p,
            RigidMotion {
                center_of_mass_m: DVec3::Y * 0.01,
                linear_velocity_m_s: -DVec3::Y,
                ..Default::default()
            },
            RigidMotion::default(),
            plane(),
        )
        .unwrap();
    close(open.elastic_energy_j, 0., 1e-14);
    close_v(open.force_on_a_n, DVec3::ZERO, 1e-14);
    println!("PATCH_POWER {count} mixed compression/rotation/damping cases; worst_relative_closure={worst}");
}
#[test]
fn common_rigid_pose_and_motion_preserve_material_response() {
    let layer = NormalLayer::new(2e6, 0.2, 500.).unwrap();
    let p = patch();
    let (a, b) = fixture();
    let r = layer.evaluate(&p, a, b, plane()).unwrap();
    let q = DQuat::from_axis_angle(DVec3::new(1., 2., 3.).normalize(), 0.73);
    let shift = DVec3::new(50., -20., 30.);
    let boost = DVec3::new(2., -1., 3.);
    let spin = DVec3::new(0.5, -0.2, 0.3);
    let transform = |a: RigidMotion| {
        let position = q * a.center_of_mass_m + shift;
        RigidMotion {
            center_of_mass_m: position,
            rotation: q * a.rotation,
            linear_velocity_m_s: q * a.linear_velocity_m_s + boost + spin.cross(position),
            angular_velocity_rad_s: q * a.angular_velocity_rad_s + spin,
        }
    };
    let transformed = layer
        .evaluate(&p, transform(a), transform(b), plane())
        .unwrap();
    close(transformed.elastic_energy_j, r.elastic_energy_j, 1e-10);
    close(transformed.geometric_area_m2, r.geometric_area_m2, 1e-10);
    close(transformed.loaded_area_m2, r.loaded_area_m2, 1e-10);
    close(transformed.body_power_w, r.body_power_w, 1e-10);
    close(transformed.dissipation_w, r.dissipation_w, 1e-10);
    close_v(transformed.force_on_a_n, q * r.force_on_a_n, 1e-10);
    close_v(transformed.torque_on_a_nm, q * r.torque_on_a_nm, 1e-10);
    close_v(transformed.torque_on_b_nm, q * r.torque_on_b_nm, 1e-10);
    close_v(r.force_on_a_n + r.force_on_b_n, DVec3::ZERO, 1e-12);
    let moment = r.torque_on_a_nm
        + r.torque_on_b_nm
        + a.center_of_mass_m.cross(r.force_on_a_n)
        + b.center_of_mass_m.cross(r.force_on_b_n);
    close_v(moment, DVec3::ZERO, 1e-9);
}
#[test]
fn triangulation_and_extra_boundary_points_do_not_change_stiffness_or_strain() {
    let layer = NormalLayer::new(2e6, 0.2, 500.).unwrap();
    let p = patch();
    let (a, b) = fixture();
    let r = layer.evaluate(&p, a, b, plane()).unwrap();
    let mut worst = 0f64;
    for level in 0..=5 {
        let refined = p.refined(level).unwrap();
        let s = layer.evaluate(&refined, a, b, plane()).unwrap();
        for (a, b) in [
            (s.elastic_energy_j, r.elastic_energy_j),
            (s.body_power_w, r.body_power_w),
            (s.dissipation_w, r.dissipation_w),
        ] {
            worst = worst.max((a - b).abs() / a.abs().max(b.abs()).max(1.));
            close(a, b, 1e-10);
        }
        close_v(s.force_on_a_n, r.force_on_a_n, 1e-10);
        close_v(s.torque_on_a_nm, r.torque_on_a_nm, 1e-10);
        close(refined.area_m2(), p.area_m2(), 1e-14);
        assert_eq!(refined.triangles(), 2 * 4usize.pow(level));
    }
    let extra = SurfacePatch::new(&[
        DVec3::new(-0.5, 0., -1.),
        DVec3::new(-0.5, 0., 0.),
        DVec3::new(-0.5, 0., 1.),
        DVec3::new(0., 0., 1.),
        DVec3::new(0.5, 0., 1.),
        DVec3::new(0.5, 0., 0.),
        DVec3::new(0.5, 0., -1.),
        DVec3::new(0., 0., -1.),
    ])
    .unwrap();
    let e = layer.evaluate(&extra, a, b, plane()).unwrap();
    close(e.elastic_energy_j, r.elastic_energy_j, 1e-10);
    close_v(e.torque_on_a_nm, r.torque_on_a_nm, 1e-10);
    println!("PATCH_REFINEMENT 2..2048 triangles, extra boundary points; worst_relative_scalar_error={worst}");
}
#[test]
fn independent_corner_springs_are_a_negative_control_for_rotational_stiffness() {
    let theta = 0.01;
    let a = RigidMotion {
        center_of_mass_m: -DVec3::Y * 0.02,
        rotation: DQuat::from_rotation_z(theta),
        ..Default::default()
    };
    let p = patch();
    let layer = NormalLayer::new(2e6, 0.2, 0.).unwrap();
    let r = layer
        .evaluate(&p, a, RigidMotion::default(), plane())
        .unwrap();
    let corners = [
        DVec3::new(-0.5, 0., -1.),
        DVec3::new(-0.5, 0., 1.),
        DVec3::new(0.5, 0., 1.),
        DVec3::new(0.5, 0., -1.),
    ];
    let mut naive = DVec3::ZERO;
    for point in corners {
        let arm = a.rotation * point;
        let gap = (a.center_of_mass_m + arm).y;
        let force = -layer.stiffness_density_n_m3() * p.area_m2() / 4. * gap * DVec3::Y;
        naive += arm.cross(force);
    }
    close(naive.z, 3. * r.torque_on_a_nm.z, 1e-11);
    assert!((naive.z - r.torque_on_a_nm.z).abs() > 1000.);
    println!("PATCH_CORNER_CONTROL naive_rotational_stiffness/exact=3");
}
#[test]
fn invalid_geometry_material_and_motion_are_rejected_explicitly() {
    for (m, h, c) in [
        (0., 1., 0.),
        (-1., 1., 0.),
        (1., 0., 0.),
        (1., 1., -1.),
        (f64::NAN, 1., 0.),
        (1e308, 1e-308, 0.),
        (1e-308, 1e308, 0.),
        (1., 1e308, 1e-308),
    ] {
        assert!(NormalLayer::new(m, h, c).is_err());
    }
    assert!(SurfacePatch::new(&[DVec3::ZERO; 3]).is_err());
    assert!(
        SurfacePatch::new(&[DVec3::ZERO, DVec3::X, DVec3::new(0.2, 0., 0.2), DVec3::Z]).is_err()
    );
    assert!(
        SurfacePatch::new(&[DVec3::ZERO, DVec3::X, DVec3::new(1., 0.1, 1.), DVec3::Z]).is_err()
    );
    assert!(patch().refined(100).is_err());
    assert!(Plane::new(DVec3::ZERO, DVec3::ZERO).is_err());
    let layer = NormalLayer::new(1e6, 0.1, 0.).unwrap();
    let invalid = RigidMotion {
        rotation: DQuat::from_xyzw(0., 0., 0., 0.),
        ..Default::default()
    };
    assert!(layer
        .evaluate(&patch(), invalid, RigidMotion::default(), plane())
        .is_err());
    let invalid = RigidMotion {
        linear_velocity_m_s: DVec3::splat(f64::INFINITY),
        ..Default::default()
    };
    assert!(layer
        .evaluate(&patch(), invalid, RigidMotion::default(), plane())
        .is_err());
}

#[test]
fn zero_pressure_does_not_claim_a_loaded_material_area() {
    let layer = NormalLayer::new(128., 1., 1.).unwrap();
    let p = patch();
    let touching = layer
        .evaluate(&p, RigidMotion::default(), RigidMotion::default(), plane())
        .unwrap();
    assert_eq!(touching.loaded_area_m2, 0.);
    let a = RigidMotion {
        center_of_mass_m: -DVec3::Y / 128.,
        linear_velocity_m_s: DVec3::Y,
        ..Default::default()
    };
    let balanced = layer
        .evaluate(&p, a, RigidMotion::default(), plane())
        .unwrap();
    assert_eq!(balanced.loaded_area_m2, 0.);
    close_v(balanced.force_on_a_n, DVec3::ZERO, 1e-14);
    close(balanced.elastic_energy_rate_w, -2., 1e-14);
    close(balanced.dissipation_w, 2., 1e-14);
}

#[test]
fn duplicate_boundary_winding_cannot_duplicate_material_area() {
    let [a, b, c] = [DVec3::ZERO, DVec3::X, DVec3::Z];
    assert!(
        SurfacePatch::new(&[a, b, c, a, b, c]).is_err(),
        "duplicate winding was admitted as twice the material area"
    );
}
