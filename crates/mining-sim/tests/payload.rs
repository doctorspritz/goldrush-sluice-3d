use glam::{DMat3, DQuat, DVec3, Quat};
use mining_sim::{
    articulation::{self, Environment},
    payload::{MassMoments, RigidParcel},
    Excavator, MachineInput, Material, Session, Site, Vec3,
};
fn material(v: f64) -> Material {
    Material {
        gravel_m3: v,
        gold_g: v * 8.,
        pore_water_m3: v * 0.1,
        ..Default::default()
    }
}
fn machine(load: f64, moving: bool) -> Excavator {
    let mut m = Excavator::new(&Site::new());
    m.position = Vec3::new(16., 10., 16.);
    m.heading = 0.4;
    m.swing = 0.3;
    m.boom = 0.9;
    m.stick = -1.3;
    m.curl = -0.2;
    m.tracks.tilt = Quat::from_rotation_x(0.11) * Quat::from_rotation_z(-0.08);
    m.bucket = material(load);
    m.enable_articulation().unwrap();
    let s = m.articulation.as_mut().unwrap();
    s.stiffness_nm_rad = [0.; 4];
    s.damping_nms_rad = [0.; 4];
    if moving {
        s.velocity = [0.2, -0.3, 0.7, 0.06, -0.05, 0.08, 0.1, -0.1, 0.2, -0.3];
        m.tracks.angular_velocity = Vec3::new(0.06, -0.05, 0.08);
    }
    articulation::step(
        &mut m,
        MachineInput::default(),
        1. / 240.,
        &Environment::free(Vec3::ZERO),
    )
    .unwrap();
    m
}
fn point_motion(m: &Excavator, p: DVec3) -> (DVec3, DVec3) {
    let v = m.articulation.as_ref().unwrap().velocity;
    let root = m.position.as_dvec3() + m.up().as_dvec3() * ((2600. * 0.325 + 700. * 0.70) / 3300.);
    let root_w = DVec3::new(v[3] as f64, v[4] as f64, v[5] as f64);
    let vertical = m.up().as_dvec3();
    let x = (m.upper_rotation() * Vec3::X).as_dvec3();
    let arm = m.arm();
    let mut out = DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64) + root_w.cross(p - root);
    for (rate, axis, anchor) in [
        (v[6], vertical, m.position.as_dvec3() + vertical * 0.95),
        (v[7], x, arm.shoulder.as_dvec3()),
        (v[8], x, arm.elbow.as_dvec3()),
        (v[9], x, arm.wrist.as_dvec3()),
    ] {
        out += axis.cross(p - anchor) * rate as f64;
    }
    (
        out,
        root_w + vertical * v[6] as f64 + x * (v[7] + v[8] + v[9]) as f64,
    )
}
fn incoming(m: &Excavator, mat: Material, moving: bool) -> RigidParcel {
    let (w, q) = articulation::payload_frame(m);
    let center = w + q * DVec3::new(0., -0.05, 0.45);
    let (mut v, mut omega) = point_motion(m, center);
    if moving {
        v += DVec3::new(1.2, -0.6, 0.8);
        omega += DVec3::new(0.3, -0.2, 0.1);
    }
    let half = DVec3::new(0.2, mat.volume() / (8. * 0.2 * 0.25), 0.25);
    RigidParcel::cuboid(mat, center, half, q, v, omega).unwrap()
}
fn state(m: &Excavator) -> String {
    serde_json::to_string(m).unwrap()
}
fn assert_momenta(m: &Excavator, expected_p: DVec3, expected_l: DVec3) {
    let f = &m.articulation.as_ref().unwrap().feedback;
    let p = f.linear_momentum_ns.as_dvec3();
    let l = f.angular_momentum_nms.as_dvec3();
    eprintln!(
        "momentum errors P={} L={}",
        (p - expected_p).length(),
        (l - expected_l).length()
    );
    assert!((p - expected_p).length() < 0.01, "{p:?} != {expected_p:?}");
    assert!((l - expected_l).length() < 0.15, "{l:?} != {expected_l:?}");
}
#[test]
fn spatial_moments_match_independent_gauss_volume_integration() {
    let boxes = [
        (
            3.,
            DVec3::new(-0.3, 0.2, 0.5),
            DVec3::new(0.2, 0.1, 0.4),
            DQuat::from_rotation_y(0.4),
        ),
        (
            7.,
            DVec3::new(0.2, -0.1, 0.3),
            DVec3::new(0.1, 0.3, 0.2),
            DQuat::from_rotation_x(-0.5),
        ),
    ];
    let mut points = Vec::new();
    for (mass, c, h, q) in boxes {
        for x in [-1., 1.] {
            for y in [-1., 1.] {
                for z in [-1., 1.] {
                    points.push((mass / 8., c + q * (DVec3::new(x, y, z) * h / 3f64.sqrt())));
                }
            }
        }
    }
    let center = points.iter().map(|(w, p)| *p * *w).sum::<DVec3>() / 10.;
    let mut inertia = DMat3::ZERO;
    for (w, p) in points {
        let r = p - center;
        let cross = DMat3::from_cols(DVec3::X.cross(r), DVec3::Y.cross(r), DVec3::Z.cross(r));
        inertia += cross.transpose() * cross * w;
    }
    let a = MassMoments::cuboid(boxes[0].0, boxes[0].1, boxes[0].2, boxes[0].3).combine(
        MassMoments::cuboid(boxes[1].0, boxes[1].1, boxes[1].2, boxes[1].3),
    );
    assert!(a.valid());
    assert!((a.center - center).length() < 1e-14);
    assert!((a.inertia() - inertia)
        .to_cols_array()
        .iter()
        .all(|x| x.abs() < 1e-13));
}
#[test]
fn stationary_capture_has_closed_linear_angular_and_passive_energy_budgets() {
    let mut m = machine(0., false);
    let parcel = incoming(&m, material(0.04), true);
    let p = parcel.state.linear_momentum();
    let l = parcel.state.angular_momentum();
    let mut source = Some(parcel);
    let report = articulation::capture_payload(&mut m, &mut source, 1.).unwrap();
    eprintln!("capture {report:?}");
    assert!(source.is_none());
    assert_momenta(&m, p, l);
    assert!(report.inelastic_heat_j > 0.);
    assert!(report.kinetic_residual_j.abs() < 0.001);
    assert!(report.kinetic_after_j <= report.kinetic_before_j + 0.001);
    assert_eq!(m.bucket, material(0.04));
    assert!(m.valid());
}
#[test]
fn co_moving_capture_preserves_joint_rates_without_invented_impact_heat() {
    let mut m = machine(0.03, true);
    let velocity = m.articulation.as_ref().unwrap().velocity;
    let parcel = incoming(&m, material(0.02), false);
    let mut source = Some(parcel);
    let report = articulation::capture_payload(&mut m, &mut source, 1.).unwrap();
    eprintln!("co-moving {report:?}");
    assert!(report.inelastic_heat_j < 1e-6);
    assert!(m
        .articulation
        .as_ref()
        .unwrap()
        .velocity
        .iter()
        .zip(velocity)
        .all(|(a, b)| (*a - b).abs() < 1e-5));
}
#[test]
fn partial_capture_retains_finite_source_and_all_material_channels() {
    let mut m = machine(0.02, true);
    let original = incoming(&m, material(0.04), true);
    let mut source = Some(original.clone());
    let p = m
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .linear_momentum_ns
        .as_dvec3()
        + original.state.linear_momentum();
    let l = m
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .angular_momentum_nms
        .as_dvec3()
        + original.state.angular_momentum();
    articulation::capture_payload(&mut m, &mut source, 0.25).unwrap();
    let left = source.unwrap();
    assert_eq!(left.material, original.material.scaled(0.75));
    assert_eq!(left.state.velocity, original.state.velocity);
    assert!(left.valid());
    assert!(
        (m.bucket.wet_mass_kg() + left.material.wet_mass_kg() - material(0.06).wet_mass_kg()).abs()
            < 1e-10
    );
    assert_momenta(
        &m,
        p - left.state.linear_momentum(),
        l - left.state.angular_momentum(),
    );
}
#[test]
fn release_inherits_actual_point_velocity_and_spin_without_a_kick() {
    let mut m = machine(0.12, true);
    let before = m.clone();
    let observed = articulation::sample_payload(&m).unwrap();
    let (velocity, omega) = point_motion(&m, observed.state.center);
    assert!((velocity - observed.state.velocity).length() < 1e-6);
    assert!((omega - observed.state.angular_velocity()).length() < 1e-6);
    let p = before
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .linear_momentum_ns
        .as_dvec3();
    let l = before
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .angular_momentum_nms
        .as_dvec3();
    let parcel = articulation::release_payload(&mut m, 0.4).unwrap();
    let report = &m
        .articulation
        .as_ref()
        .unwrap()
        .exchange
        .as_ref()
        .unwrap()
        .last;
    eprintln!("release {report:?}");
    assert!((parcel.state.velocity - velocity).length() < 1e-6);
    assert!((parcel.state.angular_velocity() - omega).length() < 1e-6);
    assert_momenta(
        &m,
        p - parcel.state.linear_momentum(),
        l - parcel.state.angular_momentum(),
    );
    assert!(report.kinetic_residual_j.abs() < 0.005);
    assert_eq!(report.inelastic_heat_j, 0.);
    assert_eq!(
        m.articulation.as_ref().unwrap().pump_energy_j,
        before.articulation.as_ref().unwrap().pump_energy_j
    );
}
#[test]
fn repeated_uniform_split_and_recapture_preserve_stock_and_motion() {
    let mut m = machine(0., true);
    let mut source = Some(incoming(&m, material(0.04), false));
    articulation::capture_payload(&mut m, &mut source, 1.).unwrap();
    let initial = m.bucket;
    let p = m
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .linear_momentum_ns
        .as_dvec3();
    let l = m
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .angular_momentum_nms
        .as_dvec3();
    for _ in 0..100 {
        let parcel = articulation::release_payload(&mut m, 0.1).unwrap();
        let mut source = Some(parcel);
        articulation::capture_payload(&mut m, &mut source, 1.).unwrap();
        assert!(source.is_none());
    }
    assert!((m.bucket.volume() - initial.volume()).abs() < 1e-12);
    assert!((m.bucket.gold_g - initial.gold_g).abs() < 1e-12);
    assert!((m.bucket.pore_water_m3 - initial.pore_water_m3).abs() < 1e-12);
    assert_momenta(&m, p, l);
}
#[test]
fn capacity_outside_geometry_and_invalid_state_refusals_are_transactional() {
    let m = machine(0.27, false);
    let sources = [
        incoming(&m, material(0.03), false),
        {
            let mut p = incoming(&m, material(0.001), false);
            p.state.center += DVec3::Y * 2.;
            p
        },
        {
            let mut p = incoming(&m, material(0.001), false);
            p.state.moments.second.x_axis.x = -1.;
            p
        },
    ];
    for p in sources {
        let mut m = m.clone();
        let mut source = Some(p);
        let old_m = state(&m);
        let old_p = serde_json::to_string(&source).unwrap();
        assert!(articulation::capture_payload(&mut m, &mut source, 1.).is_err());
        assert_eq!(state(&m), old_m);
        assert_eq!(serde_json::to_string(&source).unwrap(), old_p);
    }
    for f in [-0.1, 0., 1.1, f64::NAN] {
        let mut m = m.clone();
        let before = state(&m);
        assert!(articulation::release_payload(&mut m, f).is_err());
        assert_eq!(state(&m), before);
    }
}
#[test]
fn exact_ballistic_parcel_motion_preserves_world_spin_and_gravity_energy() {
    let mut p = RigidParcel::cuboid(
        material(0.02),
        DVec3::new(16., 10., 16.),
        DVec3::splat((0.02f64 / 8.).cbrt()),
        DQuat::from_rotation_z(0.2),
        DVec3::new(0.4, 0.7, -0.2),
        DVec3::new(0.3, 0.7, 0.1),
    )
    .unwrap();
    let initial = p.clone();
    let energy = p.state.kinetic_j() + p.material.wet_mass_kg() * 9.81 * p.state.center.y;
    let dt = 1. / 120.;
    for _ in 0..60 {
        p.state.advance(DVec3::new(0., -9.81, 0.), dt).unwrap();
    }
    let t = 0.5;
    let exact =
        initial.state.center + initial.state.velocity * t + DVec3::Y * (-0.5 * 9.81 * t * t);
    assert!((p.state.center - exact).length() < 1e-12);
    assert!((p.state.velocity - (initial.state.velocity - DVec3::Y * (9.81 * t))).length() < 1e-12);
    assert_eq!(p.state.spin_nms, initial.state.spin_nms);
    assert!(
        (p.state.kinetic_j() + p.material.wet_mass_kg() * 9.81 * p.state.center.y - energy).abs()
            < 1e-8
    );
}
#[test]
fn asymmetric_torque_free_rotation_converges_to_a_finer_reference() {
    let make = || {
        RigidParcel::cuboid(
            material(0.02),
            DVec3::ZERO,
            DVec3::new(0.2, 0.1, 0.125),
            DQuat::IDENTITY,
            DVec3::ZERO,
            DVec3::new(0.5, 0.8, 1.1),
        )
        .unwrap()
    };
    let mut reference = make();
    for _ in 0..2048 {
        reference.state.advance(DVec3::ZERO, 0.5 / 2048.).unwrap();
    }
    let mut errors = Vec::new();
    let mut energy_errors = Vec::new();
    for n in [15, 30, 60] {
        let mut p = make();
        let e = p.state.kinetic_j();
        for _ in 0..n {
            p.state.advance(DVec3::ZERO, 0.5 / n as f64).unwrap();
        }
        let dot = p.state.rotation.dot(reference.state.rotation).abs().min(1.);
        errors.push(2. * dot.acos());
        energy_errors.push((p.state.kinetic_j() - e).abs());
        assert_eq!(p.state.spin_nms, reference.state.spin_nms);
    }
    eprintln!("free rotation orientation {errors:?}, energy {energy_errors:?}");
    assert!(errors[1] < errors[0] * 0.4 && errors[2] < errors[1] * 0.4);
    assert!(energy_errors[1] < energy_errors[0] * 0.4 && energy_errors[2] < energy_errors[1] * 0.4);
}
fn finite_session() -> Session {
    let mut s = Session::new();
    s.machine.enable_articulation().unwrap();
    let p = Vec3::new(19., 0., 20.);
    let m = s.site.dig(p, 0.4, 0.2, 0.04);
    assert!(m.volume() > 0.001);
    let parcel = incoming(&s.machine, m, true);
    s.falling.push(parcel.into_falling());
    s
}
#[test]
fn session_exchange_preserves_material_water_gold_and_exact_saved_continuation() {
    let mut s = finite_session();
    let material = s.accounted_material();
    let gold = s.accounted_gold();
    let water = s.stored_water_m3();
    s.capture_rigid_load(0, 0.6).unwrap();
    s.release_rigid_payload(0.3).unwrap();
    assert!((s.accounted_material().mass_kg() - material.mass_kg()).abs() < 1e-8);
    assert!((s.accounted_gold() - gold).abs() < 1e-9);
    assert!((s.stored_water_m3() - water).abs() < 1e-9);
    let json = s.save_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["version"], 10);
    let mut restored = Session::load_json(&json).unwrap();
    for _ in 0..30 {
        s.step(Default::default());
        restored.step(Default::default());
    }
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
}
#[test]
fn versioned_material_state_and_invalid_mass_moments_are_rejected() {
    let mut s = finite_session();
    s.capture_rigid_load(0, 0.5).unwrap();
    let value: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    for v in [6, 7] {
        let mut bad = value.clone();
        bad["version"] = v.into();
        assert!(Session::load_json(&bad.to_string()).is_err());
    }
    for field in ["mass_kg", "center", "second"] {
        let mut bad = value.clone();
        let slot = &mut bad["machine"]["articulation"]["payload_moments"][field];
        if field == "mass_kg" {
            *slot = (-1.).into();
        } else if field == "center" {
            *slot = serde_json::json!([99., 0., 0.]);
        } else {
            *slot = serde_json::json!([-1., 0., 0., 0., 0., 0., 0., 0., 0.]);
        }
        assert!(Session::load_json(&bad.to_string()).is_err(), "{field}");
    }
    let mut bad = value;
    bad["falling"][0]["position"][0] = 0.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut ordinary = Session::new();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&ordinary.save_json().unwrap()).unwrap()
            ["version"],
        6
    );
    ordinary.machine.enable_articulation().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&ordinary.save_json().unwrap()).unwrap()
            ["version"],
        10
    );
}
#[test]
fn rigid_parcel_handoff_records_static_boundary_momentum_without_losing_stock() {
    let mut s = finite_session();
    s.machine.position.y += 5.;
    let m = s.falling[0].material;
    let p = Vec3::new(24., s.site.height(24., 20.) + 0.1, 20.);
    let parcel = RigidParcel::cuboid(
        m,
        p.as_dvec3(),
        DVec3::splat((m.volume() / 8.).cbrt()),
        DQuat::IDENTITY,
        DVec3::new(0.1, -0.2, 0.2),
        DVec3::new(0., 0.3, 0.),
    )
    .unwrap();
    s.falling[0] = parcel.into_falling();
    let gold = s.accounted_gold();
    let water = s.stored_water_m3();
    let mass = s.accounted_dry_mass_kg();
    for _ in 0..30 {
        if s.falling.is_empty() {
            break;
        }
        s.step(Default::default());
    }
    assert!(s.falling.is_empty());
    let receipt = &s.material_motion.as_ref().unwrap().terrain;
    assert_eq!(receipt.parcels, 1);
    assert!(receipt.impulse_ns.y < 0. && receipt.received_kinetic_j > 0.);
    assert!(receipt.valid());
    assert!((s.accounted_dry_mass_kg() - mass).abs() < 1e-6);
    assert!((s.accounted_gold() - gold).abs() < 1e-9);
    assert!((s.stored_water_m3() - water).abs() < 1e-7);
    assert!(Session::load_json(&s.save_json().unwrap()).is_ok());
}

#[test]
fn rendered_ellipsoid_reproduces_the_spatial_second_moment() {
    let a = MassMoments::cuboid(
        3.,
        DVec3::new(-0.1, 0.3, 0.4),
        DVec3::new(0.1, 0.2, 0.3),
        DQuat::from_rotation_y(0.7),
    );
    let b = MassMoments::cuboid(
        7.,
        DVec3::new(0.4, 0.1, -0.2),
        DVec3::new(0.3, 0.1, 0.2),
        DQuat::from_rotation_x(0.4),
    );
    let moments = a.combine(b);
    let (q, e) = moments.visual_ellipsoid();
    let r = DMat3::from_quat(q);
    let reproduced = r * DMat3::from_diagonal(e * e * (moments.mass_kg / 5.)) * r.transpose();
    assert!((reproduced - moments.second)
        .to_cols_array()
        .iter()
        .all(|x| x.abs() < 1e-11));
    assert!((r.determinant() - 1.).abs() < 1e-12);
}
#[test]
fn pure_spin_capture_transfers_intrinsic_angular_momentum_without_linear_impulse() {
    let mut m = machine(0., false);
    let mut p = incoming(&m, material(0.03), false);
    p.state.velocity = DVec3::ZERO;
    p.state.spin_nms = p.state.world_inertia() * DVec3::new(2., 3., -1.);
    let spin = p.state.spin_nms;
    let mut source = Some(p);
    let report = articulation::capture_payload(&mut m, &mut source, 1.).unwrap();
    assert_momenta(&m, DVec3::ZERO, spin);
    assert!(report.inelastic_heat_j > 0.);
    assert!(report.kinetic_residual_j.abs() < 0.001);
}
#[test]
fn translating_the_experiment_preserves_capture_response_and_heat() {
    let mut near = machine(0., true);
    near.position = Vec3::new(1., 10., 1.);
    let mut far = near.clone();
    far.position += Vec3::new(15., 0., 15.);
    let mut near_source = Some(incoming(&near, material(0.04), true));
    let mut far_source = Some(incoming(&far, material(0.04), true));
    let a = articulation::capture_payload(&mut near, &mut near_source, 1.).unwrap();
    let b = articulation::capture_payload(&mut far, &mut far_source, 1.).unwrap();
    eprintln!(
        "frame translation heat delta={} rate delta={:?}",
        (a.inelastic_heat_j - b.inelastic_heat_j).abs(),
        near.articulation
            .as_ref()
            .unwrap()
            .velocity
            .iter()
            .zip(far.articulation.as_ref().unwrap().velocity)
            .map(|(x, y)| x - y)
            .collect::<Vec<_>>()
    );
    assert!((a.inelastic_heat_j - b.inelastic_heat_j).abs() < 0.001);
    assert!(near
        .articulation
        .as_ref()
        .unwrap()
        .velocity
        .iter()
        .zip(far.articulation.as_ref().unwrap().velocity)
        .all(|(x, y)| (*x - y).abs() < 1e-5));
    assert!(a.linear_residual_ns.length() < 0.002 && b.linear_residual_ns.length() < 0.002);
    assert!(a.angular_residual_nms.length() < 0.02 && b.angular_residual_nms.length() < 0.02);
}
#[test]
fn positive_subthreshold_inventory_is_not_dropped_by_exchange() {
    let mut m = machine(0., false);
    let mat = material(1e-14);
    assert!(mat.is_empty());
    let (w, q) = articulation::payload_frame(&m);
    let parcel = RigidParcel::cuboid(
        mat,
        w + q * DVec3::new(0., -0.05, 0.45),
        DVec3::splat((mat.volume() / 8.).cbrt()),
        q,
        DVec3::ZERO,
        DVec3::ZERO,
    )
    .unwrap();
    let mut source = Some(parcel);
    articulation::capture_payload(&mut m, &mut source, 1.).unwrap();
    assert_eq!(m.bucket, mat);
    assert!(source.is_none());
    let released = articulation::release_payload(&mut m, 1.).unwrap();
    assert_eq!(released.material, mat);
    assert_eq!(m.bucket.volume(), 0.);
}
#[test]
fn inconsistent_render_projection_and_invalid_free_motion_are_refused_without_mutation() {
    let mut s = finite_session();
    s.falling[0].position += Vec3::X;
    let before = s.save_json().unwrap();
    assert!(s.capture_rigid_load(0, 0.5).is_err());
    assert_eq!(s.save_json().unwrap(), before);
    let p = incoming(&machine(0., false), material(0.01), false);
    for dt in [-0.1, f64::NAN, 0.1] {
        let mut state = p.state.clone();
        assert!(state.advance(DVec3::ZERO, dt).is_err());
        assert_eq!(state, p.state);
    }
    let mut state = p.state.clone();
    state.advance(DVec3::ZERO, 0.).unwrap();
    assert_eq!(state, p.state);
}

#[test]
fn covariance_validation_is_relative_at_small_and_large_scales() {
    for scale in [1e-24, 1., 1e6] {
        let good = MassMoments {
            mass_kg: 1.,
            center: DVec3::ZERO,
            second: DMat3::from_diagonal(DVec3::new(1., 2., 3.) * scale),
        };
        assert!(good.valid());
        let bad = MassMoments {
            second: DMat3::from_diagonal(DVec3::new(-0.1, 1., 1.) * scale),
            ..good
        };
        assert!(!bad.valid());
    }
}
#[test]
fn a_parcel_with_impossible_bucket_spatial_moments_is_refused() {
    let mut m = machine(0., false);
    let mut parcel = incoming(&m, material(0.02), false);
    parcel.state.moments = MassMoments::cuboid(
        parcel.material.wet_mass_kg(),
        DVec3::ZERO,
        DVec3::new(1., 0.1, 1.),
        DQuat::IDENTITY,
    );
    assert!(parcel.valid());
    let mut source = Some(parcel);
    let before = state(&m);
    assert!(articulation::capture_payload(&mut m, &mut source, 1.).is_err());
    assert_eq!(state(&m), before);
    assert!(source.is_some());
}
