use glam::Mat3;
use mining_sim::{
    outflow::{self, OutflowParticle},
    sluice::{HydraulicProfile, WaterParticle, WaterSampling},
    Placement, Session, Vec3,
};

#[test]
fn marker_sampling_changes_only_empty_custody_and_preserves_existing_credits() {
    let mut s = Session::new();
    s.sluice.water_credit_m3 = 0.000013;
    s.sluice.feed_water_credit_m3 = 0.000007;
    let before = s.sluice.accounted_water_m3();
    for sampling in [
        WaterSampling::Four,
        WaterSampling::Two,
        WaterSampling::Eight,
    ] {
        s.sluice.set_water_sampling(sampling).unwrap();
        assert_eq!(s.sluice.accounted_water_m3(), before);
    }
    s.sluice.water.push(WaterParticle {
        id: 23,
        position: Vec3::new(0.4, 0.2, 0.2),
        velocity: Vec3::X,
        affine: Mat3::IDENTITY,
    });
    let saved = s.save_json().unwrap();
    assert!(s.sluice.set_water_sampling(WaterSampling::Four).is_err());
    assert_eq!(s.save_json().unwrap(), saved);
    s.sluice.set_water_sampling(WaterSampling::Eight).unwrap();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.water[0].position.x = 1.51;
    outflow::release(&mut s.sluice);
    assert!(s.sluice.water.is_empty() && s.sluice.outflow.len() == 1);
    assert!(s.sluice.set_water_sampling(WaterSampling::Four).is_err());
}

#[test]
fn sampled_lip_flight_and_ground_receipt_keep_the_same_finite_volume_and_momentum() {
    for sampling in [
        WaterSampling::Eight,
        WaterSampling::Four,
        WaterSampling::Two,
    ] {
        let mut s = Session::new();
        s.sluice.set_water_sampling(sampling).unwrap();
        s.sluice.placement = Some(Placement::suggested(&s.site));
        let n = 4000 / (8 / sampling.markers_per_cell()) as usize;
        let one = s.sluice.water_particle_m3();
        s.sluice.water = (0..n)
            .map(|id| WaterParticle {
                id: id as u64 + 1,
                position: Vec3::new(1.51, 0.05, 0.2),
                velocity: Vec3::new(0.8, -2., -0.4),
                affine: Mat3::ZERO,
            })
            .collect();
        let before = s.sluice.accounted_water_m3();
        assert!((before - 4000. * s.sluice.profile.particle_m3()).abs() < 1e-15);
        outflow::release(&mut s.sluice);
        assert_eq!(s.sluice.outflow.len(), n);
        assert_eq!(s.sluice.accounted_water_m3(), before);
        let x = 14.625;
        let z = 20.625;
        for p in &mut s.sluice.outflow {
            p.position = Vec3::new(x, s.site.height(x, z) + 0.002, z);
            p.velocity = Vec3::new(0.8, -2., -0.4);
        }
        let stored = s.site.water_volume_m3();
        outflow::step(&mut s.sluice, &mut s.site, 1. / 60.);
        assert!(s.sluice.outflow.is_empty());
        assert!((s.sluice.water_returned_m3 - n as f64 * one).abs() < 1e-15);
        assert!((s.site.water_volume_m3() - stored - before).abs() < 1e-8);
        let (mut mx, mut mz) = (0., 0.);
        for i in 0..128 * 128 {
            let v = s.site.world.water_depth(i % 128, i / 128) as f64 * 0.0625;
            mx += v * s.site.hydraulic.velocity_x[i] as f64;
            mz += v * s.site.hydraulic.velocity_z[i] as f64;
        }
        assert!((mx - before * 0.8).abs() < 1e-8);
        assert!((mz + before * 0.4).abs() < 1e-8);
        assert!((s.sluice.accounted_water_m3() - before).abs() < 1e-15);
    }
}

#[test]
fn reduced_sampling_has_an_explicit_save_envelope_and_exact_owned_state_roundtrip() {
    for sampling in [WaterSampling::Four, WaterSampling::Two] {
        for mut s in [Session::new(), Session::new_operation()] {
            s.sluice.set_water_sampling(sampling).unwrap();
            s.sluice.water = vec![WaterParticle {
                id: 52,
                position: Vec3::new(0.6, 0.15, 0.2),
                velocity: Vec3::new(0.3, -0.1, 0.02),
                affine: Mat3::from_diagonal(Vec3::new(0.1, 0.2, 0.3)),
            }];
            s.sluice.outflow = vec![OutflowParticle {
                id: 53,
                position: Vec3::new(14., 2., 20.),
                velocity: Vec3::new(0.8, -0.4, 0.1),
            }];
            s.sluice.water_credit_m3 = 0.000013;
            s.sluice.feed_water_credit_m3 = 0.000007;
            let text = s.save_json().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(parsed["version"], 15);
            assert!(parsed.get("site").is_none()); // older fixed-volume Save refuses this envelope
            let loaded = Session::load_json(&text).unwrap();
            assert_eq!(loaded.sluice.water_sampling(), sampling);
            assert_eq!(
                loaded.sluice.accounted_water_m3(),
                s.sluice.accounted_water_m3()
            );
            assert_eq!(loaded.save_json().unwrap(), text);
            let stripped = parsed["shift"].to_string();
            assert!(Session::load_json(&stripped).is_err());
        }
    }
}

#[test]
fn reference_format_stays_unwrapped_and_wrong_sampling_formats_are_refused() {
    let s = Session::new();
    let mut value: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    assert_ne!(value["version"], 15);
    assert!(value["sluice"].get("water_sampling").is_none());
    assert_eq!(
        Session::load_json(&value.to_string())
            .unwrap()
            .sluice
            .water_sampling(),
        WaterSampling::Eight
    );
    assert!(
        Session::load_json(&serde_json::json!({"version":15,"shift":value}).to_string()).is_err()
    );
    value["sluice"]["water_sampling"] = serde_json::json!("Three");
    assert!(Session::load_json(&value.to_string()).is_err());
    value["sluice"]["water_sampling"] = serde_json::json!("Four");
    assert!(Session::load_json(&value.to_string()).is_err());
    assert!(WaterSampling::try_from(0).is_err());
    assert!(WaterSampling::try_from(3).is_err());
    let mut s = Session::new();
    s.sluice.profile = HydraulicProfile::Legacy;
    assert!(s.sluice.set_water_sampling(WaterSampling::Four).is_err());
    assert_eq!(s.sluice.water_sampling(), WaterSampling::Eight);
}
