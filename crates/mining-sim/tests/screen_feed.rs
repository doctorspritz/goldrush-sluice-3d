use glam::{Mat3, Quat};
use mining_sim::{
    sluice::{Grain, ScreenFeed, SCREEN_FEED_BATCH_MAX},
    GrainKind, Material, Placement, Session, Sluice, Vec3, FIXED_DT,
};

fn source() -> Sluice {
    let s = Session::new();
    let placement = Placement::suggested(&s.site);
    let mut out = s.sluice;
    out.placement = Some(placement);
    out.pump_on = true;
    out.feed_on = true;
    out
}
fn stock(s: &Sluice, batch: &ScreenFeed) -> Material {
    let mut m = s.active_material();
    for g in &batch.grains {
        m.add(g.material);
    }
    m.pore_water_m3 += batch.moisture_m3;
    m
}
fn close(a: Material, b: Material) {
    for (x, y) in [
        (a.soil_m3, b.soil_m3),
        (a.sand_m3, b.sand_m3),
        (a.gravel_m3, b.gravel_m3),
        (a.gold_g, b.gold_g),
        (a.pore_water_m3, b.pore_water_m3),
    ] {
        assert!((x - y).abs() <= x.abs() * 1e-10 + 1e-30, "{x} != {y}");
    }
}
fn sentinel() -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Gold,
        position: Vec3::new(0.7, 0.14, 0.2),
        velocity: Vec3::new(0.2, -0.1, 0.3),
        material: Material {
            gold_g: 0.001,
            ..Default::default()
        },
        resting: false,
        diameter_m: 0.00015,
        affine: Mat3::from_diagonal(Vec3::new(0.1, 0.2, 0.3)),
        rotation: Quat::from_rotation_z(0.3),
        angular_velocity: Vec3::new(0.2, -0.3, 0.4),
    }
}

#[test]
fn screen_batch_keeps_existing_full_grains_and_water_buffers_exact() {
    let mut s = source();
    s.grains.push(sentinel());
    s.hopper = Material {
        soil_m3: 0.01,
        sand_m3: 0.02,
        gravel_m3: 0.03,
        gold_g: 0.1,
        pore_water_m3: 0.005,
    };
    s.feed_m3s = 0.02;
    s.feed_water_credit_m3 = 0.0001;
    s.water_credit_m3 = 0.0002;
    s.water_used_m3 = 4.;
    s.processed_m3 = 2.;
    let before = s.active_material();
    let existing = serde_json::to_value(&s.grains).unwrap();
    let batch = s.take_screen_feed(FIXED_DT, true).unwrap();
    assert!(!batch.grains.is_empty());
    assert!(batch.grains.len() <= SCREEN_FEED_BATCH_MAX);
    assert_eq!(serde_json::to_value(&s.grains).unwrap(), existing);
    assert_eq!(s.feed_water_credit_m3, 0.0001);
    assert_eq!(s.water_credit_m3, 0.0002);
    assert_eq!(s.water_used_m3, 4.);
    assert_eq!(s.processed_m3, 2.);
    assert_eq!(batch.moisture_m3, 0.);
    assert!(batch.grains.iter().all(|g| g.material.valid()));
    assert!(batch
        .grains
        .iter()
        .filter(|g| g.kind != GrainKind::Gold)
        .any(|g| g.material.pore_water_m3 > 0.));
    assert!(batch
        .grains
        .iter()
        .filter(|g| g.kind == GrainKind::Gold)
        .all(|g| g.material.pore_water_m3 == 0.));
    close(stock(&s, &batch), before);
}

#[test]
fn modern_dry_source_matches_legacy_feed_and_pending_drain_exactly() {
    let mut modern = source();
    modern.hopper = Material {
        soil_m3: 0.01,
        sand_m3: 0.02,
        gravel_m3: 0.03,
        gold_g: 0.001,
        ..Default::default()
    };
    let mut legacy = modern.clone();
    for _ in 0..20 {
        let batch = modern.take_screen_feed(FIXED_DT, true).unwrap();
        legacy.feed(FIXED_DT, true);
        legacy.drain_pending_gold();
        assert_eq!(
            serde_json::to_value(&batch.grains).unwrap(),
            serde_json::to_value(&legacy.grains).unwrap()
        );
        legacy.grains.clear();
        assert_eq!(modern.hopper, legacy.hopper);
        close(modern.active_material(), legacy.active_material());
    }
}

#[test]
fn pure_gold_emits_physical_grains_without_duplicate_stock_or_macro_parcels() {
    let mut s = source();
    s.hopper = Material {
        gold_g: 0.01,
        ..Default::default()
    };
    let initial = s.active_material();
    let mut emitted = Material::default();
    for _ in 0..20 {
        let batch = s.take_screen_feed(FIXED_DT, true).unwrap();
        for g in batch.grains {
            assert_eq!(g.kind, GrainKind::Gold);
            let one = std::f64::consts::PI / 6. * (g.diameter_m as f64).powi(3) * 19300. * 1000.;
            assert!((g.material.gold_g - one).abs() < one * 1e-12);
            emitted.add(g.material);
        }
        let mut total = s.active_material();
        total.add(emitted);
        close(total, initial);
    }
    assert!(emitted.gold_g > 0.);
    assert!(s.grains.is_empty());
}

#[test]
fn no_water_stopped_pump_and_stopped_feed_emit_nothing() {
    for mode in 0..3 {
        let mut s = source();
        s.hopper = Material {
            sand_m3: 0.01,
            gold_g: 0.001,
            ..Default::default()
        };
        s.feed_m3s = 0.02;
        if mode == 1 {
            s.pump_on = false;
        }
        if mode == 2 {
            s.feed_on = false;
        }
        let initial = s.active_material();
        let batch = s.take_screen_feed(FIXED_DT, mode != 0).unwrap();
        assert!(batch.grains.is_empty());
        assert_eq!(batch.moisture_m3, 0.);
        close(stock(&s, &batch), initial);
    }
}

#[test]
fn blocked_water_room_retains_wet_feed_at_source() {
    let mut s = source();
    s.hopper = Material {
        sand_m3: 0.001,
        pore_water_m3: 0.0002,
        ..Default::default()
    };
    s.feed_m3s = 0.02;
    s.water_credit_m3 = s.profile.capacity() as f64 * s.profile.particle_m3();
    let before = s.hopper;
    let batch = s.take_screen_feed(FIXED_DT, true).unwrap();
    assert!(batch.grains.is_empty());
    assert_eq!(s.hopper, before);
    assert_eq!(s.feed_water_credit_m3, 0.);
}

#[test]
fn stopping_feed_retains_pending_gold_when_hopper_is_full() {
    let mut s = source();
    s.feed_m3s = 0.02;
    s.hopper = Material {
        gold_g: 10.,
        ..Default::default()
    };
    let first = s.take_screen_feed(FIXED_DT, true).unwrap();
    assert!(!first.grains.is_empty());
    assert!(
        serde_json::to_value(&s).unwrap()["pending_gold_g"]
            .as_f64()
            .unwrap()
            > 9.
    );
    s.hopper = Material {
        sand_m3: 0.85,
        ..Default::default()
    };
    s.feed_on = false;
    let before = s.active_material();
    let batch = s.take_screen_feed(FIXED_DT, true).unwrap();
    assert!(batch.grains.is_empty());
    assert_eq!(s.hopper.handling_volume_m3(), 0.85);
    close(stock(&s, &batch), before);
}

#[test]
fn tiny_positive_phase_and_water_stocks_are_retained_or_emitted_without_tolerance_loss() {
    for amount in [1e-11, 1e-14, 1e-22] {
        let mut s = source();
        s.feed_m3s = 0.02;
        s.hopper = Material {
            soil_m3: amount,
            sand_m3: amount * 2.,
            gravel_m3: amount * 3.,
            gold_g: 1e-14,
            pore_water_m3: amount * 0.1,
        };
        let before = s.active_material();
        let batch = s.take_screen_feed(FIXED_DT, true).unwrap();
        assert!(
            batch
                .grains
                .iter()
                .filter(|g| g.kind != GrainKind::Gold)
                .count()
                >= 3
        );
        assert_eq!(batch.moisture_m3, 0.);
        close(stock(&s, &batch), before);
        assert!(batch.grains.iter().all(|g| g.material.valid()));
    }
    let mut water_only = source();
    water_only.feed_m3s = 0.02;
    water_only.hopper = Material {
        pore_water_m3: 5e-13,
        ..Default::default()
    };
    let before = water_only.active_material();
    let batch = water_only.take_screen_feed(FIXED_DT, true).unwrap();
    assert!(batch.grains.is_empty());
    assert_eq!(batch.moisture_m3, 0.);
    close(stock(&water_only, &batch), before);
}

#[test]
fn invalid_timestep_or_exhausted_sequence_refuses_without_mutating_any_source() {
    let mut s = source();
    s.hopper = Material {
        sand_m3: 0.01,
        gold_g: 0.001,
        ..Default::default()
    };
    s.grains.push(sentinel());
    for dt in [-1., f32::NAN, FIXED_DT * 2.] {
        let before = serde_json::to_value(&s).unwrap();
        assert!(s.take_screen_feed(dt, true).is_err());
        assert_eq!(serde_json::to_value(&s).unwrap(), before);
    }
    let mut encoded = serde_json::to_value(&s).unwrap();
    encoded["sequence"] = serde_json::json!(u64::MAX);
    let mut exhausted: Sluice = serde_json::from_value(encoded).unwrap();
    let before = serde_json::to_value(&exhausted).unwrap();
    assert!(exhausted.take_screen_feed(FIXED_DT, true).is_err());
    assert_eq!(serde_json::to_value(&exhausted).unwrap(), before);
}

#[test]
fn source_sequence_and_owned_residues_continue_exactly_after_save() {
    let mut a = source();
    a.hopper = Material {
        soil_m3: 0.01,
        sand_m3: 0.02,
        gravel_m3: 0.03,
        gold_g: 0.01,
        pore_water_m3: 0.005,
    };
    a.take_screen_feed(FIXED_DT, true).unwrap();
    let mut b: Sluice = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
    for _ in 0..30 {
        let x = a.take_screen_feed(FIXED_DT, true).unwrap();
        let y = b.take_screen_feed(FIXED_DT, true).unwrap();
        assert_eq!(
            serde_json::to_value(&x.grains).unwrap(),
            serde_json::to_value(&y.grains).unwrap()
        );
        assert_eq!(x.moisture_m3, y.moisture_m3);
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
    }
}
