use glam::{DVec3, Mat3, Quat};
use mining_sim::{
    screen::{BulkDeck, Screen},
    sluice::Grain,
    GrainKind, Material, Placement, Session, Vec3,
};
use std::time::Instant;

fn grain(screen: &Screen, id: usize, diameter: f32, mut material: Material) -> Grain {
    if material.mass_kg() == 0. {
        material.sand_m3 = 1e-10;
    }
    let x = 0.05 + (id % 16) as f64 * 0.04;
    let z = -0.14 + (id / 16 % 8) as f64 * 0.04;
    let p = screen.to_world(DVec3::new(
        x,
        0.03 - x * screen.config.slope_deg.to_radians().tan(),
        z,
    ));
    Grain {
        kind: GrainKind::Sand,
        material,
        diameter_m: diameter,
        position: p.as_vec3(),
        velocity: Vec3::ZERO,
        affine: Mat3::ZERO,
        rotation: Quat::IDENTITY,
        angular_velocity: Vec3::ZERO,
        resting: false,
        volume_resolved: true,
    }
}
fn screen(bulk: bool, count: usize, diameter: f32, material: Material) -> Screen {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    if bulk {
        s.set_bulk_deck(Some(BulkDeck::default())).unwrap();
    }
    let grains: Vec<_> = (0..count)
        .map(|i| grain(&s, i, diameter, material))
        .collect();
    s.admit(&grains, 0.).unwrap();
    s.motor_on = true;
    s
}
fn run(s: &mut Screen, ticks: usize, dt: f32) {
    for _ in 0..ticks {
        s.step(dt).unwrap();
        assert!(s.valid());
    }
}
fn budget(s: &Screen) {
    let mut actual = s.material();
    actual.add(s.delivered);
    actual.add(s.released);
    for (a, b) in [
        (actual.soil_m3, s.admitted.soil_m3),
        (actual.sand_m3, s.admitted.sand_m3),
        (actual.gravel_m3, s.admitted.gravel_m3),
        (actual.gold_g, s.admitted.gold_g),
        (actual.pore_water_m3, s.admitted.pore_water_m3),
    ] {
        assert!((a - b).abs() < 1e-12);
    }
}
#[test]
fn finite_grading_uses_clearance_and_not_the_material_tag() {
    let mut fine = screen(true, 128, 0.0005, Material::default());
    let mut tagged = fine.clone();
    for p in &mut tagged.particles {
        p.grain.kind = GrainKind::Gold;
    }
    let mut coarse = screen(true, 128, 0.006, Material::default());
    run(&mut fine, 360, 1. / 60.);
    run(&mut tagged, 360, 1. / 60.);
    run(&mut coarse, 360, 1. / 60.);
    assert!(
        fine.undersize.len() > 60,
        "fine underflow {}",
        fine.undersize.len()
    );
    assert_eq!(fine.undersize.len(), tagged.undersize.len());
    assert_eq!(coarse.undersize.len(), 0);
    assert!(coarse.oversize.len() + coarse.spills.len() > 30);
    for s in [&fine, &coarse, &tagged] {
        budget(s);
    }
    println!(
        "grading fine={} coarse={} coarse_reject={} coarse_spill={}",
        fine.undersize.len(),
        coarse.undersize.len(),
        coarse.oversize.len(),
        coarse.spills.len()
    );
}
#[test]
fn aperture_upgrade_changes_passage_of_the_same_finite_cohort() {
    let mut narrow = screen(true, 128, 0.004, Material::default());
    let mut wide = Screen::new(Vec3::ZERO, 0.);
    let mut config = wide.config.clone();
    config.aperture_m = 0.010;
    wide.set_config(config).unwrap();
    wide.set_bulk_deck(Some(BulkDeck::default())).unwrap();
    let grains: Vec<_> = (0..128)
        .map(|i| grain(&wide, i, 0.004, Material::default()))
        .collect();
    wide.admit(&grains, 0.).unwrap();
    wide.motor_on = true;
    run(&mut narrow, 360, 1. / 60.);
    run(&mut wide, 360, 1. / 60.);
    assert_eq!(narrow.undersize.len(), 0);
    assert!(wide.undersize.len() > 30);
    assert_eq!(narrow.admitted, wide.admitted);
    budget(&wide);
}
#[test]
fn dense_local_feed_builds_bed_and_wet_fines_can_blind_it() {
    let dry = Material {
        soil_m3: 2e-6,
        gold_g: 1e-8,
        ..Default::default()
    };
    let wet = Material {
        pore_water_m3: dry.pore_volume_m3() * 0.7,
        ..dry
    };
    let mut a = screen(true, 128, 0.0001, dry);
    let mut b = screen(true, 128, 0.0001, wet);
    run(&mut a, 90, 1. / 60.);
    run(&mut b, 90, 1. / 60.);
    let dry_blind: f64 = a.bulk_bed().unwrap().iter().map(|c| c.1).sum();
    let wet_bed = b.bulk_bed().unwrap();
    let wet_blind: f64 = wet_bed.iter().map(|c| c.1).sum();
    assert!(wet_blind > dry_blind + 1.);
    assert!(wet_bed.iter().any(|c| c.0 > 0.001));
    assert!(
        wet_bed.iter().any(|c| c.0 == 0.),
        "local obstruction must not span whole deck"
    );
    assert!(b.particles.len() > a.particles.len());
    budget(&a);
    budget(&b);
    b.motor_on = false;
    let all = b.material();
    let out = b.cleanout().unwrap();
    let mut removed = Material::default();
    for g in out {
        removed.add(g.material);
    }
    assert!((all.gold_g - removed.gold_g).abs() < 1e-15);
    assert!(b.bulk_bed().unwrap().iter().all(|c| c.0 == 0. && c.1 == 0.));
    assert!(b.valid());
    println!(
        "wet local coverage cells={} dry={} wet_active={} dry_active={}",
        wet_blind,
        dry_blind,
        b.particles.len(),
        a.particles.len()
    );
}
#[test]
fn overload_has_more_bed_depth_and_slower_exposure() {
    let mut dilute = screen(
        true,
        128,
        0.0005,
        Material {
            sand_m3: 1e-10,
            ..Default::default()
        },
    );
    let mut loaded = screen(
        true,
        128,
        0.0005,
        Material {
            sand_m3: 8e-6,
            ..Default::default()
        },
    );
    run(&mut dilute, 15, 1. / 60.);
    run(&mut loaded, 15, 1. / 60.);
    let low_depth: f64 = dilute.bulk_bed().unwrap().iter().map(|c| c.0).sum();
    let high_depth: f64 = loaded.bulk_bed().unwrap().iter().map(|c| c.0).sum();
    assert!(high_depth > low_depth);
    run(&mut dilute, 105, 1. / 60.);
    run(&mut loaded, 105, 1. / 60.);
    println!("load dilute_active={} loaded_active={} dilute_under={} loaded_under={} loaded_spills={} loaded_reject={} initial_depth_ratio={}", dilute.particles.len(),loaded.particles.len(),dilute.undersize.len(),loaded.undersize.len(),loaded.spills.len(),loaded.oversize.len(),high_depth/low_depth.max(1e-30));
    assert!(
        loaded.undersize.len() < dilute.undersize.len(),
        "Overload must alter finite product grading"
    );
    budget(&loaded);
    budget(&dilute);
}
#[test]
fn saved_residence_continues_exactly_without_an_extra_passage_attempt() {
    let mut a = screen(
        true,
        128,
        0.002,
        Material {
            sand_m3: 2e-7,
            ..Default::default()
        },
    );
    run(&mut a, 45, 1. / 60.);
    let json = serde_json::to_string(&a).unwrap();
    let mut b: Screen = serde_json::from_str(&json).unwrap();
    for _ in 0..120 {
        a.step(1. / 60.).unwrap();
        b.step(1. / 60.).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }
    budget(&a);
}
#[test]
fn motor_stop_freezes_drive_but_does_not_freeze_gravity_or_duplicate_stock() {
    let mut s = screen(true, 128, 0.004, Material::default());
    run(&mut s, 30, 1. / 60.);
    s.motor_on = false;
    let clock = s.time;
    let elapsed = s.elapsed_time_s;
    let before = serde_json::to_string(&s.particles).unwrap();
    run(&mut s, 60, 1. / 60.);
    assert_eq!(s.time, clock);
    assert!(s.elapsed_time_s > elapsed);
    assert_ne!(serde_json::to_string(&s.particles).unwrap(), before);
    budget(&s);
    s.motor_on = true;
    run(&mut s, 60, 1. / 60.);
    assert!(s.time > clock);
    budget(&s);
}
#[test]
fn model_switch_and_invalid_state_refuse_atomically() {
    let mut s = screen(true, 4, 0.001, Material::default());
    let before = serde_json::to_string(&s).unwrap();
    assert!(s.set_bulk_deck(None).is_err());
    assert_eq!(serde_json::to_string(&s).unwrap(), before);
    assert!(s.step(f32::NAN).is_err());
    assert_eq!(serde_json::to_string(&s).unwrap(), before);
    let mut json = serde_json::to_value(&s).unwrap();
    json["bulk"]["residence"]["1"]["exposure"] = (-1.).into();
    let mut invalid: Screen = serde_json::from_value(json).unwrap();
    let bad = serde_json::to_string(&invalid).unwrap();
    assert!(!invalid.valid());
    assert!(invalid.step(1. / 60.).is_err());
    assert_eq!(serde_json::to_string(&invalid).unwrap(), bad);
    let mut empty = Screen::new(Vec3::ZERO, 0.);
    assert!(empty
        .set_bulk_deck(Some(BulkDeck {
            cells_across: 0,
            ..Default::default()
        }))
        .is_err());
    assert!(empty.bulk_deck().is_none());
}
#[test]
fn time_refinement_preserves_grading_statistics_and_complete_inventory() {
    let mut a = screen(true, 128, 0.0015, Material::default());
    let mut b = a.clone();
    run(&mut a, 240, 1. / 60.);
    run(&mut b, 480, 1. / 120.);
    assert!(a.undersize.len().abs_diff(b.undersize.len()) < 12);
    budget(&a);
    budget(&b);
    println!(
        "refinement 60Hz={} 120Hz={}",
        a.undersize.len(),
        b.undersize.len()
    );
}
#[test]
fn actual_dug_feed_bulk_screen_saved_plant_and_sluice_keep_finite_custody() {
    let mut s = Session::new_operation();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.enable_screen().unwrap();
    s.plant
        .as_mut()
        .unwrap()
        .set_bulk_screen(Some(BulkDeck::default()))
        .unwrap();
    let at = Vec3::new(19., 0., 20.);
    let spoil = s.site.dig(at, 0.4, 0.10, 0.20);
    s.site.try_deposit(at + Vec3::X * 2., spoil).unwrap();
    let pay = s.site.dig(at, 0.4, 0.15, 0.04);
    assert!(pay.gravel_m3 > 0. && pay.gold_g > 0.);
    assert!(s.sluice.add_to_hopper(pay).is_empty());
    s.sluice.feed_on = true;
    s.sluice.pump_on = true;
    s.plant.as_mut().unwrap().screen.motor_on = true;
    let baseline = s.accounted_material();
    let water = s.stored_water_m3();
    for _ in 0..120 {
        s.step(Default::default());
        assert!(!s.paused, "{}", s.message);
    }
    let saved = s.save_json().unwrap();
    let mut restored = Session::load_json(&saved).unwrap();
    for _ in 0..60 {
        s.step(Default::default());
        restored.step(Default::default());
        assert!(!s.paused, "{}", s.message);
    }
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
    let p = s.plant.as_ref().unwrap();
    assert_eq!(p.model_version, 2);
    assert!(p.screen.delivered.mass_kg() > 0. && !s.sluice.grains.is_empty());
    assert!((s.accounted_material().volume() - baseline.volume()).abs() < 1e-7);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
    assert!((s.stored_water_m3() - water).abs() < 1e-7);
    let mut bad: serde_json::Value = serde_json::from_str(&saved).unwrap();
    bad["plant"]["model_version"] = 1.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    if let Some(path) = std::env::var_os("SLUICE_BULK_FIXTURE") {
        std::fs::write(path, saved).unwrap();
    }
}
#[test]
fn original_wire_reference_and_bulk_are_measured_separately() {
    let material = Material {
        sand_m3: 1e-10,
        ..Default::default()
    };
    let mut reference = screen(false, 32, 0.001, material);
    let mut bulk = screen(true, 32, 0.001, material);
    let now = Instant::now();
    run(&mut reference, 120, 1. / 60.);
    let wire_ms = now.elapsed().as_secs_f64() * 1000.;
    let now = Instant::now();
    run(&mut bulk, 120, 1. / 60.);
    let bulk_ms = now.elapsed().as_secs_f64() * 1000.;
    assert!(reference.bulk_deck().is_none());
    budget(&reference);
    budget(&bulk);
    assert!(!reference.undersize.is_empty() && !bulk.undersize.is_empty());
    println!("screen_only ticks=120 parcels=32 wire_ms={wire_ms:.4} bulk_ms={bulk_ms:.4} reference_underflow={} bulk_underflow={} not_native_or_real_time=true",reference.undersize.len(),bulk.undersize.len());
}

#[test]
fn static_dilute_passage_agrees_with_independent_open_clearance_area() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let mut config = s.config.clone();
    config.slope_deg = 0.;
    s.set_config(config).unwrap();
    s.set_bulk_deck(Some(BulkDeck::default())).unwrap();
    let grains: Vec<_> = (0..1024)
        .map(|i| {
            grain(
                &s,
                i,
                0.0005,
                Material {
                    sand_m3: 1e-14,
                    ..Default::default()
                },
            )
        })
        .collect();
    s.admit(&grains, 0.).unwrap();
    run(&mut s, 240, 1. / 60.);
    let measured = s.undersize.len() as f64 / 1024.;
    let expected = ((0.003_f64 - 0.0005) / (0.003 + 2. * 0.001)).powi(2);
    assert!(
        (measured - expected).abs() < 0.04,
        "clearance area expected {expected} actual {measured}"
    );
    budget(&s);
    println!("static clearance probability expected={expected} actual={measured}");
}

#[test]
fn saturated_sand_loses_pendular_bridging_while_damp_sand_can_blind() {
    let dry = Material {
        sand_m3: 2e-7,
        ..Default::default()
    };
    let damp = Material {
        pore_water_m3: dry.pore_volume_m3() * 0.2,
        ..dry
    };
    let saturated = Material {
        pore_water_m3: dry.pore_volume_m3(),
        ..dry
    };
    let mut a = screen(true, 128, 0.0001, damp);
    let mut b = screen(true, 128, 0.0001, saturated);
    run(&mut a, 240, 1. / 60.);
    run(&mut b, 240, 1. / 60.);
    assert!(a.particles.len() > b.particles.len());
    assert!(a.bulk_bed().unwrap().iter().map(|c| c.1).sum::<f64>() > 0.);
    assert!(b.bulk_bed().unwrap().iter().all(|c| c.1 == 0.));
    budget(&a);
    budget(&b);
    println!(
        "moisture damp_under={} saturated_under={} damp_held={} saturated_held={}",
        a.undersize.len(),
        b.undersize.len(),
        a.particles.len(),
        b.particles.len()
    );
}

#[test]
fn offcentre_wet_stock_blinds_its_lane_without_a_width_wide_clog_flag() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    s.set_bulk_deck(Some(BulkDeck::default())).unwrap();
    let dry = Material {
        soil_m3: 2e-6,
        ..Default::default()
    };
    let wet = Material {
        pore_water_m3: dry.pore_volume_m3() * 0.7,
        ..dry
    };
    let grains: Vec<_> = (0..32)
        .map(|i| {
            let mut g = grain(&s, i, 0.0001, wet);
            let mut local = s.to_local(g.position.as_dvec3());
            local.z = -0.13;
            g.position = s.to_world(local).as_vec3();
            g
        })
        .collect();
    s.admit(&grains, 0.).unwrap();
    s.motor_on = true;
    run(&mut s, 90, 1. / 60.);
    let bed = s.bulk_bed().unwrap();
    assert!(bed.iter().enumerate().any(|(i, c)| i % 4 == 0 && c.1 > 0.));
    assert!(bed
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 != 0)
        .all(|(_, c)| c.0 == 0. && c.1 == 0.));
    assert_eq!(s.particles.len(), 32);
    budget(&s);
}
