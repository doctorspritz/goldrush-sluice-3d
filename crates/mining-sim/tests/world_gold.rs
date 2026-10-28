use glam::{Mat3, Quat};
use mining_sim::{
    loose_gold::LooseGold, session::FallingLoad, site::WIDTH, sluice::Grain, GrainKind, Material,
    Session, Site, Sluice, Vec3,
};
fn flat(depth: f32) -> Site {
    let mut s = Site::new();
    s.world.bedrock_elevation.fill(0.);
    s.world.overburden_thickness.fill(0.);
    s.world.paydirt_thickness.fill(0.);
    s.world.gravel_thickness.fill(0.);
    s.world.terrain_sediment.fill(0.);
    s.world.water_surface.fill(depth);
    s.gold_g.fill(0.);
    s.hydraulic = Default::default();
    s.soil_water = Default::default();
    s.loose_gold = Some(LooseGold::default());
    s
}
fn grain(p: Vec3, gold: f64) -> Grain {
    Grain {
        kind: GrainKind::Gold,
        position: p,
        velocity: Vec3::ZERO,
        material: Material {
            gold_g: gold,
            ..Default::default()
        },
        resting: false,
        diameter_m: 0.0002,
        rotation: Quat::IDENTITY,
        affine: Mat3::ZERO,
        angular_velocity: Vec3::ZERO,
        volume_resolved: true,
    }
}
fn owner_step(site: &mut Site, dt: f32) {
    let mut g = site.loose_gold.take().unwrap();
    g.step(site, dt).unwrap();
    site.loose_gold = Some(g);
}
#[test]
fn real_session_accepts_gold_contact_and_preserves_global_gold_through_restore() {
    let mut s = Session::new_operation();
    let gold = 0.01;
    let i = s.site.gold_g.iter().position(|g| *g > gold).unwrap();
    s.site.gold_g[i] -= gold;
    let p = Vec3::new(10., s.site.height(10., 10.) + 0.01, 10.);
    s.tailings
        .as_mut()
        .unwrap()
        .accept(&grain(p, gold))
        .unwrap();
    for _ in 0..30 {
        s.step(Default::default());
    }
    assert!(s.tailings.as_ref().unwrap().packets.is_empty());
    assert_eq!(s.tailings.as_ref().unwrap().received.gold_g, gold);
    assert_eq!(s.site.loose_gold.as_ref().unwrap().particles.len(), 1);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
    let saved = s.save_json().unwrap();
    let mut restored = Session::load_json(&saved).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&saved).unwrap()["version"],
        13
    );
    for _ in 0..120 {
        s.step(Default::default());
        restored.step(Default::default());
    }
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
}
#[test]
fn v13_rejects_downgrades_and_corruption_and_v12_retains_legacy_policy() {
    let s = Session::new_operation();
    let saved: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    for version in [6, 11, 12] {
        let mut bad = saved.clone();
        bad["version"] = version.into();
        assert!(Session::load_json(&bad.to_string()).is_err());
    }
    let mut bad = saved.clone();
    bad["site"].as_object_mut().unwrap().remove("loose_gold");
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut bad = saved;
    bad["site"]["loose_gold"]["model_version"] = 99.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut old = Session::new();
    old.enable_spatial_tailings().unwrap();
    let json = old.save_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap()["version"],
        12
    );
    let mut modern = Session::load_json(&json).unwrap();
    assert!(modern.site.loose_gold.is_none());
    modern.enable_loose_gold().unwrap();
    assert!(modern.site.loose_gold.is_some());
}
#[test]
fn surface_gold_can_be_dug_refed_and_redeposited_without_a_mineral_carrier() {
    let mut site = flat(0.);
    let g = grain(Vec3::new(10., 0.0001, 10.), 0.01);
    site.loose_gold.as_mut().unwrap().admit(&g).unwrap();
    owner_step(&mut site, 1. / 60.);
    let material = site.dig(g.position, 0.22, 0.12, f64::MAX);
    assert_eq!(material.gold_g, 0.01);
    assert_eq!(material.volume(), 0.);
    assert_eq!(site.remaining_gold(), 0.);
    site.try_deposit(g.position, material).unwrap();
    owner_step(&mut site, 1. / 60.);
    let material = site.dig(g.position, 0.22, 0.12, f64::MAX);
    assert_eq!(material.gold_g, 0.01);
    let mut sluice = Sluice::default();
    sluice.placement = Some(mining_sim::Placement::suggested(&site));
    assert!(sluice.add_to_hopper(material).is_empty());
    sluice.feed_on = true;
    sluice.pump_on = true;
    for _ in 0..600 {
        sluice.feed(1. / 60., true);
        sluice.drain_pending_gold();
    }
    assert!((sluice.active_material().gold_g - 0.01).abs() < 1e-12);
    assert!(sluice.grains.iter().map(|g| g.material.gold_g).sum::<f64>() > 0.009);
    assert!(site.gold_g.iter().all(|g| *g == 0.));
}
#[test]
fn mixed_dump_does_not_reattach_gold_to_unrelated_paydirt_tags() {
    let mut site = flat(0.);
    let m = Material {
        sand_m3: 0.0001,
        gold_g: 0.01,
        ..Default::default()
    };
    site.try_deposit(Vec3::new(10., 0., 10.), m).unwrap();
    assert!(site.gold_g.iter().all(|g| *g == 0.));
    assert_eq!(
        site.loose_gold.as_ref().unwrap().material().gold_g,
        m.gold_g
    );
    let cut = site.dig(Vec3::new(10., 0., 10.), 0.8, 0.2, 1.);
    assert_eq!(cut.gold_g, m.gold_g);
    assert!((site.stored_material().volume() + cut.volume() - m.volume()).abs() < 1e-12);
    assert_eq!(site.remaining_gold(), 0.);
}
#[test]
fn mobile_intake_and_boundary_exports_have_separate_finite_custody() {
    let mut site = flat(0.3);
    site.loose_gold
        .as_mut()
        .unwrap()
        .admit(&grain(Vec3::new(10., 0.15, 10.), 0.01))
        .unwrap();
    site.loose_gold
        .as_mut()
        .unwrap()
        .admit(&grain(Vec3::new(10., 0.0001, 10.), 0.02))
        .unwrap();
    owner_step(&mut site, 1. / 120.);
    let packet = site.withdraw_intake(Vec3::new(10., 0., 10.), 0.001);
    assert!(packet.suspension.gold_g > 0. && packet.suspension.gold_g < 0.01);
    assert!((site.remaining_gold() + packet.suspension.gold_g - 0.03).abs() < 1e-15);
    assert_eq!(
        site.loose_gold.as_ref().unwrap().particles[1]
            .grain
            .material
            .gold_g,
        0.02
    );
    let mut s = Session::new_operation();
    let gold = 0.01;
    let i = s.site.gold_g.iter().position(|g| *g > gold).unwrap();
    s.site.gold_g[i] -= gold;
    let mut g = grain(Vec3::new(31.999, 10., 10.), gold);
    g.velocity.x = 1.;
    s.site.loose_gold.as_mut().unwrap().admit(&g).unwrap();
    s.step(Default::default());
    assert_eq!(s.site.loose_gold.as_ref().unwrap().exported_g, gold);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
    assert!((s.accounted_material().gold_g - s.initial_gold_g).abs() < 1e-7);
}
#[test]
fn pure_gold_capacity_and_failed_deposition_retain_the_donor() {
    let mut s = Sluice::default();
    let material = Material {
        gold_g: 19_300. * 1000.,
        ..Default::default()
    };
    let predicted = s.hopper_remainder(material);
    let refused = s.add_to_hopper(material);
    assert_eq!(predicted, refused);
    assert!(s.hopper.handling_volume_m3() <= 0.85 + 1e-15);
    assert!(refused.gold_g > 0.);
    assert!((s.hopper.gold_g + refused.gold_g - material.gold_g).abs() < 1e-8);
    let mut session = Session::new_operation();
    session.site = flat(0.);
    session.initial_gold_g = 0.01;
    session.site.loose_gold.as_mut().unwrap().next_id = u64::MAX;
    session.falling.push(FallingLoad {
        position: Vec3::new(10., 0.00001, 10.),
        velocity: Vec3::ZERO,
        material: Material {
            gold_g: 0.01,
            ..Default::default()
        },
        rigid: None,
    });
    session.step(Default::default());
    assert!(session.paused);
    assert_eq!(session.falling.len(), 1);
    assert_eq!(session.falling[0].material.gold_g, 0.01);
    assert_eq!(session.accounted_gold(), session.initial_gold_g);
    assert!(Session::load_json(&session.save_json().unwrap()).is_ok());
}
#[test]
fn gold_step_refusal_does_not_advance_tailings_or_the_physical_clock() {
    let mut s = Session::new_operation();
    s.site = flat(0.);
    let mut fast = grain(Vec3::new(10., 1., 10.), 0.01);
    fast.velocity.x = 1e20;
    s.site.loose_gold.as_mut().unwrap().admit(&fast).unwrap();
    let mineral = Grain {
        kind: GrainKind::Sand,
        material: Material {
            sand_m3: 1e-6,
            ..Default::default()
        },
        ..grain(Vec3::new(10., 1., 10.), 0.)
    };
    s.tailings.as_mut().unwrap().accept(&mineral).unwrap();
    let before = serde_json::to_value(&s.tailings).unwrap();
    s.step(Default::default());
    assert!(s.paused);
    assert_eq!(s.sim_time, 0.);
    assert_eq!(serde_json::to_value(&s.tailings).unwrap(), before);
    assert_eq!(s.site.world.water_surface.len(), WIDTH * WIDTH);
}

#[test]
fn stopping_feed_with_a_full_hopper_keeps_pending_gold_without_overfill() {
    let mut s = Session::new_operation();
    s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
    s.sluice.feed_on = true;
    s.sluice.pump_on = true;
    s.sluice.hopper.gold_g = 0.01;
    s.sluice.feed(0.25, true);
    let emitted = s
        .sluice
        .grains
        .iter()
        .map(|g| g.material.gold_g)
        .sum::<f64>();
    assert!(emitted > 0. && emitted < 0.01);
    assert_eq!(s.sluice.hopper.gold_g, 0.);
    s.sluice.add_to_hopper(Material {
        sand_m3: 1.,
        ..Default::default()
    });
    let before = s.sluice.active_material();
    s.sluice.feed_on = false;
    for _ in 0..5 {
        s.sluice.feed(1. / 60., true);
    }
    assert!(s.sluice.hopper.handling_volume_m3() <= 0.85 + 1e-15);
    assert_eq!(s.sluice.active_material(), before);
    assert!(s.sluice.valid());
    let mut saved: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    saved["sluice"]["hopper"]["gold_g"] = 19300.into();
    assert!(Session::load_json(&saved.to_string())
        .err()
        .unwrap()
        .contains("physical capacity"));
}

#[test]
fn tiny_positive_dump_stock_and_failed_gold_receipts_are_not_discarded() {
    let mut site = flat(0.);
    let tiny = Material {
        sand_m3: 1e-11,
        gold_g: 1e-12,
        pore_water_m3: 1e-13,
        ..Default::default()
    };
    assert!(tiny.is_empty());
    site.try_deposit(Vec3::new(10., 0., 10.), tiny).unwrap();
    let stored = site.stored_material();
    assert!((stored.sand_m3 - tiny.sand_m3).abs() < 1e-20);
    assert!((stored.pore_water_m3 - tiny.pore_water_m3).abs() < 1e-20);
    assert_eq!(stored.gold_g, tiny.gold_g);
    let mut s = Session::new_operation();
    s.site = flat(0.);
    s.initial_gold_g = tiny.gold_g;
    s.site.loose_gold.as_mut().unwrap().next_id = u64::MAX;
    s.falling.push(FallingLoad {
        position: Vec3::new(10., 0., 10.),
        velocity: Vec3::ZERO,
        material: Material {
            gold_g: tiny.gold_g,
            ..Default::default()
        },
        rigid: None,
    });
    s.step(Default::default());
    assert!(s.paused);
    assert_eq!(s.falling.len(), 1);
    assert_eq!(s.accounted_gold(), tiny.gold_g);
}
