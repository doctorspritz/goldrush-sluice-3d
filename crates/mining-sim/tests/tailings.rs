use glam::{Mat3, Quat};
use mining_sim::{
    sluice::{Grain, GrainKind},
    tailings::{world_grain, Tailings},
    Material, Placement, Session, Site, Sluice, Vec3,
};
fn flat(water: f32) -> Site {
    let mut s = Site::new();
    s.world.bedrock_elevation.fill(0.);
    s.world.overburden_thickness.fill(0.);
    s.world.paydirt_thickness.fill(0.);
    s.world.gravel_thickness.fill(0.);
    s.world.terrain_sediment.fill(0.);
    s.world.water_surface.fill(water);
    s.gold_g.fill(0.);
    s.hydraulic = Default::default();
    s.soil_water = Default::default();
    s
}
fn grain(p: Vec3, v: Vec3) -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Sand,
        position: p,
        velocity: v,
        material: Material {
            sand_m3: 1e-6,
            gold_g: 0.,
            ..Default::default()
        },
        resting: false,
        diameter_m: 0.001,
        affine: Mat3::from_cols(
            Vec3::new(1., 2., 3.),
            Vec3::new(4., 5., 6.),
            Vec3::new(7., 8., 9.),
        ),
        rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.6),
        angular_velocity: Vec3::new(0.3, -0.5, 0.7),
    }
}
fn close(a: f64, b: f64, e: f64) {
    assert!((a - b).abs() <= e, "{a} != {b} within {e}");
}
#[test]
fn world_transfer_preserves_polar_axial_and_affine_motion() {
    let mut s = Sluice::default();
    let site = Site::new();
    s.placement = Some(Placement {
        yaw: 0.7,
        ..Placement::suggested(&site)
    });
    let g = grain(Vec3::new(1.62, -0.02, 0.17), Vec3::new(0.8, -0.2, 0.3));
    let out = world_grain(&s, &g).unwrap();
    let p = s.placement.unwrap();
    let b = Mat3::from_cols(p.forward(), Vec3::Y, p.right());
    assert_eq!(out.material, g.material);
    assert_eq!(out.diameter_m, g.diameter_m);
    assert_eq!(out.volume_resolved, g.volume_resolved);
    assert!((out.position - p.to_world(s.grid_to_local(g.position))).length() < 1e-6);
    assert!((out.velocity - b * g.velocity).length() < 1e-6);
    let r = Vec3::new(0.02, -0.01, 0.03);
    assert!(
        (out.angular_velocity.cross(b * r) - b * g.angular_velocity.cross(r)).length() < 1e-6,
        "axial reflection sign"
    );
    assert!((out.affine * (b * r) - b * (g.affine * r)).length() < 1e-6);
    assert!((out.rotation * (b * r) - b * (g.rotation * r)).length() < 1e-6);
}
#[test]
fn ballistic_landing_matches_analytic_contact_and_refines() {
    let g = grain(Vec3::new(10.1, 1., 10.1), Vec3::new(4., -1., 2.));
    let t = (-1. + (1. + 2. * 9.81 * (1. - g.collision_radius() as f64)).sqrt()) / 9.81;
    for dt in [1. / 30., 1. / 60., 1. / 240.] {
        let mut site = flat(0.);
        let mut tails = Tailings::default();
        tails.prepare_batch(1).unwrap();
        tails.accept(&g).unwrap();
        for _ in 0..240 {
            if tails.packets.is_empty() {
                break;
            }
            tails.step(&mut site, dt);
        }
        assert!(tails.packets.is_empty() && tails.valid());
        close(tails.received.gold_g, g.material.gold_g, 1e-15);
        let mut x = 0.;
        let mut z = 0.;
        for i in 0..mining_sim::site::WIDTH.pow(2) {
            let m = site.bed_class_volume_m3(i, 0);
            x += m * (i % mining_sim::site::WIDTH) as f64 * 0.25;
            z += m * (i / mining_sim::site::WIDTH) as f64 * 0.25;
        }
        close(x / g.material.sand_m3, 10.1 + 4. * t, 3e-5);
        close(z / g.material.sand_m3, 10.1 + 2. * t, 3e-5);
        close(site.stored_material().volume(), g.material.volume(), 1e-15);
    }
}
#[test]
fn interior_bilinear_ridge_cannot_be_tunnelled_between_clear_endpoints() {
    let mut site = flat(0.);
    let width = mining_sim::site::WIDTH;
    site.world.bedrock_elevation[40 * width + 41] = 4.;
    site.world.bedrock_elevation[41 * width + 40] = 4.;
    let mut t = Tailings::default();
    t.accept(&grain(Vec3::new(10., 1.5, 10.), Vec3::new(1.25, 0., 1.25)))
        .unwrap();
    t.step(&mut site, 0.2);
    assert!(t.packets.is_empty());
    assert!(t.received.volume() > 0.);
    assert!(t.valid());
}
#[test]
fn water_impact_enters_actual_suspension_and_retains_gold_and_water() {
    let mut site = flat(0.4);
    let water = site.water_volume_m3() + site.soil_water.stored_m3();
    let mut t = Tailings::default();
    let g = grain(Vec3::new(10.1, 0.45, 10.1), Vec3::new(1., -1., 0.));
    let m = g.material;
    t.accept(&g).unwrap();
    for _ in 0..30 {
        t.step(&mut site, 1. / 60.);
    }
    assert!(t.packets.is_empty() && t.valid());
    close(
        site.hydraulic.suspended_gold_g.iter().sum(),
        m.gold_g,
        1e-15,
    );
    close(site.gold_g.iter().sum(), 0., 0.);
    close(site.stored_material().volume(), m.volume(), 1e-15);
    close(
        site.water_volume_m3() + site.soil_water.stored_m3(),
        water,
        1e-12,
    );
    assert!(site.world.terrain_sediment.iter().all(|h| *h == 0.));
}
#[test]
fn tiny_local_bed_transfers_preserve_sub_ulp_stock_and_free_water() {
    let mut site = flat(1.13);
    site.world.terrain_sediment.fill(1.);
    let initial = site.stored_material().sand_m3;
    let water = site.water_volume_m3() + site.soil_water.stored_m3();
    let m = Material {
        sand_m3: 1e-10,
        gold_g: 0.,
        ..Default::default()
    };
    for _ in 0..1000 {
        site.receive_tailings(Vec3::new(10.125, 1., 10.125), m, true)
            .unwrap();
    }
    close(site.stored_material().sand_m3 - initial, 1e-7, 1e-11);
    close(site.gold_g.iter().sum(), 0., 0.);
    close(
        site.water_volume_m3() + site.soil_water.stored_m3(),
        water,
        1e-12,
    );
    assert!(site
        .hydraulic
        .bed_remainder_m3
        .iter()
        .flatten()
        .any(|v| *v > 0.));
}
#[test]
fn boundary_export_and_capacity_refusal_never_lose_a_packet() {
    let mut site = flat(0.);
    let mut t = Tailings::default();
    let g = grain(Vec3::new(31.9, 2., 10.), Vec3::new(2., 0., 0.));
    let m = g.material;
    t.accept(&g).unwrap();
    let before = serde_json::to_string(&t).unwrap();
    assert!(t.prepare_batch(usize::MAX).is_err());
    assert_eq!(serde_json::to_string(&t).unwrap(), before);
    let e = t.step(&mut site, 0.1);
    assert_eq!(e, m);
    assert_eq!(t.exported, m);
    assert!(t.packets.is_empty() && t.valid());
    assert_eq!(site.revision, 0);
}
#[test]
fn spatial_save_retains_finite_airborne_state_and_refuses_downgrade_or_corruption() {
    let mut s = Session::new_operation();
    let m = s.site.dig(Vec3::new(19., 0., 22.), 0.2, 0.1, 0.01);
    let mut g = grain(Vec3::new(10., 2., 10.), Vec3::X);
    g.material = m;
    s.tailings.as_mut().unwrap().accept(&g).unwrap();
    let saved = s.save_json().unwrap();
    let restored = Session::load_json(&saved).unwrap();
    let mut replay = Session::load_json(&saved).unwrap();
    for _ in 0..120 {
        s.step(Default::default());
        replay.step(Default::default());
    }
    assert_eq!(s.save_json().unwrap(), replay.save_json().unwrap());
    let s = restored.clone();
    let value: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(value["version"], 13);
    assert_eq!(value["mechanical_version"], 6);
    assert_eq!(
        serde_json::to_value(&restored.tailings).unwrap(),
        serde_json::to_value(&s.tailings).unwrap()
    );
    close(s.accounted_gold(), s.initial_gold_g, 1e-7);
    for key in ["tailings", "mechanical_version", "water_balance"] {
        let mut bad = value.clone();
        bad.as_object_mut().unwrap().remove(key);
        assert!(
            Session::load_json(&bad.to_string()).is_err(),
            "missing {key}"
        );
    }
    let mut bad = value.clone();
    bad["version"] = 6.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut bad = value.clone();
    bad["tailings"]["packets"][0]["grain"]["material"]["gold_g"] = (-1.).into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut bad = value.clone();
    bad["tailings"]["received"]["sand_m3"] = 1.0.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
}
#[test]
fn legacy_pooled_stock_migrates_once_without_changing_any_inventory() {
    let mut s = Session::new();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    let m = s.site.dig(Vec3::new(19., 0., 22.), 0.2, 0.1, 0.01);
    s.sluice.tailings.add(m);
    let old = s.save_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&old).unwrap()["version"],
        6
    );
    assert!(Session::load_json(&old).unwrap().tailings.is_none());
    let bulk = s.accounted_bulk_material_m3();
    let water = s.stored_water_m3();
    s.enable_spatial_tailings().unwrap();
    close(s.sluice.tailings.gold_g, m.gold_g, 0.);
    assert_eq!(s.sluice.tailings.volume(), 0.);
    let mut mineral = m;
    mineral.gold_g = 0.;
    assert_eq!(s.tailings.as_ref().unwrap().historic_bed, mineral);
    close(s.accounted_bulk_material_m3(), bulk, 1e-8);
    close(s.stored_water_m3(), water, 1e-10);
    close(s.accounted_gold(), s.initial_gold_g, 1e-7);
    let once = s.save_json().unwrap();
    s.enable_spatial_tailings().unwrap();
    assert_eq!(s.save_json().unwrap(), once);
    assert!(Session::load_json(&once).is_ok());
}
#[test]
fn modern_envelope_keeps_mechanism_version_guards() {
    let mut s = Session::new_operation();
    s.machine.enable_articulation().unwrap();
    let json = s.save_json().unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["version"], 13);
    assert_eq!(value["mechanical_version"], 11);
    assert!(Session::load_json(&json).is_ok());
    s.activate_soil(&[(100 * mining_sim::site::WIDTH + 104, 0.02)], 1)
        .unwrap();
    assert!(Session::load_json(&s.save_json().unwrap())
        .unwrap()
        .active_soil
        .is_some());
    value["mechanical_version"] = 6.into();
    assert!(Session::load_json(&value.to_string()).is_err());
}

#[test]
fn gold_contact_retains_complete_owned_packets_without_fake_carrier() {
    for (water, pure_gold) in [(0., false), (0.4, false), (0., true), (0.4, true)] {
        let mut site = flat(water);
        let mut t = Tailings::default();
        let mut g = grain(Vec3::new(10.1, 0.45, 10.1), Vec3::new(1., -1., 0.));
        g.material.gold_g = 0.002;
        if pure_gold {
            g.material.sand_m3 = 0.;
            g.kind = GrainKind::Gold;
        }
        t.accept(&g).unwrap();
        for _ in 0..60 {
            t.step(&mut site, 1. / 60.);
        }
        assert_eq!(t.packets.len(), 1);
        let p = &t.packets[0];
        assert!(p.contact_pending);
        assert_eq!(p.grain.material, g.material);
        assert_eq!(p.grain.diameter_m, g.diameter_m);
        assert_eq!(p.grain.affine, g.affine);
        assert_eq!(p.grain.angular_velocity, g.angular_velocity);
        assert_eq!(t.active_material(), g.material);
        assert!(t.received.is_empty() && t.valid());
        assert_eq!(site.remaining_gold(), 0.);
        assert_eq!(site.stored_material().volume(), 0.);
        let json = serde_json::to_string(&t).unwrap();
        let restored: Tailings = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&restored).unwrap());
        assert!(restored.valid());
        assert!(site
            .receive_tailings(g.position, g.material, false)
            .is_err());
    }
}
