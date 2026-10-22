use mining_sim::{
    session::Controls,
    site::{CELL, WIDTH},
    water_balance::WaterBalance,
    Action, Material, Session, Site, Vec3,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-11, "water changed: {a:.15} vs {b:.15}");
}
fn wet_site() -> (Site, usize, Vec3) {
    let mut s = Site::new();
    let p = Vec3::new(19., s.height(19., 17.), 17.);
    let i = 17usize * 4 * WIDTH + 19usize * 4;
    s.soil_water.ensure();
    s.soil_water.pore_m3[i] = s.column_pore_volume_m3(i) * 0.6;
    (s, i, p)
}
#[test]
fn pore_capacity_and_wet_mass_follow_independent_phase_volumes() {
    let mut m = Material {
        soil_m3: 0.1,
        sand_m3: 0.02,
        gravel_m3: 0.03,
        gold_g: 1.,
        ..Default::default()
    };
    let dry = 0.1 * 1800. + 0.02 * 1600. + 0.03 * 2000. + 0.001;
    let void = 0.15 - (dry - 0.001) / 2650. - 0.001 / 19300.;
    close(m.pore_volume_m3(), void);
    m.pore_water_m3 = void * 0.7;
    close(m.mass_kg(), dry);
    close(m.wet_mass_kg(), dry + 1000. * void * 0.7);
    assert!(m.valid());
    let left = m.take_fraction(0.3);
    close(m.pore_water_m3 + left.pore_water_m3, void * 0.7);
    m.pore_water_m3 = void * 2.;
    assert!(!m.valid());
}
#[test]
fn wet_cut_carries_its_share_of_finite_column_water_and_redeposit_preserves_it() {
    let (mut s, i, p) = wet_site();
    let water = s.water_volume_m3() + s.soil_water.stored_m3();
    let old = s.soil_water.pore_m3[i];
    let load = s.dig(p, 0.05, 0.06, 1.);
    let expected = load.soil_m3 * (1. - 1800. / 2650.) * 0.6;
    close(load.pore_water_m3, expected);
    assert!(load.pore_water_m3 > 0.);
    close(s.soil_water.pore_m3[i] + load.pore_water_m3, old);
    close(
        s.water_volume_m3() + s.soil_water.stored_m3() + load.pore_water_m3,
        water,
    );
    s.deposit(Vec3::new(22., s.height(22., 17.), 17.), load);
    close(s.water_volume_m3() + s.soil_water.stored_m3(), water);
    assert!(s.soil_water.valid());
}
#[test]
fn pore_water_follows_hand_bucket_flight_hopper_and_feed_without_creating_solids() {
    let (site, _, p) = wet_site();
    let mut s = Session::new();
    s.site = site;
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    let initial = s.stored_water_m3();
    close(
        s.accounted_material().pore_water_m3
            + s.site.soil_water.surface_m3.iter().sum::<f64>()
            + s.site.water_volume_m3()
            + s.sluice.stored_water_m3(),
        initial,
    );
    let mass = s.accounted_dry_mass_kg();
    let m = s.site.dig(p, 0.05, 0.06, 1.);
    s.player.hand.add(m);
    close(s.stored_water_m3(), initial);
    s.player.position = Vec3::new(22., s.site.height(22., 17.), 17.);
    s.player.yaw = 0.;
    s.on_action(Action::DumpHand).unwrap();
    assert!(!s.falling.is_empty());
    close(s.stored_water_m3(), initial);
    let save = s.save_json().unwrap();
    s = Session::load_json(&save).unwrap();
    close(s.stored_water_m3(), initial);
    for _ in 0..90 {
        s.step(Controls::default());
    }
    assert!(s.falling.is_empty());
    close(s.stored_water_m3(), initial);
    let wet = s
        .site
        .dig(Vec3::new(22., s.site.height(22., 17.), 17.), 0.8, 0.1, 1.);
    s.machine.bucket.add(wet);
    assert!(s.machine.bucket.pore_water_m3 > 0.);
    close(s.stored_water_m3(), initial);
    let parcel = s.machine.bucket.take_fraction(1.);
    s.sluice.add_to_hopper(parcel);
    close(s.stored_water_m3(), initial);
    s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    s.sluice.feed_on = true;
    let before = s.sluice.hopper.pore_water_m3;
    for _ in 0..120 {
        s.sluice.feed(mining_sim::FIXED_DT, true);
    }
    assert!(s.sluice.hopper.pore_water_m3 < before);
    assert!(s.sluice.feed_water_credit_m3 > 0.);
    assert!(s
        .sluice
        .grains
        .iter()
        .all(|g| g.material.pore_water_m3 == 0.));
    close(s.stored_water_m3(), initial);
    assert!((s.accounted_dry_mass_kg() - mass).abs() < 1e-5);
}
#[test]
fn lost_void_space_expels_water_without_losing_sub_ulp_films_or_creating_sediment() {
    let (mut s, i, _) = wet_site();
    let initial = s.soil_water.pore_m3[i];
    // Remove the column's void space without transporting its water: the
    // production GPU erosion hook must return it as actual surface water.
    s.world.overburden_thickness[i] = 0.;
    s.world.paydirt_thickness[i] = 0.;
    s.gold_g[i] = 0.;
    s.world.bedrock_elevation[i] = 1024.;
    s.world.water_surface[i] = 1024.;
    let before = s.water_volume_m3() + s.soil_water.stored_m3();
    s.reconcile_soil_water();
    close(s.water_volume_m3() + s.soil_water.stored_m3(), before);
    assert_eq!(s.soil_water.pore_m3[i], 0.);
    assert!(s.world.water_depth(i % WIDTH, i / WIDTH) > 0.);
    assert!(s.soil_water.surface_m3[i] < (1024f32.next_up() - 1024.) as f64 * (CELL * CELL) as f64);
    assert!(
        s.world.water_depth(i % WIDTH, i / WIDTH) as f64 * (CELL * CELL) as f64
            + s.soil_water.surface_m3[i]
            >= initial - 1e-12
    );
    let before = s.world.water_depth(i % WIDTH, i / WIDTH);
    s.soil_water.surface_m3[i] += 1e-14;
    let total = s.water_volume_m3() + s.soil_water.stored_m3();
    s.reconcile_soil_water();
    close(s.water_volume_m3() + s.soil_water.stored_m3(), total);
    assert_eq!(s.world.water_depth(i % WIDTH, i / WIDTH), before);
}
#[test]
fn expelled_clean_water_preserves_suspended_load_gold_and_horizontal_momentum() {
    let (mut s, i, _) = wet_site();
    s.world.overburden_thickness[i] = 0.;
    s.world.paydirt_thickness[i] = 0.;
    s.gold_g[i] = 0.;
    let ground = s.world.ground_height(i % WIDTH, i / WIDTH);
    s.world.water_surface[i] = ground + 0.02;
    let h = s.world.water_depth(i % WIDTH, i / WIDTH) as f64;
    for (field, c) in [
        (&mut s.hydraulic.suspended_sediment, 0.01),
        (&mut s.hydraulic.suspended_overburden, 0.02),
        (&mut s.hydraulic.suspended_gravel, 0.03),
        (&mut s.hydraulic.suspended_paydirt, 0.04),
    ] {
        *field = vec![0.; WIDTH * WIDTH];
        field[i] = c;
    }
    s.hydraulic.velocity_x = vec![0.; WIDTH * WIDTH];
    s.hydraulic.velocity_x[i] = 2.;
    s.hydraulic.velocity_z = vec![0.; WIDTH * WIDTH];
    s.hydraulic.velocity_z[i] = -1.;
    s.hydraulic.suspended_gold_g = vec![0.; WIDTH * WIDTH];
    s.hydraulic.suspended_gold_g[i] = 0.007;
    s.reconcile_soil_water();
    let after = s.world.water_depth(i % WIDTH, i / WIDTH) as f64;
    assert!(after > h);
    for (field, c) in [
        (&s.hydraulic.suspended_sediment, 0.01f32),
        (&s.hydraulic.suspended_overburden, 0.02),
        (&s.hydraulic.suspended_gravel, 0.03),
        (&s.hydraulic.suspended_paydirt, 0.04),
    ] {
        assert!((field[i] as f64 * after - c as f64 * h).abs() < c as f64 * h * 1e-6);
    }
    assert!((s.hydraulic.velocity_x[i] as f64 * after - 2. * h).abs() < h * 1e-6);
    assert!((s.hydraulic.velocity_z[i] as f64 * after + h).abs() < h * 1e-6);
    assert_eq!(s.hydraulic.suspended_gold_g[i], 0.007);
}

#[test]
fn old_saves_gain_no_water_and_new_saves_retain_wet_custody() {
    let mut s = Session::new();
    let wet = s.site.soil_water.stored_m3();
    assert!(wet > 0.);
    let save = s.save_json().unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&save).unwrap();
    assert_eq!(json["version"], 4);
    close(
        Session::load_json(&save).unwrap().stored_water_m3(),
        s.stored_water_m3(),
    );
    json["version"] = 3.into();
    assert!(Session::load_json(&json.to_string()).is_err());
    s.site.soil_water = Default::default();
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    let mut old: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    old["version"] = 3.into();
    old["site"].as_object_mut().unwrap().remove("soil_water");
    let loaded = Session::load_json(&old.to_string()).unwrap();
    close(loaded.stored_water_m3(), s.stored_water_m3());
    assert_eq!(loaded.site.soil_water.stored_m3(), 0.);
    let mut corrupt: serde_json::Value = serde_json::from_str(&save).unwrap();
    corrupt["site"]["soil_water"]["pore_m3"][0] = 1.0.into();
    assert!(Session::load_json(&corrupt.to_string()).is_err());
}
