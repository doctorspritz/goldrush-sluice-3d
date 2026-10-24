use mining_sim::{
    site::{CELL, WIDTH},
    water_balance::WaterBalance,
    Material, Session, Vec3,
};
const AREA: f64 = (CELL as f64) * (CELL as f64);
fn fixture() -> Session {
    let mut s = Session::new();
    s.site.soil_water = Default::default();
    for z in 62..=66 {
        for x in 54..=58 {
            let i = z * WIDTH + x;
            s.site.world.terrain_sediment[i] = 0.125;
            s.site.world.overburden_thickness[i] = 0.125;
            s.site.world.paydirt_thickness[i] = 0.125;
            s.site.world.gravel_thickness[i] = 0.;
            s.site.gold_g[i] = 0.;
            s.site.world.water_surface[i] = s.site.world.ground_height(x, z);
        }
    }
    s.initial_gold_g = s.accounted_gold();
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    s
}
#[test]
fn deposits_smaller_than_geometry_remain_owned_through_digging_and_restore() {
    let mut s = fixture();
    let position = Vec3::new(14., 0., 16.);
    let i = 64 * WIDTH + 56;
    let source = Material {
        soil_m3: 1e-9,
        sand_m3: 1e-9,
        gravel_m3: 1e-9,
        gold_g: 1e-9,
        ..Default::default()
    };
    s.site.deposit(position, source);
    s.initial_gold_g = s.accounted_gold();
    for (class, expected) in [
        (0, source.sand_m3),
        (1, source.soil_m3),
        (3, source.gravel_m3),
    ] {
        let low = s
            .site
            .hydraulic
            .bed_remainder_m3
            .iter()
            .map(|v| v[class])
            .sum::<f64>();
        assert!(
            (low / expected - 1.).abs() < 1e-6,
            "subgrid deposit {class}: {low}"
        );
    }
    let saved = s.save_json().unwrap();
    let mut restored = Session::load_json(&saved).unwrap();
    assert_eq!(
        restored.site.hydraulic.bed_remainder_m3,
        s.site.hydraulic.bed_remainder_m3
    );
    let cut = restored.site.dig(position, 0.01, 2., 1.);
    for actual in [cut.sand_m3, cut.soil_m3, cut.gravel_m3] {
        assert!(
            (actual - (0.125 * AREA + 1e-9 / 25.)).abs() < 1e-17,
            "cut lost precision stock: {actual}"
        );
    }
    assert!((cut.gold_g - 1e-9 / 25.).abs() < 1e-24);
    assert_eq!(restored.site.hydraulic.bed_remainder_m3[i], [0.; 4]);
    assert!((0..4).all(|ch| restored.site.bed_class_volume_m3(i, ch) == 0.));
}
#[test]
fn legacy_saves_cannot_hide_new_inventory_and_invalid_subgrid_quantities_are_rejected() {
    let mut s = fixture();
    s.site.deposit(
        Vec3::new(14., 0., 16.),
        Material {
            sand_m3: 1e-9,
            ..Default::default()
        },
    );
    let value: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    assert_eq!(value["version"], 6);
    for version in [3, 4] {
        let mut old = value.clone();
        old["version"] = version.into();
        assert!(Session::load_json(&old.to_string()).is_err());
    }
    for invalid in [-1e-12, 0.01] {
        let mut corrupt = value.clone();
        corrupt["site"]["hydraulic"]["bed_remainder_m3"][64 * WIDTH + 56][0] = invalid.into();
        assert!(Session::load_json(&corrupt.to_string()).is_err());
    }
    let mut old: serde_json::Value = serde_json::from_str(&fixture().save_json().unwrap()).unwrap();
    old["site"]["hydraulic"]
        .as_object_mut()
        .unwrap()
        .remove("bed_remainder_m3");
    for version in [3, 4] {
        old["version"] = version.into();
        let restored = Session::load_json(&old.to_string()).unwrap();
        assert!(restored.site.hydraulic.bed_remainder_m3.is_empty());
    }
}
