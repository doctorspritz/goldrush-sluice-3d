use mining_sim::{session::Controls, Material, Session, Site, Vec3};

#[test]
fn excavation_and_deposition_preserve_dry_wet_and_thin_water() {
    for depth in [0.0, 0.0005, 0.12] {
        let mut site = Site::new();
        for z in 80..98 {
            for x in 70..90 {
                let i = site.world.idx(x, z);
                site.world.water_surface[i] = site.world.ground_height(x, z) + depth;
            }
        }
        let before = site.water_volume_m3();
        let batch = site.dig(Vec3::new(19.8, 0., 22.), 0.55, 0.55, 0.27);
        assert!(batch.volume() > 0.2);
        assert!(
            (site.water_volume_m3() - before).abs() < 2e-7,
            "dig at {depth}m"
        );
        site.deposit(Vec3::new(19.8, 0., 22.), batch);
        assert!(
            (site.water_volume_m3() - before).abs() < 4e-7,
            "deposit at {depth}m"
        );
    }
}

#[test]
fn water_ledger_survives_save_and_legacy_saves_rebase_without_deleting_water() {
    let mut s = Session::new();
    s.water_balance.external_inflow_m3 = 0.25;
    s.water_balance.boundary_outflow_m3 = 0.12;
    let error = s.water_balance_error_m3();
    let saved = s.save_json().unwrap();
    let restored = Session::load_json(&saved).unwrap();
    assert_eq!(error, restored.water_balance_error_m3());
    let mut old: serde_json::Value = serde_json::from_str(&saved).unwrap();
    old.as_object_mut().unwrap().remove("water_balance");
    let legacy = Session::load_json(&old.to_string()).unwrap();
    assert!(legacy.water_balance.legacy_baseline);
    assert_eq!(legacy.stored_water_m3(), s.stored_water_m3());
    assert_eq!(legacy.water_balance_error_m3(), 0.);
}

#[test]
fn dry_dump_and_idle_player_do_not_create_water() {
    let mut s = Session::new();
    let before = s.stored_water_m3();
    s.site.deposit(
        Vec3::new(20., 0., 20.),
        Material {
            soil_m3: 0.2,
            ..Default::default()
        },
    );
    for _ in 0..120 {
        s.step(Controls::default());
    }
    assert!((s.stored_water_m3() - before).abs() < 1e-7);
}
