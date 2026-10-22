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
#[test]
fn clean_river_source_dilutes_all_suspension_without_creating_solid() {
    let mut s = mining_sim::Session::new();
    let n = mining_sim::site::WIDTH.pow(2);
    s.site.hydraulic.suspended_sediment = vec![0.01; n];
    s.site.hydraulic.suspended_overburden = vec![0.02; n];
    s.site.hydraulic.suspended_gravel = vec![0.03; n];
    s.site.hydraulic.suspended_paydirt = vec![0.04; n];
    s.site.hydraulic.suspended_gold_g = vec![0.1; n];
    let mass = |site: &mining_sim::Site| {
        [
            &site.hydraulic.suspended_sediment,
            &site.hydraulic.suspended_overburden,
            &site.hydraulic.suspended_gravel,
            &site.hydraulic.suspended_paydirt,
        ]
        .map(|field| {
            field
                .iter()
                .enumerate()
                .map(|(i, c)| *c as f64 * site.world.water_depth(i % 128, i / 128) as f64 * 0.0625)
                .sum::<f64>()
        })
    };
    let before = mass(&s.site);
    let water = s.site.water_volume_m3();
    let gold = s.site.remaining_gold();
    let mut added = 0.;
    for _ in 0..100 {
        added += s.site.add_clean_inflow(
            mining_sim::Vec3::new(mining_sim::Site::river_center(30.5), 0., 30.5),
            0.12,
            0.8,
            0.05,
        );
    }
    assert!((added - 0.6).abs() < 1e-5);
    assert!((s.site.water_volume_m3() - water - added).abs() < 1e-10);
    for (a, b) in before.into_iter().zip(mass(&s.site)) {
        assert!((a - b).abs() < 1e-7, "source created solids: {a} -> {b}");
    }
    assert_eq!(gold, s.site.remaining_gold());
    assert!(s.site.hydraulic.impact_dirty);
}
