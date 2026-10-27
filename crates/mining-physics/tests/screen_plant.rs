use mining_physics::Hydraulics;
use mining_sim::{session::Controls, Action, Placement, Session, Vec3};

#[test]
#[ignore = "requires an actual Metal/Vulkan adapter"]
fn finite_dug_feed_crosses_actual_screen_and_continues_in_original_wet_sluice() {
    let mut s = Session::new_operation();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.enable_screen().unwrap();
    // Initial operator pose for the separate native K/L/M maintenance replay.
    // No runtime teleport or material injection is part of that replay.
    s.player.position = mining_sim::Sluice::hopper_position(&s.site) + Vec3::X * 1.5;
    let at = Vec3::new(19., 0., 20.);
    let spoil = s.site.dig(at, 0.45, 0.10, 0.20);
    assert!(
        spoil.gravel_m3 < 1e-12 && spoil.gold_g < 1e-10,
        "strip the real overburden first"
    );
    s.site.try_deposit(at + Vec3::X * 2., spoil).unwrap();
    let m = s.site.dig(at, 0.45, 0.15, 0.06);
    assert!(
        m.gravel_m3 > 0. && m.gold_g > 0.,
        "actual finite pay/gravel cohort required"
    );
    assert!(m.volume() > 0.01);
    assert!(s.sluice.add_to_hopper(m).is_empty());
    s.sluice.feed_on = true;
    s.sluice.pump_on = true;
    s.sluice.feed_m3s = 0.0003;
    s.plant.as_mut().unwrap().screen.motor_on = true;
    let baseline = s.accounted_material();
    let mut h = Hydraulics::new().unwrap();
    let mut full_state_matches = 0usize;
    let mut restored = false;
    let mut saved_fixture = None;
    let mut maximum_deck = 0;
    for tick in 0..480 {
        let previous = s.plant.as_ref().unwrap().screen.undersize.clone();
        let before = s.sluice.grains.len();
        s.step(Controls::default());
        assert!(!s.paused, "tick {tick}: {}", s.message);
        let new = &s.sluice.grains[before..];
        if !new.is_empty() {
            assert_eq!(
                new.len(),
                previous.len(),
                "exclusive finite whole-batch screen admission"
            );
            for (grid, world) in new.iter().zip(&previous) {
                let returned = mining_sim::tailings::world_grain(&s.sluice, grid).unwrap();
                assert!((returned.position - world.position).length() < 3e-6);
                assert!((returned.velocity - world.velocity).length() < 2e-6);
                assert!((returned.affine - world.affine)
                    .to_cols_array()
                    .iter()
                    .all(|v| v.abs() < 2e-6));
                assert!(returned.rotation.dot(world.rotation).abs() > 1. - 2e-6);
                assert!((returned.angular_velocity - world.angular_velocity).length() < 2e-5);
                assert_eq!(returned.diameter_m, world.diameter_m);
                let mut dry = world.material;
                dry.pore_water_m3 = 0.;
                assert_eq!(returned.material, dry);
                full_state_matches += 1;
            }
        }
        h.step(&mut s).unwrap();
        let p = s.plant.as_ref().unwrap();
        assert!(p.valid());
        maximum_deck = maximum_deck.max(p.screen.particles.len());
        if !restored && tick > 120 && full_state_matches > 10 {
            let json = s.save_json().unwrap();
            let loaded = Session::load_json(&json).unwrap();
            assert_eq!(
                serde_json::to_value(&loaded.plant).unwrap(),
                serde_json::to_value(&s.plant).unwrap()
            );
            saved_fixture = Some(json);
            s = loaded;
            h.restore();
            restored = true;
        }
        assert!(
            (s.accounted_gold() - s.initial_gold_g).abs() < 1e-7,
            "gold tick {tick}"
        );
        assert!(
            (s.accounted_material().volume() - baseline.volume()).abs() < 1e-5,
            "bulk tick {tick}"
        );
        assert!(s.water_balance_error_m3().abs() < 2e-5, "water tick {tick}");
    }
    let p = s.plant.as_ref().unwrap();
    assert!(restored && full_state_matches > 30 && p.screen.admitted.volume() > 0.001);
    assert!(
        p.screen.delivered.volume() > 0.0001 && !p.screen.oversize.is_empty(),
        "real underflow and caught oversize required"
    );
    let before_clear = s.accounted_material();
    let rejects = p.material();
    let hopper = mining_sim::Sluice::hopper_position(&s.site);
    s.player.position = hopper + Vec3::X * 1.5;
    s.on_action(Action::ToggleScreen).unwrap();
    s.on_action(Action::ClearScreen).unwrap();
    assert_eq!(s.player.hand, rejects);
    assert!(s.plant.as_ref().unwrap().screen.oversize.is_empty());
    assert!((s.accounted_material().volume() - before_clear.volume()).abs() < 1e-10);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
    println!(
        "{}",
        serde_json::json!({"ticks":480,"full_state_underflow_matches":full_state_matches,"maximum_deck_particles":maximum_deck,"live_plant_restored":restored,"screen_admitted":s.plant.as_ref().unwrap().screen.admitted,"screen_delivered":s.plant.as_ref().unwrap().screen.delivered,"collected_rejects":rejects,"gold_error_g":s.accounted_gold()-s.initial_gold_g,"bulk_error_m3":s.accounted_material().volume()-baseline.volume(),"water_error_m3":s.water_balance_error_m3(),"limits":"Dilute physical-diameter point parcels in a dry moving screen; no dense packing/blinding or screen-water hydraulics. Original wet sluice solver retained."})
    );
    if let Some(path) = std::env::var_os("SLUICE_SCREEN_OUTPUT") {
        let dir = std::path::PathBuf::from(path);
        std::fs::create_dir(&dir).unwrap();
        let modern = saved_fixture.unwrap();
        std::fs::write(dir.join("modern.save.json"), &modern).unwrap();
        let mut old = Session::load_json(&modern).unwrap();
        let plant = old.plant.take().unwrap();
        let remainder = old.sluice.add_to_hopper(plant.material());
        assert!(remainder.is_empty());
        old.sluice.feed_water_credit_m3 += plant.screen.free_water_m3 + plant.pending_moisture_m3;
        let legacy = old.save_json().unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&legacy).unwrap()["version"],
            13
        );
        assert!(Session::load_json(&legacy).is_ok());
        std::fs::write(dir.join("legacy-render.save.json"), legacy).unwrap();
    }
}
