use mining_physics::{grain_exits::Snapshot, Hydraulics};
use mining_sim::{session::Controls, Material, Placement, Session, Vec3};
#[test]
#[ignore = "requires a real Metal/Vulkan adapter"]
fn actual_sluice_exits_enter_spatial_tailings_and_continue_after_restore() {
    let mut s = Session::new_operation();
    let baseline = s.accounted_material();
    let batch = s.site.dig(Vec3::new(19., 0., 22.), 0.4, 0.3, 0.08);
    assert!(batch.volume() > 0.02);
    assert!(s.sluice.add_to_hopper(batch).is_empty());
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    s.sluice.feed_on = true;
    let mut h = Hydraulics::new().unwrap();
    h.enable_grain_exit_capture(4096).unwrap();
    let mut snapshot = Snapshot::default();
    let mut observed = Material::default();
    let mut exits = 0;
    let mut saved_airborne = false;
    let mut max_owned_packets = 0;
    for tick in 0..600 {
        s.step(Controls::default());
        let previous = s.tailings.as_ref().unwrap().released_count;
        h.step(&mut s).unwrap_or_else(|e| panic!("tick{tick}: {e}"));
        h.take_grain_exit_snapshot(&mut snapshot).unwrap();
        assert!(
            snapshot.is_complete()
                || (snapshot.status == mining_physics::grain_exits::Status::NoRemovalStage
                    && snapshot.observed == 0)
        );
        let new: Vec<_> = s
            .tailings
            .as_ref()
            .unwrap()
            .packets
            .iter()
            .filter(|g| g.id > previous)
            .collect();
        assert_eq!(new.len(), snapshot.exits().len());
        for (p, e) in new.iter().zip(snapshot.exits()) {
            let expected = mining_sim::tailings::world_grain(&s.sluice, &e.grain).unwrap();
            assert_eq!(
                serde_json::to_value(&p.grain).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
            observed.add(e.grain.material);
            exits += 1;
        }
        assert!(s.tailings.as_ref().unwrap().valid());
        max_owned_packets = max_owned_packets.max(s.tailings.as_ref().unwrap().packets.len());
        if !saved_airborne && !new.is_empty() {
            let json = s.save_json().unwrap();
            let restored = Session::load_json(&json).unwrap();
            assert_eq!(
                serde_json::to_value(&s.tailings).unwrap(),
                serde_json::to_value(&restored.tailings).unwrap()
            );
            s = restored;
            h.restore();
            saved_airborne = true;
        }
        assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
    }
    let t = s.tailings.as_ref().unwrap();
    assert!(exits > 10 && saved_airborne && t.received.volume() > 0.);
    assert!(s.sluice.tailings.is_empty());
    for (a, b) in [
        (observed.soil_m3, t.released.soil_m3),
        (observed.sand_m3, t.released.sand_m3),
        (observed.gravel_m3, t.released.gravel_m3),
        (observed.gold_g, t.released.gold_g),
    ] {
        assert!((a - b).abs() < 1e-12);
    }
    let bulk_error = s.accounted_bulk_material_m3() - baseline.volume();
    let mass_error = s.accounted_dry_mass_kg() - baseline.mass_kg();
    assert!(bulk_error.abs() < 1e-5);
    assert!(mass_error.abs() < 0.02);
    assert!(s.water_balance_error_m3().abs() < 2e-5);
    println!(
        "{}",
        serde_json::json!({"ticks":600,"actual_exit_records":exits,"maximum_owned_packets":max_owned_packets,"released":t.released,"received":t.received,"exported":t.exported,"bulk_error_m3":bulk_error,"dry_mass_error_kg":mass_error,"gold_error_g":s.accounted_gold()-s.initial_gold_g,"water_error_m3":s.water_balance_error_m3(),"live_airborne_restored":saved_airborne,"solver_steps":s.sluice.solver_steps})
    );
    if let Some(path) = std::env::var_os("SLUICE_TAILINGS_OUTPUT") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("modern.save.json"), s.save_json().unwrap()).unwrap();
        let t = s.tailings.take().unwrap();
        assert!(t.exported.is_empty());
        s.sluice.tailings.add(t.active_material());
        let old = s.save_json().unwrap();
        assert!(Session::load_json(&old).is_ok());
        std::fs::write(path.join("legacy-render.save.json"), old).unwrap();
    }
}
