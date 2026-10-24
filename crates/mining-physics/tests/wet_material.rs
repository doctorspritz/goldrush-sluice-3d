//! Moisture must reach physical FLIP particles, rather than a visual water tag.
use mining_physics::Hydraulics;
use mining_sim::{session::Controls, water_balance::WaterBalance, Material, Placement, Session};
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --nocapture"]
fn wet_feed_releases_its_water_into_real_flip_and_survives_restore() {
    let mut s = Session::new();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    let mut wet = Material {
        soil_m3: 0.001,
        ..Default::default()
    };
    wet.pore_water_m3 = wet.pore_volume_m3() * 0.8;
    s.sluice.hopper = wet;
    s.sluice.pump_on = true;
    s.sluice.feed_on = true;
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    let initial = s.stored_water_m3();
    for _ in 0..120 {
        s.sluice.feed(mining_sim::FIXED_DT, true);
    }
    let released = wet.pore_water_m3 - s.sluice.hopper.pore_water_m3;
    assert!(released > 1e-6);
    assert!((released - s.sluice.feed_water_credit_m3).abs() < 1e-12);
    s.sluice.pump_on = false;
    s.sluice.feed_on = false;
    assert!((s.stored_water_m3() - initial).abs() < 1e-12);
    let mut solver = Hydraulics::new().unwrap();
    let mut max_actual_particles = 0;
    for tick in 0..90 {
        s.step(Controls::default());
        solver.step(&mut s).unwrap();
        max_actual_particles = max_actual_particles.max(s.sluice.water.len());
        if tick == 45 {
            let moisture = s.site.soil_water.stored_m3();
            let free = s.sluice.stored_water_m3();
            let json = s.save_json().unwrap();
            s = Session::load_json(&json).unwrap();
            solver.restore();
            assert_eq!(s.site.soil_water.stored_m3(), moisture);
            assert_eq!(s.sluice.stored_water_m3(), free);
        }
    }
    assert!(
        max_actual_particles > 20,
        "moisture never reached actual solver particles"
    );
    assert!((s.sluice.accounted_water_m3() - released).abs() < 1e-12);
    assert!(s.water_balance_error_m3().abs() < 1e-4);
    assert!(s
        .sluice
        .grains
        .iter()
        .all(|g| g.material.pore_water_m3 == 0.));
    println!(
        "WET FEED: {}",
        serde_json::json!({"released_water_m3":released,"max_actual_flip_particles":max_actual_particles,"sluice_water_error_m3":s.sluice.accounted_water_m3()-released,"whole_water_error_m3":s.water_balance_error_m3(),"source":"only hopper moisture; pump stopped before physical steps","restored_during_flow":true})
    );
}
