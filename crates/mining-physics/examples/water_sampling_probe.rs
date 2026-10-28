//! Reproducible finite full-width plant for marker-quadrature comparisons.
//! Writes one requested snapshot; callers should remove it after validation.
use mining_sim::{session::Controls, sluice::WaterSampling, Action, Placement, Session, Vec3};
use std::{path::Path, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let sampling = WaterSampling::try_from(args[1].parse::<u32>().unwrap()).unwrap();
    let ticks = args.get(3).map_or(480, |v| v.parse::<u32>().unwrap());
    let mut s = Session::new_operation();
    s.sluice.set_water_sampling(sampling).unwrap();
    let pay = s.site.dig(Vec3::new(19., 0., 22.), 0.44, 0.3, 0.08);
    assert!(s.sluice.add_to_hopper(pay).is_empty());
    s.player.position = Vec3::new(14.6, 0.35, 15.);
    s.on_action(Action::Place(Placement::suggested(&s.site)))
        .unwrap();
    s.enable_screen().unwrap();
    s.plant.as_mut().unwrap().screen.motor_on = true;
    s.on_action(Action::TogglePump).unwrap();
    s.on_action(Action::ToggleFeed).unwrap();
    let initial_mass = s.accounted_dry_mass_kg();
    let initial_gold = s.accounted_gold();
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    if sampling.is_reference() {
        solver = solver.with_conservative_phases();
    }
    let started = Instant::now();
    for _ in 0..ticks {
        s.step(Controls::default());
        assert!(!s.paused, "{}", s.message);
        solver.step(&mut s).unwrap();
    }
    let wall = started.elapsed().as_secs_f64();
    assert!((s.accounted_dry_mass_kg() - initial_mass).abs() < 0.0001);
    assert!((s.accounted_gold() - initial_gold).abs() < 1e-7);
    assert!(s.water_balance_error_m3().abs() < 2e-6);
    mining_sim::storage::save(&s, Path::new(&format!("{}.save.json", args[2]))).unwrap();
    let report = serde_json::json!({"markers_per_cell":sampling.markers_per_cell(),"ticks":ticks,
        "physical_seconds":s.sim_time,"wall_seconds":wall,"physical_seconds_per_wall_second":s.sim_time/wall,
        "water_markers":s.sluice.water.len(),"marker_m3":s.sluice.water_particle_m3(),
        "water_depth_m":s.sluice.water_depth(),"downstream_ms":s.sluice.water_speed(),
        "water_returned_m3":s.sluice.water_returned_m3,"water_error_m3":s.water_balance_error_m3(),
        "gold_error_g":s.accounted_gold()-initial_gold,"grains":s.sluice.grains.len(),
        "dry_mass_error_kg":s.accounted_dry_mass_kg()-initial_mass,"headless":true,
        "riffle_observation":mining_physics::calibration::observe(&s),
        "slurry":mining_sim::slurry::diagnostics(&s.sluice)});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("{report}");
}
