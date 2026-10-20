//! Reproducible finite demo: mine real terrain, feed it, then warm the original
//! solver until water and gangue actually reach the outlet. No synthetic yield.
use mining_sim::{session::Controls, storage, Action, Placement, Session, Vec3};
fn main() {
    let path = std::env::args().nth(1).expect("output save path");
    let mut s = Session::new();
    for x in [18.8, 19.8, 20.8] {
        let batch = s.site.dig(Vec3::new(x, 0.0, 22.0), 0.55, 0.55, 0.27);
        assert!(batch.volume() > 0.20);
        let volume = batch.volume();
        assert!(s.sluice.add_to_hopper(batch).is_empty());
        s.delivered_m3 += volume;
    }
    s.player.position = Vec3::new(16.3, s.site.height(16.3, 20.0), 20.0);
    s.player.yaw = -2.223643;
    s.player.pitch = -0.40;
    s.player.third_person = false;
    s.on_action(Action::Place(Placement::suggested(&s.site)))
        .unwrap();
    s.sluice.feed_m3s = 0.0001;
    s.on_action(Action::TogglePump).unwrap();
    s.on_action(Action::ToggleFeed).unwrap();
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    for tick in 0..3000 {
        s.step(Controls::default());
        solver.step(&mut s).unwrap();
        if tick % 240 == 239 {
            println!(
                "DEMO sim={:.1}s water={} returned={:.4}m3 processed={:.3}m3 gangue={:.3}m3",
                s.sim_time,
                s.sluice.water.len(),
                s.sluice.water_returned_m3,
                s.sluice.processed_m3,
                s.sluice.tailings.volume()
            );
        }
        if tick >= 720 && s.sluice.water_returned_m3 > 0.005 && s.sluice.tailings.volume() > 0.0001
        {
            break;
        }
    }
    assert!(s.sluice.water_returned_m3 > 0.005 && !s.sluice.grains.is_empty());
    assert!(s.sluice.hopper.volume() > 0.4);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
    s.message = "Flowing demo. Click to look around; Esc releases the mouse. P water, O feed. Demo / New shift buttons restart.".into();
    storage::save(&s, std::path::Path::new(&path)).unwrap();
    println!(
        "DEMO SAVED {path} at {:.1}s; gold error={:.10}g",
        s.sim_time,
        s.accounted_gold() - s.initial_gold_g
    );
}
