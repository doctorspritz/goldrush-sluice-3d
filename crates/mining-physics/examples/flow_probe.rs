//! Short original-solver flow-through diagnostic, without a rendering window.
use mining_sim::{session::Controls, Action, Placement, Session, Vec3};
fn main() {
    let mut s = Session::new();
    s.player.position = Vec3::new(14.6, 0.35, 15.0);
    s.on_action(Action::Place(Placement::suggested(&s.site)))
        .unwrap();
    s.on_action(Action::TogglePump).unwrap();
    let mut physics = mining_physics::Hydraulics::new().unwrap();
    let started = std::time::Instant::now();
    for tick in 0..2400 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
        if tick % 240 == 239 {
            let mut bins = [0usize; 7];
            for p in &s.sluice.water {
                bins[((p.position.x / 0.5) as usize).min(6)] += 1;
            }
            let max_x = s
                .sluice
                .water
                .iter()
                .map(|p| p.position.x)
                .fold(0.0, f32::max);
            let max_y = s
                .sluice
                .water
                .iter()
                .map(|p| p.position.y)
                .fold(0.0, f32::max);
            let max_v = s
                .sluice
                .water
                .iter()
                .map(|p| p.velocity.length())
                .fold(0.0, f32::max);
            println!("t={:.1} wall={:.1} water={} returned={:.5} mean_vx={:.3} max_v={max_v:.2} max_x={max_x:.2} max_y={max_y:.2} bins={bins:?}",s.sim_time,started.elapsed().as_secs_f64(),s.sluice.water.len(),s.sluice.water_returned_m3,s.sluice.water_speed());
        }
    }
}
