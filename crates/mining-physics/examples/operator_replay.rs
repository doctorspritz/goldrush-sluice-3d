use mining_sim::{
    replay::{replay_receipt, OperatorReplay},
    Session,
};
use std::time::Instant;
fn main() {
    let mut s = Session::new();
    let mut physics = mining_physics::Hydraulics::new().expect("GPU physics");
    let mut r = OperatorReplay::default();
    let mut last = u32::MAX;
    let started = Instant::now();
    let wall_seconds: f64 = std::env::var("SLUICE_REPLAY_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300.0);
    let mut saved = false;
    for tick in 0..25000 {
        let controls = r.controls(&mut s);
        s.step(controls);
        physics.step(&mut s).expect("Original simulation step");
        if r.phase != last || tick % 600 == 0 {
            println!(
                "phase {} {} tick={tick} sim={:.1}s wall={:.1}s: bucket {:.3} m3 / {:.3} g; delivered {:.3}; processed {:.3}; water={}; grains={}; active_gold={:.3}; tailings_gold={:.3}; collected={:.3}",
                r.phase, r.label(), s.sim_time, started.elapsed().as_secs_f64(),
                s.machine.bucket.volume(), s.machine.bucket.gold_g, s.delivered_m3,
                s.sluice.processed_m3, s.sluice.water.len(), s.sluice.grains.len(),
                s.sluice.active_material().gold_g, s.sluice.tailings.gold_g, s.sluice.concentrate.gold_g
            );
            let max_x = s
                .sluice
                .water
                .iter()
                .map(|p| p.position.x)
                .fold(0.0, f32::max);
            let max_v = s
                .sluice
                .water
                .iter()
                .map(|p| p.velocity.length())
                .fold(0.0, f32::max);
            println!("  fluid front={max_x:.2}m, max speed={max_v:.2}m/s, returned={:.5}m3; grain x={:?}",s.sluice.water_returned_m3,
                s.sluice.grains.iter().map(|g| g.position.x).reduce(f32::max));
            last = r.phase;
        }
        if r.phase == 20 && s.sluice.processed_m3 > 0.04 && !saved {
            let before = s.sluice.water.len();
            s = Session::load_json(&s.save_json().expect("Save live shift"))
                .expect("Restore live shift");
            assert_eq!(before, s.sluice.water.len());
            physics.restore();
            saved = true;
            println!("Restored live original solver state: {before} water particles");
        }
        if r.complete || r.error.is_some() {
            break;
        }
        if started.elapsed().as_secs_f64() > wall_seconds {
            eprintln!(
                "Replay stopped at its {wall_seconds:.0}-second wall-clock bound; phase {}",
                r.phase
            );
            std::process::exit(2);
        }
    }
    match replay_receipt(&s, &r) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
