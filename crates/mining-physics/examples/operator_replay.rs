use mining_sim::{
    replay::{replay_receipt, OperatorReplay},
    Session,
};
fn main() {
    let mut s = Session::new();
    let mut physics = mining_physics::Hydraulics::new().expect("GPU physics");
    let mut r = OperatorReplay::default();
    let mut last = u32::MAX;
    for _ in 0..25000 {
        let controls = r.controls(&mut s);
        s.step(controls);
        physics.step(&mut s).expect("Original simulation step");
        if r.phase != last {
            println!(
                "phase {} {}: bucket {:.3} m3 / {:.3} g; delivered {:.3}; lip {:?}",
                r.phase,
                r.label(),
                s.machine.bucket.volume(),
                s.machine.bucket.gold_g,
                s.delivered_m3,
                s.machine.arm().lip
            );
            last = r.phase;
        }
        if r.complete || r.error.is_some() {
            break;
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
