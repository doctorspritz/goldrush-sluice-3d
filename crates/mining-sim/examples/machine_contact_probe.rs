//! Actual command replay with machine/contact evidence, without GPU water.
use mining_sim::{replay::OperatorReplay, Session};
fn main() {
    let path = std::env::args().nth(1).expect("output.json");
    let mut session = Session::new();
    let mut replay = OperatorReplay::default();
    let mut frames = Vec::new();
    let mut previous = u32::MAX;
    for tick in 0..30_000 {
        let controls = replay.controls(&mut session);
        session.step(controls);
        if replay.phase != previous || tick % 600 == 0 {
            let a = session.machine.arm();
            let f = serde_json::json!({"phase":replay.phase,"tick":tick,"sim_s":session.sim_time,
                "machine":session.machine,"mouth":a.wrist+a.forward*0.35-a.up*0.1,
                "hopper_top":mining_sim::Sluice::hopper_top(&session.site),
                "delivered_m3":session.delivered_m3,"mined_m3":session.site.excavated.volume(),
                "falling":session.falling.len()});
            eprintln!("phase={} t={:.2} pos={:?} tilt={:.2} arm={:.3}/{:.3}/{:.3} mouth={:?} hopperY={} bucket={:.5} delivered={:.5} limit={:?}",
                replay.phase,session.sim_time,session.machine.position,session.machine.tracks.feedback.tilt_deg,
                session.machine.boom,session.machine.stick,session.machine.curl,a.wrist+a.forward*0.35-a.up*0.1,
                mining_sim::Sluice::hopper_top(&session.site),session.machine.bucket.volume(),session.delivered_m3,session.machine.digging.reason);
            frames.push(f);
            previous = replay.phase;
        }
        if replay.error.is_some() || replay.phase >= 15 {
            break;
        }
    }
    let passed = replay.phase == 15 && replay.error.is_none() && session.delivered_m3 >= 0.4;
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&serde_json::json!({"passed":passed,"frames":frames})).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{path}.save.json"), session.save_json().unwrap()).unwrap();
    if !passed {
        std::process::exit(1);
    }
}
