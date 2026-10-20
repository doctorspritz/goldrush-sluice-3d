//! Bounded real-solver probe, separate render and physical-time measurements.
use mining_sim::{session::Controls, storage};
use std::{path::Path, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut s = storage::load(Path::new(&args[1])).unwrap();
    let ticks: u32 = args[2].parse().unwrap();
    let output = &args[3];
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    let before = s.sim_time;
    let start = Instant::now();
    let mut phases = [0.0; 5];
    for _ in 0..ticks {
        s.step(Controls::default());
        solver.step(&mut s).unwrap();
        let t = solver.timings;
        for (sum, value) in phases.iter_mut().zip([
            t.prepare_ms,
            t.fluid_ms,
            t.dem_ms,
            t.finish_ms,
            t.terrain_ms,
        ]) {
            *sum += value;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    for value in &mut phases {
        *value /= ticks as f64;
    }
    let w = &s.sluice.water;
    let water = w.len() as f64 * s.sluice.profile.particle_m3()
        + s.sluice.water_returned_m3
        + s.sluice.water_credit_m3;
    let data = serde_json::json!({"simulated_seconds":s.sim_time-before,"wall_seconds":elapsed,"simulation_wall_ratio":(s.sim_time-before)/elapsed,"timings_ms":{"prepare":phases[0],"fluid_and_readback":phases[1],"dem":phases[2],"finish":phases[3],"terrain":phases[4]},"water_particles":w.len(),"mean_downstream_ms":s.sluice.water_speed(),"backflow_fraction":w.iter().filter(|p|p.velocity.x < -0.05).count() as f64/w.len().max(1) as f64,"water_depth_m":s.sluice.water_depth(),"grains":s.sluice.grains.len(),"water_balance_error_m3":s.sluice.water_used_m3-water,"gold_balance_error_g":s.accounted_gold()-s.initial_gold_g});
    std::fs::write(output, serde_json::to_vec_pretty(&data).unwrap()).unwrap();
    storage::save(&s, Path::new(&format!("{output}.save.json"))).unwrap();
    println!("{data}");
}
