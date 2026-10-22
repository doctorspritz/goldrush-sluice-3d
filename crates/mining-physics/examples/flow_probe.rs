//! Bounded real-solver probe, separate render and physical-time measurements.
use mining_sim::{session::Controls, storage};
use std::{path::Path, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut s = storage::load(Path::new(&args[1])).unwrap();
    let ticks: u32 = args[2].parse().unwrap();
    let output = &args[3];
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    let warmup: u32 = args.get(4).and_then(|n| n.parse().ok()).unwrap_or(0);
    for _ in 0..warmup {
        s.step(Controls::default());
        solver.step(&mut s).unwrap();
    }
    let before = s.sim_time;
    let start = Instant::now();
    let mut phases = [0.0; 7];
    let mut gpu = std::collections::BTreeMap::<String, f64>::new();
    let mut gpu_device = std::collections::BTreeMap::<String, f64>::new();
    let mut capacity_blocked_ticks = 0u32;
    let mut grain_limit_ticks = 0u32;
    let mut nominal_water_m3 = 0.;
    let mut admitted_water_m3 = 0.;
    for _ in 0..ticks {
        if s.sluice.pump_on && s.sluice.placement.is_some() {
            nominal_water_m3 += (s.sluice.flow_m3s * mining_sim::FIXED_DT) as f64;
            capacity_blocked_ticks += u32::from(!s.sluice.intake_has_room());
            grain_limit_ticks +=
                u32::from(s.sluice.grains.len() + 5 > mining_sim::sluice::MAX_GRAINS);
        }
        s.step(Controls::default());
        admitted_water_m3 += s.sluice.inlet.last_water_m3;
        solver.step(&mut s).unwrap();
        let t = solver.timings;
        for (stage, ms) in &solver.gpu_device_stages {
            *gpu_device.entry(stage.clone()).or_default() += ms / ticks as f64;
        }
        for (key, ms) in &solver.gpu_stages {
            *gpu.entry(key.clone()).or_default() += ms / ticks as f64;
        }
        for (sum, value) in phases.iter_mut().zip([
            t.prepare_ms,
            t.fluid_ms,
            t.dem_ms,
            t.finish_ms,
            t.terrain_ms,
            t.dem_sdf_ms,
            t.dem_pairs_ms,
        ]) {
            *sum += value;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    for value in &mut phases {
        *value /= ticks as f64;
    }
    let w = &s.sluice.water;
    let water = s.sluice.accounted_water_m3();
    let data = serde_json::json!({"admission":{"capacity_blocked_ticks":capacity_blocked_ticks,"grain_limit_ticks":grain_limit_ticks,"nominal_water_m3":nominal_water_m3,"admitted_water_m3":admitted_water_m3},"warmup_ticks":warmup,"serialized_stage_ms":gpu,"gpu_device_stage_ms":gpu_device,"simulated_seconds":s.sim_time-before,"wall_seconds":elapsed,"simulation_wall_ratio":(s.sim_time-before)/elapsed,"timings_ms":{"prepare":phases[0],"fluid_and_readback":phases[1],"dem":phases[2],"dem_sdf":phases[5],"dem_pairs":phases[6],"finish":phases[3],"terrain":phases[4]},"water_particles":w.len(),"mean_downstream_ms":s.sluice.water_speed(),"backflow_fraction":w.iter().filter(|p|p.velocity.x < -0.05).count() as f64/w.len().max(1) as f64,"water_depth_m":s.sluice.water_depth(),"grains":s.sluice.grains.len(),"water_balance_error_m3":s.sluice.water_used_m3-water,"gold_balance_error_g":s.accounted_gold()-s.initial_gold_g});
    std::fs::write(output, serde_json::to_vec_pretty(&data).unwrap()).unwrap();
    storage::save(&s, Path::new(&format!("{output}.save.json"))).unwrap();
    println!("{data}");
}
