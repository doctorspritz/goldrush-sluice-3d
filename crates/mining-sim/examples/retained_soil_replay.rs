//! Fixed-command mechanical comparison. No Session fuel, GPU water or terrain ticks.
use mining_sim::{
    articulation::{self, retained::RetainedSoil, Environment},
    bucket,
    soil::ActiveSoil,
    Excavator, MachineInput, Session, Vec3,
};
use serde::Deserialize;
use std::{error::Error, fs, io::Read, path::Path, time::Instant};
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    drive: f32,
    steer: f32,
    swing: f32,
    boom: f32,
    stick: f32,
    curl: f32,
    cut: bool,
    dump: bool,
}
impl From<Input> for MachineInput {
    fn from(i: Input) -> Self {
        Self {
            drive: i.drive,
            steer: i.steer,
            swing: i.swing,
            boom: i.boom,
            stick: i.stick,
            curl: i.curl,
            cut: i.cut,
            dump: i.dump,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tape {
    version: u32,
    initial_session_json: String,
    commands: Vec<Input>,
}
enum State {
    Cold(Box<(Excavator, ActiveSoil)>),
    Retained(Box<RetainedSoil>),
}
impl State {
    fn views(&self) -> (&Excavator, &ActiveSoil) {
        match self {
            Self::Cold(s) => (&s.0, &s.1),
            Self::Retained(s) => (s.machine(), s.soil()),
        }
    }
    fn step(&mut self, input: MachineInput, h: f32, env: &Environment<'_>) -> Result<(), String> {
        match self {
            Self::Cold(s) => articulation::step_soil(&mut s.0, input, h, env, &mut s.1),
            Self::Retained(s) => s.step(input, h, env),
        }
    }
    fn snapshot(&self, out: &Path, tick: usize) -> Result<(), Box<dyn Error>> {
        let (machine, soil) = self.views();
        fs::write(
            out.join(format!("tick-{tick:04}.json")),
            serde_json::to_vec(&serde_json::json!({"machine":machine,"soil":soil}))?,
        )?;
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let usage="retained_soil_replay cold|retained|restore OUTPUT_DIR TICKS HZ < commands.json\nMechanical comparison only; fresh directories,1..3600 command ticks,240/480/960Hz.";
    if args.len() == 2 && args[1] == "--help" {
        println!("{usage}");
        return Ok(());
    }
    if args.len() != 5 || !matches!(args[1].as_str(), "cold" | "retained" | "restore") {
        return Err(usage.into());
    }
    let ticks: usize = args[3].parse()?;
    let hz: u32 = args[4].parse()?;
    if !(1..=3600).contains(&ticks) || ![240, 480, 960].contains(&hz) {
        return Err(usage.into());
    }
    let mut data = String::new();
    std::io::stdin()
        .take(64 * 1024 * 1024 + 1)
        .read_to_string(&mut data)?;
    if data.len() > 64 * 1024 * 1024 {
        return Err("command tape too large".into());
    }
    let tape: Tape = serde_json::from_str(&data)?;
    if tape.version != 1
        || tape.commands.len() < ticks
        || tape.commands.len() > 3600
        || tape.commands.iter().any(|c| {
            [c.drive, c.steer, c.swing, c.boom, c.stick, c.curl]
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 1.)
        })
    {
        return Err("invalid bounded command tape".into());
    }
    let mut session = Session::load_json(&tape.initial_session_json)?;
    let mut machine = session.machine.clone();
    machine.tracks.mechanical_hz = hz as u16;
    let soil = session.active_soil.take().ok_or("active bank required")?;
    let initial_material = soil.material();
    let clods = soil.clods.len();
    let obstacles = session.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&session.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let out = Path::new(&args[2]);
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut state = if args[1] == "cold" {
        State::Cold(Box::new((machine, soil)))
    } else {
        State::Retained(Box::new(RetainedSoil::new(machine, soil, &env)?))
    };
    let startup = started.elapsed().as_secs_f64();
    state.snapshot(out, 0)?;
    let mut step_wall = 0.;
    let mut restore_wall = 0.;
    let mut max_held = 0f64;
    let mut maximum_residual = f64::NEG_INFINITY;
    let mut contact_steps = 0;
    let mut restores = 0;
    for (tick, input) in tape.commands.iter().take(ticks).enumerate() {
        let timer = Instant::now();
        for _ in 0..hz / 60 {
            state.step((*input).into(), 1. / hz as f32, &env)?;
            let (_, soil) = state.views();
            maximum_residual = maximum_residual.max(soil.feedback.coupled_passivity_residual_j);
            contact_steps += usize::from(soil.feedback.bucket_contacts > 0);
        }
        step_wall += timer.elapsed().as_secs_f64();
        let (m, s) = state.views();
        max_held = max_held.max(bucket::estimated_soil_contents(m, s).volume());
        if (tick + 1) % 60 == 0 || tick + 1 == ticks {
            if args[1] == "restore" {
                let timer = Instant::now();
                if let State::Retained(s) = &mut state {
                    **s = RetainedSoil::decode(&s.encode()?, &env)?;
                    restores += 1;
                }
                restore_wall += timer.elapsed().as_secs_f64();
            }
            state.snapshot(out, tick + 1)?;
            eprintln!(
                "tick={} step_wall={step_wall:.3} max_held={max_held:.9}",
                tick + 1
            );
        }
    }
    let (machine, soil) = state.views();
    let final_material = soil.material();
    if serde_json::to_value(initial_material)? != serde_json::to_value(final_material)?
        || !machine.valid()
        || !soil.valid()
    {
        return Err("mechanical replay changed finite material or became invalid".into());
    }
    let metrics = serde_json::json!({"completed":true,"scope":"Fixed commands, CPU mechanical steps only; no fuel/GPU water/terrain evolution; actual equipment collision boxes included",
 "mode":args[1],"ticks":ticks,"mechanical_hz":hz,"mechanical_steps":ticks*(hz as usize/60),"simulated_seconds":ticks as f64/60.,"startup_seconds":startup,"step_wall_seconds":step_wall,"restore_wall_seconds":restore_wall,"total_wall_seconds":started.elapsed().as_secs_f64(),
 "clods":clods,"initial_material":initial_material,"final_material":final_material,"max_held_estimate_m3":max_held,"final_held_estimate_m3":bucket::estimated_soil_contents(machine,soil).volume(),"contact_steps":contact_steps,"maximum_coupled_passivity_residual_j":maximum_residual,"restores":restores});
    fs::write(
        out.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics)?,
    )?;
    if let State::Retained(s) = &state {
        fs::write(out.join("final-boundary.bin"), s.encode()?)?;
    }
    println!("{metrics}");
    Ok(())
}
