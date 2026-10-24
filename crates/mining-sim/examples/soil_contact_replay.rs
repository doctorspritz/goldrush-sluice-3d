//! Bounded CPU soil command recording/replay; no GPU terrain/water updates.
use mining_sim::{
    bucket, replay::soil_trial_controls, session::Controls, ControlMode, MachineInput, Session,
};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, io::Read, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_TICKS: usize = 3600;

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Command {
    drive: f32,
    steer: f32,
    swing: f32,
    boom: f32,
    stick: f32,
    curl: f32,
    cut: bool,
    dump: bool,
}
impl From<MachineInput> for Command {
    fn from(i: MachineInput) -> Self {
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
impl From<Command> for MachineInput {
    fn from(i: Command) -> Self {
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
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tape {
    version: u32,
    initial_session_json: String,
    commands: Vec<Command>,
}
fn write_tape(out: &Path, tape: &Tape) -> Result<()> {
    fs::write(out.join("commands.json"), serde_json::to_vec(tape)?)?;
    Ok(())
}
fn snapshot(out: &Path, tick: usize, session: &Session) -> Result<()> {
    fs::write(
        out.join(format!("tick-{tick:04}.json")),
        session.save_json()?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let usage = "soil_contact_replay record idle|stroke OUTPUT_DIR TICKS\nsoil_contact_replay replay TAPE_JSON OUTPUT_DIR\nFresh output directories only; 1..3600 ticks at the session's fixed 60 Hz control clock.";
    if args.len() == 2 && args[1] == "--help" {
        println!("{usage}");
        return Ok(());
    }
    let (mut session, mut tape, out, guided, replaying, ticks) =
        match args.get(1).map(String::as_str) {
            Some("record") if args.len() == 5 && matches!(args[2].as_str(), "idle" | "stroke") => {
                let ticks: usize = args[4].parse()?;
                let mut s = Session::new();
                s.prepare_soil_trial()?;
                s.player.mode = ControlMode::Excavator;
                let tape = Tape {
                    version: 1,
                    initial_session_json: s.save_json()?,
                    commands: Vec::new(),
                };
                (
                    s,
                    tape,
                    Path::new(&args[3]),
                    args[2] == "stroke",
                    false,
                    ticks,
                )
            }
            Some("replay") if args.len() == 4 => {
                let bytes = if args[2] == "-" {
                    let mut bytes = Vec::new();
                    std::io::stdin().read_to_end(&mut bytes)?;
                    bytes
                } else {
                    fs::read(&args[2])?
                };
                let tape: Tape = serde_json::from_slice(&bytes)?;
                if tape.version != 1 {
                    return Err("unsupported command tape version".into());
                }
                for c in &tape.commands {
                    if [c.drive, c.steer, c.swing, c.boom, c.stick, c.curl]
                        .iter()
                        .any(|x| !x.is_finite() || x.abs() > 1.)
                    {
                        return Err("invalid command axes".into());
                    }
                }
                let s = Session::load_json(&tape.initial_session_json)?;
                let ticks = tape.commands.len();
                (s, tape, Path::new(&args[3]), false, true, ticks)
            }
            _ => return Err(usage.into()),
        };
    if !(1..=MAX_TICKS).contains(&ticks) {
        return Err("ticks must be 1..3600".into());
    }
    if session.paused
        || session.active_soil.is_none()
        || session.player.mode != ControlMode::Excavator
    {
        return Err("replay needs active, unpaused excavator soil".into());
    }
    // Never overwrite an earlier experiment, including an interrupted one.
    fs::create_dir(out)?;
    snapshot(out, 0, &session)?;
    write_tape(out, &tape)?;
    let start_sim_time = session.sim_time;
    let initial_material = session.active_soil.as_ref().unwrap().material();
    let mut step_wall = 0.;
    let mut max_held = 0f64;
    let started = Instant::now();
    for tick in 0..ticks {
        let input = if replaying {
            tape.commands[tick].into()
        } else {
            let input = if guided {
                soil_trial_controls(&session, session.sim_time)
            } else {
                MachineInput::default()
            };
            tape.commands.push(input.into());
            input
        };
        let timer = Instant::now();
        session.step(Controls {
            machine: input,
            ..Default::default()
        });
        step_wall += timer.elapsed().as_secs_f64();
        if session.paused {
            write_tape(out, &tape)?;
            return Err(format!("paused at tick {}: {}", tick + 1, session.message).into());
        }
        let held = bucket::estimated_soil_contents(
            &session.machine,
            session.active_soil.as_ref().unwrap(),
        )
        .volume();
        max_held = max_held.max(held);
        if (tick + 1) % 60 == 0 || tick + 1 == ticks {
            snapshot(out, tick + 1, &session)?;
            if !replaying {
                write_tape(out, &tape)?;
            }
            eprintln!(
                "tick={} sim={} step_wall={} held_m3={}",
                tick + 1,
                session.sim_time,
                step_wall,
                held
            );
        }
    }
    let soil = session.active_soil.as_ref().unwrap();
    let metrics = serde_json::json!({"completed":true,"scope":"CPU mechanics; no GPU terrain/water updates",
        "ticks":ticks,"start_sim_time":start_sim_time,"sim_time":session.sim_time,"simulated_seconds":session.sim_time-start_sim_time,"mechanical_hz":session.machine.tracks.mechanical_hz,
        "step_wall_seconds":step_wall,"total_wall_seconds":started.elapsed().as_secs_f64(),
        "clods":soil.clods.len(),"initial_material":initial_material,"final_material":soil.material(),
        "max_held_estimate_m3":max_held,"final_held_estimate_m3":bucket::estimated_soil_contents(&session.machine,soil).volume()});
    fs::write(
        out.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics)?,
    )?;
    println!("{metrics}");
    Ok(())
}
