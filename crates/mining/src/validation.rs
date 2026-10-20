//! Rendered operator replay and benchmark. Screenshots use the real game camera;
//! all mining and transfers run through the same domain controls as interactive play.
use crate::app::{save_path, Game};
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use mining_physics::worker::Command;
use mining_sim::replay::{replay_receipt, OperatorReplay};
use std::{collections::HashSet, path::PathBuf, time::Instant};

#[derive(Resource)]
pub struct Validation {
    pub directory: Option<PathBuf>,
    capture: Option<PathBuf>,
    load_path: Option<PathBuf>,
    pub replay: OperatorReplay,
    frames: u64,
    photos: HashSet<&'static str>,
    finished: Option<Instant>,
    frame_ms: Vec<f64>,
    benchmark_seconds: f64,
    pub saved: bool,
    pub restore_requested: bool,
    pub failure: Option<String>,
    wash_frame_ms: Vec<f64>,
    wash_started: Option<(Instant, f64)>,
    wash_seconds: (f64, f64),
    started: Instant,
    last_progress: Instant,
    last_phase: u32,
}
impl Default for Validation {
    fn default() -> Self {
        let mut directory = None;
        let mut capture = None;
        let mut load_path = None;
        let mut benchmark_seconds = 60.0;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--validate" => directory = args.next().map(PathBuf::from),
                "--capture" => capture = args.next().map(PathBuf::from),
                "--load" => load_path = args.next().map(PathBuf::from),
                "--benchmark-seconds" => {
                    benchmark_seconds = args.next().and_then(|v| v.parse().ok()).unwrap_or(60.0)
                }
                _ => {}
            }
        }
        Self {
            directory,
            capture,
            load_path,
            replay: OperatorReplay::default(),
            frames: 0,
            photos: HashSet::new(),
            finished: None,
            frame_ms: Vec::new(),
            benchmark_seconds,
            saved: false,
            restore_requested: false,
            failure: None,
            wash_frame_ms: Vec::new(),
            wash_started: None,
            wash_seconds: (0.0, 0.0),
            started: Instant::now(),
            last_progress: Instant::now(),
            last_phase: u32::MAX,
        }
    }
}
impl Validation {
    pub fn automated(&self) -> bool {
        self.active() || self.capture.is_some()
    }
    pub fn ready(&self) -> bool {
        self.frames >= 180
    }
    pub fn active(&self) -> bool {
        self.directory.is_some()
    }
}

fn photo(commands: &mut Commands, v: &mut Validation, name: &'static str) {
    if v.photos.insert(name) {
        if let Some(dir) = &v.directory {
            let path = dir.join(format!("{name}.png"));
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
        }
    }
}
pub fn update(
    mut commands: Commands,
    mut game: ResMut<Game>,
    mut v: ResMut<Validation>,
    time: Res<Time<Real>>,
    mut exit: MessageWriter<AppExit>,
) {
    v.frames += 1;
    if v.frames == 1 {
        if let Some(path) = v.load_path.clone() {
            game.command(Command::Load(path));
            v.restore_requested = true;
        }
    }
    if v.active() && (v.last_phase != v.replay.phase || v.last_progress.elapsed().as_secs() >= 10) {
        println!("OPERATOR phase={} {} sim={:.1}s wall={:.1}s delivered={:.3}m3 processed={:.3}m3 water={} returned={:.4}m3 tailings={:.3}m3 gold={:.3}g",
            v.replay.phase, v.replay.label(), game.session.sim_time, v.started.elapsed().as_secs_f64(),
            game.session.delivered_m3, game.session.sluice.processed_m3,
            game.session.sluice.water.len(), game.session.sluice.water_returned_m3,
            game.session.sluice.tailings.volume(), game.session.sluice.concentrate.gold_g);
        v.last_phase = v.replay.phase;
        v.last_progress = Instant::now();
    }
    if v.automated() {
        if let Some(error) = &v.failure {
            eprintln!("MINING VALIDATION FAILED: {error}");
            if let Some(dir) = &v.directory {
                let _ = std::fs::write(dir.join("failure.txt"), error);
            }
            exit.write(AppExit::error());
            return;
        }
    }
    if v.replay.phase == 20 {
        v.wash_started
            .get_or_insert_with(|| (Instant::now(), game.session.sim_time));
        v.wash_frame_ms.push(time.delta_secs_f64() * 1000.0);
    } else if let Some((started, sim)) = v.wash_started.take() {
        v.wash_seconds = (started.elapsed().as_secs_f64(), game.session.sim_time - sim);
    }
    if let Some(path) = v.capture.clone() {
        let capture_frame = if v.load_path.is_some() { 1050 } else { 180 };
        let final_frame = capture_frame + 60;
        if v.frames > 300 {
            v.frame_ms.push(time.delta_secs_f64() * 1000.0);
        }
        if v.frames == capture_frame {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
        }
        if v.frames > final_frame {
            if v.load_path.is_some() && !v.saved {
                eprintln!("Capture did not restore the requested live shift");
                exit.write(AppExit::error());
            } else {
                let mean = v.frame_ms.iter().sum::<f64>() / v.frame_ms.len().max(1) as f64;
                let config = mining_physics::geometry::config(&game.session.sluice);
                let below_floor = game
                    .session
                    .sluice
                    .water
                    .iter()
                    .filter(|p| {
                        let x = (p.position.x / config.cell_size) as usize;
                        p.position.y
                            < (config.floor_height_at(x.min(config.grid_width - 1)) + 1) as f32
                                * config.cell_size
                                - 0.002
                    })
                    .count();
                let report = serde_json::json!({"scenario":if v.load_path.is_some() { "live saved shift" } else { "initial scene" },
                    "mean_fps":if mean > 0.0 { Some(1000.0/mean) } else { None },
                    "frames":v.frame_ms.len(), "save_restored":v.saved,
                    "water_rendering":"GPU-instanced sphere splats",
                    "water_particles":game.session.sluice.water.len(),
                    "sampled_splats":crate::water::visible_count(&game.session.sluice),
                    "particles_below_floor":below_floor,
                    "ledger_error_g":game.session.accounted_gold()-game.session.initial_gold_g});
                let _ = std::fs::write(
                    path.with_extension("timing.json"),
                    serde_json::to_string_pretty(&report).unwrap(),
                );
                exit.write(AppExit::Success);
            }
        }
        return;
    }
    let Some(dir) = v.directory.clone() else {
        return;
    };
    if v.frames == 1 {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            v.failure = Some(e.to_string());
        }
    }
    if v.frames == 150 {
        photo(&mut commands, &mut v, "01-site");
    }
    if v.replay.phase == 4 && v.replay.ticks > 30 {
        photo(&mut commands, &mut v, "02-cab");
    }
    if v.replay.phase == 5 && v.replay.ticks > 70 {
        photo(&mut commands, &mut v, "03-delivery");
    }
    if v.replay.phase == 20 && v.replay.ticks > 1800 {
        photo(&mut commands, &mut v, "04-washing");
    }
    if v.replay.phase == 20 && game.session.sluice.processed_m3 > 0.04 && !v.restore_requested {
        game.command(Command::Save(save_path()));
        game.command(Command::Load(save_path()));
        v.restore_requested = true;
    }
    if v.replay.complete && v.finished.is_none() {
        if !v.saved {
            v.failure = Some("The replay did not verify live save/restore".into());
        }
        match replay_receipt(&game.session, &v.replay) {
            Ok(receipt) => {
                if let Err(e) = std::fs::write(dir.join("operation.json"), receipt) {
                    v.failure = Some(e.to_string());
                }
            }
            Err(e) => v.failure = Some(e),
        }
        photo(&mut commands, &mut v, "05-cleanup");
        v.finished = Some(Instant::now());
    }
    if let Some(done) = v.finished {
        // Discard five seconds of shader/scene warmup after the completed shift.
        if done.elapsed().as_secs_f64() > 5.0 {
            v.frame_ms.push(time.delta_secs_f64() * 1000.0);
        }
        if done.elapsed().as_secs_f64() > v.benchmark_seconds + 5.0 {
            let mut times = v.frame_ms.clone();
            times.sort_by(f64::total_cmp);
            let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
            let percentile = |p: f64| {
                times
                    .get(((times.len() as f64 * p) as usize).min(times.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0)
            };
            let report = serde_json::json!({"frame_count":times.len(),"mean_ms":mean,"mean_fps":1000.0/mean,
                "p50_ms":percentile(0.50),"p95_ms":percentile(0.95),"p99_ms":percentile(0.99),
                "benchmark_seconds":v.benchmark_seconds,"resolution":"1440x900","save_restored_during_processing":v.saved,
                "ledger_error_g":game.session.accounted_gold()-game.session.initial_gold_g,
                "wash_frames":v.wash_frame_ms.len(),
                "wash_mean_fps":1000.0/(v.wash_frame_ms.iter().sum::<f64>()/v.wash_frame_ms.len().max(1) as f64),
                "wash_wall_seconds":v.wash_seconds.0,
                "wash_simulated_seconds":v.wash_seconds.1,
                "wash_simulation_speed":v.wash_seconds.1/v.wash_seconds.0.max(1e-9),
                "solver":"original FLIP/APIC + CPU DEM + GPU heightfield",
                "flip_substeps_per_tick":mining_physics::FLIP_SUBSTEPS});
            if let Err(e) = std::fs::write(
                dir.join("render-timing.json"),
                serde_json::to_string_pretty(&report).unwrap(),
            ) {
                v.failure = Some(e.to_string());
            }
            println!("MINING VALIDATION: {report}");
            exit.write(AppExit::Success);
        }
    }
    if v.started.elapsed().as_secs() > 1800 {
        v.failure = Some("Rendered validation exceeded thirty minutes".into());
    }
    if let Some(error) = &v.failure {
        eprintln!("MINING VALIDATION FAILED: {error}");
        let _ = std::fs::write(dir.join("failure.txt"), error);
        exit.write(AppExit::error());
    }
}
