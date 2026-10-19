//! Rendered operator replay and benchmark. Screenshots use the real game camera;
//! all mining and transfers run through the same domain controls as interactive play.
use crate::app::{load, save, Game};
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use mining_sim::{
    replay::{replay_receipt, OperatorReplay},
    Action,
};
use std::{collections::HashSet, path::PathBuf, time::Instant};

#[derive(Resource)]
pub struct Validation {
    pub directory: Option<PathBuf>,
    capture: Option<PathBuf>,
    pub replay: OperatorReplay,
    frames: u64,
    photos: HashSet<&'static str>,
    finished: Option<Instant>,
    frame_ms: Vec<f64>,
    benchmark_seconds: f64,
    saved: bool,
    pub failure: Option<String>,
    started: Instant,
}
impl Default for Validation {
    fn default() -> Self {
        let mut directory = None;
        let mut capture = None;
        let mut benchmark_seconds = 60.0;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--validate" => directory = args.next().map(PathBuf::from),
                "--capture" => capture = args.next().map(PathBuf::from),
                "--benchmark-seconds" => {
                    benchmark_seconds = args.next().and_then(|v| v.parse().ok()).unwrap_or(60.0)
                }
                _ => {}
            }
        }
        Self {
            directory,
            capture,
            replay: OperatorReplay::default(),
            frames: 0,
            photos: HashSet::new(),
            finished: None,
            frame_ms: Vec::new(),
            benchmark_seconds,
            saved: false,
            failure: None,
            started: Instant::now(),
        }
    }
}
impl Validation {
    pub fn active(&self) -> bool {
        self.directory.is_some()
    }
}

pub fn fixed(
    mut game: ResMut<Game>,
    mut validation: ResMut<Validation>,
    mut hydraulics: NonSendMut<mining_physics::Hydraulics>,
) {
    if !validation.active() {
        let input = game.controls;
        game.session.step(input);
        if let Err(e) = hydraulics.step(&mut game.session) {
            game.session.message = e;
            game.session.paused = true;
        }
        return;
    }
    if validation.frames < 180 {
        return;
    }
    let loops = 1;
    for _ in 0..loops {
        let controls = validation.replay.controls(&mut game.session);
        game.controls = controls;
        game.session.step(controls);
        if let Err(e) = hydraulics.step(&mut game.session) {
            validation.failure = Some(e);
            break;
        }
        if let Some(e) = &validation.replay.error {
            validation.failure = Some(e.clone());
            break;
        }
    }
}
fn photo(commands: &mut Commands, v: &mut Validation, name: &'static str) {
    if v.photos.insert(name) {
        if let Some(dir) = &v.directory {
            let path = dir.join(format!("{name}.png"));
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
        }
    }
}
pub fn update(
    mut commands: Commands,
    mut game: ResMut<Game>,
    mut v: ResMut<Validation>,
    time: Res<Time<Real>>,
    mut exit: MessageWriter<AppExit>,
    mut hydraulics: NonSendMut<mining_physics::Hydraulics>,
) {
    v.frames += 1;
    if let Some(path) = v.capture.clone() {
        if v.frames == 180 {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
        }
        if v.frames > 210 {
            exit.write(AppExit::Success);
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
    if v.replay.phase == 2 && game.session.player.third_person {
        let _ = game.session.on_action(Action::Camera);
    }
    if v.replay.phase == 4 && v.replay.ticks > 30 {
        photo(&mut commands, &mut v, "02-cab");
    }
    if v.replay.phase == 5 {
        if !game.session.player.third_person {
            let _ = game.session.on_action(Action::Camera);
        }
        if v.replay.ticks > 70 {
            photo(&mut commands, &mut v, "03-delivery");
        }
    }
    if v.replay.phase == 20 && v.replay.ticks > 1800 {
        photo(&mut commands, &mut v, "04-washing");
    }
    if v.replay.phase == 20 && game.session.sluice.processed_m3 > 0.04 && !v.saved {
        match save(&game.session).and_then(|_| load()) {
            Ok(mut restored) => {
                restored.site.revision = game.session.site.revision + 1;
                if restored.sluice.grains.len() != game.session.sluice.grains.len() {
                    v.failure = Some("Save/load lost live parcels".into());
                }
                game.session = restored;
                hydraulics.restore();
                v.saved = true;
            }
            Err(e) => v.failure = Some(format!("Save/load during processing: {e}")),
        }
    }
    if v.replay.complete && v.finished.is_none() {
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
                "ledger_error_g":game.session.accounted_gold()-game.session.initial_gold_g});
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
