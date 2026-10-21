//! Opt-in renderer-owned evidence capture. Records real frame times and particle
//! identities; it does not prescribe, slow, or otherwise edit physical motion.
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use std::{collections::HashSet, path::PathBuf, time::Instant};

#[derive(Resource)]
pub struct MotionCapture {
    /// Diagnostic visibility control only; sediment physics/optics still run.
    pub water_only: bool,
    directory: Option<PathBuf>,
    started: Instant,
    recording: Option<Instant>,
    finished: Option<Instant>,
    next: f64,
    ids: HashSet<u64>,
    frames: Vec<serde_json::Value>,
}
impl Default for MotionCapture {
    fn default() -> Self {
        let args: Vec<_> = std::env::args().collect();
        let directory = args
            .windows(2)
            .find(|p| p[0] == "--capture-motion")
            .map(|p| PathBuf::from(&p[1]));
        if let Some(dir) = &directory {
            assert!(
                !dir.join("frames.json").exists(),
                "preserve existing motion evidence"
            );
            std::fs::create_dir_all(dir).expect("motion capture directory");
        }
        Self {
            water_only: directory.is_some() && args.iter().any(|a| a == "--capture-water-only"),
            directory,
            started: Instant::now(),
            recording: None,
            finished: None,
            next: 0.,
            ids: HashSet::new(),
            frames: Vec::new(),
        }
    }
}
pub fn update(
    mut commands: Commands,
    game: Res<crate::app::Game>,
    water: Res<crate::water::WaterSplats>,
    mut capture: ResMut<MotionCapture>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(directory) = capture.directory.clone() else {
        return;
    };
    if crate::water::surface_failed(&water) {
        exit.write(AppExit::error());
        return;
    }
    if capture.started.elapsed().as_secs_f64() < 5. || !crate::water::surface_ready(&water) {
        return;
    }
    if let Some(finished) = capture.finished {
        let written = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|p| p.path().extension().is_some_and(|p| p == "png"))
            .count();
        if finished.elapsed().as_secs_f64() > 1. && written == capture.frames.len() {
            let report = serde_json::json!({"source":"native Bevy screenshot, actual GPU output; capture overhead affects cadence","wall_seconds":capture.recording.unwrap().elapsed().as_secs_f64(),"frames":capture.frames,"water_presentation":water.presentation_report(),"water_balance_error_m3":game.session.water_balance_error_m3(),"gold_balance_error_g":game.session.accounted_gold()-game.session.initial_gold_g});
            std::fs::write(
                directory.join("frames.json"),
                serde_json::to_vec(&report).unwrap(),
            )
            .expect("motion receipt");
            println!(
                "MOTION CAPTURE: {} frames in {}",
                written,
                directory.display()
            );
            exit.write(AppExit::Success);
            capture.directory = None;
        } else if finished.elapsed().as_secs_f64() > 10. {
            eprintln!(
                "Missing screenshot files: {written}/{}",
                capture.frames.len()
            );
            exit.write(AppExit::error());
        }
        return;
    }
    if capture.recording.is_none() {
        let mut candidates: Vec<_> = game
            .session
            .sluice
            .water
            .iter()
            .filter_map(|p| {
                let local = game.session.sluice.grid_to_local(p.position);
                (p.position.x > 0.3
                    && p.position.x < 1.35
                    && p.position.z > 0.07
                    && p.position.z < 0.33)
                    .then_some((p.id, local.y))
            })
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        capture.ids = candidates.into_iter().take(128).map(|p| p.0).collect();
        capture.recording = Some(Instant::now());
    }
    let elapsed = capture.recording.unwrap().elapsed().as_secs_f64();
    if elapsed >= 6. || capture.frames.len() >= 120 {
        capture.finished = Some(Instant::now());
        return;
    }
    if elapsed < capture.next {
        return;
    }
    capture.next = elapsed + 0.05;
    let sluice = &game.session.sluice;
    let placement = sluice.placement.expect("flowing demo placement");
    let tracks:Vec<_>=sluice.water.iter().filter(|p|capture.ids.contains(&p.id)).map(|p|serde_json::json!({"id":p.id,"kind":"channel","grid":p.position.to_array(),"world":placement.to_world(sluice.grid_to_local(p.position)).to_array(),"velocity":sluice.grid_velocity_to_world(p.velocity).to_array(),"height_above_plane_m":sluice.grid_to_local(p.position).y})).chain(sluice.outflow.iter().filter(|p|capture.ids.contains(&p.id)).map(|p|serde_json::json!({"id":p.id,"kind":"outlet","world":p.position.to_array(),"velocity":p.velocity.to_array()}))).collect();
    let name = format!("frame-{:04}.png", capture.frames.len());
    let record = serde_json::json!({"file":name,"wall":elapsed,"sim_time":game.session.sim_time,"fluid_clock":sluice.fluid_clock,"solver_steps":sluice.solver_steps,"water_only":capture.water_only,"tracks":tracks,"surface":water.motion_samples(sluice),"presentation":water.presentation_report()});
    capture.frames.push(record);
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(directory.join(name)));
}
