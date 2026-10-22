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
    /// Opt-in capture channels: final, scene, normal, column, sediment.
    pub water_debug: u32,
    /// Keep the physical pause while removing its obstructing evidence overlay.
    pub frozen: bool,
    directory: Option<PathBuf>,
    started: Instant,
    recording: Option<Instant>,
    finished: Option<Instant>,
    next: f64,
    ids: HashSet<u64>,
    frames: Vec<serde_json::Value>,
    duration: f64,
    interval: f64,
    frame_limit: usize,
    view: Option<String>,
}
impl Default for MotionCapture {
    fn default() -> Self {
        let args: Vec<_> = std::env::args().collect();
        let directory = args
            .windows(2)
            .find(|p| p[0] == "--capture-motion")
            .map(|p| PathBuf::from(&p[1]));
        let number = |flag: &str, default: f64, max: f64| {
            let value = args
                .windows(2)
                .find(|p| p[0] == flag)
                .map(|p| p[1].parse::<f64>().expect("numeric capture setting"))
                .unwrap_or(default);
            assert!(
                value.is_finite() && value > 0. && value <= max,
                "bounded capture setting: {flag}"
            );
            value
        };
        let duration = number("--capture-seconds", 6., 20.);
        let hz = number("--capture-hz", 20., 40.);
        let view = args
            .windows(2)
            .find(|p| p[0] == "--capture-view")
            .map(|p| p[1].clone());
        assert!(
            view.as_deref().is_none_or(|v| matches!(
                v,
                "sluice" | "outlet" | "river" | "tailrace" | "bucket" | "bank"
            )),
            "capture view"
        );
        if let Some(dir) = &directory {
            assert!(
                !dir.join("frames.json").exists(),
                "preserve existing motion evidence"
            );
            std::fs::create_dir_all(dir).expect("motion capture directory");
        }
        let water_debug = args
            .windows(2)
            .find(|p| p[0] == "--capture-water-debug")
            .map(|p| match p[1].as_str() {
                "final" => 0,
                "scene" => 1,
                "normal" => 2,
                "column" => 3,
                "sediment" => 4,
                _ => panic!("unknown capture water channel"),
            })
            .unwrap_or(0);
        assert!(
            directory.is_some() || water_debug == 0,
            "water diagnostics require --capture-motion"
        );
        let frozen = directory.is_some() && args.iter().any(|a| a == "--freeze");
        Self {
            water_debug,
            frozen,
            water_only: directory.is_some() && args.iter().any(|a| a == "--capture-water-only"),
            directory,
            started: Instant::now(),
            recording: None,
            finished: None,
            next: 0.,
            ids: HashSet::new(),
            frames: Vec::new(),
            duration,
            interval: 1. / hz,
            frame_limit: (duration * hz).ceil() as usize,
            view,
        }
    }
}

/// Fixed diagnostic cameras change presentation only, never player or solver
/// state. The default capture keeps the ordinary gameplay camera.
pub fn camera(
    capture: Res<MotionCapture>,
    game: Res<crate::app::Game>,
    validation: Res<crate::validation::Validation>,
    mut camera: Query<&mut Transform, With<crate::app::GameCamera>>,
) {
    if validation.earthworks_only {
        let m = &game.session.machine;
        let target = crate::app::bvec(m.position + m.forward() * 2.3) + Vec3::Y * 1.1;
        if let Ok(mut camera) = camera.single_mut() {
            *camera = Transform::from_translation(target + Vec3::new(6., 4.6, 5.5))
                .looking_at(target, Vec3::Y);
        }
        return;
    }
    if capture.directory.is_none() {
        return;
    }
    if capture.view.as_deref() == Some("bucket") {
        let arm = game.session.machine.arm();
        let forward = crate::app::bvec(arm.forward);
        let right = Vec3::new(forward.z, 0., -forward.x);
        let target = crate::app::bvec((arm.heel + arm.lip) * 0.5) + Vec3::Y * 0.13;
        if let Ok(mut camera) = camera.single_mut() {
            *camera = Transform::from_translation(
                target + right * 1.25 + forward * 0.85 + Vec3::Y * 1.25,
            )
            .looking_at(target, Vec3::Y);
        }
        return;
    }
    let Some(p) = game.session.sluice.placement else {
        return;
    };
    let Some(view) = capture.view.as_deref() else {
        return;
    };
    let f = crate::app::bvec(p.forward());
    let r = crate::app::bvec(p.right());
    let (target, offset) = match view {
        "sluice" => (
            crate::app::bvec(p.to_world(mining_sim::Vec3::new(0.8, 0.035, 0.))),
            r * 0.75 + Vec3::Y * 0.95 - f * 0.9,
        ),
        "outlet" => (
            crate::app::bvec(p.outlet()) + f * 0.4 - Vec3::Y * 0.3,
            r * 1.1 + Vec3::Y * 0.7 + f * 0.25,
        ),
        "tailrace" => {
            let q = crate::app::bvec(p.outlet()) + f * 1.7;
            (
                Vec3::new(q.x, game.session.site.height(q.x, q.z) + 0.02, q.z),
                r * 3.2 + Vec3::Y * 3. - f * 2.5,
            )
        }
        "river" => {
            let z = 18.5;
            let x = mining_sim::Site::river_center(z);
            (
                Vec3::new(x, game.session.site.height(x, z) + 0.05, z),
                r * 3.2 + Vec3::Y * 3. - f * 2.5,
            )
        }
        "bank" => {
            let z = 18.5;
            let x = mining_sim::Site::river_center(z);
            (
                Vec3::new(x, game.session.site.height(x, z) + 1.5, z),
                r * 7. + Vec3::Y * 2.5 - f * 6.,
            )
        }
        _ => unreachable!(),
    };
    if let Ok(mut camera) = camera.single_mut() {
        *camera = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
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
            let (passed, advancing_intervals) =
                recording_status(&capture.frames, capture.frozen, game.session.paused);
            let report = serde_json::json!({"source":"native Bevy screenshot, actual GPU output; capture overhead affects cadence","passed":passed,"expected_frozen":capture.frozen,"paused":game.session.paused,"session_message":game.session.message,"advancing_intervals":advancing_intervals,"wall_seconds":capture.recording.unwrap().elapsed().as_secs_f64(),"frames":capture.frames,"water_presentation":water.presentation_report(),"water_balance_error_m3":game.session.water_balance_error_m3(),"gold_balance_error_g":game.session.accounted_gold()-game.session.initial_gold_g});
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
            if !passed {
                eprintln!(
                    "MOTION CAPTURE FAILED: paused={}, advancing_intervals={}, message={}",
                    game.session.paused, advancing_intervals, game.session.message
                );
            }
            exit.write(if passed {
                AppExit::Success
            } else {
                AppExit::error()
            });
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
    if elapsed >= capture.duration || capture.frames.len() >= capture.frame_limit {
        capture.finished = Some(Instant::now());
        return;
    }
    if elapsed < capture.next {
        return;
    }
    capture.next = elapsed + capture.interval;
    let sluice = &game.session.sluice;
    let tracks = water_tracks(sluice, &capture.ids);
    let name = format!("frame-{:04}.png", capture.frames.len());
    let record = serde_json::json!({"file":name,"wall":elapsed,"sim_time":game.session.sim_time,"paused":game.session.paused,"session_message":game.session.message,"fluid_clock":sluice.fluid_clock,"solver_steps":sluice.solver_steps,"water_only":capture.water_only,"water_debug":capture.water_debug,"flow":flow_metrics(sluice),"tracks":tracks,"surface":water.motion_samples(sluice),"presentation":water.presentation_report(),"machine":game.session.machine,"excavated_m3":game.session.site.excavated.volume(),"delivered_m3":game.session.delivered_m3});
    capture.frames.push(record);
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(directory.join(name)));
}

fn water_tracks(sluice: &mining_sim::Sluice, ids: &HashSet<u64>) -> Vec<serde_json::Value> {
    sluice.water.iter().filter(|p|ids.contains(&p.id)).filter_map(|p| {
        let placement=sluice.placement?;
        Some(serde_json::json!({"id":p.id,"kind":"channel","grid":p.position.to_array(),"world":placement.to_world(sluice.grid_to_local(p.position)).to_array(),"velocity":sluice.grid_velocity_to_world(p.velocity).to_array(),"height_above_plane_m":sluice.grid_to_local(p.position).y}))
    }).chain(sluice.outflow.iter().filter(|p|ids.contains(&p.id)).map(|p|serde_json::json!({"id":p.id,"kind":"outlet","world":p.position.to_array(),"velocity":p.velocity.to_array()}))).collect()
}

/// Population measurements in the same central channel region as the raised
/// tracks. This avoids interpreting the highest 128 as typical of all water.
fn flow_metrics(sluice: &mining_sim::Sluice) -> serde_json::Value {
    let (mut count, mut high, mut backflow, mut high_vertical) = (0usize, 0usize, 0usize, 0usize);
    let (mut vx, mut vy, mut high_vx, mut high_vy) = (0f64, 0f64, 0f64, 0f64);
    for p in &sluice.water {
        if !(0.3..1.35).contains(&p.position.x) || !(0.07..0.33).contains(&p.position.z) {
            continue;
        }
        count += 1;
        vx += p.velocity.x as f64;
        vy += p.velocity.y.abs() as f64;
        backflow += usize::from(p.velocity.x < -0.05);
        if sluice.grid_to_local(p.position).y > 0.05 {
            high += 1;
            high_vx += p.velocity.x.abs() as f64;
            high_vy += p.velocity.y.abs() as f64;
            high_vertical += usize::from(p.velocity.y.abs() > p.velocity.x.abs());
        }
    }
    serde_json::json!({"central_water":count,"mean_downstream_mps":vx/count.max(1) as f64,
        "mean_abs_vertical_mps":vy/count.max(1) as f64,"backflow_fraction":backflow as f64/count.max(1) as f64,
        "above_5cm":high,"above_5cm_fraction":high as f64/count.max(1) as f64,
        "high_mean_abs_downstream_mps":high_vx/high.max(1) as f64,"high_mean_abs_vertical_mps":high_vy/high.max(1) as f64,
        "high_vertical_dominance":high_vertical as f64/high.max(1) as f64})
}

#[cfg(test)]
mod tests {
    #[test]
    fn fresh_claim_motion_capture_does_not_require_an_installed_sluice() {
        let session = mining_sim::Session::new();
        assert!(super::water_tracks(&session.sluice, &Default::default()).is_empty());
    }
}

fn recording_status(frames: &[serde_json::Value], frozen: bool, paused: bool) -> (bool, usize) {
    let Some(times) = frames
        .iter()
        .map(|r| r.get("sim_time").and_then(serde_json::Value::as_f64))
        .collect::<Option<Vec<_>>>()
    else {
        return (false, 0);
    };
    let advancing = times.windows(2).filter(|p| p[1] > p[0]).count();
    let monotone = times.windows(2).all(|p| p[1] >= p[0]);
    let passed = times.len() >= 2
        && monotone
        && if frozen {
            advancing == 0
        } else {
            !paused && advancing > 0
        };
    (passed, advancing)
}

#[cfg(test)]
mod recording_tests {
    use super::recording_status;
    fn frames(times: &[f64]) -> Vec<serde_json::Value> {
        times
            .iter()
            .map(|t| serde_json::json!({"sim_time":t}))
            .collect()
    }
    #[test]
    fn live_capture_rejects_static_or_missing_progress() {
        assert_eq!(
            recording_status(&frames(&[1., 1., 1.]), false, false),
            (false, 0)
        );
        assert_eq!(recording_status(&[], false, false), (false, 0));
        assert_eq!(
            recording_status(&[serde_json::json!({})], false, false),
            (false, 0)
        );
    }
    #[test]
    fn frozen_capture_requires_a_constant_known_clock() {
        assert_eq!(recording_status(&frames(&[1., 1.]), true, true), (true, 0));
        assert_eq!(recording_status(&frames(&[1., 1.]), true, false), (true, 0));
        assert_eq!(
            recording_status(&frames(&[1., 1.1]), true, false),
            (false, 1)
        );
        assert_eq!(
            recording_status(&[serde_json::json!({}), serde_json::json!({})], true, false),
            (false, 0)
        );
    }
    #[test]
    fn live_capture_records_monotone_progress_and_holds() {
        assert_eq!(
            recording_status(&frames(&[1., 1.1, 1.1, 1.2]), false, false),
            (true, 2)
        );
    }
    #[test]
    fn paused_or_rewound_capture_is_not_a_pass() {
        assert_eq!(
            recording_status(&frames(&[1., 1.1]), false, true),
            (false, 1)
        );
        assert_eq!(
            recording_status(&frames(&[1., 0.9, 1.]), false, false),
            (false, 1)
        );
    }
}
