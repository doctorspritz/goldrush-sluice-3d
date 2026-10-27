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
    end_sim_time: Option<f64>,
    interval: f64,
    frame_limit: usize,
    view: Option<String>,
    initial_bulk: Option<f64>,
    initial_dry_mass: Option<f64>,
    initial_sim_time: Option<f64>,
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
        // Long physical checks can need more wall time without changing the
        // solver cadence. The explicit opt-in recording remains bounded.
        let duration = number("--capture-seconds", 6., 300.);
        let hz = number("--capture-hz", 20., 40.);
        let end_sim_time = args
            .windows(2)
            .find(|p| p[0] == "--capture-until-sim-time")
            .map(|_| number("--capture-until-sim-time", 18., 120.));
        let view = args
            .windows(2)
            .find(|p| p[0] == "--capture-view")
            .map(|p| p[1].clone());
        assert!(
            view.as_deref().is_none_or(|v| matches!(
                v,
                "sluice" | "outlet" | "river" | "tailrace" | "bucket" | "bank" | "machine" | "soil"
            )),
            "capture view"
        );
        if let Some(dir) = &directory {
            assert!(
                !dir.join("frames.json").exists()
                    && !dir.join("frames.partial.jsonl").exists()
                    && !dir.join("frame-0000.png").exists(),
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
        assert!(
            !frozen || end_sim_time.is_none(),
            "a physical-time capture endpoint cannot be frozen"
        );
        Self {
            end_sim_time,
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
            initial_bulk: None,
            initial_dry_mass: None,
            initial_sim_time: None,
        }
    }
}

impl MotionCapture {
    fn observe_baseline(&mut self, session: &mining_sim::Session, scene_ready: bool) -> bool {
        if !scene_ready {
            return false;
        }
        if self.initial_bulk.is_none() {
            self.initial_bulk = Some(session.accounted_bulk_material_m3());
            self.initial_dry_mass = Some(session.accounted_dry_mass_kg());
            self.initial_sim_time = Some(session.sim_time);
        }
        true
    }

    fn report(
        &self,
        session: &mining_sim::Session,
        validation: &crate::validation::Validation,
        presentation: serde_json::Value,
        failure: Option<&str>,
    ) -> serde_json::Value {
        let (base_passed, advancing_intervals) =
            recording_status(&self.frames, self.frozen, session.paused);
        let end_reached = self.end_sim_time.is_none_or(|t| {
            self.frames
                .last()
                .and_then(|f| f["sim_time"].as_f64())
                .is_some_and(|last| last >= t)
        });
        let baseline_selected = self.initial_bulk.is_some()
            && self.initial_dry_mass.is_some()
            && self.initial_sim_time.is_some();
        let passed = base_passed
            && end_reached
            && baseline_selected
            && validation.capture_scene_ready()
            && failure.is_none();
        serde_json::json!({
            "source":"native Bevy screenshot, actual GPU output; capture overhead affects cadence",
            "passed":passed,"failure":failure,
            "requested_end_sim_time":self.end_sim_time,"end_sim_time_reached":end_reached,
            "expected_frozen":self.frozen,"paused":session.paused,"session_message":session.message,
            "advancing_intervals":advancing_intervals,
            "wall_seconds":self.recording.map_or(0.,|started|started.elapsed().as_secs_f64()),
            "frames":self.frames,"water_presentation":presentation,
            "reference_sim_time":self.initial_sim_time,
            "restore_requested":validation.restore_requested,"restore_acknowledged":validation.saved,
            "reference_bulk_m3":self.initial_bulk,"accounted_bulk_m3":session.accounted_bulk_material_m3(),
            "reference_dry_mass_kg":self.initial_dry_mass,"accounted_dry_mass_kg":session.accounted_dry_mass_kg(),
            "water_balance_error_m3":session.water_balance_error_m3(),
            "gold_balance_error_g":session.accounted_gold()-session.initial_gold_g,
        })
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
    if capture.view.as_deref() == Some("machine") {
        let m = &game.session.machine;
        let target = crate::app::bvec(m.position + m.forward() * 2.0) + Vec3::Y * 2.7;
        if let Ok(mut camera) = camera.single_mut() {
            *camera = Transform::from_translation(target + Vec3::new(9., 7., 10.))
                .looking_at(target, Vec3::Y);
        }
        return;
    }
    if capture.view.as_deref() == Some("soil") {
        if let Some(soil) = &game.session.active_soil {
            let m = &game.session.machine;
            let arm = m.arm();
            let center = soil
                .clods
                .iter()
                .map(|c| c.center.as_dvec3() * c.material().wet_mass_kg())
                .reduce(|a, b| a + b)
                .unwrap_or_default()
                / soil.material().wet_mass_kg();
            let shell = m.bucket_model().is_shell();
            let target = crate::app::bvec(if shell {
                center.as_vec3() * 0.5 + (arm.heel + arm.lip) * 0.25
            } else {
                center.as_vec3() * 0.75 + (arm.heel + arm.lip) * 0.125
            }) + Vec3::Y * 0.1;
            let right = crate::app::bvec(m.upper_rotation() * mining_sim::Vec3::X);
            let forward = crate::app::bvec(m.upper_rotation() * mining_sim::Vec3::Z);
            if let Ok(mut camera) = camera.single_mut() {
                *camera = Transform::from_translation(
                    target + right * 2.3 + forward * if shell { -2. } else { 2. } + Vec3::Y * 2.1,
                )
                .looking_at(target, Vec3::Y);
            }
        }
        return;
    }
    if capture.view.as_deref() == Some("bucket") {
        let arm = game.session.machine.arm();
        let forward = crate::app::bvec(arm.forward);
        let right = Vec3::new(forward.z, 0., -forward.x);
        let shell = game.session.machine.bucket_model().is_shell();
        let target = crate::app::bvec((arm.heel + arm.lip) * 0.5) + Vec3::Y * 0.13;
        if let Ok(mut camera) = camera.single_mut() {
            let mounted =
                game.session.machine.bucket_model() == mining_sim::bucket::Model::MountedShellV2;
            // The deeper mounted bowl hides its floor behind the near cheek
            // from the old oblique inspection view. Look through the mouth.
            let offset = if mounted {
                // Keep the sightline outside the stick while looking through
                // the mouth; the old centreline view hid a real retained load.
                right * 0.95 - forward * 0.35 + Vec3::Y * 2.2
            } else {
                right * 1.25 + forward * if shell { -0.85 } else { 0.85 } + Vec3::Y * 1.25
            };
            *camera = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
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
type SoilRenderQuery<'w, 's> = Query<
    'w,
    's,
    (
        Option<&'static GlobalTransform>,
        Option<&'static ViewVisibility>,
        &'static Mesh3d,
    ),
    With<crate::scene::SoilVisual>,
>;
pub fn update(
    mut commands: Commands,
    (game, validation): (Res<crate::app::Game>, Res<crate::validation::Validation>),
    water: Res<crate::water::WaterSplats>,
    mut capture: ResMut<MotionCapture>,
    mut exit: MessageWriter<AppExit>,
    soil_visuals: SoilRenderQuery,
    meshes: Res<Assets<Mesh>>,
) {
    let Some(directory) = capture.directory.clone() else {
        return;
    };
    let failure = validation
        .failure
        .as_deref()
        .or_else(|| crate::water::surface_failed(&water).then_some("Water render pipeline failed"));
    if let Some(error) = failure {
        // A failed load/worker can never reach render readiness. Preserve an
        // explicit nonpass receipt and stop instead of waiting for warmup.
        let report = capture.report(
            &game.session,
            &validation,
            water.presentation_report(),
            Some(error),
        );
        std::fs::write(
            directory.join("frames.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .expect("failed motion receipt");
        eprintln!("MOTION CAPTURE FAILED: {error}");
        capture.directory = None;
        exit.write(AppExit::error());
        return;
    }
    if !capture.observe_baseline(&game.session, validation.capture_scene_ready()) {
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
            let report = capture.report(
                &game.session,
                &validation,
                water.presentation_report(),
                None,
            );
            let passed = report["passed"].as_bool().unwrap();
            let advancing_intervals = report["advancing_intervals"].as_u64().unwrap();
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
    let end_now = capture
        .end_sim_time
        .is_some_and(|t| game.session.sim_time >= t);
    if elapsed < capture.next && !end_now {
        return;
    }
    capture.next = elapsed + capture.interval;
    let sluice = &game.session.sluice;
    let tracks = water_tracks(sluice, &capture.ids);
    let name = format!("frame-{:04}.png", capture.frames.len());
    let record = serde_json::json!({"file":name,"wall":elapsed,"sim_time":game.session.sim_time,"paused":game.session.paused,"session_message":game.session.message,"fluid_clock":sluice.fluid_clock,"solver_steps":sluice.solver_steps,"water_only":capture.water_only,"water_debug":capture.water_debug,"flow":flow_metrics(sluice),"tracks":tracks,"surface":water.motion_samples(sluice),"presentation":water.presentation_report(),"machine":game.session.machine,"rigid_falling":game.session.falling.iter().filter(|l|l.rigid.is_some()).collect::<Vec<_>>(),"material_motion":game.session.material_motion,"loose_gold":game.session.site.loose_gold.as_ref().map(|g|serde_json::json!({"model_version":g.model_version,"friction_mu":g.friction_mu,"particles":g.particles.len(),"material":g.material(),"exported_g":g.exported_g,"excavated_g":g.excavated_g,"withdrawn_g":g.withdrawn_g,"admitted_g":g.admitted_g,"sample":g.particles.iter().take(64).collect::<Vec<_>>()})),"tailings":game.session.tailings.as_ref().map(|t|serde_json::json!({"airborne":t.packets.iter().filter(|p|!p.contact_pending).count(),"contact_pending":t.packets.iter().filter(|p|p.contact_pending).count(),"released_count":t.released_count,"released":t.released,"received":t.received,"exported":t.exported,"historic_bed":t.historic_bed,"active":t.active_material(),"sample":t.packets.iter().take(32).collect::<Vec<_>>()})),"soil_render":{"entities":soil_visuals.iter().count(),"visible":soil_visuals.iter().filter(|(_,v,_)|v.is_some_and(|v|v.get())).count(),"samples":soil_visuals.iter().take(3).map(|(t,v,m)|serde_json::json!({"translation":t.map(|t|t.translation().to_array()),"visible":v.map(|v|v.get()),"vertices":meshes.get(&m.0).map(|m|m.count_vertices()),"indexed":meshes.get(&m.0).map(|m|m.indices().is_some())})).collect::<Vec<_>>()},"active_soil":game.session.active_soil.as_ref().map(|s|serde_json::json!({"clods":s.clods.len(),"bucket_contents_estimate":mining_sim::bucket::estimated_soil_contents(&game.session.machine,s),"all_clods":if capture.frozen {Some(&s.clods)} else {None},"terrain_support":soil_support_report(&game.session.site,s),"material":s.material(),"totals":s.totals(mining_sim::Vec3::new(0.,-9.81,0.)),"feedback":s.feedback,"sample":s.clods.iter().take(32).map(|c|serde_json::json!({"center":c.center,"rotation":c.rotation,"velocity":c.velocity,"angular_velocity":c.angular_velocity})).collect::<Vec<_>>()})),"excavated_m3":game.session.site.excavated.volume(),"delivered_m3":game.session.delivered_m3});
    // Retain physical timing/support evidence even if validation exits before
    // the final capture report. This append is opt-in capture work only.
    let mut line = serde_json::to_vec(&record).expect("partial motion record");
    line.push(b'\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("frames.partial.jsonl"))
        .expect("partial motion log")
        .write_all(&line)
        .expect("partial motion write");
    capture.frames.push(record);
    if end_now {
        capture.finished = Some(Instant::now());
    }
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(directory.join(name)));
}

fn soil_support_report(
    site: &mining_sim::Site,
    soil: &mining_sim::soil::ActiveSoil,
) -> serde_json::Value {
    let mut worst = (f32::INFINITY, 0);
    let mut lost = 0;
    for (i, clod) in soil.clods.iter().enumerate() {
        let highest = clod
            .vertices()
            .iter()
            .map(|p| clod.center + clod.rotation * *p)
            .map(|p| p.y - mining_sim::soil::substrate_height(site, p.x, p.z))
            .fold(f32::NEG_INFINITY, f32::max);
        if highest < worst.0 {
            worst = (highest, i);
        }
        lost += usize::from(highest < -0.025);
    }
    let clod = &soil.clods[worst.1];
    serde_json::json!({"minimum_highest_vertex_clearance_m":worst.0,"fully_below_terrain_25mm":lost,
        "worst_clod_index":worst.1,"worst_center":clod.center,"worst_velocity":clod.velocity})
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

#[cfg(test)]
mod baseline_tests {
    use super::*;
    use crate::{app::Game, validation::Validation, water::WaterSplats};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture {
        app: App,
        directory: PathBuf,
    }
    impl Fixture {
        fn new(validation: Validation) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "sluice-motion-baseline-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            assert!(!directory.exists(), "preserve prior test evidence");
            std::fs::create_dir(&directory).unwrap();
            let capture = MotionCapture {
                directory: Some(directory.clone()),
                ..MotionCapture::default()
            };
            let mut app = App::new();
            // Exercise the real capture system without a window, renderer,
            // GPU, simulation worker or screenshot request during warmup.
            app.add_message::<AppExit>()
                .insert_resource(Game::default())
                .insert_resource(validation)
                .insert_resource(capture)
                .init_resource::<WaterSplats>()
                .init_resource::<Assets<Mesh>>()
                .add_systems(Update, update);
            Self { app, directory }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    #[test]
    fn pending_restore_skips_placeholder_then_keeps_first_restored_stock_before_warmup() {
        let mut validation = Validation::default();
        validation.restore_requested = true;
        validation.saved = false;
        let mut fixture = Fixture::new(validation);
        fixture.app.update();
        let capture = fixture.app.world().resource::<MotionCapture>();
        assert!(capture.initial_bulk.is_none());
        assert!(capture.frames.is_empty());
        assert!(capture.recording.is_none());
        let expected = {
            let mut game = fixture.app.world_mut().resource_mut::<Game>();
            // Distinct finite scene stock makes a placeholder baseline observable.
            game.session.sluice.hopper.soil_m3 = 0.02;
            game.session.sim_time = 37.;
            game.session.message = "Shift restored".into();
            (
                game.session.accounted_bulk_material_m3(),
                game.session.accounted_dry_mass_kg(),
            )
        };
        fixture.app.world_mut().resource_mut::<Validation>().saved = true;
        fixture.app.update();
        let capture = fixture.app.world().resource::<MotionCapture>();
        assert_eq!(capture.initial_bulk, Some(expected.0));
        assert_eq!(capture.initial_dry_mass, Some(expected.1));
        assert_eq!(capture.initial_sim_time, Some(37.));
        assert!(
            capture.recording.is_none(),
            "readiness warmup must not start recording"
        );
        fixture
            .app
            .world_mut()
            .resource_mut::<Game>()
            .session
            .sluice
            .hopper
            .soil_m3 += 0.01;
        fixture.app.update();
        let capture = fixture.app.world().resource::<MotionCapture>();
        assert_eq!(
            capture.initial_bulk,
            Some(expected.0),
            "warmup must not move the baseline"
        );
    }

    #[test]
    fn new_scene_selects_its_baseline_without_requiring_a_restore() {
        let mut fixture = Fixture::new(Validation::default());
        let expected = fixture
            .app
            .world()
            .resource::<Game>()
            .session
            .accounted_bulk_material_m3();
        fixture.app.update();
        let capture = fixture.app.world().resource::<MotionCapture>();
        assert_eq!(capture.initial_bulk, Some(expected));
        assert_eq!(capture.initial_sim_time, Some(0.));
        assert!(capture.frames.is_empty());
    }

    #[test]
    fn load_and_worker_failures_write_nonpass_and_exit_before_surface_readiness() {
        for error in [
            "Load failed: missing fixture",
            "Simulation worker disconnected",
        ] {
            let mut validation = Validation::default();
            validation.restore_requested = true;
            validation.failure = Some(error.into());
            let mut fixture = Fixture::new(validation);
            fixture.app.update();
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(fixture.directory.join("frames.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(report["passed"], false);
            assert_eq!(report["failure"], error);
            assert!(report["reference_bulk_m3"].is_null());
            assert_eq!(report["frames"].as_array().unwrap().len(), 0);
            assert!(fixture
                .app
                .world_mut()
                .resource_mut::<Messages<AppExit>>()
                .drain()
                .any(|exit| matches!(exit, AppExit::Error(_))));
            assert!(fixture
                .app
                .world()
                .resource::<MotionCapture>()
                .directory
                .is_none());
        }
    }

    #[test]
    fn recorded_progress_and_intended_frozen_pause_cannot_hide_a_worker_failure() {
        let validation = Validation::default();
        let mut capture = MotionCapture::default();
        let mut session = mining_sim::Session::new();
        session.paused = true;
        capture.frozen = true;
        capture.frames = vec![
            serde_json::json!({"sim_time":1.}),
            serde_json::json!({"sim_time":1.}),
        ];
        assert!(capture.observe_baseline(&session, true));
        let healthy = capture.report(&session, &validation, serde_json::Value::Null, None);
        assert_eq!(
            healthy["passed"], true,
            "intended frozen capture remains valid"
        );
        let failed = capture.report(
            &session,
            &validation,
            serde_json::Value::Null,
            Some("worker stopped"),
        );
        assert_eq!(failed["passed"], false);
        assert_eq!(failed["failure"], "worker stopped");
    }
}
