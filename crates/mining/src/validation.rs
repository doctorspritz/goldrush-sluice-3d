//! Rendered operator replay and benchmark. Screenshots use the real game camera;
//! all mining and transfers run through the same domain controls as interactive play.
use crate::app::{save_path, Game};
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use mining_physics::worker::Command;
use mining_sim::replay::{earthworks_receipt, replay_receipt, OperatorReplay};
use std::{collections::HashSet, path::PathBuf, time::Instant};

#[derive(Resource)]
pub struct Validation {
    pub directory: Option<PathBuf>,
    pub input_check: bool,
    pub earthworks_only: bool,
    earthwork_started: Option<(Instant, f64)>,
    earthwork_seconds: (f64, f64),
    flow_benchmark: Option<PathBuf>,
    flow_result: Option<bool>,
    flow_started: Option<(Instant, f64)>,
    flow_initial_bulk_m3: Option<f64>,
    flow_initial_mass_kg: Option<f64>,
    flow_sample: Option<(Instant, f64)>,
    flow_windows: Vec<f64>,
    worker_ms: Vec<f64>,
    worker_phases: [f64; 5],
    pub input_complete: bool,
    capture: Option<PathBuf>,
    motion_capture: bool,
    freeze: bool,
    detail_walk: bool,
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
        let mut flow_benchmark = None;
        let mut benchmark_seconds = 60.0;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--flow-benchmark" => flow_benchmark = args.next().map(PathBuf::from),
                "--validate" => directory = args.next().map(PathBuf::from),
                "--capture" => capture = args.next().map(PathBuf::from),
                "--load" => load_path = args.next().map(PathBuf::from),
                "--benchmark-seconds" => {
                    benchmark_seconds = args.next().and_then(|v| v.parse().ok()).unwrap_or(60.0)
                }
                _ => {}
            }
        }
        let earthworks_only = std::env::args().any(|a| a == "--earthworks-only");
        assert!(
            !earthworks_only || directory.is_some(),
            "--earthworks-only requires --validate DIRECTORY"
        );
        if earthworks_only {
            assert!(
                !directory.as_ref().unwrap().join("earthworks.json").exists(),
                "preserve existing earthwork evidence"
            );
        }
        Self {
            directory,
            earthworks_only,
            earthwork_started: None,
            earthwork_seconds: (0., 0.),
            flow_benchmark,
            flow_result: None,
            flow_started: None,
            flow_initial_bulk_m3: None,
            flow_initial_mass_kg: None,
            flow_sample: None,
            flow_windows: Vec::new(),
            worker_ms: Vec::new(),
            worker_phases: [0.; 5],
            input_check: std::env::args().any(|a| a == "--check-controls"),
            input_complete: false,
            capture,
            motion_capture: std::env::args().any(|a| a == "--capture-motion"),
            freeze: std::env::args().any(|a| a == "--freeze"),
            detail_walk: std::env::args().any(|a| a == "--detail-walk"),
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
    pub fn record_step(&mut self, ms: f64, phases: [f64; 5]) {
        if self.flow_started.is_some() {
            self.worker_ms.push(ms);
            for (sum, p) in self.worker_phases.iter_mut().zip(phases) {
                *sum += p;
            }
        }
    }
    pub fn automated(&self) -> bool {
        self.active() || self.capture.is_some() || self.input_check || self.flow_benchmark.is_some()
    }
    pub fn ready(&self) -> bool {
        self.frames >= 180
    }
    pub fn active(&self) -> bool {
        self.directory.is_some()
    }
}

fn photo(commands: &mut Commands, v: &mut Validation, name: &'static str) {
    // Bevy skips duplicate requests for a render target in the same frame.
    // Motion capture owns the primary target; do not race it with still photos.
    if v.motion_capture {
        return;
    }
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
    water: Res<crate::water::WaterSplats>,
    detail: Res<crate::environment::Detail>,
    mut exit: MessageWriter<AppExit>,
) {
    v.frames += 1;
    // Render-only camera traversal checks the cost of LOD transitions without
    // modifying the frozen solver state or requiring desktop key injection.
    if v.detail_walk && v.freeze && v.capture.is_some() && v.saved && v.frames > 240 {
        let phase = ((v.frames - 240) as f32 / 240.) % 2.;
        let distance = if phase < 1. { phase } else { 2. - phase };
        let x = 8. + distance * 8.;
        let z = game.session.player.position.z;
        game.session.player.position.x = x;
        game.session.player.position.y = game.session.site.height(x, z);
    }
    if v.automated() && crate::water::surface_failed(&water) {
        v.failure = Some("Water render pipeline failed".into());
    }

    if v.frames == 1 {
        if let Some(path) = v.load_path.clone() {
            game.command(Command::Load(path));
            v.restore_requested = true;
        }
        // A fixed saved scene allows before/after rendering checks without
        // different solver timings changing its water depth or particle state.
        if v.freeze {
            game.command(Command::Pause(true));
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
    if let Some(path) = v.flow_benchmark.clone() {
        if let Some(passed) = v.flow_result {
            exit.write(if passed {
                AppExit::Success
            } else {
                AppExit::error()
            });
            return;
        }
        if v.started.elapsed().as_secs_f64() < 5.0 || (v.load_path.is_some() && !v.saved) {
            return;
        }
        let now = Instant::now();
        let sim = game.session.sim_time;
        let (started, sim_start) = *v.flow_started.get_or_insert((now, sim));
        let initial_bulk = *v
            .flow_initial_bulk_m3
            .get_or_insert_with(|| game.session.accounted_bulk_material_m3());
        let initial_mass = *v
            .flow_initial_mass_kg
            .get_or_insert_with(|| game.session.accounted_dry_mass_kg());
        let (sample, previous) = *v.flow_sample.get_or_insert((now, sim));
        v.frame_ms.push(time.delta_secs_f64() * 1000.0);
        if sample.elapsed().as_secs_f64() >= 1.0 {
            v.flow_windows
                .push((sim - previous) / sample.elapsed().as_secs_f64());
            v.flow_sample = Some((now, sim));
            if v.flow_windows.len().is_multiple_of(10) {
                println!(
                    "FLOW progress wall={:.1}s sim={:.2}s rate={:.3}",
                    started.elapsed().as_secs_f64(),
                    sim - sim_start,
                    v.flow_windows.last().unwrap()
                );
            }
        }
        // Capture the measured live scene in the same bounded app run.
        if started.elapsed().as_secs_f64() >= 10. && v.photos.insert("flow-benchmark") {
            if let Some(path) = v.capture.clone() {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            }
        }
        if started.elapsed().as_secs_f64() >= v.benchmark_seconds {
            let wall = started.elapsed().as_secs_f64();
            let simulated = sim - sim_start;
            let mean = v.frame_ms.iter().sum::<f64>() / v.frame_ms.len().max(1) as f64;
            let mut sorted = v.frame_ms.clone();
            sorted.sort_by(f64::total_cmp);
            let p95 = sorted[(sorted.len().saturating_sub(1) as f64 * 0.95) as usize];
            let sluice = &game.session.sluice;
            let sluice_water_error = sluice.water_used_m3 - sluice.accounted_water_m3();
            let water_error = game.session.water_balance_error_m3();
            let gold_error = game.session.accounted_gold() - game.session.initial_gold_g;
            let bulk_error = game.session.accounted_bulk_material_m3() - initial_bulk;
            let bulk_tolerance = initial_bulk * 1e-6;
            let bulk_passed = bulk_error.abs() <= bulk_tolerance;
            let mass_error = game.session.accounted_dry_mass_kg() - initial_mass;
            let mass_tolerance = initial_mass * 1e-6;
            let mass_passed = mass_error.abs() <= mass_tolerance;
            let passed = simulated / wall >= 0.98
                && 1000. / mean >= 50.
                && crate::water::surface_ready(&water)
                && water_error.abs() < 1e-4
                && sluice_water_error.abs() < 1e-6
                && gold_error.abs() < 1e-6;
            let passed = passed && bulk_passed && mass_passed;
            v.flow_result = Some(passed);
            let report = serde_json::json!({"passed":passed,"water_balance_error_m3":water_error,
                "bulk_material_error_m3":bulk_error,"initial_bulk_material_m3":initial_bulk,
                "bulk_tolerance_m3":bulk_tolerance,"bulk_budget_passed":bulk_passed,
                "dry_mass_error_kg":mass_error,"initial_dry_mass_kg":initial_mass,
                "dry_mass_tolerance_kg":mass_tolerance,"dry_mass_budget_passed":mass_passed,
                "inlet":sluice.inlet,
                "sluice_water_balance_error_m3":sluice_water_error,
                "water_balance":game.session.water_balance,
                "stored_water_m3":game.session.stored_water_m3(),
                "exported_material_m3":game.session.site.hydraulic.exported_material_m3,
                "exported_gold_g":game.session.site.hydraulic.exported_gold_g,
                "warmup_seconds":5,"wall_seconds":wall,"simulation_seconds":simulated,
                "simulation_wall_ratio":simulated/wall,"one_second_ratios":v.flow_windows,"mean_fps":1000.0/mean,"p95_frame_ms":p95,
                "last_worker_tick_ms":game.step_ms,"worker_packets":v.worker_ms.len(),"worker_busy_fraction":v.worker_ms.iter().sum::<f64>()/(wall*1000.),"worker_mean_ms":v.worker_ms.iter().sum::<f64>()/v.worker_ms.len().max(1) as f64,
                "worker_phases_mean_ms":v.worker_phases.map(|n|n/v.worker_ms.len().max(1) as f64),"water_particles":game.session.sluice.water.len(),
                    "grains":game.session.sluice.grains.len(),
                    "displayed_grains":crate::sediment::visible_grains(&game.session.sluice),
                    "slurry":mining_sim::slurry::diagnostics(&game.session.sluice),
                    "water_presentation": water.presentation_report(),
                "water_mean_downstream_ms":game.session.sluice.water_speed(),"fluid_surface_ready":crate::water::surface_ready(&water),
                    "outflow_particles":sluice.outflow.len(),"terrain_water_splats":water.terrain_count(),"terrain_fine_cells":water.terrain_fine_count(),"detail_revision":detail.revision,"gold_balance_error_g":game.session.accounted_gold()-game.session.initial_gold_g});
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap())
                .expect("write flow benchmark");
            mining_sim::storage::save(&game.session, &path.with_extension("save.json"))
                .expect("save measured flow state for headless comparison");
            println!("FLOW BENCHMARK: {report}");
            // A benchmark that ran but missed throughput must not appear passed.
            exit.write(if passed {
                AppExit::Success
            } else {
                AppExit::error()
            });
        }
        return;
    }
    if let Some(path) = v.capture.clone() {
        // Keep the deadline stable while a native check switches demo/shift.
        let capture_frame = 1050;
        let final_frame = capture_frame + 60;
        if v.frames > if v.input_check { 900 } else { 300 } {
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
            if v.input_check && !v.input_complete {
                eprintln!("Native input checks did not complete");
                exit.write(AppExit::error());
            } else if !game.session.sluice.water.is_empty() && !crate::water::surface_ready(&water)
            {
                eprintln!("Water surface was not rendered");
                exit.write(AppExit::error());
            } else if v.load_path.is_some() && !v.saved {
                eprintln!("Capture did not restore the requested live shift");
                exit.write(AppExit::error());
            } else {
                let mean = v.frame_ms.iter().sum::<f64>() / v.frame_ms.len().max(1) as f64;
                let mut sorted = v.frame_ms.clone();
                sorted.sort_by(f64::total_cmp);
                let p95 = sorted[(sorted.len().saturating_sub(1) as f64 * 0.95) as usize];
                let config = mining_physics::geometry::config(&game.session.sluice);
                let below_floor = game
                    .session
                    .sluice
                    .water
                    .iter()
                    .filter(|p| {
                        let x = (p.position.x / config.cell_size) as usize;
                        x < config.grid_width
                            && p.position.y
                                < (config.floor_height_at(x.min(config.grid_width - 1)) + 1) as f32
                                    * config.cell_size
                                    - 0.002
                    })
                    .count();
                let report = serde_json::json!({"scenario":if v.load_path.is_some() { "live saved shift" } else if game.demo { "flowing demo" } else { "initial scene" },
                    "mean_fps":if mean > 0.0 { Some(1000.0/mean) } else { None },
                    "frames":v.frame_ms.len(), "save_restored":v.saved,
                    "frozen":v.freeze,"camera_traversal":v.detail_walk,"p95_frame_ms":p95,"detail_revision":detail.revision,"terrain_fine_cells":water.terrain_fine_count(),"terrain_water_cells":water.terrain_count(),
                    "water_rendering":"anisotropic particle surface + optical thickness + connected terrain water",
                    "fluid_surface_ready":crate::water::surface_ready(&water),
                    "input_check_passed":if v.input_check { Some(v.input_complete) } else { None },
                    "hydraulic_profile":game.session.sluice.profile,
                    "simulation_time":game.session.sim_time,
                    "wall_seconds":v.started.elapsed().as_secs_f64(),
                    "water_mean_downstream_ms":game.session.sluice.water_speed(),
                    "water_particles":game.session.sluice.water.len(),
                    "airborne_water_particles":game.session.sluice.outflow.len(),
                    "water_balance_error_m3":game.session.water_balance_error_m3(),
                    "sluice_water_balance_error_m3":game.session.sluice.water_used_m3-game.session.sluice.accounted_water_m3(),
                    "sampled_splats":crate::water::visible_count(&game.session.sluice),
                    "displayed_grains":crate::sediment::visible_grains(&game.session.sluice),
                    "slurry":mining_sim::slurry::diagnostics(&game.session.sluice),
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
    if v.earthworks_only && v.replay.phase == 2 && v.replay.ticks > 100 {
        photo(&mut commands, &mut v, "02-earth-cut");
    }
    if v.replay.phase == 4 && v.replay.ticks > 30 {
        photo(&mut commands, &mut v, "02-cab");
    }
    if v.replay.phase == 5 && v.replay.ticks > 70 {
        photo(&mut commands, &mut v, "03-delivery");
    }
    if v.earthworks_only {
        if v.replay.phase >= 2 {
            v.earthwork_started
                .get_or_insert_with(|| (Instant::now(), game.session.sim_time));
        }
        if v.started.elapsed().as_secs_f64() > 5. && v.finished.is_none() {
            v.frame_ms.push(time.delta_secs_f64() * 1000.);
        }
        if v.replay.phase == 10 && v.replay.ticks > 30 && !v.restore_requested {
            game.command(Command::Save(save_path()));
            game.command(Command::Load(save_path()));
            v.restore_requested = true;
        }
        if v.replay.phase >= 15 && v.finished.is_none() {
            if let Some((start, sim)) = v.earthwork_started {
                v.earthwork_seconds = (start.elapsed().as_secs_f64(), game.session.sim_time - sim);
            }
            match earthworks_receipt(&game.session, &v.replay) {
                Ok(report) => {
                    if let Err(e) = std::fs::write(dir.join("earthworks.json"), report) {
                        v.failure = Some(e.to_string());
                    }
                }
                Err(error) => v.failure = Some(error),
            }
            game.command(Command::Save(save_path()));
            game.command(Command::Pause(true));
            photo(&mut commands, &mut v, "04-earthworks-delivered");
            v.finished = Some(Instant::now());
        }
        let screenshots_done = v
            .photos
            .iter()
            .all(|name| dir.join(format!("{name}.png")).exists());
        if v.finished
            .is_some_and(|done| done.elapsed().as_secs_f64() > 10.)
            && !screenshots_done
        {
            v.failure = Some("Earthwork screenshots did not finish writing".into());
        }
        if v.finished
            .is_some_and(|done| done.elapsed().as_secs_f64() > 2.)
            && (screenshots_done || v.failure.is_some())
        {
            if !v.saved {
                v.failure = Some("Earthwork live save/restore was not verified".into());
            }
            if let Some(error) = &v.failure {
                let _ = std::fs::write(dir.join("failure.txt"), error);
                eprintln!("EARTHWORK VALIDATION FAILED: {error}");
                exit.write(AppExit::error());
            } else {
                let mut times = v.frame_ms.clone();
                times.sort_by(f64::total_cmp);
                let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
                let report = serde_json::json!({"phase":"earthwork only; original GPU terrain flow running, sluice idle",
                    "mean_fps":1000./mean,"p95_ms":times.get(times.len()*95/100),"frames":times.len(),
                    "simulated_seconds":game.session.sim_time,"wall_seconds":v.started.elapsed().as_secs_f64(),
                    "active_simulated_seconds":v.earthwork_seconds.1,"active_wall_seconds":v.earthwork_seconds.0,
                    "simulation_wall_ratio":v.earthwork_seconds.1/v.earthwork_seconds.0.max(1e-9),
                    "save_restored_during_cut":v.saved,"fixed_inspection_camera":true});
                if let Err(error) = std::fs::write(
                    dir.join("render-timing.json"),
                    serde_json::to_string_pretty(&report).unwrap(),
                ) {
                    eprintln!("Earthwork timing receipt failed: {error}");
                    exit.write(AppExit::error());
                    return;
                }
                println!("EARTHWORK VALIDATION: {report}");
                exit.write(AppExit::Success);
            }
        }
        return;
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
