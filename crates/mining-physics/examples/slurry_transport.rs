//! Finite pulses through the real FLIP/APIC/DEM adapter. The laboratory supplies
//! declared clean external water; it does not prescribe particle transport.
#[path = "support/slurry_pulse.rs"]
mod pulse;
use glam::Vec3;
use mining_physics::{
    grain_exits::{Snapshot, Status, POS_X},
    Hydraulics,
};
use mining_sim::{
    intake::IntakePacket, session::Controls, GrainKind, Placement, Session, FIXED_DT,
};
use serde_json::{json, Value};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

fn directory(path: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(path);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir(&p).map_err(|e| format!("preserve output {p:?}: {e}"))?;
    Ok(p)
}
fn write(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("preserve receipt {path:?}: {e}"))?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}
fn first_failure(failure: &mut Option<String>, error: impl Into<String>) {
    if failure.is_none() {
        *failure = Some(error.into());
    }
}
struct WaterBudget {
    whole_tolerance_m3: f64,
    maximum_whole_error_m3: f64,
    maximum_sluice_error_m3: f64,
    finite: bool,
}
impl WaterBudget {
    fn new(initial_stored_m3: f64) -> Self {
        Self {
            whole_tolerance_m3: initial_stored_m3 * 1e-6,
            maximum_whole_error_m3: 0.,
            maximum_sluice_error_m3: 0.,
            finite: initial_stored_m3.is_finite() && initial_stored_m3 >= 0.,
        }
    }
    fn observe(&mut self, whole_error_m3: f64, sluice_error_m3: f64) {
        self.finite &= whole_error_m3.is_finite() && sluice_error_m3.is_finite();
        if whole_error_m3.is_finite() {
            self.maximum_whole_error_m3 = self.maximum_whole_error_m3.max(whole_error_m3.abs());
        }
        if sluice_error_m3.is_finite() {
            self.maximum_sluice_error_m3 = self.maximum_sluice_error_m3.max(sluice_error_m3.abs());
        }
    }
    fn session(&mut self, s: &Session) {
        self.observe(
            s.water_balance_error_m3(),
            s.sluice.water_used_m3 - s.sluice.accounted_water_m3(),
        );
    }
    fn passed(&self) -> bool {
        self.finite
            && self.maximum_whole_error_m3 <= self.whole_tolerance_m3
            && self.maximum_sluice_error_m3 <= 1e-10
    }
    fn report(&self) -> Value {
        json!({"passed":self.passed(),"all_observations_finite":self.finite,
            "whole_water_passed":self.finite&&self.maximum_whole_error_m3<=self.whole_tolerance_m3,
            "sluice_water_passed":self.finite&&self.maximum_sluice_error_m3<=1e-10,
            "whole_water_tolerance_m3":self.whole_tolerance_m3,"sluice_water_tolerance_m3":1e-10,
            "maximum_finite_abs_whole_error_m3":self.maximum_whole_error_m3,
            "maximum_finite_abs_sluice_error_m3":self.maximum_sluice_error_m3})
    }
}
fn progress_error(
    initial_time: f64,
    final_time: f64,
    initial_steps: u64,
    final_steps: u64,
    ticks: usize,
    paused: bool,
    placed: bool,
) -> Option<String> {
    if paused || !placed {
        return Some("equivalence requires an unpaused placed sluice".into());
    }
    let elapsed = final_time - initial_time;
    let expected = ticks as f64 * FIXED_DT as f64;
    if !elapsed.is_finite() || elapsed <= 0. || (elapsed - expected).abs() > 1e-8 {
        return Some("equivalence did not advance the requested physical time".into());
    }
    if final_steps <= initial_steps {
        return Some("equivalence did not execute positive solver progress".into());
    }
    None
}
fn custody_errors(
    initial: [f64; 3],
    resident: [f64; 3],
    exits: [f64; 3],
    actual_tailings_m3: f64,
) -> Option<(f64, f64)> {
    if initial.iter().any(|v| !v.is_finite() || *v <= 0.)
        || resident.iter().chain(&exits).any(|v| !v.is_finite())
        || !actual_tailings_m3.is_finite()
    {
        return None;
    }
    let relative = (0..3)
        .map(|i| (resident[i] + exits[i] - initial[i]).abs() / initial[i])
        .fold(0_f64, f64::max);
    let tailings =
        (actual_tailings_m3 - exits.iter().sum::<f64>()).abs() / initial.iter().sum::<f64>();
    Some((relative, tailings))
}
fn observe_custody(
    initial: [f64; 3],
    resident: [f64; 3],
    exits: [f64; 3],
    actual_tailings_m3: f64,
    maximum_relative: &mut f64,
    maximum_tailings: &mut f64,
) -> bool {
    if let Some((relative, tailings)) = custody_errors(initial, resident, exits, actual_tailings_m3)
    {
        *maximum_relative = maximum_relative.max(relative);
        *maximum_tailings = maximum_tailings.max(tailings);
        true
    } else {
        false
    }
}
fn snapshot_evidence(snapshot: &Snapshot) -> Value {
    json!({"status":snapshot.status.as_str(),"is_complete":snapshot.is_complete(),
        "sim_time":snapshot.sim_time,"domain_max_m":snapshot.domain_max_m,
        "removal_stage_completed":snapshot.removal_stage_completed,
        "observed":snapshot.observed,"records":snapshot.exits().len(),
        "dropped":snapshot.dropped,"invalid":snapshot.invalid,"metadata_valid":snapshot.metadata_valid})
}
fn retain_setup_failure(
    dir: &Path,
    request: &Value,
    result: Result<(), String>,
) -> Result<(), String> {
    if let Err(error) = &result {
        let path = dir.join("report.json");
        if !path.exists() {
            if let Err(receipt_error) = write(
                &path,
                &json!({"kind":"experiment_setup_or_preloop_failure",
                "passed":false,"failure":error,"request":request,
                "state_save_exists":dir.join("state.save.json").exists()}),
            ) {
                return Err(format!(
                    "{error}; failure receipt could not be retained: {receipt_error}"
                ));
            }
        }
    }
    result
}
fn seconds(value: &str, minimum: f64, maximum: f64) -> Result<usize, String> {
    let n: f64 = value.parse().map_err(|_| "invalid duration")?;
    if !n.is_finite() || !(minimum..=maximum).contains(&n) {
        return Err(format!("duration must be {minimum}..={maximum} seconds"));
    }
    Ok((n * 60.).round() as usize)
}
fn solver(conservative: bool) -> Result<Hydraulics, String> {
    Ok(if conservative {
        Hydraulics::new()?.with_conservative_phases()
    } else {
        Hydraulics::new()?
    })
}
/// Explicit external source with complete water custody. Feed remains off.
/// Nozzle velocity, admission, outflow, cadence and real solvers remain active.
fn laboratory_tick(s: &mut Session, h: &mut Hydraulics) -> Result<f64, String> {
    s.sim_time += FIXED_DT as f64;
    mining_sim::outflow::step(&mut s.sluice, &mut s.site, FIXED_DT);
    let room = s
        .sluice
        .profile
        .capacity()
        .saturating_sub(s.sluice.water.len() + s.sluice.outflow.len()) as f64
        * s.sluice.profile.particle_m3()
        - s.sluice.water_credit_m3
        - s.sluice.feed_water_credit_m3;
    let water = if s.sluice.pump_on && s.sluice.intake_has_room() {
        ((s.sluice.flow_m3s * FIXED_DT) as f64).min(room.max(0.))
    } else {
        0.
    };
    s.sluice.water_used_m3 += water;
    s.water_balance.external_inflow_m3 += water;
    s.sluice.inlet.last_water_m3 = water;
    s.sluice.accept_intake(IntakePacket {
        water_m3: water,
        ..Default::default()
    });
    s.sluice.feed(FIXED_DT, water > 0.);
    h.step(s)?;
    Ok(water)
}
fn finish(dir: &Path, s: &Session, mut report: Value) -> Result<(), String> {
    let save_error = mining_sim::storage::save(s, &dir.join("state.save.json")).err();
    report["state_save_error"] = json!(save_error);
    if let Some(error) = save_error {
        report["passed"] = json!(false);
        if report["failure"].is_null() {
            report["failure"] = json!(format!("state save failed: {error}"));
        }
    }
    if let Err(error) = write(&dir.join("report.json"), &report) {
        return Err(format!(
            "{}; report write failed: {error}",
            report["failure"]
                .as_str()
                .unwrap_or("experiment receipt could not be written")
        ));
    }
    println!(
        "{}",
        json!({"output":dir,"passed":report["passed"],"failure":report["failure"]})
    );
    if report["passed"] == true {
        Ok(())
    } else {
        Err("experiment did not pass; receipt retained".into())
    }
}
fn warm(dir: &Path, name: &str, ticks: usize, conservative: bool) -> Result<(), String> {
    let (riffles, flow) = match name {
        "bare_low" => (false, 0.00125),
        "bare_nominal" => (false, 0.00375),
        "bare_high" => (false, 0.0075),
        "riffles_low" => (true, 0.00125),
        "riffles_nominal" => (true, 0.00375),
        "riffles_high" => (true, 0.0075),
        _ => return Err("unknown bare/riffles low/nominal/high case".into()),
    };
    let mut s = Session::new();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.riffles_enabled = riffles;
    s.sluice.flow_m3s = flow;
    s.sluice.pump_on = true;
    s.sluice.feed_on = false;
    let mut h = solver(conservative)?;
    let began = Instant::now();
    let mut failure = None;
    let water_start = s.sluice.water_used_m3;
    let mut water_budget = WaterBudget::new(s.water_balance.initial_stored_m3);
    water_budget.session(&s);
    let mut completed_ticks = 0;
    let mut series = Vec::new();
    for tick in 0..ticks {
        let result = laboratory_tick(&mut s, &mut h);
        water_budget.session(&s);
        if let Err(error) = result {
            first_failure(&mut failure, error);
            break;
        }
        completed_ticks += 1;
        if tick % 60 == 59 {
            series.push(json!({"time_s":s.sim_time,"water_particles":s.sluice.water.len(),"depth_m":s.sluice.water_depth(),
                "mean_downstream_mps":s.sluice.water_speed(),"water_balance_error_m3":s.water_balance_error_m3()}));
            println!(
                "WARM {name} t={:.1}s particles={}",
                s.sim_time,
                s.sluice.water.len()
            );
        }
    }
    water_budget.session(&s);
    if !water_budget.passed() {
        first_failure(&mut failure, "warmup water budget gate failed");
    }
    if !s.sluice.grains.is_empty() || !s.sluice.valid() {
        first_failure(
            &mut failure,
            "warmup requires finite clean grain-free stock",
        );
    }
    let passed = failure.is_none() && completed_ticks == ticks;
    finish(
        dir,
        &s,
        json!({"kind":"prescribed_clean_water_warmup","case":name,"conservative_phases":conservative,
        "riffles":riffles,"flow_m3s":flow,"passed":passed,"failure":failure,"physical_seconds":s.sim_time,
        "wall_seconds":began.elapsed().as_secs_f64(),"delivered_external_water_m3":s.sluice.water_used_m3-water_start,
        "requested_ticks":ticks,"completed_ticks":completed_ticks,"water_budget":water_budget.report(),
        "grain_parcels":s.sluice.grains.len(),"series":series,
        "protocol":"Prescribed clean external water with finite admission and water ledger; isolated laboratory boundary, not native intake circuit or calibration."}),
    )
}
fn volumes(s: &Session) -> [f64; 3] {
    let mut a = [0.; 3];
    for g in &s.sluice.grains {
        if let Some(i) = pulse::bin(g.diameter_m).filter(|_| g.kind == GrainKind::Sand) {
            a[i] += g.solid_volume();
        }
    }
    a
}
fn histogram(s: &Session) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.; 32]; 3];
    for g in &s.sluice.grains {
        if let Some(i) = pulse::bin(g.diameter_m).filter(|_| g.kind == GrainKind::Sand) {
            let x = (g.position.x / 0.05).floor().clamp(0., 31.) as usize;
            a[i][x] += g.solid_volume();
        }
    }
    a
}
fn observation(s: &Session, t: f64, exits: [f64; 3], downstream: [f64; 3]) -> Value {
    let wet = mining_physics::geometry::grain_wetness(&s.sluice);
    json!({"time_s":t,"resident":pulse::resident(&s.sluice.grains,&wet),"cumulative_all_boundary_exit_m3":exits,
        "cumulative_positive_x_only_exit_m3":downstream,"streamwise_histogram_50mm_m3":histogram(s),
        "histogram_last_bin_includes_buffer":true,"water_particles":s.sluice.water.len(),
        "water_mean_downstream_mps":s.sluice.water_speed(),"water_depth_m":s.sluice.water_depth()})
}
fn require_clean_pulse_stock(s: &mining_sim::Sluice) -> Result<(), String> {
    // active_material includes the hopper, ready/pending inlet stock, private
    // pending feed gold and every resident grain. Exact zero is intentional:
    // Material::is_empty permits trace stock below its bulk/display threshold.
    if s.feed_on
        || !s.grains.is_empty()
        || s.active_material() != mining_sim::Material::default()
        || s.feed_water_credit_m3 != 0.
    {
        return Err(
            "pulse requires zero resident, inlet, hopper and pending-feed stock with feed off"
                .into(),
        );
    }
    Ok(())
}
fn run_pulse(
    source_dir: &Path,
    dir: &Path,
    order: usize,
    ticks: usize,
    conservative: bool,
) -> Result<(), String> {
    let source: Value = serde_json::from_slice(
        &std::fs::read(source_dir.join("report.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if source["kind"] != "prescribed_clean_water_warmup"
        || source["passed"] != true
        || source["conservative_phases"] != conservative
    {
        return Err("pulse source must be a passed prescribed clean-water warmup using the same phase model".into());
    }
    let mut s = mining_sim::storage::load(&source_dir.join("state.save.json"))?;
    if s.paused || s.sluice.placement.is_none() {
        return Err("pulse requires an unpaused placed warmup".into());
    }
    require_clean_pulse_stock(&s.sluice)?;
    let mut h = solver(conservative)?;
    // Identical physical restart ticks reconstruct the caches before injection.
    for _ in 0..2 {
        laboratory_tick(&mut s, &mut h)?;
    }
    require_clean_pulse_stock(&s.sluice)?;
    let c = mining_physics::geometry::config(&s.sluice);
    let dx = s.sluice.profile.cell();
    let center = Vec3::new(
        0.32,
        (c.floor_height_at(32) + 1) as f32 * dx + 0.008,
        s.sluice.profile.width() * 0.5 + dx,
    );
    let p = pulse::build(order, center, Vec3::new(0.35, 0., 0.))?;
    let pulse_report = p.report;
    s.sluice.grains = p.grains;
    let initial = volumes(&s);
    let initial_tailings = s.sluice.tailings.mineral_solid_volume_m3();
    let initial_time = s.sim_time;
    let initial_mass = s.accounted_dry_mass_kg();
    let initial_solver_steps = s.sluice.solver_steps;
    let water_start = s.sluice.water_used_m3;
    let initial_boundary = h.boundary_diagnostics(&s);
    let mut failure = match initial_boundary["max_grain_surface_penetration_m"].as_f64() {
        Some(penetration) if penetration.is_finite() && penetration <= 1e-7 => None,
        _ => Some("initial pulse is unsupported or penetrates the actual solver geometry".into()),
    };
    let mut capture_complete = true;
    if let Err(error) = h.enable_grain_exit_capture(4096) {
        first_failure(&mut failure, error);
        capture_complete = false;
    }
    let mut snapshot = Snapshot::default();
    let mut last_snapshot = snapshot_evidence(&snapshot);
    let mut last_domain = None;
    let mut exits = [0.; 3];
    let mut downstream = [0.; 3];
    let mut first_moment = [0.; 3];
    let mut resident_integral = [0.; 3];
    let mut masks = vec![vec![0.; 64]; 3];
    let mut events = Vec::new();
    let mut series = vec![observation(&s, 0., exits, downstream)];
    let mut maximum_relative_closure = 0_f64;
    let mut maximum_tailings_closure = 0_f64;
    let mut closure_finite = true;
    let mut water_budget = WaterBudget::new(s.water_balance.initial_stored_m3);
    water_budget.session(&s);
    let mut completed_ticks = 0;
    let mut elapsed = 0.;
    let began = Instant::now();
    for tick in 0..ticks {
        if failure.is_some() {
            break;
        }
        let before = volumes(&s);
        let result = laboratory_tick(&mut s, &mut h);
        let mut complete_tick = result.is_ok();
        if let Err(error) = result {
            first_failure(&mut failure, error);
        }
        elapsed = s.sim_time - initial_time;
        water_budget.session(&s);
        // A whole-step failure may happen after removals. Preserve its known
        // records and metadata while refusing complete custody qualification.
        match h.take_grain_exit_snapshot(&mut snapshot) {
            Err(error) => {
                last_snapshot = json!({"take_error":error,"status":"unavailable"});
                first_failure(&mut failure, error);
                capture_complete = false;
                complete_tick = false;
            }
            Ok(()) => {
                last_snapshot = snapshot_evidence(&snapshot);
                if snapshot.domain_max_m.is_some() {
                    last_domain = snapshot.domain_max_m;
                }
                if snapshot.status != Status::NoRemovalStage && !snapshot.is_complete() {
                    first_failure(
                        &mut failure,
                        format!("incomplete exit capture: {:?}", snapshot.status),
                    );
                    capture_complete = false;
                    complete_tick = false;
                }
                for event in snapshot.exits() {
                    let bin = pulse::bin(event.grain.diameter_m)
                        .filter(|_| event.grain.kind == GrainKind::Sand);
                    let v = event.grain.solid_volume();
                    if let Some(bin) = bin {
                        exits[bin] += v;
                        first_moment[bin] += v * elapsed;
                        if event.faces == POS_X {
                            downstream[bin] += v;
                        }
                        masks[bin][event.faces as usize] += v;
                    } else {
                        first_failure(&mut failure, "unlabelled grain exited clean experiment");
                        complete_tick = false;
                    }
                    events.push(json!({"time_s":elapsed,"diameter_m":event.grain.diameter_m,"solid_volume_m3":v,
                        "faces":event.faces,"position_m":event.grain.position.to_array(),"velocity_mps":event.grain.velocity.to_array(),
                        "cohort_bin":bin,"grain_index":event.grain_index,"grain":event.grain,
                        "step_capture_status":snapshot.status.as_str(),"whole_step_capture_complete":snapshot.is_complete()}));
                }
            }
        }
        let after = volumes(&s);
        if complete_tick {
            completed_ticks += 1;
            for bin in 0..3 {
                // Left-end inventory integral over completed ticks only;
                // removal-time quantization is at most one fluid interval.
                resident_integral[bin] += before[bin] * FIXED_DT as f64 / initial[bin];
            }
        }
        let actual = s.sluice.tailings.mineral_solid_volume_m3() - initial_tailings;
        closure_finite &= observe_custody(
            initial,
            after,
            exits,
            actual,
            &mut maximum_relative_closure,
            &mut maximum_tailings_closure,
        );
        if tick % 6 == 5 || tick + 1 == ticks || failure.is_some() {
            series.push(observation(&s, elapsed, exits, downstream));
        }
        if tick % 60 == 59 {
            println!("PULSE order={order} t={elapsed:.1}s parcels={} closure={maximum_relative_closure:e}",s.sluice.grains.len());
        }
        if failure.is_some() {
            break;
        }
    }
    // Refresh final evidence even when the last attempted tick failed or
    // refused capture. Previous successful-prefix maxima are not a final pass.
    let final_resident = volumes(&s);
    let final_tailings = s.sluice.tailings.mineral_solid_volume_m3() - initial_tailings;
    closure_finite &= observe_custody(
        initial,
        final_resident,
        exits,
        final_tailings,
        &mut maximum_relative_closure,
        &mut maximum_tailings_closure,
    );
    water_budget.session(&s);
    if !water_budget.passed() {
        first_failure(&mut failure, "pulse water budget gate failed");
    }
    let untracked = s
        .sluice
        .grains
        .iter()
        .any(|g| g.kind != GrainKind::Sand || pulse::bin(g.diameter_m).is_none());
    let custody_passed = failure.is_none()
        && capture_complete
        && completed_ticks == ticks
        && closure_finite
        && maximum_relative_closure <= 1e-6
        && maximum_tailings_closure <= 1e-6
        && !untracked;
    if !custody_passed && failure.is_none() {
        failure = Some("pulse-relative independent custody gate failed".into());
    }
    let passed = failure.is_none()
        && custody_passed
        && s.sluice.valid()
        && water_budget.passed()
        && s.sluice.solver_steps > initial_solver_steps
        && elapsed >= ticks as f64 * FIXED_DT as f64 - 1e-8;
    let conditional_mean: Vec<_> = (0..3)
        .map(|i| {
            if exits[i] > 0. {
                Some(first_moment[i] / exits[i])
            } else {
                None
            }
        })
        .collect();
    finish(
        dir,
        &s,
        json!({"kind":"dilute_fine_pulse","source_warmup":source_dir,"case":source["case"],
        "conservative_phases":conservative,"passed":passed,"custody_passed":custody_passed,"failure":failure,
        "source_clean_stock_verified_before_and_after_restart":true,
        "numerical_transport_convergence_qualified":false,"empirically_calibrated":false,
        "initial_pulse":pulse_report,"initial_boundary_diagnostics":initial_boundary,
        "initial_per_bin_solid_m3":initial,"series":series,"exit_events":events,
        "final_observation":observation(&s,s.sim_time-initial_time,exits,downstream),
        "final_resident_per_bin_solid_m3":final_resident,"actual_final_tailings_delta_m3":final_tailings,
        "exit_face_mask_volumes_m3":masks,"exit_domain_max_m":last_domain,"last_exit_snapshot":last_snapshot,
        "cumulative_all_boundary_exit_m3":exits,"cumulative_positive_x_only_exit_m3":downstream,
        "conditional_mean_observed_exit_time_s":conditional_mean,"restricted_resident_integral_s":resident_integral,
        "maximum_pulse_relative_closure":maximum_relative_closure,"maximum_relative_independent_tailings_error":maximum_tailings_closure,
        "whole_scene_dry_mass_error_kg":s.accounted_dry_mass_kg()-initial_mass,"water_balance_error_m3":s.water_balance_error_m3(),
        "delivered_external_water_m3":s.sluice.water_used_m3-water_start,"physical_seconds":s.sim_time-initial_time,
        "requested_ticks":ticks,"completed_ticks":completed_ticks,"closure_observations_finite":closure_finite,
        "exit_capture_complete":capture_complete&&completed_ticks==ticks,"water_budget":water_budget.report(),
        "wall_seconds":began.elapsed().as_secs_f64(),
        "limits":["Positive X is buffered solver-domain exit, not sluice-lip discharge or cleanup recovery.",
            "Finite-radius weighted contact parcels remain; this is refinement evidence, not resolved-grain calibration.",
            "Cohort custody is independent of whole-site tolerance; model accuracy is a separate gate.",
            "GL rules are nonnested; fixed per-class first two moments do not fix higher moments, contact envelopes or initial overlaps.",
            "Each diameter has a distinct source rotation; cross-size timing is not a size-only comparison.",
            "Exact diameter cohorts and no background grains identify this pulse; no persistent grain IDs are added.",
            "Prescribed clean external water retains original nozzle, admission, outflow and actual FLIP/DEM/terrain stages."]}),
    )
}
fn equivalence(
    load: &Path,
    dir: &Path,
    ticks: usize,
    warmup: usize,
    conservative: bool,
) -> Result<(), String> {
    let mut s = mining_sim::storage::load(load)?;
    if s.paused || s.sluice.placement.is_none() || s.sluice.solver_steps == 0 {
        return Err(
            "equivalence requires an unpaused, placed and previously advanced sluice".into(),
        );
    }
    let initial_time = s.sim_time;
    let initial_steps = s.sluice.solver_steps;
    let requested_ticks = ticks + warmup;
    let mut h = solver(conservative)?;
    h.enable_grain_exit_capture(4096)?;
    let mut snapshot = Snapshot::default();
    let mut records = 0;
    let mut failure = None;
    let mut completed_ticks = 0;
    let mut last_snapshot = snapshot_evidence(&snapshot);
    let mut water_budget = WaterBudget::new(s.water_balance.initial_stored_m3);
    water_budget.session(&s);
    for _ in 0..requested_ticks {
        s.step(Controls::default());
        if let Err(error) = h.step(&mut s) {
            first_failure(&mut failure, error);
        }
        water_budget.session(&s);
        match h.take_grain_exit_snapshot(&mut snapshot) {
            Err(error) => {
                last_snapshot = json!({"take_error":error,"status":"unavailable"});
                first_failure(&mut failure, error);
            }
            Ok(()) => {
                last_snapshot = snapshot_evidence(&snapshot);
                records += snapshot.exits().len();
                if snapshot.status != Status::NoRemovalStage && !snapshot.is_complete() {
                    first_failure(
                        &mut failure,
                        format!("incomplete capture: {:?}", snapshot.status),
                    );
                }
            }
        }
        if failure.is_some() {
            break;
        }
        completed_ticks += 1;
    }
    if let Some(error) = progress_error(
        initial_time,
        s.sim_time,
        initial_steps,
        s.sluice.solver_steps,
        requested_ticks,
        s.paused,
        s.sluice.placement.is_some(),
    ) {
        first_failure(&mut failure, error);
    }
    if !water_budget.passed() {
        first_failure(&mut failure, "equivalence water budget gate failed");
    }
    finish(
        dir,
        &s,
        json!({"kind":"ordinary_pipeline_with_exit_capture","source_save":load,"passed":failure.is_none(),"failure":failure,
        "steps":completed_ticks,"requested_steps":requested_ticks,"observed_exit_records":records,
        "initial_sim_time":initial_time,"final_sim_time":s.sim_time,"physical_seconds":s.sim_time-initial_time,
        "initial_solver_steps":initial_steps,"final_solver_steps":s.sluice.solver_steps,
        "last_exit_snapshot":last_snapshot,"water_budget":water_budget.report(),
        "canonical_comparison_required":true,"conservative_phases":conservative}),
    )
}
enum Operation {
    Warm {
        name: String,
        ticks: usize,
    },
    Pulse {
        source: PathBuf,
        order: usize,
        ticks: usize,
    },
    Equivalence {
        source: PathBuf,
        ticks: usize,
        warmup: usize,
    },
}
fn parse(a: &[String]) -> Result<(Operation, String), String> {
    match a.first().map(String::as_str) {
        Some("warm") if a.len() == 4 => {
            if !["bare_low", "bare_nominal", "bare_high", "riffles_low", "riffles_nominal", "riffles_high"].contains(&a[1].as_str()) {
                return Err("unknown bare/riffles low/nominal/high case".into());
            }
            Ok((Operation::Warm { name: a[1].clone(), ticks: seconds(&a[3], 4., 32.)? }, a[2].clone()))
        }
        Some("pulse") if a.len() == 5 => {
            let order = a[3].parse().map_err(|_| "invalid order")?;
            if ![2, 4, 6].contains(&order) {
                return Err("pulse order must be 2, 4 or 6".into());
            }
            Ok((Operation::Pulse { source: PathBuf::from(&a[1]), order, ticks: seconds(&a[4], 1., 120.)? }, a[2].clone()))
        }
        Some("equivalence") if a.len() == 5 => {
            let ticks = a[3].parse().map_err(|_| "invalid ticks")?;
            let warmup = a[4].parse().map_err(|_| "invalid warmup")?;
            if !(1..=600).contains(&ticks) || warmup > 120 {
                return Err("comparison tick count exceeds bound".into());
            }
            Ok((Operation::Equivalence { source: PathBuf::from(&a[1]), ticks, warmup }, a[2].clone()))
        }
        _ => Err("slurry_transport warm CASE NEW_DIR SECONDS | pulse WARM_DIR NEW_DIR ORDER SECONDS | equivalence SAVE NEW_DIR TICKS WARMUP [--conservative]".into()),
    }
}
fn main() -> Result<(), String> {
    let mut a: Vec<_> = std::env::args().skip(1).collect();
    let conservative = a.iter().any(|s| s == "--conservative");
    a.retain(|s| s != "--conservative");
    let (operation, output) = parse(&a)?;
    let dir = directory(&output)?;
    let request = json!({"arguments":a,"conservative_phases":conservative});
    let result = match operation {
        Operation::Warm { name, ticks } => warm(&dir, &name, ticks, conservative),
        Operation::Pulse {
            source,
            order,
            ticks,
        } => run_pulse(&source, &dir, order, ticks, conservative),
        Operation::Equivalence {
            source,
            ticks,
            warmup,
        } => equivalence(&source, &dir, ticks, warmup, conservative),
    };
    retain_setup_failure(&dir, &request, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_pulse_guard_rejects_trace_stock_and_grains_emitted_during_restart() {
        let mut s = mining_sim::Sluice::default();
        assert!(require_clean_pulse_stock(&s).is_ok());
        for pending in [false, true] {
            for channel in 0..4 {
                let stock = if pending {
                    &mut s.inlet.pending
                } else {
                    &mut s.inlet.ready
                };
                stock.solid_m3[channel] = 1e-12;
                assert!(
                    s.active_material().is_empty(),
                    "trace fixture must exercise the old tolerance hole"
                );
                assert!(require_clean_pulse_stock(&s).is_err());
                s.inlet = Default::default();
            }
        }
        s.hopper.gold_g = 1e-12;
        assert!(require_clean_pulse_stock(&s).is_err());
        s.hopper = Default::default();
        let mut saved = serde_json::to_value(&s).unwrap();
        saved["pending_gold_g"] = json!(1e-12);
        let pending_feed: mining_sim::Sluice = serde_json::from_value(saved).unwrap();
        assert!(pending_feed.active_material().is_empty());
        assert!(require_clean_pulse_stock(&pending_feed).is_err());
        s.feed_water_credit_m3 = 1e-12;
        assert!(require_clean_pulse_stock(&s).is_err());
        s.feed_water_credit_m3 = 0.;
        s.placement = Some(Placement {
            inlet: Vec3::ZERO,
            yaw: 0.,
            slope_deg: 6.,
        });
        s.pump_on = true;
        s.inlet.pending.solid_m3[0] = 2e-6;
        assert!(require_clean_pulse_stock(&s).is_err());
        s.emit_intake_grains(1. / 30.);
        assert!(
            !s.grains.is_empty(),
            "actual inlet emission must populate the formerly empty array"
        );
        assert!(
            require_clean_pulse_stock(&s).is_err(),
            "post-restart check must precede pulse replacement"
        );
    }

    #[test]
    fn water_gate_retains_intermediate_losses_and_tests_local_custody_independently() {
        let mut budget = WaterBudget::new(2.);
        budget.observe(1e-6, -5e-11);
        assert!(budget.passed());
        budget.observe(-3e-6, 0.);
        budget.observe(0., 0.);
        assert!(
            !budget.passed(),
            "later balance cannot erase the earlier loss"
        );
        assert_eq!(budget.report()["maximum_finite_abs_whole_error_m3"], 3e-6);
        let mut large_scene = WaterBudget::new(1e12);
        large_scene.observe(1., 1.01e-10);
        assert_eq!(large_scene.report()["whole_water_passed"], true);
        assert_eq!(large_scene.report()["sluice_water_passed"], false);
        assert!(
            !large_scene.passed(),
            "whole-scene tolerance must not hide local loss"
        );
    }

    #[test]
    fn nonfinite_water_observations_never_recover_a_passing_status() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for local in [false, true] {
                let mut budget = WaterBudget::new(2.);
                budget.observe(
                    if local { 0. } else { invalid },
                    if local { invalid } else { 0. },
                );
                budget.observe(0., 0.);
                assert!(!budget.passed());
                assert_eq!(budget.report()["all_observations_finite"], false);
            }
        }
    }

    #[test]
    fn equivalence_requires_time_and_solver_progress_and_a_live_placed_scene() {
        let end = 90. + 12. * FIXED_DT as f64;
        assert!(progress_error(90., end, 100, 124, 12, false, true).is_none());
        for (time, steps, paused, placed) in [
            (90., 100, false, true),
            (end, 100, false, true),
            (end, 124, true, true),
            (end, 124, false, false),
            (end + 1., 124, false, true),
            (f64::NAN, 124, false, true),
        ] {
            assert!(progress_error(90., time, 100, steps, 12, paused, placed).is_some());
        }
    }

    #[test]
    fn actual_tailings_supply_an_independent_pulse_closure_gate() {
        let initial = [1e-8; 3];
        let resident = [8e-9, 9e-9, 5e-9];
        let exits = [2e-9, 1e-9, 5e-9];
        let correct = custody_errors(initial, resident, exits, 8e-9).unwrap();
        assert!(correct.0 < 1e-14 && correct.1 < 1e-14);
        let wrong_tailings = custody_errors(initial, resident, exits, 9e-9).unwrap();
        assert!(wrong_tailings.0 < 1e-14 && wrong_tailings.1 > 0.03);
        let wrong_capture = custody_errors(initial, resident, [1e-9, 1e-9, 5e-9], 8e-9).unwrap();
        assert!(wrong_capture.0 > 0.09 && wrong_capture.1 > 0.03);
        assert!(custody_errors(initial, resident, exits, f64::NAN).is_none());
    }

    #[test]
    fn secondary_failure_cannot_replace_the_original_solver_error() {
        let mut failure = None;
        first_failure(&mut failure, "original solver refusal");
        first_failure(&mut failure, "incomplete capture after refusal");
        first_failure(&mut failure, "water gate failed after refusal");
        assert_eq!(failure.as_deref(), Some("original solver refusal"));
    }

    #[test]
    fn setup_failures_get_a_nonpass_receipt_without_overwriting_existing_evidence() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "sluice-transport-receipt-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        let request = json!({"fixture":"missing source"});
        let result = retain_setup_failure(&dir, &request, Err("original load failure".into()));
        assert_eq!(result.unwrap_err(), "original load failure");
        let path = dir.join("report.json");
        let before = std::fs::read(&path).unwrap();
        let receipt: Value = serde_json::from_slice(&before).unwrap();
        assert_eq!(receipt["passed"], false);
        assert_eq!(receipt["failure"], "original load failure");
        assert!(retain_setup_failure(&dir, &request, Err("later failure".into())).is_err());
        assert!(write(&path, &json!({"passed":true})).is_err());
        assert_eq!(before, std::fs::read(&path).unwrap());
        let unavailable = retain_setup_failure(
            &dir.join("missing-parent"),
            &request,
            Err("first error".into()),
        )
        .unwrap_err();
        assert!(unavailable.starts_with("first error; failure receipt could not be retained:"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
