//! Serial laboratory trials with explicit finite inputs and honest
//! residence/cleanup/outlet assays. Run one case per process.
use glam::{Mat3, Quat};
use mining_physics::calibration::{self, Assay, DIAMETERS_M, TRACERS_PER_SIZE};
use mining_sim::{session::Controls, sluice::Grain, GrainKind, Material, Placement, Session, Vec3};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("sluice_accuracy OUTPUT_DIRECTORY CASE [TRACER_SECONDS=30]");
        for c in calibration::CASES {
            println!(
                "{} water={}L/s feed={}L/s parcel={}mL",
                c.name,
                c.flow_m3s * 1000.,
                c.feed_m3s * 1000.,
                c.parcel_m3 * 1e6
            );
        }
        return;
    }
    let dir = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&dir).unwrap();
    if args[2] == "--cleanup" {
        let path = PathBuf::from(args.get(3).expect("saved trial path"));
        let name = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_suffix(".save.json")
            .expect("trial.save.json");
        let output = dir.join(format!("{name}.cleanup.json"));
        assert!(!output.exists(), "preserve existing cleanup receipt");
        let mut s = mining_sim::storage::load(&path).unwrap();
        let before = Assay::resident(&s);
        let result = s.sluice.cleanup();
        let after = Assay::resident(&s);
        let gold: Vec<f64> = before
            .gold_g
            .iter()
            .zip(after.gold_g)
            .map(|(a, b)| a - b)
            .collect();
        let report = json!({"source_save":path,"protocol":"settled_mat_fraction",
            "sim_seconds":s.sim_time,"cleanup_allowed":result.is_ok(),"cleanup_error":result.err(),
            "diameters_m":DIAMETERS_M,"collected_gold_by_size_g":gold,
            "resident_before":before.json(),"resident_after":after.json(),
            "water_error_m3":s.water_balance_error_m3(),"gold_error_g":s.accounted_gold()-s.initial_gold_g});
        std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("{report}");
        return;
    }
    let case = calibration::CASES
        .iter()
        .find(|c| c.name == args[2])
        .expect("known case");
    let duration: f32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(30.);
    assert!((8.0..=600.0).contains(&duration));
    let report = dir.join(format!("{}.json", case.name));
    assert!(
        !report.exists(),
        "preserve existing trial; choose a new output directory"
    );
    let started = Instant::now();
    let mut s = Session::new();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.flow_m3s = case.flow_m3s;
    s.sluice.feed_m3s = case.feed_m3s;
    s.sluice.parcel_volume_m3 = Some(case.parcel_m3);
    s.sluice.contact_model = case.contact;
    s.sluice.riffles_enabled = case.riffles;
    s.sluice.pump_on = true;
    // Explicit laboratory feed: homogeneous quartz gravel, no hidden assay.
    s.sluice.hopper = Material {
        gravel_m3: 0.08,
        ..Default::default()
    };
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    let mut series = Vec::new();
    let mut advance =
        |s: &mut Session, seconds: f32, phase: &str, solver: &mut mining_physics::Hydraulics| {
            for tick in 0..(seconds * 60.).round() as usize {
                s.step(Controls::default());
                solver.step(s).unwrap();
                if tick % 300 == 299 {
                    let mut observation = calibration::observe(s);
                    observation["phase"] = json!(phase);
                    println!(
                        "TRIAL {} {} t={:.1}s grains={} water_error={:.9}m3",
                        case.name,
                        phase,
                        s.sim_time,
                        s.sluice.grains.len(),
                        s.water_balance_error_m3()
                    );
                    series.push(observation);
                }
            }
        };
    advance(&mut s, 8., "warmup", &mut solver);
    s.sluice.feed_on = true;
    advance(&mut s, 12., "bed_loading", &mut solver);
    for (bin, d) in DIAMETERS_M.into_iter().enumerate() {
        for i in 0..TRACERS_PER_SIZE {
            for (kind, density) in [(GrainKind::Gold, 19300.), (GrainKind::Sand, 2650.)] {
                let volume = std::f64::consts::PI / 6. * (d as f64).powi(3);
                let gold = kind == GrainKind::Gold;
                let material = if gold {
                    Material {
                        gold_g: volume * density * 1000.,
                        ..Default::default()
                    }
                } else {
                    Material::sand_from_solid_volume(volume)
                };
                s.sluice.grains.push(Grain {
                    volume_resolved: true,
                    kind,
                    diameter_m: d,
                    material,
                    position: Vec3::new(
                        0.34 + bin as f32 * 0.008,
                        (s.sluice.floor_cells() + 1) as f32 * 0.01
                            - 0.34 * s.sluice.placement.unwrap().slope_deg.to_radians().tan()
                            + 0.04,
                        0.05 + i as f32 * 0.012 + if gold { 0.003 } else { 0. },
                    ),
                    velocity: Vec3::new(0.35, 0., 0.),
                    resting: false,
                    affine: Mat3::ZERO,
                    rotation: Quat::IDENTITY,
                    angular_velocity: Vec3::ZERO,
                });
                s.initial_gold_g += material.gold_g;
            }
        }
    }
    let input = Assay::resident(&s);
    let initial_mass = s.accounted_dry_mass_kg();
    let mass_start = s.sim_time;
    let pumped = s.sluice.water_used_m3;
    let processed = s.sluice.processed_m3;
    advance(&mut s, duration, "tracer_with_feed", &mut solver);
    let wash_assay = Assay::resident(&s);
    let delivered_water = (s.sluice.water_used_m3 - pumped) / duration as f64;
    let delivered_feed = (s.sluice.processed_m3 - processed) / duration as f64;
    s.sluice.feed_on = false;
    advance(&mut s, 10., "clear_water_flush", &mut solver);
    let flushed = Assay::resident(&s);
    s.sluice.pump_on = false;
    advance(&mut s, 15., "draining", &mut solver);
    let before_cleanup = Assay::resident(&s);
    let registered_material = solver.material_diagnostics(&s);
    let cleanup = s.sluice.cleanup();
    let cleanup_allowed = cleanup.is_ok();
    let after_cleanup = Assay::resident(&s);
    let mut bins = Vec::new();
    for (i, d) in DIAMETERS_M.iter().enumerate() {
        let injected = input.gold_g[i];
        let recovered = before_cleanup.gold_g[i] - after_cleanup.gold_g[i];
        let lost = injected - before_cleanup.gold_g[i];
        bins.push(json!({"diameter_m":d,"injected_gold_g":injected,
            "outlet_gold_g":lost,"cleanup_gold_g":recovered,"still_resident_gold_g":after_cleanup.gold_g[i],
            "outlet_loss_fraction":lost/injected,
            "actual_cleanup_fraction":if cleanup_allowed {Some(recovered/injected)} else {None},
            "wash_resident_fraction":wash_assay.gold_g[i]/injected,
            "post_clear_flush_resident_fraction":flushed.gold_g[i]/injected,
            "quartz_outlet_fraction":1.-before_cleanup.quartz_m3[i]/input.quartz_m3[i],
            "gold_mass_residual_g":injected-lost-recovered-after_cleanup.gold_g[i]}));
    }
    let loss_from_bins: f64 = bins
        .iter()
        .map(|b| b["outlet_gold_g"].as_f64().unwrap())
        .sum();
    assert!((loss_from_bins - s.sluice.tailings.gold_g).abs() < 1e-7);
    let dry_mass_error = s.accounted_dry_mass_kg() - initial_mass;
    let valid = s.water_balance_error_m3().abs() < 1e-4
        && dry_mass_error.abs() < initial_mass * 1e-6
        && (s.accounted_gold() - s.initial_gold_g).abs() < 1e-7
        && s.sluice.valid();
    let achieved = (0.98..=1.02).contains(&(delivered_feed / case.feed_m3s as f64))
        && (0.98..=1.02).contains(&(delivered_water / case.flow_m3s as f64));
    let mut data = json!({"schema_version":4,"component":"sluice","case":case.name,
        "model_revision":"2026-10-22-reference-bulk-mass",
        "volume_basis":{"feed_and_parcels":"reference_bulk","quartz_assay":"mineral_grain_solids"},
        "requested_conditions_achieved":achieved,
        "boundaries":solver.boundary_diagnostics(&s),
        "contact_model":case.contact,"riffles":case.riffles,"cell_m":s.sluice.profile.cell(),
        "fluid_hz":120,"pressure_iterations":[80,40],"tracers_per_size":TRACERS_PER_SIZE,
        "set_water_m3s":case.flow_m3s,"delivered_water_m3s":delivered_water,
        "set_feed_m3s":case.feed_m3s,"delivered_feed_m3s":delivered_feed,
        "feed_delivery_fraction":delivered_feed/case.feed_m3s as f64,
        "water_delivery_fraction":delivered_water/case.flow_m3s as f64,
        "parcel_m3":case.parcel_m3,"slope_deg":s.sluice.placement.unwrap().slope_deg,
        "warmup_seconds":8,"bed_loading_seconds":12,"tracer_seconds":duration,
        "clear_flush_seconds":10,"drain_seconds":15,
        "cleanup_allowed":cleanup_allowed,"cleanup_error":cleanup.err(),
        "mass_and_finite_checks_passed":valid,"interphase_momentum_verified":false,
        "externally_validated_recovery":false,
        "sim_seconds":s.sim_time,"wall_seconds":started.elapsed().as_secs_f64(),
        "whole_water_error_m3":s.water_balance_error_m3(),"gold_error_g":s.accounted_gold()-s.initial_gold_g,
        "size_assay":bins,"observations":series,
        "limits":["weighted bulk parcels","voxel ramp and riffles","unresolved mat fibres","no calibrated flake drag","not matched to an experimental geometry","24 deterministic lateral positions per size; no statistical confidence claim"]});
    data["mass_budget"] = json!({"start_sim_time":mass_start,
        "dry_mass_error_kg":dry_mass_error,"dry_mass_tolerance_kg":initial_mass*1e-6,
        "registered_material_before_cleanup":registered_material});
    mining_sim::storage::save(&s, &dir.join(format!("{}.save.json", case.name))).unwrap();
    std::fs::write(report, serde_json::to_vec_pretty(&data).unwrap()).unwrap();
    println!(
        "TRIAL {} complete mass_checks={} cleanup={} wall={:.1}s",
        case.name,
        valid,
        cleanup_allowed,
        started.elapsed().as_secs_f64()
    );
    assert!(valid, "numerical checks failed; receipt retained");
}
