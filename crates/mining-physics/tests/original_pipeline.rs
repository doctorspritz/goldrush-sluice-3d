//! GPU integration, explicitly opt in. Failure to obtain an adapter is a failure,
//! never a passing test. CI compiles this; a Metal/Vulkan host runs it with --ignored.
use mining_physics::Hydraulics;
use mining_sim::{session::Controls, Action, Placement, Session, Vec3};
#[test]
#[ignore = "requires real Metal/Vulkan compute adapter; run with --ignored --nocapture"]
fn original_pipeline_survives_excavation_pumping_camera_and_live_restore() {
    run_pipeline(false, false, false, false);
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --nocapture"]
fn conservative_pipeline_survives_excavation_pumping_camera_and_live_restore() {
    run_pipeline(true, false, false, false);
}
#[test]
#[ignore = "requires actual GPU; run explicitly with --ignored --nocapture"]
fn articulated_machine_survives_original_water_pipeline_and_full_live_restore() {
    run_pipeline(false, true, false, false);
}
#[test]
#[ignore = "requires real GPU; explicitly run with --ignored --nocapture"]
fn spatial_payload_exchange_survives_the_water_pipeline_and_shell_restore() {
    run_pipeline(false, true, true, false);
}
#[test]
#[ignore = "requires actual GPU; run explicitly with --ignored"]
fn moving_soil_survives_original_water_pipeline_and_shell_restore() {
    run_pipeline(false, true, false, true);
}
fn run_pipeline(conservative: bool, articulated: bool, exchange: bool, soil: bool) {
    let mut s = Session::new();
    if articulated {
        s.machine.enable_articulation().unwrap();
    }
    if soil {
        s.activate_soil(&[(100 * mining_sim::site::WIDTH + 104, 0.12)], 2)
            .unwrap();
    }
    let initial_mass = s.accounted_dry_mass_kg();
    if exchange {
        let batch = s.site.dig(Vec3::new(19., 0., 20.), 0.4, 0.2, 0.04);
        let (w, q) = mining_sim::articulation::payload_frame(&s.machine);
        let parcel = mining_sim::payload::RigidParcel::cuboid(
            batch,
            w + q * glam::DVec3::new(0., -0.05, 0.45),
            glam::DVec3::new(0.2, batch.volume() / (8. * 0.2 * 0.25), 0.25),
            q,
            glam::DVec3::new(0.1, -0.2, 0.3),
            glam::DVec3::new(0., 0.3, 0.),
        )
        .unwrap();
        s.falling.push(parcel.into_falling());
        s.capture_rigid_load(0, 1.).unwrap();
        s.release_rigid_payload(0.3).unwrap();
        assert!(s
            .machine
            .articulation
            .as_ref()
            .unwrap()
            .payload_moments
            .is_some());
    }

    let mut physics = Hydraulics::new().unwrap();
    if conservative {
        physics = physics.with_conservative_phases();
    }
    let initial_layers = s.site.world.overburden_thickness.clone();
    let batch = s.site.dig(Vec3::new(19.0, 0.0, 22.0), 0.44, 0.3, 0.08);
    assert!(batch.volume() > 0.05);
    assert!(s.sluice.add_to_hopper(batch).is_empty());
    s.player.position = Vec3::new(14.6, 0.35, 15.0);
    s.on_action(Action::Place(Placement::suggested(&s.site)))
        .unwrap();
    s.on_action(Action::TogglePump).unwrap();
    s.on_action(Action::ToggleFeed).unwrap();
    let started = std::time::Instant::now();
    for tick in 0..360 {
        s.step(Controls::default());
        assert!(!s.paused, "pipeline paused: {}", s.message);
        physics
            .step(&mut s)
            .unwrap_or_else(|error| panic!("pipeline tick {tick}: {error}"));
        if tick == 300 {
            let positions: Vec<_> = s.sluice.water.iter().map(|p| p.position).collect();
            let outflow: Vec<_> = s
                .sluice
                .outflow
                .iter()
                .map(|p| (p.position, p.velocity))
                .collect();
            assert!(!outflow.is_empty(), "save the real airborne stream");
            let affine: Vec<_> = s.sluice.water.iter().map(|p| p.affine).collect();
            let momentum = s.site.hydraulic.velocity_x.clone();
            let suspend = s.site.hydraulic.suspended_paydirt.clone();
            let bed_age = s.site.hydraulic.settling_time.clone();
            let exchange_error = s.site.hydraulic.bed_exchange_error.clone();
            assert!(
                exchange_error.iter().flatten().any(|error| *error != 0.),
                "roundoff state was not exercised"
            );
            assert!(
                bed_age.iter().any(|age| *age > 255),
                "physical bed clock never advanced"
            );
            assert_eq!(
                s.site.hydraulic.settling_time_unit,
                mining_sim::site::BedAgeUnit::Microseconds
            );
            println!(
                "PRE SAVE: hydraulic={} gold_min={:?} suspended_min={:?} clock={} max_vel={}",
                s.site.hydraulic.valid(),
                s.site.gold_g.iter().copied().reduce(f64::min),
                s.site
                    .hydraulic
                    .suspended_gold_g
                    .iter()
                    .copied()
                    .reduce(f64::min),
                s.site.hydraulic.clock,
                s.site
                    .world
                    .water_flow_x
                    .iter()
                    .chain(&s.site.world.water_flow_z)
                    .map(|v| v.abs())
                    .fold(0.0, f32::max)
            );
            let machine = serde_json::to_string(&s.machine).unwrap();
            let body_motion = serde_json::to_string(&s.material_motion).unwrap();
            let moving_soil = serde_json::to_string(&s.active_soil).unwrap();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&s.save_json().unwrap()).unwrap()
                    ["version"],
                if soil || exchange || articulated {
                    10
                } else {
                    6
                }
            );
            if exchange {
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&s.save_json().unwrap()).unwrap()
                        ["version"],
                    10
                );
            }

            s = Session::load_json(&s.save_json().unwrap()).unwrap();
            assert_eq!(serde_json::to_string(&s.machine).unwrap(), machine);
            assert_eq!(serde_json::to_string(&s.active_soil).unwrap(), moving_soil);
            assert_eq!(
                serde_json::to_string(&s.material_motion).unwrap(),
                body_motion
            );

            assert_eq!(s.machine.articulation.is_some(), articulated);
            physics.restore();
            assert_eq!(
                positions,
                s.sluice
                    .water
                    .iter()
                    .map(|p| p.position)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                affine,
                s.sluice.water.iter().map(|p| p.affine).collect::<Vec<_>>()
            );
            assert_eq!(momentum, s.site.hydraulic.velocity_x);
            assert_eq!(suspend, s.site.hydraulic.suspended_paydirt);
            assert_eq!(bed_age, s.site.hydraulic.settling_time);
            assert_eq!(exchange_error, s.site.hydraulic.bed_exchange_error);
            assert_eq!(
                s.site.hydraulic.settling_time_unit,
                mining_sim::site::BedAgeUnit::Microseconds
            );
            s.on_action(Action::Camera).unwrap();
            s.on_action(Action::Camera).unwrap();
            assert_eq!(positions.len(), s.sluice.water.len());
            assert_eq!(
                outflow,
                s.sluice
                    .outflow
                    .iter()
                    .map(|p| (p.position, p.velocity))
                    .collect::<Vec<_>>()
            );
        }
    }
    assert_eq!(s.site.hydraulic.steps, 120);
    let mass = physics.material_diagnostics(&s);
    assert_eq!(mass["ready"], true);
    assert_eq!(mass["legacy_grains"], 0);
    let inventory = mass["inventory_dry_mass_kg"].as_f64().unwrap();
    assert!(inventory > 0.);
    assert!(
        mass["dem_mass_error_kg"].as_f64().unwrap().abs() < inventory * 1e-6,
        "DEM mass handoff: {mass}"
    );
    assert!(
        (s.accounted_dry_mass_kg() - initial_mass).abs() < initial_mass * 1e-6,
        "full dry-mass budget"
    );
    println!("ACTUAL MATERIAL MASS: {mass}");
    assert!(s.sluice.solver_steps >= 180 * mining_physics::FLIP_SUBSTEPS as u64);
    assert!(
        s.sluice.water.iter().any(|p| p.position.x > 0.4),
        "original FLIP advection must move inlet water downstream"
    );
    assert!(s
        .site
        .hydraulic
        .velocity_x
        .iter()
        .chain(&s.site.hydraulic.velocity_z)
        .any(|v| v.abs() > 0.01));
    assert!(s
        .site
        .world
        .overburden_thickness
        .iter()
        .zip(initial_layers)
        .any(|(a, b)| (a - b).abs() > 0.0001));
    assert!(
        s.sluice
            .water
            .iter()
            .any(|p| p.velocity.length_squared() > 1e-4),
        "water must retain velocity from the original FLIP/PIC update"
    );
    assert!(!s.sluice.grains.is_empty());
    assert!(
        s.sluice
            .water
            .iter()
            .all(|p| p.position.x < s.sluice.profile.length()),
        "no water may remain on the old outlet-buffer floor"
    );
    assert!(
        !s.sluice.outflow.is_empty(),
        "real GPU discharge must reach free flight"
    );
    assert!(
        s.sluice.water_returned_m3 > 0.,
        "discharge must land on terrain"
    );
    assert!(s.site.hydraulic.valid() && s.sluice.valid());
    assert!(
        s.water_balance_error_m3().abs() < 1e-4,
        "whole-system water balance: {}",
        s.water_balance_error_m3()
    );
    assert!(
        s.water_balance.boundary_outflow_m3 > 0.,
        "river must drain downhill"
    );
    let water = s.sluice.accounted_water_m3();
    assert!(
        (s.sluice.water_used_m3 - water).abs() < 1e-7,
        "intake != FLIP + airborne + returned + credit"
    );
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-3);
    println!("ORIGINAL PIPELINE: adapter={} real_seconds={:.2} water={} sediment={} SWE_steps={} FLIP_steps={} water_error={:.9} gold_error={:.9}",
        physics.adapter_name,started.elapsed().as_secs_f64(),s.sluice.water.len(),s.sluice.grains.len(),s.site.hydraulic.steps,s.sluice.solver_steps,
        s.sluice.water_used_m3-water,s.accounted_gold()-s.initial_gold_g);
}
