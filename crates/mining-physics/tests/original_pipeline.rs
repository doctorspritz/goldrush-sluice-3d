//! GPU integration, explicitly opt in. Failure to obtain an adapter is a failure,
//! never a passing test. CI compiles this; a Metal/Vulkan host runs it with --ignored.
use mining_physics::Hydraulics;
use mining_sim::{session::Controls, sluice::WATER_PARTICLE_M3, Action, Placement, Session, Vec3};
#[test]
#[ignore = "requires real Metal/Vulkan compute adapter; run with --ignored --nocapture"]
fn original_pipeline_survives_excavation_pumping_camera_and_live_restore() {
    let mut s = Session::new();
    let mut physics = Hydraulics::new().unwrap();
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
        physics.step(&mut s).unwrap();
        if tick == 200 {
            let positions: Vec<_> = s.sluice.water.iter().map(|p| p.position).collect();
            let affine: Vec<_> = s.sluice.water.iter().map(|p| p.affine).collect();
            let momentum = s.site.hydraulic.velocity_x.clone();
            let suspend = s.site.hydraulic.suspended_paydirt.clone();
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
            s = Session::load_json(&s.save_json().unwrap()).unwrap();
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
            s.on_action(Action::Camera).unwrap();
            s.on_action(Action::Camera).unwrap();
            assert_eq!(positions.len(), s.sluice.water.len());
        }
    }
    assert_eq!(s.site.hydraulic.steps, 120);
    assert!(s.sluice.solver_steps >= 1440);
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
    assert!(!s.sluice.grains.is_empty());
    assert!(s.site.hydraulic.valid() && s.sluice.valid());
    let water = s.sluice.water.len() as f64 * WATER_PARTICLE_M3
        + s.sluice.water_credit_m3
        + s.sluice.water_returned_m3;
    assert!(
        (s.sluice.water_used_m3 - water).abs() < 1e-7,
        "intake != FLIP + returned + credit"
    );
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-3);
    println!("ORIGINAL PIPELINE: adapter={} real_seconds={:.2} water={} sediment={} SWE_steps={} FLIP_steps={} water_error={:.9} gold_error={:.9}",
        physics.adapter_name,started.elapsed().as_secs_f64(),s.sluice.water.len(),s.sluice.grains.len(),s.site.hydraulic.steps,s.sluice.solver_steps,
        s.sluice.water_used_m3-water,s.accounted_gold()-s.initial_gold_g);
}
