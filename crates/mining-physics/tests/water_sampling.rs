//! Actual full-width FLIP/pressure/DEM and saved finite-pulse checks. Reduced
//! marker quadrature is opt-in; these are not empirical recovery calibration.
use glam::{Mat3, Quat};
use mining_physics::Hydraulics;
use mining_sim::{
    session::Controls,
    sluice::{Grain, WaterParticle, WaterSampling},
    Action, GrainKind, Material, Placement, Session, Vec3,
};
fn modes() -> [WaterSampling; 3] {
    [
        WaterSampling::Eight,
        WaterSampling::Four,
        WaterSampling::Two,
    ]
}
fn canonical(s: &Session) -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    let shift = if v["version"] == 15 {
        &mut v["shift"]
    } else {
        &mut v
    };
    for name in ["spheres", "planes", "sdf"] {
        if let Some(rows) = shift["sluice"]["contacts"][name].as_array_mut() {
            rows.sort_by_key(|v| {
                serde_json::to_string(&v.as_array().unwrap()[..v.as_array().unwrap().len() - 1])
                    .unwrap()
            });
        }
    }
    v
}
fn cloud(sampling: WaterSampling) -> Session {
    let mut s = Session::new();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.set_water_sampling(sampling).unwrap();
    // Co-located equal-velocity submarkers describe identical quadrature and
    // physical carrier volume at every support point, independent of count.
    for x in 58..64 {
        for y in 24..28 {
            for z in 17..21 {
                for _ in 0..sampling.markers_per_cell() {
                    let id = s.sluice.allocate_water_id();
                    s.sluice.water.push(WaterParticle {
                        id,
                        position: Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * 0.01,
                        velocity: Vec3::new(0.25, 0., 0.),
                        affine: Mat3::ZERO,
                    });
                }
            }
        }
    }
    for (kind, position, material, diameter_m) in [
        (
            GrainKind::Sand,
            Vec3::new(0.595, 0.255, 0.185),
            Material {
                sand_m3: 0.0000000625,
                ..Default::default()
            },
            0.0008,
        ),
        (
            GrainKind::Gold,
            Vec3::new(0.625, 0.255, 0.185),
            Material {
                gold_g: 0.0003,
                ..Default::default()
            },
            0.0004,
        ),
    ] {
        s.sluice.grains.push(Grain {
            volume_resolved: true,
            resting: false,
            material,
            kind,
            diameter_m,
            position,
            velocity: Vec3::new(-0.125, 0., 0.),
            affine: Mat3::ZERO,
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        });
    }
    s
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn changing_water_quadrature_preserves_carrier_volume_and_actual_grain_reaction() {
    let mut outputs = Vec::new();
    for sampling in modes() {
        let mut s = cloud(sampling);
        let initial = s.sluice.stored_water_m3();
        let mut physics = Hydraulics::new().unwrap();
        if sampling == WaterSampling::Eight {
            physics = physics.with_conservative_phases();
        }
        for _ in 0..2 {
            physics.step(&mut s).unwrap();
        }
        assert_eq!(
            s.sluice.water.len(),
            96 * sampling.markers_per_cell() as usize
        );
        assert_eq!(s.sluice.stored_water_m3(), initial);
        let diagnostic = physics.material_diagnostics(&s);
        assert!(diagnostic["dem_mass_error_kg"].as_f64().unwrap().abs() < 1e-9);
        outputs.push(s);
    }
    for candidate in &outputs[1..] {
        assert_eq!(
            candidate.sluice.stored_water_m3(),
            outputs[0].sluice.stored_water_m3()
        );
        for (reference, grain) in outputs[0]
            .sluice
            .grains
            .iter()
            .zip(&candidate.sluice.grains)
        {
            assert_eq!(reference.material, grain.material);
            assert_eq!(reference.diameter_m, grain.diameter_m);
            assert!(
                (reference.position - grain.position).length() < 2e-6,
                "position {:?} vs {:?}",
                reference.position,
                grain.position
            );
            assert!(
                (reference.velocity - grain.velocity).length() < 2e-4,
                "velocity {:?} vs {:?}",
                reference.velocity,
                grain.velocity
            );
        }
    }
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn reduced_water_sampling_runs_full_width_feed_outlet_and_saved_continuation() {
    let mut reports = Vec::new();
    for sampling in modes() {
        let mut s = Session::new();
        s.sluice.set_water_sampling(sampling).unwrap();
        let feed = s.site.dig(Vec3::new(19., 0., 22.), 0.44, 0.3, 0.08);
        assert!(feed.volume() > 0.05);
        assert!(s.sluice.add_to_hopper(feed).is_empty());
        s.player.position = Vec3::new(14.6, 0.35, 15.);
        s.on_action(Action::Place(Placement::suggested(&s.site)))
            .unwrap();
        s.enable_screen().unwrap();
        s.plant.as_mut().unwrap().screen.motor_on = true;
        s.on_action(Action::TogglePump).unwrap();
        s.on_action(Action::ToggleFeed).unwrap();
        let initial_mass = s.accounted_dry_mass_kg();
        let initial_gold = s.accounted_gold();
        let mut physics = Hydraulics::new().unwrap();
        if sampling == WaterSampling::Eight {
            physics = physics.with_conservative_phases();
        }
        let start = std::time::Instant::now();
        let mut fluid_ms = 0.;
        for tick in 0..480 {
            s.step(Controls::default());
            assert!(!s.paused, "tick {tick}: {}", s.message);
            physics
                .step(&mut s)
                .unwrap_or_else(|e| panic!("sampling {sampling:?} tick {tick}: {e}"));
            fluid_ms += physics.timings.fluid_ms;
        }
        let wall = start.elapsed().as_secs_f64();
        assert!((s.accounted_dry_mass_kg() - initial_mass).abs() < 0.0001);
        assert!((s.accounted_gold() - initial_gold).abs() < 1e-7);
        assert!(s.water_balance_error_m3().abs() < 2e-6);
        assert!((s.sluice.water_used_m3 - s.sluice.accounted_water_m3()).abs() < 1e-10);
        assert!(s.sluice.water_returned_m3 > 0.0001);
        assert!(!s.sluice.grains.is_empty());
        assert!(s.sluice.water_speed() > 0.05 && s.sluice.water_speed() < 3.);
        let text = s.save_json().unwrap();
        let mut restored = Session::load_json(&text).unwrap();
        assert_eq!(canonical(&s), canonical(&restored));
        let mut continued = Hydraulics::new().unwrap();
        let mut reloaded = Hydraulics::new().unwrap();
        if sampling == WaterSampling::Eight {
            continued = continued.with_conservative_phases();
            reloaded = reloaded.with_conservative_phases();
        }
        for _ in 0..24 {
            s.step(Controls::default());
            restored.step(Controls::default());
            continued.step(&mut s).unwrap();
            reloaded.step(&mut restored).unwrap();
        }
        assert_eq!(canonical(&s), canonical(&restored));
        reports.push(
            serde_json::json!({"markers_per_cell":sampling.markers_per_cell(),
            "reference_grid_m":s.sluice.profile.cell(),"channel_width_m":s.sluice.profile.width(),
            "ticks_measured":480,"physical_seconds":8.,"wall_seconds":wall,
            "physical_seconds_per_wall_second":8./wall,"mean_fluid_ms_per_fixed_tick":fluid_ms/480.,
            "water_markers":s.sluice.water.len(),"water_depth_m":s.sluice.water_depth(),
            "downstream_ms":s.sluice.water_speed(),"returned_m3":s.sluice.water_returned_m3,
            "grains":s.sluice.grains.len(),"water_error_m3":s.water_balance_error_m3(),
            "gold_error_g":s.accounted_gold()-initial_gold,"paired_saved_continuation_ticks":24}),
        );
    }
    eprintln!(
        "WATER_SAMPLING_OPERATION {}",
        serde_json::to_string(&reports).unwrap()
    );
}
