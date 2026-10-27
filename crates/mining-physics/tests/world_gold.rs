use glam::{Mat3, Quat};
use mining_physics::{
    grain_exits::{Snapshot, Status},
    Hydraulics,
};
use mining_sim::{session::Controls, sluice::Grain, GrainKind, Material, Placement, Session, Vec3};
#[test]
#[ignore = "requires an actual Metal/Vulkan adapter"]
fn actual_buffer_exits_feed_saved_independent_gold_and_continue_in_the_live_solver() {
    let mut s = Session::new_operation();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    let baseline = s.accounted_material();
    let mut h = Hydraulics::new().unwrap();
    for _ in 0..180 {
        s.step(Controls::default());
        h.step(&mut s).unwrap();
    }
    let mut pulse = Material::default();
    for (index, diameter) in [0.00005, 0.0001, 0.0002, 0.0005].into_iter().enumerate() {
        let mass = std::f64::consts::PI / 6. * (diameter as f64).powi(3) * 19300. * 1000. * 10.;
        let source = s.site.gold_g.iter().position(|g| *g > mass).unwrap();
        s.site.gold_g[source] -= mass;
        let material = Material {
            gold_g: mass,
            ..Default::default()
        };
        pulse.add(material);
        s.sluice.grains.push(Grain {
            kind: GrainKind::Gold,
            // Distinct physical parcels must not start in an artificial DEM overlap.
            position: Vec3::new(1.55, 0.015, 0.19 + index as f32 * 0.007),
            velocity: Vec3::new(0.6, -0.5, 0.),
            material,
            resting: false,
            diameter_m: diameter,
            affine: Mat3::from_diagonal(Vec3::new(0.1, 0.2, 0.3)),
            rotation: Quat::from_rotation_z(0.2),
            angular_velocity: Vec3::new(0.1, 0.2, 0.3),
            volume_resolved: true,
        });
    }
    h.enable_grain_exit_capture(4096).unwrap();
    let mut exits = Snapshot::default();
    let mut gold_records = 0;
    let mut restored_flight = false;
    let mut restored_world = false;
    let mut fixture = None;
    let mut max_world = 0;
    for tick in 0..360 {
        s.step(Controls::default());
        let previous = s.tailings.as_ref().unwrap().released_count;
        h.step(&mut s).unwrap();
        h.take_grain_exit_snapshot(&mut exits).unwrap();
        assert!(
            exits.is_complete() || (exits.status == Status::NoRemovalStage && exits.observed == 0)
        );
        let new: Vec<_> = s
            .tailings
            .as_ref()
            .unwrap()
            .packets
            .iter()
            .filter(|p| p.id > previous)
            .collect();
        assert_eq!(new.len(), exits.exits().len());
        for (p, e) in new.iter().zip(exits.exits()) {
            assert_eq!(
                serde_json::to_value(&p.grain).unwrap(),
                serde_json::to_value(
                    mining_sim::tailings::world_grain(&s.sluice, &e.grain).unwrap()
                )
                .unwrap()
            );
            if e.grain.kind == GrainKind::Gold {
                gold_records += 1;
            }
        }
        if !restored_flight && new.iter().any(|p| p.grain.kind == GrainKind::Gold) {
            s = Session::load_json(&s.save_json().unwrap()).unwrap();
            h.restore();
            restored_flight = true;
        }
        let g = s.site.loose_gold.as_ref().unwrap();
        assert!(g.valid());
        max_world = max_world.max(g.particles.len());
        if !restored_world && g.particles.len() >= 4 {
            let saved = s.save_json().unwrap();
            let loaded = Session::load_json(&saved).unwrap();
            assert_eq!(
                serde_json::to_value(&loaded.site.loose_gold).unwrap(),
                serde_json::to_value(&s.site.loose_gold).unwrap()
            );
            fixture = Some(saved);
            s = loaded;
            h.restore();
            restored_world = true;
        }
        assert!(
            (s.accounted_gold() - s.initial_gold_g).abs() < 1e-7,
            "tick {tick}"
        );
        assert!((s.accounted_material().volume() - baseline.volume()).abs() < 1e-5);
        assert!(s.water_balance_error_m3().abs() < 2e-5);
    }
    let g = s.site.loose_gold.as_ref().unwrap();
    assert!(gold_records >= 4 && restored_flight && restored_world);
    assert!(g.admitted_g >= pulse.gold_g * (1. - 1e-9));
    assert!(s
        .tailings
        .as_ref()
        .unwrap()
        .packets
        .iter()
        .all(|p| !p.contact_pending || p.grain.kind != GrainKind::Gold));
    println!(
        "{}",
        serde_json::json!({"actual_gold_exit_records":gold_records,"pulse_gold_g":pulse.gold_g,"admitted_g":g.admitted_g,"active_g":g.material().gold_g,"exported_g":g.exported_g,"maximum_world_particles":max_world,"restored_airborne":restored_flight,"restored_world":restored_world,"gold_error_g":s.accounted_gold()-s.initial_gold_g,"bulk_error_m3":s.accounted_material().volume()-baseline.volume(),"water_error_m3":s.water_balance_error_m3(),"ticks_after_pulse":360,"limitations":"Buffered-lip handoff fixture, not an inlet-through-riffles recovery assay; dilute one-way point parcels and bulk inlet/excavation reconstitution remain provisional."})
    );
    if let Some(path) = std::env::var_os("SLUICE_WORLD_GOLD_OUTPUT") {
        let dir = std::path::PathBuf::from(path);
        std::fs::create_dir(&dir).unwrap();
        let saved = fixture.unwrap();
        std::fs::write(dir.join("modern.save.json"), &saved).unwrap();
        let mut old = Session::load_json(&saved).unwrap();
        let gold = old.site.loose_gold.take().unwrap();
        old.sluice.tailings.add(gold.material());
        old.sluice.tailings.gold_g += gold.exported_g;
        let legacy = old.save_json().unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&legacy).unwrap()["version"],
            12
        );
        assert!(Session::load_json(&legacy).is_ok());
        std::fs::write(dir.join("legacy-render.save.json"), legacy).unwrap();
    }
}
