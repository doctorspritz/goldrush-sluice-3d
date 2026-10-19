use mining_sim::{
    session::Controls, Action, ControlMode, MachineInput, Material, Placement, Session, Site,
    Sluice, Vec3, FIXED_DT,
};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() < tolerance, "{a} != {b}");
}
#[test]
fn dry_or_missing_water_cannot_process_free_material() {
    let mut s = Sluice::default();
    s.placement = Some(Placement::suggested(&Site::new()));
    s.pump_on = true;
    s.feed_on = true;
    s.hopper = Material {
        gravel_m3: 0.2,
        gold_g: 2.0,
        ..Material::default()
    };
    let before = s.hopper;
    for _ in 0..300 {
        s.feed(FIXED_DT, false);
    }
    assert_eq!(s.hopper, before);
    assert!(s.grains.is_empty());
}

#[test]
fn placement_rejects_bad_support_overlap_and_disconnected_inlet() {
    let site = Site::new();
    let p = Placement::suggested(&site);
    assert!(p.validate(&site, Vec3::new(19.0, 0.35, 17.0)).is_ok());
    assert!(Placement {
        inlet: p.inlet + Vec3::X * 5.0,
        ..p
    }
    .validate(&site, Vec3::ZERO)
    .is_err());
    assert!(Placement {
        slope_deg: 30.0,
        ..p
    }
    .validate(&site, Vec3::ZERO)
    .is_err());
    assert!(Placement {
        inlet: p.inlet + Vec3::Y * 3.0,
        ..p
    }
    .validate(&site, Vec3::ZERO)
    .is_err());
    assert!(p
        .validate(&site, p.to_world(Vec3::new(1.0, 0.0, 0.0)))
        .is_err());
}

#[test]
fn bucket_cut_is_finite_capacity_limited_and_conserves_gold() {
    let mut s = Session::new();
    s.on_action(Action::EnterExit).unwrap();
    for _ in 0..100 {
        s.step(Controls {
            machine: MachineInput {
                cut: true,
                boom: -0.22,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    println!(
        "bucket {:?}, lip {:?}",
        s.machine.bucket,
        s.machine.arm().lip
    );
    assert!(
        s.machine.bucket.volume() > 0.02,
        "actual bucket trajectory must intersect the paydirt"
    );
    assert!(s.machine.bucket.volume() <= 0.28001);
    assert!(s.machine.bucket.gold_g > 0.0);
    close(s.accounted_gold(), s.initial_gold_g, 1e-5);
    let before = s.machine.bucket;
    for _ in 0..500 {
        s.step(Controls {
            machine: MachineInput {
                cut: true,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    assert!(s.machine.bucket.volume() <= 0.28001);
    assert!(s.machine.bucket.volume() >= before.volume());
    close(s.accounted_gold(), s.initial_gold_g, 1e-5);
}

#[test]
fn distant_tools_and_camera_changes_do_not_edit_or_reset_operation() {
    let mut s = Session::new();
    let before = s.site.remaining_gold();
    assert!(s.on_action(Action::Dig(Vec3::new(1.0, 0.1, 1.0))).is_err());
    close(s.site.remaining_gold(), before, 1e-10);
    s.on_action(Action::EnterExit).unwrap();
    for _ in 0..90 {
        s.step(Controls {
            machine: MachineInput {
                cut: true,
                boom: -0.22,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    let bucket = s.machine.bucket;
    s.on_action(Action::Camera).unwrap();
    s.on_action(Action::Camera).unwrap();
    assert_eq!(s.machine.bucket, bucket);
    assert_eq!(s.player.mode, ControlMode::Excavator);
}

#[test]
fn terrain_cannot_be_mined_below_bedrock_or_duplicate_trace_gold() {
    let mut site = Site::new();
    let initial = site.remaining_gold();
    let p = Vec3::new(19.0, 0.0, 22.0);
    let mut removed = Material::default();
    for _ in 0..20 {
        removed.add(site.dig(p, 0.44, 1.0, 100.0));
    }
    assert!(site.dig(p, 0.44, 1.0, 100.0).is_empty());
    close(initial, site.remaining_gold() + removed.gold_g, 1e-6);
    site.deposit(p, removed);
    close(initial, site.remaining_gold(), 1e-6);
}

#[test]
fn save_restore_continues_in_flight_loads_and_grain_trajectories() {
    let mut s = Session::new();
    s.on_action(Action::EnterExit).unwrap();
    for _ in 0..90 {
        s.step(Controls {
            machine: MachineInput {
                cut: true,
                boom: -0.22,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    let json = s.save_json().unwrap();
    let mut restored = Session::load_json(&json).unwrap();
    assert_eq!(s.machine.bucket, restored.machine.bucket);
    assert_eq!(s.player.mode, restored.player.mode);
    for _ in 0..60 {
        let input = Controls {
            machine: MachineInput {
                boom: 1.0,
                stick: -0.3,
                ..Default::default()
            },
            ..Default::default()
        };
        s.step(input);
        restored.step(input);
    }
    assert_eq!(s.machine.arm().lip, restored.machine.arm().lip);
    close(s.accounted_gold(), restored.accounted_gold(), 1e-6);
    close(s.sim_time, restored.sim_time, 1e-9);
}

#[test]
fn invalid_saves_are_rejected_without_mutating_live_state() {
    let s = Session::new();
    let before = s.initial_gold_g;
    let mut json: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    json["machine"]["boom"] = serde_json::json!(1000.0);
    assert!(Session::load_json(&json.to_string()).is_err());
    json["machine"]["boom"] = serde_json::json!(0.48);
    json["site"]["paydirt"] = serde_json::json!([0.0]);
    assert!(Session::load_json(&json.to_string()).is_err());
    close(s.initial_gold_g, before, 1e-10);
}

#[test]
fn material_splitting_and_hopper_overflow_preserve_all_components() {
    let mut hopper = Sluice::default();
    let input = Material {
        soil_m3: 0.2,
        sand_m3: 0.3,
        gravel_m3: 0.5,
        gold_g: 17.0,
    };
    let overflow = hopper.add_to_hopper(input);
    close(
        hopper.hopper.volume() + overflow.volume(),
        input.volume(),
        1e-10,
    );
    close(hopper.hopper.gold_g + overflow.gold_g, input.gold_g, 1e-10);
    let mut collected = Material::default();
    for _ in 0..1000 {
        collected.add(hopper.hopper.take_volume(0.003));
    }
    collected.add(hopper.hopper);
    collected.add(overflow);
    close(collected.mass_kg(), input.mass_kg(), 1e-7);
}
