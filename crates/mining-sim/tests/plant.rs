use glam::{Mat3, Quat};
use mining_sim::{
    session::Controls,
    sluice::{Grain, HOPPER_X, HOPPER_Z},
    Action, GrainKind, Material, Placement, Session, Vec3,
};
fn fixture() -> Session {
    let mut s = Session::new_operation();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.enable_screen().unwrap();
    let at = Vec3::new(19., 0., 20.);
    let spoil = s.site.dig(at, 0.4, 0.10, 0.20);
    assert!(
        spoil.gravel_m3 < 1e-12 && spoil.gold_g < 1e-10,
        "strip the real overburden first"
    );
    s.site.try_deposit(at + Vec3::X * 2., spoil).unwrap();
    let m = s.site.dig(at, 0.4, 0.15, 0.04);
    assert!(
        m.gravel_m3 > 0. && m.gold_g > 0.,
        "actual finite pay/gravel cohort required"
    );
    assert!(m.volume() > 0.001);
    assert!(s.sluice.add_to_hopper(m).is_empty());
    s.sluice.feed_on = true;
    s.sluice.pump_on = true;
    s.plant.as_mut().unwrap().screen.motor_on = true;
    s
}
fn close(a: Material, b: Material) {
    for (x, y) in [
        (a.soil_m3, b.soil_m3),
        (a.sand_m3, b.sand_m3),
        (a.gravel_m3, b.gravel_m3),
        (a.gold_g, b.gold_g),
        (a.pore_water_m3, b.pore_water_m3),
    ] {
        assert!((x - y).abs() < 1e-7, "material {x} != {y}");
    }
}
#[test]
fn reflected_full_state_handoff_round_trips_without_added_impulse() {
    let mut s = Session::new_operation();
    let mut p = Placement::suggested(&s.site);
    p.yaw = 0.73;
    s.sluice.placement = Some(p);
    let g = Grain {
        kind: GrainKind::Gold,
        position: Vec3::new(0.17, 0.4, 0.21),
        velocity: Vec3::new(0.3, -0.7, 0.1),
        material: Material {
            gold_g: 0.001,
            ..Default::default()
        },
        resting: false,
        diameter_m: 0.0002,
        affine: Mat3::from_cols(
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.4, 0.5, 0.6),
            Vec3::new(0.7, 0.8, 0.9),
        ),
        rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.2, 0.5, -0.3),
        angular_velocity: Vec3::new(0.4, 0.5, 0.6),
        volume_resolved: true,
    };
    let world = mining_sim::tailings::world_grain(&s.sluice, &g).unwrap();
    let back = mining_sim::tailings::grid_grain(&s.sluice, &world).unwrap();
    assert!((back.position - g.position).length() < 2e-6);
    assert!((back.velocity - g.velocity).length() < 1e-6);
    assert!((back.affine - g.affine)
        .to_cols_array()
        .iter()
        .all(|v| v.abs() < 1e-6));
    assert!(back.rotation.dot(g.rotation).abs() > 1. - 1e-6);
    assert!((back.angular_velocity - g.angular_velocity).length() < 1e-6);
    assert_eq!(back.material, g.material);
    assert_eq!(back.diameter_m, g.diameter_m);
}
#[test]
fn finite_dug_hopper_runs_through_screen_gutter_and_into_existing_sluice() {
    let mut s = fixture();
    let before = s.accounted_material();
    let water = s.stored_water_m3();
    for _ in 0..240 {
        s.step(Controls::default());
        assert!(!s.paused, "{}", s.message);
    }
    let p = s.plant.as_ref().unwrap();
    assert!(p.screen.admitted.volume() > 0.0001);
    assert!(p.screen.delivered.volume() > 0.);
    assert!(s.sluice.processed_m3 > 0. && !s.sluice.grains.is_empty());
    assert!(p.valid());
    close(s.accounted_material(), before);
    assert!((s.stored_water_m3() - water).abs() < 2e-7);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
}
#[test]
fn saved_circuit_resumes_exactly_and_older_envelopes_cannot_hide_plant_stocks() {
    let mut s = fixture();
    for _ in 0..45 {
        s.step(Default::default());
        assert!(!s.paused, "{}", s.message);
    }
    let json = s.save_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["version"], 14);
    let mut restored = Session::load_json(&json).unwrap();
    for _ in 0..30 {
        s.step(Default::default());
        restored.step(Default::default());
    }
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
    for version in [6, 12, 13] {
        let mut bad = value.clone();
        bad["version"] = version.into();
        assert!(Session::load_json(&bad.to_string()).is_err());
    }
    let mut bad = value.clone();
    bad.as_object_mut().unwrap().remove("plant");
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut bad = value.clone();
    bad["plant"]["connections"][0]["destination_module"] = 1.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut bad = value.clone();
    bad["plant"]["screen"]["origin"][0] = 19.0.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let mut bad = value;
    bad["plant"]["screen"]["config"]["aperture_m"] = 0.03.into();
    assert!(Session::load_json(&bad.to_string()).is_err());
    let legacy = Session::new_operation();
    let json = legacy.save_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap()["version"],
        13
    );
    assert!(Session::load_json(&json).unwrap().plant.is_none());
}
#[test]
fn real_hand_dump_reaches_the_raised_hopper_and_keeps_finite_stock() {
    let mut s = Session::new_operation();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.enable_screen().unwrap();
    let m = s.site.dig(Vec3::new(19., 0., 20.), 0.3, 0.05, 0.01);
    s.player.hand = m;
    s.player.position = Vec3::new(
        HOPPER_X,
        s.site.height(HOPPER_X, HOPPER_Z - 1.2),
        HOPPER_Z - 1.2,
    );
    s.player.yaw = 0.;
    let before = s.accounted_material();
    s.on_action(Action::DumpHand).unwrap();
    assert!(s.falling[0].position.y > s.hopper_top());
    for _ in 0..90 {
        s.step(Default::default());
    }
    assert!(!s.paused, "{}", s.message);
    assert!(s.falling.is_empty());
    assert!(s.player.hand.is_empty());
    close(s.sluice.hopper, m);
    close(s.accounted_material(), before);
}
#[test]
fn screen_installation_requires_a_real_hopper_connection_and_empty_reconfiguration() {
    let mut s = Session::new_operation();
    assert!(s.enable_screen().is_err());
    assert!(s.plant.is_none());
    let mut p = Placement::suggested(&s.site);
    p.inlet.x += 3.;
    s.sluice.placement = Some(p);
    assert!(s.enable_screen().is_err());
    assert!(s.plant.is_none());
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.enable_screen().unwrap();
    s.player.position = Vec3::new(
        HOPPER_X + 1.5,
        s.site.height(HOPPER_X + 1.5, HOPPER_Z),
        HOPPER_Z,
    );
    s.on_action(Action::ScreenAperture).unwrap();
    assert_eq!(s.plant.as_ref().unwrap().screen.config.aperture_m, 0.006);
    s.on_action(Action::ToggleScreen).unwrap();
    let before = s.save_json().unwrap();
    assert!(s.on_action(Action::ScreenAperture).is_err());
    assert_eq!(s.save_json().unwrap(), before);
    assert!(s.validate_placement(Placement::suggested(&s.site)).is_err());
}

#[test]
fn stopped_cleanout_preserves_pay_and_permits_a_physical_aperture_change() {
    let mut s = fixture();
    for _ in 0..24 {
        s.step(Default::default());
        assert!(!s.paused, "{}", s.message);
    }
    let hopper = mining_sim::Sluice::hopper_position(&s.site);
    s.player.position = hopper + Vec3::X * 1.5;
    let held = s.plant.as_ref().unwrap().material();
    assert!(held.volume() > 0.);
    let before = s.accounted_material();
    s.on_action(Action::ToggleScreen).unwrap();
    s.on_action(Action::ClearScreen).unwrap();
    close(s.player.hand, held);
    assert_eq!(s.plant.as_ref().unwrap().material(), Material::default());
    s.on_action(Action::ScreenAperture).unwrap();
    assert_eq!(s.plant.as_ref().unwrap().screen.config.aperture_m, 0.006);
    close(s.accounted_material(), before);
    assert!(s.plant.as_ref().unwrap().valid());
}

#[test]
fn underflow_waits_for_the_start_of_the_unchanged_two_tick_sluice_interval() {
    let mut s = fixture();
    for _ in 0..240 {
        s.step(Default::default());
        assert!(!s.paused, "{}", s.message);
        if !s.plant.as_ref().unwrap().screen.undersize.is_empty() {
            break;
        }
    }
    let waiting = s.plant.as_ref().unwrap().screen.undersize.clone();
    assert!(!waiting.is_empty());
    s.sluice.feed_on = false;
    s.sluice.fluid_clock = mining_sim::FIXED_DT;
    let before = serde_json::to_value(&s.sluice.grains).unwrap();
    s.step(Default::default());
    assert_eq!(serde_json::to_value(&s.sluice.grains).unwrap(), before);
    assert_eq!(
        serde_json::to_value(&s.plant.as_ref().unwrap().screen.undersize[..waiting.len()]).unwrap(),
        serde_json::to_value(&waiting).unwrap()
    );
    s.sluice.fluid_clock = 0.;
    let previous_count = s.sluice.grains.len();
    s.step(Default::default());
    let delivered = s.plant.as_ref().unwrap().screen.delivered;
    // Gold-only first arrivals have zero bulk volume but positive mass.
    assert!(delivered.mass_kg() > 0., "delivered={delivered:?}");
    assert!(s.sluice.grains.len() >= previous_count + waiting.len());
}

#[test]
fn pending_water_is_received_before_any_new_batch_can_replace_it() {
    let mut s = fixture();
    let p = s.plant.as_mut().unwrap();
    p.pending_moisture_m3 = 0.0001;
    let hopper = s.sluice.hopper;
    let water = p.water_m3();
    p.step(&mut s.sluice, mining_sim::FIXED_DT, true).unwrap();
    assert_eq!(s.sluice.hopper, hopper);
    assert_eq!(p.pending_moisture_m3, 0.);
    assert_eq!(p.screen.free_water_m3, 0.0001);
    assert_eq!(p.water_m3(), water);
    assert!(p.valid());
}

#[test]
fn moving_an_empty_plant_keeps_its_upgrade_and_refuses_disconnection_atomically() {
    let mut s = Session::new_operation();
    let mut at = Placement::suggested(&s.site);
    s.sluice.placement = Some(at);
    s.enable_screen().unwrap();
    let hopper = mining_sim::Sluice::hopper_position(&s.site);
    s.player.position = hopper + Vec3::X * 1.5;
    s.on_action(Action::ScreenAperture).unwrap();
    at.inlet.y += 0.02;
    s.on_action(Action::Place(at)).unwrap();
    assert_eq!(s.plant.as_ref().unwrap().screen.config.aperture_m, 0.006);
    assert!(s.plant.as_ref().unwrap().connected_to(&s.sluice));
    let before = s.save_json().unwrap();
    at.inlet.x += 1.8;
    assert!(s.on_action(Action::Place(at)).is_err());
    assert_eq!(s.save_json().unwrap(), before);
}

#[test]
fn out_of_domain_underflow_retains_the_entire_donor_batch() {
    let mut s = fixture();
    for _ in 0..240 {
        s.step(Default::default());
        assert!(!s.paused, "{}", s.message);
        if !s.plant.as_ref().unwrap().screen.undersize.is_empty() {
            break;
        }
    }
    let p = s.plant.as_mut().unwrap();
    assert!(!p.screen.undersize.is_empty());
    p.screen.undersize[0].position.y += 2.;
    let donor = serde_json::to_value(&p.screen.undersize).unwrap();
    let delivered = p.screen.delivered;
    s.sluice.fluid_clock = 0.;
    s.sluice.feed_on = false;
    let received = serde_json::to_value(&s.sluice.grains).unwrap();
    p.step(&mut s.sluice, 0., true).unwrap();
    assert_eq!(serde_json::to_value(&p.screen.undersize).unwrap(), donor);
    assert_eq!(p.screen.delivered, delivered);
    assert_eq!(serde_json::to_value(&s.sluice.grains).unwrap(), received);
}

#[test]
fn paused_maintenance_collects_all_held_water_and_material_once() {
    let mut s = fixture();
    let p = s.plant.as_mut().unwrap();
    p.screen.admit(&[], 0.0002).unwrap();
    p.pending_moisture_m3 = 0.0001;
    let before = s.stored_water_m3();
    let material = s.accounted_material();
    s.player.position = mining_sim::Sluice::hopper_position(&s.site) + Vec3::X * 1.5;
    s.paused = true;
    s.on_action(Action::ToggleScreen).unwrap();
    s.on_action(Action::ClearScreen).unwrap();
    assert!(s.paused);
    assert_eq!(s.player.hand.pore_water_m3, 0.0002 + 0.0001);
    assert_eq!(s.plant.as_ref().unwrap().water_m3(), 0.);
    assert!((s.stored_water_m3() - before).abs() < 1e-12);
    // Free-water pools join carried water; dry mineral/gold custody is unchanged.
    assert_eq!(s.accounted_material().mass_kg(), material.mass_kg());
    assert_eq!(s.accounted_material().volume(), material.volume());
    assert_eq!(s.accounted_material().gold_g, material.gold_g);
    s.on_action(Action::ClearScreen).unwrap();
    assert_eq!(s.player.hand.pore_water_m3, 0.0002 + 0.0001);
    s.on_action(Action::ScreenAperture).unwrap();
}
