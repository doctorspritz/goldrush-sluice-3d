use glam::{DVec3, Mat3, Quat};
use mining_sim::{
    screen::{Screen, ScreenConfig, MAX_SCREEN_PARTICLES},
    sluice::Grain,
    GrainKind, Material, Vec3,
};

fn grain(p: DVec3, v: DVec3, diameter: f32) -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Sand,
        position: p.as_vec3(),
        velocity: v.as_vec3(),
        material: Material {
            sand_m3: 1e-8,
            pore_water_m3: 1e-9,
            ..Default::default()
        },
        resting: false,
        diameter_m: diameter,
        affine: Mat3::from_cols(
            Vec3::new(1., 2., 3.),
            Vec3::new(4., 5., 6.),
            Vec3::new(7., 8., 9.),
        ),
        rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.6),
        angular_velocity: Vec3::ZERO,
    }
}
fn deck_point(s: &Screen, x: f64, z: f64, height: f64) -> DVec3 {
    s.to_world(DVec3::new(
        x,
        -x * s.config.slope_deg.to_radians().tan() + height,
        z,
    )) + s.offset_at(s.time).0
}
fn hole(s: &Screen) -> (f64, f64) {
    let pitch = s.config.aperture_m + 2. * s.config.wire_radius_m;
    (
        20.5 * pitch * s.config.slope_deg.to_radians().cos(),
        pitch * 0.5,
    )
}
fn state(s: &Screen) -> String {
    serde_json::to_string(s).unwrap()
}
fn run(s: &mut Screen, seconds: f64, dt: f32) {
    for _ in 0..(seconds / dt as f64).round() as usize {
        s.step(dt).unwrap();
        assert!(s.valid());
    }
}
fn energy(s: &Screen) -> f64 {
    s.particles
        .iter()
        .map(|p| {
            let r = p.grain.diameter_m as f64 * 0.5;
            let m = p.grain.material.wet_mass_kg();
            m * (0.5 * p.velocity.length_squared()
                + 9.81 * p.position.y
                + 0.2 * r * r * p.angular_velocity.length_squared())
        })
        .sum()
}

#[test]
fn aperture_and_corner_contacts_do_not_sort_by_material_tag() {
    for kind in [GrainKind::Gold, GrainKind::Sand, GrainKind::Gravel] {
        let mut s = Screen::new(Vec3::ZERO, 0.);
        let (x, z) = hole(&s);
        let mut g = grain(deck_point(&s, x, z, 0.025), DVec3::ZERO, 0.001);
        g.kind = kind;
        s.admit(&[g], 0.).unwrap();
        let mut touched_gutter = false;
        for _ in 0..360 {
            s.step(1. / 120.).unwrap();
            touched_gutter |= s.particles.iter().any(|p| p.in_gutter);
            if !s.undersize.is_empty() {
                break;
            }
        }
        assert!(
            touched_gutter,
            "through-aperture passage must first reach the gutter"
        );
        assert_eq!(s.undersize.len(), 1);
        assert!(s.oversize.is_empty() && s.spills.is_empty());
        assert_eq!(s.undersize[0].kind, kind);
        assert!(s.valid());
    }
    // Directly above a crossing, the same fine particle must meet real wire.
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let pitch = s.config.aperture_m + 2. * s.config.wire_radius_m;
    let x = 20. * pitch * s.config.slope_deg.to_radians().cos();
    let p = deck_point(&s, x, 0., 0.0032);
    s.admit(&[grain(p, DVec3::ZERO, 0.001)], 0.).unwrap();
    run(&mut s, 0.04, 1. / 240.);
    assert!(s.last_step.maximum_repair_m < 1e-7);
    assert!(s.particles.iter().all(|p| !p.in_gutter));
    assert!(s.undersize.is_empty());
}

#[test]
fn continuous_wire_cast_stops_a_fast_crossing_with_clear_endpoints() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let x = 0.1;
    let p = deck_point(&s, x, 0., 0.03);
    s.admit(&[grain(p, -DVec3::Y * 30., 0.002)], 0.).unwrap();
    // An endpoint-only 2 ms update would put this centre 3 cm below the wire.
    s.step(0.002).unwrap();
    assert_eq!(s.particles.len(), 1);
    let local = s.to_local(s.particles[0].position);
    assert!(local.y + x * s.config.slope_deg.to_radians().tan() > 0.002);
    assert!(s.last_step.contacts > 0 && s.last_step.maximum_repair_m < 1e-7);
    assert!(!s.particles[0].in_gutter && s.valid());
}

#[test]
fn frozen_drive_retains_its_shifted_geometry() {
    let mut shifted = Screen::new(Vec3::ZERO, 0.);
    shifted.time = 0.017;
    shifted.elapsed_time_s = 0.017;
    let (offset, velocity) = shifted.offset_at(100.);
    assert_eq!(velocity, DVec3::ZERO);
    assert_eq!(offset, shifted.offset_at(0.).0);
    let mut translated = Screen::new(offset.as_vec3(), 0.);
    let p = deck_point(&shifted, 0.1, 0., 0.0033);
    let g = grain(p, DVec3::ZERO, 0.001);
    shifted.admit(std::slice::from_ref(&g), 0.).unwrap();
    translated.admit(&[g], 0.).unwrap();
    run(&mut shifted, 0.05, 1. / 240.);
    run(&mut translated, 0.05, 1. / 240.);
    assert!((shifted.particles[0].position - translated.particles[0].position).length() < 2e-7);
    assert!((shifted.particles[0].velocity - translated.particles[0].velocity).length() < 2e-5);
    assert_eq!(shifted.time, 0.017);
    let mut moving = Screen::new(Vec3::ZERO, 0.);
    moving.motor_on = true;
    run(&mut moving, 0.05, 1. / 240.);
    let before = moving.wire_segments(moving.time);
    moving.motor_on = false;
    assert_eq!(before, moving.wire_segments(moving.time + 1.));
}

#[test]
fn moving_contact_energy_is_bounded_by_actual_actuator_work() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    s.motor_on = true;
    s.time = 1. / (4. * s.config.frequency_hz);
    s.elapsed_time_s = s.time;
    let pitch = s.config.aperture_m + 2. * s.config.wire_radius_m;
    let x = 20.5 * pitch * s.config.slope_deg.to_radians().cos();
    let g = grain(deck_point(&s, x, 0., 0.0033), -DVec3::Y * 0.05, 0.001);
    s.admit(&[g], 0.).unwrap();
    let before = energy(&s);
    s.step(0.004).unwrap();
    assert_eq!(s.particles.len(), 1);
    assert!(
        s.last_step.motor_work_j > 0.,
        "rising wire must supply observed work"
    );
    let excess = energy(&s) - before - s.last_step.motor_work_j - s.last_step.projection_work_j;
    assert!(excess <= 1e-9, "unexplained energy {excess} J");
    assert!(s.last_step.maximum_repair_m < 2e-5);
}

#[test]
fn moving_contact_refines_with_the_external_step() {
    let mut a = Screen::new(Vec3::ZERO, 0.);
    a.motor_on = true;
    let g = grain(deck_point(&a, 0.1, 0., 0.02), DVec3::ZERO, 0.001);
    a.admit(&[g], 0.).unwrap();
    let mut b = a.clone();
    run(&mut a, 0.1, 1. / 30.);
    run(&mut b, 0.1, 1. / 240.);
    assert_eq!(a.particles.len(), 1);
    assert_eq!(b.particles.len(), 1);
    assert!((a.particles[0].position - b.particles[0].position).length() < 2e-4);
    assert!((a.particles[0].velocity - b.particles[0].velocity).length() < 0.01);
}

#[test]
fn save_continuation_preserves_full_grain_custody_and_drive_phase() {
    let mut s = Screen::new(Vec3::new(2., 3., 4.), 0.7);
    s.motor_on = true;
    let (x, z) = hole(&s);
    let mut g = grain(
        deck_point(&s, x, z, 0.03),
        DVec3::new(0.03, 0., 0.04),
        0.001,
    );
    g.volume_resolved = false;
    g.angular_velocity = Vec3::new(0.2, -0.3, 0.4);
    let material = g.material;
    let affine = g.affine;
    s.admit(&[g], 0.002).unwrap();
    run(&mut s, 0.025, 1. / 240.);
    s.particles[0].angular_velocity += DVec3::new(1e-10, -2e-10, 3e-10);
    s.particles[0].grain.angular_velocity = s.particles[0].angular_velocity.as_vec3();
    let mut restored: Screen = serde_json::from_str(&state(&s)).unwrap();
    assert!(restored.valid());
    assert_eq!(
        restored.config.simultaneous_impact,
        mining_sim::screen::SimultaneousImpactPolicy::Plastic
    );
    assert_eq!(
        restored.particles[0].angular_velocity,
        s.particles[0].angular_velocity
    );
    assert_eq!(state(&s), state(&restored));
    let encoded: serde_json::Value = serde_json::from_str(&state(&s)).unwrap();
    assert_eq!(encoded["config"]["simultaneous_impact"], "plastic");
    let mut invalid_spin = s.clone();
    invalid_spin.particles[0].grain.angular_velocity.x += 1.;
    assert!(!invalid_spin.valid());
    let before = state(&invalid_spin);
    assert!(invalid_spin.step(1. / 240.).is_err());
    assert_eq!(state(&invalid_spin), before);
    for _ in 0..12 {
        s.step(1. / 240.).unwrap();
        restored.step(1. / 240.).unwrap();
    }
    assert_eq!(state(&s), state(&restored));
    let g = &s.particles[0].grain;
    assert!(!g.volume_resolved);
    assert_eq!(g.material, material);
    assert_eq!(g.affine, affine);
    assert!((g.rotation.length_squared() - 1.).abs() < 1e-6);
    assert_eq!(s.material(), material);
    assert_eq!(s.take_free_water(), 0.002);
    assert!(s.valid());
    let before = state(&s);
    assert!(s.set_config(ScreenConfig::default()).is_err());
    assert_eq!(before, state(&s));
    let mut invalid = s.clone();
    invalid.config.aperture_m *= 2.;
    let before = state(&invalid);
    assert!(invalid.step(1. / 30.).is_err());
    assert_eq!(before, state(&invalid));
}

#[test]
fn full_output_stock_refuses_feed_without_losing_or_duplicating_material() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let (x, z) = hole(&s);
    let g = grain(deck_point(&s, x, z, 0.03), DVec3::ZERO, 0.001);
    s.admit(&vec![g.clone(); MAX_SCREEN_PARTICLES], 0.).unwrap();
    // A saved, owned product stock occupies the same bounded storage budget.
    s.undersize = s.particles.drain(..).map(|p| p.grain).collect();
    assert!(s.valid());
    let before = state(&s);
    assert!(s.admit(std::slice::from_ref(&g), 0.).is_err());
    assert_eq!(before, state(&s));
    assert!(s.admit(&[g], f64::NAN).is_err());
    assert_eq!(before, state(&s));
    assert_eq!(s.take_undersize().len(), MAX_SCREEN_PARTICLES);
    assert_eq!(s.material(), Material::default());
    assert!(s.valid());
}

#[test]
fn coarse_reject_reaches_the_real_bin_without_entering_the_gutter() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let g = grain(
        deck_point(&s, s.config.length_m - 0.008, 0., 0.014),
        DVec3::Z * 0.3,
        0.006,
    );
    let inventory = g.material;
    s.admit(&[g], 0.).unwrap();
    for _ in 0..360 {
        s.step(1. / 240.).unwrap();
        if !s.oversize.is_empty() {
            break;
        }
    }
    assert_eq!(
        s.oversize.len(),
        1,
        "rejected grain must land in the actual bin"
    );
    assert!(s.undersize.is_empty() && s.spills.is_empty());
    let local = s.to_local(s.oversize[0].position.as_dvec3());
    let floor = -s.config.length_m * s.config.slope_deg.to_radians().tan() - 0.1;
    assert!((local.y - floor - 0.003).abs() < 2e-5);
    assert!(local.x >= s.config.length_m && local.x <= s.config.length_m + 0.18);
    assert_eq!(s.take_oversize()[0].material, inventory);
    assert!(s.valid());
}

#[test]
fn tilted_lower_cross_wire_keeps_its_leading_grazing_contact() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let config = ScreenConfig {
        slope_deg: 25.,
        ..Default::default()
    };
    s.set_config(config).unwrap();
    let (a, b, wire_radius) = s
        .wire_segments(0.)
        .into_iter()
        .find(|(a, b, _)| {
            let a = s.to_local(*a);
            let b = s.to_local(*b);
            (a.x - b.x).abs() < 1e-12 && a.x > 0.05 && a.x < 0.15
        })
        .unwrap();
    let centre = s.to_local((a + b) * 0.5);
    let r = 0.0005;
    let normal_shift = wire_radius * s.config.slope_deg.to_radians().sin();
    let x = centre.x - (wire_radius + r) + normal_shift * 0.4;
    let z = (s.config.aperture_m + 2. * wire_radius) * 0.5;
    let p = deck_point(&s, x, z, 0.006);
    s.admit(&[grain(p, -DVec3::Y * 0.2, (2. * r) as f32)], 0.)
        .unwrap();
    let mut contacts = 0;
    for _ in 0..18 {
        s.step(1. / 240.).unwrap();
        contacts += s.last_step.contacts;
        assert!(s.last_step.maximum_repair_m < 1e-7);
    }
    assert!(
        contacts > 0,
        "actual capsule was omitted by its unshifted lattice index"
    );
    assert!(s.valid());
}

#[test]
fn a_bin_wall_top_is_not_a_supported_bin_floor_product() {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    let g = grain(deck_point(&s, 0.1, 0., 0.02), -DVec3::Y * 0.1, 0.002);
    s.admit(&[g], 0.).unwrap();
    // A valid airborne state arriving over the downstream wall's top edge.
    let floor = -s.config.length_m * s.config.slope_deg.to_radians().tan() - 0.1;
    let p = s.to_world(DVec3::new(
        s.config.length_m + 0.18,
        floor + 0.12 + 0.001 + 0.00001,
        0.,
    ));
    s.particles[0].position = p;
    s.particles[0].grain.position = p.as_vec3();
    s.step(0.001).unwrap();
    assert!(s.last_step.contacts > 0);
    assert!(
        s.oversize.is_empty(),
        "wall-lip contact cannot admit floor product stock"
    );
    assert!(s.undersize.is_empty() && s.valid());
}

#[test]
fn actual_pay_feeder_gravel_keeps_simultaneous_wire_support() {
    use mining_sim::{Placement, Session, FIXED_DT};
    let mut operation = Session::new_operation();
    let at = Vec3::new(19., 0., 20.);
    let spoil = operation.site.dig(at, 0.4, 0.10, 0.20);
    operation
        .site
        .try_deposit(at + Vec3::X * 2., spoil)
        .unwrap();
    let pay = operation.site.dig(at, 0.4, 0.15, 0.04);
    assert!(pay.gravel_m3 > 0. && pay.gold_g > 0.);
    let placement = Placement::suggested(&operation.site);
    operation.sluice.placement = Some(placement);
    operation.sluice.add_to_hopper(pay);
    operation.sluice.feed_on = true;
    operation.sluice.pump_on = true;
    let mut screen = Screen::new(
        placement.inlet - placement.forward() * 0.70 + Vec3::Y * 0.60,
        placement.yaw,
    );
    screen.motor_on = true;
    let mut selected = 0;
    // Other phases remain held by this fixture; parcels do not interact in the
    // explicitly dilute screen, so selecting gravel preserves its exact input.
    let mut held_other_phases = Vec::new();
    for tick in 0..64 {
        screen
            .step(FIXED_DT)
            .unwrap_or_else(|e| panic!("tick={tick}: {e}"));
        if tick < 8 {
            let batch = operation.sluice.take_screen_feed(FIXED_DT, true).unwrap();
            let mut incoming = Vec::new();
            for g in batch.grains {
                if g.kind != GrainKind::Gravel {
                    held_other_phases.push(g);
                    continue;
                }
                let across = g.position.z
                    - operation.sluice.profile.width() * 0.5
                    - operation.sluice.profile.cell();
                let mut world = mining_sim::tailings::world_grain(&operation.sluice, &g).unwrap();
                world.position = screen
                    .to_world(DVec3::new(
                        0.035 + (g.position.x as f64 - 0.10),
                        0.055,
                        across as f64,
                    ))
                    .as_vec3();
                world.position += screen.offset_at(screen.time).0.as_vec3();
                incoming.push(world);
            }
            selected += incoming.len();
            screen.admit(&incoming, batch.moisture_m3).unwrap();
        }
        assert!(screen.valid());
    }
    assert!(selected > 0 && !held_other_phases.is_empty());
    assert_eq!(
        screen.particles.len()
            + screen.undersize.len()
            + screen.oversize.len()
            + screen.spills.len(),
        selected
    );
    assert!(
        screen.undersize.iter().all(|g| g.diameter_m <= 0.0030001),
        "actual oversize gravel cannot leak through the 3 mm aperture"
    );
}
