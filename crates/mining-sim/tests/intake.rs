use mining_sim::{
    intake::{IntakePacket, Suspension},
    session::Controls,
    site::{CELL, WIDTH},
    sluice::{GrainKind, MAX_GRAINS},
    Material, Placement, Session, Site, Vec3, FIXED_DT,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a:.16e} != {b:.16e}");
}
fn fixture() -> (Session, Vec3, usize) {
    let mut s = Session::new();
    // This fixture replaces the claim with impermeable bare rock.
    s.site.soil_water = Default::default();
    for layer in [
        &mut s.site.world.bedrock_elevation,
        &mut s.site.world.paydirt_thickness,
        &mut s.site.world.overburden_thickness,
        &mut s.site.world.gravel_thickness,
        &mut s.site.world.terrain_sediment,
        &mut s.site.world.water_surface,
    ] {
        layer.fill(0.);
    }
    s.site.gold_g.fill(0.);
    let p = Vec3::new(Site::river_center(mining_sim::sluice::HOPPER_Z), 0., 17.);
    let i = s
        .site
        .world
        .idx((p.x / CELL) as usize, (p.z / CELL) as usize);
    s.site.world.water_surface[i] = 0.08;
    let n = WIDTH * WIDTH;
    for (field, concentration) in [
        (&mut s.site.hydraulic.suspended_sediment, 0.01),
        (&mut s.site.hydraulic.suspended_overburden, 0.02),
        (&mut s.site.hydraulic.suspended_gravel, 0.03),
        (&mut s.site.hydraulic.suspended_paydirt, 0.04),
    ] {
        *field = vec![0.; n];
        field[i] = concentration;
    }
    s.site.hydraulic.suspended_gold_g = vec![0.; n];
    s.site.hydraulic.suspended_gold_g[i] = 0.2;
    s.initial_gold_g = s.site.remaining_gold();
    s.water_balance.initial_stored_m3 = s.stored_water_m3();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    (s, p, i)
}

fn suspended(site: &Site) -> [f64; 4] {
    [
        &site.hydraulic.suspended_sediment,
        &site.hydraulic.suspended_overburden,
        &site.hydraulic.suspended_gravel,
        &site.hydraulic.suspended_paydirt,
    ]
    .map(|field| {
        field
            .iter()
            .enumerate()
            .map(|(i, c)| {
                *c as f64
                    * site.world.water_depth(i % WIDTH, i / WIDTH) as f64
                    * (CELL as f64).powi(2)
            })
            .sum()
    })
}

#[test]
fn intake_withdraws_the_same_local_water_suspension_and_gold_sample() {
    let (mut s, p, i) = fixture();
    let before = suspended(&s.site);
    let water = s.site.water_volume_m3();
    let gold = s.site.remaining_gold();
    let packet = s.site.withdraw_intake(p, 0.001);
    assert!(packet.water_m3 > 0.000999);
    assert!(packet.water_m3 <= 0.001_f32 as f64);
    close(water - s.site.water_volume_m3(), packet.water_m3);
    for (channel, initial) in before.iter().enumerate() {
        close(
            *initial,
            suspended(&s.site)[channel] + packet.suspension.solid_m3[channel],
        );
        close(
            packet.suspension.solid_m3[channel],
            initial * packet.water_m3 / water,
        );
    }
    close(gold, s.site.remaining_gold() + packet.suspension.gold_g);
    close(packet.suspension.gold_g, gold * packet.water_m3 / water);
    assert_eq!(s.site.gold_g[i], 0.);
    assert!(s.site.hydraulic.impact_dirty);
}

#[test]
fn sub_ulp_requests_dry_or_invalid_intakes_cannot_create_a_pump_credit() {
    let (mut s, p, i) = fixture();
    s.site.world.bedrock_elevation[i] = 1000.;
    s.site.world.water_surface[i] = 1000.1;
    let before = s.site.world.water_surface[i];
    let gold = s.site.remaining_gold();
    assert_eq!(s.site.withdraw_intake(p, 1e-9).water_m3, 0.);
    assert_eq!(s.site.world.water_surface[i], before);
    close(s.site.remaining_gold(), gold);
    for (position, amount) in [
        (Vec3::NAN, 1.),
        (Vec3::splat(-3.), 1.),
        (p, f32::NAN),
        (p, -1.),
    ] {
        assert_eq!(s.site.withdraw_intake(position, amount).water_m3, 0.);
    }
    s.site.world.water_surface[i] = 1000.;
    assert_eq!(s.site.withdraw_intake(p, 1.).water_m3, 0.);
}

#[test]
fn an_empty_intake_stops_without_losing_material_or_processing_the_hopper() {
    let (mut s, _, _) = fixture();
    // This domain-only run retains the entire stream at the inlet. Choose a
    // dilute source that fits its finite material buffer before testing dry-up.
    for field in [
        &mut s.site.hydraulic.suspended_sediment,
        &mut s.site.hydraulic.suspended_overburden,
        &mut s.site.hydraulic.suspended_gravel,
        &mut s.site.hydraulic.suspended_paydirt,
    ] {
        for c in field {
            *c *= 0.001;
        }
    }
    let water = s.stored_water_m3();
    let material = s.accounted_bulk_material_m3();
    let gold = s.accounted_gold();
    for _ in 0..180 {
        s.step(Controls::default());
    }
    assert_eq!(s.sluice.inlet.last_water_m3, 0.);
    assert!(s.site.water_volume_m3() < 1e-9);
    close(s.stored_water_m3(), water);
    close(s.accounted_bulk_material_m3(), material);
    close(s.accounted_gold(), gold);
    assert_eq!(s.sluice.hopper, Material::default());
    assert!(s.sluice.grains.is_empty());
    assert!(s.sluice.inlet.ready.material().volume() > 0.);
}

#[test]
fn intake_samples_the_local_mixture_without_draining_a_distant_corner() {
    let (mut s, p, i) = fixture();
    let neighbour = i + 1;
    s.site.world.water_surface[neighbour] = 0.16;
    s.site.hydraulic.suspended_sediment[neighbour] = 0.03;
    let distant = i - 3 * WIDTH - 3;
    s.site.world.water_surface[distant] = 1.;
    s.site.hydraulic.suspended_sediment[distant] = 10.;
    let packet = s.site.withdraw_intake(p, 0.001);
    let expected_c = (0.01_f32 as f64 + 2. * 0.03_f32 as f64) / 3.;
    close(packet.suspension.solid_m3[0], packet.water_m3 * expected_c);
    assert_eq!(s.site.world.water_surface[distant], 1.);
    assert!(s.site.world.water_depth(i % WIDTH, i / WIDTH) < 0.08);
    assert!(
        s.site
            .world
            .water_depth(neighbour % WIDTH, neighbour / WIDTH)
            < 0.16
    );
}

#[test]
fn matching_carrier_fraction_enters_real_grains_at_the_water_inlet() {
    let (mut s, p, _) = fixture();
    let packet = s.site.withdraw_intake(p, 0.001);
    s.sluice.accept_intake(packet);
    s.sluice.admit_intake_water(packet.water_m3 * 0.5);
    for channel in 0..4 {
        close(
            s.sluice.inlet.ready.solid_m3[channel],
            packet.suspension.solid_m3[channel] * 0.5,
        );
    }
    let before = s.sluice.active_material();
    s.sluice.emit_intake_grains(0.25);
    assert!(s.sluice.grains.len() >= 5);
    assert_eq!(s.sluice.hopper, Material::default());
    let after = s.sluice.active_material();
    close(before.volume(), after.volume());
    close(before.gold_g, after.gold_g);
    for (g, diameter) in s
        .sluice
        .grains
        .iter()
        .zip([0.0001, 0.001, 0.01, 0.002, 0.0001])
    {
        assert_eq!(g.diameter_m, diameter);
        assert!(g.position.x < 0.06);
        assert!(g.position.x >= g.collision_radius() + 2. * s.sluice.profile.cell());
        assert_eq!(g.velocity, s.sluice.inlet_velocity());
        assert!(g.volume_resolved);
        assert!(g.represented_grains() >= 1. - 1e-6);
        assert!(g.solid_volume() <= s.sluice.parcel_volume() * 1.000001);
    }
    assert_eq!(s.sluice.grains[4].kind, GrainKind::Gold);
}

#[test]
fn dense_admission_keeps_up_without_oversized_parcels_or_a_pulsing_pump() {
    for (water_m3s, concentration) in [(0.00375, [0., 0.04, 0., 0.04]), (0.02, [0., 0.05, 0., 0.])]
    {
        let (mut s, _, _) = fixture();
        let packet = IntakePacket {
            water_m3: water_m3s / 30.,
            suspension: Suspension {
                solid_m3: concentration.map(|c| c * water_m3s / 30.),
                gold_g: 0.,
            },
        };
        for _ in 0..60 {
            assert!(
                s.sluice.intake_has_room(),
                "ordinary dense river input must not pulse the pump"
            );
            s.sluice.accept_intake(packet);
            s.sluice.admit_intake_water(packet.water_m3);
            let before = s.sluice.grains.len();
            s.sluice.emit_intake_grains(1. / 30.);
            assert!(s.sluice.grains.len() - before <= 32);
            assert!(s.sluice.inlet.material().volume() < 1e-10);
        }
        for g in &s.sluice.grains {
            assert!(g.material.volume() <= mining_sim::sluice::PARCEL_VOLUME * 1.000001);
        }
        close(
            s.sluice.active_material().volume(),
            packet.suspension.material().volume() * 60.,
        );
    }
}

#[test]
fn high_flow_session_preserves_both_domain_ticks_before_fluid_admission() {
    let (mut s, _, i) = fixture();
    s.site.world.water_surface[i] = 10.;
    for field in [
        &mut s.site.hydraulic.suspended_sediment,
        &mut s.site.hydraulic.suspended_overburden,
        &mut s.site.hydraulic.suspended_gravel,
        &mut s.site.hydraulic.suspended_paydirt,
    ] {
        field.fill(0.);
    }
    s.site.hydraulic.suspended_overburden[i] = 0.05;
    s.sluice.flow_m3s = 0.02;
    s.water_balance.initial_stored_m3 = s.stored_water_m3();
    let water = s.stored_water_m3();
    let material = s.accounted_bulk_material_m3();
    let gold = s.accounted_gold();
    let mut delivered = 0.;
    // A known absorbing water receiver replaces FLIP for this boundary-only
    // test. Admission follows the actual two-domain-ticks/one-fluid-tick order.
    for tick in 0..120 {
        s.step(Controls::default());
        if tick % 2 == 1 {
            let volume = s.sluice.water_credit_m3;
            s.sluice.admit_intake_water(volume);
            s.sluice.emit_intake_grains(FIXED_DT * 2.);
            delivered += volume;
            s.water_balance.boundary_outflow_m3 += volume;
        }
    }
    assert!(
        delivered > 0.03996 && delivered <= 0.040001,
        "20 L/s must not be halved by the numerical buffer: {delivered}"
    );
    close(s.stored_water_m3() + delivered, water);
    close(s.accounted_bulk_material_m3(), material);
    close(s.accounted_gold(), gold);
    close(s.water_balance_error_m3(), 0.);
}

#[test]
fn grain_capacity_applies_backpressure_before_the_river_is_modified() {
    let (mut s, p, _) = fixture();
    let packet = s.site.withdraw_intake(p, 0.001);
    s.sluice.accept_intake(packet);
    s.sluice.admit_intake_water(packet.water_m3);
    s.sluice.emit_intake_grains(0.25);
    s.sluice
        .grains
        .resize(MAX_GRAINS, s.sluice.grains[0].clone());
    let water = s.site.water_volume_m3();
    let solids = suspended(&s.site);
    s.step(Controls::default());
    assert_eq!(s.site.water_volume_m3(), water);
    assert_eq!(suspended(&s.site), solids);
    assert_eq!(s.sluice.inlet.last_water_m3, 0.);
    s.sluice.grains.clear();
    s.sluice.inlet.pending = Suspension::default();
    s.sluice.inlet.ready = Suspension::default();
    s.step(Controls::default());
    assert!(s.sluice.inlet.last_water_m3 > 0.);
}

#[test]
fn save_continuation_preserves_partly_admitted_stream_and_subgrain_gold() {
    let (mut s, _, _) = fixture();
    for _ in 0..3 {
        s.step(Controls::default());
    }
    let n = (s.sluice.water_credit_m3 * 0.5 / s.sluice.profile.particle_m3()).floor() as usize;
    for _ in 0..n {
        let id = s.sluice.allocate_water_id();
        s.sluice.water.push(mining_sim::sluice::WaterParticle {
            id,
            position: Vec3::new(0.02, 0.4, 0.2),
            velocity: s.sluice.inlet_velocity(),
            affine: glam::Mat3::ZERO,
        });
    }
    s.sluice
        .admit_intake_water(n as f64 * s.sluice.profile.particle_m3());
    let saved = s.save_json().unwrap();
    let mut restored = Session::load_json(&saved).unwrap();
    for _ in 0..30 {
        for session in [&mut s, &mut restored] {
            session.step(Controls::default());
            session.sluice.emit_intake_grains(FIXED_DT * 2.);
        }
    }
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
    close(s.accounted_gold(), s.initial_gold_g);
    let mut old = Session::new();
    old.sluice.water_credit_m3 = 0.000123;
    let mut legacy: serde_json::Value = serde_json::from_str(&old.save_json().unwrap()).unwrap();
    legacy["sluice"].as_object_mut().unwrap().remove("inlet");
    let restored = Session::load_json(&legacy.to_string()).unwrap();
    assert_eq!(restored.sluice.water_credit_m3, old.sluice.water_credit_m3);
    assert_eq!(restored.sluice.inlet.material(), Material::default());
    let mut invalid: serde_json::Value = serde_json::from_str(&saved).unwrap();
    invalid["sluice"]["inlet"]["pending"]["gold_g"] = (-1.).into();
    assert!(Session::load_json(&invalid.to_string()).is_err());
    let mut invalid: serde_json::Value = serde_json::from_str(&saved).unwrap();
    invalid["sluice"]["inlet"]["pending"]["solid_m3"][0] = 1.0.into();
    assert!(Session::load_json(&invalid.to_string()).is_err());
    let mut invalid: serde_json::Value = serde_json::from_str(&saved).unwrap();
    invalid["site"]["hydraulic"]["suspended_sediment"][0] = (-0.01).into();
    assert!(Session::load_json(&invalid.to_string()).is_err());
}

#[test]
fn a_fine_gold_remainder_stays_at_the_inlet_when_the_hopper_feed_stops() {
    let (mut s, _, _) = fixture();
    let packet = IntakePacket {
        water_m3: 0.0001,
        suspension: Suspension {
            gold_g: 1e-7,
            ..Default::default()
        },
    };
    s.sluice.accept_intake(packet);
    s.sluice.admit_intake_water(packet.water_m3);
    s.sluice.pump_on = false;
    for _ in 0..60 {
        s.sluice.feed(FIXED_DT, false);
        s.sluice.emit_intake_grains(FIXED_DT);
    }
    assert!(s.sluice.grains.is_empty());
    assert_eq!(s.sluice.hopper, Material::default());
    close(s.sluice.inlet.pending.gold_g, packet.suspension.gold_g);
    close(s.sluice.active_material().gold_g, packet.suspension.gold_g);
}
