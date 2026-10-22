use glam::{Mat3, Quat};
use mining_sim::{sluice::Grain, GrainKind, Material, Placement, Site, Sluice, Vec3, FIXED_DT};

fn grain(material: Material, diameter_m: f32) -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Sand,
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        material,
        resting: false,
        diameter_m,
        affine: Mat3::ZERO,
        rotation: Quat::IDENTITY,
        angular_velocity: Vec3::ZERO,
    }
}

#[test]
fn bulk_inventory_and_physical_grains_have_the_same_dry_mass() {
    for material in [
        Material {
            soil_m3: 0.02,
            ..Default::default()
        },
        Material {
            sand_m3: 0.03,
            ..Default::default()
        },
        Material {
            gravel_m3: 0.04,
            ..Default::default()
        },
        Material {
            soil_m3: 0.02,
            sand_m3: 0.03,
            gravel_m3: 0.04,
            gold_g: 7.,
            ..Default::default()
        },
    ] {
        let expected = material.soil_m3 * 1800.
            + material.sand_m3 * 1600.
            + material.gravel_m3 * 2000.
            + material.gold_g * 0.001;
        let g = grain(material, 0.0003);
        assert!(
            (g.mass_kg() as f64 - expected).abs() < expected * 1e-6,
            "handoff creates mass: {} vs {expected}",
            g.mass_kg()
        );
        let solid = (expected - material.gold_g * 0.001) / 2650. + material.gold_g * 0.001 / 19300.;
        assert!(
            (g.solid_volume() - solid).abs() < 1e-12,
            "bulk pores became solid grains"
        );
        assert!((g.density_kg_m3() as f64 * g.solid_volume() - expected).abs() < expected * 1e-6);
    }
}

#[test]
#[allow(
    clippy::field_reassign_with_default,
    reason = "Sluice persistence fields are private"
)]
fn feeding_cannot_increase_the_mass_held_by_the_operation() {
    let mut s = Sluice::default();
    s.placement = Some(Placement::suggested(&Site::new()));
    s.pump_on = true;
    s.feed_on = true;
    s.hopper = Material {
        soil_m3: 0.003,
        sand_m3: 0.002,
        gravel_m3: 0.001,
        gold_g: 0.0002,
        ..Default::default()
    };
    let initial = s.hopper.mass_kg();
    for _ in 0..120 {
        s.feed(FIXED_DT, true);
    }
    let physical = s.active_material().mass_kg()
        - s.grains.iter().map(|g| g.material.mass_kg()).sum::<f64>()
        + s.grains.iter().map(|g| g.mass_kg() as f64).sum::<f64>();
    assert!(
        (physical - initial).abs() < initial * 1e-6,
        "feed changed dry mass: {initial} -> {physical}"
    );
}

#[test]
fn excavation_custody_deposition_and_save_preserve_component_dry_mass() {
    let mut s = mining_sim::Session::new();
    let initial = s.accounted_material();
    let sample = s.site.dig(Vec3::new(19.8, 0., 22.), 0.55, 0.25, 0.08);
    assert!(sample.volume() > 0.05);
    s.player.hand.add(sample);
    let check = |s: &mining_sim::Session| {
        let actual = s.accounted_material();
        for (a, b) in [
            (initial.soil_m3, actual.soil_m3),
            (initial.sand_m3, actual.sand_m3),
            (initial.gravel_m3, actual.gravel_m3),
        ] {
            assert!((a - b).abs() < sample.volume() * 1e-6);
        }
        assert!((s.accounted_dry_mass_kg() - initial.mass_kg()).abs() < sample.mass_kg() * 1e-6);
        assert!((actual.gold_g - s.accounted_gold()).abs() < 1e-8);
    };
    check(&s);
    s.machine.bucket.add(s.player.hand.take_volume(0.02));
    s.falling.push(mining_sim::session::FallingLoad {
        position: Vec3::new(20., 2., 20.),
        velocity: Vec3::ZERO,
        material: s.player.hand.take_fraction(0.3),
    });
    assert!(s
        .sluice
        .add_to_hopper(s.player.hand.take_fraction(0.5))
        .is_empty());
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    s.sluice.feed_on = true;
    for _ in 0..120 {
        s.sluice.feed(FIXED_DT, true);
    }
    check(&s);
    s.sluice.pump_on = false;
    s.sluice.feed_on = false;
    for g in &mut s.sluice.grains {
        g.resting = true;
        g.position.x = 0.5;
    }
    s.sluice.cleanup().unwrap();
    check(&s);
    let dumped = s.machine.bucket.take_fraction(1.);
    s.site.deposit(Vec3::new(20., 0., 20.), dumped);
    for load in std::mem::take(&mut s.falling) {
        s.site.deposit(load.position, load.material);
    }
    check(&s);
    let loaded = mining_sim::Session::load_json(&s.save_json().unwrap()).unwrap();
    check(&loaded);
}

#[test]
fn one_quartz_grain_does_not_expand_into_a_porous_parcel() {
    let d = 0.0003_f32;
    let solid = std::f64::consts::PI / 6. * (d as f64).powi(3);
    let g = grain(
        Material {
            sand_m3: solid * 2650. / 1600.,
            ..Default::default()
        },
        d,
    );
    assert!(
        (g.represented_grains() - 1.).abs() < 1e-12,
        "grain population includes pore space"
    );
    assert_eq!(g.collision_radius(), d * 0.5);
}
