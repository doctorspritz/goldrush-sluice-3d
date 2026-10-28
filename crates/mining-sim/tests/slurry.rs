#![allow(
    clippy::field_reassign_with_default,
    reason = "Sluice persistence fields are private"
)]
use glam::{Mat3, Quat};
use mining_sim::{
    sluice::{FeedGrading, Grain, WaterParticle},
    slurry::SlurryField,
    GrainKind, Material, Placement, Session, Sluice, Vec3, FIXED_DT,
};
fn fine(position: Vec3, volume: f64, d: f32) -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Sand,
        position,
        velocity: Vec3::ZERO,
        material: Material {
            soil_m3: volume,
            ..Default::default()
        },
        resting: false,
        diameter_m: d,
        affine: Mat3::ZERO,
        rotation: Quat::IDENTITY,
        angular_velocity: Vec3::ZERO,
    }
}
#[test]
fn new_grading_separates_fines_and_sand_without_changing_any_material() {
    let mut s = Session::new();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    s.sluice.feed_on = true;
    s.sluice.hopper = Material {
        soil_m3: 0.002,
        sand_m3: 0.002,
        gravel_m3: 0.002,
        gold_g: 0.002,
        ..Default::default()
    };
    s.initial_gold_g += s.sluice.hopper.gold_g;
    let before = s.sluice.active_material();
    for _ in 0..120 {
        s.sluice.feed(FIXED_DT, true);
    }
    let after = s.sluice.active_material();
    for (a, b) in [
        (before.soil_m3, after.soil_m3),
        (before.sand_m3, after.sand_m3),
        (before.gravel_m3, after.gravel_m3),
        (before.gold_g, after.gold_g),
    ] {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(s.sluice.grains.iter().any(|g| g.material.soil_m3 > 0.));
    for g in &s.sluice.grains {
        if g.material.soil_m3 > 0. {
            assert!((0.00002..=0.0002).contains(&g.diameter_m));
            assert_eq!(g.material.sand_m3, 0.);
        }
        if g.material.sand_m3 > 0. {
            assert!((0.0002..=0.0012).contains(&g.diameter_m));
            assert_eq!(g.material.soil_m3, 0.);
        }
    }
    assert!(s.sluice.valid());
    let loaded = Session::load_json(&s.save_json().unwrap()).unwrap();
    assert_eq!(loaded.sluice.feed_grading, FeedGrading::Alluvial);
    let mut old: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    old["sluice"]
        .as_object_mut()
        .unwrap()
        .remove("feed_grading");
    let old = Session::load_json(&old.to_string()).unwrap();
    assert_eq!(old.sluice.feed_grading, FeedGrading::Legacy);
}
#[test]
fn one_parcel_represents_many_fines_but_one_gold_grain_stays_one_grain() {
    let p = fine(Vec3::ZERO, 2e-6, 0.0001);
    let expected = 2e-6 * 1800. / 2650. / (std::f64::consts::PI / 6. * (0.0001_f32 as f64).powi(3));
    assert!((p.represented_grains() / expected - 1.).abs() < 1e-12);
    let mut g = p;
    g.kind = GrainKind::Gold;
    g.material = Material {
        gold_g: std::f64::consts::PI / 6. * (g.diameter_m as f64).powi(3) * 19300. * 1000.,
        ..Default::default()
    };
    assert!((g.represented_grains() - 1.).abs() < 1e-12);
}
fn column() -> Sluice {
    let mut s = Sluice::default();
    s.water = (0..300)
        .map(|_| WaterParticle {
            id: 0,
            position: Vec3::new(0.6, 0.28, 0.2),
            velocity: Vec3::ZERO,
            affine: Mat3::ZERO,
        })
        .collect();
    s.grains.push(fine(Vec3::new(0.6, 0.28, 0.2), 2e-6, 0.0001));
    s
}
#[test]
fn projection_is_local_and_conserves_volume_even_at_domain_edges() {
    let mut s = column();
    s.grains.push(fine(Vec3::ZERO, 1e-6, 0.00005));
    let before = serde_json::to_string(&s).unwrap();
    let f = SlurryField::new(&s);
    let (w, v) = f.projected_volumes();
    assert!((w - s.water.len() as f64 * s.profile.particle_m3()).abs() < 1e-15);
    assert!((v - 3e-6 * 1800. / 2650.).abs() < 1e-15);
    assert!(f.sample(Vec3::new(0.6, 0.28, 0.2)).1 > 0.);
    assert_eq!(f.sample(Vec3::new(1.2, 0.28, 0.2)).1, 0.);
    assert_eq!(before, serde_json::to_string(&s).unwrap());
}
#[test]
fn optics_depend_on_represented_volume_and_size_not_parcel_count() {
    let mut s = column();
    let p = s.water[0].position;
    let beta = SlurryField::new(&s).sample(p).1;
    s.grains[0].material.soil_m3 *= 0.5;
    s.grains.push(s.grains[0].clone());
    assert!((SlurryField::new(&s).sample(p).1 - beta).abs() < 1e-4);
    for g in &mut s.grains {
        g.diameter_m *= 0.5;
    }
    assert!((SlurryField::new(&s).sample(p).1 - 2. * beta).abs() < 1e-3);
    for g in &mut s.grains {
        g.resting = true;
    }
    assert_eq!(SlurryField::new(&s).sample(p).1, 0.);
}

#[test]
fn a_uniform_mixture_keeps_extinction_across_optical_cell_phases() {
    let mut s = column();
    let water = s.water.len() as f64 * s.profile.particle_m3();
    let expected = 3. * s.grains[0].solid_volume() / (s.grains[0].diameter_m as f64 * water);
    for i in 0..=24 {
        let p = Vec3::new(0.60 + i as f32 * 0.00125, 0.28, 0.20);
        for w in &mut s.water {
            w.position = p;
        }
        s.grains[0].position = p;
        for channel in [false, true] {
            let mut field = SlurryField::new(&s);
            if channel {
                field.reconstruct_channel();
            }
            for offset in [-0.025, 0., 0.025] {
                let sample = p + Vec3::new(offset, 0., -offset * 0.3);
                let beta = if channel {
                    field.sample_channel(sample)
                } else {
                    field.sample(sample)
                }
                .1 as f64;
                assert!(
                    (beta / expected - 1.).abs() < 2e-6,
                    "channel {channel}, phase {i}, offset {offset}: {beta} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn channel_suspension_mask_preserves_carrier_and_rejects_only_unwetted_sand() {
    let mut s = column();
    let p = s.grains[0].position;
    s.grains.push(fine(p + Vec3::Y * 0.1, 7e-6, 0.00005));
    let mut bed = fine(p, 3e-6, 0.00008);
    bed.resting = true;
    s.grains.push(bed);
    let before = serde_json::to_string(&s).unwrap();
    let legacy = SlurryField::new(&s);
    let legacy_outlet = legacy.outlet_extinction(&s).to_bits();
    let mut channel = SlurryField::new_with_suspension(&s, &[true, false, true]);
    assert!((channel.projected_volumes().0 - legacy.projected_volumes().0).abs() < 1e-15);
    assert!((channel.projected_volumes().1 - s.grains[0].solid_volume()).abs() < 1e-15);
    let mut expected = s.clone();
    expected.grains.remove(1);
    let mut reference = SlurryField::new(&expected);
    channel.reconstruct_channel();
    reference.reconstruct_channel();
    for i in 0..101 {
        let query = p + Vec3::new((i as f32 - 50.) * 0.006, 0.001, 0.003);
        let actual = channel.sample_channel(query);
        let expected = reference.sample_channel(query);
        assert_eq!(
            [actual.0.to_bits(), actual.1.to_bits()],
            [expected.0.to_bits(), expected.1.to_bits()]
        );
    }
    assert_eq!(
        SlurryField::new(&s).outlet_extinction(&s).to_bits(),
        legacy_outlet
    );
    assert_eq!(before, serde_json::to_string(&s).unwrap());
    let dry = SlurryField::new_with_suspension(&s, &[false; 3]);
    assert_eq!(dry.projected_volumes().1, 0.);
    assert_eq!(dry.sample_channel(p).1, 0.);
    assert_eq!(
        dry.projected_volumes().0.to_bits(),
        legacy.projected_volumes().0.to_bits()
    );
}

#[test]
fn wet_channel_projection_is_invariant_to_coincident_parcel_splitting() {
    let mut s = column();
    let p = s.grains[0].position;
    let mut full = SlurryField::new_with_suspension(&s, &[true]);
    full.reconstruct_channel();
    s.grains[0].material.soil_m3 *= 0.5;
    s.grains.push(s.grains[0].clone());
    let mut split = SlurryField::new_with_suspension(&s, &[true, true]);
    split.reconstruct_channel();
    for offset in [-0.05, -0.01, 0., 0.015, 0.05] {
        let q = p + Vec3::X * offset;
        let a = full.sample_channel(q);
        let b = split.sample_channel(q);
        assert!((a.0 - b.0).abs() < 1e-7);
        assert!((a.1 - b.1).abs() < 1e-3);
    }
}

#[test]
#[should_panic(expected = "complete grain wetness mask")]
fn suspension_mask_cannot_silently_omit_grain_entries() {
    SlurryField::new_with_suspension(&column(), &[]);
}
