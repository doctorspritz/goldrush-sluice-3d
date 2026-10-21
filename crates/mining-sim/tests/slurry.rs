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
    assert!(p.represented_grains() > 3_000_000.);
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
    assert!((v - 3e-6).abs() < 1e-15);
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
