#![allow(
    clippy::field_reassign_with_default,
    reason = "Sluice persistence fields are private; operating controls are public"
)]
use glam::{Mat3, Quat};
use mining_sim::{sluice::Grain, GrainKind, Material, Placement, Site, Sluice, Vec3, FIXED_DT};
#[test]
fn feed_represents_volume_without_inventing_large_gold_nuggets() {
    let mut s = Sluice::default();
    s.placement = Some(Placement::suggested(&Site::new()));
    s.pump_on = true;
    s.feed_on = true;
    s.hopper = Material {
        sand_m3: 0.005,
        gravel_m3: 0.005,
        gold_g: 0.001,
        ..Default::default()
    };
    let initial = s.hopper;
    for _ in 0..300 {
        s.feed(FIXED_DT, true);
    }
    let active = s.active_material();
    assert!((active.volume() - initial.volume()).abs() < 1e-10);
    assert!((active.gold_g - initial.gold_g).abs() < 1e-12);
    assert!(s.grains.len() > 100);
    let positions: std::collections::HashSet<_> = s
        .grains
        .iter()
        .map(|g| g.position.to_array().map(f32::to_bits))
        .collect();
    assert_eq!(
        positions.len(),
        s.grains.len(),
        "feed must not spawn coincident DEM parcels"
    );
    assert!(s
        .grains
        .iter()
        .all(|g| (0.10..=0.14).contains(&g.position.x)));
    for g in &s.grains {
        assert!(g.volume_resolved);
        if g.kind == GrainKind::Gold {
            let physical_mass =
                std::f64::consts::PI * (g.diameter_m as f64).powi(3) / 6.0 * 19300.0 * 1000.0;
            assert!((g.material.gold_g - physical_mass).abs() < 1e-12);
        }
    }
    s.feed_on = false;
    s.feed(FIXED_DT, true);
    assert!((s.active_material().gold_g - initial.gold_g).abs() < 1e-12);
    assert!(s.valid());
    let restored: Sluice = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert!(restored.valid());
    assert_eq!(
        restored.active_material().volume(),
        s.active_material().volume()
    );
    assert!((restored.active_material().gold_g - s.active_material().gold_g).abs() < 1e-15);
}
fn grain(x: f32, resting: bool) -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Sand,
        position: Vec3::new(x, 0.1, 0.1),
        velocity: Vec3::ZERO,
        material: Material {
            sand_m3: 0.000002,
            ..Default::default()
        },
        resting,
        diameter_m: 0.0005,
        affine: Mat3::ZERO,
        rotation: Quat::IDENTITY,
        angular_velocity: Vec3::ZERO,
    }
}
#[test]
fn cleanup_requires_settling_and_only_removes_mat_material() {
    let mut s = Sluice::default();
    s.grains = vec![grain(0.2, true), grain(0.5, false)];
    assert!(s.cleanup().is_err());
    assert_eq!(s.grains.len(), 2);
    s.grains[1].resting = true;
    let result = s.cleanup().unwrap();
    assert_eq!(result.sand_m3, 0.000002);
    assert_eq!(s.grains.len(), 1);
    assert_eq!(s.grains[0].position.x, 0.2);
    assert!(s.cleanup().is_err());
}
#[test]
fn old_saves_keep_their_original_parcel_interpretation() {
    let mut json = serde_json::to_value(Sluice::default()).unwrap();
    for field in [
        "granular_packets",
        "riffles_enabled",
        "fluid_clock",
        "pending_gold_g",
    ] {
        json.as_object_mut().unwrap().remove(field);
    }
    let s: Sluice = serde_json::from_value(json).unwrap();
    assert!(!s.granular_packets);
    assert!(s.riffles_enabled);
    assert_eq!(s.parcel_volume(), 0.002);
    assert!(s.valid());
}

#[test]
fn cleanup_collects_only_the_settled_mat_fraction_and_preserves_moving_material() {
    let mut s = Sluice::default();
    s.grains = vec![grain(0.2, false), grain(0.5, true), grain(0.8, false)];
    let before = s.active_material();
    let collected = s.cleanup().unwrap();
    assert_eq!(collected.sand_m3, 0.000002);
    assert_eq!(s.grains.len(), 2);
    assert!(s.grains.iter().all(|g| !g.resting));
    assert!((s.active_material().volume() + collected.volume() - before.volume()).abs() < 1e-12);
    assert!((s.active_material().gold_g + collected.gold_g - before.gold_g).abs() < 1e-12);
}

#[test]
fn cleanup_preserves_a_single_fine_quartz_grain_below_bulk_empty_threshold() {
    let mut s = Sluice::default();
    let mut fine = grain(0.5, true);
    fine.diameter_m = 0.00008;
    fine.material = Material::sand_from_solid_volume(
        std::f64::consts::PI / 6. * (fine.diameter_m as f64).powi(3),
    );
    let volume = fine.material.sand_m3;
    assert!(
        fine.material.is_empty(),
        "bulk tolerance exceeds a real fine grain"
    );
    s.grains.push(fine);
    let collected = s.cleanup().unwrap();
    assert_eq!(collected.sand_m3, volume);
    assert_eq!(s.concentrate.sand_m3, volume);
    assert!(s.grains.is_empty());
}
