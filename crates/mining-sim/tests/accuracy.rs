#![allow(
    clippy::field_reassign_with_default,
    reason = "Sluice persistence fields are private"
)]
use glam::{Mat3, Quat};
use mining_sim::{
    sluice::{ContactModel, Grain},
    GrainKind, Material, Session, Sluice, Vec3,
};

#[test]
fn one_physical_grain_has_its_physical_contact_radius() {
    let d = 0.0003_f32;
    let volume = std::f64::consts::PI / 6. * (d as f64).powi(3);
    let grain = Grain {
        volume_resolved: true,
        kind: GrainKind::Gold,
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        material: Material {
            gold_g: volume * 19300. * 1000.,
            ..Default::default()
        },
        resting: false,
        diameter_m: d,
        affine: Mat3::ZERO,
        rotation: Quat::IDENTITY,
        angular_velocity: Vec3::ZERO,
    };
    assert_eq!(grain.collision_radius(), d * 0.5);
    let packet = Grain {
        material: grain.material.scaled(1000.),
        ..grain
    };
    assert!(packet.collision_radius() > d * 4.);
}

#[test]
fn contact_model_and_parcel_resolution_persist_and_legacy_defaults_are_explicit() {
    let mut s = Session::new();
    assert_eq!(s.sluice.contact_model, ContactModel::Coulomb);
    s.sluice.parcel_volume_m3 = Some(0.5e-6);
    let restored = Session::load_json(&s.save_json().unwrap()).unwrap();
    assert_eq!(restored.sluice.parcel_volume(), 0.5e-6);
    assert_eq!(restored.sluice.contact_model, ContactModel::Coulomb);
    let mut old: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    old["sluice"]
        .as_object_mut()
        .unwrap()
        .remove("contact_model");
    old["sluice"]
        .as_object_mut()
        .unwrap()
        .remove("parcel_volume_m3");
    let legacy = Session::load_json(&old.to_string()).unwrap();
    assert_eq!(legacy.sluice.contact_model, ContactModel::LegacyDamping);
    assert_eq!(legacy.sluice.parcel_volume(), 2e-6);
}

#[test]
fn invalid_packet_resolution_is_rejected() {
    for volume in [0., -1., f64::NAN, 3e-6] {
        let mut s = Sluice::default();
        s.parcel_volume_m3 = Some(volume);
        assert!(!s.valid());
    }
}
