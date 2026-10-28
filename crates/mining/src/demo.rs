//! Finite, already-running inspection scene, warmed by the real solver.
use mining_sim::Session;
use std::io::Read;
pub fn session() -> Result<Session, String> {
    configured_session(std::env::args().any(|a| a == "--bulk-screen"))
}
fn configured_session(bulk: bool) -> Result<Session, String> {
    let mut decoder =
        flate2::read::ZlibDecoder::new(&include_bytes!("../assets/demo/flowing.json.zlib")[..]);
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .map_err(|e| e.to_string())?;
    let mut session = Session::load_json(&json)?;
    session.enable_spatial_tailings()?;
    session.enable_loose_gold()?;
    session.enable_screen()?;
    if bulk {
        session
            .plant
            .as_mut()
            .unwrap()
            .set_bulk_screen(Some(mining_sim::screen::BulkDeck::default()))?;
    }
    session.plant.as_mut().unwrap().screen.motor_on = true;
    session.message = "Running plant: finite hopper / dry screen / wet sluice. K screen · M empty-deck aperture · L empty stopped screen.".into();
    Ok(session)
}
pub fn initial() -> (Session, bool) {
    let fresh =
        std::env::args().any(|a| matches!(a.as_str(), "--new-shift" | "--validate" | "--load"));
    if fresh {
        (Session::new_operation(), false)
    } else {
        (session().expect("Bundled flowing demo must be valid"), true)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn opt_in_bulk_demo_and_restarts_use_the_same_saved_finite_circuit() {
        for _ in 0..2 {
            let s = super::configured_session(true).unwrap();
            let p = s.plant.as_ref().unwrap();
            assert!(p.screen.motor_on && p.screen.bulk_deck().is_some());
            assert_eq!(p.model_version, 2);
            assert!(p.connected_to(&s.sluice));
            assert!(s.sluice.water.len() > 10000 && s.sluice.hopper.volume() > 0.4);
            assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
            let restored = mining_sim::Session::load_json(&s.save_json().unwrap()).unwrap();
            assert_eq!(
                serde_json::to_value(&restored.plant).unwrap(),
                serde_json::to_value(&s.plant).unwrap()
            );
        }
        assert!(super::configured_session(false)
            .unwrap()
            .plant
            .unwrap()
            .screen
            .bulk_deck()
            .is_none());
    }

    #[test]
    fn bundled_demo_has_real_flow_finite_paydirt_and_balanced_ledger() {
        let s = super::session().unwrap();
        for p in [s.machine.arm().lip, s.machine.arm().heel] {
            assert!(
                p.y >= s.site.height(p.x, p.z) + 0.05,
                "fresh demo tool starts in terrain"
            );
        }
        assert!(s.sluice.water.len() > 10_000 && s.sluice.granular_packets);
        assert_eq!(
            s.sluice.contact_model,
            mining_sim::sluice::ContactModel::Coulomb
        );
        assert_eq!(
            s.sluice.feed_grading,
            mining_sim::sluice::FeedGrading::Alluvial
        );
        assert!(s
            .sluice
            .grains
            .iter()
            .any(|g| g.material.soil_m3 > 0. && g.diameter_m < 0.0002));
        assert_eq!(
            s.sluice.profile,
            mining_sim::sluice::HydraulicProfile::OriginalDemo
        );
        assert!(s.sluice.pump_on && s.sluice.feed_on);
        assert!(s.sluice.hopper.volume() > 0.4 && s.sluice.hopper.volume() < 0.85);
        assert!(
            s.sluice.water_returned_m3 > 0.005
                && s.tailings.as_ref().unwrap().historic_bed.volume() > 0.0001
        );
        assert!(!s.sluice.grains.is_empty());
        assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
        let (volume, positions) = (
            s.sluice.hopper.volume(),
            s.sluice
                .water
                .iter()
                .map(|p| p.position)
                .collect::<Vec<_>>(),
        );
        let restored = mining_sim::Session::load_json(&s.save_json().unwrap()).unwrap();
        assert_eq!(restored.sluice.hopper.volume(), volume);
        assert_eq!(
            restored
                .sluice
                .water
                .iter()
                .map(|p| p.position)
                .collect::<Vec<_>>(),
            positions
        );
    }
}
