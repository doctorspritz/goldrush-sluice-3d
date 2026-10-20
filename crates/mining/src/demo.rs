//! Finite, already-running inspection scene, warmed by the real solver.
use mining_sim::Session;
use std::io::Read;
pub fn session() -> Result<Session, String> {
    let mut decoder =
        flate2::read::ZlibDecoder::new(&include_bytes!("../assets/demo/flowing.json.zlib")[..]);
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .map_err(|e| e.to_string())?;
    let mut session = Session::load_json(&json)?;
    session.message = "Flowing demo: real paydirt and water. Click to look; Esc frees the mouse. P water, O feed.".into();
    Ok(session)
}
pub fn initial() -> (Session, bool) {
    let fresh =
        std::env::args().any(|a| matches!(a.as_str(), "--new-shift" | "--validate" | "--load"));
    if fresh {
        (Session::new(), false)
    } else {
        (session().expect("Bundled flowing demo must be valid"), true)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn bundled_demo_has_real_flow_finite_paydirt_and_balanced_ledger() {
        let s = super::session().unwrap();
        assert!(s.sluice.water.len() > 100_000);
        assert_eq!(
            s.sluice.profile,
            mining_sim::sluice::HydraulicProfile::OriginalDemo
        );
        assert!(s.sluice.pump_on && s.sluice.feed_on);
        assert!(s.sluice.hopper.volume() > 0.4 && s.sluice.hopper.volume() < 0.85);
        assert!(s.sluice.water_returned_m3 > 0.005 && s.sluice.tailings.volume() > 0.001);
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
