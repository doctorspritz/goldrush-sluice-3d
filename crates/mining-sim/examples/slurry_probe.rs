//! Read-only population and projection-cost receipt from a real saved shift.
use mining_sim::{slurry, storage};
use std::{path::PathBuf, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "slurry_probe SAVE_JSON OUTPUT_JSON");
    let path = PathBuf::from(&args[2]);
    assert!(!path.exists(), "preserve existing receipt");
    let s = storage::load(std::path::Path::new(&args[1])).unwrap();
    let before = s.save_json().unwrap();
    let started = Instant::now();
    for _ in 0..20 {
        std::hint::black_box(slurry::SlurryField::new(&s.sluice));
    }
    let mean_ms = started.elapsed().as_secs_f64() * 1000. / 20.;
    let mut result = slurry::diagnostics(&s.sluice);
    result["mean_field_build_ms"] = serde_json::json!(mean_ms);
    result["sim_seconds"] = serde_json::json!(s.sim_time);
    result["source_save"] = serde_json::json!(args[1]);
    assert_eq!(before, s.save_json().unwrap());
    std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("{result}");
}
