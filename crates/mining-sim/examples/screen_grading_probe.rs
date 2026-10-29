//! Bounded finite-cohort grading diagnostic; no injected recovery targets.
//! Detailed wires are a dilute numerical reference, not measured plant data.
use glam::{DVec3, Mat3, Quat};
use mining_sim::{
    screen::{BulkDeck, Screen},
    sluice::Grain,
    GrainKind, Material, Vec3,
};
use std::{error::Error, fs, path::Path, time::Instant};

fn run(
    bulk: bool,
    diameter: f32,
    weighted: bool,
    motor: bool,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut s = Screen::new(Vec3::ZERO, 0.);
    if bulk {
        s.set_bulk_deck(Some(BulkDeck::default()))?;
    }
    let volume = if weighted { 2e-6 } else { 1e-10 };
    let grains: Vec<_> = (0..64)
        .map(|i| {
            // Low-discrepancy phases cover cells without aligning all grain centres
            // to the wire pitch. Both models receive identical world positions.
            let x = 0.04 + (i % 8) as f64 * 0.08 + ((i * 37 % 101) as f64 / 101.) * 0.005;
            let z = -0.14 + (i / 8) as f64 * 0.04 + ((i * 53 % 103) as f64 / 103.) * 0.005;
            Grain {
                kind: GrainKind::Gravel,
                material: Material {
                    gravel_m3: volume,
                    ..Default::default()
                },
                diameter_m: diameter,
                position: s
                    .to_world(DVec3::new(
                        x,
                        0.025 - x * s.config.slope_deg.to_radians().tan(),
                        z,
                    ))
                    .as_vec3(),
                velocity: Vec3::ZERO,
                affine: Mat3::ZERO,
                rotation: Quat::IDENTITY,
                angular_velocity: Vec3::ZERO,
                resting: false,
                volume_resolved: true,
            }
        })
        .collect();
    s.admit(&grains, 0.)?;
    s.motor_on = motor;
    let start = Instant::now();
    let mut failure = None;
    let mut contact_intervals = 0;
    let mut max_depth = 0.0_f64;
    for tick in 0..240 {
        if let Err(e) = s.step(1. / 60.) {
            failure = Some(format!("tick {tick}: {e}"));
            break;
        }
        if !s.valid() {
            return Err("invalid finite screen state".into());
        }
        contact_intervals += s.last_step.contacts;
        if let Some(field) = s.bulk_bed_field() {
            max_depth = max_depth.max(field.cells().iter().map(|c| c.depth_m).fold(0., f64::max));
        }
    }
    let mut accounted = s.material();
    accounted.add(s.released);
    let error = accounted.volume() - s.admitted.volume();
    if error.abs() > 1e-12 {
        return Err(format!("finite stock error {error}").into());
    }
    Ok(
        serde_json::json!({"model":s.processing_model_version(),"diameter_m":diameter,"weighted":weighted,"motor":motor,
        "ticks":240,"physical_time_s":s.elapsed_time_s,"cpu_seconds":start.elapsed().as_secs_f64(),"failure":failure,"contacts":contact_intervals,
        "active":s.particles.len(),"under":s.undersize.len(),"reject":s.oversize.len(),"spill":s.spills.len(),
        "max_bed_depth_m":max_depth,"material_error_m3":error,"admitted":s.admitted,"retained":accounted}),
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("screen_grading_probe FRESH_OUTPUT_DIRECTORY".into());
    }
    let dir = Path::new(&args[1]);
    fs::create_dir(dir)?;
    let mut rows = Vec::new();
    for diameter in [0.0001, 0.0015, 0.0028, 0.0032] {
        for weighted in [false, true] {
            for motor in [false, true] {
                for bulk in [false, true] {
                    let row = run(bulk, diameter, weighted, motor)?;
                    println!("{}", row);
                    rows.push(row);
                }
            }
        }
    }
    fs::write(dir.join("grading.json"), serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
