#[cfg(feature = "legacy-scenarios")]
use game::scenario::Scenario;
#[cfg(feature = "legacy-scenarios")]
use game::example_utils::simulation::SimulationManager;
use game::example_utils::testing::{SimulationMetrics, GoldenValue};
use std::path::Path;
use std::collections::HashMap;
use glam::Vec3;

#[cfg(feature = "legacy-scenarios")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scenarios = [
        "crates/game/scenarios/dem_floor_collision.json",
        "crates/game/scenarios/dem_wall_collision.json",
        "crates/game/scenarios/dem_density_separation.json",
        "crates/game/scenarios/dem_settling_time.json",
    ];

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                 SIMULATION REGRESSION TEST SUITE                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let args: Vec<String> = std::env::args().collect();
    let generate_mode = args.contains(&"--generate".to_string());
    
    if generate_mode {
        println!("MODE: GENERATING GOLDEN VALUES");
    } else {
        println!("MODE: VERIFYING AGAINST GOLDEN VALUES");
    }

    let mut all_passed = true;

    for scenario_path in scenarios {
        let path = Path::new(scenario_path);
        if !path.exists() {
            println!("Skipping missing scenario: {}", scenario_path);
            continue;
        }

        let scenario = Scenario::load_json(path)?;
        println!("Running Scenario: {}", scenario.name);
        
        let mut manager = SimulationManager::new(60, 60, 60, 0.01);
        manager.load_scenario(&scenario);

        // Run for 300 steps (5 seconds) to allow settling
        let dt = 1.0 / 60.0;
        let total_frames = 300;
        
        // Simple progress bar
        print!("  Progress: [");
        for i in 0..total_frames {
            manager.update(dt);
            if i % (total_frames / 20) == 0 {
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }
        }
        println!("] Done.");

        // Calculate Metrics
        let particle_count = manager.dem.clumps.len();
        let mut vel_sum = Vec3::ZERO;
        let mut density_sum = 0.0;
        let mut settled_count = 0;

        for clump in &manager.dem.clumps {
            vel_sum += clump.velocity;
            // Density is property of material, but here we can check if it matches expected
            // For now, let's just use Clump mass/volume? No, that's constant.
            // Let's use kinetic energy as a proxy for "activity"?
            // Or just check mean velocity.
            if clump.velocity.length() < 0.01 {
                settled_count += 1;
            }
        }

        let mean_velocity = if particle_count > 0 {
            vel_sum / particle_count as f32
        } else {
            Vec3::ZERO
        };

        // For density, we need to access the FLIP grid if there are fluid particles.
        // But these are DEM tests mostly.
        
        let metrics = SimulationMetrics {
            frame: manager.frame,
            particle_count,
            mean_velocity: mean_velocity.to_array(),
            mean_density: 0.0, // TODO: FLIP density
            settled_count,
            additional_metrics: HashMap::from([
                ("mean_velocity_x".to_string(), mean_velocity.x),
                ("mean_velocity_y".to_string(), mean_velocity.y),
                ("mean_velocity_z".to_string(), mean_velocity.z),
            ]),
        };

        println!("  - Final Particle Count: {}", metrics.particle_count);
        println!("  - Settled Particles: {}/{}", metrics.settled_count, metrics.particle_count);
        println!("  - Mean Velocity: {:.4}, {:.4}, {:.4}", metrics.mean_velocity[0], metrics.mean_velocity[1], metrics.mean_velocity[2]);

        let golden_path = path.with_extension("golden.json");

        if generate_mode {
            let golden = GoldenValue {
                scenario_name: scenario.name.clone(),
                target_frame: total_frames,
                metrics: metrics.clone(),
                tolerances: HashMap::from([
                    ("mean_velocity_x".to_string(), 0.01),
                    ("mean_velocity_y".to_string(), 0.01),
                    ("mean_velocity_z".to_string(), 0.01),
                ]),
            };
            if let Err(e) = golden.save_json(&golden_path) {
                println!("  [ERROR] Failed to save golden value: {}", e);
                all_passed = false;
            } else {
                println!("  [SAVED] Golden value written to {:?}", golden_path);
            }
        } else {
            if golden_path.exists() {
                match GoldenValue::load_json(&golden_path) {
                    Ok(golden) => {
                        if let Err(errors) = golden.compare(&metrics) {
                            println!("  [FAIL] Regression check failed:");
                            for err in errors {
                                println!("    - {}", err);
                            }
                            all_passed = false;
                        } else {
                            println!("  [PASS] Matches golden values.");
                        }
                    }
                    Err(e) => {
                        println!("  [ERROR] Failed to load golden value: {}", e);
                        all_passed = false;
                    }
                }
            } else {
                println!("  [WARN] No golden value found. Run with --generate to create one.");
            }
        }
        println!();
    }

    if all_passed {
        println!("OVERALL STATUS: SUCCESS");
    } else {
        println!("OVERALL STATUS: FAILURE");
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(not(feature = "legacy-scenarios"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scenario regression harness is disabled (missing legacy scenario APIs).");
    Ok(())
}
