//! Riffle Geometry Test Harness
//!
//! Tests each riffle mode to verify:
//! 1. Vortex formation behind riffles (using enstrophy)
//! 2. Sediment trapping behavior
//! 3. Flow persistence
//!
//! Run with: cargo run --example riffle_test --release

use sim::{
    create_sluice_with_mode, FlipSimulation, RiffleMode, SluiceConfig,
    ParticleMaterial,
};

// Simulation parameters
const SIM_WIDTH: usize = 200;
const SIM_HEIGHT: usize = 100;
const CELL_SIZE: f32 = 4.0;
const DT: f32 = 0.008;

/// Test result for a single riffle mode
struct RiffleTestResult {
    mode: RiffleMode,
    // Vortex metrics
    peak_enstrophy: f32,
    final_enstrophy: f32,
    enstrophy_stable: bool,
    // Flow metrics
    avg_flow_velocity: f32,
    // Sediment metrics (if applicable)
    sand_trapped: usize,
    gold_trapped: usize,
}

impl std::fmt::Display for RiffleTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:16} | Peak E: {:6.1} | Final E: {:6.1} | Stable: {:5} | Flow: {:5.1} | Sand: {:4} | Gold: {:4}",
            self.mode.name(),
            self.peak_enstrophy,
            self.final_enstrophy,
            if self.enstrophy_stable { "YES" } else { "NO" },
            self.avg_flow_velocity,
            self.sand_trapped,
            self.gold_trapped,
        )
    }
}

/// Test a single riffle mode with water only
fn test_riffle_water_only(mode: RiffleMode, steps: usize) -> RiffleTestResult {
    let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    let config = SluiceConfig {
        slope: 0.25,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: mode,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(&mut sim, &config);

    let inlet_x = 10.0 * CELL_SIZE;
    let inlet_y = (SIM_HEIGHT as f32 / 5.0) * CELL_SIZE;
    let inlet_vx = 80.0;
    let inlet_vy = 5.0;

    let mut enstrophy_history = Vec::with_capacity(steps);
    let mut peak_enstrophy = 0.0f32;

    // Warm-up: fill sluice with water
    for _ in 0..100 {
        sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, 4);
        sim.update(DT);
    }

    // Main simulation loop
    for step in 0..steps {
        // Spawn water at inlet
        sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, 4);

        sim.update(DT);

        // Compute enstrophy every 10 steps
        if step % 10 == 0 {
            let enstrophy = sim.update_and_compute_enstrophy();
            enstrophy_history.push(enstrophy);
            peak_enstrophy = peak_enstrophy.max(enstrophy);
        }
    }

    // Analyze stability: check if enstrophy variance in last 20% is low
    let final_portion = &enstrophy_history[enstrophy_history.len() * 4 / 5..];
    let final_enstrophy: f32 = final_portion.iter().sum::<f32>() / final_portion.len() as f32;
    let variance: f32 = final_portion.iter()
        .map(|e| (e - final_enstrophy).powi(2))
        .sum::<f32>() / final_portion.len() as f32;
    let enstrophy_stable = variance.sqrt() < final_enstrophy * 0.3; // 30% variation threshold

    // Compute average flow velocity
    let avg_flow: f32 = sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity.x)
        .sum::<f32>() / sim.particles.len().max(1) as f32;

    RiffleTestResult {
        mode,
        peak_enstrophy,
        final_enstrophy,
        enstrophy_stable,
        avg_flow_velocity: avg_flow,
        sand_trapped: 0,
        gold_trapped: 0,
    }
}

/// Test a riffle mode with sediment (sand + gold)
fn test_riffle_with_sediment(mode: RiffleMode, steps: usize) -> RiffleTestResult {
    let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    let config = SluiceConfig {
        slope: 0.25,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: mode,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(&mut sim, &config);

    let inlet_x = 10.0 * CELL_SIZE;
    let inlet_y = (SIM_HEIGHT as f32 / 5.0) * CELL_SIZE;
    let inlet_vx = 80.0;
    let inlet_vy = 5.0;

    let mut peak_enstrophy = 0.0f32;

    // Warm-up: fill with water
    for _ in 0..100 {
        sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, 4);
        sim.update(DT);
    }

    // Main loop with sediment
    for step in 0..steps {
        sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, 4);

        // Spawn sand every 4 frames
        if step % 4 == 0 {
            sim.spawn_sand(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
        }

        // Spawn gold every 20 frames
        if step % 20 == 0 {
            sim.spawn_gold(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
        }

        sim.update(DT);

        if step % 10 == 0 {
            let enstrophy = sim.update_and_compute_enstrophy();
            peak_enstrophy = peak_enstrophy.max(enstrophy);
        }
    }

    let final_enstrophy = sim.update_and_compute_enstrophy();

    // Count trapped sediment (particles with low velocity behind riffles)
    let trap_velocity_threshold = 10.0;
    let riffle_zone_start = config.slick_plate_len as f32 * CELL_SIZE;

    let mut sand_trapped = 0usize;
    let mut gold_trapped = 0usize;

    for particle in sim.particles.iter() {
        if particle.position.x > riffle_zone_start && particle.velocity.length() < trap_velocity_threshold {
            match particle.material {
                ParticleMaterial::Sand => sand_trapped += 1,
                ParticleMaterial::Gold => gold_trapped += 1,
                _ => {}
            }
        }
    }

    let avg_flow: f32 = sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity.x)
        .sum::<f32>() / sim.particles.iter().filter(|p| p.material == ParticleMaterial::Water).count().max(1) as f32;

    RiffleTestResult {
        mode,
        peak_enstrophy,
        final_enstrophy,
        enstrophy_stable: true, // Not checking stability in sediment test
        avg_flow_velocity: avg_flow,
        sand_trapped,
        gold_trapped,
    }
}

fn main() {
    println!("=== RIFFLE GEOMETRY TEST HARNESS ===\n");

    let modes = [
        RiffleMode::None,
        RiffleMode::ClassicBattEdge,
        RiffleMode::DoublePocket,
        RiffleMode::ParallelBoards,
        RiffleMode::VNotch,
        RiffleMode::StepCascade,
    ];

    // Test 1: Water-only (vortex formation)
    println!("TEST 1: WATER-ONLY (Vortex Formation)");
    println!("{}", "=".repeat(100));
    println!("{:16} | {:12} | {:12} | {:7} | {:7} | {:6} | {:6}",
        "Mode", "Peak E", "Final E", "Stable", "Flow", "Sand", "Gold");
    println!("{}", "-".repeat(100));

    for mode in &modes {
        let result = test_riffle_water_only(*mode, 500);
        println!("{}", result);
    }

    println!("\n");

    // Test 2: With sediment (trapping behavior)
    println!("TEST 2: WITH SEDIMENT (Trapping Behavior)");
    println!("{}", "=".repeat(100));
    println!("{:16} | {:12} | {:12} | {:7} | {:7} | {:6} | {:6}",
        "Mode", "Peak E", "Final E", "Stable", "Flow", "Sand", "Gold");
    println!("{}", "-".repeat(100));

    for mode in &modes {
        let result = test_riffle_with_sediment(*mode, 800);
        println!("{}", result);
    }

    println!("\n=== TEST COMPLETE ===");
    println!("\nExpected behavior:");
    println!("- Enstrophy should be higher with riffles than without (vortex formation)");
    println!("- Stable = YES indicates persistent vortices (not drifting)");
    println!("- Sand trapped > Gold trapped indicates density-based stratification");
    println!("- Different modes should show visibly different behavior");
    println!("\nIf two modes look identical, the geometry implementation may have failed.");
}
