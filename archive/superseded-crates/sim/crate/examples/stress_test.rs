//! FLIP Stress Test Suite - Known Risk Scenarios
//! Run with: cargo run --example stress_test -p sim --release
//!
//! Tests scenarios known to break FLIP simulations:
//! 1. Dam break (column collapse) - incompressibility under compression
//! 2. High-speed jet impact - velocity explosion risk
//! 3. Dense particle packing - pressure solver stress
//! 4. Boundary corner trap - particles stuck in corners
//! 5. Long-term stability - divergence drift over time + FPS degradation

use sim::grid::{CellType, Grid};
use sim::FlipSimulation;
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

/// Track frame timing for FPS metrics
struct FpsTracker {
    frame_times: Vec<f64>, // in milliseconds
}

impl FpsTracker {
    fn new() -> Self {
        Self { frame_times: Vec::new() }
    }

    fn record(&mut self, ms: f64) {
        self.frame_times.push(ms);
    }

    fn avg_fps(&self) -> f64 {
        if self.frame_times.is_empty() { return 0.0; }
        let avg_ms = self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64;
        1000.0 / avg_ms
    }

    fn min_fps(&self) -> f64 {
        if self.frame_times.is_empty() { return 0.0; }
        let max_ms = self.frame_times.iter().cloned().fold(0.0f64, f64::max);
        1000.0 / max_ms
    }

    fn first_half_avg(&self) -> f64 {
        if self.frame_times.len() < 2 { return 0.0; }
        let half = self.frame_times.len() / 2;
        let avg_ms = self.frame_times[..half].iter().sum::<f64>() / half as f64;
        1000.0 / avg_ms
    }

    fn second_half_avg(&self) -> f64 {
        if self.frame_times.len() < 2 { return 0.0; }
        let half = self.frame_times.len() / 2;
        let avg_ms = self.frame_times[half..].iter().sum::<f64>() / (self.frame_times.len() - half) as f64;
        1000.0 / avg_ms
    }

    fn degradation_ratio(&self) -> f64 {
        let first = self.first_half_avg();
        let second = self.second_half_avg();
        if first <= 0.0 { return 1.0; }
        second / first
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         FLIP STRESS TEST - KNOWN RISK SCENARIOS               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let mut total_pass = 0;
    let mut total_fail = 0;

    // Test 1: Dam break
    let (p, f) = test_dam_break();
    total_pass += p;
    total_fail += f;

    // Test 2: High-speed jet
    let (p, f) = test_high_speed_jet();
    total_pass += p;
    total_fail += f;

    // Test 3: Dense packing
    let (p, f) = test_dense_packing();
    total_pass += p;
    total_fail += f;

    // Test 4: Corner trap
    let (p, f) = test_corner_trap();
    total_pass += p;
    total_fail += f;

    // Test 5: Long-term stability
    let (p, f) = test_long_term_stability();
    total_pass += p;
    total_fail += f;

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  TOTAL: {} PASSED, {} FAILED                                   ║", total_pass, total_fail);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    if total_fail == 0 {
        println!("\n✅ ALL STRESS TESTS PASSED");
    } else {
        println!("\n❌ {} STRESS TESTS FAILED", total_fail);
        std::process::exit(1);
    }
}

/// Test 1: Dam Break (Column Collapse)
/// Risk: Water compresses instead of spreading horizontally
/// Success: Particles spread out, velocities stay bounded
fn test_dam_break() -> (usize, usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: DAM BREAK (Column Collapse)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Risk: Water compresses instead of spreading");
    println!("  Setup: 10x10 particle column, no obstacles\n");

    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 32;
    const HEIGHT: usize = 24;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create floor
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }

    // Spawn dense column of water on the left
    let column_x = 20.0;
    let column_y = 40.0;
    for row in 0..10 {
        for col in 0..10 {
            let x = column_x + col as f32 * 3.0;
            let y = column_y + row as f32 * 3.0;
            sim.spawn_water(x, y, 0.0, 0.0, 1);
        }
    }

    let initial_count = sim.particles.len();
    println!("  Initial: {} particles", initial_count);

    // Track metrics
    let mut max_velocity: f32 = 0.0;
    let mut min_x: f32 = f32::MAX;
    let mut max_x: f32 = f32::MIN;

    // Run for 120 frames (2 seconds)
    for frame in 0..120 {
        sim.update(DT);

        for p in sim.particles.iter() {
            let speed = p.velocity.length();
            max_velocity = max_velocity.max(speed);
            min_x = min_x.min(p.position.x);
            max_x = max_x.max(p.position.x);
        }

        if frame % 30 == 0 {
            sim.grid.compute_divergence();
            let div = sim.grid.total_divergence();
            let spread = max_x - min_x;
            println!("  Frame {:3}: spread={:5.1}px, max_v={:5.1}, div={:5.1}",
                frame, spread, max_velocity, div);
        }
    }

    let final_spread = max_x - min_x;
    let final_count = sim.particles.len();

    println!("\n  Results:");
    println!("    Spread: {:.1}px (started ~30px)", final_spread);
    println!("    Max velocity: {:.1}", max_velocity);
    println!("    Particles: {} → {}", initial_count, final_count);

    // Check 1: Water should spread out (at least 45px wide, started ~30px)
    // 50% spread is reasonable for 2 seconds of dam break
    print!("  [1.1] Water spreads (>45px): ");
    if final_spread > 45.0 {
        println!("PASS ({:.1}px)", final_spread);
        pass += 1;
    } else {
        println!("FAIL ({:.1}px - water compressed!)", final_spread);
        fail += 1;
    }

    // Check 2: Velocity bounded
    print!("  [1.2] Velocity bounded (<300): ");
    if max_velocity < 300.0 {
        println!("PASS ({:.1})", max_velocity);
        pass += 1;
    } else {
        println!("FAIL ({:.1} - explosion!)", max_velocity);
        fail += 1;
    }

    (pass, fail)
}

/// Test 2: High-Speed Jet Impact
/// Risk: Velocity explosion on impact, particles flying off
/// Success: Velocities damped on impact, particles retained
fn test_high_speed_jet() -> (usize, usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: HIGH-SPEED JET IMPACT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Risk: Velocity explosion on impact");
    println!("  Setup: Fast particles hitting a wall\n");

    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 32;
    const HEIGHT: usize = 24;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create floor and right wall
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(WIDTH - 1, j);
    }

    // Spawn high-speed water jet aimed at the wall
    let jet_x = 20.0;
    let jet_y = 50.0;
    let jet_vx = 200.0; // High speed!
    for i in 0..20 {
        sim.spawn_water(jet_x + i as f32 * 2.0, jet_y, jet_vx, 0.0, 1);
    }

    let initial_count = sim.particles.len();
    println!("  Initial: {} particles at vx={}", initial_count, jet_vx);

    let mut max_velocity: f32 = 0.0;
    let mut velocity_after_impact: f32 = 0.0;

    // Run for 60 frames (1 second)
    for frame in 0..60 {
        sim.update(DT);

        let mut frame_max_v: f32 = 0.0;
        for p in sim.particles.iter() {
            frame_max_v = frame_max_v.max(p.velocity.length());
        }
        max_velocity = max_velocity.max(frame_max_v);

        if frame == 30 {
            velocity_after_impact = frame_max_v;
        }

        if frame % 15 == 0 {
            sim.grid.compute_divergence();
            let div = sim.grid.total_divergence();
            println!("  Frame {:2}: {} particles, max_v={:5.1}, div={:5.1}",
                frame, sim.particles.len(), frame_max_v, div);
        }
    }

    let final_count = sim.particles.len();

    println!("\n  Results:");
    println!("    Max velocity ever: {:.1}", max_velocity);
    println!("    Velocity after impact: {:.1}", velocity_after_impact);
    println!("    Particles: {} → {}", initial_count, final_count);

    // Check 1: Velocity never explodes beyond clamp
    print!("  [2.1] No velocity explosion (<350): ");
    if max_velocity < 350.0 {
        println!("PASS ({:.1})", max_velocity);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", max_velocity);
        fail += 1;
    }

    // Check 2: Most particles retained
    print!("  [2.2] Particles retained (>80%): ");
    let retention = final_count as f32 / initial_count as f32;
    if retention > 0.8 {
        println!("PASS ({:.0}%)", retention * 100.0);
        pass += 1;
    } else {
        println!("FAIL ({:.0}%)", retention * 100.0);
        fail += 1;
    }

    (pass, fail)
}

/// Test 3: Dense Particle Packing
/// Risk: Pressure solver fails under extreme density, particles overlap
/// Success: Particles separate, divergence reduces
fn test_dense_packing() -> (usize, usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: DENSE PARTICLE PACKING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Risk: Pressure solver fails under extreme density");
    println!("  Setup: Many particles in small area\n");

    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 16;
    const HEIGHT: usize = 16;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
        sim.grid.set_solid(i, 0);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }

    // Spawn VERY dense cluster in center
    let center_x = 32.0;
    let center_y = 32.0;
    for i in 0..50 {
        let angle = i as f32 * 0.1256; // Golden angle
        let r = (i as f32).sqrt() * 1.5;
        let x = center_x + r * angle.cos();
        let y = center_y + r * angle.sin();
        sim.spawn_water(x, y, 0.0, 0.0, 1);
    }

    let initial_count = sim.particles.len();
    println!("  Initial: {} particles in tight cluster", initial_count);

    // Measure initial divergence
    sim.grid.compute_divergence();
    let initial_div = sim.grid.total_divergence();
    println!("  Initial divergence: {:.1}", initial_div);

    let mut max_velocity: f32 = 0.0;
    let mut final_div: f32 = 0.0;

    // Run for 120 frames
    for frame in 0..120 {
        sim.update(DT);

        for p in sim.particles.iter() {
            max_velocity = max_velocity.max(p.velocity.length());
        }

        if frame % 30 == 0 {
            sim.grid.compute_divergence();
            let div = sim.grid.total_divergence();
            println!("  Frame {:3}: {} particles, max_v={:5.1}, div={:5.1}",
                frame, sim.particles.len(), max_velocity, div);
            final_div = div;
        }
    }

    sim.grid.compute_divergence();
    final_div = sim.grid.total_divergence();

    println!("\n  Results:");
    println!("    Divergence: {:.1} → {:.1}", initial_div, final_div);
    println!("    Max velocity: {:.1}", max_velocity);

    // Check 1: Divergence doesn't explode
    print!("  [3.1] Divergence bounded (<500): ");
    if final_div < 500.0 {
        println!("PASS ({:.1})", final_div);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", final_div);
        fail += 1;
    }

    // Check 2: Velocity bounded
    print!("  [3.2] Velocity bounded (<300): ");
    if max_velocity < 300.0 {
        println!("PASS ({:.1})", max_velocity);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", max_velocity);
        fail += 1;
    }

    (pass, fail)
}

/// Test 4: Corner Trap
/// Risk: Particles get stuck in corners with high pressure
/// Success: Particles flow out of corners, no velocity explosion
fn test_corner_trap() -> (usize, usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: CORNER TRAP");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Risk: Particles trapped in corners with pressure buildup");
    println!("  Setup: Particles in corner with gravity\n");

    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 16;
    const HEIGHT: usize = 16;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create L-shaped corner
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1); // floor
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j); // left wall
    }

    // Spawn particles in the corner
    for i in 0..5 {
        for j in 0..5 {
            let x = 8.0 + i as f32 * 3.0;
            let y = (HEIGHT as f32 - 2.0) * CELL_SIZE - j as f32 * 3.0;
            sim.spawn_water(x, y, 0.0, 0.0, 1);
        }
    }

    let initial_count = sim.particles.len();
    println!("  Initial: {} particles in corner", initial_count);

    let mut max_velocity: f32 = 0.0;
    let mut pressure_readings: Vec<f32> = Vec::new();

    // Run for 120 frames
    for frame in 0..120 {
        sim.update(DT);

        for p in sim.particles.iter() {
            max_velocity = max_velocity.max(p.velocity.length());
        }

        if frame % 30 == 0 {
            sim.grid.compute_divergence();
            let div = sim.grid.total_divergence();
            let (p_min, p_max, p_avg) = sim.grid.pressure_stats();
            pressure_readings.push(p_max);
            println!("  Frame {:3}: max_v={:5.1}, div={:5.1}, pressure=[{:.1}, {:.1}, {:.1}]",
                frame, max_velocity, div, p_min, p_max, p_avg);
        }
    }

    let max_pressure = pressure_readings.iter().cloned().fold(0.0f32, f32::max);

    println!("\n  Results:");
    println!("    Max velocity: {:.1}", max_velocity);
    println!("    Max pressure: {:.1}", max_pressure);

    // Check 1: Velocity bounded (corner shouldn't cause explosion)
    print!("  [4.1] Velocity bounded (<200): ");
    if max_velocity < 200.0 {
        println!("PASS ({:.1})", max_velocity);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", max_velocity);
        fail += 1;
    }

    // Check 2: Pressure bounded
    print!("  [4.2] Pressure bounded (<100): ");
    if max_pressure < 100.0 {
        println!("PASS ({:.1})", max_pressure);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", max_pressure);
        fail += 1;
    }

    (pass, fail)
}

/// Test 5: Long-Term Stability
/// Risk: Divergence/velocity drift over many frames
/// Success: Metrics stay bounded over extended simulation
fn test_long_term_stability() -> (usize, usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 5: LONG-TERM STABILITY (60 seconds)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Risk: Divergence or velocity drift over time");
    println!("  Setup: Continuous water flow for 3600 frames\n");

    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 48;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const FRAMES: usize = 3600; // 60 seconds at 60fps

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create sloped floor
    for i in 0..WIDTH {
        let floor_height = HEIGHT - 1 - (i / 8);
        for j in floor_height..HEIGHT {
            sim.grid.set_solid(i, j);
        }
    }
    // Right wall to collect water
    for j in 0..HEIGHT {
        sim.grid.set_solid(WIDTH - 1, j);
    }

    // Track metrics over time
    let mut div_samples: Vec<f32> = Vec::new();
    let mut vel_samples: Vec<f32> = Vec::new();
    let mut max_velocity: f32 = 0.0;
    let mut max_divergence: f32 = 0.0;
    let mut fps_tracker = FpsTracker::new();

    for frame in 0..FRAMES {
        // Spawn water continuously from left
        if frame % 10 == 0 && sim.particles.len() < 500 {
            sim.spawn_water(20.0, 30.0, 50.0, 0.0, 5);
        }

        let frame_start = Instant::now();
        sim.update(DT);
        let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        fps_tracker.record(frame_ms);

        // Sample every 300 frames (5 seconds)
        if frame % 300 == 0 {
            let mut frame_max_v: f32 = 0.0;
            for p in sim.particles.iter() {
                frame_max_v = frame_max_v.max(p.velocity.length());
            }

            sim.grid.compute_divergence();
            let div = sim.grid.total_divergence();

            div_samples.push(div);
            vel_samples.push(frame_max_v);
            max_velocity = max_velocity.max(frame_max_v);
            max_divergence = max_divergence.max(div);

            let current_fps = 1000.0 / frame_ms;
            println!("  t={:4.1}s: {:3} particles, max_v={:5.1}, div={:5.1}, fps={:5.0}",
                frame as f32 / 60.0, sim.particles.len(), frame_max_v, div, current_fps);
        }
    }

    // Calculate trend (is divergence growing?)
    let first_half_avg: f32 = div_samples[..div_samples.len()/2].iter().sum::<f32>()
        / (div_samples.len()/2) as f32;
    let second_half_avg: f32 = div_samples[div_samples.len()/2..].iter().sum::<f32>()
        / (div_samples.len()/2) as f32;
    let div_growth = second_half_avg / first_half_avg.max(0.1);

    println!("\n  Results:");
    println!("    Max velocity: {:.1}", max_velocity);
    println!("    Max divergence: {:.1}", max_divergence);
    println!("    Divergence trend: {:.2}x (1st half avg: {:.1}, 2nd half avg: {:.1})",
        div_growth, first_half_avg, second_half_avg);
    println!("    FPS: avg={:.0}, min={:.0}", fps_tracker.avg_fps(), fps_tracker.min_fps());
    println!("    FPS trend: {:.2}x (1st half: {:.0}, 2nd half: {:.0})",
        fps_tracker.degradation_ratio(), fps_tracker.first_half_avg(), fps_tracker.second_half_avg());

    // Check 1: Velocity stays bounded
    print!("  [5.1] Velocity bounded (<300): ");
    if max_velocity < 300.0 {
        println!("PASS ({:.1})", max_velocity);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", max_velocity);
        fail += 1;
    }

    // Check 2: Divergence stays bounded
    print!("  [5.2] Divergence bounded (<500): ");
    if max_divergence < 500.0 {
        println!("PASS ({:.1})", max_divergence);
        pass += 1;
    } else {
        println!("FAIL ({:.1})", max_divergence);
        fail += 1;
    }

    // Check 3: Divergence not growing unboundedly
    print!("  [5.3] No divergence drift (<3x growth): ");
    if div_growth < 3.0 {
        println!("PASS ({:.2}x)", div_growth);
        pass += 1;
    } else {
        println!("FAIL ({:.2}x)", div_growth);
        fail += 1;
    }

    // Check 4: FPS doesn't degrade significantly (stays above 50% of initial)
    let fps_degradation = fps_tracker.degradation_ratio();
    print!("  [5.4] No FPS degradation (>0.5x): ");
    if fps_degradation > 0.5 {
        println!("PASS ({:.2}x)", fps_degradation);
        pass += 1;
    } else {
        println!("FAIL ({:.2}x - performance degraded!)", fps_degradation);
        fail += 1;
    }

    // Check 5: Minimum FPS stays playable (>30 fps in release)
    let min_fps = fps_tracker.min_fps();
    print!("  [5.5] Min FPS playable (>30): ");
    if min_fps > 30.0 {
        println!("PASS ({:.0} fps)", min_fps);
        pass += 1;
    } else {
        println!("FAIL ({:.0} fps)", min_fps);
        fail += 1;
    }

    (pass, fail)
}
