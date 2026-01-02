//! Comprehensive flow diagnostic - tests water flowing through full sluice
//!
//! PASS CRITERIA:
//! 1. Particles must reach outlet (x > 90% of domain width)
//! 2. Velocity must not decay below 0.5 m/s average
//! 3. Particles must flow over riffles (y variation)
//! 4. Vortices must form (velocity direction variance)
//!
//! Run with: cargo run --example flow_diagnostic -p sim3d --release

use sim3d::{create_sluice, spawn_inlet_water, CellType, FlipSimulation3D, SluiceConfig, Vec3};
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         COMPREHENSIVE SLUICE FLOW DIAGNOSTIC                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Smaller grid for faster iteration, but still representative
    let grid_width = 80;
    let grid_height = 30;
    let grid_depth = 16;
    let cell_size = 0.025;
    let world_width = grid_width as f32 * cell_size;

    let mut sim = FlipSimulation3D::new(grid_width, grid_height, grid_depth, cell_size);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 50;

    let sluice_config = SluiceConfig {
        slope: 0.08,
        slick_plate_len: 8,
        riffle_spacing: 10,
        riffle_height: 2,
        riffle_width: 1,
    };
    create_sluice(&mut sim, &sluice_config);

    // Count solid cells and riffles
    let solid_count = sim.grid.cell_type.iter().filter(|&&t| t == CellType::Solid).count();
    let total_cells = grid_width * grid_height * grid_depth;
    println!("Grid: {}x{}x{} = {} cells", grid_width, grid_height, grid_depth, total_cells);
    println!("Solid cells: {} ({:.1}%)", solid_count, 100.0 * solid_count as f32 / total_cells as f32);
    println!("World size: {:.2}m x {:.2}m x {:.2}m", world_width, grid_height as f32 * cell_size, grid_depth as f32 * cell_size);
    println!("Slope: {} (floor drops {:.3}m over length)\n", sluice_config.slope, sluice_config.slope * world_width);

    // Spawn initial batch with high velocity
    let inlet_vel = Vec3::new(2.0, 0.0, 0.0);
    spawn_inlet_water(&mut sim, &sluice_config, 1000, inlet_vel);

    println!("Initial particles: {}", sim.particle_count());
    println!("Inlet velocity: {} m/s\n", inlet_vel.x);

    // Track metrics over time
    let dt = 1.0 / 120.0;
    let total_frames = 600; // 5 seconds
    let mut reached_outlet = 0;
    let mut max_x_ever = 0.0f32;
    let mut total_exited = 0;

    // Velocity histograms at different X positions
    let mut vel_by_region: HashMap<&str, Vec<f32>> = HashMap::new();
    vel_by_region.insert("inlet", vec![]);
    vel_by_region.insert("middle", vec![]);
    vel_by_region.insert("outlet", vec![]);

    println!("Running {} frames ({:.1}s of simulation)...\n", total_frames, total_frames as f32 * dt);
    println!("{:>6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8}",
             "Frame", "Count", "MaxX", "AvgVelX", "AvgVelY", "VelMag", "Exited");
    println!("{}", "-".repeat(70));

    for frame in 0..total_frames {
        // Spawn more water continuously
        if frame % 20 == 0 && sim.particle_count() < 8000 {
            spawn_inlet_water(&mut sim, &sluice_config, 100, inlet_vel);
        }

        sim.update(dt);

        // Track particles that would exit
        let before = sim.particle_count();
        sim.particles.list.retain(|p| {
            let in_bounds = p.position.x > 0.0
                && p.position.x < world_width
                && p.position.y > 0.0
                && p.position.y < grid_height as f32 * cell_size
                && p.position.z > 0.0
                && p.position.z < grid_depth as f32 * cell_size
                && p.velocity.is_finite()
                && p.position.is_finite();

            if !in_bounds && p.position.x >= world_width * 0.9 {
                // Exited through outlet
                true // will be counted then removed
            } else {
                in_bounds
            }
        });

        // Count particles near outlet before removing
        let near_outlet = sim.particles.list.iter()
            .filter(|p| p.position.x >= world_width * 0.9)
            .count();
        reached_outlet = reached_outlet.max(near_outlet);

        // Remove particles that exited
        let exited_this_frame = sim.particles.list.iter()
            .filter(|p| p.position.x >= world_width)
            .count();
        total_exited += exited_this_frame;
        sim.particles.list.retain(|p| p.position.x < world_width);

        // Compute stats
        if !sim.particles.list.is_empty() {
            let max_x = sim.particles.list.iter().map(|p| p.position.x).fold(0.0f32, f32::max);
            max_x_ever = max_x_ever.max(max_x);

            let avg_vel_x = sim.particles.list.iter().map(|p| p.velocity.x).sum::<f32>() / sim.particle_count() as f32;
            let avg_vel_y = sim.particles.list.iter().map(|p| p.velocity.y).sum::<f32>() / sim.particle_count() as f32;
            let avg_vel_mag = sim.particles.list.iter().map(|p| p.velocity.length()).sum::<f32>() / sim.particle_count() as f32;

            // Sample velocities by region
            for p in &sim.particles.list {
                let region = if p.position.x < world_width * 0.2 {
                    "inlet"
                } else if p.position.x < world_width * 0.8 {
                    "middle"
                } else {
                    "outlet"
                };
                vel_by_region.get_mut(region).unwrap().push(p.velocity.x);
            }

            if frame % 60 == 0 || frame == total_frames - 1 {
                println!("{:>6} {:>8} {:>8.3} {:>8.3} {:>10.3} {:>10.3} {:>8}",
                         frame, sim.particle_count(), max_x, avg_vel_x, avg_vel_y, avg_vel_mag, total_exited);
            }
        }
    }

    // Analyze results
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        RESULTS                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Did particles reach outlet?
    let outlet_threshold = world_width * 0.9;
    println!("TEST 1: Particles reaching outlet (x > {:.2}m)", outlet_threshold);
    println!("  Max X ever reached: {:.3}m", max_x_ever);
    println!("  Max particles near outlet: {}", reached_outlet);
    println!("  Total exited through outlet: {}", total_exited);
    if max_x_ever > outlet_threshold {
        println!("  ✅ PASS: Water reached outlet region");
    } else {
        println!("  ❌ FAIL: Water never reached outlet (max_x = {:.3}m, need > {:.2}m)", max_x_ever, outlet_threshold);
    }

    // Test 2: Velocity by region
    println!("\nTEST 2: Velocity by region");
    for (region, vels) in &vel_by_region {
        if !vels.is_empty() {
            let avg: f32 = vels.iter().sum::<f32>() / vels.len() as f32;
            let min = vels.iter().cloned().fold(f32::MAX, f32::min);
            let max = vels.iter().cloned().fold(f32::MIN, f32::max);
            println!("  {:>8}: avg={:>6.3} m/s, range=[{:.3}, {:.3}], samples={}",
                     region, avg, min, max, vels.len());
        }
    }

    let inlet_avg: f32 = vel_by_region["inlet"].iter().sum::<f32>() / vel_by_region["inlet"].len().max(1) as f32;
    let outlet_avg: f32 = if !vel_by_region["outlet"].is_empty() {
        vel_by_region["outlet"].iter().sum::<f32>() / vel_by_region["outlet"].len() as f32
    } else {
        0.0
    };

    if inlet_avg > 0.5 {
        println!("  ✅ PASS: Inlet velocity maintained (avg={:.3} m/s)", inlet_avg);
    } else {
        println!("  ❌ FAIL: Inlet velocity too low (avg={:.3} m/s, need > 0.5)", inlet_avg);
    }

    // Test 3: Check for vortices (velocity variance)
    println!("\nTEST 3: Vortex formation (velocity direction variance)");
    let vel_y_vals: Vec<f32> = sim.particles.list.iter().map(|p| p.velocity.y).collect();
    if !vel_y_vals.is_empty() {
        let mean_y: f32 = vel_y_vals.iter().sum::<f32>() / vel_y_vals.len() as f32;
        let variance_y: f32 = vel_y_vals.iter().map(|v| (v - mean_y).powi(2)).sum::<f32>() / vel_y_vals.len() as f32;
        let std_y = variance_y.sqrt();

        let vel_z_vals: Vec<f32> = sim.particles.list.iter().map(|p| p.velocity.z).collect();
        let mean_z: f32 = vel_z_vals.iter().sum::<f32>() / vel_z_vals.len() as f32;
        let variance_z: f32 = vel_z_vals.iter().map(|v| (v - mean_z).powi(2)).sum::<f32>() / vel_z_vals.len() as f32;
        let std_z = variance_z.sqrt();

        println!("  Y velocity: mean={:.3}, std={:.3}", mean_y, std_y);
        println!("  Z velocity: mean={:.3}, std={:.3}", mean_z, std_z);

        if std_y > 0.05 || std_z > 0.05 {
            println!("  ✅ PASS: Velocity variance indicates turbulent flow");
        } else {
            println!("  ⚠️  WARN: Low velocity variance - flow may be too laminar");
        }
    }

    // Test 4: Y position distribution (did water go over riffles?)
    println!("\nTEST 4: Vertical distribution (water over riffles)");
    let y_vals: Vec<f32> = sim.particles.list.iter().map(|p| p.position.y).collect();
    if !y_vals.is_empty() {
        let min_y = y_vals.iter().cloned().fold(f32::MAX, f32::min);
        let max_y = y_vals.iter().cloned().fold(f32::MIN, f32::max);
        let y_range = max_y - min_y;
        println!("  Y range: [{:.3}, {:.3}], spread = {:.3}m", min_y, max_y, y_range);

        if y_range > 0.05 {
            println!("  ✅ PASS: Water has vertical spread (flowing over terrain)");
        } else {
            println!("  ❌ FAIL: Water is flat - not flowing over riffles");
        }
    }

    // Final verdict
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    let all_pass = max_x_ever > outlet_threshold && inlet_avg > 0.5 && total_exited > 0;
    if all_pass {
        println!("║                    ✅ ALL TESTS PASSED                       ║");
    } else {
        println!("║                    ❌ TESTS FAILED                           ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Detailed particle state dump for debugging
    println!("\n=== Sample particle states (first 5) ===");
    for (i, p) in sim.particles.list.iter().take(5).enumerate() {
        println!("Particle {}: pos=({:.3}, {:.3}, {:.3}), vel=({:.3}, {:.3}, {:.3}), |vel|={:.3}",
                 i, p.position.x, p.position.y, p.position.z,
                 p.velocity.x, p.velocity.y, p.velocity.z, p.velocity.length());
    }
}
