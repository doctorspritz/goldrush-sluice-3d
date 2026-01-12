//! Visual Collapse Demo (Terminal)
//!
//! Shows material piles collapsing to their angle of repose in ASCII art.
//! Run: cargo run -p sim3d --example collapse_visual

use sim3d::{World, TerrainMaterial};
use std::{thread, time::Duration};

const WIDTH: usize = 41;
const DEPTH: usize = 41;
const CELL_SIZE: f32 = 0.1;
const CENTER: usize = 20;

fn main() {
    println!("=== ANGLE OF REPOSE COLLAPSE DEMO ===\n");
    println!("Material angles:");
    println!("  Sand:   {:.1}°", TerrainMaterial::Sand.angle_of_repose().to_degrees());
    println!("  Dirt:   {:.1}°", TerrainMaterial::Dirt.angle_of_repose().to_degrees());
    println!("  Gravel: {:.1}°\n", TerrainMaterial::Gravel.angle_of_repose().to_degrees());

    // Create flat world
    let mut world = World::new(WIDTH, DEPTH, CELL_SIZE, 0.0);

    // Set flat bedrock base, clear other layers
    let cell_count = WIDTH * DEPTH;
    world.bedrock_elevation = vec![0.5; cell_count];
    world.paydirt_thickness = vec![0.0; cell_count];
    world.gravel_thickness = vec![0.0; cell_count];
    world.overburden_thickness = vec![0.0; cell_count];
    world.terrain_sediment = vec![0.0; cell_count];

    // Use slower transfer for accurate convergence
    world.params.collapse_transfer_rate = 0.15;
    world.params.collapse_max_outflow = 0.3;

    // Add a tall pile of sediment at center
    let pile_height = 0.4;
    let idx = CENTER * WIDTH + CENTER;
    world.terrain_sediment[idx] = pile_height;

    println!("Initial pile: {:.2}m tall at center", pile_height);
    println!("Expected final angle: {:.1}° (sand)\n", TerrainMaterial::Sand.angle_of_repose().to_degrees());

    print_cross_section(&world, CENTER);

    // Run collapse with animation
    let mut iteration = 0;
    let max_iterations = 100;

    loop {
        // Debug: Check slope at center before collapse
        let center_idx = CENTER * WIDTH + CENTER;
        let neighbor_idx = CENTER * WIDTH + (CENTER + 1);
        let center_h = 0.5 + world.terrain_sediment[center_idx];
        let neighbor_h = 0.5 + world.terrain_sediment[neighbor_idx];
        let diff = center_h - neighbor_h;
        let max_diff = TerrainMaterial::Sand.angle_of_repose().tan() * CELL_SIZE;

        if iteration % 5 == 0 {
            println!("  Debug: center_h={:.4}, neighbor_h={:.4}", center_h, neighbor_h);
            println!("  Debug: diff={:.4}, max_diff={:.4} ({})", diff, max_diff, if diff > max_diff { "SHOULD COLLAPSE" } else { "stable" });
        }

        let changed = world.update_terrain_collapse();
        iteration += 1;

        if iteration % 5 == 0 || !changed {
            println!("\n--- Iteration {} ---", iteration);
            print_cross_section(&world, CENTER);

            let (max_h, spread, slope_angle) = measure_pile(&world, CENTER);
            println!("Peak height: {:.3}m, Spread: {} cells", max_h, spread);
            println!("Slope angle (cell-to-cell): {:.1}° (target: {:.1}°)",
                     slope_angle, TerrainMaterial::Sand.angle_of_repose().to_degrees());
        }

        if !changed {
            println!("\n=== STABLE after {} iterations ===", iteration);
            println!("  Final diff: {:.4}, max_diff: {:.4}", diff, max_diff);
            break;
        }

        if iteration >= max_iterations {
            println!("\n=== Max iterations reached ===");
            break;
        }

        thread::sleep(Duration::from_millis(50));
    }

    // Final analysis
    println!("\n=== FINAL STATE ===");
    print_cross_section(&world, CENTER);

    let (max_h, spread, slope_angle) = measure_pile(&world, CENTER);
    let expected = TerrainMaterial::Sand.angle_of_repose().to_degrees();

    println!("\n=== RESULTS ===");
    println!("Final slope angle: {:.1}°", slope_angle);
    println!("Expected angle:    {:.1}°", expected);
    println!("Difference:        {:.1}°", (slope_angle - expected).abs());

    if (slope_angle - expected).abs() < 2.0 {
        println!("\n✓ SUCCESS: Collapse converged to correct angle of repose!");
    } else {
        println!("\n✗ FAILED: Angle not within tolerance");
    }

    // Print total mass check
    let total_sediment: f32 = world.terrain_sediment.iter().sum();
    println!("\nMass conservation: {:.4} / {:.4} ({:.1}%)",
             total_sediment, pile_height, 100.0 * total_sediment / pile_height);
}

fn print_cross_section(world: &World, z: usize) {
    // Print a cross-section at z through the center
    let base = 0.5; // bedrock height
    let max_display_height = 1.0;
    let rows = 12;

    print!("     ");
    for x in 10..31 {
        if x == CENTER {
            print!("v");
        } else {
            print!(" ");
        }
    }
    println!();

    for row in (0..rows).rev() {
        let h = base + (row as f32 / rows as f32) * (max_display_height - base);
        print!("{:4.2} |", h);

        for x in 10..31 {
            let idx = z * WIDTH + x;
            let ground = base + world.terrain_sediment[idx];

            if ground >= h {
                print!("#");
            } else if ground >= h - 0.04 {
                print!(".");
            } else {
                print!(" ");
            }
        }
        println!("|");
    }

    print!("     +");
    for _ in 10..31 {
        print!("-");
    }
    println!("+");
}

fn measure_pile(world: &World, center: usize) -> (f32, usize, f32) {
    let idx = center * WIDTH + center;
    let max_h = 0.5 + world.terrain_sediment[idx];

    // Find spread (how far sediment extends from center)
    let mut spread = 0;
    for dx in 1..=CENTER {
        let check_idx = center * WIDTH + (center + dx);
        if world.terrain_sediment[check_idx] > 0.001 {
            spread = dx;
        } else {
            break;
        }
    }

    // Measure actual slope angle between center and first neighbor
    let neighbor_idx = center * WIDTH + (center + 1);
    let neighbor_h = 0.5 + world.terrain_sediment[neighbor_idx];
    let slope_angle = ((max_h - neighbor_h) / CELL_SIZE).atan().to_degrees();

    (max_h, spread, slope_angle)
}
