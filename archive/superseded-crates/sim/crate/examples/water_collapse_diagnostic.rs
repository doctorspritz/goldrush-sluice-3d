//! Water Collapse Diagnostic
//!
//! Track what happens to water particles frame by frame.
//! Run with: cargo run --example water_collapse_diagnostic

use sim::flip::FlipSimulation;
use sim::grid::CellType;
use sim::particle::ParticleMaterial;

// Match flip_dem_coupling_test dimensions
const WIDTH: usize = 180;
const HEIGHT: usize = 70;
const CELL_SIZE: f32 = 5.0;

fn main() {
    let mut flip = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Match flip_dem_coupling_test geometry: sluice with riffles
    // CRITICAL: Use set_solid() to set BOTH solid[] and cell_type[]
    // Otherwise classify_cells() will lose terrain on frame 2!
    let floor_row = HEIGHT - 8;
    for i in 0..WIDTH {
        for j in 0..HEIGHT {
            // Floor
            if j >= floor_row {
                flip.grid.set_solid(i, j);
            }
            // Left wall
            if i < 2 && j > 5 {
                flip.grid.set_solid(i, j);
            }
            // Right wall
            if i >= WIDTH - 2 && j > 5 {
                flip.grid.set_solid(i, j);
            }
        }
    }

    // Add riffles
    let riffle_positions = [WIDTH / 4, WIDTH / 2, 3 * WIDTH / 4];
    for &rx in &riffle_positions {
        for di in 0..3 {
            let i = rx + di;
            if i < WIDTH {
                for dj in 1..=4 {
                    let j = floor_row - dj;
                    if j < HEIGHT {
                        flip.grid.set_solid(i, j);
                    }
                }
            }
        }
    }
    flip.grid.compute_sdf();

    // Match flip_dem_coupling_test water initialization
    let particles_per_cell = 4;
    let spacing = CELL_SIZE / 2.0;
    let water_top = floor_row - 12;
    let water_bottom = floor_row - 1;

    for j in water_top..water_bottom {
        for i in 3..(WIDTH - 3) {
            let idx = j * WIDTH + i;
            if flip.grid.cell_type[idx] != CellType::Solid {
                let base_x = i as f32 * CELL_SIZE;
                let base_y = j as f32 * CELL_SIZE;

                for pi in 0..2 {
                    for pj in 0..2 {
                        let px = base_x + (pi as f32 + 0.25) * spacing;
                        let py = base_y + (pj as f32 + 0.25) * spacing;
                        // Initial rightward flow matching the example
                        flip.particles.spawn_water(px, py, 60.0, 0.0);
                    }
                }
            }
        }
    }

    let initial_count = flip.particles.len();
    println!("Initial water particles: {}", initial_count);
    println!("Grid size: {}x{}", WIDTH, HEIGHT);
    println!();

    let dt = 1.0 / 60.0;

    for frame in 0..120 {
        // Diagnostics before update
        let water_count_before: usize = flip.particles.iter()
            .filter(|p| p.material == ParticleMaterial::Water)
            .count();

        // Track velocity stats before
        let (avg_vel_before, max_vel_before) = velocity_stats(&flip);

        // Track cell classification
        let fluid_cells: usize = flip.grid.cell_type.iter()
            .filter(|&&c| c == CellType::Fluid)
            .count();

        flip.update(dt);

        // Diagnostics after update
        let water_count_after: usize = flip.particles.iter()
            .filter(|p| p.material == ParticleMaterial::Water)
            .count();

        let (avg_vel_after, max_vel_after) = velocity_stats(&flip);

        // Track particle positions - how many are in valid bounds?
        let in_bounds: usize = flip.particles.iter()
            .filter(|p| {
                p.position.x > 0.0 &&
                p.position.x < WIDTH as f32 * CELL_SIZE &&
                p.position.y > 0.0 &&
                p.position.y < HEIGHT as f32 * CELL_SIZE
            })
            .count();

        // Track divergence
        flip.grid.compute_divergence();
        let total_div = flip.grid.total_divergence();
        let max_div = flip.grid.divergence.iter().cloned().fold(0.0f32, f32::max);

        let lost = water_count_before as i32 - water_count_after as i32;

        if frame < 20 || frame % 10 == 0 || lost > 0 {
            println!(
                "Frame {:3}: water={:5} (lost={:3}) fluid_cells={:4} div_total={:8.1} div_max={:6.1} avg_vel={:6.1} max_vel={:7.1}",
                frame, water_count_after, lost, fluid_cells, total_div, max_div, avg_vel_after, max_vel_after
            );
        }

        // Stop if water collapsed
        if water_count_after < initial_count / 2 {
            println!("\n!!! Water collapsed to < 50% at frame {} !!!", frame);

            // Print detailed position analysis
            let mut below_floor = 0;
            let mut left_of_domain = 0;
            let mut right_of_domain = 0;
            let mut above_domain = 0;

            for p in flip.particles.iter() {
                if p.position.y > floor_row as f32 * CELL_SIZE {
                    below_floor += 1;
                }
                if p.position.x < 0.0 {
                    left_of_domain += 1;
                }
                if p.position.x > WIDTH as f32 * CELL_SIZE {
                    right_of_domain += 1;
                }
                if p.position.y < 0.0 {
                    above_domain += 1;
                }
            }

            println!("Position breakdown:");
            println!("  Below floor: {}", below_floor);
            println!("  Left of domain: {}", left_of_domain);
            println!("  Right of domain: {}", right_of_domain);
            println!("  Above domain: {}", above_domain);

            break;
        }
    }

    println!("\nFinal water count: {} / {} ({:.1}%)",
        flip.particles.len(), initial_count,
        flip.particles.len() as f32 / initial_count as f32 * 100.0);
}

fn velocity_stats(flip: &FlipSimulation) -> (f32, f32) {
    let velocities: Vec<f32> = flip.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity.length())
        .collect();

    if velocities.is_empty() {
        return (0.0, 0.0);
    }

    let avg = velocities.iter().sum::<f32>() / velocities.len() as f32;
    let max = velocities.iter().cloned().fold(0.0f32, f32::max);
    (avg, max)
}
