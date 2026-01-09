//! Analyze grid utilization - where do particles actually go?

use sim::{create_sluice_with_mode, FlipSimulation, SluiceConfig, RiffleMode};

fn main() {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 384;
    const CELL_SIZE: f32 = 1.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    let config = SluiceConfig {
        slope: 0.25,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(&mut sim, &config);

    let dt = 1.0 / 60.0;
    let inlet_vx = 80.0;
    let inlet_vy = 5.0;
    let spawn_rate = 4;

    // Run for 30 seconds of sim time (water should reach outlet)
    let total_frames = 1800;

    for frame in 0..total_frames {
        // Spawn water at inlet
        let base_y = HEIGHT / 4;
        for i in 0..spawn_rate {
            let y = (base_y - 20 + i * 5) as f32;
            sim.spawn_water(5.0, y, inlet_vx, inlet_vy, 1);
        }

        sim.update(dt);

        // Every 5 seconds, report bounds
        if frame % 300 == 299 {
            let (min_x, max_x, min_y, max_y) = particle_bounds(&sim);
            let seconds = (frame + 1) as f32 / 60.0;

            println!("\n=== {:.1}s ({} particles) ===", seconds, sim.particles.len());
            println!("X range: {:.1} - {:.1} (using {:.1}% of {} width)",
                min_x, max_x, (max_x - min_x) / WIDTH as f32 * 100.0, WIDTH);
            println!("Y range: {:.1} - {:.1} (using {:.1}% of {} height)",
                min_y, max_y, (max_y - min_y) / HEIGHT as f32 * 100.0, HEIGHT);

            // Grid cell utilization
            let cells_with_particles = count_occupied_cells(&sim, WIDTH, HEIGHT, CELL_SIZE);
            let total_cells = WIDTH * HEIGHT;
            println!("Cells with particles: {} / {} ({:.1}%)",
                cells_with_particles, total_cells,
                cells_with_particles as f32 / total_cells as f32 * 100.0);
        }
    }

    // Final summary
    println!("\n=== FINAL SUMMARY ===");
    let (min_x, max_x, min_y, max_y) = particle_bounds(&sim);
    println!("Grid size: {} x {} = {} cells", WIDTH, HEIGHT, WIDTH * HEIGHT);
    println!("Particle X range: {:.1} - {:.1}", min_x, max_x);
    println!("Particle Y range: {:.1} - {:.1}", min_y, max_y);
    println!("Effective area needed: {:.0} x {:.0} = {:.0} cells",
        max_x - min_x, max_y - min_y, (max_x - min_x) * (max_y - min_y));

    // Suggest smaller grid
    let suggested_width = ((max_x + 20.0) as usize).next_power_of_two().max(256);
    let suggested_height = ((max_y + 20.0) as usize).next_power_of_two().max(192);
    println!("\nSuggested grid size: {} x {} ({:.1}x smaller)",
        suggested_width, suggested_height,
        (WIDTH * HEIGHT) as f32 / (suggested_width * suggested_height) as f32);
}

fn particle_bounds(sim: &FlipSimulation) -> (f32, f32, f32, f32) {
    if sim.particles.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for p in sim.particles.iter() {
        min_x = min_x.min(p.position.x);
        max_x = max_x.max(p.position.x);
        min_y = min_y.min(p.position.y);
        max_y = max_y.max(p.position.y);
    }

    (min_x, max_x, min_y, max_y)
}

fn count_occupied_cells(sim: &FlipSimulation, width: usize, height: usize, cell_size: f32) -> usize {
    let mut occupied = vec![false; width * height];

    for p in sim.particles.iter() {
        let i = (p.position.x / cell_size) as usize;
        let j = (p.position.y / cell_size) as usize;
        if i < width && j < height {
            occupied[j * width + i] = true;
        }
    }

    occupied.iter().filter(|&&x| x).count()
}
