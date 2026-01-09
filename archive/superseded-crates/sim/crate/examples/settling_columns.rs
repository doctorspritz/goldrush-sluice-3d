//! Static Water Test
//!
//! Simple test: a container half full of water.
//! The water should maintain volume and reach hydrostatic equilibrium.
//!
//! Run with: cargo run --example settling_columns --release

use macroquad::prelude::*;
use sim::flip::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial};

const CELL_SIZE: f32 = 4.0;
const DT: f32 = 1.0 / 60.0;
const WIDTH: usize = 80;
const HEIGHT: usize = 60;
const SCALE: f32 = 2.5;

fn window_conf() -> Conf {
    Conf {
        window_title: "Static Water Test".to_owned(),
        window_width: (WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Build a simple box: floor + walls
    build_box(&mut sim);

    // Fill bottom half with water
    fill_with_water(&mut sim);

    let initial_count = sim.particles.len();
    println!("Grid: {}x{}", sim.grid.width, sim.grid.height);
    println!("Initial water particles: {}", initial_count);

    // Track initial bounding box
    let (init_min_y, init_max_y) = particle_y_bounds(&sim);
    println!("Initial Y bounds: {:.1} to {:.1}", init_min_y, init_max_y);

    let mut time = 0.0f32;
    let mut frame = 0u64;
    let mut paused = false;

    loop {
        // Input
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::R) {
            sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
            build_box(&mut sim);
            fill_with_water(&mut sim);
            time = 0.0;
            frame = 0;
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // Update
        if !paused {
            sim.update(DT);
            time += DT;
            frame += 1;

            // Diagnostic every second
            if frame % 60 == 0 {
                let count = sim.particles.len();
                let (min_y, max_y) = particle_y_bounds(&sim);
                let height = max_y - min_y;

                sim.grid.compute_divergence();
                let div = sim.grid.total_divergence();

                // Pressure stats
                let (p_min, p_max, p_avg) = sim.grid.pressure_stats();

                // Count fluid cells
                let fluid_cells = sim.grid.cell_type.iter()
                    .filter(|&&t| t == sim::grid::CellType::Fluid).count();

                println!("t={:.1}s: count={}, Y=[{:.1},{:.1}], h={:.1}, div={:.1}, p=[{:.1},{:.1},{:.1}], fluid={}",
                    time, count, min_y, max_y, height, div, p_min, p_max, p_avg, fluid_cells);
            }
        }

        // Render
        clear_background(Color::from_rgba(20, 25, 40, 255));

        // Draw solid walls
        for j in 0..sim.grid.height {
            for i in 0..sim.grid.width {
                if sim.grid.is_solid(i, j) {
                    let x = i as f32 * CELL_SIZE * SCALE;
                    let y = j as f32 * CELL_SIZE * SCALE;
                    let size = CELL_SIZE * SCALE;
                    draw_rectangle(x, y, size, size, Color::from_rgba(60, 65, 75, 255));
                }
            }
        }

        // Draw particles
        let particle_size = CELL_SIZE * SCALE * 0.7;
        for p in sim.particles.iter() {
            let x = p.position.x * SCALE;
            let y = p.position.y * SCALE;
            let rgba = p.material.color();
            let color = Color::from_rgba(rgba[0], rgba[1], rgba[2], 200);
            draw_circle(x, y, particle_size * 0.4, color);
        }

        // Draw initial water level line
        let init_surface_y = init_min_y * SCALE;
        draw_line(0.0, init_surface_y, screen_width(), init_surface_y, 2.0, Color::from_rgba(255, 100, 100, 150));

        // Draw current bounds
        let (min_y, max_y) = particle_y_bounds(&sim);
        let curr_surface_y = min_y * SCALE;
        draw_line(0.0, curr_surface_y, screen_width(), curr_surface_y, 2.0, Color::from_rgba(100, 255, 100, 150));

        // Draw info
        draw_text(&format!("Time: {:.1}s  Particles: {}", time, sim.particles.len()), 10.0, 25.0, 20.0, WHITE);
        draw_text("[SPACE] Pause  [R] Reset  [ESC] Quit", 10.0, 50.0, 16.0, GRAY);
        draw_text("Red line = initial surface, Green = current", 10.0, 75.0, 16.0, LIGHTGRAY);

        next_frame().await;
    }
}

fn build_box(sim: &mut FlipSimulation) {
    let w = sim.grid.width;
    let h = sim.grid.height;

    // Floor
    for i in 0..w {
        sim.grid.set_solid(i, h - 1);
        sim.grid.set_solid(i, h - 2);
    }

    // Left wall
    for j in 0..h {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(1, j);
    }

    // Right wall
    for j in 0..h {
        sim.grid.set_solid(w - 1, j);
        sim.grid.set_solid(w - 2, j);
    }

    sim.grid.compute_sdf();
    println!("Box built: {}x{}, walls=2 cells thick", w, h);
}

fn fill_with_water(sim: &mut FlipSimulation) {
    let h = sim.grid.height;

    // Fill bottom 50% with water
    let water_top = h / 2;
    let water_bottom = h - 2; // Above floor

    println!("Water fill: y = {} to {} (rows)", water_top, water_bottom - 1);

    for j in water_top..water_bottom {
        for i in 2..(sim.grid.width - 2) {
            if !sim.grid.is_solid(i, j) {
                let x = (i as f32 + 0.5) * CELL_SIZE;
                let y = (j as f32 + 0.5) * CELL_SIZE;
                sim.particles.list.push(Particle::water(
                    glam::Vec2::new(x, y),
                    glam::Vec2::ZERO,
                ));
            }
        }
    }
}

fn particle_y_bounds(sim: &FlipSimulation) -> (f32, f32) {
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for p in sim.particles.iter() {
        min_y = min_y.min(p.position.y);
        max_y = max_y.max(p.position.y);
    }

    if min_y == f32::MAX {
        (0.0, 0.0)
    } else {
        (min_y, max_y)
    }
}
