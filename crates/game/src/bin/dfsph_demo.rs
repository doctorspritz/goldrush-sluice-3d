//! DFSPH Fluid Simulation Demo
//!
//! Simple demo showing the new DFSPH solver with a sloped floor.

use macroquad::prelude::*;
use dfsph::{DfsphSimulation, ParticleMaterial};

const SIM_WIDTH: usize = 200;
const SIM_HEIGHT: usize = 150;
const CELL_SIZE: f32 = 2.0;
const SCALE: f32 = 5.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "DFSPH Fluid Demo".to_owned(),
        window_width: (SIM_WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (SIM_HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut sim = DfsphSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    // Setup sloped floor with riffles
    let grid_w = (sim.width / sim.cell_size).ceil() as usize;
    let grid_h = (sim.height / sim.cell_size).ceil() as usize;

    sim.grid_solid.fill(false);

    // Create sloped floor
    for i in 0..grid_w {
        // Slope from y=40 at left to y=90 at right
        let floor_y = (40.0 + (i as f32 * 0.25)) as usize;

        for j in floor_y..grid_h {
            let idx = j * grid_w + i;
            if idx < sim.grid_solid.len() {
                sim.grid_solid[idx] = true;
            }
        }
    }

    // Add riffles (small bumps)
    for riffle in 0..5 {
        let riffle_x = 30 + riffle * 30;
        for dx in 0..4 {
            let x = riffle_x + dx;
            if x < grid_w {
                let base_y = (40.0 + (x as f32 * 0.25)) as usize;
                for dy in 0..6 {
                    let y = base_y.saturating_sub(dy + 1);
                    let idx = y * grid_w + x;
                    if idx < sim.grid_solid.len() {
                        sim.grid_solid[idx] = true;
                    }
                }
            }
        }
    }

    // Side walls
    for j in 0..grid_h {
        sim.grid_solid[j * grid_w] = true;
        sim.grid_solid[j * grid_w + grid_w - 1] = true;
    }

    let mut paused = false;
    let mut frame_count = 0u32;

    loop {
        // Controls
        if is_key_pressed(KeyCode::Space) { paused = !paused; }
        if is_key_pressed(KeyCode::R) {
            sim.particles.list.clear();
            sim.old_positions.clear();
            sim.densities.clear();
            sim.lambdas.clear();
        }
        if is_key_pressed(KeyCode::Escape) { break; }

        // Spawn water from top-left
        if !paused && frame_count % 2 == 0 {
            sim.spawn_particles(25.0, 25.0, 40.0, 5.0, 3, ParticleMaterial::Water, 3.0);
        }

        // Update physics
        if !paused {
            sim.update(1.0 / 60.0);
        }

        // Render
        clear_background(Color::from_rgba(20, 20, 35, 255));

        // Draw solids
        for (i, &solid) in sim.grid_solid.iter().enumerate() {
            if solid {
                let x = (i % grid_w) as f32 * CELL_SIZE;
                let y = (i / grid_w) as f32 * CELL_SIZE;
                draw_rectangle(
                    x * SCALE, y * SCALE,
                    CELL_SIZE * SCALE, CELL_SIZE * SCALE,
                    Color::from_rgba(80, 70, 60, 255)
                );
            }
        }

        // Draw particles
        for p in &sim.particles.list {
            let color = match p.material {
                ParticleMaterial::Water => Color::from_rgba(100, 150, 255, 220),
                ParticleMaterial::Sand => Color::from_rgba(194, 178, 128, 255),
                ParticleMaterial::Mud => Color::from_rgba(139, 90, 43, 255),
                ParticleMaterial::Gold => Color::from_rgba(255, 215, 0, 255),
                ParticleMaterial::Magnetite => Color::from_rgba(50, 50, 50, 255),
            };
            draw_circle(
                p.position.x * SCALE,
                p.position.y * SCALE,
                CELL_SIZE * SCALE * 0.4,
                color
            );
        }

        // UI
        draw_text(&format!("FPS: {}", get_fps()), 10.0, 25.0, 24.0, WHITE);
        draw_text(&format!("Particles: {}", sim.particles.len()), 10.0, 50.0, 24.0, WHITE);
        draw_text("[SPACE] Pause  [R] Reset  [ESC] Quit", 10.0, 75.0, 20.0, GRAY);
        if paused {
            draw_text("PAUSED", screen_width() / 2.0 - 50.0, screen_height() / 2.0, 40.0, YELLOW);
        }

        frame_count = frame_count.wrapping_add(1);
        next_frame().await
    }
}
