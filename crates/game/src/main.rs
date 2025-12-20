//! Goldrush Fluid Miner - Game
//!
//! PIC/FLIP fluid simulation demo with sluice box vortex formation.
//! Uses GPU-accelerated instanced circle rendering for fluid particles.

mod render;

use macroquad::prelude::*;
use render::ParticleRenderer;
use sim::{create_sluice, FlipSimulation};

// Simulation size - larger for better sluice dynamics
const SIM_WIDTH: usize = 256;
const SIM_HEIGHT: usize = 192;
const CELL_SIZE: f32 = 2.0;
const SCALE: f32 = 2.5; // Slightly smaller scale to fit on screen

// Render buffer size (simulation space, not screen space)
const RENDER_WIDTH: usize = (SIM_WIDTH as f32 * CELL_SIZE) as usize;
const RENDER_HEIGHT: usize = (SIM_HEIGHT as f32 * CELL_SIZE) as usize;

fn window_conf() -> Conf {
    Conf {
        window_title: "Goldrush Fluid Miner - FLIP Sluice".to_owned(),
        window_width: (SIM_WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (SIM_HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Create simulation
    let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    // Create GPU particle renderer
    let renderer = ParticleRenderer::new();

    // Create terrain texture (static, never changes)
    let terrain_texture = {
        let bg_color = Color::from_rgba(20, 20, 30, 255);
        let terrain_color = Color::from_rgba(80, 80, 80, 255);

    // Set up sloped sluice with riffles
    // slope=0.25 means floor rises 0.25 cells per horizontal cell (about 14 degrees)
    // riffle_spacing=30, riffle_height=8, riffle_width=3
    create_sluice(&mut sim, 0.25, 30, 8, 3);

        let mut img = Image::gen_image_color(RENDER_WIDTH as u16, RENDER_HEIGHT as u16, bg_color);
        for j in 0..sim.grid.height {
            for i in 0..sim.grid.width {
                if sim.grid.is_solid(i, j) {
                    let start_x = (i as f32 * CELL_SIZE) as usize;
                    let start_y = (j as f32 * CELL_SIZE) as usize;
                    let end_x = ((i + 1) as f32 * CELL_SIZE) as usize;
                    let end_y = ((j + 1) as f32 * CELL_SIZE) as usize;

                    for py in start_y..end_y.min(RENDER_HEIGHT) {
                        for px in start_x..end_x.min(RENDER_WIDTH) {
                            img.set_pixel(px as u32, py as u32, terrain_color);
                        }
                    }
                }
            }
        }
        let tex = Texture2D::from_image(&img);
        tex.set_filter(FilterMode::Nearest);
        tex
    };

    let mut paused = false;
    let mut show_velocity = false;
    let mut spawn_mud = false;
    let mut frame_count = 0u64;

    loop {
        // --- INPUT ---
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::V) {
            show_velocity = !show_velocity;
        }
        if is_key_pressed(KeyCode::M) {
            spawn_mud = !spawn_mud;
        }
        if is_key_pressed(KeyCode::R) {
            // Reset simulation
            sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
            create_sluice(&mut sim, 0.25, 30, 8, 3);
        }
        if is_key_pressed(KeyCode::C) {
            // Clear particles
            sim.particles.list.clear();
        }

        // Mouse spawning
        if is_mouse_button_down(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let wx = mx / SCALE;
            let wy = my / SCALE;

            if spawn_mud {
                sim.spawn_mud(wx, wy, 20.0, 0.0, 3);
            } else {
                sim.spawn_water(wx, wy, 20.0, 0.0, 5);
            }
        }

        // --- UPDATE ---
        if !paused {
            // Spawn particles at inlet (left side, above the sluice floor)
            if frame_count % 3 == 0 {
                let inlet_x = 30.0;
                let inlet_y = (SIM_HEIGHT as f32 * CELL_SIZE) * 0.15; // Higher up (15% from top)

                // Spawn water - reduced to prevent washout
                sim.spawn_water(inlet_x, inlet_y, 60.0, 0.0, 2);

                // Occasionally add mud
                if frame_count % 20 == 0 {
                    sim.spawn_mud(inlet_x, inlet_y + 5.0, 60.0, 0.0, 1);
                }

                // Spawn sediment particles (integrated into FLIP simulation)
                // Sand is most common
                if frame_count % 6 == 0 {
                    sim.spawn_sand(inlet_x, inlet_y, 60.0, 0.0, 1);
                }
                // Magnetite less common (black sand indicator)
                if frame_count % 15 == 0 {
                    sim.spawn_magnetite(inlet_x, inlet_y, 60.0, 0.0, 1);
                }
                // Gold is rare!
                if frame_count % 50 == 0 {
                    sim.spawn_gold(inlet_x, inlet_y, 60.0, 0.0, 1);
                }
            }

            // Remove particles that reached the end (right side outflow)
            let outflow_x = (SIM_WIDTH as f32 - 5.0) * CELL_SIZE;
            sim.particles.list.retain(|p| p.position.x < outflow_x);

            // Run simulation step
            let dt = 1.0 / 60.0;
            sim.update(dt);

            frame_count += 1;
        }

        // --- RENDER ---

        // Draw terrain background
        draw_texture_ex(
            &terrain_texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(
                    RENDER_WIDTH as f32 * SCALE,
                    RENDER_HEIGHT as f32 * SCALE,
                )),
                ..Default::default()
            },
        );

        // Draw particles as smooth GPU-accelerated circles
        renderer.draw_sorted(&sim.particles, SCALE);

        // Draw velocity field on top (optional debug visualization)
        if show_velocity {
            let spacing = 4;
            for j in (0..sim.grid.height).step_by(spacing) {
                for i in (0..sim.grid.width).step_by(spacing) {
                    let x = (i as f32 + 0.5) * CELL_SIZE * SCALE;
                    let y = (j as f32 + 0.5) * CELL_SIZE * SCALE;

                    let pos = glam::Vec2::new(
                        (i as f32 + 0.5) * CELL_SIZE,
                        (j as f32 + 0.5) * CELL_SIZE,
                    );
                    let vel = sim.grid.sample_velocity(pos);

                    let vscale = 0.5;
                    let vx = vel.x * vscale * SCALE;
                    let vy = vel.y * vscale * SCALE;

                    draw_line(x, y, x + vx, y + vy, 1.0, Color::from_rgba(255, 255, 255, 100));
                }
            }
        }

        // --- UI ---
        let (water, mud, sand, magnetite, gold) = sim.particles.count_by_material();
        draw_text(
            &format!(
                "Particles: {} | FPS: {} | {}",
                sim.particles.len(),
                get_fps(),
                if paused { "PAUSED" } else { "Running" }
            ),
            10.0,
            25.0,
            20.0,
            WHITE,
        );

        draw_text(
            &format!(
                "Water: {} Mud: {} Sand: {} Mag: {} Gold: {}",
                water, mud, sand, magnetite, gold
            ),
            10.0,
            45.0,
            16.0,
            Color::from_rgba(255, 215, 0, 255), // Gold color
        );

        draw_text(
            &format!(
                "Spawning: {} | [Space]=Pause [V]=Velocity [M]=Toggle Mud [R]=Reset [C]=Clear",
                if spawn_mud { "Mud" } else { "Water" }
            ),
            10.0,
            65.0,
            16.0,
            GRAY,
        );

        draw_text(
            "Watch gold (yellow) settle faster than sand (tan) through vortices!",
            10.0,
            screen_height() - 10.0,
            16.0,
            GRAY,
        );

        next_frame().await
    }
}
