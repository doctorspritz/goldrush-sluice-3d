//! Goldrush Fluid Miner - Game
//!
//! PIC/FLIP fluid simulation demo with sluice box vortex formation.
//! Uses batched pixel rendering for high performance.

use macroquad::prelude::*;
use sim::{create_sluice, FlipSimulation, Sediment, SedimentType};

// Simulation size
const SIM_WIDTH: usize = 128;
const SIM_HEIGHT: usize = 128;
const CELL_SIZE: f32 = 2.0;
const SCALE: f32 = 3.0;

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

    // Create sediment system (drift-flux particles coupled to FLIP velocity)
    let mut sediment = Sediment::new();

    // Create render buffer (pixels in simulation space)
    let mut render_buffer = Image::gen_image_color(RENDER_WIDTH as u16, RENDER_HEIGHT as u16, BLACK);
    let render_texture = Texture2D::from_image(&render_buffer);
    render_texture.set_filter(FilterMode::Nearest); // Crisp pixel art look

    // Set up sloped sluice with riffles
    // slope=0.3 means floor rises 0.3 cells per horizontal cell (about 17 degrees)
    create_sluice(&mut sim, 0.3, 20, 5); // Slope, riffle spacing, riffle height

    // Pre-render terrain WITH background (static base layer, never changes)
    let bg_color = Color::from_rgba(20, 20, 30, 255);
    let terrain_color = Color::from_rgba(80, 80, 80, 255);
    let base_pixels: Vec<[u8; 4]> = {
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
        img.get_image_data().to_vec()
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
            create_sluice(&mut sim, 0.3, 20, 5);
            sediment = Sediment::new();
        }
        if is_key_pressed(KeyCode::C) {
            // Clear particles
            sim.particles.list.clear();
            sediment.particles.clear();
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
            if frame_count % 2 == 0 {
                let inlet_x = 20.0;
                let inlet_y = (SIM_HEIGHT as f32 * CELL_SIZE) * 0.2; // Higher up (20% from top)

                // Spawn water flowing right
                sim.spawn_water(inlet_x, inlet_y, 50.0, 0.0, 3);

                // Occasionally add mud
                if frame_count % 20 == 0 {
                    sim.spawn_mud(inlet_x, inlet_y + 5.0, 50.0, 0.0, 1);
                }

                // Spawn sediment with the water (mixed ore feed)
                let sed_pos = glam::Vec2::new(inlet_x, inlet_y);
                let sed_vel = glam::Vec2::new(50.0, 0.0);

                // Sand is most common
                if frame_count % 4 == 0 {
                    sediment.spawn_sand(sed_pos, sed_vel);
                }
                // Magnetite less common (black sand indicator)
                if frame_count % 12 == 0 {
                    sediment.spawn_magnetite(sed_pos, sed_vel);
                }
                // Gold is rare!
                if frame_count % 60 == 0 {
                    sediment.spawn_gold(sed_pos, sed_vel);
                }
            }

            // Run simulation step
            let dt = 1.0 / 60.0;
            sim.update(dt);

            // Update sediment (drift-flux: advected by FLIP velocity + settling)
            // With collision detection against sluice terrain
            let grid = &sim.grid;
            sediment.update_with_collision(
                dt,
                |pos| grid.sample_velocity(pos),
                |pos| {
                    // Convert pixel position to grid cell
                    let gi = (pos.x / grid.cell_size) as usize;
                    let gj = (pos.y / grid.cell_size) as usize;
                    grid.is_solid(gi, gj)
                },
            );

            // Remove sediment that exits the sluice
            sediment.remove_out_of_bounds(RENDER_WIDTH as f32, RENDER_HEIGHT as f32);

            frame_count += 1;
        }

        // --- RENDER (Batched to single texture) ---

        // Fast copy base layer (terrain + background) - single memcpy
        render_buffer.get_image_data_mut().copy_from_slice(&base_pixels);

        // Draw particles to buffer (as 2x2 pixel blocks)
        let water_rgba: [u8; 4] = [30, 100, 200, 255];
        let mud_rgba: [u8; 4] = [139, 90, 43, 255];
        let pixels = render_buffer.get_image_data_mut();

        for particle in sim.particles.iter() {
            let px = particle.position.x as usize;
            let py = particle.position.y as usize;

            if px + 1 >= RENDER_WIDTH || py + 1 >= RENDER_HEIGHT {
                continue;
            }

            let rgba = if particle.is_mud() { mud_rgba } else { water_rgba };

            // Draw 2x2 block (pixels array is [u8; 4] per pixel)
            let base = py * RENDER_WIDTH + px;
            pixels[base] = rgba;
            pixels[base + 1] = rgba;
            let base2 = base + RENDER_WIDTH;
            pixels[base2] = rgba;
            pixels[base2 + 1] = rgba;
        }

        // Draw sediment particles on top (sand, magnetite, gold)
        for particle in sediment.particles.iter() {
            let px = particle.position.x as usize;
            let py = particle.position.y as usize;

            if px + 1 >= RENDER_WIDTH || py + 1 >= RENDER_HEIGHT {
                continue;
            }

            let rgba = particle.material.color();

            // Draw 2x2 block
            let base = py * RENDER_WIDTH + px;
            pixels[base] = rgba;
            pixels[base + 1] = rgba;
            let base2 = base + RENDER_WIDTH;
            pixels[base2] = rgba;
            pixels[base2 + 1] = rgba;
        }

        // Upload buffer to GPU and draw (SINGLE DRAW CALL!)
        render_texture.update(&render_buffer);
        draw_texture_ex(
            &render_texture,
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

        // Draw velocity field on top (optional, still uses draw calls but rarely enabled)
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
        let (sand, magnetite, gold) = sediment.count_by_type();
        draw_text(
            &format!(
                "Water: {} | Sediment: {} | FPS: {} | {}",
                sim.particles.len(),
                sediment.len(),
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
                "Sand: {} | Magnetite: {} | Gold: {}",
                sand, magnetite, gold
            ),
            10.0,
            45.0,
            18.0,
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
