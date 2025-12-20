//! Goldrush Fluid Miner - Game
//!
//! PIC/FLIP fluid simulation demo with sluice box vortex formation.
//! Uses GPU-accelerated instanced circle rendering for fluid particles.

mod render;

use macroquad::prelude::*;
use render::ParticleRenderer;
use sim::{create_sluice, FlipSimulation};

// Simulation size
const SIM_WIDTH: usize = 128;
const SIM_HEIGHT: usize = 96;
const CELL_SIZE: f32 = 2.0;
const SCALE: f32 = 5.0;

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
    // Riffle tests: (slope, spacing, HEIGHT, WIDTH)
    // Baseline: (0.25, 30, 3, 4)
    // R1: spacing=20 (more riffles)
    // R2: spacing=45 (fewer riffles)
    // R3: width=6 (thicker)
    // R4: width=2 (thinner)
    // R5: height=5 (taller)
    create_sluice(&mut sim, 0.25, 30, 5, 4); // R5: taller riffles (height 3→5)

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
    let mut fps_samples: Vec<i32> = Vec::new();
    let start_time = std::time::Instant::now();

    // Frame rate limiting
    let target_frame_time = std::time::Duration::from_secs_f64(1.0 / 60.0);

    // === TUNABLE PARAMETERS ===
    // Inlet flow
    let mut inlet_vx: f32 = 69.0;
    let mut inlet_vy: f32 = 40.0;
    let mut spawn_rate: usize = 14;

    // Riffle geometry
    let mut riffle_spacing: usize = 30;
    let mut riffle_height: usize = 4;
    let mut riffle_width: usize = 4;
    let mut riffle_dirty = false; // rebuild terrain when true

    // Sediment blend (spawn every N frames, 0 = disabled)
    let mut mud_rate: usize = 6;
    let mut sand_rate: usize = 2;
    let mut magnetite_rate: usize = 5;
    let mut gold_rate: usize = 15;

    // Mutable terrain texture
    let mut terrain_texture = terrain_texture;

    loop {
        let frame_start = std::time::Instant::now();

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
            // Reset simulation with current riffle params
            sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
            create_sluice(&mut sim, 0.25, riffle_spacing, riffle_height, riffle_width);
            riffle_dirty = true; // rebuild texture
        }
        if is_key_pressed(KeyCode::C) {
            // Clear particles
            sim.particles.list.clear();
        }

        // === PARAMETER CONTROLS ===
        // Velocity: Arrow keys (with shift for vy)
        if is_key_pressed(KeyCode::Right) {
            if is_key_down(KeyCode::LeftShift) {
                inlet_vy = (inlet_vy + 5.0).min(80.0);
            } else {
                inlet_vx = (inlet_vx + 5.0).min(100.0);
            }
        }
        if is_key_pressed(KeyCode::Left) {
            if is_key_down(KeyCode::LeftShift) {
                inlet_vy = (inlet_vy - 5.0).max(0.0);
            } else {
                inlet_vx = (inlet_vx - 5.0).max(20.0);
            }
        }

        // Spawn rate: Up/Down arrows
        if is_key_pressed(KeyCode::Up) {
            spawn_rate = (spawn_rate + 2).min(30);
        }
        if is_key_pressed(KeyCode::Down) {
            spawn_rate = spawn_rate.saturating_sub(2).max(2);
        }

        // Riffle spacing: Q/A
        if is_key_pressed(KeyCode::Q) {
            riffle_spacing = (riffle_spacing + 5).min(60);
            riffle_dirty = true;
        }
        if is_key_pressed(KeyCode::A) {
            riffle_spacing = riffle_spacing.saturating_sub(5).max(15);
            riffle_dirty = true;
        }

        // Riffle height: W/S
        if is_key_pressed(KeyCode::W) {
            riffle_height = (riffle_height + 1).min(8);
            riffle_dirty = true;
        }
        if is_key_pressed(KeyCode::S) {
            riffle_height = riffle_height.saturating_sub(1).max(2);
            riffle_dirty = true;
        }

        // Riffle width: E/D
        if is_key_pressed(KeyCode::E) {
            riffle_width = (riffle_width + 1).min(8);
            riffle_dirty = true;
        }
        if is_key_pressed(KeyCode::D) {
            riffle_width = riffle_width.saturating_sub(1).max(1);
            riffle_dirty = true;
        }

        // Sediment rates: 1-4 to cycle
        if is_key_pressed(KeyCode::Key1) {
            mud_rate = if mud_rate == 0 { 6 } else if mud_rate > 2 { mud_rate - 2 } else { 0 };
        }
        if is_key_pressed(KeyCode::Key2) {
            sand_rate = if sand_rate == 0 { 4 } else if sand_rate > 1 { sand_rate - 1 } else { 0 };
        }
        if is_key_pressed(KeyCode::Key3) {
            magnetite_rate = if magnetite_rate == 0 { 8 } else if magnetite_rate > 2 { magnetite_rate - 2 } else { 0 };
        }
        if is_key_pressed(KeyCode::Key4) {
            gold_rate = if gold_rate == 0 { 20 } else if gold_rate > 5 { gold_rate - 5 } else { 0 };
        }

        // Rebuild terrain if riffle params changed
        if riffle_dirty {
            riffle_dirty = false;
            sim.particles.list.clear();
            sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
            create_sluice(&mut sim, 0.25, riffle_spacing, riffle_height, riffle_width);

            // Rebuild terrain texture
            let bg_color = Color::from_rgba(20, 20, 30, 255);
            let terrain_color = Color::from_rgba(80, 80, 80, 255);
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
            terrain_texture = Texture2D::from_image(&img);
            terrain_texture.set_filter(FilterMode::Nearest);
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
            // Spawn water and sediments using tunable parameters
            {
                let inlet_x = 5.0;
                let inlet_y = 43.0;

                // Water
                sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, spawn_rate);

                // Sediments (spawn every N frames, 0 = disabled)
                if mud_rate > 0 && frame_count % mud_rate as u64 == 0 {
                    sim.spawn_mud(inlet_x, inlet_y + 3.0, inlet_vx, inlet_vy, 2);
                }
                if sand_rate > 0 && frame_count % sand_rate as u64 == 0 {
                    sim.spawn_sand(inlet_x, inlet_y, inlet_vx, inlet_vy, 2);
                }
                if magnetite_rate > 0 && frame_count % magnetite_rate as u64 == 0 {
                    sim.spawn_magnetite(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
                }
                if gold_rate > 0 && frame_count % gold_rate as u64 == 0 {
                    sim.spawn_gold(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
                }
            }

            // Remove particles that reached the end (right side outflow)
            let outflow_x = (SIM_WIDTH as f32 - 5.0) * CELL_SIZE;
            sim.particles.list.retain(|p| p.position.x < outflow_x);

            // Run simulation step
            let dt = 1.0 / 60.0;
            sim.update(dt);

            frame_count += 1;

            // Log diagnostics every second
            if frame_count % 60 == 0 {
                // Check post-update divergence (after pressure solve)
                sim.grid.compute_divergence();
                let div = sim.grid.total_divergence();

                let mut max_vx: f32 = 0.0;  // Horizontal
                let mut max_vy_down: f32 = 0.0;  // Positive Y = downward
                let mut max_vy_up: f32 = 0.0;    // Negative Y = upward ejection
                for p in sim.particles.iter() {
                    max_vx = max_vx.max(p.velocity.x.abs());
                    if p.velocity.y > 0.0 {
                        max_vy_down = max_vy_down.max(p.velocity.y);
                    } else {
                        max_vy_up = max_vy_up.max(-p.velocity.y);
                    }
                }
                let fps = get_fps();
                fps_samples.push(fps);
                let elapsed = start_time.elapsed().as_secs();
                println!("t={:3}s: {:4} p, fps={:3}, div={:4.1}, vx={:.0}, vy=↓{:.0}/↑{:.0}",
                    elapsed, sim.particles.len(), fps, div, max_vx, max_vy_down, max_vy_up);
            }
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

        // Top status bar
        draw_text(
            &format!(
                "Particles: {} | FPS: {} | {}",
                sim.particles.len(),
                get_fps(),
                if paused { "PAUSED" } else { "Running" }
            ),
            10.0, 20.0, 18.0, WHITE,
        );

        // Current parameters - LEFT SIDE
        draw_text(
            &format!("FLOW: vx={:.0} vy={:.0} rate={}", inlet_vx, inlet_vy, spawn_rate),
            10.0, 42.0, 16.0, Color::from_rgba(100, 200, 255, 255),
        );
        draw_text(
            &format!("RIFFLES: spacing={} height={} width={}", riffle_spacing, riffle_height, riffle_width),
            10.0, 58.0, 16.0, Color::from_rgba(150, 150, 150, 255),
        );
        draw_text(
            &format!("SEDIMENT: mud=1/{} sand=1/{} mag=1/{} gold=1/{}",
                if mud_rate > 0 { mud_rate.to_string() } else { "off".to_string() },
                if sand_rate > 0 { sand_rate.to_string() } else { "off".to_string() },
                if magnetite_rate > 0 { magnetite_rate.to_string() } else { "off".to_string() },
                if gold_rate > 0 { gold_rate.to_string() } else { "off".to_string() },
            ),
            10.0, 74.0, 14.0, Color::from_rgba(255, 215, 0, 200),
        );

        // Material counts
        draw_text(
            &format!("W:{} M:{} S:{} Mag:{} Au:{}", water, mud, sand, magnetite, gold),
            10.0, 92.0, 14.0, Color::from_rgba(200, 200, 200, 255),
        );

        // Controls - BOTTOM
        draw_text(
            "←→ vx | Shift+←→ vy | ↑↓ spawn | Q/A spacing | W/S height | E/D width | 1234 sediment",
            10.0, screen_height() - 28.0, 14.0, GRAY,
        );
        draw_text(
            "[Space]=Pause [V]=Velocity [R]=Reset [C]=Clear",
            10.0, screen_height() - 10.0, 14.0, GRAY,
        );

        // Frame rate limiting - sleep if we finished early
        let elapsed = frame_start.elapsed();
        if elapsed < target_frame_time {
            std::thread::sleep(target_frame_time - elapsed);
        }

        next_frame().await
    }
}
