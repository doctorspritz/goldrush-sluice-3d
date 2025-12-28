//! Goldrush Fluid Miner - Game
//!
//! PIC/FLIP fluid simulation demo with sluice box vortex formation.
//! Uses GPU-accelerated instanced circle rendering for fluid particles.

mod render;

use macroquad::prelude::*;
use render::{MetaballRenderer, ParticleRenderer, draw_particles_fast, draw_particles_fast_debug, draw_particles_rect, draw_particles_mesh};
use sim::{
    create_sluice_with_mode, FlipSimulation, RiffleMode, SluiceConfig,
    compute_surface_heightfield,
};

/// Rendering mode selection
#[derive(Clone, Copy, PartialEq)]
enum RenderMode {
    Metaball,      // Two-pass metaball (slowest, best look)
    Hybrid,        // Water as metaballs, Solids as specific shapes
    Shader,        // Per-particle shader circles
    FastCircle,    // macroquad draw_circle batching
    FastRect,      // macroquad draw_rectangle batching
    Mesh,          // Single mesh with vertex colors (fastest)
}

// Simulation size (high resolution for better vortex formation)
// Simulation size (high resolution for better vortex formation)
const SIM_WIDTH: usize = 512;
const SIM_HEIGHT: usize = 256;
const CELL_SIZE: f32 = 1.0;
const SCALE: f32 = 2.5; // Reduced scale to fit screen with double resolution
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

    // Create GPU particle renderers
    let particle_renderer = ParticleRenderer::new();
    let screen_w = (SIM_WIDTH as f32 * CELL_SIZE * SCALE) as u32;
    let screen_h = (SIM_HEIGHT as f32 * CELL_SIZE * SCALE) as u32;
    let mut metaball_renderer = MetaballRenderer::new(screen_w, screen_h);

    // Initial sluice configuration
    let mut sluice_config = SluiceConfig {
        slope: 0.12, // Shallower gradient for better sediment settling
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 50,
    };

    // Set up sloped sluice with riffles
    create_sluice_with_mode(&mut sim, &sluice_config);

    // Helper to generate terrain texture at cell resolution (matches physics)
    fn generate_terrain_texture(
        sim: &FlipSimulation,
        config: &SluiceConfig,
        cell_size: f32,
        render_w: usize,
        render_h: usize,
    ) -> Texture2D {
        let bg_color = Color::from_rgba(20, 20, 30, 255);
        let terrain_color = Color::from_rgba(80, 80, 80, 255);
        let riffle_color = config.riffle_mode.debug_color();
        let riffle_color = Color::from_rgba(riffle_color[0], riffle_color[1], riffle_color[2], riffle_color[3]);

        let mut img = Image::gen_image_color(render_w as u16, render_h as u16, bg_color);

        // Calculate floor for each column to distinguish riffles from floor
        let base_height = sim.grid.height / 4;

        // Render solid cells as blocks (matches physics collision)
        for j in 0..sim.grid.height {
            for i in 0..sim.grid.width {
                if sim.grid.is_solid(i, j) {
                    // Determine if this is a riffle cell (above floor)
                    let floor_y = if i < config.slick_plate_len {
                        base_height
                    } else {
                        base_height + ((i - config.slick_plate_len) as f32 * config.slope) as usize
                    };

                    // Use riffle color for cells above floor line, terrain color for floor
                    let color = if j < floor_y && config.riffle_mode != RiffleMode::None {
                        riffle_color
                    } else {
                        terrain_color
                    };

                    let start_x = (i as f32 * cell_size) as usize;
                    let start_y = (j as f32 * cell_size) as usize;
                    let end_x = ((i + 1) as f32 * cell_size) as usize;
                    let end_y = ((j + 1) as f32 * cell_size) as usize;
                    for py in start_y..end_y.min(render_h) {
                        for px in start_x..end_x.min(render_w) {
                            img.set_pixel(px as u32, py as u32, color);
                        }
                    }
                }
            }
        }

        let tex = Texture2D::from_image(&img);
        tex.set_filter(FilterMode::Nearest);
        tex
    }

    // Create initial terrain texture
    let terrain_texture = generate_terrain_texture(
        &sim, &sluice_config, CELL_SIZE, RENDER_WIDTH, RENDER_HEIGHT
    );

    let mut paused = false;
    let mut show_velocity = false;
    let mut show_surface_line = false;
    let mut debug_state_colors = false; // D key: Bedload=red, Suspended=blue
    let mut render_mode = RenderMode::Mesh; // Default to Mesh for best performance (batches 8000 particles per draw call)
    let mut metaball_threshold: f32 = 0.08;
    let mut metaball_scale: f32 = 6.0;
    let mut fast_particle_size: f32 = CELL_SIZE * SCALE * 1.5;  // Larger for visibility
    let mut frame_count = 0u64;
    let mut fps_samples: Vec<i32> = Vec::new();
    let start_time = std::time::Instant::now();

    // Frame timing (no artificial limiting - vsync handles it)
    let _target_frame_time = std::time::Duration::from_secs_f64(1.0 / 60.0);

    // === TUNABLE PARAMETERS ===
    // Inlet flow (higher spawn rate for finer grid to maintain ~6-8 particles/cell)
    let mut inlet_vx: f32 = 80.0;
    let mut inlet_vy: f32 = 5.0;
    let mut spawn_rate: usize = 40;  // Water particles per frame (10x for strong flow)

    // Riffle geometry is in sluice_config (defined earlier)
    let mut riffle_dirty = false; // rebuild terrain when true

    // Sediment spawn rate (spawn every N frames, 0 = disabled)
    let mut sand_rate: usize = 4;       // 1 per 4 frames = 0.25/frame
    let mut magnetite_rate: usize = 8;  // 1 per 8 frames = heavier material spawns less
    let mut gold_rate: usize = 20;      // 1 per 20 frames = rare, very heavy

    // Global flow multiplier (F/G to adjust) - scales all spawn rates
    let mut flow_multiplier: usize = 1;

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
        if is_key_pressed(KeyCode::R) {
            // Reset simulation with current riffle params
            sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
            create_sluice_with_mode(&mut sim, &sluice_config);
            riffle_dirty = true; // rebuild texture
        }
        // Riffle mode toggle (N key)
        if is_key_pressed(KeyCode::N) {
            sluice_config.riffle_mode = sluice_config.riffle_mode.next();
            riffle_dirty = true;
        }
        // Surface line toggle (L key)
        if is_key_pressed(KeyCode::L) {
            show_surface_line = !show_surface_line;
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
                inlet_vx = (inlet_vx + 5.0).min(200.0); // Higher max for testing
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
            spawn_rate = (spawn_rate + 5).min(100);
        }
        if is_key_pressed(KeyCode::Down) {
            spawn_rate = spawn_rate.saturating_sub(5).max(5);
        }

        // Flow multiplier: =/- (scales everything together)
        if is_key_pressed(KeyCode::Equal) {
            flow_multiplier = (flow_multiplier + 1).min(10);
        }
        if is_key_pressed(KeyCode::Minus) {
            flow_multiplier = flow_multiplier.saturating_sub(1).max(1);
        }

        // Riffle spacing: Q/A (doubled limits for finer grid)
        if is_key_pressed(KeyCode::Q) {
            sluice_config.riffle_spacing = (sluice_config.riffle_spacing + 10).min(120);
            riffle_dirty = true;
        }
        if is_key_pressed(KeyCode::A) {
            sluice_config.riffle_spacing = sluice_config.riffle_spacing.saturating_sub(10).max(30);
            riffle_dirty = true;
        }

        // Riffle height: W/S (doubled limits for finer grid)
        if is_key_pressed(KeyCode::W) {
            sluice_config.riffle_height = (sluice_config.riffle_height + 2).min(16);
            riffle_dirty = true;
        }
        if is_key_pressed(KeyCode::S) {
            sluice_config.riffle_height = sluice_config.riffle_height.saturating_sub(2).max(4);
            riffle_dirty = true;
        }

        // Riffle width: E/D (doubled limits for finer grid)
        if is_key_pressed(KeyCode::E) {
            sluice_config.riffle_width = (sluice_config.riffle_width + 2).min(16);
            riffle_dirty = true;
        }
        if is_key_pressed(KeyCode::D) {
            sluice_config.riffle_width = sluice_config.riffle_width.saturating_sub(2).max(2);
            riffle_dirty = true;
        }
        // X = toggle debug state colors (Bedload=red, Suspended=blue)
        if is_key_pressed(KeyCode::X) {
            debug_state_colors = !debug_state_colors;
        }

        // Sand rate: Key2 to cycle
        if is_key_pressed(KeyCode::Key2) {
            sand_rate = if sand_rate == 0 { 4 } else if sand_rate > 1 { sand_rate - 1 } else { 0 };
        }

        // Magnetite (black sand) rate: Key3 to cycle
        if is_key_pressed(KeyCode::Key3) {
            magnetite_rate = if magnetite_rate == 0 { 8 } else if magnetite_rate > 1 { magnetite_rate - 1 } else { 0 };
        }

        // Gold rate: Key4 to cycle
        if is_key_pressed(KeyCode::Key4) {
            gold_rate = if gold_rate == 0 { 20 } else if gold_rate > 5 { gold_rate - 5 } else { 0 };
        }

        // Sand PIC ratio: ]/[ to adjust (0.0 = pure FLIP, 1.0 = pure PIC)
        if is_key_pressed(KeyCode::RightBracket) {
            sim.sand_pic_ratio = (sim.sand_pic_ratio + 0.1).min(1.0);
            eprintln!("Sand PIC ratio: {:.1}", sim.sand_pic_ratio);
        }
        if is_key_pressed(KeyCode::LeftBracket) {
            sim.sand_pic_ratio = (sim.sand_pic_ratio - 0.1).max(0.0);
            eprintln!("Sand PIC ratio: {:.1}", sim.sand_pic_ratio);
        }

        // Render mode controls - B cycles through modes
        if is_key_pressed(KeyCode::B) {
            render_mode = match render_mode {
                RenderMode::Hybrid => RenderMode::Metaball,
                RenderMode::Metaball => RenderMode::Shader,
                RenderMode::Shader => RenderMode::FastCircle,
                RenderMode::FastCircle => RenderMode::FastRect,
                RenderMode::FastRect => RenderMode::Mesh,
                RenderMode::Mesh => RenderMode::Hybrid,
            };
        }
        if is_key_pressed(KeyCode::T) {
            metaball_threshold = (metaball_threshold + 0.02).min(0.5);
            metaball_renderer.set_threshold(metaball_threshold);
        }
        if is_key_pressed(KeyCode::G) {
            metaball_threshold = (metaball_threshold - 0.02).max(0.02);
            metaball_renderer.set_threshold(metaball_threshold);
        }
        if is_key_pressed(KeyCode::Y) {
            metaball_scale = (metaball_scale + 2.0).min(30.0);
            metaball_renderer.set_particle_scale(metaball_scale);
        }
        if is_key_pressed(KeyCode::H) {
            metaball_scale = (metaball_scale - 2.0).max(6.0);
            metaball_renderer.set_particle_scale(metaball_scale);
        }

        // Particle size tuning: 9/0 to adjust
        if is_key_pressed(KeyCode::Key9) {
            fast_particle_size = (fast_particle_size - 0.5).max(1.0);
        }
        if is_key_pressed(KeyCode::Key0) {
            fast_particle_size = (fast_particle_size + 0.5).min(8.0);
        }

        // Viscosity controls: I to toggle, O/P to adjust
        // Viscosity creates boundary layers needed for vortex shedding
        if is_key_pressed(KeyCode::I) {
            sim.use_viscosity = !sim.use_viscosity;
        }
        if is_key_pressed(KeyCode::O) {
            sim.viscosity = (sim.viscosity - 0.25).max(0.25);
        }
        if is_key_pressed(KeyCode::P) {
            sim.viscosity = (sim.viscosity + 0.25).min(5.0);
        }

        // Rebuild terrain if riffle params changed
        if riffle_dirty {
            riffle_dirty = false;
            sim.particles.list.clear();
            sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
            create_sluice_with_mode(&mut sim, &sluice_config);

            // Rebuild terrain texture with riffle mode color
            terrain_texture = generate_terrain_texture(
                &sim, &sluice_config, CELL_SIZE, RENDER_WIDTH, RENDER_HEIGHT
            );
        }

        // Mouse spawning
        if is_mouse_button_down(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let wx = mx / SCALE;
            let wy = my / SCALE;

            sim.spawn_water(wx, wy, 20.0, 0.0, 5);
        }

        // --- UPDATE ---
        if !paused {
            // Spawn water and sediments using tunable parameters
            // flow_multiplier scales everything: more water, more frequent sediments
            {
                let inlet_x = 5.0;
                // Spawn above the floor (base_height = SIM_HEIGHT/4)
                let inlet_y = (SIM_HEIGHT / 4 - 10) as f32;

                // Water (spawn_rate * flow_multiplier per frame)
                sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, spawn_rate * flow_multiplier);

                // Sand (spawn every N/multiplier frames, 0 = disabled)
                let effective_sand = sand_rate / flow_multiplier.max(1);
                if effective_sand > 0 && frame_count % effective_sand as u64 == 0 {
                    sim.spawn_sand(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
                }

                // Magnetite/black sand (spawn every N/multiplier frames, 0 = disabled)
                let effective_magnetite = magnetite_rate / flow_multiplier.max(1);
                if effective_magnetite > 0 && frame_count % effective_magnetite as u64 == 0 {
                    sim.spawn_magnetite(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
                }

                // Gold (spawn every N/multiplier frames, 0 = disabled) - rare and heavy!
                let effective_gold = gold_rate / flow_multiplier.max(1);
                if effective_gold > 0 && frame_count % effective_gold as u64 == 0 {
                    sim.spawn_gold(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
                }
            }

            // Remove particles that reached the end (right side outflow)
            let outflow_x = (SIM_WIDTH as f32 - 5.0) * CELL_SIZE;
            sim.particles.list.retain(|p| p.position.x < outflow_x);

            // Run simulation step with timing
            let sim_start = std::time::Instant::now();
            let dt = 1.0 / 60.0;
            sim.update(dt);
            let sim_time = sim_start.elapsed();

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
                println!("t={:3}s: {:5} p, fps={:3}, sim={:4.1}ms, div={:4.1}",
                    elapsed, sim.particles.len(), fps, sim_time.as_secs_f32() * 1000.0, div);
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

        // Draw particles with selected renderer
        match render_mode {
            RenderMode::Hybrid => {
                // Pass 1: Water as metaballs for fluid look
                metaball_renderer.draw_filtered(&sim.particles, SCALE, true);
                // Pass 2: Solids as sharp sprites for clarity
                particle_renderer.draw_filtered(&sim.particles, SCALE, false);
            }
            // Legacy modes use full draw (passing true/false doesn't matter for non-filtered methods if we didn't update them,
            // but we only updated filtered ones. Wait, I only added `draw_filtered`. 
            // The original `draw` methods still exist and draw everything.
            RenderMode::Metaball => metaball_renderer.draw(&sim.particles, SCALE), 
            RenderMode::Shader => particle_renderer.draw_sorted(&sim.particles, SCALE),
            RenderMode::FastCircle => draw_particles_fast_debug(&sim.particles, SCALE, fast_particle_size, debug_state_colors),
            RenderMode::FastRect => draw_particles_rect(&sim.particles, SCALE, fast_particle_size),
            RenderMode::Mesh => draw_particles_mesh(&sim.particles, SCALE, fast_particle_size),
        }

        // Draw deposited sediment cells ON TOP of particles (so visible in all render modes)
        // Uses composition-based coloring for mixed-material beds
        for j in 0..sim.grid.height {
            for i in 0..sim.grid.width {
                if sim.grid.is_deposited(i, j) {
                    let x = i as f32 * CELL_SIZE * SCALE;
                    let y = j as f32 * CELL_SIZE * SCALE;
                    let size = CELL_SIZE * SCALE;
                    // Get composition-based color from the deposited cell
                    let cell = &sim.grid.deposited[sim.grid.cell_index(i, j)];
                    let rgba = cell.color();
                    let deposit_color = Color::from_rgba(rgba[0], rgba[1], rgba[2], rgba[3]);
                    draw_rectangle(x, y, size, size, deposit_color);
                }
            }
        }

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

        // Draw surface heightfield line (optional debug visualization)
        if show_surface_line {
            let surface = compute_surface_heightfield(&sim);
            let surface_color = Color::from_rgba(100, 200, 255, 200);

            for i in 1..sim.grid.width {
                let x0 = (i - 1) as f32 * CELL_SIZE * SCALE;
                let x1 = i as f32 * CELL_SIZE * SCALE;
                let y0 = surface[i - 1] * SCALE;
                let y1 = surface[i] * SCALE;

                // Only draw if valid (not MAX)
                if y0 < screen_height() && y1 < screen_height() {
                    draw_line(x0, y0, x1, y1, 2.0, surface_color);
                }
            }
        }

        // --- UI ---
        let (water, mud, sand, magnetite, gold) = sim.particles.count_by_material();
        let total_sediment = mud + sand + magnetite + gold;

        // Top status bar
        let mode_str = match render_mode {
            RenderMode::Metaball => "METABALL",
            RenderMode::Hybrid => "HYBRID",
            RenderMode::Shader => "SHADER",
            RenderMode::FastCircle => "FAST-CIRCLE",
            RenderMode::FastRect => "FAST-RECT",
            RenderMode::Mesh => "MESH-BATCH",
        };
        draw_text(
            &format!(
                "Particles: {} | FPS: {} | {} | {}",
                sim.particles.len(),
                get_fps(),
                if paused { "PAUSED" } else { "Running" },
                mode_str
            ),
            10.0, 20.0, 18.0, WHITE,
        );

        // Current parameters - LEFT SIDE
        draw_text(
            &format!("FLOW: vx={:.0} vy={:.0} rate={} x{}", inlet_vx, inlet_vy, spawn_rate, flow_multiplier),
            10.0, 42.0, 16.0, Color::from_rgba(100, 200, 255, 255),
        );
        // Riffle mode with mode-specific color
        let riffle_color = sluice_config.riffle_mode.debug_color();
        draw_text(
            &format!("RIFFLES: {} | spacing={} height={} width={}",
                sluice_config.riffle_mode.name(),
                sluice_config.riffle_spacing, sluice_config.riffle_height, sluice_config.riffle_width),
            10.0, 58.0, 16.0, Color::from_rgba(riffle_color[0], riffle_color[1], riffle_color[2], 255),
        );
        draw_text(
            &format!("SAND[2]: 1/{} | MAG[3]: 1/{}",
                if sand_rate > 0 { sand_rate.to_string() } else { "off".to_string() },
                if magnetite_rate > 0 { magnetite_rate.to_string() } else { "off".to_string() },
            ),
            10.0, 74.0, 14.0, Color::from_rgba(255, 215, 0, 200),
        );

        // Material counts with slurry percentage
        let total = water + total_sediment;
        let solids_pct = if total > 0 { (total_sediment as f32 / total as f32) * 100.0 } else { 0.0 };
        draw_text(
            &format!("Water:{} Sand:{} Mag:{} Gold:{} | Solids: {:.1}%",
                water, sand, magnetite, gold, solids_pct),
            10.0, 92.0, 14.0, Color::from_rgba(200, 200, 200, 255),
        );

        // Metaball params (when active)
        if render_mode == RenderMode::Metaball || render_mode == RenderMode::Hybrid {
            draw_text(
                &format!("METABALL: threshold={:.2} scale={:.0}", metaball_threshold, metaball_scale),
                10.0, 108.0, 14.0, Color::from_rgba(180, 100, 255, 255),
            );
        }

        // Particle size
        draw_text(
            &format!("PARTICLE SIZE={:.1}", fast_particle_size),
            10.0, 124.0, 14.0, Color::from_rgba(100, 200, 100, 255),
        );

        // Viscosity params (for vortex shedding)
        let visc_status = if sim.use_viscosity {
            format!("VISCOSITY: ON (ν={:.2})", sim.viscosity)
        } else {
            "VISCOSITY: OFF".to_string()
        };
        let visc_color = if sim.use_viscosity {
            Color::from_rgba(255, 150, 100, 255)
        } else {
            Color::from_rgba(100, 100, 100, 255)
        };
        draw_text(&visc_status, 10.0, 140.0, 14.0, visc_color);

        // Sand PIC ratio
        draw_text(
            &format!("SAND PIC: {:.0}% ([/] to adjust)", sim.sand_pic_ratio * 100.0),
            10.0, 156.0, 14.0, Color::from_rgba(220, 180, 100, 255),
        );

        // Controls - BOTTOM
        draw_text(
            "←→ vx | Shift+←→ vy | ↑↓ spawn | +/- flow×  | Q/A spacing | W/S height | E/D width",
            10.0, screen_height() - 90.0, 14.0, GRAY,
        );
        draw_text(
            "[N]=Riffle Mode | [L]=Surface Line | [B]=Render | T/G threshold | Y/H scale",
            10.0, screen_height() - 74.0, 14.0, GRAY,
        );
        draw_text(
            "5/6 H | 7/8 rest | 9/0 size | [I]=Viscosity | O/P ν",
            10.0, screen_height() - 58.0, 14.0, GRAY,
        );
        draw_text(
            "[Space]=Pause [V]=Velocity [R]=Reset [C]=Clear",
            10.0, screen_height() - 42.0, 14.0, GRAY,
        );

        // === CURSOR DEBUG ===
        {
            let (mx, my) = mouse_position();
            let wx = mx / SCALE;
            let wy = my / SCALE;
            let col = (wx / CELL_SIZE) as usize;

            // Get pile height at cursor column
            let pile_h = if col < sim.pile_height.len() {
                sim.pile_height[col]
            } else {
                f32::INFINITY
            };

            // Count particles near cursor (within 10 world units)
            let mut bedload_near = 0;
            let mut suspended_near = 0;
            for p in &sim.particles.list {
                let dx = p.position.x - wx;
                let dy = p.position.y - wy;
                if dx*dx + dy*dy < 100.0 {
                    if p.is_sediment() {
                        match p.state {
                            sim::particle::ParticleState::Bedload => bedload_near += 1,
                            sim::particle::ParticleState::Suspended => suspended_near += 1,
                        }
                    }
                }
            }

            let pile_str = if pile_h < f32::INFINITY {
                format!("{:.1}", pile_h)
            } else {
                "none".to_string()
            };

            let debug_text = format!(
                "cursor: ({:.0},{:.0}) col:{} pile:{} | nearby: bed={} sus={}",
                wx, wy, col, pile_str, bedload_near, suspended_near
            );
            draw_text(&debug_text, mx + 15.0, my - 10.0, 14.0, YELLOW);
        }

        // Frame rate limiting disabled - we're CPU-bound on simulation
        // The macroquad vsync will limit to monitor refresh rate
        let _ = frame_start; // suppress unused warning

        next_frame().await
    }
}
