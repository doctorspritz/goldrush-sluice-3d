//! FLIP + DEM Coupling Test
//!
//! Full two-phase simulation: FLIP water flow + DEM sediment.
//! Watch gold settle in vortices while sand gets carried downstream.

use macroquad::prelude::*;
use sim::dem::DemSimulation;
use sim::flip::FlipSimulation;
use sim::grid::CellType;
use sim::particle::ParticleMaterial;

const WIDTH: usize = 180;
const HEIGHT: usize = 70;
const CELL_SIZE: f32 = 5.0;
const SCALE: f32 = 1.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "FLIP + DEM Coupling Test".to_owned(),
        window_width: (WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

struct FlipDemScene {
    flip: FlipSimulation,
    dem: DemSimulation,
    paused: bool,
    spawn_material: ParticleMaterial,
    frame_times: Vec<f32>,
    continuous_spawn: bool,
    spawn_timer: f32,
    show_water: bool,
    show_velocity: bool,
}

impl FlipDemScene {
    fn new() -> Self {
        // Create FLIP simulation with sluice geometry
        let mut flip = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

        // Build sluice: SLOPED floor with riffles
        // CRITICAL: Use set_solid() to set BOTH solid[] and cell_type[]
        // Otherwise classify_cells() will lose terrain on frame 2!

        // Sloped floor: higher on left (row 58), lower on right (row 64)
        let floor_left = HEIGHT - 12;  // row 58
        let floor_right = HEIGHT - 6;  // row 64
        let slope = (floor_right as f32 - floor_left as f32) / WIDTH as f32;

        for i in 0..WIDTH {
            // Calculate floor row at this x position (sloped)
            let floor_row = (floor_left as f32 + slope * i as f32) as usize;

            for j in 0..HEIGHT {
                // Sloped floor
                if j >= floor_row {
                    flip.grid.set_solid(i, j);
                }
                // Left wall (inlet region) - shorter to allow water entry
                if i < 2 && j > 15 {
                    flip.grid.set_solid(i, j);
                }
                // Right wall - leave open for outlet
                // (particles exit naturally)
            }
        }

        // Add riffles on the sloped floor
        let riffle_positions = [WIDTH / 4, WIDTH / 2, 3 * WIDTH / 4];
        for &rx in &riffle_positions {
            let base_floor = (floor_left as f32 + slope * rx as f32) as usize;
            for di in 0..4 {
                let i = rx + di;
                if i < WIDTH {
                    for dj in 1..=5 {
                        let j = base_floor.saturating_sub(dj);
                        if j < HEIGHT {
                            flip.grid.set_solid(i, j);
                        }
                    }
                }
            }
        }

        flip.grid.compute_sdf();

        // No initial water - continuous flow fills the sluice
        Self::initialize_water(&mut flip, floor_left);

        let dem = DemSimulation::new();

        Self {
            flip,
            dem,
            paused: false,
            spawn_material: ParticleMaterial::Sand,
            frame_times: Vec::with_capacity(60),
            continuous_spawn: false,
            spawn_timer: 0.0,
            show_water: true,
            show_velocity: false,
        }
    }

    fn initialize_water(_flip: &mut FlipSimulation, _floor_row: usize) {
        // No initial water - continuous flow will fill the sluice
    }

    /// Spawn water at inlet (left side) - called every frame for continuous flow
    fn spawn_water_at_inlet(&mut self) {
        let cell_size = CELL_SIZE;
        let floor_left = HEIGHT - 12; // Match sloped floor
        let spacing = cell_size / 2.5;

        // Spawn water in a column at the left inlet
        let spawn_x = cell_size * 3.0; // Just past the left wall
        let water_top = floor_left - 14;  // Higher water column
        let water_bottom = floor_left - 1;

        // Spawn ~25 particles per frame at inlet (denser, faster flow)
        for j in water_top..water_bottom {
            let base_y = j as f32 * cell_size;
            for pj in 0..2 {
                let py = base_y + (pj as f32 + 0.25) * spacing;
                let jitter_x = rand::gen_range(0.0, 1.0) * spacing * 0.3;
                let jitter_y = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;
                // Fast rightward flow velocity
                self.flip.particles.spawn_water(spawn_x + jitter_x, py + jitter_y, 150.0, 0.0);
            }
        }
    }

    fn spawn_sediment_at_inlet(&mut self, count: usize) {
        // Spawn sediment at inlet (left side, above water)
        let spawn_x = CELL_SIZE * 10.0;
        let spawn_y = CELL_SIZE * 20.0; // Above water surface
        let spacing = CELL_SIZE * 0.8;

        for i in 0..count {
            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * spacing;
            let jitter_y = (rand::gen_range(0.0, 1.0) - 0.5) * spacing;
            let y = spawn_y + (i as f32) * spacing * 0.3;

            match self.spawn_material {
                ParticleMaterial::Sand => self.flip.particles.spawn_sand(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Gold => self.flip.particles.spawn_gold(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Magnetite => self.flip.particles.spawn_magnetite(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
                _ => self.flip.particles.spawn_sand(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
            }
        }
    }

    fn spawn_mixed_batch(&mut self) {
        let spawn_x = CELL_SIZE * 10.0;
        let spawn_y = CELL_SIZE * 20.0;
        let spacing = CELL_SIZE * 0.7;

        for i in 0..10 {
            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 2.0;
            let jitter_y = i as f32 * spacing * 0.4;

            // 1 in 5 is gold
            if rand::gen_range(0, 5) == 0 {
                self.flip.particles.spawn_gold(spawn_x + jitter_x, spawn_y + jitter_y, 0.0, 0.0);
            } else {
                self.flip.particles.spawn_sand(spawn_x + jitter_x, spawn_y + jitter_y, 0.0, 0.0);
            }
        }
    }

    fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        // Continuous water inflow at left inlet
        self.spawn_water_at_inlet();

        // Continuous sediment spawning
        if self.continuous_spawn {
            self.spawn_timer += dt;
            if self.spawn_timer >= 0.2 {
                self.spawn_timer = 0.0;
                self.spawn_mixed_batch();
            }
        }

        let start = get_time();

        // Step 1: Run FLIP for water
        self.flip.update(dt);

        // Step 2: Run DEM for sediment, coupled with FLIP velocities
        // Build spatial hash for DEM
        let mut cell_head = vec![-1i32; WIDTH * HEIGHT];
        let mut particle_next = vec![-1i32; self.flip.particles.len()];

        for (idx, p) in self.flip.particles.iter().enumerate() {
            if p.is_sediment() {
                let gi = ((p.position.x / CELL_SIZE) as usize).min(WIDTH - 1);
                let gj = ((p.position.y / CELL_SIZE) as usize).min(HEIGHT - 1);
                let cell_idx = gj * WIDTH + gi;

                particle_next[idx] = cell_head[cell_idx];
                cell_head[cell_idx] = idx as i32;
            }
        }

        // Run DEM with FLIP coupling
        self.dem.update_coupled(
            &mut self.flip.particles,
            &self.flip.grid,
            &cell_head,
            &particle_next,
            dt,
            200.0, // gravity
            None,  // Use grid cell types for water detection
            true,  // Use FLIP velocities for drag
        );

        // Remove particles that escaped
        let max_x = WIDTH as f32 * CELL_SIZE;
        self.flip.particles.list.retain(|p| p.position.x < max_x && p.position.x > 0.0);

        let elapsed = (get_time() - start) as f32 * 1000.0;
        self.frame_times.push(elapsed);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(20, 25, 30, 255));

        // Draw grid cells (floor/walls/riffles)
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                if self.flip.grid.cell_type[j * WIDTH + i] == CellType::Solid {
                    let x = i as f32 * CELL_SIZE * SCALE;
                    let y = j as f32 * CELL_SIZE * SCALE;
                    draw_rectangle(x, y, CELL_SIZE * SCALE, CELL_SIZE * SCALE, DARKGRAY);
                }
            }
        }

        // Draw velocity field if enabled
        if self.show_velocity {
            self.draw_velocity_field();
        }

        // Draw particles
        let water_radius = CELL_SIZE * 0.3 * SCALE;
        let sediment_radius = CELL_SIZE * 0.4 * SCALE;

        for p in self.flip.particles.iter() {
            let x = p.position.x * SCALE;
            let y = p.position.y * SCALE;

            match p.material {
                ParticleMaterial::Water => {
                    if self.show_water {
                        draw_circle(x, y, water_radius, Color::from_rgba(60, 140, 220, 180));
                    }
                }
                ParticleMaterial::Sand => {
                    draw_circle(x, y, sediment_radius, Color::from_rgba(194, 178, 128, 255));
                }
                ParticleMaterial::Gold => {
                    draw_circle(x, y, sediment_radius * 0.7, Color::from_rgba(255, 215, 0, 255));
                }
                ParticleMaterial::Magnetite => {
                    draw_circle(x, y, sediment_radius * 1.2, Color::from_rgba(40, 40, 40, 255));
                }
                _ => {}
            }
        }

        self.draw_diagnostics();
    }

    fn draw_velocity_field(&self) {
        let step = 6;
        let arrow_scale = 0.15;

        for gx in (0..WIDTH).step_by(step) {
            for gy in (0..HEIGHT).step_by(step) {
                let idx = gy * WIDTH + gx;
                if self.flip.grid.cell_type[idx] == CellType::Solid {
                    continue;
                }

                let pos = glam::Vec2::new(
                    (gx as f32 + 0.5) * CELL_SIZE,
                    (gy as f32 + 0.5) * CELL_SIZE,
                );

                let vel = self.flip.grid.sample_velocity(pos);
                let speed = vel.length();

                if speed > 1.0 {
                    let x1 = pos.x * SCALE;
                    let y1 = pos.y * SCALE;
                    let x2 = x1 + vel.x * arrow_scale;
                    let y2 = y1 + vel.y * arrow_scale;

                    let alpha = (speed / 150.0).min(1.0);
                    let color = Color::from_rgba(100, 200, 255, (alpha * 200.0) as u8);

                    draw_line(x1, y1, x2, y2, 1.0, color);
                }
            }
        }
    }

    fn draw_diagnostics(&self) {
        let mut water_count = 0;
        let mut sand_count = 0;
        let mut gold_count = 0;

        for p in self.flip.particles.iter() {
            match p.material {
                ParticleMaterial::Water => water_count += 1,
                ParticleMaterial::Sand => sand_count += 1,
                ParticleMaterial::Gold => gold_count += 1,
                _ => {}
            }
        }

        let avg_frame_ms = if !self.frame_times.is_empty() {
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
        } else {
            0.0
        };

        let y_start = 10.0;
        let line_height = 16.0;
        let x = 10.0;

        draw_text(&format!("Water: {}", water_count), x, y_start, 16.0, Color::from_rgba(60, 140, 220, 255));
        draw_text(&format!("Sand: {}", sand_count), x, y_start + line_height, 16.0, Color::from_rgba(194, 178, 128, 255));
        draw_text(&format!("Gold: {}", gold_count), x, y_start + line_height * 2.0, 16.0, GOLD);
        draw_text(&format!("Sim: {:.1} ms", avg_frame_ms), x, y_start + line_height * 3.0, 16.0,
            if avg_frame_ms < 16.0 { GREEN } else { RED });

        // Controls
        let help_y = screen_height() - 70.0;
        draw_text("Click: spawn sediment | M: mixed batch | F: continuous", x, help_y, 13.0, GRAY);
        draw_text("1/2/3: Sand/Gold/Magnetite | W: toggle water | V: velocity", x, help_y + 14.0, 13.0, GRAY);
        draw_text("R: reset | Space: pause", x, help_y + 28.0, 13.0, GRAY);

        if self.continuous_spawn {
            draw_text("FLOWING", screen_width() - 70.0, y_start, 16.0, GREEN);
        }

        if self.paused {
            draw_text("PAUSED", screen_width() / 2.0 - 35.0, 25.0, 25.0, RED);
        }
    }

    fn handle_input(&mut self) {
        if is_mouse_button_pressed(MouseButton::Left) {
            self.spawn_sediment_at_inlet(8);
        }

        if is_key_pressed(KeyCode::M) {
            self.spawn_mixed_batch();
        }

        if is_key_pressed(KeyCode::Key1) {
            self.spawn_material = ParticleMaterial::Sand;
        }
        if is_key_pressed(KeyCode::Key2) {
            self.spawn_material = ParticleMaterial::Gold;
        }
        if is_key_pressed(KeyCode::Key3) {
            self.spawn_material = ParticleMaterial::Magnetite;
        }

        if is_key_pressed(KeyCode::F) {
            self.continuous_spawn = !self.continuous_spawn;
        }

        if is_key_pressed(KeyCode::W) {
            self.show_water = !self.show_water;
        }

        if is_key_pressed(KeyCode::V) {
            self.show_velocity = !self.show_velocity;
        }

        if is_key_pressed(KeyCode::R) {
            *self = Self::new();
        }

        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut scene = FlipDemScene::new();

    loop {
        scene.handle_input();

        let dt = 1.0 / 60.0;
        scene.update(dt);
        scene.draw();

        next_frame().await;
    }
}
