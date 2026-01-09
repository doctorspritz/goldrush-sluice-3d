//! Water Settling Test - Sediment falling through water
//!
//! Tests buoyancy-reduced settling of sediment particles in water.
//! Sand falls slowly, gold sinks fast.

use macroquad::prelude::*;
use sim::dem::DemSimulation;
use sim::grid::{CellType, Grid};
use sim::particle::{ParticleMaterial, Particles};

const WIDTH: usize = 120;
const HEIGHT: usize = 100;
const CELL_SIZE: f32 = 6.0;
const SCALE: f32 = 1.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "Water Settling Test".to_owned(),
        window_width: (WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

struct WaterSettlingScene {
    grid: Grid,
    particles: Particles,
    dem: DemSimulation,
    cell_head: Vec<i32>,
    particle_next: Vec<i32>,
    paused: bool,
    water_level: f32,  // Y coordinate of water surface
    spawn_material: ParticleMaterial,
    frame_times: Vec<f32>,
}

impl WaterSettlingScene {
    fn new() -> Self {
        let mut grid = Grid::new(WIDTH, HEIGHT, CELL_SIZE);

        // Create container with floor and walls
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                // Floor (bottom 8%)
                if j >= HEIGHT - HEIGHT / 12 {
                    grid.cell_type[j * WIDTH + i] = CellType::Solid;
                }
                // Left wall
                if i < 2 {
                    grid.cell_type[j * WIDTH + i] = CellType::Solid;
                }
                // Right wall
                if i >= WIDTH - 2 {
                    grid.cell_type[j * WIDTH + i] = CellType::Solid;
                }
            }
        }

        grid.compute_sdf();

        let particles = Particles::new();
        let dem = DemSimulation::new();

        // Water fills bottom 60% of container
        let water_level = HEIGHT as f32 * CELL_SIZE * 0.35;

        Self {
            grid,
            particles,
            dem,
            cell_head: vec![-1; WIDTH * HEIGHT],
            particle_next: Vec::new(),
            paused: false,
            water_level,
            spawn_material: ParticleMaterial::Sand,
            frame_times: Vec::with_capacity(60),
        }
    }

    fn spawn_above_water(&mut self, count: usize) {
        let center_x = WIDTH as f32 * CELL_SIZE / 2.0;
        let spawn_y = self.water_level - CELL_SIZE * 8.0; // Above water
        let spacing = CELL_SIZE * 0.9;

        let cols = (count as f32).sqrt().ceil() as usize;

        for i in 0..count {
            let col = i % cols;
            let row = i / cols;
            let x = center_x + (col as f32 - cols as f32 / 2.0) * spacing;
            let y = spawn_y - row as f32 * spacing;

            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;
            let jitter_y = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;

            match self.spawn_material {
                ParticleMaterial::Sand => self.particles.spawn_sand(x + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Gold => self.particles.spawn_gold(x + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Magnetite => self.particles.spawn_magnetite(x + jitter_x, y + jitter_y, 0.0, 0.0),
                _ => self.particles.spawn_sand(x + jitter_x, y + jitter_y, 0.0, 0.0),
            }
        }
    }

    fn spawn_mixed_batch(&mut self) {
        let center_x = WIDTH as f32 * CELL_SIZE / 2.0;
        let spawn_y = self.water_level - CELL_SIZE * 6.0;
        let spacing = CELL_SIZE * 1.0;

        // Spawn mix of sand and gold
        for i in 0..20 {
            let x = center_x + (i as f32 - 10.0) * spacing * 0.5;
            let jitter = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;

            if i % 4 == 0 {
                // Gold (1 in 4)
                self.particles.spawn_gold(x + jitter, spawn_y, 0.0, 0.0);
            } else {
                // Sand
                self.particles.spawn_sand(x + jitter, spawn_y, 0.0, 0.0);
            }
        }
    }

    fn rebuild_spatial_hash(&mut self) {
        self.cell_head.fill(-1);
        self.particle_next.resize(self.particles.len(), -1);
        self.particle_next.fill(-1);

        for (idx, p) in self.particles.iter().enumerate() {
            let gi = ((p.position.x / CELL_SIZE) as usize).min(WIDTH - 1);
            let gj = ((p.position.y / CELL_SIZE) as usize).min(HEIGHT - 1);
            let cell_idx = gj * WIDTH + gi;

            self.particle_next[idx] = self.cell_head[cell_idx];
            self.cell_head[cell_idx] = idx as i32;
        }
    }

    fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        let start = get_time();

        self.rebuild_spatial_hash();

        let gravity = 300.0;
        self.dem.update_with_water_level(
            &mut self.particles,
            &self.grid,
            &self.cell_head,
            &self.particle_next,
            dt,
            gravity,
            self.water_level,
        );

        let elapsed = (get_time() - start) as f32 * 1000.0;
        self.frame_times.push(elapsed);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(20, 25, 30, 255));

        // Draw water region (semi-transparent blue)
        let water_top_y = self.water_level * SCALE;
        let water_height = screen_height() - water_top_y;
        draw_rectangle(
            0.0,
            water_top_y,
            screen_width(),
            water_height,
            Color::from_rgba(30, 80, 140, 100),
        );

        // Draw water surface line
        draw_line(0.0, water_top_y, screen_width(), water_top_y, 2.0, Color::from_rgba(60, 120, 180, 200));

        // Draw grid cells (floor/walls)
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                if self.grid.cell_type[j * WIDTH + i] == CellType::Solid {
                    let x = i as f32 * CELL_SIZE * SCALE;
                    let y = j as f32 * CELL_SIZE * SCALE;
                    draw_rectangle(x, y, CELL_SIZE * SCALE, CELL_SIZE * SCALE, DARKGRAY);
                }
            }
        }

        // Draw particles
        let radius = CELL_SIZE * 0.4 * SCALE;
        for (idx, p) in self.particles.iter().enumerate() {
            let x = p.position.x * SCALE;
            let y = p.position.y * SCALE;

            let is_sleeping = self.dem.is_particle_sleeping(idx);
            let in_water = p.position.y > self.water_level;

            let base_color = match p.material {
                ParticleMaterial::Sand => Color::from_rgba(194, 178, 128, 255),
                ParticleMaterial::Gold => Color::from_rgba(255, 215, 0, 255),
                ParticleMaterial::Magnetite => Color::from_rgba(40, 40, 40, 255),
                _ => Color::from_rgba(100, 100, 100, 255),
            };

            // Dim if sleeping, tint blue if in water
            let color = if is_sleeping {
                Color::from_rgba(
                    (base_color.r * 255.0 * 0.5) as u8,
                    (base_color.g * 255.0 * 0.5) as u8,
                    (base_color.b * 255.0 * 0.6) as u8,
                    255,
                )
            } else if in_water {
                Color::from_rgba(
                    (base_color.r * 255.0 * 0.9) as u8,
                    (base_color.g * 255.0 * 0.9) as u8,
                    ((base_color.b * 255.0 * 0.9) as u8).saturating_add(30),
                    255,
                )
            } else {
                base_color
            };

            let size_mult = match p.material {
                ParticleMaterial::Gold => 0.7,
                ParticleMaterial::Magnetite => 1.2,
                _ => 1.0,
            };

            draw_circle(x, y, radius * size_mult, color);
        }

        self.draw_diagnostics();
    }

    fn draw_diagnostics(&self) {
        let total = self.particles.len();
        let sleeping = self.dem.sleeping_count();

        // Count particles in water vs air
        let mut in_water = 0;
        let mut in_air = 0;
        for p in self.particles.iter() {
            if p.position.y > self.water_level {
                in_water += 1;
            } else {
                in_air += 1;
            }
        }

        let avg_frame_ms = if !self.frame_times.is_empty() {
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
        } else {
            0.0
        };

        let y_start = 10.0;
        let line_height = 18.0;
        let x = 10.0;

        draw_text(&format!("Particles: {}", total), x, y_start, 18.0, WHITE);
        draw_text(&format!("In water: {}", in_water), x, y_start + line_height, 18.0, Color::from_rgba(100, 150, 255, 255));
        draw_text(&format!("In air: {}", in_air), x, y_start + line_height * 2.0, 18.0, WHITE);
        draw_text(&format!("Sleeping: {}", sleeping), x, y_start + line_height * 3.0, 18.0, GRAY);
        draw_text(&format!("DEM: {:.2} ms", avg_frame_ms), x, y_start + line_height * 4.0, 18.0,
            if avg_frame_ms < 5.0 { GREEN } else { RED });

        // Material info
        let mat_name = match self.spawn_material {
            ParticleMaterial::Sand => "SAND (density 2.65)",
            ParticleMaterial::Gold => "GOLD (density 19.3)",
            ParticleMaterial::Magnetite => "MAGNETITE (density 5.2)",
            _ => "OTHER",
        };
        draw_text(mat_name, screen_width() - 200.0, y_start, 16.0,
            match self.spawn_material {
                ParticleMaterial::Sand => Color::from_rgba(194, 178, 128, 255),
                ParticleMaterial::Gold => GOLD,
                ParticleMaterial::Magnetite => DARKGRAY,
                _ => WHITE,
            });

        // Controls
        let help_y = screen_height() - 80.0;
        draw_text("Click: spawn 25 particles above water", x, help_y, 14.0, GRAY);
        draw_text("1/2/3: Sand/Gold/Magnetite | M: mixed batch", x, help_y + 16.0, 14.0, GRAY);
        draw_text("R: reset | Space: pause", x, help_y + 32.0, 14.0, GRAY);

        if self.paused {
            draw_text("PAUSED", screen_width() / 2.0 - 40.0, 30.0, 30.0, RED);
        }
    }

    fn handle_input(&mut self) {
        if is_mouse_button_pressed(MouseButton::Left) {
            self.spawn_above_water(25);
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

        if is_key_pressed(KeyCode::R) {
            self.particles = Particles::new();
        }

        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut scene = WaterSettlingScene::new();

    // Spawn initial mixed batch
    scene.spawn_mixed_batch();

    loop {
        scene.handle_input();

        let dt = 1.0 / 60.0;
        scene.update(dt);
        scene.draw();

        next_frame().await;
    }
}
