//! DEM Test Scene - Dry Granular Simulation
//!
//! Visual test for the unified DEM with diagnostics.
//! Click to spawn particles, watch them settle into piles.

use macroquad::prelude::*;
use sim::dem::DemSimulation;
use sim::grid::{CellType, Grid};
use sim::particle::{ParticleMaterial, Particles};

const WIDTH: usize = 160;
const HEIGHT: usize = 120;
const CELL_SIZE: f32 = 5.0;
const SCALE: f32 = 1.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "DEM Test - Dry Granular".to_owned(),
        window_width: (WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

struct DemTestScene {
    grid: Grid,
    particles: Particles,
    dem: DemSimulation,
    cell_head: Vec<i32>,
    particle_next: Vec<i32>,
    paused: bool,
    show_sleeping: bool,
    spawn_material: ParticleMaterial,
    frame_times: Vec<f32>,
    continuous_spawn: bool,
    spawn_timer: f32,
}

impl DemTestScene {
    fn new() -> Self {
        let mut grid = Grid::new(WIDTH, HEIGHT, CELL_SIZE);

        // Create a box with floor and walls
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                // Floor (bottom 10%)
                if j >= HEIGHT - HEIGHT / 10 {
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

        // Compute SDF for collision
        grid.compute_sdf();

        let particles = Particles::new();
        let dem = DemSimulation::new();

        Self {
            grid,
            particles,
            dem,
            cell_head: vec![-1; WIDTH * HEIGHT],
            particle_next: Vec::new(),
            paused: false,
            show_sleeping: true,
            spawn_material: ParticleMaterial::Sand,
            frame_times: Vec::with_capacity(60),
            continuous_spawn: false,
            spawn_timer: 0.0,
        }
    }

    fn spawn_stream(&mut self, x: f32, count: usize) {
        let y = CELL_SIZE * 3.0;
        let spacing = CELL_SIZE * 0.7;

        for i in 0..count {
            let px = x + (i as f32 - count as f32 / 2.0) * spacing;
            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;
            let jitter_y = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;

            match self.spawn_material {
                ParticleMaterial::Sand => self.particles.spawn_sand(px + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Gold => self.particles.spawn_gold(px + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Magnetite => self.particles.spawn_magnetite(px + jitter_x, y + jitter_y, 0.0, 0.0),
                _ => self.particles.spawn_sand(px + jitter_x, y + jitter_y, 0.0, 0.0),
            }
        }
    }

    fn spawn_particles_at(&mut self, screen_x: f32, screen_y: f32, count: usize) {
        let sim_x = screen_x / SCALE;
        let sim_y = screen_y / SCALE;

        let spacing = CELL_SIZE * 0.8;
        let cols = (count as f32).sqrt().ceil() as usize;

        for i in 0..count {
            let col = i % cols;
            let row = i / cols;
            let x = sim_x + (col as f32 - cols as f32 / 2.0) * spacing;
            let y = sim_y + row as f32 * spacing;

            // Check bounds
            if x < CELL_SIZE * 3.0 || x > (WIDTH as f32 - 3.0) * CELL_SIZE {
                continue;
            }
            if y < CELL_SIZE * 2.0 || y > (HEIGHT as f32 - HEIGHT as f32 / 10.0 - 2.0) * CELL_SIZE {
                continue;
            }

            // Add small random offset to prevent perfect stacking
            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;
            let jitter_y = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.3;

            let px = x + jitter_x;
            let py = y + jitter_y;

            match self.spawn_material {
                ParticleMaterial::Sand => self.particles.spawn_sand(px, py, 0.0, 0.0),
                ParticleMaterial::Gold => self.particles.spawn_gold(px, py, 0.0, 0.0),
                ParticleMaterial::Magnetite => self.particles.spawn_magnetite(px, py, 0.0, 0.0),
                _ => self.particles.spawn_sand(px, py, 0.0, 0.0),
            }
        }
    }

    fn spawn_column(&mut self, x: f32, height: usize) {
        let start_y = CELL_SIZE * 5.0;
        let spacing = CELL_SIZE * 0.9;

        for i in 0..height {
            let y = start_y + i as f32 * spacing;
            let jitter = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.2;

            let px = x + jitter;

            match self.spawn_material {
                ParticleMaterial::Sand => self.particles.spawn_sand(px, y, 0.0, 0.0),
                ParticleMaterial::Gold => self.particles.spawn_gold(px, y, 0.0, 0.0),
                ParticleMaterial::Magnetite => self.particles.spawn_magnetite(px, y, 0.0, 0.0),
                _ => self.particles.spawn_sand(px, y, 0.0, 0.0),
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

        // Continuous spawning
        if self.continuous_spawn {
            self.spawn_timer += dt;
            if self.spawn_timer >= 0.05 { // 20 particles/sec rate
                self.spawn_timer = 0.0;
                let center_x = WIDTH as f32 * CELL_SIZE / 2.0;
                self.spawn_stream(center_x, 5);
            }
        }

        let start = get_time();

        // Rebuild spatial hash
        self.rebuild_spatial_hash();

        // Run DEM physics (dry mode = no water)
        let gravity = 300.0; // pixels/s^2
        self.dem.update(
            &mut self.particles,
            &self.grid,
            &self.cell_head,
            &self.particle_next,
            dt,
            gravity,
            false, // in_water = false (dry simulation)
        );

        let elapsed = (get_time() - start) as f32 * 1000.0;
        self.frame_times.push(elapsed);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(20, 25, 30, 255));

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

            let base_color = match p.material {
                ParticleMaterial::Sand => Color::from_rgba(194, 178, 128, 255),
                ParticleMaterial::Gold => Color::from_rgba(255, 215, 0, 255),
                ParticleMaterial::Magnetite => Color::from_rgba(40, 40, 40, 255),
                _ => Color::from_rgba(100, 100, 100, 255),
            };

            let color = if self.show_sleeping && is_sleeping {
                // Dim sleeping particles
                Color::from_rgba(
                    (base_color.r * 255.0 * 0.5) as u8,
                    (base_color.g * 255.0 * 0.5) as u8,
                    (base_color.b * 255.0 * 0.5) as u8,
                    255,
                )
            } else {
                base_color
            };

            // Size based on material
            let size_mult = match p.material {
                ParticleMaterial::Gold => 0.7,
                ParticleMaterial::Magnetite => 1.2,
                _ => 1.0,
            };

            draw_circle(x, y, radius * size_mult, color);
        }

        // Draw diagnostics
        self.draw_diagnostics();
    }

    fn draw_diagnostics(&self) {
        let sleeping = self.dem.sleeping_count();
        let total = self.particles.len();
        let awake = total.saturating_sub(sleeping);

        // Compute average velocity of awake particles
        let (avg_vel, max_vel) = if awake > 0 {
            let mut sum = 0.0f32;
            let mut max = 0.0f32;
            for (idx, p) in self.particles.iter().enumerate() {
                if !self.dem.is_particle_sleeping(idx) {
                    let speed = p.velocity.length();
                    sum += speed;
                    max = max.max(speed);
                }
            }
            (sum / awake as f32, max)
        } else {
            (0.0, 0.0)
        };

        // Compute average frame time
        let avg_frame_ms = if !self.frame_times.is_empty() {
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
        } else {
            0.0
        };

        let y_start = 10.0;
        let line_height = 20.0;
        let x = 10.0;

        draw_text(&format!("Particles: {}", total), x, y_start, 20.0, WHITE);
        draw_text(&format!("Sleeping: {} ({:.0}%)", sleeping,
            if total > 0 { sleeping as f32 / total as f32 * 100.0 } else { 0.0 }),
            x, y_start + line_height, 20.0, GRAY);
        draw_text(&format!("Awake: {}", awake), x, y_start + line_height * 2.0, 20.0, YELLOW);
        draw_text(&format!("Avg vel: {:.1} px/s", avg_vel), x, y_start + line_height * 3.0, 20.0,
            if avg_vel < 5.0 { GREEN } else { ORANGE });
        draw_text(&format!("Max vel: {:.1} px/s", max_vel), x, y_start + line_height * 4.0, 20.0, WHITE);
        draw_text(&format!("DEM: {:.2} ms", avg_frame_ms), x, y_start + line_height * 5.0, 20.0,
            if avg_frame_ms < 5.0 { GREEN } else { RED });

        // Controls help
        let help_y = screen_height() - 100.0;
        draw_text("Click: spawn 25 particles", x, help_y, 16.0, GRAY);
        draw_text("1/2/3: Sand/Gold/Magnetite", x, help_y + 18.0, 16.0, GRAY);
        draw_text("C: spawn column | R: reset | Space: pause", x, help_y + 36.0, 16.0, GRAY);
        draw_text("S: toggle sleep viz | F: continuous flow", x, help_y + 54.0, 16.0, GRAY);

        // Continuous spawn indicator
        if self.continuous_spawn {
            draw_text("FLOWING", screen_width() - 100.0, help_y + 72.0, 20.0, GREEN);
        }

        // Current material indicator
        let mat_name = match self.spawn_material {
            ParticleMaterial::Sand => "SAND",
            ParticleMaterial::Gold => "GOLD",
            ParticleMaterial::Magnetite => "MAGNETITE",
            _ => "OTHER",
        };
        draw_text(&format!("Material: {}", mat_name), screen_width() - 150.0, y_start, 20.0,
            match self.spawn_material {
                ParticleMaterial::Sand => Color::from_rgba(194, 178, 128, 255),
                ParticleMaterial::Gold => GOLD,
                ParticleMaterial::Magnetite => DARKGRAY,
                _ => WHITE,
            });

        if self.paused {
            draw_text("PAUSED", screen_width() / 2.0 - 40.0, 30.0, 30.0, RED);
        }
    }

    fn handle_input(&mut self) {
        // Spawn particles on click
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            self.spawn_particles_at(mx, my, 25);
        }

        // Spawn column
        if is_key_pressed(KeyCode::C) {
            let center_x = WIDTH as f32 * CELL_SIZE / 2.0;
            self.spawn_column(center_x, 30);
        }

        // Material selection
        if is_key_pressed(KeyCode::Key1) {
            self.spawn_material = ParticleMaterial::Sand;
        }
        if is_key_pressed(KeyCode::Key2) {
            self.spawn_material = ParticleMaterial::Gold;
        }
        if is_key_pressed(KeyCode::Key3) {
            self.spawn_material = ParticleMaterial::Magnetite;
        }

        // Reset
        if is_key_pressed(KeyCode::R) {
            self.particles = Particles::new();
        }

        // Pause
        if is_key_pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }

        // Toggle sleep visualization
        if is_key_pressed(KeyCode::S) {
            self.show_sleeping = !self.show_sleeping;
        }

        // Toggle continuous spawn
        if is_key_pressed(KeyCode::F) {
            self.continuous_spawn = !self.continuous_spawn;
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut scene = DemTestScene::new();

    // Spawn initial column - taller for more dramatic flow
    scene.spawn_column(WIDTH as f32 * CELL_SIZE / 2.0, 60);

    loop {
        scene.handle_input();

        let dt = 1.0 / 60.0; // Fixed timestep
        scene.update(dt);
        scene.draw();

        next_frame().await;
    }
}
