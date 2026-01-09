//! Vortex Settling Test - Sediment in flowing water with vortices
//!
//! Tests sediment behavior in water with flow patterns.
//! Shows how gold settles in vortex traps while sand gets carried.

use macroquad::prelude::*;
use sim::dem::DemSimulation;
use sim::grid::{CellType, Grid};
use sim::particle::{ParticleMaterial, Particles};
use glam::Vec2;

const WIDTH: usize = 160;
const HEIGHT: usize = 100;
const CELL_SIZE: f32 = 5.0;
const SCALE: f32 = 1.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "Vortex Settling Test - Flow + Sediment".to_owned(),
        window_width: (WIDTH as f32 * CELL_SIZE * SCALE) as i32,
        window_height: (HEIGHT as f32 * CELL_SIZE * SCALE) as i32,
        ..Default::default()
    }
}

/// Simple flow field with horizontal flow and vortex behind riffle
struct FlowField {
    /// Base horizontal flow speed
    base_flow: f32,
    /// Riffle X positions (create vortices behind these)
    riffle_x: Vec<f32>,
    /// Riffle height (from floor)
    riffle_height: f32,
    /// Water surface Y
    water_level: f32,
    /// Floor Y
    floor_y: f32,
}

impl FlowField {
    fn new(water_level: f32, floor_y: f32) -> Self {
        // Place riffles at 1/3 and 2/3 of width
        let riffle_x = vec![
            WIDTH as f32 * CELL_SIZE * 0.33,
            WIDTH as f32 * CELL_SIZE * 0.66,
        ];

        Self {
            base_flow: 80.0, // pixels/sec to the right
            riffle_x,
            riffle_height: CELL_SIZE * 4.0,
            water_level,
            floor_y,
        }
    }

    /// Get flow velocity at a position
    fn velocity_at(&self, pos: Vec2) -> Vec2 {
        // Not in water = no flow
        if pos.y < self.water_level {
            return Vec2::ZERO;
        }

        let mut vel = Vec2::new(self.base_flow, 0.0);

        // Check for vortex influence from each riffle
        for &rx in &self.riffle_x {
            // Vortex center is behind and above the riffle
            let vortex_center = Vec2::new(rx + CELL_SIZE * 3.0, self.floor_y - self.riffle_height * 0.7);
            let to_center = vortex_center - pos;
            let dist = to_center.length();

            // Vortex influence radius
            let vortex_radius = CELL_SIZE * 6.0;

            if dist < vortex_radius && dist > 1.0 {
                // Circular flow around vortex center (clockwise)
                let tangent = Vec2::new(-to_center.y, to_center.x).normalize();
                let strength = (1.0 - dist / vortex_radius) * 60.0;
                vel += tangent * strength;

                // Slight inward pull (trapping effect)
                let inward = to_center.normalize() * strength * 0.2;
                vel += inward;
            }

            // Slow down flow directly behind riffle (wake region)
            let behind_riffle = pos.x > rx && pos.x < rx + CELL_SIZE * 8.0;
            let near_floor = pos.y > self.floor_y - self.riffle_height * 2.0;
            if behind_riffle && near_floor {
                vel.x *= 0.3; // Reduced flow in wake
            }
        }

        // Reduce flow near floor (boundary layer)
        let floor_dist = self.floor_y - pos.y;
        if floor_dist < CELL_SIZE * 3.0 && floor_dist > 0.0 {
            let factor = floor_dist / (CELL_SIZE * 3.0);
            vel *= factor;
        }

        vel
    }
}

struct VortexSettlingScene {
    grid: Grid,
    particles: Particles,
    dem: DemSimulation,
    cell_head: Vec<i32>,
    particle_next: Vec<i32>,
    paused: bool,
    water_level: f32,
    floor_y: f32,
    flow_field: FlowField,
    spawn_material: ParticleMaterial,
    frame_times: Vec<f32>,
    continuous_spawn: bool,
    spawn_timer: f32,
    show_flow: bool,
}

impl VortexSettlingScene {
    fn new() -> Self {
        let mut grid = Grid::new(WIDTH, HEIGHT, CELL_SIZE);

        let floor_row = HEIGHT - HEIGHT / 10;
        let floor_y = floor_row as f32 * CELL_SIZE;

        // Create sluice-like container
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                // Floor
                if j >= floor_row {
                    grid.cell_type[j * WIDTH + i] = CellType::Solid;
                }
                // Left wall (partial - inlet)
                if i < 2 && j > HEIGHT / 3 {
                    grid.cell_type[j * WIDTH + i] = CellType::Solid;
                }
                // Right wall (partial - outlet)
                if i >= WIDTH - 2 && j > HEIGHT / 3 {
                    grid.cell_type[j * WIDTH + i] = CellType::Solid;
                }
            }
        }

        // Add riffles on floor
        let riffle_positions = [
            (WIDTH as f32 * 0.33) as usize,
            (WIDTH as f32 * 0.66) as usize,
        ];
        for &rx in &riffle_positions {
            for di in 0..3 {
                let i = rx + di;
                if i < WIDTH {
                    // Riffle is 3-4 cells high
                    for dj in 1..=4 {
                        let j = floor_row - dj;
                        if j < HEIGHT {
                            grid.cell_type[j * WIDTH + i] = CellType::Solid;
                        }
                    }
                }
            }
        }

        grid.compute_sdf();

        let particles = Particles::new();
        let dem = DemSimulation::new();

        // Water fills most of the container
        let water_level = HEIGHT as f32 * CELL_SIZE * 0.25;
        let flow_field = FlowField::new(water_level, floor_y);

        Self {
            grid,
            particles,
            dem,
            cell_head: vec![-1; WIDTH * HEIGHT],
            particle_next: Vec::new(),
            paused: false,
            water_level,
            floor_y,
            flow_field,
            spawn_material: ParticleMaterial::Sand,
            frame_times: Vec::with_capacity(60),
            continuous_spawn: false,
            spawn_timer: 0.0,
            show_flow: true,
        }
    }

    fn spawn_at_inlet(&mut self, count: usize) {
        let spawn_x = CELL_SIZE * 8.0;
        let spawn_y = self.water_level + CELL_SIZE * 5.0; // Just below surface
        let spacing = CELL_SIZE * 0.8;

        for i in 0..count {
            let y = spawn_y + (i as f32) * spacing * 0.5;
            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * spacing;
            let jitter_y = (rand::gen_range(0.0, 1.0) - 0.5) * spacing * 0.5;

            match self.spawn_material {
                ParticleMaterial::Sand => self.particles.spawn_sand(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Gold => self.particles.spawn_gold(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
                ParticleMaterial::Magnetite => self.particles.spawn_magnetite(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
                _ => self.particles.spawn_sand(spawn_x + jitter_x, y + jitter_y, 0.0, 0.0),
            }
        }
    }

    fn spawn_mixed_stream(&mut self) {
        let spawn_x = CELL_SIZE * 8.0;
        let spawn_y = self.water_level + CELL_SIZE * 3.0;

        for i in 0..5 {
            let jitter_x = (rand::gen_range(0.0, 1.0) - 0.5) * CELL_SIZE;
            let jitter_y = i as f32 * CELL_SIZE * 0.4;

            // 1 in 5 is gold
            if rand::gen_range(0, 5) == 0 {
                self.particles.spawn_gold(spawn_x + jitter_x, spawn_y + jitter_y, 0.0, 0.0);
            } else {
                self.particles.spawn_sand(spawn_x + jitter_x, spawn_y + jitter_y, 0.0, 0.0);
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

    fn apply_flow_forces(&mut self, dt: f32) {
        // Apply water flow drag to particles in water
        let drag_coeff = 5.0; // How strongly particles follow flow

        for p in self.particles.list.iter_mut() {
            if !p.is_sediment() {
                continue;
            }

            // Only apply in water
            if p.position.y <= self.water_level {
                continue;
            }

            let flow_vel = self.flow_field.velocity_at(p.position);
            let vel_diff = flow_vel - p.velocity;

            // Drag force proportional to velocity difference
            // Lighter particles (lower density) are affected more
            let density = p.material.density();
            let drag_factor = drag_coeff / density;

            p.velocity += vel_diff * drag_factor * dt;
        }
    }

    fn remove_escaped_particles(&mut self) {
        // Remove particles that escaped through outlet
        let max_x = WIDTH as f32 * CELL_SIZE - CELL_SIZE;
        self.particles.list.retain(|p| p.position.x < max_x);
    }

    fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        // Continuous spawning
        if self.continuous_spawn {
            self.spawn_timer += dt;
            if self.spawn_timer >= 0.15 {
                self.spawn_timer = 0.0;
                self.spawn_mixed_stream();
            }
        }

        let start = get_time();

        self.rebuild_spatial_hash();

        // Apply flow forces before DEM
        self.apply_flow_forces(dt);

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

        self.remove_escaped_particles();

        let elapsed = (get_time() - start) as f32 * 1000.0;
        self.frame_times.push(elapsed);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(20, 25, 30, 255));

        // Draw water region
        let water_top_y = self.water_level * SCALE;
        draw_rectangle(
            0.0,
            water_top_y,
            screen_width(),
            screen_height() - water_top_y,
            Color::from_rgba(30, 80, 140, 80),
        );

        // Draw flow vectors if enabled
        if self.show_flow {
            self.draw_flow_field();
        }

        // Draw water surface
        draw_line(0.0, water_top_y, screen_width(), water_top_y, 2.0, Color::from_rgba(60, 120, 180, 200));

        // Draw grid cells (floor/walls/riffles)
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
        for p in self.particles.iter() {
            let x = p.position.x * SCALE;
            let y = p.position.y * SCALE;

            let base_color = match p.material {
                ParticleMaterial::Sand => Color::from_rgba(194, 178, 128, 255),
                ParticleMaterial::Gold => Color::from_rgba(255, 215, 0, 255),
                ParticleMaterial::Magnetite => Color::from_rgba(40, 40, 40, 255),
                _ => Color::from_rgba(100, 100, 100, 255),
            };

            let size_mult = match p.material {
                ParticleMaterial::Gold => 0.7,
                ParticleMaterial::Magnetite => 1.2,
                _ => 1.0,
            };

            draw_circle(x, y, radius * size_mult, base_color);
        }

        self.draw_diagnostics();
    }

    fn draw_flow_field(&self) {
        let step = CELL_SIZE * 4.0;
        let arrow_scale = 0.3;

        for gx in (0..WIDTH).step_by(4) {
            for gy in (0..HEIGHT).step_by(4) {
                let pos = Vec2::new(
                    gx as f32 * CELL_SIZE + CELL_SIZE * 2.0,
                    gy as f32 * CELL_SIZE + CELL_SIZE * 2.0,
                );

                // Only draw in water
                if pos.y <= self.water_level {
                    continue;
                }

                let vel = self.flow_field.velocity_at(pos);
                let speed = vel.length();

                if speed > 1.0 {
                    let x1 = pos.x * SCALE;
                    let y1 = pos.y * SCALE;
                    let x2 = x1 + vel.x * arrow_scale;
                    let y2 = y1 + vel.y * arrow_scale;

                    let alpha = (speed / 100.0).min(1.0);
                    let color = Color::from_rgba(100, 180, 255, (alpha * 150.0) as u8);

                    draw_line(x1, y1, x2, y2, 1.5, color);
                }
            }
        }
    }

    fn draw_diagnostics(&self) {
        let total = self.particles.len();

        // Count by material
        let mut sand_count = 0;
        let mut gold_count = 0;
        for p in self.particles.iter() {
            match p.material {
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
        let line_height = 18.0;
        let x = 10.0;

        draw_text(&format!("Particles: {}", total), x, y_start, 18.0, WHITE);
        draw_text(&format!("Sand: {}", sand_count), x, y_start + line_height, 18.0, Color::from_rgba(194, 178, 128, 255));
        draw_text(&format!("Gold: {}", gold_count), x, y_start + line_height * 2.0, 18.0, GOLD);
        draw_text(&format!("DEM: {:.2} ms", avg_frame_ms), x, y_start + line_height * 3.0, 18.0,
            if avg_frame_ms < 5.0 { GREEN } else { RED });

        // Controls
        let help_y = screen_height() - 90.0;
        draw_text("Click: spawn 10 at inlet | M: mixed batch", x, help_y, 14.0, GRAY);
        draw_text("1/2/3: Sand/Gold/Magnetite", x, help_y + 16.0, 14.0, GRAY);
        draw_text("F: continuous flow | V: toggle flow viz", x, help_y + 32.0, 14.0, GRAY);
        draw_text("R: reset | Space: pause", x, help_y + 48.0, 14.0, GRAY);

        if self.continuous_spawn {
            draw_text("FLOWING", screen_width() - 80.0, y_start, 18.0, GREEN);
        }

        if self.paused {
            draw_text("PAUSED", screen_width() / 2.0 - 40.0, 30.0, 30.0, RED);
        }
    }

    fn handle_input(&mut self) {
        if is_mouse_button_pressed(MouseButton::Left) {
            self.spawn_at_inlet(10);
        }

        if is_key_pressed(KeyCode::M) {
            self.spawn_mixed_stream();
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

        if is_key_pressed(KeyCode::V) {
            self.show_flow = !self.show_flow;
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
    let mut scene = VortexSettlingScene::new();

    loop {
        scene.handle_input();

        let dt = 1.0 / 60.0;
        scene.update(dt);
        scene.draw();

        next_frame().await;
    }
}
