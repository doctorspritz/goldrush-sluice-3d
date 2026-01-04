//! Visual stage runner for sediment diagnostics.
//! Run with: cargo run --example sediment_stages_visual -p game --release

mod gpu {
    pub use game::gpu::*;
}

use gpu::{dem::GpuDemSolver, renderer::ParticleRenderer, GpuContext};
use sim::particle::{Particle, ParticleMaterial};
use sim::stages::{
    stage_clump_drop, stage_dry_gold_stream, stage_dry_mixed_stream, stage_dry_sand_stream,
    stage_sand_then_gold, stage_sediment_water_dem, stage_sediment_water_no_dem,
    stage_two_way_coupling, stage_water_sluice, StageMode, StageSpec,
};
use sim::FlipSimulation;
use glam::Vec2;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const DT: f32 = sim::stages::STAGE_DT;
const TARGET_WIDTH: u32 = 1920;
const TARGET_HEIGHT: u32 = 1080;
const DRY_SAND_SPAWN_MULTIPLIER: usize = 1;  // Spawn 1 particle at a time
const DRY_SAND_STREAM_INTERVAL: usize = 3;   // Every 3 frames (~20 particles/sec)
const DRY_SAND_STAGE: &str = "dry_sand_stream";

fn zoom_for_width(stage: &StageSpec, view_width: u32) -> f32 {
    let sim_width = stage.width as f32 * stage.cell_size;
    let width = view_width.max(1) as f32;
    width / sim_width.max(1.0)
}

/// Material mode for Level 4 testing
#[derive(Clone, Copy, PartialEq)]
enum MaterialMode {
    Sand,
    Gold,
    Mixed, // Original behavior
}

struct App {
    gpu: Option<GpuContext>,
    renderer: Option<ParticleRenderer>,
    dem_solver: Option<GpuDemSolver>,
    window: Option<Arc<Window>>,
    sim: FlipSimulation,
    stage: StageSpec,
    paused: bool,
    flow_enabled: bool,
    zoom: f32,
    camera_offset: (f32, f32),
    spawn_multiplier: usize,
    spawn_interval: usize,
    spawn_frame: usize,
    frame: usize,
    /// For Level 4 testing: allows switching material without resetting
    material_mode: MaterialMode,
    /// Mouse position in screen coordinates
    mouse_pos: (f32, f32),
    /// Mouse position in simulation coordinates
    mouse_sim_pos: Vec2,
    /// Left mouse button held
    mouse_left_held: bool,
    /// Right mouse button held
    mouse_right_held: bool,
}

impl App {
    fn new() -> Self {
        let stage = stage_dry_gold_stream();
        let mut sim = FlipSimulation::new(stage.width, stage.height, stage.cell_size);
        (stage.init)(&mut sim);
        let zoom = zoom_for_width(&stage, TARGET_WIDTH);
        let spawn_multiplier = if stage.name == DRY_SAND_STAGE {
            DRY_SAND_SPAWN_MULTIPLIER
        } else {
            1
        };
        let spawn_interval = if stage.name == DRY_SAND_STAGE {
            DRY_SAND_STREAM_INTERVAL
        } else {
            1
        };

        println!("Stage: {}", stage.name);
        println!("Controls: 1-9 switch stage, SPACE pause, F toggle flow, R reset, +/- zoom");
        println!("Level 4 test: G=switch to gold, T=switch to sand, M=mixed mode");
        println!("Mouse: Left=add particles, Right=remove particles");

        Self {
            gpu: None,
            renderer: None,
            dem_solver: None,
            window: None,
            sim,
            stage,
            paused: false,
            flow_enabled: true,
            zoom,
            camera_offset: (0.0, 0.0),
            spawn_multiplier,
            spawn_interval,
            spawn_frame: 0,
            frame: 0,
            material_mode: MaterialMode::Mixed, // Default to original behavior
            mouse_pos: (0.0, 0.0),
            mouse_sim_pos: Vec2::ZERO,
            mouse_left_held: false,
            mouse_right_held: false,
        }
    }

    /// Spawn particles at the stream location based on current material_mode
    fn spawn_particles_by_mode(&mut self) {
        let center_x = (self.stage.width as f32 / 2.0) * self.stage.cell_size;
        let drop_y = 20.0; // Near top
        let spacing = self.stage.cell_size * 0.7;

        let material = match self.material_mode {
            MaterialMode::Sand => ParticleMaterial::Sand,
            MaterialMode::Gold => ParticleMaterial::Gold,
            MaterialMode::Mixed => {
                // Let the stage's per_frame handle mixed spawning
                return;
            }
        };

        // Spawn 3x2 cluster
        for row in 0..2 {
            for col in 0..3 {
                let x = center_x + (col as f32 - 1.0) * spacing;
                let y = drop_y + row as f32 * spacing;
                self.sim.particles.list.push(Particle::new(
                    Vec2::new(x, y),
                    Vec2::ZERO,
                    material,
                ));
            }
        }
    }

    /// Spawn particles at the current mouse cursor position (silent, for held mouse)
    fn spawn_at_cursor_silent(&mut self) {
        let material = match self.material_mode {
            MaterialMode::Sand => ParticleMaterial::Sand,
            MaterialMode::Gold => ParticleMaterial::Gold,
            MaterialMode::Mixed => ParticleMaterial::Sand,
        };
        // Spawn small cluster at cursor
        let spacing = self.stage.cell_size * 0.5;
        for row in 0..2 {
            for col in 0..2 {
                let pos = self.mouse_sim_pos + Vec2::new(
                    (col as f32 - 0.5) * spacing,
                    (row as f32 - 0.5) * spacing,
                );
                self.sim.particles.list.push(Particle::new(pos, Vec2::ZERO, material));
            }
        }
    }

    /// Remove particles near the current mouse cursor position (silent, for held mouse)
    fn remove_near_cursor_silent(&mut self) {
        let radius = self.stage.cell_size * 3.0;
        self.sim.particles.list.retain(|p| {
            (p.position - self.mouse_sim_pos).length() > radius
        });
    }

    fn update_camera(&mut self, view_width: u32, view_height: u32) {
        let sim_height = self.stage.height as f32 * self.stage.cell_size;
        let view_h = view_height.max(1) as f32 / self.zoom.max(0.0001);
        let offset_y = if sim_height > view_h {
            sim_height - view_h
        } else {
            0.0
        };
        self.camera_offset = (0.0, offset_y);
    }

    fn update_camera_from_gpu(&mut self) {
        if let Some(gpu) = &self.gpu {
            self.update_camera(gpu.size.0, gpu.size.1);
        }
    }

    fn reset_stage(&mut self, stage: StageSpec) {
        let mut sim = FlipSimulation::new(stage.width, stage.height, stage.cell_size);
        (stage.init)(&mut sim);
        self.sim = sim;
        self.stage = stage;
        let fallback_zoom = zoom_for_width(&self.stage, TARGET_WIDTH);
        self.zoom = self
            .gpu
            .as_ref()
            .map(|gpu| zoom_for_width(&self.stage, gpu.size.0))
            .unwrap_or(fallback_zoom);
        self.spawn_multiplier = if self.stage.name == DRY_SAND_STAGE {
            DRY_SAND_SPAWN_MULTIPLIER
        } else {
            1
        };
        self.spawn_interval = if self.stage.name == DRY_SAND_STAGE {
            DRY_SAND_STREAM_INTERVAL
        } else {
            1
        };
        self.spawn_frame = 0;
        self.flow_enabled = true;
        self.material_mode = MaterialMode::Mixed; // Reset to default
        self.update_camera_from_gpu();
        if let Some(renderer) = &mut self.renderer {
            renderer.invalidate_terrain();
        }
        // Recreate DEM solver for new dimensions
        if let Some(gpu) = &self.gpu {
            self.dem_solver = Some(GpuDemSolver::new(
                gpu,
                self.stage.width as u32,
                self.stage.height as u32,
                100_000,
            ));
        }
        self.frame = 0;
        println!("Stage: {}", self.stage.name);
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }

        // Mouse-held spawning/removing (every 3rd frame to avoid spam)
        if self.frame % 3 == 0 {
            if self.mouse_left_held {
                self.spawn_at_cursor_silent();
            }
            if self.mouse_right_held {
                self.remove_near_cursor_silent();
            }
        }

        let is_dry_sand = self.stage.name == DRY_SAND_STAGE;
        let should_spawn = self.flow_enabled
            && (!is_dry_sand || (self.frame % self.spawn_interval == 0));
        if should_spawn {
            // Use material_mode if not Mixed, otherwise use stage's per_frame
            if self.material_mode != MaterialMode::Mixed {
                self.spawn_particles_by_mode();
            } else {
                let repeats = if is_dry_sand {
                    self.spawn_multiplier.max(1)
                } else {
                    1
                };
                for _ in 0..repeats {
                    let spawn_frame = if is_dry_sand { self.spawn_frame } else { self.frame };
                    (self.stage.per_frame)(&mut self.sim, spawn_frame);
                }
                if is_dry_sand {
                    self.spawn_frame = self.spawn_frame.wrapping_add(1);
                }
            }
        }
        match self.stage.mode {
            StageMode::Dry => {
                // Use GPU DEM for dry stages
                if let (Some(gpu), Some(dem)) = (&self.gpu, &mut self.dem_solver) {
                    // Compute and upload SDF
                    self.sim.grid.compute_sdf();
                    dem.upload_sdf(gpu, &self.sim.grid.sdf);
                    // Execute GPU DEM with no water (-1.0)
                    dem.execute(
                        gpu,
                        &mut self.sim.particles,
                        self.stage.cell_size,
                        DT,
                        sim::physics::GRAVITY,
                        -1.0, // No water level for dry stages
                    );
                    // Debug: print static state info every 60 frames
                    if self.frame % 60 == 0 && !self.sim.particles.list.is_empty() {
                        let particle_count = self.sim.particles.list.len();
                        // Download static states to check how many are static
                        let static_states = dem.download_static_states_headless(&gpu.device, &gpu.queue, particle_count);
                        let static_count = static_states.iter().filter(|&&s| s == 1).count();
                        let static_pct = if particle_count > 0 { static_count * 100 / particle_count } else { 0 };

                        let avg_vel: f32 = self.sim.particles.iter()
                            .map(|p| p.velocity.length())
                            .sum::<f32>() / particle_count.max(1) as f32;

                        println!(
                            "STATIC: frame={} particles={} static={}/{}({}%) avg_vel={:.2} mode={:?}",
                            self.frame, particle_count, static_count, particle_count, static_pct, avg_vel,
                            match self.material_mode {
                                MaterialMode::Sand => "Sand",
                                MaterialMode::Gold => "Gold",
                                MaterialMode::Mixed => "Mixed",
                            }
                        );

                        // Density diagnostic: check particle spacing near floor
                        let floor_y = (self.stage.height as f32 - 3.0) * self.stage.cell_size;
                        let radius = 0.8 * 0.35 * self.stage.cell_size;  // Expected particle radius
                        let ideal_spacing = radius * 2.0;  // Particles touching but not overlapping

                        // Find particles near floor (within 10 cell heights)
                        let floor_zone_height = 10.0 * self.stage.cell_size;
                        let near_floor: Vec<&Particle> = self.sim.particles.iter()
                            .filter(|p| p.position.y > floor_y - floor_zone_height)
                            .collect();

                        if near_floor.len() >= 2 {
                            // Compute average nearest-neighbor distance
                            let mut total_min_dist = 0.0;
                            let mut count = 0;
                            for (i, p1) in near_floor.iter().enumerate() {
                                let mut min_dist = f32::MAX;
                                for (j, p2) in near_floor.iter().enumerate() {
                                    if i != j {
                                        let dist = (p1.position - p2.position).length();
                                        min_dist = min_dist.min(dist);
                                    }
                                }
                                if min_dist < f32::MAX {
                                    total_min_dist += min_dist;
                                    count += 1;
                                }
                            }
                            let avg_min_dist = if count > 0 { total_min_dist / count as f32 } else { 0.0 };
                            let compression_ratio = avg_min_dist / ideal_spacing;

                            // Count severely compressed particles (overlapping by >20%)
                            let mut severely_compressed = 0;
                            for (i, p1) in near_floor.iter().enumerate() {
                                for (j, p2) in near_floor.iter().enumerate() {
                                    if i < j {
                                        let dist = (p1.position - p2.position).length();
                                        if dist < ideal_spacing * 0.8 {
                                            severely_compressed += 1;
                                        }
                                    }
                                }
                            }

                            println!(
                                "  FLOOR_DENSITY: near_floor={} avg_nn_dist={:.2} ideal={:.2} compression={:.0}% severely_compressed_pairs={}",
                                near_floor.len(), avg_min_dist, ideal_spacing, compression_ratio * 100.0, severely_compressed
                            );
                        }
                    }

                    // Remove out-of-bounds particles
                    let w = self.stage.width as f32 * self.stage.cell_size;
                    let h = self.stage.height as f32 * self.stage.cell_size;
                    self.sim.particles.remove_out_of_bounds(w, h);
                } else {
                    // Fallback if GPU not ready
                    self.sim.update_dry(DT);
                }
            }
            StageMode::Full => self.sim.update(DT),
        }
        self.frame = self.frame.wrapping_add(1);
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(renderer) = &mut self.renderer else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render"),
        });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.06,
                            g: 0.06,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        renderer.draw_terrain_with_offset(
            gpu,
            &mut encoder,
            &view,
            &self.sim.grid,
            self.stage.cell_size,
            self.zoom,
            self.camera_offset,
        );
        renderer.draw_with_offset(
            gpu,
            &mut encoder,
            &view,
            &self.sim.particles,
            self.zoom,
            self.stage.cell_size * self.zoom * 1.5,
            self.camera_offset,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        if !pressed {
            return;
        }

        match key {
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::KeyF => {
                self.flow_enabled = !self.flow_enabled;
                println!("Flow: {}", if self.flow_enabled { "ON" } else { "OFF" });
            }
            KeyCode::KeyR => self.reset_stage(self.stage),
            KeyCode::Equal => self.zoom = (self.zoom + 0.5).min(8.0),
            KeyCode::Minus => self.zoom = (self.zoom - 0.5).max(1.0),
            KeyCode::KeyD => {
                if self.stage.name == DRY_SAND_STAGE {
                    self.spawn_multiplier = if self.spawn_multiplier == 1 {
                        DRY_SAND_SPAWN_MULTIPLIER
                    } else {
                        1
                    };
                    println!("Dry sand spawn multiplier: {}", self.spawn_multiplier);
                }
            }
            KeyCode::KeyS => {
                if self.stage.name == DRY_SAND_STAGE {
                    self.spawn_interval = if self.spawn_interval == 1 {
                        DRY_SAND_STREAM_INTERVAL
                    } else {
                        1
                    };
                    println!("Dry sand spawn interval: {}", self.spawn_interval);
                }
            }
            KeyCode::Digit1 => self.reset_stage(stage_water_sluice()),
            KeyCode::Digit2 => self.reset_stage(stage_dry_sand_stream()),
            KeyCode::Digit3 => self.reset_stage(stage_dry_gold_stream()),
            KeyCode::Digit4 => self.reset_stage(stage_dry_mixed_stream()),
            KeyCode::Digit5 => self.reset_stage(stage_sediment_water_no_dem()),
            KeyCode::Digit6 => self.reset_stage(stage_sediment_water_dem()),
            KeyCode::Digit7 => self.reset_stage(stage_two_way_coupling()),
            KeyCode::Digit8 => self.reset_stage(stage_clump_drop()),
            KeyCode::Digit9 => self.reset_stage(stage_sand_then_gold()),
            // Level 4 material switching (no reset!)
            KeyCode::KeyG => {
                self.material_mode = MaterialMode::Gold;
                println!("Material: GOLD (Level 4 test - gold on existing pile)");
            }
            KeyCode::KeyT => {
                self.material_mode = MaterialMode::Sand;
                println!("Material: SAND");
            }
            KeyCode::KeyM => {
                self.material_mode = MaterialMode::Mixed;
                println!("Material: MIXED (stage default)");
            }
            _ => {}
        }
        self.update_camera_from_gpu();
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let gpu = pollster::block_on(GpuContext::new(window));
        let renderer = ParticleRenderer::new(&gpu, 500_000);
        let dem_solver = GpuDemSolver::new(
            &gpu,
            self.stage.width as u32,
            self.stage.height as u32,
            100_000,
        );
        self.zoom = zoom_for_width(&self.stage, gpu.size.0);
        self.update_camera(gpu.size.0, gpu.size.1);
        self.renderer = Some(renderer);
        self.dem_solver = Some(dem_solver);
        self.gpu = Some(gpu);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Sediment Stages")
            .with_inner_size(LogicalSize::new(TARGET_WIDTH, TARGET_HEIGHT));
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.window = Some(window.clone());
        self.init_gpu(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
                self.zoom = zoom_for_width(&self.stage, size.width);
                self.update_camera(size.width, size.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
                // Convert to simulation coords
                let screen_height = self.gpu.as_ref().map(|g| g.size.1).unwrap_or(TARGET_HEIGHT) as f32;
                let sim_x = self.mouse_pos.0 / self.zoom + self.camera_offset.0;
                let sim_y = (screen_height - self.mouse_pos.1) / self.zoom + self.camera_offset.1;
                self.mouse_sim_pos = Vec2::new(sim_x, sim_y);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.mouse_left_held = pressed,
                    MouseButton::Right => self.mouse_right_held = pressed,
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state,
                    ..
                } = event
                {
                    self.handle_key(code, state == ElementState::Pressed);
                }
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
