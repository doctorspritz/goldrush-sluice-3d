//! Visual stage runner for sediment diagnostics.
//! Run with: cargo run --example sediment_stages_visual -p game --release

mod gpu {
    pub use game::gpu::*;
}

use gpu::{dem::GpuDemSolver, renderer::ParticleRenderer, GpuContext};
use sim::stages::{
    stage_clump_drop, stage_dry_gold_stream, stage_dry_mixed_stream, stage_dry_sand_stream,
    stage_sand_then_gold, stage_sediment_water_dem, stage_sediment_water_no_dem,
    stage_two_way_coupling, stage_water_sluice, StageMode, StageSpec,
};
use sim::FlipSimulation;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
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
}

impl App {
    fn new() -> Self {
        let stage = stage_dry_sand_stream();
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
        println!("Controls: 1-8 switch stage, SPACE pause, F toggle flow, R reset, +/- zoom, D dense sand, S slow stream");

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
        }
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

        let is_dry_sand = self.stage.name == DRY_SAND_STAGE;
        let should_spawn = self.flow_enabled
            && (!is_dry_sand || (self.frame % self.spawn_interval == 0));
        if should_spawn {
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
                    // Debug: print first particle info every 60 frames
                    if self.frame % 60 == 0 && !self.sim.particles.list.is_empty() {
                        let p = &self.sim.particles.list[0];
                        let sdf = self.sim.grid.sample_sdf(p.position);
                        let h_domain = self.stage.height as f32 * self.stage.cell_size;
                        println!(
                            "DEBUG: frame={} pos=({:.1},{:.1}) vel=({:.1},{:.1}) sdf={:.2} floor_y={:.0}",
                            self.frame, p.position.x, p.position.y,
                            p.velocity.x, p.velocity.y, sdf, h_domain
                        );
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
