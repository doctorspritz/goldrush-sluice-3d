//! Goldrush Fluid Miner - Game
//!
//! PIC/FLIP fluid simulation demo with sluice box vortex formation.
//! Uses wgpu for GPU-accelerated rendering and compute.

mod gpu;

use gpu::{g2p::GpuG2pSolver, p2g::GpuP2gSolver, pressure::GpuPressureSolver, renderer::ParticleRenderer, GpuContext};
use sim::{create_sluice_with_mode, FlipSimulation, RiffleMode, SluiceConfig};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Simulation constants
const SIM_WIDTH: usize = 512;
const SIM_HEIGHT: usize = 256;
const CELL_SIZE: f32 = 1.0;
const SCALE: f32 = 2.5;

// Emitter configuration
const PARTICLES_PER_EMITTER: usize = 10;
const EMITTER_SPACING: f32 = 3.0; // Cells between emitters

/// Application state
struct App {
    // Rendering
    gpu: Option<GpuContext>,
    particle_renderer: Option<ParticleRenderer>,
    pressure_solver: Option<GpuPressureSolver>,
    p2g_solver: Option<GpuP2gSolver>,
    g2p_solver: Option<GpuG2pSolver>,
    window: Option<Arc<Window>>,

    // Simulation
    sim: FlipSimulation,
    sluice_config: SluiceConfig,
    paused: bool,

    // Input state
    zoom: f32,
    inlet_x: f32,
    inlet_y: f32,
    inlet_vx: f32,
    inlet_vy: f32,
    num_emitters: usize,
    flow_multiplier: usize,
    sand_rate: usize,
    magnetite_rate: usize,
    gold_rate: usize,
    fast_particle_size: f32,

    // Mouse state
    mouse_pos: (f32, f32),
    mouse_left_down: bool,

    // Frame counting
    frame_count: u64,
    start_time: std::time::Instant,
    profile_accum: [f32; 7],
    profile_count: u32,

    // Keyboard state for modifiers
    shift_down: bool,

    // GPU toggles
    use_gpu_p2g: bool,
    use_gpu_g2p: bool,
}

impl App {
    fn new() -> Self {
        let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

        let sluice_config = SluiceConfig {
            slope: 0.12,
            riffle_spacing: 60,
            riffle_height: 6,
            riffle_width: 4,
            riffle_mode: RiffleMode::ClassicBattEdge,
            slick_plate_len: 0,
        };
        create_sluice_with_mode(&mut sim, &sluice_config);

        // TEST: Add barrier at end of sluice to stress-test pressure solver
        let barrier_x = SIM_WIDTH - 10; // 10 cells from right edge
        for j in 0..SIM_HEIGHT {
            for i in barrier_x..SIM_WIDTH {
                let idx = j * SIM_WIDTH + i;
                sim.grid.solid[idx] = true;
            }
        }

        Self {
            gpu: None,
            particle_renderer: None,
            pressure_solver: None,
            p2g_solver: None,
            g2p_solver: None,
            window: None,
            sim,
            sluice_config,
            paused: false,
            zoom: SCALE,
            inlet_x: 5.0,
            inlet_y: (SIM_HEIGHT / 4 - 10) as f32,
            inlet_vx: 80.0,
            inlet_vy: 5.0,
            num_emitters: 4,
            flow_multiplier: 1,
            sand_rate: 4,
            magnetite_rate: 8,
            gold_rate: 20,
            fast_particle_size: CELL_SIZE * SCALE * 1.5,
            mouse_pos: (0.0, 0.0),
            mouse_left_down: false,
            frame_count: 0,
            start_time: std::time::Instant::now(),
            profile_accum: [0.0; 7],
            profile_count: 0,
            shift_down: false,
            use_gpu_p2g: true,  // Start with GPU P2G enabled
            use_gpu_g2p: true,  // Start with GPU G2P enabled
        }
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }

        // Mouse spawning
        if self.mouse_left_down {
            let wx = self.mouse_pos.0 / self.zoom;
            let wy = self.mouse_pos.1 / self.zoom;
            self.sim.spawn_water(wx, wy, 20.0, 0.0, 5);
        }

        // Spawn water from multiple emitters (spaced vertically)
        for i in 0..self.num_emitters {
            let emitter_y = self.inlet_y - (i as f32 * EMITTER_SPACING);
            self.sim.spawn_water(
                self.inlet_x,
                emitter_y,
                self.inlet_vx,
                self.inlet_vy,
                PARTICLES_PER_EMITTER * self.flow_multiplier,
            );
        }

        // Sediments spawn from all emitters (cycled round-robin)
        let emitter_idx = (self.frame_count as usize) % self.num_emitters;
        let sediment_y = self.inlet_y - (emitter_idx as f32 * EMITTER_SPACING);

        // Sand
        let effective_sand = self.sand_rate / self.flow_multiplier.max(1);
        if effective_sand > 0 && self.frame_count % effective_sand as u64 == 0 {
            self.sim
                .spawn_sand(self.inlet_x, sediment_y, self.inlet_vx, self.inlet_vy, 1);
        }

        // Magnetite
        let effective_magnetite = self.magnetite_rate / self.flow_multiplier.max(1);
        if effective_magnetite > 0 && self.frame_count % effective_magnetite as u64 == 0 {
            self.sim
                .spawn_magnetite(self.inlet_x, sediment_y, self.inlet_vx, self.inlet_vy, 1);
        }

        // Gold
        let effective_gold = self.gold_rate / self.flow_multiplier.max(1);
        if effective_gold > 0 && self.frame_count % effective_gold as u64 == 0 {
            self.sim
                .spawn_gold(self.inlet_x, sediment_y, self.inlet_vx, self.inlet_vy, 1);
        }

        // Remove particles at outflow - DISABLED for barrier test
        // let outflow_x = (SIM_WIDTH as f32 - 5.0) * CELL_SIZE;
        // self.sim.particles.list.retain(|p| p.position.x < outflow_x);

        // Run simulation with profiling
        let dt = 1.0 / 60.0;

        // TODO: Debug GPU pressure solver - produces similar divergence numbers but
        // visually worse results than CPU. Keep CPU for now until GPU is fixed.
        // See diagnostics: GPU div_out=6-16, CPU div_out=2-20, but CPU looks correct.
        let use_gpu_pressure = false;
        if use_gpu_pressure {
        if let (Some(gpu), Some(solver)) = (&self.gpu, &self.pressure_solver) {
            use std::time::Instant;

            // Phase 1: CPU prepares for pressure solve
            let pre_timings = self.sim.prepare_pressure_solve(dt);

            // Phase 2: GPU pressure solve with warm start
            let press_start = Instant::now();

            // Convert cell types to u32 for GPU
            let cell_types: Vec<u32> = self
                .sim
                .grid
                .cell_type
                .iter()
                .map(|&ct| ct as u32)
                .collect();

            // Upload with warm start from previous pressure
            solver.upload_warm(
                gpu,
                &self.sim.grid.divergence,
                &cell_types,
                &self.sim.grid.pressure,
                1.9,
            );

            // Run GPU pressure solve (30 iterations with warm start)
            solver.solve(gpu, 30);

            // Download pressure results
            solver.download(gpu, &mut self.sim.grid.pressure);

            let press_time = press_start.elapsed().as_secs_f32() * 1000.0;

            // Store input divergence for diagnostics
            let pre_max_div = self.sim.grid.divergence.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));

            // Phase 3: CPU finishes simulation (applies pressure to velocities)
            let post_timings = self.sim.finalize_after_pressure(dt);

            // DIAGNOSTICS: Check pressure solve quality
            if self.frame_count % 60 == 0 {
                // Check for NaN/inf in pressure
                let nan_count = self.sim.grid.pressure.iter().filter(|p| p.is_nan()).count();
                let inf_count = self.sim.grid.pressure.iter().filter(|p| p.is_infinite()).count();
                let max_p = self.sim.grid.pressure.iter().cloned().fold(0.0f32, f32::max);
                let min_p = self.sim.grid.pressure.iter().cloned().fold(0.0f32, f32::min);

                // Compute divergence AFTER pressure applied to velocities (should be ~0)
                self.sim.grid.compute_divergence();
                let post_max_div = self.sim.grid.divergence.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));

                // Check max velocity
                let max_vel = self.sim.particles.list.iter()
                    .map(|p| p.velocity.length())
                    .fold(0.0f32, f32::max);

                eprintln!("GPU: div_in={:.2} -> div_out={:.2} | p[{:.1}..{:.1}] nan={} inf={} | vel={:.1}",
                    pre_max_div, post_max_div, min_p, max_p, nan_count, inf_count, max_vel);
            }

            // Combine timings: [classify, sdf, p2g, press, g2p, neigh, rest]
            self.profile_accum[0] += pre_timings[0];
            self.profile_accum[1] += pre_timings[1];
            self.profile_accum[2] += pre_timings[2];
            self.profile_accum[3] += press_time;
            self.profile_accum[4] += post_timings[1]; // g2p
            self.profile_accum[5] += post_timings[2]; // neigh
            self.profile_accum[6] += post_timings[3]; // rest
        }
        } else if self.use_gpu_p2g {
            // GPU P2G path (optionally with GPU G2P)
            if let (Some(gpu), Some(p2g_solver)) = (&self.gpu, &self.p2g_solver) {
                use std::time::Instant;

                // Phase 1a: Classify + SDF (CPU)
                let pre_timings = self.sim.prepare_for_p2g();

                // Phase 1b: GPU P2G
                let p2g_start = Instant::now();
                p2g_solver.execute(
                    gpu,
                    &self.sim.particles.list,
                    CELL_SIZE,
                    &mut self.sim.grid.u,
                    &mut self.sim.grid.v,
                );
                let p2g_time = p2g_start.elapsed().as_secs_f32() * 1000.0;

                // Phase 1c: Complete P2G phase (extrapolate, store_old, forces, divergence)
                // Note: This now also stores grid.u_old/v_old for GPU G2P FLIP delta
                self.sim.complete_p2g_phase(dt);

                // Phase 2: CPU pressure solve (multigrid)
                let press_start = Instant::now();
                self.sim.grid.solve_pressure_multigrid(4);
                let press_time = press_start.elapsed().as_secs_f32() * 1000.0;

                // Phase 3a: Pre-G2P (apply pressure gradient to grid)
                let pre_g2p_time = self.sim.finalize_pre_g2p(dt);

                // Phase 3b: Hybrid G2P (GPU for water, CPU for sediment)
                let g2p_start = Instant::now();
                if self.use_gpu_g2p {
                    if let Some(g2p_solver) = &self.g2p_solver {
                        // GPU G2P for water particles
                        g2p_solver.execute(
                            gpu,
                            &mut self.sim.particles.list,
                            &self.sim.grid.u,
                            &self.sim.grid.v,
                            &self.sim.grid.u_old,
                            &self.sim.grid.v_old,
                            CELL_SIZE,
                            dt,
                        );
                    }
                    // CPU G2P for sediment only (preserves all sediment physics)
                    self.sim.grid_to_particles_sediment_only(dt);
                } else {
                    // Full CPU G2P for all particles
                    self.sim.grid_to_particles(dt);
                }
                let g2p_time = g2p_start.elapsed().as_secs_f32() * 1000.0;

                // Phase 3c: Post-G2P (advection, neighbor, DEM)
                let post_g2p_timings = self.sim.finalize_post_g2p(dt);

                // Combine timings: [classify, sdf, p2g, press, g2p, neigh, rest]
                self.profile_accum[0] += pre_timings[0];  // classify
                self.profile_accum[1] += pre_timings[1];  // sdf
                self.profile_accum[2] += p2g_time;        // p2g (GPU)
                self.profile_accum[3] += press_time + pre_g2p_time;  // pressure + pre-g2p
                self.profile_accum[4] += g2p_time;        // g2p (GPU/CPU hybrid)
                self.profile_accum[5] += post_g2p_timings[0]; // neigh
                self.profile_accum[6] += post_g2p_timings[1]; // rest

                // GPU diagnostics
                if self.frame_count % 60 == 0 {
                    self.sim.grid.compute_divergence();
                    let max_div = self.sim.grid.divergence.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                    let max_vel = self.sim.particles.list.iter()
                        .map(|p| p.velocity.length())
                        .fold(0.0f32, f32::max);
                    let g2p_mode = if self.use_gpu_g2p { "GPU" } else { "CPU" };
                    eprintln!("GPU_P2G + {}_G2P: div_out={:.4} | vel={:.1} | p2g={:.2}ms g2p={:.2}ms",
                        g2p_mode, max_div, max_vel, p2g_time, g2p_time);
                }
            }
        } else {
            // CPU fallback
            let timings = self.sim.update_profiled(dt);
            for (i, t) in timings.iter().enumerate() {
                self.profile_accum[i] += t;
            }

            // CPU diagnostics
            if self.frame_count % 60 == 0 {
                self.sim.grid.compute_divergence();
                let max_div = self.sim.grid.divergence.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                let max_vel = self.sim.particles.list.iter()
                    .map(|p| p.velocity.length())
                    .fold(0.0f32, f32::max);
                eprintln!("CPU: div_out={:.4} | vel={:.1}", max_div, max_vel);
            }
        }

        self.profile_count += 1;
        self.frame_count += 1;

        // Log diagnostics every second
        if self.frame_count % 60 == 0 {
            self.sim.grid.compute_divergence();

            let elapsed = self.start_time.elapsed().as_secs();
            let n = self.profile_count.max(1) as f32;
            let avg: Vec<f32> = self.profile_accum.iter().map(|&t| t / n).collect();
            let total: f32 = avg.iter().sum();

            println!(
                "t={:3}s: {:6} p, sim={:5.1}ms | classify:{:4.2} sdf:{:4.2} p2g:{:4.2} press:{:5.2} g2p:{:4.2} neigh:{:4.2} rest:{:4.2}",
                elapsed,
                self.sim.particles.len(),
                total,
                avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6]
            );

            self.profile_accum = [0.0; 7];
            self.profile_count = 0;
        }
    }

    fn render(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let Some(renderer) = &mut self.particle_renderer else {
            return;
        };

        // Get surface texture
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::error!("Failed to get surface texture: {:?}", e);
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Clear with background color
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.2,  // Dark blue background
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

        // STEP 2: Terrain + water only (separate buffers now)
        renderer.draw_terrain(
            gpu,
            &mut encoder,
            &view,
            &self.sim.grid,
            CELL_SIZE,
            self.zoom,
        );

        // Water particles (filtered in renderer)
        renderer.draw(
            gpu,
            &mut encoder,
            &view,
            &self.sim.particles,
            self.zoom,
            self.fast_particle_size,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        // Track modifier state
        if key == KeyCode::ShiftLeft || key == KeyCode::ShiftRight {
            self.shift_down = pressed;
        }

        if !pressed {
            return;
        }

        match key {
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::KeyR => {
                // Reset simulation
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::KeyC => self.sim.particles.list.clear(),
            KeyCode::KeyN => {
                self.sluice_config.riffle_mode = self.sluice_config.riffle_mode.next();
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::ArrowRight => {
                if self.shift_down {
                    self.inlet_vy = (self.inlet_vy + 5.0).min(80.0);
                } else {
                    self.inlet_vx = (self.inlet_vx + 5.0).min(200.0);
                }
            }
            KeyCode::ArrowLeft => {
                if self.shift_down {
                    self.inlet_vy = (self.inlet_vy - 5.0).max(0.0);
                } else {
                    self.inlet_vx = (self.inlet_vx - 5.0).max(20.0);
                }
            }
            KeyCode::ArrowUp => self.num_emitters = (self.num_emitters + 1).min(20),
            KeyCode::ArrowDown => self.num_emitters = self.num_emitters.saturating_sub(1).max(1),
            KeyCode::Equal => {
                if !self.shift_down {
                    self.zoom = (self.zoom + 0.25).min(6.0);
                    self.fast_particle_size = CELL_SIZE * self.zoom * 1.5;
                } else {
                    self.flow_multiplier = (self.flow_multiplier + 1).min(10);
                }
            }
            KeyCode::Minus => {
                if !self.shift_down {
                    self.zoom = (self.zoom - 0.25).max(0.5);
                    self.fast_particle_size = CELL_SIZE * self.zoom * 1.5;
                } else {
                    self.flow_multiplier = self.flow_multiplier.saturating_sub(1).max(1);
                }
            }
            KeyCode::KeyQ => {
                self.sluice_config.riffle_spacing =
                    (self.sluice_config.riffle_spacing + 10).min(120);
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::KeyA => {
                self.sluice_config.riffle_spacing =
                    self.sluice_config.riffle_spacing.saturating_sub(10).max(30);
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::KeyW => {
                self.sluice_config.riffle_height = (self.sluice_config.riffle_height + 2).min(16);
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::KeyS => {
                self.sluice_config.riffle_height =
                    self.sluice_config.riffle_height.saturating_sub(2).max(4);
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::KeyZ => {
                if self.shift_down {
                    self.sluice_config.slope = (self.sluice_config.slope + 0.02).min(0.5);
                } else {
                    self.sluice_config.slope = (self.sluice_config.slope - 0.02).max(0.0);
                }
                self.sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
                create_sluice_with_mode(&mut self.sim, &self.sluice_config);
                if let Some(ref mut renderer) = self.particle_renderer {
                    renderer.invalidate_terrain();
                }
            }
            KeyCode::Digit2 => {
                self.sand_rate = if self.sand_rate == 0 {
                    4
                } else if self.sand_rate > 1 {
                    self.sand_rate - 1
                } else {
                    0
                };
            }
            KeyCode::Digit3 => {
                self.magnetite_rate = if self.magnetite_rate == 0 {
                    8
                } else if self.magnetite_rate > 1 {
                    self.magnetite_rate - 1
                } else {
                    0
                };
            }
            KeyCode::Digit4 => {
                self.gold_rate = if self.gold_rate == 0 {
                    20
                } else if self.gold_rate > 5 {
                    self.gold_rate - 5
                } else {
                    0
                };
            }
            KeyCode::Digit9 => self.fast_particle_size = (self.fast_particle_size - 0.5).max(1.0),
            KeyCode::Digit0 => self.fast_particle_size = (self.fast_particle_size + 0.5).min(8.0),
            KeyCode::KeyP => {
                self.use_gpu_p2g = !self.use_gpu_p2g;
                log::info!("GPU P2G: {}", if self.use_gpu_p2g { "ENABLED" } else { "DISABLED" });
            }
            KeyCode::KeyG => {
                self.use_gpu_g2p = !self.use_gpu_g2p;
                log::info!("GPU G2P: {}", if self.use_gpu_g2p { "ENABLED" } else { "DISABLED" });
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Goldrush Fluid Miner - wgpu")
            .with_inner_size(winit::dpi::LogicalSize::new(
                (SIM_WIDTH as f32 * CELL_SIZE * SCALE) as u32,
                (SIM_HEIGHT as f32 * CELL_SIZE * SCALE) as u32,
            ));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.window = Some(window.clone());

        // Initialize GPU (blocking on async)
        let gpu = pollster::block_on(GpuContext::new(window.clone()));

        // Create renderers and GPU compute solvers
        let particle_renderer = ParticleRenderer::new(&gpu, 300_000);
        let pressure_solver =
            GpuPressureSolver::new(&gpu, SIM_WIDTH as u32, SIM_HEIGHT as u32);
        let p2g_solver = GpuP2gSolver::new(&gpu, SIM_WIDTH as u32, SIM_HEIGHT as u32, 300_000);
        let g2p_solver = GpuG2pSolver::new(&gpu, SIM_WIDTH as u32, SIM_HEIGHT as u32, 300_000);

        self.particle_renderer = Some(particle_renderer);
        self.pressure_solver = Some(pressure_solver);
        self.p2g_solver = Some(p2g_solver);
        self.g2p_solver = Some(g2p_solver);
        self.gpu = Some(gpu);

        log::info!("GPU initialized successfully (P2G + G2P solvers ready)");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                self.handle_key(key, state == ElementState::Pressed);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if button == MouseButton::Left {
                    self.mouse_left_down = state == ElementState::Pressed;
                }
                if button == MouseButton::Right && state == ElementState::Pressed {
                    // Set emitter position
                    let wx = self.mouse_pos.0 / self.zoom;
                    let wy = self.mouse_pos.1 / self.zoom;
                    self.inlet_x = wx.clamp(2.0, (SIM_WIDTH - 50) as f32);
                    let base_floor = (SIM_HEIGHT / 4) as f32;
                    let floor_at_x = base_floor
                        + (self.inlet_x - self.sluice_config.slick_plate_len as f32).max(0.0)
                            * self.sluice_config.slope;
                    self.inlet_y = wy.clamp(5.0, floor_at_x - 5.0);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };
                self.zoom = (self.zoom + scroll * 0.2).clamp(0.5, 6.0);
                self.fast_particle_size = CELL_SIZE * self.zoom * 1.5;
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop failed");
}
