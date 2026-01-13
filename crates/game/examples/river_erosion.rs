//! River Erosion Demo - Erosion & Deposition
//!
//! Demonstrates both erosion and deposition in a slope → pool system.
//! Water flows down a steep slope, then enters a flat pool where it slows.
//!
//! Physics demonstrated:
//! - EROSION on slope: Fast water erodes material (muddy brown water)
//! - DEPOSITION in pool: Slow water drops sediment (clear water, building sediment layer)
//! - Velocity-dependent erosion (faster water erodes more)
//! - Layer exposure (sediment → overburden → gravel → paydirt → bedrock)
//! - Material-specific critical velocities
//! - Sediment transport visualization via water color
//! - Settling velocity threshold for deposition
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look around
//! - P: Pause/Resume
//! - Q/Esc: Quit

use glam::{Mat4, Vec3};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use game::gpu::heightfield::GpuHeightfield;
use sim3d::World;

// Larger world to see erosion patterns develop
const WORLD_WIDTH: usize = 96;
const WORLD_DEPTH: usize = 64;
const CELL_SIZE: f32 = 0.5;

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new() -> Self {
        // Position camera to view both slope and pool
        Self {
            position: Vec3::new(24.0, 22.0, -12.0),
            yaw: 1.9,    // Looking toward slope and pool
            pitch: -0.5, // Looking down to see deposition
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }
}

struct InputState {
    keys: HashSet<KeyCode>,
    right_mouse: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

// Track erosion at sample points
struct ErosionMetrics {
    initial_heights: Vec<f32>,
    sample_points: Vec<(usize, usize)>,
}

impl ErosionMetrics {
    fn new(world: &World) -> Self {
        // Sample points along the valley center
        let center_z = WORLD_DEPTH / 2;
        let sample_points: Vec<(usize, usize)> = vec![
            (10, center_z),  // Upstream
            (30, center_z),  // Mid-upstream
            (50, center_z),  // Center
            (70, center_z),  // Mid-downstream
            (90, center_z),  // Downstream
        ];

        let initial_heights: Vec<f32> = sample_points
            .iter()
            .map(|&(x, z)| world.ground_height(x, z))
            .collect();

        Self { initial_heights, sample_points }
    }

    fn get_erosion_depths(&self, world: &World) -> Vec<f32> {
        self.sample_points
            .iter()
            .zip(self.initial_heights.iter())
            .map(|(&(x, z), &initial)| initial - world.ground_height(x, z))
            .collect()
    }
}

struct App {
    window: Option<Arc<Window>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    heightfield: Option<GpuHeightfield>,
    world: World,
    metrics: ErosionMetrics,

    camera: Camera,
    input: InputState,

    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,

    last_frame: Instant,
    start_time: Instant,
    total_water_added: f32,
    paused: bool,
}

impl App {
    fn new() -> Self {
        // Create world with slope → pool terrain for erosion/deposition demo
        let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 10.0);

        // Terrain: steep slope (left 60%) → flat pool (right 40%)
        let slope_grade = 0.25; // 25% grade - steep for fast erosion
        let top_elevation = 14.0; // Height at top (left edge)
        let slope_end_x = (WORLD_WIDTH as f32 * 0.6) as usize; // Slope ends at 60% width
        let pool_elevation = top_elevation - (slope_end_x as f32 * CELL_SIZE * slope_grade);

        // Layer thicknesses (meters) - generous layers to see erosion
        let paydirt_thickness = 0.3;
        let gravel_thickness = 0.2;
        let overburden_thickness = 0.8;
        let sediment_thickness = 0.4; // Thick sediment to see erosion progress

        // Create slope → pool terrain
        for z in 0..WORLD_DEPTH {
            for x in 0..WORLD_WIDTH {
                let idx = world.idx(x, z);

                // Bedrock height: slope → flat pool
                let bedrock_height = if x < slope_end_x {
                    // Slope region: descends from left to slope_end
                    let x_dist = x as f32 * CELL_SIZE;
                    top_elevation - x_dist * slope_grade
                } else {
                    // Pool region: flat at low elevation
                    pool_elevation
                };

                // Slight channel depression in center (z) to concentrate flow on slope
                let center_z = WORLD_DEPTH as f32 * 0.5;
                let z_offset = ((z as f32 - center_z).abs() / center_z).min(1.0);
                let channel_depth = if x < slope_end_x {
                    0.4 * (1.0 - z_offset * z_offset) // Channel on slope
                } else {
                    0.0 // Flat pool - no channel
                };

                // Set all layers
                world.bedrock_elevation[idx] = bedrock_height - channel_depth;
                world.paydirt_thickness[idx] = paydirt_thickness;
                world.gravel_thickness[idx] = gravel_thickness;
                world.overburden_thickness[idx] = overburden_thickness;

                // Less initial sediment in pool to see deposition clearly
                world.terrain_sediment[idx] = if x < slope_end_x {
                    sediment_thickness
                } else {
                    0.05 // Minimal sediment in pool - watch it build up!
                };
            }
        }

        // No pre-fill - let water flow down naturally from emitter

        let metrics = ErosionMetrics::new(&world);

        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            config: None,
            heightfield: None,
            world,
            metrics,
            camera: Camera::new(),
            input: InputState {
                keys: HashSet::new(),
                right_mouse: false,
                last_mouse_pos: None,
            },
            depth_texture: None,
            depth_view: None,
            last_frame: Instant::now(),
            start_time: Instant::now(),
            total_water_added: 0.0,
            paused: false,
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        if let (Some(config), Some(surface), Some(device)) =
            (&mut self.config, &self.surface, &self.device)
        {
            config.width = size.width;
            config.height = size.height;
            surface.configure(device, config);

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view =
                Some(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.depth_texture = Some(depth_texture);
        }
    }

    fn update_camera(&mut self, dt: f32) {
        let speed = 10.0 * dt;
        let forward = self.camera.forward();
        let right = self.camera.right();

        if self.input.keys.contains(&KeyCode::KeyW) {
            self.camera.position += forward * speed;
        }
        if self.input.keys.contains(&KeyCode::KeyS) {
            self.camera.position -= forward * speed;
        }
        if self.input.keys.contains(&KeyCode::KeyA) {
            self.camera.position -= right * speed;
        }
        if self.input.keys.contains(&KeyCode::KeyD) {
            self.camera.position += right * speed;
        }
        if self.input.keys.contains(&KeyCode::Space) {
            self.camera.position.y += speed;
        }
        if self.input.keys.contains(&KeyCode::ShiftLeft) {
            self.camera.position.y -= speed;
        }
    }

    fn redraw(&mut self) {
        if self.device.is_none() || self.queue.is_none() || self.surface.is_none() || self.heightfield.is_none() {
            return;
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.update_camera(dt);

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let hf = self.heightfield.as_ref().unwrap();

        if !self.paused {
            // Use multiple substeps for numerical stability
            // Smaller timesteps = less oscillation
            let num_substeps = 4;
            let total_dt = dt.min(0.02);
            let sub_dt = total_dt / num_substeps as f32;

            // Emit water at the upstream end of the valley
            let emit_x = 5.0 * CELL_SIZE;
            let emit_z = (WORLD_DEPTH as f32 / 2.0) * CELL_SIZE;
            let emit_rate = 8.0; // m³/s - high flow for visible erosion
            let emit_radius = 5.0 * CELL_SIZE;

            for _ in 0..num_substeps {
                // Update simulation parameters with substep dt
                hf.update_params(queue, sub_dt);

                // Emit water (rate is per-second, so use sub_dt)
                hf.update_emitter(
                    queue,
                    emit_x,
                    emit_z,
                    emit_radius,
                    emit_rate,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    sub_dt,
                    true,
                );

                // Run GPU simulation substep
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                hf.dispatch_emitter(&mut encoder);
                hf.dispatch_tile(&mut encoder, WORLD_WIDTH as u32, WORLD_DEPTH as u32);

                queue.submit(Some(encoder.finish()));
            }

            self.total_water_added += emit_rate * total_dt;
        }

        // Render
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.depth_view.as_ref().unwrap();

        let view_matrix = self.camera.view_matrix();
        let config = self.config.as_ref().unwrap();
        let proj_matrix = Mat4::perspective_rh(
            0.8,
            config.width as f32 / config.height as f32,
            0.1,
            500.0,
        );
        let view_proj = proj_matrix * view_matrix;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        hf.render(
            &mut encoder,
            &view,
            depth_view,
            queue,
            view_proj.to_cols_array_2d(),
            self.camera.position.into(),
            self.start_time.elapsed().as_secs_f32(),
            true,
        );

        queue.submit(Some(encoder.finish()));
        frame.present();

        // Print erosion stats every second
        static mut LAST_PRINT: Option<Instant> = None;
        unsafe {
            if LAST_PRINT.is_none() || LAST_PRINT.unwrap().elapsed().as_secs_f32() > 1.0 {
                let elapsed = self.start_time.elapsed().as_secs_f32();

                // Note: We can't easily read GPU state back, so this shows initial metrics
                // In a full implementation, we'd async-read the GPU buffers
                println!(
                    "t={:>5.1}s | Water: {:>6.1}m³ | Flow: {:.1}m³/s | {}",
                    elapsed,
                    self.total_water_added,
                    if elapsed > 0.0 { self.total_water_added / elapsed } else { 0.0 },
                    if self.paused { "PAUSED" } else { "Running - watch colors change as layers erode!" }
                );

                LAST_PRINT = Some(Instant::now());
            }
        }

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("River Erosion Demo - Watch layers erode!")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 800)),
                )
                .unwrap(),
        );
        self.window = Some(window.clone());

        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .unwrap();

            let size = window.inner_size();
            let caps = surface.get_capabilities(&adapter);
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: caps.formats[0],
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            // Create heightfield GPU simulation
            let hf = GpuHeightfield::new(
                &device,
                WORLD_WIDTH as u32,
                WORLD_DEPTH as u32,
                CELL_SIZE,
                10.0,
                config.format,
            );

            // Upload initial terrain state with all layers
            hf.upload_from_world(&queue, &self.world);

            self.device = Some(device);
            self.queue = Some(queue);
            self.surface = Some(surface);
            self.config = Some(config);
            self.heightfield = Some(hf);

            self.resize(size);
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size),
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.input.right_mouse = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.input.right_mouse {
                    if let Some((lx, ly)) = self.input.last_mouse_pos {
                        let dx = (position.x - lx) as f32 * 0.003;
                        let dy = (position.y - ly) as f32 * 0.003;
                        self.camera.yaw += dx;
                        self.camera.pitch -= dy;
                        self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
                    }
                }
                self.input.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state.is_pressed() {
                        self.input.keys.insert(code);

                        match code {
                            KeyCode::KeyQ | KeyCode::Escape => event_loop.exit(),
                            KeyCode::KeyP => {
                                self.paused = !self.paused;
                                println!("Simulation {}", if self.paused { "PAUSED" } else { "RESUMED" });
                            }
                            _ => {}
                        }
                    } else {
                        self.input.keys.remove(&code);
                    }
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

fn main() {
    println!("River Erosion Demo - Erosion & Deposition");
    println!("==========================================");
    println!();
    println!("Terrain: SLOPE (left 60%) → FLAT POOL (right 40%)");
    println!("  - Water emitted at TOP LEFT, flows down slope");
    println!("  - SLOPE: Fast water erodes material (muddy brown water)");
    println!("  - POOL: Slow water deposits sediment (clear water, growing sediment layer)");
    println!();
    println!("Material layers (top to bottom, critical velocity):");
    println!("  - Sediment (tan)      - v_crit = 0.10 m/s (easiest)");
    println!("  - Overburden (brown)  - v_crit = 0.20 m/s");
    println!("  - Gravel (grey)       - v_crit = 0.50 m/s");
    println!("  - Paydirt (gold)      - v_crit = 0.40 m/s");
    println!("  - Bedrock (dark grey) - does not erode");
    println!();
    println!("Watch: Muddy water from slope becomes clear as sediment settles in pool");
    println!();
    println!("Controls:");
    println!("  WASD        - Move camera");
    println!("  Space/Shift - Move up/down");
    println!("  Right Mouse - Look around");
    println!("  P           - Pause/Resume simulation");
    println!("  Q/Esc       - Quit");
    println!();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
