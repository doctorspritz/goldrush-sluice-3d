//! River Flow Demo - Physics-Accurate 2.5D Shallow Water Equations
//!
//! Demonstrates a realistic river flowing through a defined channel using
//! GPU-accelerated Shallow Water Equations (SWE) simulation.
//!
//! Features:
//! - Parabolic river channel with raised banks
//! - Gravity-driven flow with 6m elevation drop
//! - Proper boundary conditions (drain only at downstream edge)
//! - Real-time mesh rendering of terrain and water surface
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

const WORLD_WIDTH: usize = 64;
const WORLD_DEPTH: usize = 64;
const CELL_SIZE: f32 = 0.5;

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new() -> Self {
        // Position camera to view the river channel from upstream
        // River flows along x-axis (left to right), channel is at z = 16m (world center)
        Self {
            position: Vec3::new(-5.0, 12.0, 16.0), // Behind and above the upstream end
            yaw: 0.0,    // Looking along +X (downstream)
            pitch: -0.4, // Slight downward angle
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

struct App {
    window: Option<Arc<Window>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    heightfield: Option<GpuHeightfield>,
    world: World,

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
        // Create world with a river channel
        let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 10.0);

        // River parameters
        let channel_center_z = WORLD_DEPTH as f32 / 2.0; // Center of grid
        let channel_half_width = 6.0; // 6 cells half-width = 12 cells total channel width
        let channel_depth = 1.5; // Maximum depth of channel below banks
        let bank_height = 0.5; // Banks rise above surrounding terrain

        // Create terrain with river channel
        for z in 0..WORLD_DEPTH {
            for x in 0..WORLD_WIDTH {
                let idx = world.idx(x, z);

                // Base slope: 6m drop from left (x=0) to right (x=max)
                let slope_height = 6.0 - (x as f32 / WORLD_WIDTH as f32) * 5.0;

                // Distance from channel center
                let dist_from_center = (z as f32 - channel_center_z).abs();

                // Channel profile: parabolic cross-section
                let channel_factor = if dist_from_center < channel_half_width {
                    // Inside channel: parabolic depression
                    let normalized_dist = dist_from_center / channel_half_width;
                    let parabolic = normalized_dist * normalized_dist; // 0 at center, 1 at edges
                    -channel_depth * (1.0 - parabolic)
                } else if dist_from_center < channel_half_width + 2.0 {
                    // Bank region: raised levee
                    let bank_factor = (dist_from_center - channel_half_width) / 2.0;
                    bank_height * (1.0 - bank_factor)
                } else {
                    // Floodplain: flat
                    0.0
                };

                world.bedrock_elevation[idx] = slope_height + channel_factor;
                world.overburden_thickness[idx] = 0.0;
                world.paydirt_thickness[idx] = 0.0;
                world.gravel_thickness[idx] = 0.0;
                world.terrain_sediment[idx] = 0.0;
            }
        }

        // Add initial water in the channel at the upstream end
        for z in (WORLD_DEPTH / 2 - 5)..(WORLD_DEPTH / 2 + 5) {
            for x in 2..15 {
                let idx = world.idx(x, z);
                let ground = world.ground_height(x, z);
                world.water_surface[idx] = ground + 0.3; // 30cm initial depth
            }
        }

        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            config: None,
            heightfield: None,
            world,
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
            let num_substeps = 4;
            let total_dt = dt.min(0.02);
            let sub_dt = total_dt / num_substeps as f32;

            // Emit water at the upstream end of the channel
            let emit_x = 3.0 * CELL_SIZE;
            let emit_z = (WORLD_DEPTH as f32 / 2.0) * CELL_SIZE;
            let emit_rate = 1.5; // m³/s - steady river flow
            let emit_radius = 4.0 * CELL_SIZE;

            for _ in 0..num_substeps {
                hf.update_params(queue, sub_dt);
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

        // Print stats every second
        static mut LAST_PRINT: Option<Instant> = None;
        unsafe {
            if LAST_PRINT.is_none() || LAST_PRINT.unwrap().elapsed().as_secs_f32() > 1.0 {
                let elapsed = self.start_time.elapsed().as_secs_f32();
                let flow_rate = if elapsed > 0.0 { self.total_water_added / elapsed } else { 0.0 };
                println!(
                    "t={:.1}s | Water added: {:.1}m³ | Flow rate: {:.2}m³/s | {}",
                    elapsed,
                    self.total_water_added,
                    flow_rate,
                    if self.paused { "PAUSED" } else { "Running" }
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
                        .with_title("River Flow Demo - WASD move, P pause, Q quit")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1024, 768)),
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

            // Upload initial terrain state
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
                                println!("Paused: {}", self.paused);
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
    println!("River Flow Demo - 2.5D Shallow Water Equations");
    println!("===============================================");
    println!();
    println!("Physics: GPU-accelerated SWE with proper boundary conditions");
    println!("- Water emitted at upstream (left) end of river channel");
    println!("- Flow governed by gravity and terrain slope (6m elevation drop)");
    println!("- Parabolic channel cross-section with raised banks");
    println!("- Water drains only at downstream (right) edge");
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
