//! Terrain Collapse Demo - Angle of Repose Visualization
//!
//! Visual demonstration of materials collapsing to their angle of repose.
//! This validates the physics shown in the unit tests.
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look
//! - 1: Add sand/sediment pile at cursor
//! - 2: Add dirt/overburden pile at cursor
//! - 3: Add gravel pile at cursor
//! - C: Toggle continuous collapse simulation
//! - R: Reset to flat terrain
//! - ESC: Quit
//!
//! Expected angles:
//! - Sand/Sediment: ~32°
//! - Dirt/Overburden: ~35°
//! - Gravel: ~38°
//!
//! Run: cargo run --example collapse_demo --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use sim3d::{TerrainMaterial, World};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Small world for focused collapse visualization
const WORLD_WIDTH: usize = 64;
const WORLD_DEPTH: usize = 64;
const CELL_SIZE: f32 = 0.1; // 10cm cells for detailed view
const BASE_HEIGHT: f32 = 1.0;

const PILE_HEIGHT: f32 = 0.4; // Height of added piles
const PILE_RADIUS: f32 = 0.8; // Pile radius in world units (8 cells = rounder look)

const MOVE_SPEED: f32 = 5.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

const STEPS_PER_FRAME: usize = 3;
const DT: f32 = 0.016;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.cos() * cos_pitch,
            self.pitch.sin(),
            self.yaw.sin() * cos_pitch,
        )
        .normalize()
    }

    fn forward_flat(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    fn right_flat(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }
}

struct InputState {
    keys: HashSet<KeyCode>,
    mouse_look: bool,
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pos: (f32, f32),
    scroll_delta: f32,
}

use game::gpu::heightfield::GpuHeightfield;

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    heightfield: GpuHeightfield,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    world: World,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    last_stats: Instant,
    start_time: Instant,
    window_size: (u32, u32),
    collapse_enabled: bool,
    pile_count: u32,
}

impl App {
    fn new() -> Self {
        let world = build_flat_world();

        // Camera position for overhead view of 64x64 world at 10cm cells (6.4m x 6.4m)
        let center = Vec3::new(3.2, 0.0, 3.2);

        Self {
            window: None,
            gpu: None,
            world,
            camera: Camera {
                position: Vec3::new(center.x, 5.0, center.z + 4.0),
                yaw: -std::f32::consts::FRAC_PI_2,
                pitch: -0.6,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                last_mouse_pos: None,
                mouse_pos: (0.0, 0.0),
                scroll_delta: 0.0,
            },
            last_frame: Instant::now(),
            last_stats: Instant::now(),
            start_time: Instant::now(),
            window_size: (1280, 720),
            collapse_enabled: false, // Start paused so you can see piles form
            pile_count: 0,
        }
    }

    fn reset_world(&mut self) {
        self.world = build_flat_world();
        self.pile_count = 0;
        if let Some(gpu) = &self.gpu {
            gpu.heightfield.upload_from_world(&gpu.queue, &self.world);
        }
        println!("=== World Reset ===");
    }

    fn add_pile_at_cursor(&mut self, material: u32) {
        if let Some(hit) = self.raycast_terrain() {
            let material_name = match material {
                0 => "Sand/Sediment",
                1 => "Dirt/Overburden",
                2 => "Gravel",
                _ => "Unknown",
            };

            let expected_angle = match material {
                0 => TerrainMaterial::Sand.angle_of_repose().to_degrees(),
                1 => TerrainMaterial::Dirt.angle_of_repose().to_degrees(),
                2 => TerrainMaterial::Gravel.angle_of_repose().to_degrees(),
                _ => 0.0,
            };

            println!(
                "\n--- Adding {} pile at ({:.2}, {:.2}) ---",
                material_name, hit.x, hit.z
            );
            println!("Expected angle of repose: {:.1}°", expected_angle);

            if let Some(gpu) = &self.gpu {
                // Use material tool to add pile
                gpu.heightfield.update_material_tool(
                    &gpu.queue,
                    hit.x,
                    hit.z,
                    PILE_RADIUS,
                    PILE_HEIGHT * 100.0, // Scale up for visibility
                    material,
                    0.016,
                    true,
                );

                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Pile Add Encoder"),
                        });
                gpu.heightfield.dispatch_material_tool(&mut encoder);
                gpu.queue.submit(Some(encoder.finish()));

                self.pile_count += 1;
            }
        }
    }

    fn update(&mut self, dt: f32) {
        self.update_camera(dt);

        if self.collapse_enabled {
            if let Some(gpu) = &self.gpu {
                let sim_dt = DT;
                let steps = ((dt / sim_dt).ceil() as usize).min(STEPS_PER_FRAME);

                for _ in 0..steps {
                    let mut encoder =
                        gpu.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Collapse Sim Encoder"),
                            });

                    gpu.heightfield.update_params(&gpu.queue, sim_dt);
                    gpu.heightfield.dispatch(&mut encoder, sim_dt);

                    gpu.queue.submit(Some(encoder.finish()));
                }
            }
        }

        // Print stats periodically
        if self.last_stats.elapsed() > Duration::from_secs(2) {
            self.print_stats();
            self.last_stats = Instant::now();
        }
    }

    fn print_stats(&self) {
        println!("\n=== Collapse Demo Stats ===");
        println!("Collapse sim: {}", if self.collapse_enabled { "ON" } else { "OFF" });
        println!("Piles added: {}", self.pile_count);
        println!(
            "Expected angles: Sand={:.1}°, Dirt={:.1}°, Gravel={:.1}°",
            TerrainMaterial::Sand.angle_of_repose().to_degrees(),
            TerrainMaterial::Dirt.angle_of_repose().to_degrees(),
            TerrainMaterial::Gravel.angle_of_repose().to_degrees(),
        );
    }

    fn update_camera(&mut self, dt: f32) {
        let mut direction = Vec3::ZERO;

        if self.input.keys.contains(&KeyCode::KeyW) {
            direction += self.camera.forward_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyS) {
            direction -= self.camera.forward_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyA) {
            direction -= self.camera.right_flat();
        }
        if self.input.keys.contains(&KeyCode::KeyD) {
            direction += self.camera.right_flat();
        }
        if self.input.keys.contains(&KeyCode::Space) {
            direction.y += 1.0;
        }
        if self.input.keys.contains(&KeyCode::ShiftLeft)
            || self.input.keys.contains(&KeyCode::ShiftRight)
        {
            direction.y -= 1.0;
        }

        if direction.length_squared() > 0.0 {
            self.camera.position += direction.normalize() * self.camera.speed * dt;
        }

        // Apply scroll zoom
        if self.input.scroll_delta != 0.0 {
            let forward = self.camera.forward();
            self.camera.position += forward * self.input.scroll_delta * 0.5;
            self.input.scroll_delta = 0.0;
        }

        // Clamp camera within bounds
        let world_size = self.world.world_size();
        self.camera.position.x = self.camera.position.x.clamp(-2.0, world_size.x + 2.0);
        self.camera.position.z = self.camera.position.z.clamp(-2.0, world_size.z + 2.0);
        self.camera.position.y = self.camera.position.y.clamp(0.5, 20.0);
    }

    fn screen_to_world_ray(&self, screen_x: f32, screen_y: f32) -> Vec3 {
        let ndc_x = (2.0 * screen_x / self.window_size.0 as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / self.window_size.1 as f32);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let inv_vp = (proj * view).inverse();

        let near = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;

        (far - near).normalize()
    }

    fn projection_matrix(&self) -> Mat4 {
        let aspect = self.window_size.0 as f32 / self.window_size.1 as f32;
        Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.01, 100.0)
    }

    fn raycast_terrain(&self) -> Option<Vec3> {
        let ray_dir = self.screen_to_world_ray(self.input.mouse_pos.0, self.input.mouse_pos.1);
        let ray_origin = self.camera.position;

        let step = 0.02;
        let max_dist = 50.0;

        let mut t = 0.0;
        while t < max_dist {
            let p = ray_origin + ray_dir * t;

            if let Some((x, z)) = self.world.world_to_cell(p) {
                let ground = self.world.ground_height(x, z);
                if p.y <= ground {
                    return Some(Vec3::new(p.x, ground, p.z));
                }
            }

            t += step;
        }

        None
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        self.window_size = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let heightfield = GpuHeightfield::new(
            &device,
            self.world.width as u32,
            self.world.depth as u32,
            self.world.cell_size,
            BASE_HEIGHT,
            config.format,
        );
        heightfield.upload_from_world(&queue, &self.world);

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            heightfield,
            depth_texture,
            depth_view,
        });

        println!("=== TERRAIN COLLAPSE DEMO ===");
        println!("Controls:");
        println!("  1 - Add Sand pile (expected: {:.1}°)", TerrainMaterial::Sand.angle_of_repose().to_degrees());
        println!("  2 - Add Dirt pile (expected: {:.1}°)", TerrainMaterial::Dirt.angle_of_repose().to_degrees());
        println!("  3 - Add Gravel pile (expected: {:.1}°)", TerrainMaterial::Gravel.angle_of_repose().to_degrees());
        println!("  C - Toggle collapse simulation");
        println!("  R - Reset world");
        println!("  WASD/Space/Shift - Move camera");
        println!("  Right Mouse - Look\n");
    }

    fn render(&mut self) {
        let window = match self.window.as_ref() {
            Some(window) => window.clone(),
            None => return,
        };

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.update(dt);

        let view = self.camera.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;

        let Some(gpu) = self.gpu.as_mut() else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        let frame_view = output.texture.create_view(&Default::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render using GpuHeightfield's render method
        gpu.heightfield.render(
            &mut encoder,
            &frame_view,
            &gpu.depth_view,
            &gpu.queue,
            view_proj.to_cols_array_2d(),
            self.camera.position.to_array(),
            self.start_time.elapsed().as_secs_f32(),
        );

        gpu.queue.submit(Some(encoder.finish()));
        output.present();

        window.request_redraw();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = (new_size.width, new_size.height);
            if let Some(gpu) = &mut self.gpu {
                gpu.config.width = new_size.width;
                gpu.config.height = new_size.height;
                gpu.surface.configure(&gpu.device, &gpu.config);

                // Recreate depth texture
                let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Depth Texture"),
                    size: wgpu::Extent3d {
                        width: gpu.config.width,
                        height: gpu.config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                gpu.depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.depth_texture = depth_texture;
            }
        }
    }
}

fn build_flat_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 0.0);

    // Set flat terrain - just bedrock base
    let cell_count = WORLD_WIDTH * WORLD_DEPTH;
    world.bedrock_elevation = vec![BASE_HEIGHT; cell_count];
    world.paydirt_thickness = vec![0.0; cell_count];
    world.gravel_thickness = vec![0.0; cell_count];
    world.overburden_thickness = vec![0.0; cell_count];
    world.terrain_sediment = vec![0.0; cell_count];

    // Water surface = ground height (no water depth)
    let ground_height = BASE_HEIGHT;
    world.water_surface = vec![ground_height; cell_count];

    world
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Terrain Collapse Demo - Angle of Repose")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
                )
                .unwrap(),
        );

        self.window = Some(window.clone());
        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.resize(size);
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.input.keys.insert(code);

                            match code {
                                KeyCode::Escape => event_loop.exit(),
                                KeyCode::KeyR => self.reset_world(),
                                KeyCode::KeyC => {
                                    self.collapse_enabled = !self.collapse_enabled;
                                    println!(
                                        "Collapse simulation: {}",
                                        if self.collapse_enabled { "ON" } else { "OFF" }
                                    );
                                }
                                KeyCode::Digit1 => self.add_pile_at_cursor(0), // Sediment
                                KeyCode::Digit2 => self.add_pile_at_cursor(1), // Overburden
                                KeyCode::Digit3 => self.add_pile_at_cursor(2), // Gravel
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.input.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Right => {
                    self.input.mouse_look = state == ElementState::Pressed;
                    if state == ElementState::Released {
                        self.input.last_mouse_pos = None;
                    }
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_pos = (position.x as f32, position.y as f32);

                if self.input.mouse_look {
                    if let Some(last) = self.input.last_mouse_pos {
                        let dx = position.x - last.0;
                        let dy = position.y - last.1;
                        self.camera.yaw += dx as f32 * self.camera.sensitivity;
                        self.camera.pitch -= dy as f32 * self.camera.sensitivity;
                        self.camera.pitch = self.camera.pitch.clamp(-1.4, 1.4);
                    }
                    self.input.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.input.scroll_delta += scroll;
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
