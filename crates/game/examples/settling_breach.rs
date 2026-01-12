//! Settling Pond Breach Simulation
//!
//! Demonstrates mid-scale dynamics:
//! - Wash plant ejecting silty water into settling pond
//! - Silt deposits in calm water (Stokes settling)
//! - Pond fills until overflow
//! - Critical shear stress creates runaway breach through weak berm
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look around
//! - 1: Increase inflow rate
//! - 2: Decrease inflow rate

use bytemuck::{Pod, Zeroable};
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

const WORLD_WIDTH: usize = 128;
const WORLD_DEPTH: usize = 128;
const CELL_SIZE: f32 = 0.25; // 25cm cells for finer detail

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    cell_size: f32,
    grid_width: u32,
    grid_depth: u32,
    time: f32,
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

    // Simulation state
    inflow_rate: f32,      // m³/s of silty water
    inflow_position: (usize, usize),
    total_inflow: f32,
    breach_detected: bool,
}

impl App {
    fn new() -> Self {
        // Settling pond inlet position (northwest corner of pond)
        let inflow_pos = (30, 30);

        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            config: None,
            heightfield: None,
            world: build_settling_pond_world(),
            camera: Camera {
                position: Vec3::new(16.0, 15.0, 16.0),
                yaw: 0.8,
                pitch: -0.5,
                speed: 10.0,
                sensitivity: 0.003,
            },
            input: InputState {
                keys: HashSet::new(),
                mouse_look: false,
                last_mouse_pos: None,
            },
            depth_texture: None,
            depth_view: None,
            last_frame: Instant::now(),
            start_time: Instant::now(),
            inflow_rate: 0.5, // Start with moderate flow
            inflow_position: inflow_pos,
            total_inflow: 0.0,
            breach_detected: false,
        }
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
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if let (Some(device), Some(surface), Some(config)) =
            (&self.device, &self.surface, &mut self.config)
        {
            config.width = new_size.width.max(1);
            config.height = new_size.height.max(1);
            surface.configure(device, config);

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
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

}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Settling Pond Breach - Press 1/2 to adjust flow")
                )
                .unwrap(),
        );
        self.window = Some(window.clone());

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .unwrap();

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Simulation Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_storage_buffers_per_shader_stage: 16,
                            ..wgpu::Limits::default()
                        }
                        .using_resolution(adapter.limits()),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .unwrap();

            let size = window.inner_size();
            let config = surface
                .get_default_config(&adapter, size.width, size.height)
                .unwrap();
            surface.configure(&device, &config);

            // Initialize Heightfield Simulation
            let mut hf = GpuHeightfield::new(
                &device,
                WORLD_WIDTH as u32,
                WORLD_DEPTH as u32,
                CELL_SIZE,
                10.0, // Max height
                config.format,
            );

            // Upload initial terrain
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
            WindowEvent::KeyboardInput {
                event: kb_input, ..
            } => {
                if let PhysicalKey::Code(code) = kb_input.physical_key {
                    if kb_input.state == ElementState::Pressed {
                        self.input.keys.insert(code);
                        match code {
                            KeyCode::Escape => event_loop.exit(),
                            KeyCode::Digit1 => {
                                self.inflow_rate = (self.inflow_rate + 0.2).min(3.0);
                                println!("Inflow rate: {:.1} m³/s", self.inflow_rate);
                            }
                            KeyCode::Digit2 => {
                                self.inflow_rate = (self.inflow_rate - 0.2).max(0.0);
                                println!("Inflow rate: {:.1} m³/s", self.inflow_rate);
                            }
                            _ => {}
                        }
                    } else {
                        self.input.keys.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.input.mouse_look = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = (position.x, position.y);
                if self.input.mouse_look {
                    if let Some(last_pos) = self.input.last_mouse_pos {
                        let dx = (current_pos.0 - last_pos.0) as f32;
                        let dy = (current_pos.1 - last_pos.1) as f32;
                        self.camera.yaw += dx * self.camera.sensitivity;
                        self.camera.pitch -= dy * self.camera.sensitivity;
                        self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
                    }
                }
                self.input.last_mouse_pos = Some(current_pos);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;

                self.update_camera(dt);

                let sim_dt = 0.016;

                let device = self.device.as_ref().unwrap();
                let queue = self.queue.as_ref().unwrap();
                let surface = self.surface.as_ref().unwrap();
                let hf = self.heightfield.as_mut().unwrap();

                // CRITICAL: Update simulation parameters (dt, cell_size, gravity, damping)
                hf.update_params(queue, sim_dt);

                // Use emitter to inject water at inflow point instead of overwriting all buffers!
                // Position in world coordinates
                let (ix, iz) = self.inflow_position;
                let pos_x = ix as f32 * CELL_SIZE;
                let pos_z = iz as f32 * CELL_SIZE;
                hf.update_emitter(
                    queue,
                    pos_x,
                    pos_z,
                    2.0 * CELL_SIZE,       // radius
                    self.inflow_rate,      // rate (m³/s -> converted to depth/s in shader)
                    sim_dt,
                    self.inflow_rate > 0.0, // enabled
                );

                // Create encoder for simulation
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                // Inject water via emitter
                hf.dispatch_emitter(&mut encoder);

                // Step heightfield simulation (SWE + erosion + collapse)
                hf.dispatch_tile(&mut encoder, WORLD_WIDTH as u32, WORLD_DEPTH as u32);

                // Render
                let frame = surface.get_current_texture().unwrap();
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

                hf.render(
                    &mut encoder,
                    &view,
                    &depth_view,
                    queue,
                    view_proj.to_cols_array_2d(),
                    self.camera.position.into(),
                    self.start_time.elapsed().as_secs_f32(),
                );

                queue.submit(std::iter::once(encoder.finish()));
                frame.present();

                // Print status periodically
                let elapsed = self.start_time.elapsed().as_secs_f32();
                if elapsed as u32 % 5 == 0 && (elapsed * 60.0) as u32 % 60 < 2 {
                    println!(
                        "Time: {:.0}s | Total inflow: {:.1}m³ | Rate: {:.1}m³/s",
                        elapsed, self.total_inflow, self.inflow_rate
                    );
                }

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    println!("=== Settling Pond Breach Demo ===");
    println!("This demonstrates critical shear stress erosion:");
    println!("- Silty water flows into settling pond");
    println!("- Silt settles in calm water, building up sediment");
    println!("- Water overflows through the weak point in the berm");
    println!("- Once flow exceeds critical velocity, erosion accelerates");
    println!("- Positive feedback creates runaway breach");
    println!();
    println!("Controls:");
    println!("  WASD - Move camera");
    println!("  Right Mouse - Look around");
    println!("  1/2 - Increase/decrease inflow rate");
    println!();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

/// Build a settling pond terrain with:
/// - A basin (the settling pond)
/// - Berms on 3 sides made of sediment/overburden
/// - A weak point in the south berm (thin, made of loose sediment)
/// - Downhill slope beyond the berm for the flood path
fn build_settling_pond_world() -> World {
    let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, 10.0);

    // Pond center and dimensions
    let pond_cx = 48;
    let pond_cz = 48;
    let pond_radius = 20;
    let pond_depth = 2.0; // 2m deep pond

    // Berm dimensions
    let berm_height = 1.5;  // 1.5m tall berm
    let berm_width = 6;     // 6 cells wide

    // Weak point location (center of south berm)
    let weak_width = 4; // 4 cells wide notch

    for z in 0..WORLD_DEPTH {
        for x in 0..WORLD_WIDTH {
            let idx = world.idx(x, z);

            // Base terrain: gentle slope down to the south
            let base_slope = (z as f32 / WORLD_DEPTH as f32) * 3.0;
            let base_height = 5.0 - base_slope;

            // Distance from pond center
            let dx = x as i32 - pond_cx as i32;
            let dz = z as i32 - pond_cz as i32;
            let dist = ((dx * dx + dz * dz) as f32).sqrt();

            // Inside the pond basin
            let in_pond = dist < pond_radius as f32;

            // On the berm (ring around pond)
            let on_berm = dist >= pond_radius as f32 && dist < (pond_radius + berm_width) as f32;

            // Check if this is the weak point in the south berm
            let is_weak_point = on_berm
                && dz > 0  // South side
                && (dx.abs() as usize) < weak_width;

            if in_pond {
                // Pond floor - dig into overburden
                let dig_factor = 1.0 - (dist / pond_radius as f32);
                let dig_depth = pond_depth * dig_factor.sqrt();

                world.bedrock_elevation[idx] = base_height * 0.4;
                world.paydirt_thickness[idx] = base_height * 0.2;
                world.gravel_thickness[idx] = 0.1;
                world.overburden_thickness[idx] = (base_height * 0.3 - dig_depth).max(0.0);
                world.terrain_sediment[idx] = 0.0;
            } else if on_berm {
                // Berm - built up material
                world.bedrock_elevation[idx] = base_height * 0.4;
                world.paydirt_thickness[idx] = base_height * 0.2;
                world.gravel_thickness[idx] = 0.1;

                if is_weak_point {
                    // Weak point: lower, made of loose sediment only
                    let notch_depth = (weak_width as f32 - dx.abs() as f32) / weak_width as f32;
                    let weak_height = berm_height * 0.3 * notch_depth; // Much lower
                    world.overburden_thickness[idx] = base_height * 0.2;
                    world.terrain_sediment[idx] = weak_height; // Loose sediment, easy to erode
                } else {
                    // Normal berm: compacted overburden
                    let berm_dist = dist - pond_radius as f32;
                    let berm_factor = 1.0 - (berm_dist / berm_width as f32);
                    world.overburden_thickness[idx] = base_height * 0.3 + berm_height * berm_factor;
                    world.terrain_sediment[idx] = 0.0;
                }
            } else {
                // Outside pond - normal terrain sloping away
                world.bedrock_elevation[idx] = base_height * 0.4;
                world.paydirt_thickness[idx] = base_height * 0.2;
                world.gravel_thickness[idx] = 0.1;
                world.overburden_thickness[idx] = base_height * 0.3;
                world.terrain_sediment[idx] = 0.0;
            }
        }
    }

    // Add initial water in the pond (partially filled)
    // water_surface is absolute height, so we need ground_height + desired_depth
    for z in 0..WORLD_DEPTH {
        for x in 0..WORLD_WIDTH {
            let dx = x as i32 - pond_cx as i32;
            let dz = z as i32 - pond_cz as i32;
            let dist = ((dx * dx + dz * dz) as f32).sqrt();

            if dist < (pond_radius - 2) as f32 {
                // Start with some water (0.3m depth)
                let ground_h = world.ground_height(x, z);
                let idx = world.idx(x, z);
                world.water_surface[idx] = ground_h + 0.3;
            }
        }
    }

    world
}
