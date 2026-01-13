//! Visual DEM Physics Tests
//!
//! Run: cargo run --example vtest_dem --release
//!
//! WINDOW TITLE shows: Test name | Expected | What to watch
//! CONSOLE shows: Live metrics
//!
//! Controls:
//!   1-9         Switch test
//!   SPACE       Pause/resume
//!   R           Reset test
//!   WASD        Move camera
//!   Mouse drag  Rotate camera
//!   Scroll      Zoom
//!   ESC         Quit

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// ============================================================================
// TEST DEFINITIONS
// ============================================================================

struct TestDef {
    key: char,
    name: &'static str,
    unit_test: &'static str,
    expect: &'static str,
    watch: &'static str,
}

const TESTS: &[TestDef] = &[
    TestDef {
        key: '1',
        name: "Floor Collision",
        unit_test: "test_dem_floor_collision",
        expect: "Bounce ~4cm then settle",
        watch: "min_y >= 0.02",
    },
    TestDef {
        key: '2',
        name: "Wall Collision",
        unit_test: "test_dem_wall_collision",
        expect: "Hit wall, bounce back",
        watch: "v_x flips sign",
    },
    TestDef {
        key: '3',
        name: "Clump Collision",
        unit_test: "test_dem_clump_collision",
        expect: "Head-on, both reverse",
        watch: "momentum ~0",
    },
    TestDef {
        key: '4',
        name: "No Penetration",
        unit_test: "test_dem_no_penetration",
        expect: "100 clumps, none through floor",
        watch: "penetrations = 0",
    },
    TestDef {
        key: '5',
        name: "Static Friction",
        unit_test: "test_dem_static_friction",
        expect: "At rest stays still",
        watch: "drift < 2cm",
    },
    TestDef {
        key: '6',
        name: "Kinetic Friction",
        unit_test: "test_dem_kinetic_friction",
        expect: "Sliding slows down",
        watch: "v < 50% initial",
    },
    TestDef {
        key: '7',
        name: "Wet vs Dry",
        unit_test: "test_dem_wet_vs_dry",
        expect: "Blue(wet) slides 2x+ farther",
        watch: "wet/dry > 2",
    },
    TestDef {
        key: '8',
        name: "Settling",
        unit_test: "test_dem_settling_time",
        expect: "All settle in 5s",
        watch: "avg_vel < 0.05",
    },
    TestDef {
        key: '9',
        name: "Density Separation",
        unit_test: "test_dem_density_separation",
        expect: "Gold(yellow) sinks below gray",
        watch: "gold_y < gray_y",
    },
];

// ============================================================================
// CONSTANTS
// ============================================================================

const RADIUS: f32 = 0.025;
const MASS: f32 = 0.1;
const GOLD_MASS: f32 = 0.7;
const GRAVITY: f32 = -9.81;
const DT: f32 = 1.0 / 120.0;
const SUBSTEPS: usize = 2;

// ============================================================================
// APP
// ============================================================================

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,

    // Simulation
    sim: ClusterSimulation3D,
    sim_dry: Option<ClusterSimulation3D>, // For wet vs dry comparison
    test_idx: usize,
    paused: bool,
    frame: u32,

    // Camera (from washplant_editor)
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vec3,
    mouse_pressed: bool,
    last_mouse: (f64, f64),

    // Tracking for specific tests
    gold_indices: Vec<usize>,
    gangue_indices: Vec<usize>,
    wet_start_x: f32,
    dry_start_x: f32,
    max_bounce_y: f32,
    initial_vel: f32,
}

impl App {
    fn new() -> Self {
        let mut app = Self {
            window: None,
            gpu: None,
            sim: ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(2.0)),
            sim_dry: None,
            test_idx: 0,
            paused: false,
            frame: 0,
            camera_yaw: 0.8,
            camera_pitch: 0.5,
            camera_distance: 4.0,
            camera_target: Vec3::new(1.0, 0.3, 1.0),
            mouse_pressed: false,
            last_mouse: (0.0, 0.0),
            gold_indices: Vec::new(),
            gangue_indices: Vec::new(),
            wet_start_x: 0.0,
            dry_start_x: 0.0,
            max_bounce_y: 0.0,
            initial_vel: 0.0,
        };
        app.setup_test(0);
        app
    }

    fn camera_pos(&self) -> Vec3 {
        self.camera_target
            + Vec3::new(
                self.camera_distance * self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_distance * self.camera_pitch.sin(),
                self.camera_distance * self.camera_yaw.sin() * self.camera_pitch.cos(),
            )
    }

    fn update_title(&self) {
        if let Some(w) = &self.window {
            let t = &TESTS[self.test_idx];
            let status = if self.paused { " [PAUSED]" } else { "" };
            let title = format!(
                "TEST {}: {} | EXPECT: {} | WATCH: {}{}",
                t.key, t.name, t.expect, t.watch, status
            );
            w.set_title(&title);
        }
    }

    fn setup_test(&mut self, idx: usize) {
        self.test_idx = idx;
        self.frame = 0;
        self.max_bounce_y = 0.0;
        self.initial_vel = 0.0;
        self.gold_indices.clear();
        self.gangue_indices.clear();
        self.sim_dry = None;

        let bounds = 2.0;
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(bounds));
        sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

        let tpl = ClumpTemplate3D::generate(ClumpShape3D::Tetra, RADIUS, MASS);
        let tpl_idx = sim.add_template(tpl);

        let t = &TESTS[idx];

        // Big clear console banner
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║  TEST {}: {:<48} ║", t.key, t.name);
        println!("║  Unit test: {:<44} ║", t.unit_test);
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║  EXPECT: {:<47} ║", t.expect);
        println!("║  WATCH:  {:<47} ║", t.watch);
        println!("╚══════════════════════════════════════════════════════════╝");
        println!();

        match idx {
            0 => {
                // Floor collision
                sim.spawn(tpl_idx, Vec3::new(1.0, 1.0 + RADIUS, 1.0), Vec3::ZERO);
            }
            1 => {
                // Wall collision
                sim.gravity = Vec3::ZERO;
                sim.spawn(tpl_idx, Vec3::new(0.3, 1.0, 1.0), Vec3::new(3.0, 0.0, 0.0));
                self.initial_vel = 3.0;
            }
            2 => {
                // Clump collision
                sim.gravity = Vec3::ZERO;
                sim.spawn(tpl_idx, Vec3::new(0.5, 1.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
                sim.spawn(tpl_idx, Vec3::new(1.5, 1.0, 1.0), Vec3::new(-1.0, 0.0, 0.0));
            }
            3 => {
                // No penetration
                for i in 0..10 {
                    for j in 0..10 {
                        let x = 0.3 + i as f32 * 0.14;
                        let z = 0.3 + j as f32 * 0.14;
                        sim.spawn(tpl_idx, Vec3::new(x, 1.5, z), Vec3::ZERO);
                    }
                }
            }
            4 => {
                // Static friction
                sim.spawn(tpl_idx, Vec3::new(1.0, RADIUS * 2.0, 1.0), Vec3::ZERO);
            }
            5 => {
                // Kinetic friction
                sim.spawn(
                    tpl_idx,
                    Vec3::new(0.3, RADIUS * 2.0, 1.0),
                    Vec3::new(2.0, 0.0, 0.0),
                );
                self.initial_vel = 2.0;
            }
            6 => {
                // Wet vs dry
                sim.wet_friction = 0.08;
                sim.floor_friction = 0.5;
                sim.spawn(
                    tpl_idx,
                    Vec3::new(0.3, RADIUS * 2.0, 0.8),
                    Vec3::new(1.5, 0.0, 0.0),
                );
                self.wet_start_x = 0.3;

                let mut dry = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(bounds));
                dry.gravity = Vec3::new(0.0, GRAVITY, 0.0);
                dry.floor_friction = 0.5;
                dry.wet_friction = 0.5;
                let dry_tpl = ClumpTemplate3D::generate(ClumpShape3D::Tetra, RADIUS, MASS);
                let dry_tpl_idx = dry.add_template(dry_tpl);
                dry.spawn(
                    dry_tpl_idx,
                    Vec3::new(0.3, RADIUS * 2.0, 1.2),
                    Vec3::new(1.5, 0.0, 0.0),
                );
                self.dry_start_x = 0.3;
                self.sim_dry = Some(dry);
                println!("Blue = WET (μ=0.08), Orange = DRY (μ=0.5)");
            }
            7 => {
                // Settling
                for i in 0..20 {
                    let x = 0.7 + (i % 5) as f32 * 0.12;
                    let z = 0.7 + (i / 5) as f32 * 0.12;
                    sim.spawn(tpl_idx, Vec3::new(x, 1.0, z), Vec3::ZERO);
                }
            }
            8 => {
                // Density separation
                let gold_tpl = ClumpTemplate3D::generate(ClumpShape3D::Tetra, RADIUS, GOLD_MASS);
                let gold_idx = sim.add_template(gold_tpl);

                for i in 0..10 {
                    let angle = i as f32 * 0.628;
                    let r = 0.3;
                    let x = 1.0 + r * angle.cos();
                    let z = 1.0 + r * angle.sin();
                    let gi =
                        sim.spawn(gold_idx, Vec3::new(x, 1.5 + i as f32 * 0.06, z), Vec3::ZERO);
                    self.gold_indices.push(gi);
                    let si = sim.spawn(
                        tpl_idx,
                        Vec3::new(x + 0.06, 1.55 + i as f32 * 0.06, z + 0.06),
                        Vec3::ZERO,
                    );
                    self.gangue_indices.push(si);
                }
                println!("Yellow = GOLD (heavy), Gray = GANGUE (light)");
            }
            _ => {}
        }

        self.sim = sim;
        self.update_title();
    }

    fn print_metrics(&self) {
        if self.frame % 30 != 0 {
            return;
        }

        match self.test_idx {
            0 => {
                if let Some(c) = self.sim.clumps.first() {
                    let settled = c.velocity.length() < 0.05;
                    println!(
                        "y={:.4} vy={:.3} max_bounce={:.4} {}",
                        c.position.y,
                        c.velocity.y,
                        self.max_bounce_y,
                        if settled { "✓ SETTLED" } else { "" }
                    );
                }
            }
            1 => {
                if let Some(c) = self.sim.clumps.first() {
                    let reflected = c.velocity.x < 0.0;
                    println!(
                        "x={:.3} vx={:.3} {}",
                        c.position.x,
                        c.velocity.x,
                        if reflected { "✓ REFLECTED" } else { "" }
                    );
                }
            }
            2 => {
                if self.sim.clumps.len() >= 2 {
                    let va = self.sim.clumps[0].velocity.x;
                    let vb = self.sim.clumps[1].velocity.x;
                    let p = MASS * (va + vb);
                    let ok = va < 0.0 && vb > 0.0;
                    println!(
                        "v_a={:.3} v_b={:.3} momentum={:.4} {}",
                        va,
                        vb,
                        p,
                        if ok { "✓ BOTH REVERSED" } else { "" }
                    );
                }
            }
            3 => {
                let min_y = self
                    .sim
                    .clumps
                    .iter()
                    .map(|c| c.position.y)
                    .fold(f32::MAX, f32::min);
                let bad = self
                    .sim
                    .clumps
                    .iter()
                    .filter(|c| c.position.y < RADIUS * 0.9)
                    .count();
                println!(
                    "min_y={:.4} penetrations={} {}",
                    min_y,
                    bad,
                    if bad == 0 { "✓ OK" } else { "✗ FAIL" }
                );
            }
            4 => {
                if let Some(c) = self.sim.clumps.first() {
                    let drift = (c.position.x - 1.0).hypot(c.position.z - 1.0);
                    println!(
                        "drift={:.4}m {}",
                        drift,
                        if drift < 0.02 { "✓ STABLE" } else { "" }
                    );
                }
            }
            5 => {
                if let Some(c) = self.sim.clumps.first() {
                    let ratio = c.velocity.x / self.initial_vel;
                    println!(
                        "vx={:.3} ratio={:.2} {}",
                        c.velocity.x,
                        ratio,
                        if ratio < 0.5 { "✓ DECELERATED" } else { "" }
                    );
                }
            }
            6 => {
                let wet_d = self
                    .sim
                    .clumps
                    .first()
                    .map(|c| c.position.x - self.wet_start_x)
                    .unwrap_or(0.0);
                let dry_d = self
                    .sim_dry
                    .as_ref()
                    .and_then(|s| s.clumps.first())
                    .map(|c| c.position.x - self.dry_start_x)
                    .unwrap_or(0.001);
                let ratio = wet_d / dry_d.max(0.001);
                println!(
                    "wet={:.3}m dry={:.3}m ratio={:.2}x {}",
                    wet_d,
                    dry_d,
                    ratio,
                    if ratio > 2.0 {
                        "✓ WET SLIDES MORE"
                    } else {
                        ""
                    }
                );
            }
            7 => {
                let avg: f32 = self
                    .sim
                    .clumps
                    .iter()
                    .map(|c| c.velocity.length())
                    .sum::<f32>()
                    / self.sim.clumps.len().max(1) as f32;
                let max = self
                    .sim
                    .clumps
                    .iter()
                    .map(|c| c.velocity.length())
                    .fold(0.0f32, f32::max);
                println!(
                    "avg_vel={:.4} max_vel={:.4} {}",
                    avg,
                    max,
                    if avg < 0.05 && max < 0.1 {
                        "✓ SETTLED"
                    } else {
                        ""
                    }
                );
            }
            8 => {
                let gold_y: f32 = self
                    .gold_indices
                    .iter()
                    .filter_map(|&i| self.sim.clumps.get(i))
                    .map(|c| c.position.y)
                    .sum::<f32>()
                    / self.gold_indices.len().max(1) as f32;
                let gang_y: f32 = self
                    .gangue_indices
                    .iter()
                    .filter_map(|&i| self.sim.clumps.get(i))
                    .map(|c| c.position.y)
                    .sum::<f32>()
                    / self.gangue_indices.len().max(1) as f32;
                let sep = gang_y - gold_y;
                println!(
                    "gold_y={:.3} gangue_y={:.3} sep={:.3}m {}",
                    gold_y,
                    gang_y,
                    sep,
                    if sep > 0.05 { "✓ SEPARATED" } else { "" }
                );
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.paused {
            return;
        }

        for _ in 0..SUBSTEPS {
            self.sim.step(DT);
            if let Some(dry) = &mut self.sim_dry {
                dry.step(DT);
            }
        }
        self.frame += 1;

        // Track max bounce for floor test
        if self.test_idx == 0 {
            if let Some(c) = self.sim.clumps.first() {
                if c.velocity.y > 0.0 && c.position.y > self.max_bounce_y {
                    self.max_bounce_y = c.position.y;
                }
            }
        }

        self.print_metrics();
    }
}

// ============================================================================
// RENDERING
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    norm: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    pos: [f32; 3],
    scale: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    floor_pipeline: wgpu::RenderPipeline,
    mesh_verts: wgpu::Buffer,
    mesh_count: u32,
    floor_verts: wgpu::Buffer,
    floor_count: u32,
    instances: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let w = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Loading...")
                    .with_inner_size(winit::dpi::LogicalSize::new(1400, 800)),
            )
            .unwrap(),
        );
        self.window = Some(w.clone());
        pollster::block_on(self.init_gpu(w));

        println!("\n══════════════════════════════════════════════════════════");
        println!("  VISUAL DEM PHYSICS TESTS");
        println!("══════════════════════════════════════════════════════════");
        println!("  1-9       = Switch test");
        println!("  SPACE     = Pause/resume");
        println!("  R         = Reset current test");
        println!("  WASD      = Move camera");
        println!("  Mouse     = Drag to rotate");
        println!("  Scroll    = Zoom in/out");
        println!("  ESC       = Quit");
        println!("══════════════════════════════════════════════════════════\n");

        self.setup_test(0);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    let dx = position.x - self.last_mouse.0;
                    let dy = position.y - self.last_mouse.1;
                    self.camera_yaw += dx as f32 * 0.005;
                    self.camera_pitch = (self.camera_pitch + dy as f32 * 0.005).clamp(0.1, 1.4);
                }
                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.3).clamp(1.0, 20.0);
            }

            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    let move_speed = 0.15;
                    let fwd = Vec3::new(-self.camera_yaw.sin(), 0.0, -self.camera_yaw.cos());
                    let right = Vec3::new(self.camera_yaw.cos(), 0.0, -self.camera_yaw.sin());

                    match k {
                        KeyCode::Digit1 => self.setup_test(0),
                        KeyCode::Digit2 => self.setup_test(1),
                        KeyCode::Digit3 => self.setup_test(2),
                        KeyCode::Digit4 => self.setup_test(3),
                        KeyCode::Digit5 => self.setup_test(4),
                        KeyCode::Digit6 => self.setup_test(5),
                        KeyCode::Digit7 => self.setup_test(6),
                        KeyCode::Digit8 => self.setup_test(7),
                        KeyCode::Digit9 => self.setup_test(8),
                        KeyCode::Space => {
                            self.paused = !self.paused;
                            self.update_title();
                        }
                        KeyCode::KeyR => self.setup_test(self.test_idx),
                        KeyCode::KeyW => self.camera_target += fwd * move_speed,
                        KeyCode::KeyS => self.camera_target -= fwd * move_speed,
                        KeyCode::KeyA => self.camera_target -= right * move_speed,
                        KeyCode::KeyD => self.camera_target += right * move_speed,
                        KeyCode::KeyQ => self.camera_target.y -= move_speed,
                        KeyCode::KeyE => self.camera_target.y += move_speed,
                        KeyCode::Escape => el.exit(),
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                self.step();
                self.render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout { array_stride: 24, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3] },
                    wgpu::VertexBufferLayout { array_stride: 32, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32, 4 => Float32x4] },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(caps.formats[0].into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: depth_stencil.clone(),
            multisample: Default::default(), multiview: None, cache: None,
        });

        let floor_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_floor"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_floor"),
                targets: &[Some(caps.formats[0].into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sphere = make_sphere(1);
        let floor = make_floor(5.0);

        let mesh_verts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&sphere),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let floor_verts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&floor),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32 * 500,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
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

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            floor_pipeline,
            mesh_verts,
            mesh_count: sphere.len() as u32,
            floor_verts,
            floor_count: floor.len() as u32,
            instances,
            uniforms,
            bind_group,
            depth_view: depth.create_view(&Default::default()),
        });
    }

    fn render(&mut self) {
        let window = match &self.window {
            Some(w) => w.clone(),
            None => return,
        };
        let gpu = match self.gpu.take() {
            Some(g) => g,
            None => return,
        };

        let eye = self.camera_pos();
        let view = Mat4::look_at_rh(eye, self.camera_target, Vec3::Y);
        let size = window.inner_size();
        let proj = Mat4::perspective_rh(
            45f32.to_radians(),
            size.width as f32 / size.height.max(1) as f32,
            0.1,
            100.0,
        );
        gpu.queue.write_buffer(
            &gpu.uniforms,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: (proj * view).to_cols_array_2d(),
            }),
        );

        // Build instances
        let mut insts: Vec<Instance> = Vec::new();
        for (i, c) in self.sim.clumps.iter().enumerate() {
            let color = if self.gold_indices.contains(&i) {
                [1.0, 0.85, 0.2, 1.0] // Gold - bright yellow
            } else if self.test_idx == 6 {
                [0.3, 0.5, 0.9, 1.0] // Wet - blue
            } else {
                [0.6, 0.6, 0.65, 1.0] // Default gray
            };
            insts.push(Instance {
                pos: c.position.to_array(),
                scale: RADIUS,
                color,
            });
        }
        if let Some(dry) = &self.sim_dry {
            for c in &dry.clumps {
                insts.push(Instance {
                    pos: c.position.to_array(),
                    scale: RADIUS,
                    color: [0.9, 0.5, 0.2, 1.0],
                }); // Dry - orange
            }
        }

        if !insts.is_empty() {
            gpu.queue
                .write_buffer(&gpu.instances, 0, bytemuck::cast_slice(&insts));
        }

        let out = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => {
                self.gpu = Some(gpu);
                return;
            }
        };
        let view = out.texture.create_view(&Default::default());
        let mut enc = gpu.device.create_command_encoder(&Default::default());

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&gpu.floor_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.floor_verts.slice(..));
            pass.draw(0..gpu.floor_count, 0..1);

            if !insts.is_empty() {
                pass.set_pipeline(&gpu.pipeline);
                pass.set_vertex_buffer(0, gpu.mesh_verts.slice(..));
                pass.set_vertex_buffer(1, gpu.instances.slice(..));
                pass.draw(0..gpu.mesh_count, 0..insts.len() as u32);
            }
        }

        gpu.queue.submit(Some(enc.finish()));
        out.present();
        self.gpu = Some(gpu);
    }
}

fn make_sphere(sub: u32) -> Vec<Vertex> {
    let phi = (1.0 + 5f32.sqrt()) / 2.0;
    let v: Vec<Vec3> = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ];
    let mut out = Vec::new();
    fn subdiv(a: Vec3, b: Vec3, c: Vec3, d: u32, o: &mut Vec<Vertex>) {
        if d == 0 {
            for p in [a, b, c] {
                o.push(Vertex {
                    pos: p.to_array(),
                    norm: p.to_array(),
                });
            }
        } else {
            let ab = ((a + b) / 2.0).normalize();
            let bc = ((b + c) / 2.0).normalize();
            let ca = ((c + a) / 2.0).normalize();
            subdiv(a, ab, ca, d - 1, o);
            subdiv(b, bc, ab, d - 1, o);
            subdiv(c, ca, bc, d - 1, o);
            subdiv(ab, bc, ca, d - 1, o);
        }
    }
    for (i, j, k) in f {
        subdiv(v[i], v[j], v[k], sub, &mut out);
    }
    out
}

fn make_floor(s: f32) -> Vec<Vertex> {
    let n = [0.0, 1.0, 0.0];
    vec![
        Vertex {
            pos: [-s, 0.0, -s],
            norm: n,
        },
        Vertex {
            pos: [s, 0.0, -s],
            norm: n,
        },
        Vertex {
            pos: [s, 0.0, s],
            norm: n,
        },
        Vertex {
            pos: [-s, 0.0, -s],
            norm: n,
        },
        Vertex {
            pos: [s, 0.0, s],
            norm: n,
        },
        Vertex {
            pos: [-s, 0.0, s],
            norm: n,
        },
    ]
}

const SHADER: &str = r#"
struct U { vp: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: U;
struct V { @location(0) p: vec3f, @location(1) n: vec3f }
struct I { @location(2) pos: vec3f, @location(3) s: f32, @location(4) col: vec4f }
struct O { @builtin(position) p: vec4f, @location(0) c: vec4f, @location(1) n: vec3f, @location(2) wp: vec3f }

@vertex fn vs_main(v: V, i: I) -> O {
    let wp = v.p * i.s + i.pos;
    return O(u.vp * vec4f(wp, 1.0), i.col, v.n, wp);
}
@fragment fn fs_main(i: O) -> @location(0) vec4f {
    let l = normalize(vec3f(0.5, 1.0, 0.3));
    let d = max(dot(normalize(i.n), l), 0.0) * 0.7 + 0.3;
    return vec4f(i.c.rgb * d, 1.0);
}
@vertex fn vs_floor(v: V) -> O { return O(u.vp * vec4f(v.p, 1.0), vec4f(0.15, 0.18, 0.15, 1.0), v.n, v.p); }
@fragment fn fs_floor(i: O) -> @location(0) vec4f {
    let g = abs(fract(i.wp.x * 2.0) - 0.5) + abs(fract(i.wp.z * 2.0) - 0.5);
    let c = mix(vec3f(0.1, 0.12, 0.1), vec3f(0.18, 0.22, 0.18), smoothstep(0.4, 0.5, g));
    return vec4f(c, 1.0);
}
"#;

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
