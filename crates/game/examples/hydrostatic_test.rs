//! Hydrostatic Test - VISUAL IISPH Validation
//!
//! Spawns a static grid of particles that should maintain equilibrium.
//! VISUAL with rendering so you can SEE if particles collapse.
//!
//! Pass criteria:
//! - Particles stay in a grid (don't collapse to a pancake)
//! - Density deviation < 5% from rest density
//!
//! Run with: cargo run --example hydrostatic_test --release

use bytemuck::{Pod, Zeroable};
use game::gpu::sph_3d::GpuSph3D;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Simulation parameters
const CELL_SIZE: f32 = 0.02;  // 2cm cells
const H: f32 = 0.04;          // Kernel radius = 2x cell size
const DT: f32 = 1.0 / 120.0;  // 120 Hz physics (frame time)
const SUB_STEPS: u32 = 10;    // Sub-steps per frame for stability
const GRID_SIZE: [u32; 3] = [32, 64, 32];  // Must be big enough for SPH allocation
const STREAM_PARTICLES: usize = 48;        // Extra stream to feed the bucket
const STREAM_LENGTH: f32 = 0.3;
const STREAM_HORIZONTAL_SPREAD: f32 = 0.02;
const STREAM_VERTICAL_GAP: f32 = 0.01;

struct SimpleRng {
    state: u32,
}

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let frac = (self.next_u32() as f32) / (u32::MAX as f32);
        lo + (hi - lo) * frac
    }
}

// HARDCODED bucket bounds - MUST MATCH shader sph_bruteforce.wgsl
const BUCKET_MIN: [f32; 3] = [0.1, 0.04, 0.1];
const BUCKET_MAX: [f32; 3] = [0.3, 1.0, 0.3];

// Test parameters - fewer particles to fit in tight bucket
const PARTICLE_GRID: usize = 6;  // 6x6x6 = 216 particles

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    // Simple particle rendering
    particle_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // Box wireframe
    line_pipeline: wgpu::RenderPipeline,
    box_vertices: wgpu::Buffer,
    box_vertex_count: u32,

    // Depth buffer
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sph: Option<GpuSph3D>,

    // Metrics
    frame: u32,
    paused: bool,
    start_time: Instant,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,

    // Camera
    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,

    // Initial grid bounds for comparison
    initial_y_min: f32,
    initial_y_max: f32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            sph: None,
            frame: 0,
            paused: false,
            start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            camera_angle: 0.5,
            camera_pitch: 0.4,
            camera_distance: 1.5,
            initial_y_min: 0.0,
            initial_y_max: 0.0,
        }
    }

    fn update_simulation(&mut self) {
        if self.paused { return; }

        let Some(ref gpu) = self.gpu else { return };
        let Some(ref mut sph) = self.sph else { return };

        // Run SUB_STEPS sub-steps per frame for stability
        for _ in 0..SUB_STEPS {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SPH Step"),
            });
            sph.step_bruteforce(&mut encoder);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    fn render(&mut self) {
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref sph) = self.sph else { return };

        // Get surface texture
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        // Update uniforms - camera looks at bucket center
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let center = Vec3::new(
            (BUCKET_MIN[0] + BUCKET_MAX[0]) / 2.0,
            (BUCKET_MIN[1] + BUCKET_MAX[1]) / 4.0,
            (BUCKET_MIN[2] + BUCKET_MAX[2]) / 2.0,
        );
        let eye = Vec3::new(
            self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
            self.camera_distance * self.camera_pitch.sin() + 0.3,
            self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
        );
        let view_matrix = Mat4::look_at_rh(eye + center, center, Vec3::Y);
        let proj_matrix = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.01, 100.0);

        let uniforms = Uniforms {
            view_proj: (proj_matrix * view_matrix).to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Render
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
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

            // Draw box wireframe first
            pass.set_pipeline(&gpu.line_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.box_vertices.slice(..));
            pass.draw(0..gpu.box_vertex_count, 0..1);

            // Draw particles as points
            pass.set_pipeline(&gpu.particle_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, sph.positions.slice(..));
            pass.draw(0..4, 0..sph.num_particles());
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Update FPS
        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.last_fps_time = now;
        }
    }

    /// Log diagnostic metrics from the simulation (blocking GPU readback)
    fn log_metrics(&self) {
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref sph) = self.sph else { return };

        if sph.num_particles() == 0 {
            return;
        }

        let metrics = sph.compute_metrics(&gpu.device, &gpu.queue);
        let time = self.start_time.elapsed().as_secs_f32();

        // Compute compression ratio
        let compression_ratio = metrics.max_density / sph.rest_density();

        // Y collapse detection
        let y_collapse_ratio = metrics.y_spread / (self.initial_y_max - self.initial_y_min);

        println!(
            "Frame {:4} ({:.1}s): density_err={:.2}%, rho=[{:.0},{:.0}], compression={:.3}",
            self.frame,
            time,
            metrics.avg_density_error * 100.0,
            metrics.min_density,
            metrics.max_density,
            compression_ratio
        );
        println!(
            "              Y: min={:.3}, max={:.3}, spread={:.3} (was {:.3}, ratio={:.2})",
            metrics.min_y,
            metrics.max_y,
            metrics.y_spread,
            self.initial_y_max - self.initial_y_min,
            y_collapse_ratio
        );

        // Warn if compression exceeds threshold
        if compression_ratio > 1.05 {
            eprintln!("  WARNING: Compression {:.1}% exceeds 5% threshold!", (compression_ratio - 1.0) * 100.0);
        }

        // Warn if Y collapsed
        if y_collapse_ratio < 0.5 {
            eprintln!("  WARNING: Y spread collapsed to {:.0}% of initial!", y_collapse_ratio * 100.0);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Hydrostatic Test - VISUAL SPH Validation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());

        // Initialize GPU
        let gpu = pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_storage_buffers_per_shader_stage: 12,
                            ..wgpu::Limits::default()
                        },
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .unwrap();

            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface.get_capabilities(&adapter).formats[0],
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            // Create depth texture
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
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&Default::default());

            // Create uniform buffer
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Uniform Layout"),
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

            let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Uniform Bind Group"),
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            // Create particle render pipeline
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(PARTICLE_SHADER)),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

            let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Particle Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 16,  // vec4<f32>
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

            // Create line pipeline for box wireframe
            let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Line Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(LINE_SHADER)),
            });

            let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Line Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &line_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,  // vec3<f32>
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &line_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

            // Create box wireframe vertices - HARDCODED bucket bounds
            let box_min = BUCKET_MIN;
            let box_max = BUCKET_MAX;

            // 12 edges of a box = 24 vertices
            let box_verts: Vec<[f32; 3]> = vec![
                // Bottom face
                [box_min[0], box_min[1], box_min[2]], [box_max[0], box_min[1], box_min[2]],
                [box_max[0], box_min[1], box_min[2]], [box_max[0], box_min[1], box_max[2]],
                [box_max[0], box_min[1], box_max[2]], [box_min[0], box_min[1], box_max[2]],
                [box_min[0], box_min[1], box_max[2]], [box_min[0], box_min[1], box_min[2]],
                // Top face
                [box_min[0], box_max[1], box_min[2]], [box_max[0], box_max[1], box_min[2]],
                [box_max[0], box_max[1], box_min[2]], [box_max[0], box_max[1], box_max[2]],
                [box_max[0], box_max[1], box_max[2]], [box_min[0], box_max[1], box_max[2]],
                [box_min[0], box_max[1], box_max[2]], [box_min[0], box_max[1], box_min[2]],
                // Vertical edges
                [box_min[0], box_min[1], box_min[2]], [box_min[0], box_max[1], box_min[2]],
                [box_max[0], box_min[1], box_min[2]], [box_max[0], box_max[1], box_min[2]],
                [box_max[0], box_min[1], box_max[2]], [box_max[0], box_max[1], box_max[2]],
                [box_min[0], box_min[1], box_max[2]], [box_min[0], box_max[1], box_max[2]],
            ];

            let box_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Box Vertices"),
                contents: bytemuck::cast_slice(&box_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });

            GpuState {
                device,
                queue,
                surface,
                config,
                particle_pipeline,
                uniform_buffer,
                uniform_bind_group,
                line_pipeline,
                box_vertices,
                box_vertex_count: box_verts.len() as u32,
                depth_view,
            }
        });

        // Create SPH simulation
        let max_particles = (PARTICLE_GRID * PARTICLE_GRID * PARTICLE_GRID + STREAM_PARTICLES) as u32;
        let mut sph = GpuSph3D::new(&gpu.device, max_particles, H, DT, GRID_SIZE);

        // HARDCODE rest_density - calibration is broken
        // Poly6 kernel sum at spacing h*0.5 with h=0.04 gives ~126222
        let hardcoded_density = 126222.0;
        println!("Using HARDCODED rest_density: {:.0}", hardcoded_density);
        sph.set_rest_density(&gpu.queue, hardcoded_density);

        // Set sub-step timestep for stability
        let sub_dt = DT / SUB_STEPS as f32;
        sph.set_timestep(&gpu.queue, sub_dt);

        // Spawn particles in a static grid INSIDE the hardcoded bucket
        let spacing = H * 0.5;  // 0.02m spacing
        let bucket_center_x = (BUCKET_MIN[0] + BUCKET_MAX[0]) / 2.0;
        let bucket_center_z = (BUCKET_MIN[2] + BUCKET_MAX[2]) / 2.0;
        let start_x = bucket_center_x - (PARTICLE_GRID as f32 * spacing) / 2.0;
        let start_y = BUCKET_MIN[1] + 0.01;  // Just above floor
        let start_z = bucket_center_z - (PARTICLE_GRID as f32 * spacing) / 2.0;

        let mut positions = Vec::with_capacity(max_particles as usize);
        let mut velocities = Vec::with_capacity(max_particles as usize);

        for z in 0..PARTICLE_GRID {
            for y in 0..PARTICLE_GRID {
                for x in 0..PARTICLE_GRID {
                    positions.push(Vec3::new(
                        start_x + x as f32 * spacing,
                        start_y + y as f32 * spacing,
                        start_z + z as f32 * spacing,
                    ));
                    velocities.push(Vec3::ZERO);  // Start at rest
                }
            }
        }

        let stream_center_x = (BUCKET_MIN[0] + BUCKET_MAX[0]) / 2.0;
        let stream_center_z = (BUCKET_MIN[2] + BUCKET_MAX[2]) / 2.0;

        let mut rng = SimpleRng::new(0xC0FFEE);
        for i in 0..STREAM_PARTICLES {
            let t = i as f32 / (STREAM_PARTICLES as f32 - 1.0);
            let base_x = stream_center_x - (STREAM_LENGTH * 0.5) + t * STREAM_LENGTH;
            let offset_x = base_x + rng.range(-STREAM_HORIZONTAL_SPREAD, STREAM_HORIZONTAL_SPREAD);
            let offset_z = stream_center_z + rng.range(-STREAM_HORIZONTAL_SPREAD, STREAM_HORIZONTAL_SPREAD);
            let offset_y = BUCKET_MAX[1] + 0.02 + i as f32 * STREAM_VERTICAL_GAP;

            positions.push(Vec3::new(offset_x, offset_y, offset_z));
            velocities.push(Vec3::new(rng.range(-0.01, 0.01), -0.35, rng.range(-0.01, 0.01)));
        }

        // Record initial Y bounds
        self.initial_y_min = start_y;
        self.initial_y_max = start_y + (PARTICLE_GRID - 1) as f32 * spacing;

        println!();
        println!("=== HYDROSTATIC TEST - VISUAL ===");
        println!("Spawned {}x{}x{} = {} particles", PARTICLE_GRID, PARTICLE_GRID, PARTICLE_GRID, positions.len());
        println!("Initial Y range: [{:.3}, {:.3}] = {:.3}m spread",
                 self.initial_y_min, self.initial_y_max, self.initial_y_max - self.initial_y_min);
        println!("Using {} sub-steps, dt = {:.6}s", SUB_STEPS, sub_dt);
        println!("Press SPACE to pause/resume");
        println!();

        // Upload particles
        sph.upload_particles(&gpu.queue, &positions, &velocities);

        self.gpu = Some(gpu);
        self.sph = Some(sph);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        match key {
                            KeyCode::Space => {
                                self.paused = !self.paused;
                                println!("Simulation {}", if self.paused { "PAUSED" } else { "RUNNING" });
                            }
                            KeyCode::Escape => {
                                event_loop.exit();
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_simulation();
                self.render();
                self.frame += 1;

                // Log metrics every 60 frames
                if self.frame % 60 == 0 {
                    self.log_metrics();
                }

                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

const LINE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4(0.8, 0.8, 0.8, 1.0);  // Gray wireframe
}
"#;

const PARTICLE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) instance_pos: vec4<f32>,
) -> VertexOutput {
    // Billboard quad
    let size = 0.008;  // 8mm particles (smaller for dense grid)
    let offsets = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0),
    );

    let offset = offsets[vertex_index] * size;

    // Billboard facing camera
    let view_dir = normalize(uniforms.camera_pos - instance_pos.xyz);
    let right = normalize(cross(vec3(0.0, 1.0, 0.0), view_dir));
    let up = cross(view_dir, right);

    let world_pos = instance_pos.xyz + right * offset.x + up * offset.y;

    var out: VertexOutput;
    out.position = uniforms.view_proj * vec4(world_pos, 1.0);
    out.uv = offsets[vertex_index] * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circular particle with soft edge
    let dist = length(in.uv - vec2(0.5));
    let alpha = 1.0 - smoothstep(0.3, 0.5, dist);

    if (alpha < 0.01) {
        discard;
    }

    // Water blue color
    return vec4(0.2, 0.5, 0.9, alpha * 0.8);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
