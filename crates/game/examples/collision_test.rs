//! Collision Test - JUST boundary collision, NO pressure
//!
//! Drops particles with gravity only. Tests if walls/floor work.
//! If particles fall through floor or escape walls, collision is broken.
//!
//! Run with: cargo run --example collision_test --release

use bytemuck::{Pod, Zeroable};
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

// Simple params - just gravity, no SPH
const CELL_SIZE: f32 = 0.02;
const DT: f32 = 1.0 / 120.0;
const GRAVITY: f32 = -9.81;

// Tight bucket
const BUCKET_MIN: [f32; 3] = [0.1, 0.04, 0.1];
const BUCKET_MAX: [f32; 3] = [0.3, 1.0, 0.3];  // 20cm x 96cm x 20cm

const MAX_PARTICLES: usize = 500;
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
    particle_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    box_vertices: wgpu::Buffer,
    box_vertex_count: u32,
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,

    // CPU-side particles (no GPU SPH)
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    position_buffer: Option<wgpu::Buffer>,

    frame: u32,
    paused: bool,
    start_time: Instant,

    camera_angle: f32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            positions: Vec::new(),
            velocities: Vec::new(),
            position_buffer: None,
            frame: 0,
            paused: false,
            start_time: Instant::now(),
            camera_angle: 0.5,
        }
    }

    fn spawn_particles(&mut self) {
        let center_x = (BUCKET_MIN[0] + BUCKET_MAX[0]) / 2.0;
        let center_z = (BUCKET_MIN[2] + BUCKET_MAX[2]) / 2.0;

        // Spawn particles in a circle pattern with outward velocities
        // This tests ALL wall directions
        let num_angles = 16;
        let speeds = [1.0, 2.0, 3.0];  // Different speeds
        let heights = [0.2, 0.4, 0.6, 0.8];  // Different heights

        for &speed in &speeds {
            for &height in &heights {
                for i in 0..num_angles {
                    let angle = (i as f32 / num_angles as f32) * std::f32::consts::PI * 2.0;
                    let vx = angle.cos() * speed;
                    let vz = angle.sin() * speed;

                    self.positions.push(Vec3::new(center_x, height, center_z));
                    self.velocities.push(Vec3::new(vx, 0.0, vz));
                }
            }
        }

        // Also add some falling straight down
        for i in 0..10 {
            let angle = (i as f32 / 10.0) * std::f32::consts::PI * 2.0;
            let r = 0.05;
            self.positions.push(Vec3::new(
                center_x + angle.cos() * r,
                0.9,
                center_z + angle.sin() * r,
            ));
            self.velocities.push(Vec3::new(0.0, -1.0, 0.0));  // Falling
        }

        println!("Spawned {} particles", self.positions.len());
        println!("  {} angles x {} speeds x {} heights = {} radial",
                 num_angles, speeds.len(), heights.len(), num_angles * speeds.len() * heights.len());
        println!("  + 10 falling straight down");
    }

    fn update_physics(&mut self) {
        if self.paused { return; }

        for i in 0..self.positions.len() {
            // Apply gravity
            self.velocities[i].y += GRAVITY * DT;

            // Update position
            self.positions[i] += self.velocities[i] * DT;

            // Box collision - simple but explicit
            let pos = &mut self.positions[i];
            let vel = &mut self.velocities[i];

            // Floor
            if pos.y < BUCKET_MIN[1] {
                pos.y = BUCKET_MIN[1];
                vel.y = 0.0;
                vel.x *= 0.9;  // Friction
                vel.z *= 0.9;
            }

            // Ceiling (open, but clamp for safety)
            if pos.y > BUCKET_MAX[1] {
                pos.y = BUCKET_MAX[1];
                vel.y = 0.0;
            }

            // X walls
            if pos.x < BUCKET_MIN[0] {
                pos.x = BUCKET_MIN[0];
                vel.x = 0.0;
            }
            if pos.x > BUCKET_MAX[0] {
                pos.x = BUCKET_MAX[0];
                vel.x = 0.0;
            }

            // Z walls
            if pos.z < BUCKET_MIN[2] {
                pos.z = BUCKET_MIN[2];
                vel.z = 0.0;
            }
            if pos.z > BUCKET_MAX[2] {
                pos.z = BUCKET_MAX[2];
                vel.z = 0.0;
            }
        }
    }

    fn upload_positions(&mut self) {
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref buffer) = self.position_buffer else { return };

        // Convert to vec4 for GPU
        let data: Vec<[f32; 4]> = self.positions.iter()
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();

        gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&data));
    }

    fn log_diagnostics(&self) {
        if self.positions.is_empty() { return; }

        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut sum_y = 0.0;
        let mut outside_count = 0;

        for pos in &self.positions {
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
            sum_y += pos.y;

            // Check if outside bucket
            if pos.x < BUCKET_MIN[0] || pos.x > BUCKET_MAX[0] ||
               pos.y < BUCKET_MIN[1] || pos.y > BUCKET_MAX[1] ||
               pos.z < BUCKET_MIN[2] || pos.z > BUCKET_MAX[2] {
                outside_count += 1;
            }
        }

        let avg_y = sum_y / self.positions.len() as f32;
        let time = self.start_time.elapsed().as_secs_f32();

        println!(
            "Frame {:4} ({:.1}s): Y=[{:.3}, {:.3}], avg={:.3}, outside={}",
            self.frame, time, min_y, max_y, avg_y, outside_count
        );

        if outside_count > 0 {
            eprintln!("  ERROR: {} particles escaped the bucket!", outside_count);
        }

        // Check if settled on floor
        if max_y - min_y < 0.01 && min_y < BUCKET_MIN[1] + 0.02 {
            println!("  GOOD: Particles settled on floor at y={:.3}", min_y);
        }
    }

    fn render(&mut self) {
        let Some(ref gpu) = self.gpu else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        // Camera
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let center = Vec3::new(
            (BUCKET_MIN[0] + BUCKET_MAX[0]) / 2.0,
            (BUCKET_MIN[1] + BUCKET_MAX[1]) / 4.0,
            (BUCKET_MIN[2] + BUCKET_MAX[2]) / 2.0,
        );
        let dist = 0.8;
        let eye = Vec3::new(
            dist * self.camera_angle.cos(),
            dist * 0.5 + 0.2,
            dist * self.camera_angle.sin(),
        );
        let view_matrix = Mat4::look_at_rh(eye + center, center, Vec3::Y);
        let proj_matrix = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.01, 100.0);

        let uniforms = Uniforms {
            view_proj: (proj_matrix * view_matrix).to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = gpu.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }),
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

            // Draw bucket wireframe
            pass.set_pipeline(&gpu.line_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.box_vertices.slice(..));
            pass.draw(0..gpu.box_vertex_count, 0..1);

            // Draw particles
            if let Some(ref buffer) = self.position_buffer {
                pass.set_pipeline(&gpu.particle_pipeline);
                pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..4, 0..self.positions.len() as u32);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Rotate camera slowly
        self.camera_angle += 0.005;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Collision Test - NO PRESSURE, just gravity + walls")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(async {
            let instance = wgpu::Instance::new(Default::default());
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }).await.unwrap();

            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None).await.unwrap();

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

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth"),
                size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&Default::default());

            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(PARTICLE_SHADER)),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
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
                        array_stride: 16,
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
                        array_stride: 12,
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

            // Bucket wireframe
            let box_verts: Vec<[f32; 3]> = vec![
                // Bottom
                BUCKET_MIN, [BUCKET_MAX[0], BUCKET_MIN[1], BUCKET_MIN[2]],
                [BUCKET_MAX[0], BUCKET_MIN[1], BUCKET_MIN[2]], [BUCKET_MAX[0], BUCKET_MIN[1], BUCKET_MAX[2]],
                [BUCKET_MAX[0], BUCKET_MIN[1], BUCKET_MAX[2]], [BUCKET_MIN[0], BUCKET_MIN[1], BUCKET_MAX[2]],
                [BUCKET_MIN[0], BUCKET_MIN[1], BUCKET_MAX[2]], BUCKET_MIN,
                // Top
                [BUCKET_MIN[0], BUCKET_MAX[1], BUCKET_MIN[2]], [BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MIN[2]],
                [BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MIN[2]], [BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MAX[2]],
                [BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MAX[2]], [BUCKET_MIN[0], BUCKET_MAX[1], BUCKET_MAX[2]],
                [BUCKET_MIN[0], BUCKET_MAX[1], BUCKET_MAX[2]], [BUCKET_MIN[0], BUCKET_MAX[1], BUCKET_MIN[2]],
                // Verticals
                BUCKET_MIN, [BUCKET_MIN[0], BUCKET_MAX[1], BUCKET_MIN[2]],
                [BUCKET_MAX[0], BUCKET_MIN[1], BUCKET_MIN[2]], [BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MIN[2]],
                [BUCKET_MAX[0], BUCKET_MIN[1], BUCKET_MAX[2]], [BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MAX[2]],
                [BUCKET_MIN[0], BUCKET_MIN[1], BUCKET_MAX[2]], [BUCKET_MIN[0], BUCKET_MAX[1], BUCKET_MAX[2]],
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
                line_pipeline,
                uniform_buffer,
                uniform_bind_group,
                box_vertices,
                box_vertex_count: box_verts.len() as u32,
                depth_view,
            }
        });

        // Create position buffer
        let position_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions"),
            size: (MAX_PARTICLES * 16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.gpu = Some(gpu);
        self.position_buffer = Some(position_buffer);

        // Spawn initial particles
        self.spawn_particles();

        println!();
        println!("=== COLLISION TEST ===");
        println!("Testing boundary collision ONLY - no SPH pressure");
        println!("Bucket: [{:.2},{:.2},{:.2}] to [{:.2},{:.2},{:.2}]",
                 BUCKET_MIN[0], BUCKET_MIN[1], BUCKET_MIN[2],
                 BUCKET_MAX[0], BUCKET_MAX[1], BUCKET_MAX[2]);
        println!("Particles should fall and settle on floor at y={:.3}", BUCKET_MIN[1]);
        println!("Press SPACE to pause, R to reset");
        println!();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        match key {
                            KeyCode::Space => {
                                self.paused = !self.paused;
                                println!("{}", if self.paused { "PAUSED" } else { "RUNNING" });
                            }
                            KeyCode::KeyR => {
                                self.positions.clear();
                                self.velocities.clear();
                                self.spawn_particles();
                                println!("RESET");
                            }
                            KeyCode::Escape => event_loop.exit(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_physics();
                self.upload_positions();
                self.render();
                self.frame += 1;

                if self.frame % 60 == 0 {
                    self.log_diagnostics();
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
    return vec4(0.8, 0.8, 0.8, 1.0);
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
    let size = 0.015;
    let offsets = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0),
        vec2(-1.0, 1.0), vec2(1.0, 1.0),
    );
    let offset = offsets[vertex_index] * size;

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
    let dist = length(in.uv - vec2(0.5));
    let alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    if (alpha < 0.01) { discard; }
    return vec4(0.2, 0.6, 1.0, alpha);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
