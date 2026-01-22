//! SPH Bucket Test - Incompressibility Validation
//!
//! Spawns water particles that fall into a bucket.
//! Measures compression ratio to validate IISPH implementation.
//!
//! Success criteria: compression ratio < 1.05 (5% compression)
//!
//! Run with: cargo run --example bucket_test --release

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
const GRID_SIZE: [u32; 3] = [32, 64, 32];  // 64cm x 128cm x 64cm bucket
// GRID MODE: Can handle 30k+ on M1/M2
const MAX_PARTICLES: u32 = 32_000;
const SPAWN_RATE: usize = 64;  // 8x8 grid spawn pattern
const SPAWN_HEIGHT: f32 = 1.0;  // Spawn at top of bucket

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

    // Bucket wireframe
    line_pipeline: wgpu::RenderPipeline,
    bucket_vertices: wgpu::Buffer,
    bucket_vertex_count: u32,

    // Depth buffer
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sph: Option<GpuSph3D>,

    // Pending new particles (cleared after upload)
    pending_positions: Vec<Vec3>,
    pending_velocities: Vec<Vec3>,

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
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            sph: None,
            pending_positions: Vec::with_capacity(SPAWN_RATE),
            pending_velocities: Vec::with_capacity(SPAWN_RATE),
            frame: 0,
            paused: false,
            start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            camera_angle: 0.5,
            camera_pitch: 0.4,
            camera_distance: 2.0,
        }
    }

    fn spawn_particles(&mut self) {
        if self.paused { return; }

        let Some(ref sph) = self.sph else { return };
        let current_count = sph.num_particles() as usize;
        if current_count >= MAX_PARTICLES as usize { return; }

        let to_spawn = SPAWN_RATE.min(MAX_PARTICLES as usize - current_count);

        // Clear pending and add new particles
        self.pending_positions.clear();
        self.pending_velocities.clear();

        // Spawn in a small grid pattern with spacing h*0.5
        let spacing = H * 0.5;  // 0.02m
        let grid_size = 8;  // 8x8 = 64 particles per spawn
        let offset_x = GRID_SIZE[0] as f32 * CELL_SIZE / 2.0 - (grid_size as f32 * spacing / 2.0);  
        let offset_z = GRID_SIZE[2] as f32 * CELL_SIZE / 2.0 - (grid_size as f32 * spacing / 2.0);

        'spawn: for z in 0..grid_size {
            for x in 0..grid_size {
                if self.pending_positions.len() >= to_spawn {
                    break 'spawn;
                }
                self.pending_positions.push(Vec3::new(
                    offset_x + x as f32 * spacing,
                    SPAWN_HEIGHT,
                    offset_z + z as f32 * spacing,
                ));
                self.pending_velocities.push(Vec3::new(0.0, -1.0, 0.0));
            }
        }
    }

    fn update_simulation(&mut self) {
        if self.paused { return; }

        let Some(ref gpu) = self.gpu else { return };
        let Some(ref mut sph) = self.sph else { return };

        // Append only new particles (don't overwrite GPU simulation state)
        if !self.pending_positions.is_empty() {
            sph.append_particles(&gpu.queue, &self.pending_positions, &self.pending_velocities);
        }

            // Run SUB_STEPS sub-steps per frame for stability
        for _ in 0..SUB_STEPS {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SPH Step"),
            });

            // Use Grid-based IISPH (O(n log n))
            sph.step(&mut encoder, &gpu.queue);

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

        // Update uniforms
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let eye = Vec3::new(
            self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
            self.camera_distance * self.camera_pitch.sin() + 0.5,
            self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
        );
        let center = Vec3::new(
            GRID_SIZE[0] as f32 * CELL_SIZE / 2.0,
            GRID_SIZE[1] as f32 * CELL_SIZE / 4.0,
            GRID_SIZE[2] as f32 * CELL_SIZE / 2.0,
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

            // Draw bucket wireframe first
            pass.set_pipeline(&gpu.line_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.bucket_vertices.slice(..));
            pass.draw(0..gpu.bucket_vertex_count, 0..1);

            // Draw particles as points
            pass.set_pipeline(&gpu.particle_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);

            // Use SPH position buffer directly (as instance data)
            pass.set_vertex_buffer(0, sph.positions.slice(..));
            // 4 vertices per particle (billboard quad), num_particles instances
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

            // Print basic FPS info
            println!(
                "FPS: {:.1} | Particles: {} | Time: {:.1}s",
                self.current_fps,
                sph.num_particles(),
                now.duration_since(self.start_time).as_secs_f32()
            );
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

        // Compute compression ratio
        let compression_ratio = metrics.max_density / sph.rest_density();

        println!(
            "Frame {}: {} particles, density_err={:.2}%, rho=[{:.0},{:.0}], p_max={:.1}, compression={:.3}",
            self.frame,
            metrics.particle_count,
            metrics.avg_density_error * 100.0,
            metrics.min_density,
            metrics.max_density,
            metrics.max_pressure,
            compression_ratio
        );
        println!(
            "         Y: min={:.3}, max={:.3}, spread={:.3}, avg={:.3}",
            metrics.min_y,
            metrics.max_y,
            metrics.y_spread,
            metrics.avg_y
        );

        // Warn if compression exceeds threshold
        if compression_ratio > 1.05 {
            eprintln!(
                "WARNING: Compression ratio {:.3} exceeds 5% threshold!",
                compression_ratio
            );
        }

        // Warn if particles collapsed to flat layer (y_spread should grow as bucket fills)
        // With 5000 particles at spacing 0.02m in a bucket, expect y_spread > 0.1m
        let expected_y_spread = (metrics.particle_count as f32 / 100.0) * 0.02; // rough estimate
        if metrics.y_spread < expected_y_spread * 0.5 && metrics.particle_count > 100 {
            eprintln!(
                "WARNING: Y spread {:.3}m is too small - particles may have collapsed! (expected ~{:.3}m)",
                metrics.y_spread,
                expected_y_spread
            );
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("SPH Bucket Test - Incompressibility Validation")
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

            // Create line pipeline for bucket wireframe
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

            // Create bucket wireframe vertices
            let bucket_min = [H, H, H];
            let bucket_max = [
                GRID_SIZE[0] as f32 * CELL_SIZE - H,
                GRID_SIZE[1] as f32 * CELL_SIZE * 0.5,  // Half height for visibility
                GRID_SIZE[2] as f32 * CELL_SIZE - H,
            ];

            // 12 edges of a box = 24 vertices
            let bucket_verts: Vec<[f32; 3]> = vec![
                // Bottom face
                [bucket_min[0], bucket_min[1], bucket_min[2]], [bucket_max[0], bucket_min[1], bucket_min[2]],
                [bucket_max[0], bucket_min[1], bucket_min[2]], [bucket_max[0], bucket_min[1], bucket_max[2]],
                [bucket_max[0], bucket_min[1], bucket_max[2]], [bucket_min[0], bucket_min[1], bucket_max[2]],
                [bucket_min[0], bucket_min[1], bucket_max[2]], [bucket_min[0], bucket_min[1], bucket_min[2]],
                // Top face
                [bucket_min[0], bucket_max[1], bucket_min[2]], [bucket_max[0], bucket_max[1], bucket_min[2]],
                [bucket_max[0], bucket_max[1], bucket_min[2]], [bucket_max[0], bucket_max[1], bucket_max[2]],
                [bucket_max[0], bucket_max[1], bucket_max[2]], [bucket_min[0], bucket_max[1], bucket_max[2]],
                [bucket_min[0], bucket_max[1], bucket_max[2]], [bucket_min[0], bucket_max[1], bucket_min[2]],
                // Vertical edges
                [bucket_min[0], bucket_min[1], bucket_min[2]], [bucket_min[0], bucket_max[1], bucket_min[2]],
                [bucket_max[0], bucket_min[1], bucket_min[2]], [bucket_max[0], bucket_max[1], bucket_min[2]],
                [bucket_max[0], bucket_min[1], bucket_max[2]], [bucket_max[0], bucket_max[1], bucket_max[2]],
                [bucket_min[0], bucket_min[1], bucket_max[2]], [bucket_min[0], bucket_max[1], bucket_max[2]],
            ];

            let bucket_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bucket Vertices"),
                contents: bytemuck::cast_slice(&bucket_verts),
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
                bucket_vertices,
                bucket_vertex_count: bucket_verts.len() as u32,
                depth_texture,
                depth_view,
            }
        });

        // Create SPH simulation
        let mut sph = GpuSph3D::new(&gpu.device, MAX_PARTICLES, H, DT, GRID_SIZE);

        // Calibrate rest density to match actual kernel sum
        let calibrated_density = sph.calibrate_rest_density(&gpu.device, &gpu.queue);
        println!("Calibrated rest_density: {:.0}", calibrated_density);
        sph.set_rest_density(&gpu.queue, calibrated_density);

        // Set sub-step timestep for stability
        let sub_dt = DT / SUB_STEPS as f32;
        sph.set_timestep(&gpu.queue, sub_dt);
        println!("Using {} sub-steps, dt = {:.6}s", SUB_STEPS, sub_dt);

        self.gpu = Some(gpu);
        self.sph = Some(sph);

        println!("SPH Bucket Test initialized (GRID IISPH MODE)");
        println!("Max particles: {} (O(n log n) complexity)", MAX_PARTICLES);
        println!("Press SPACE to pause/resume");
        println!("Target: compression ratio < 1.05");
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
                self.spawn_particles();
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
    let size = 0.01;  // 1cm particles
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
