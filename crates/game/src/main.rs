//! Filling Box Test with Visual Rendering
//!
//! Closed box with solid walls. Particles spawn continuously from a "faucet" at the top.
//! Water level should rise proportionally to particles added (volume conservation test).

mod gpu;

use crate::gpu::GpuFlip2D;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const GRID_WIDTH: u32 = 64;
const GRID_HEIGHT: u32 = 64;
const CELL_SIZE: f32 = 0.05;
const MAX_PARTICLES: usize = 50_000;

const DT: f32 = 1.0 / 60.0;
const PRESSURE_ITERS: u32 = 500;
const GRAVITY: f32 = -9.8;

// Faucet settings: spawn particles each frame
const SPAWN_PER_FRAME: usize = 20;  // 20 particles per frame = 1200/sec

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 800;

struct App {
    window: Option<Arc<Window>>,
    state: Option<RenderState>,
}

struct RenderState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuFlip2D,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    frame: u32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
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
                        .with_title("FLIP Box Test")
                        .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
                )
                .unwrap(),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        println!("GPU: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .unwrap();

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create simulation
        let sim = GpuFlip2D::new(&device, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, MAX_PARTICLES);

        // Start with no particles - will fill from faucet
        println!("Starting filling box test - particles will spawn from top");
        println!("Domain: {:.2}m x {:.2}m", GRID_WIDTH as f32 * CELL_SIZE, GRID_HEIGHT as f32 * CELL_SIZE);

        // Create render pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (MAX_PARTICLES * 8) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.window = Some(window);
        self.state = Some(RenderState {
            device,
            queue,
            surface,
            surface_config,
            sim,
            render_pipeline,
            vertex_buffer,
            frame: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    // Spawn particles from faucet at top-center
                    let domain_size = GRID_WIDTH as f32 * CELL_SIZE;
                    let faucet_x = domain_size * 0.5;  // Center
                    let faucet_y = domain_size * 0.85;  // Near top
                    let spacing = CELL_SIZE * 0.45;

                    let mut spawn_pos = Vec::new();
                    let mut spawn_vel = Vec::new();
                    for i in 0..SPAWN_PER_FRAME {
                        // Spread particles horizontally within a small region
                        let offset = (i as f32 - SPAWN_PER_FRAME as f32 / 2.0) * spacing * 0.5;
                        spawn_pos.push([faucet_x + offset, faucet_y]);
                        spawn_vel.push([0.0, -2.0]);  // Falling downward
                    }
                    state.sim.spawn_particles(&state.queue, &spawn_pos, &spawn_vel);

                    // Step simulation
                    state.sim.step(&state.device, &state.queue, DT, GRAVITY, PRESSURE_ITERS);
                    state.device.poll(wgpu::Maintain::Wait);
                    state.frame += 1;

                    // Read positions and upload to vertex buffer
                    let positions = state.sim.read_positions(&state.device, &state.queue);
                    let particle_count = state.sim.particle_count();

                    // Print diagnostics every second
                    if state.frame % 60 == 0 && !positions.is_empty() {
                        let min_y = positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
                        let max_y = positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);

                        // Expected water level based on particle count
                        // Each particle represents about (cell_size/2)^2 of fluid area
                        // Volume = n * (dx/2)^2, Level = Volume / box_width
                        let particle_area = (CELL_SIZE / 2.0) * (CELL_SIZE / 2.0);
                        let expected_level = (particle_count as f32 * particle_area) / domain_size;

                        // Post-correction residual
                        let residual = state.sim.compute_residual(&state.device, &state.queue);

                        println!(
                            "t={:.1}s | n={} | y: {:.2}..{:.2} | surface={:.2}m expected={:.2}m | vel_res={:.3}",
                            state.frame as f32 / 60.0, particle_count, min_y, max_y, max_y, expected_level, residual
                        );
                    }

                    // Transform positions to NDC
                    let domain_size = GRID_WIDTH as f32 * CELL_SIZE;
                    let ndc_positions: Vec<[f32; 2]> = positions
                        .iter()
                        .map(|p| {
                            [
                                (p[0] / domain_size) * 2.0 - 1.0,
                                (p[1] / domain_size) * 2.0 - 1.0,
                            ]
                        })
                        .collect();

                    state.queue.write_buffer(
                        &state.vertex_buffer,
                        0,
                        bytemuck::cast_slice(&ndc_positions),
                    );

                    // Render
                    let frame = state.surface.get_current_texture().unwrap();
                    let view = frame.texture.create_view(&Default::default());

                    let mut encoder = state.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Render Pass"),
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
                            depth_stencil_attachment: None,
                            ..Default::default()
                        });

                        pass.set_pipeline(&state.render_pipeline);
                        pass.set_vertex_buffer(0, state.vertex_buffer.slice(..));
                        pass.draw(0..positions.len() as u32, 0..1);
                    }

                    state.queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

const RENDER_SHADER: &str = r#"
@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.6, 1.0, 1.0);
}
"#;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
