//! Washplant Editor (Clean Slate)
//!
//! Minimal rendering scaffold for future washplant editor work.
//!
//! Run with:
//!   cargo run --example washplant_editor -- <optional-scenario.json>
//!
//! Controls:
//!   Mouse drag = rotate camera
//!   Scroll     = zoom

use bytemuck::{Pod, Zeroable};
use game::editor::ScenarioConfig;
use game::example_utils::{create_depth_view, Camera, Pos3Color4Vertex, WgpuContext, BASIC_SHADER};
use glam::Vec3;
use std::path::PathBuf;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const GRID_SIZE_METERS: f32 = 1.25;
const BOX_SIZE_METERS: f32 = 1.0;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let scenario_path = std::env::args().nth(1).map(PathBuf::from);
    let mut app = App::new(scenario_path);
    event_loop.run_app(&mut app).unwrap();
}

struct ScenarioState {
    path: Option<PathBuf>,
    scenario: Option<ScenarioConfig>,
}

impl ScenarioState {
    fn new(path: Option<PathBuf>) -> Self {
        let scenario = path
            .as_ref()
            .and_then(|path| ScenarioConfig::load_json(path).ok());

        match (&path, &scenario) {
            (Some(path), Some(config)) => {
                let label = config
                    .name
                    .clone()
                    .or_else(|| path.file_stem().map(|s| s.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "Unnamed".to_string());
                println!("Loaded scenario: {label}");
            }
            (Some(path), None) => {
                eprintln!("Scenario load placeholder: unable to load {path:?}");
            }
            (None, _) => {
                println!("No scenario specified; using default placeholder scene");
            }
        }

        Self { path, scenario }
    }

    fn title_suffix(&self) -> Option<String> {
        let config_name = self.scenario.as_ref().and_then(|config| config.name.clone());
        let path_name = self
            .path
            .as_ref()
            .and_then(|path| path.file_stem().map(|s| s.to_string_lossy().to_string()));
        config_name.or(path_name)
    }
}

struct App {
    window: Option<Arc<Window>>,
    ctx: Option<WgpuContext>,
    depth_view: Option<wgpu::TextureView>,

    line_pipeline: Option<wgpu::RenderPipeline>,
    mesh_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    uniform_bind_group: Option<wgpu::BindGroup>,

    grid_buffer: Option<wgpu::Buffer>,
    grid_vertex_count: u32,
    box_buffer: Option<wgpu::Buffer>,
    box_vertex_count: u32,

    camera: Camera,
    dragging: bool,
    last_mouse: (f32, f32),

    scenario: ScenarioState,
}

impl App {
    fn new(scenario_path: Option<PathBuf>) -> Self {
        let scenario = ScenarioState::new(scenario_path);
        let grid_center = Vec3::splat(GRID_SIZE_METERS * 0.5);

        Self {
            window: None,
            ctx: None,
            depth_view: None,
            line_pipeline: None,
            mesh_pipeline: None,
            uniform_buffer: None,
            uniform_bind_group: None,
            grid_buffer: None,
            grid_vertex_count: 0,
            box_buffer: None,
            box_vertex_count: 0,
            camera: Camera::new(1.1, 0.45, 2.1, grid_center),
            dragging: false,
            last_mouse: (0.0, 0.0),
            scenario,
        }
    }

    fn render(&mut self) {
        let Some(window) = &self.window else { return; };
        let Some(line_pipeline) = &self.line_pipeline else { return; };
        let Some(mesh_pipeline) = &self.mesh_pipeline else { return; };
        let Some(uniform_buffer) = &self.uniform_buffer else { return; };
        let Some(uniform_bind_group) = &self.uniform_bind_group else { return; };
        let Some(grid_buffer) = &self.grid_buffer else { return; };
        let Some(box_buffer) = &self.box_buffer else { return; };
        let Some(depth_view) = &self.depth_view else { return; };
        if self.ctx.is_none() {
            return;
        }

        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let aspect = size.width as f32 / size.height as f32;
        let view_proj = self.camera.view_proj_matrix(aspect);
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position().to_array(),
            _pad: 0.0,
        };
        if let Some(ctx) = &self.ctx {
            ctx.queue.write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        let output = {
            let ctx = self.ctx.as_ref().unwrap();
            ctx.surface.get_current_texture()
        };

        let output = match output {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                if let Some(ctx) = &mut self.ctx {
                    let size = window.inner_size();
                    ctx.resize(size.width, size.height);
                    self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));
                }
                return;
            }
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Timeout) => return,
            Err(err) => {
                eprintln!("Surface error: {err:?}");
                return;
            }
        };

        let Some(ctx) = &self.ctx else { return; };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.04,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, uniform_bind_group, &[]);

            render_pass.set_pipeline(line_pipeline);
            render_pass.set_vertex_buffer(0, grid_buffer.slice(..));
            render_pass.draw(0..self.grid_vertex_count, 0..1);

            render_pass.set_pipeline(mesh_pipeline);
            render_pass.set_vertex_buffer(0, box_buffer.slice(..));
            render_pass.draw(0..self.box_vertex_count, 0..1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = self
            .scenario
            .title_suffix()
            .map(|suffix| format!("Washplant Editor - {suffix}"))
            .unwrap_or_else(|| "Washplant Editor".to_string());

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title(title)
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
                )
                .unwrap(),
        );
        self.window = Some(window.clone());

        let ctx = pollster::block_on(WgpuContext::init(window));
        let depth_view = create_depth_view(&ctx.device, &ctx.config);

        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
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

        let uniform_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Basic Shader"),
            source: wgpu::ShaderSource::Wgsl(BASIC_SHADER.into()),
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mesh_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Pos3Color4Vertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let line_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Pos3Color4Vertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let grid_vertices = grid_boundary_lines(GRID_SIZE_METERS, [0.2, 0.6, 0.9, 1.0]);
        let grid_vertex_count = grid_vertices.len() as u32;
        let grid_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&grid_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let box_center = Vec3::splat(GRID_SIZE_METERS * 0.5);
        let box_vertices = cube_mesh(box_center, BOX_SIZE_METERS, [0.85, 0.55, 0.2, 1.0]);
        let box_vertex_count = box_vertices.len() as u32;
        let box_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Box Vertex Buffer"),
            contents: bytemuck::cast_slice(&box_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.ctx = Some(ctx);
        self.depth_view = Some(depth_view);
        self.uniform_buffer = Some(uniform_buffer);
        self.uniform_bind_group = Some(uniform_bind_group);
        self.line_pipeline = Some(line_pipeline);
        self.mesh_pipeline = Some(mesh_pipeline);
        self.grid_buffer = Some(grid_buffer);
        self.grid_vertex_count = grid_vertex_count;
        self.box_buffer = Some(box_buffer);
        self.box_vertex_count = box_vertex_count;

        println!("Washplant editor scaffold ready.");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.resize(size.width, size.height);
                    self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(window) = &self.window {
                    let size = window.inner_size();
                    if let Some(ctx) = &mut self.ctx {
                        ctx.resize(size.width, size.height);
                        self.depth_view = Some(create_depth_view(&ctx.device, &ctx.config));
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if self.dragging {
                    let dx = x - self.last_mouse.0;
                    let dy = y - self.last_mouse.1;
                    self.camera.handle_mouse_move(dx, dy);
                }
                self.last_mouse = (x, y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.handle_zoom(scroll);
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn grid_boundary_lines(size: f32, color: [f32; 4]) -> Vec<Pos3Color4Vertex> {
    let min = Vec3::ZERO;
    let max = Vec3::splat(size);

    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];

    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    let mut lines = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        lines.push(Pos3Color4Vertex {
            position: corners[a].to_array(),
            color,
        });
        lines.push(Pos3Color4Vertex {
            position: corners[b].to_array(),
            color,
        });
    }

    lines
}

fn cube_mesh(center: Vec3, size: f32, color: [f32; 4]) -> Vec<Pos3Color4Vertex> {
    let half = size * 0.5;
    let min = center - Vec3::splat(half);
    let max = center + Vec3::splat(half);

    let p000 = Vec3::new(min.x, min.y, min.z);
    let p100 = Vec3::new(max.x, min.y, min.z);
    let p110 = Vec3::new(max.x, max.y, min.z);
    let p010 = Vec3::new(min.x, max.y, min.z);
    let p001 = Vec3::new(min.x, min.y, max.z);
    let p101 = Vec3::new(max.x, min.y, max.z);
    let p111 = Vec3::new(max.x, max.y, max.z);
    let p011 = Vec3::new(min.x, max.y, max.z);

    let mut verts = Vec::with_capacity(36);

    push_quad(&mut verts, p000, p100, p110, p010, color); // -Z
    push_quad(&mut verts, p101, p001, p011, p111, color); // +Z
    push_quad(&mut verts, p001, p000, p010, p011, color); // -X
    push_quad(&mut verts, p100, p101, p111, p110, color); // +X
    push_quad(&mut verts, p010, p110, p111, p011, color); // +Y
    push_quad(&mut verts, p001, p101, p100, p000, color); // -Y

    verts
}

fn push_quad(
    verts: &mut Vec<Pos3Color4Vertex>,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    color: [f32; 4],
) {
    push_tri(verts, a, b, c, color);
    push_tri(verts, a, c, d, color);
}

fn push_tri(verts: &mut Vec<Pos3Color4Vertex>, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
    verts.push(Pos3Color4Vertex {
        position: a.to_array(),
        color,
    });
    verts.push(Pos3Color4Vertex {
        position: b.to_array(),
        color,
    });
    verts.push(Pos3Color4Vertex {
        position: c.to_array(),
        color,
    });
}
