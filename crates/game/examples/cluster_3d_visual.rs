//! 3D Cluster Prototype - Visual Demo
//!
//! Renders rigid clump clusters using an instanced low-poly rock mesh.
//! Run with: cargo run --example cluster_3d_visual --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use sim3d::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, IrregularStyle3D};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const MAX_INSTANCES: usize = 50_000;
const DT: f32 = 1.0 / 240.0;
const SUBSTEPS: usize = 4;
const CLUSTER_VISUAL_SCALE: f32 = 0.9;
const SPAWN_GAP: f32 = 0.8;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneVertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterInstance {
    position: [f32; 3],
    scale: f32,
    rotation: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    base_scale: f32,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sim: ClusterSimulation3D,
    template_colors: Vec<[f32; 4]>,
    spawn_queue: Vec<SpawnSpec>,
    spawn_timer: f32,
    spawn_cursor: usize,
    instances: Vec<ClusterInstance>,
    round_instances: Vec<ClusterInstance>,
    sharp_instances: Vec<ClusterInstance>,
    paused: bool,
    camera_angle: f32,
    camera_distance: f32,
    camera_height: f32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    plane_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    plane_vertex_buffer: wgpu::Buffer,
    plane_vertex_count: u32,
    round_mesh_vertex_buffer: wgpu::Buffer,
    round_mesh_vertex_count: u32,
    sharp_mesh_vertex_buffer: wgpu::Buffer,
    sharp_mesh_vertex_count: u32,
    instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl App {
    fn new() -> Self {
        let (sim, template_colors, spawn_queue) = build_sim();
        let (camera_distance, camera_height) = camera_defaults(&sim);
        let mut app = Self {
            window: None,
            gpu: None,
            sim,
            template_colors,
            spawn_queue,
            spawn_timer: 0.0,
            spawn_cursor: 0,
            instances: Vec::new(),
            round_instances: Vec::new(),
            sharp_instances: Vec::new(),
            paused: false,
            camera_angle: 0.6,
            camera_distance,
            camera_height,
        };

        app.spawn_next();
        println!(
            "Templates: {}, spawned: {}",
            app.sim.templates.len(),
            app.sim.clumps.len()
        );
        println!("Controls: SPACE=pause, R=reset, arrows=orbit/zoom");
        app
    }

    fn reset_sim(&mut self) {
        let (sim, template_colors, spawn_queue) = build_sim();
        let (camera_distance, camera_height) = camera_defaults(&sim);
        self.sim = sim;
        self.template_colors = template_colors;
        self.spawn_queue = spawn_queue;
        self.spawn_timer = 0.0;
        self.spawn_cursor = 0;
        self.camera_distance = camera_distance;
        self.camera_height = camera_height;
        self.spawn_next();
    }

    fn spawn_next(&mut self) {
        if self.spawn_cursor >= self.spawn_queue.len() {
            return;
        }
        let spec = self.spawn_queue[self.spawn_cursor];
        self.sim
            .spawn(spec.template_idx, spec.position, spec.velocity);
        self.spawn_cursor += 1;
    }

    fn update_spawns(&mut self, dt: f32) {
        if self.spawn_cursor >= self.spawn_queue.len() {
            return;
        }
        self.spawn_timer += dt;
        if self.spawn_timer >= SPAWN_GAP {
            self.spawn_timer -= SPAWN_GAP;
            self.spawn_next();
        }
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

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
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cluster Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let round_profile = RockProfile {
            radial_jitter: 0.08,
            lateral_jitter: 0.04,
            spike_chance: 0.0,
            spike_scale: 0.0,
            seed: 0xB2D4_09A7,
        };
        let sharp_profile = RockProfile {
            radial_jitter: 0.32,
            lateral_jitter: 0.2,
            spike_chance: 0.6,
            spike_scale: 0.7,
            seed: 0x65A1_F07C,
        };

        let round_mesh_vertices = build_rock_vertices(round_profile);
        let sharp_mesh_vertices = build_rock_vertices(sharp_profile);

        let round_mesh_vertex_count = round_mesh_vertices.len() as u32;
        let sharp_mesh_vertex_count = sharp_mesh_vertices.len() as u32;

        let round_mesh_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Round Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(&round_mesh_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let sharp_mesh_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sharp Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(&sharp_mesh_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (MAX_INSTANCES * std::mem::size_of::<ClusterInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let plane_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plane Shader"),
            source: wgpu::ShaderSource::Wgsl(PLANE_SHADER.into()),
        });

        let plane_color = [0.12, 0.12, 0.16, 1.0];
        let bounds_min = self.sim.bounds_min;
        let bounds_max = self.sim.bounds_max;
        let y = bounds_min.y;
        let plane_vertices = [
            PlaneVertex {
                position: [bounds_min.x, y, bounds_min.z],
                color: plane_color,
            },
            PlaneVertex {
                position: [bounds_max.x, y, bounds_min.z],
                color: plane_color,
            },
            PlaneVertex {
                position: [bounds_max.x, y, bounds_max.z],
                color: plane_color,
            },
            PlaneVertex {
                position: [bounds_min.x, y, bounds_min.z],
                color: plane_color,
            },
            PlaneVertex {
                position: [bounds_max.x, y, bounds_max.z],
                color: plane_color,
            },
            PlaneVertex {
                position: [bounds_min.x, y, bounds_max.z],
                color: plane_color,
            },
        ];
        let plane_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Plane Vertex Buffer"),
            contents: bytemuck::cast_slice(&plane_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let plane_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Plane Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &plane_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PlaneVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &plane_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cluster Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ClusterInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 5,
                            },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (depth_texture, depth_view) = create_depth_texture(&device, &config);

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            plane_pipeline,
            pipeline,
            plane_vertex_buffer,
            plane_vertex_count: plane_vertices.len() as u32,
            round_mesh_vertex_buffer,
            round_mesh_vertex_count,
            sharp_mesh_vertex_buffer,
            sharp_mesh_vertex_count,
            instance_buffer,
            uniform_buffer,
            bind_group,
            depth_texture,
            depth_view,
        });
    }

    fn build_instances(&mut self) -> (usize, usize) {
        self.instances.clear();
        self.round_instances.clear();
        self.sharp_instances.clear();

        for clump in &self.sim.clumps {
            let template = &self.sim.templates[clump.template_idx];
            let color = self.template_colors[clump.template_idx % self.template_colors.len()];

            if self.round_instances.len() + self.sharp_instances.len() >= MAX_INSTANCES {
                break;
            }
            let scale = template.bounding_radius * CLUSTER_VISUAL_SCALE;
            let instance = ClusterInstance {
                position: clump.position.to_array(),
                scale,
                rotation: clump.rotation.to_array(),
                color,
            };
            match template.shape {
                ClumpShape3D::Irregular {
                    style: IrregularStyle3D::Sharp,
                    ..
                } => self.sharp_instances.push(instance),
                _ => self.round_instances.push(instance),
            }
        }

        let round_count = self.round_instances.len();
        let sharp_count = self.sharp_instances.len();
        self.instances.extend_from_slice(&self.round_instances);
        self.instances.extend_from_slice(&self.sharp_instances);

        (round_count, sharp_count)
    }

    fn render(&mut self) {
        let window = match &self.window {
            Some(window) => Arc::clone(window),
            None => return,
        };
        let mut gpu = match self.gpu.take() {
            Some(gpu) => gpu,
            None => return,
        };

        if !self.paused {
            for _ in 0..SUBSTEPS {
                self.update_spawns(DT);
                self.sim.step(DT);
            }
        }

        let bounds_min = self.sim.bounds_min;
        let bounds_max = self.sim.bounds_max;
        let center = (bounds_min + bounds_max) * 0.5;
        let camera_pos = center
            + Vec3::new(
                self.camera_angle.cos() * self.camera_distance,
                self.camera_height,
                self.camera_angle.sin() * self.camera_distance,
            );

        let view = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj * view;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            base_scale: 1.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let (round_count, sharp_count) = self.build_instances();
        let instance_count = round_count + sharp_count;
        if instance_count > 0 {
            gpu.queue.write_buffer(
                &gpu.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances[..instance_count]),
            );
        }

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = window.inner_size();
                gpu.config.width = size.width.max(1);
                gpu.config.height = size.height.max(1);
                gpu.surface.configure(&gpu.device, &gpu.config);
                let (depth_texture, depth_view) = create_depth_texture(&gpu.device, &gpu.config);
                gpu.depth_texture = depth_texture;
                gpu.depth_view = depth_view;
                self.gpu = Some(gpu);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                self.gpu = Some(gpu);
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => {
                self.gpu = Some(gpu);
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.03,
                            b: 0.06,
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

            pass.set_pipeline(&gpu.plane_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.plane_vertex_buffer.slice(..));
            pass.draw(0..gpu.plane_vertex_count, 0..1);

            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));

            let round_count_u32 = round_count as u32;
            let sharp_count_u32 = sharp_count as u32;
            if round_count_u32 > 0 {
                pass.set_vertex_buffer(0, gpu.round_mesh_vertex_buffer.slice(..));
                pass.draw(0..gpu.round_mesh_vertex_count, 0..round_count_u32);
            }
            if sharp_count_u32 > 0 {
                pass.set_vertex_buffer(0, gpu.sharp_mesh_vertex_buffer.slice(..));
                pass.draw(
                    0..gpu.sharp_mesh_vertex_count,
                    round_count_u32..round_count_u32 + sharp_count_u32,
                );
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        window.request_redraw();
        self.gpu = Some(gpu);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("3D Cluster Prototype")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 800));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => self.paused = !self.paused,
                        PhysicalKey::Code(KeyCode::KeyR) => self.reset_sim(),
                        PhysicalKey::Code(KeyCode::ArrowLeft) => self.camera_angle -= 0.1,
                        PhysicalKey::Code(KeyCode::ArrowRight) => self.camera_angle += 0.1,
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.camera_distance = (self.camera_distance - 0.5).max(2.0)
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => self.camera_distance += 0.5,
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    let (depth_texture, depth_view) =
                        create_depth_texture(&gpu.device, &gpu.config);
                    gpu.depth_texture = depth_texture;
                    gpu.depth_view = depth_view;
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

const PLANE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    base_scale: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    base_scale: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_pos: vec3<f32>,
    @location(3) instance_scale: f32,
    @location(4) instance_rot: vec4<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let scaled = in.position * (in.instance_scale * uniforms.base_scale);
    let world_pos = in.instance_pos + quat_rotate(in.instance_rot, scaled);
    let normal = normalize(quat_rotate(in.instance_rot, in.normal));
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.2));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let view_dir = normalize(uniforms.camera_pos - world_pos);
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
    let shade = 0.35 + 0.65 * diffuse;
    let tint = in.color.rgb * shade + vec3<f32>(0.08) * rim;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(tint, in.color.a);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
"#;

#[derive(Clone, Copy)]
struct RockProfile {
    radial_jitter: f32,
    lateral_jitter: f32,
    spike_chance: f32,
    spike_scale: f32,
    seed: u32,
}

fn build_rock_vertices(profile: RockProfile) -> Vec<MeshVertex> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let inv_len = 1.0 / (1.0 + phi * phi).sqrt();
    let a = inv_len;
    let b = phi * inv_len;

    let mut verts = [
        Vec3::new(-a, b, 0.0),
        Vec3::new(a, b, 0.0),
        Vec3::new(-a, -b, 0.0),
        Vec3::new(a, -b, 0.0),
        Vec3::new(0.0, -a, b),
        Vec3::new(0.0, a, b),
        Vec3::new(0.0, -a, -b),
        Vec3::new(0.0, a, -b),
        Vec3::new(b, 0.0, -a),
        Vec3::new(b, 0.0, a),
        Vec3::new(-b, 0.0, -a),
        Vec3::new(-b, 0.0, a),
    ];

    let seed = profile.seed;
    for (idx, pos) in verts.iter_mut().enumerate() {
        let idx_u = idx as u32;
        let radial = 1.0 + profile.radial_jitter * hash_to_unit(seed ^ idx_u.wrapping_mul(11));
        let lateral = Vec3::new(
            hash_to_unit(seed ^ idx_u.wrapping_mul(13)),
            hash_to_unit(seed ^ idx_u.wrapping_mul(17)),
            hash_to_unit(seed ^ idx_u.wrapping_mul(19)),
        ) * profile.lateral_jitter;
        let mut adjusted = (*pos * radial) + lateral;

        let spike_roll = hash_to_01(seed ^ idx_u.wrapping_mul(23));
        if spike_roll < profile.spike_chance {
            let axis_pick = (hash_to_01(seed ^ idx_u.wrapping_mul(29)) * 3.0) as u32;
            let spike = 1.0 + profile.spike_scale * hash_to_01(seed ^ idx_u.wrapping_mul(31));
            match axis_pick {
                0 => adjusted.x *= spike,
                1 => adjusted.y *= spike,
                _ => adjusted.z *= spike,
            }
        }

        *pos = adjusted;
    }

    let mut max_len = 0.0_f32;
    for pos in &verts {
        max_len = max_len.max(pos.length());
    }
    if max_len > 0.0 {
        for pos in &mut verts {
            *pos /= max_len;
        }
    }

    let indices: [[usize; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    let mut vertices = Vec::with_capacity(indices.len() * 3);
    for tri in indices {
        let a = verts[tri[0]];
        let b = verts[tri[1]];
        let c = verts[tri[2]];
        let normal = (b - a).cross(c - a).normalize();
        for pos in [a, b, c] {
            vertices.push(MeshVertex {
                position: pos.to_array(),
                normal: normal.to_array(),
            });
        }
    }

    vertices
}

fn hash_to_unit(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    let unit = x as f32 / u32::MAX as f32;
    unit * 2.0 - 1.0
}

fn hash_to_01(x: u32) -> f32 {
    hash_to_unit(x) * 0.5 + 0.5
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn camera_defaults(sim: &ClusterSimulation3D) -> (f32, f32) {
    let extent = sim.bounds_max - sim.bounds_min;
    let span = extent.x.max(extent.z);
    let distance = (span * 0.9 + 2.0).max(6.0);
    let height = (extent.y * 0.7 + 1.0).max(2.5);
    (distance, height)
}

#[derive(Clone, Copy)]
struct SpawnSpec {
    template_idx: usize,
    position: Vec3,
    velocity: Vec3,
}

#[derive(Clone, Copy)]
struct TemplateSpec {
    count: usize,
    style: IrregularStyle3D,
    particle_radius: f32,
    particle_mass: f32,
    color: [f32; 4],
}

fn push_specs(
    specs: &mut Vec<TemplateSpec>,
    counts: &[usize],
    styles: &[IrregularStyle3D],
    particle_radius: f32,
    particle_mass: f32,
    color: [f32; 4],
) {
    for &count in counts {
        for &style in styles {
            specs.push(TemplateSpec {
                count,
                style,
                particle_radius,
                particle_mass,
                color,
            });
        }
    }
}

fn build_sim() -> (ClusterSimulation3D, Vec<[f32; 4]>, Vec<SpawnSpec>) {
    let round_sharp = [IrregularStyle3D::Round, IrregularStyle3D::Sharp];
    let mut specs = Vec::new();

    specs.push(TemplateSpec {
        count: 1,
        style: IrregularStyle3D::Round,
        particle_radius: 0.02,
        particle_mass: 0.6,
        color: [0.82, 0.75, 0.6, 0.95],
    });

    specs.push(TemplateSpec {
        count: 1,
        style: IrregularStyle3D::Round,
        particle_radius: 0.02,
        particle_mass: 2.2,
        color: [0.95, 0.82, 0.2, 0.95],
    });

    push_specs(
        &mut specs,
        &[4, 7, 10],
        &round_sharp,
        0.03,
        1.0,
        [0.55, 0.55, 0.58, 0.95],
    );

    push_specs(
        &mut specs,
        &[4, 7, 10],
        &round_sharp,
        0.028,
        2.4,
        [0.95, 0.78, 0.25, 0.95],
    );

    push_specs(
        &mut specs,
        &[20, 30, 40],
        &round_sharp,
        0.04,
        1.2,
        [0.48, 0.5, 0.54, 0.95],
    );

    push_specs(
        &mut specs,
        &[60, 120, 200],
        &round_sharp,
        0.055,
        1.4,
        [0.35, 0.38, 0.42, 0.95],
    );

    let mut templates = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let shape = ClumpShape3D::Irregular {
            count: spec.count,
            seed: 900 + idx as u64 * 17,
            style: spec.style,
        };
        templates.push(ClumpTemplate3D::generate(
            shape,
            spec.particle_radius,
            spec.particle_mass,
        ));
    }

    let max_radius = templates
        .iter()
        .map(|template| template.bounding_radius)
        .fold(0.0_f32, f32::max);
    let grid_cols = (templates.len() as f32).sqrt().ceil() as usize;
    let grid_rows = (templates.len() + grid_cols - 1) / grid_cols;
    let spacing = (max_radius * 2.8).max(0.3);
    let margin = max_radius * 1.5 + 0.5;
    let width = (grid_cols as f32 - 1.0).max(0.0) * spacing + margin * 2.0;
    let depth = (grid_rows as f32 - 1.0).max(0.0) * spacing + margin * 2.0;
    let height = max_radius * 6.0 + 2.0;

    let bounds_min = Vec3::new(0.0, 0.0, 0.0);
    let bounds_max = Vec3::new(width, height, depth);
    let mut sim = ClusterSimulation3D::new(bounds_min, bounds_max);
    sim.use_dem = true;
    sim.restitution = 0.25;
    sim.friction = 0.7;
    sim.floor_friction = 1.1;
    sim.normal_stiffness = 45_000.0;
    sim.tangential_stiffness = 20_000.0;
    sim.rolling_friction = 0.05;

    let mut template_colors = Vec::with_capacity(templates.len());
    for (idx, template) in templates.into_iter().enumerate() {
        sim.add_template(template);
        template_colors.push(specs[idx].color);
    }

    let spawn_origin = Vec3::new(width * 0.5, bounds_max.y - max_radius * 0.7, depth * 0.5);
    let jitter = max_radius * 0.35;
    let mut spawn_queue = Vec::with_capacity(sim.templates.len());
    for idx in 0..sim.templates.len() {
        let idx_u = idx as u32;
        let jx = hash_to_unit(0xA53A_91C3_u32 ^ idx_u.wrapping_mul(31)) * jitter;
        let jz = hash_to_unit(0xC1B2_9ED1_u32 ^ idx_u.wrapping_mul(27)) * jitter;
        let position = spawn_origin + Vec3::new(jx, 0.0, jz);
        spawn_queue.push(SpawnSpec {
            template_idx: idx,
            position,
            velocity: Vec3::ZERO,
        });
    }

    (sim, template_colors, spawn_queue)
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
