//! Shaker Deck Demo
//!
//! Demonstrates a vibrating shaker deck using DEM clumps to wash fines
//! (gold + gangue) off rocks and separate undersize through screen holes.
//!
//! Run with: cargo run --example shaker_deck --release

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

const MAX_INSTANCES: usize = 80_000;
const DT: f32 = 1.0 / 240.0;
const SUBSTEPS: usize = 4;
const CLUSTER_VISUAL_SCALE: f32 = 0.9;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const DECK_LENGTH: f32 = 6.0;
const DECK_WIDTH: f32 = 2.2;
const DECK_HEIGHT: f32 = 2.0;
const DECK_ANGLE_DEG: f32 = 12.0;
const HOLE_SPACING: f32 = 0.18;
const APERTURE_DIAMETER: f32 = 0.06;

const SHAKE_FREQ: f32 = 18.0;
const SHAKE_AMP: f32 = 0.012;
const SHAKE_TANGENT: f32 = 0.55;
const WATER_DRAG: f32 = 1.6;
const WATER_FLOW: f32 = 2.8;
const BREAK_ACCEL: f32 = 16.0;

const FINE_GANGUE_DENSITY: f32 = 2.7;
const FINE_GOLD_DENSITY: f32 = 19.3;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaterialTag {
    Rock,
    Gangue,
    Gold,
}

#[derive(Clone, Copy)]
struct Attachment {
    fine_idx: usize,
    rock_idx: usize,
    local_offset: Vec3,
    break_accel: f32,
    attached: bool,
}

struct DeckSpec {
    origin: Vec3,
    length: f32,
    width: f32,
    slope: f32,
    hole_spacing: f32,
    hole_radius: f32,
}

impl DeckSpec {
    fn height_at(&self, x: f32) -> f32 {
        self.origin.y + self.slope * (x - self.origin.x)
    }

    fn inside_bounds(&self, pos: Vec3) -> bool {
        pos.x >= self.origin.x
            && pos.x <= self.origin.x + self.length
            && pos.z >= self.origin.z
            && pos.z <= self.origin.z + self.width
    }

    fn in_hole(&self, pos: Vec3) -> bool {
        let local_x = pos.x - self.origin.x;
        let local_z = pos.z - self.origin.z;
        let gx = (local_x / self.hole_spacing).round();
        let gz = (local_z / self.hole_spacing).round();
        let cx = gx * self.hole_spacing;
        let cz = gz * self.hole_spacing;
        let dx = local_x - cx;
        let dz = local_z - cz;
        dx * dx + dz * dz <= self.hole_radius * self.hole_radius
    }

    fn normal(&self) -> Vec3 {
        Vec3::new(-self.slope, 1.0, 0.0).normalize()
    }

    fn tangent(&self) -> Vec3 {
        Vec3::new(1.0, self.slope, 0.0).normalize()
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sim: ClusterSimulation3D,
    template_colors: Vec<[f32; 4]>,
    instances: Vec<ClusterInstance>,
    round_instances: Vec<ClusterInstance>,
    sharp_instances: Vec<ClusterInstance>,
    materials: Vec<MaterialTag>,
    attachments: Vec<Attachment>,
    time: f32,
    paused: bool,
    camera_angle: f32,
    camera_distance: f32,
    camera_height: f32,
    deck: DeckSpec,
    gold_underflow: usize,
    gangue_underflow: usize,
    gold_overflow: usize,
    gangue_overflow: usize,
    counted: Vec<bool>,
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
        let (sim, template_colors, materials, attachments, deck) = build_sim();
        let (camera_distance, camera_height) = camera_defaults(&sim);
        let counted = vec![false; sim.clumps.len()];

        println!(
            "Shaker deck: rocks={} fines={} (gold+gangue)",
            materials.iter().filter(|m| **m == MaterialTag::Rock).count(),
            materials.len() - materials.iter().filter(|m| **m == MaterialTag::Rock).count()
        );
        println!("Controls: SPACE=pause, R=reset, arrows=orbit/zoom");

        Self {
            window: None,
            gpu: None,
            sim,
            template_colors,
            instances: Vec::new(),
            round_instances: Vec::new(),
            sharp_instances: Vec::new(),
            materials,
            attachments,
            time: 0.0,
            paused: false,
            camera_angle: 0.6,
            camera_distance,
            camera_height,
            deck,
            gold_underflow: 0,
            gangue_underflow: 0,
            gold_overflow: 0,
            gangue_overflow: 0,
            counted,
        }
    }

    fn reset_sim(&mut self) {
        let (sim, template_colors, materials, attachments, deck) = build_sim();
        let (camera_distance, camera_height) = camera_defaults(&sim);
        self.sim = sim;
        self.template_colors = template_colors;
        self.materials = materials;
        self.attachments = attachments;
        self.deck = deck;
        self.time = 0.0;
        self.gold_underflow = 0;
        self.gangue_underflow = 0;
        self.gold_overflow = 0;
        self.gangue_overflow = 0;
        self.counted = vec![false; self.sim.clumps.len()];
        self.camera_distance = camera_distance;
        self.camera_height = camera_height;
    }

    fn update_sim(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        let sub_dt = dt / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.time += sub_dt;
            let shake_accel = self.shake_accel();
            self.apply_shake(shake_accel, sub_dt);
            self.sim.step(sub_dt);
            self.apply_deck_constraints(sub_dt);
            self.apply_water_drag(sub_dt);
            self.update_attachments(shake_accel);
            self.update_counts();
        }
    }

    fn shake_accel(&self) -> Vec3 {
        let omega = std::f32::consts::TAU * SHAKE_FREQ;
        let phase = (omega * self.time).sin();
        let deck_normal = self.deck.normal();
        let deck_tangent = self.deck.tangent();
        let normal_accel = deck_normal * (SHAKE_AMP * omega * omega * phase);
        let tangent_accel = deck_tangent * (SHAKE_TANGENT * phase.cos());
        normal_accel + tangent_accel
    }

    fn apply_shake(&mut self, accel: Vec3, dt: f32) {
        for clump in &mut self.sim.clumps {
            clump.velocity += accel * dt;
        }
    }

    fn apply_water_drag(&mut self, dt: f32) {
        let deck_tangent = self.deck.tangent();
        for (idx, clump) in self.sim.clumps.iter_mut().enumerate() {
            if matches!(self.materials[idx], MaterialTag::Rock) {
                continue;
            }
            let drag = (1.0 - WATER_DRAG * dt).max(0.0);
            clump.velocity *= drag;
            clump.velocity += deck_tangent * WATER_FLOW * dt;
        }
    }

    fn apply_deck_constraints(&mut self, dt: f32) {
        let deck = &self.deck;
        let normal = deck.normal();
        let friction = 0.55;
        for (idx, clump) in self.sim.clumps.iter_mut().enumerate() {
            let template = &self.sim.templates[clump.template_idx];
            let mut max_penetration = 0.0f32;
            let mut blocked = false;

            for offset in &template.local_offsets {
                let local = clump.rotation * *offset;
                let pos = clump.position + local;
                if !deck.inside_bounds(pos) {
                    continue;
                }
                let plane_y = deck.height_at(pos.x);
                let penetration = plane_y + template.particle_radius - pos.y;
                if penetration > 0.0 {
                    let in_hole = deck.in_hole(pos);
                    let fine = matches!(self.materials[idx], MaterialTag::Gold | MaterialTag::Gangue);
                    if in_hole && fine && template.particle_radius * 2.0 <= APERTURE_DIAMETER {
                        continue;
                    }
                    blocked = true;
                    if penetration > max_penetration {
                        max_penetration = penetration;
                    }
                }
            }

            if blocked && max_penetration > 0.0 {
                clump.position += normal * max_penetration;
                let v_n = clump.velocity.dot(normal);
                if v_n < 0.0 {
                    let v_t = clump.velocity - normal * v_n;
                    let v_t = v_t * (1.0 - friction * dt * 8.0).max(0.0);
                    clump.velocity = v_t;
                }
            }
        }
    }

    fn update_attachments(&mut self, shake_accel: Vec3) {
        for attachment in &mut self.attachments {
            if !attachment.attached {
                continue;
            }
            if shake_accel.length() > attachment.break_accel {
                attachment.attached = false;
                continue;
            }
            if attachment.rock_idx >= self.sim.clumps.len() {
                attachment.attached = false;
                continue;
            }
            let rock = self.sim.clumps[attachment.rock_idx];
            if attachment.fine_idx >= self.sim.clumps.len() {
                attachment.attached = false;
                continue;
            }
            let fine = &mut self.sim.clumps[attachment.fine_idx];
            fine.position = rock.position + rock.rotation * attachment.local_offset;
            fine.velocity = rock.velocity;
            fine.angular_velocity = rock.angular_velocity;
        }
    }

    fn update_counts(&mut self) {
        let deck = &self.deck;
        let underflow_y = deck.origin.y - 0.25;
        let overflow_x = deck.origin.x + deck.length + 0.4;

        for (idx, clump) in self.sim.clumps.iter().enumerate() {
            if self.counted[idx] {
                continue;
            }
            let pos = clump.position;
            if pos.y < underflow_y {
                match self.materials[idx] {
                    MaterialTag::Gold => self.gold_underflow += 1,
                    MaterialTag::Gangue => self.gangue_underflow += 1,
                    MaterialTag::Rock => {}
                }
                self.counted[idx] = true;
            } else if pos.x > overflow_x {
                match self.materials[idx] {
                    MaterialTag::Gold => self.gold_overflow += 1,
                    MaterialTag::Gangue => self.gangue_overflow += 1,
                    MaterialTag::Rock => {}
                }
                self.counted[idx] = true;
            }
        }
    }

    fn rebuild_instances(&mut self) {
        self.instances.clear();
        self.round_instances.clear();
        self.sharp_instances.clear();

        for (_idx, clump) in self.sim.clumps.iter().enumerate() {
            let template = &self.sim.templates[clump.template_idx];
            let color = self.template_colors[clump.template_idx];
            let instance = ClusterInstance {
                position: clump.position.to_array(),
                scale: template.bounding_radius * CLUSTER_VISUAL_SCALE,
                rotation: clump.rotation.to_array(),
                color,
            };

            match template.shape {
                ClumpShape3D::Irregular { style, .. } => match style {
                    IrregularStyle3D::Round => self.round_instances.push(instance),
                    IrregularStyle3D::Sharp => self.sharp_instances.push(instance),
                },
                _ => self.round_instances.push(instance),
            }

        }

        self.instances.clear();
        self.instances.extend_from_slice(&self.round_instances);
        self.instances.extend_from_slice(&self.sharp_instances);
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cluster Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let plane_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plane Shader"),
            source: wgpu::ShaderSource::Wgsl(PLANE_SHADER.into()),
        });

        let plane_color = [0.16, 0.18, 0.2, 1.0];
        let deck = &self.deck;
        let x0 = deck.origin.x;
        let x1 = deck.origin.x + deck.length;
        let z0 = deck.origin.z;
        let z1 = deck.origin.z + deck.width;
        let y0 = deck.height_at(x0);
        let y1 = deck.height_at(x1);

        let plane_vertices = [
            PlaneVertex {
                position: [x0, y0, z0],
                color: plane_color,
            },
            PlaneVertex {
                position: [x1, y1, z0],
                color: plane_color,
            },
            PlaneVertex {
                position: [x1, y1, z1],
                color: plane_color,
            },
            PlaneVertex {
                position: [x0, y0, z0],
                color: plane_color,
            },
            PlaneVertex {
                position: [x1, y1, z1],
                color: plane_color,
            },
            PlaneVertex {
                position: [x0, y0, z1],
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
                compilation_options: Default::default(),
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
            },
            fragment: Some(wgpu::FragmentState {
                module: &plane_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            cache: None,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cluster Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
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
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            cache: None,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let round_mesh = build_round_mesh();
        let sharp_mesh = build_sharp_mesh();

        let round_mesh_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Round Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&round_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sharp_mesh_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sharp Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&sharp_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<ClusterInstance>() * MAX_INSTANCES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (depth_texture, depth_view) = create_depth_texture(&device, config.width, config.height);

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
            round_mesh_vertex_count: round_mesh.len() as u32,
            sharp_mesh_vertex_buffer,
            sharp_mesh_vertex_count: sharp_mesh.len() as u32,
            instance_buffer,
            uniform_buffer,
            bind_group,
            depth_texture,
            depth_view,
        });
    }

    fn update_gpu(&mut self) {
        let view_proj = self.camera_matrix();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [
                self.camera_distance * self.camera_angle.cos(),
                self.camera_height,
                self.camera_distance * self.camera_angle.sin(),
            ],
            base_scale: 1.0,
        };
        let gpu = self.gpu.as_mut().unwrap();
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let total_instances = self.instances.len().min(MAX_INSTANCES);
        let instance_data = &self.instances[..total_instances];
        gpu.queue.write_buffer(
            &gpu.instance_buffer,
            0,
            bytemuck::cast_slice(instance_data),
        );
    }

    fn camera_matrix(&self) -> Mat4 {
        let eye = Vec3::new(
            self.camera_distance * self.camera_angle.cos(),
            self.camera_height,
            self.camera_distance * self.camera_angle.sin(),
        );
        let target = Vec3::new(
            self.deck.origin.x + self.deck.length * 0.4,
            self.deck.origin.y + 0.2,
            self.deck.origin.z + self.deck.width * 0.5,
        );
        let up = Vec3::Y;
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh_gl(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        proj * view
    }

    fn render(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        let frame = match gpu.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                gpu.surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
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
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&gpu.plane_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.plane_vertex_buffer.slice(..));
            pass.draw(0..gpu.plane_vertex_count, 0..1);

            let total_round = self.round_instances.len().min(MAX_INSTANCES);
            if total_round > 0 {
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.round_mesh_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
                pass.draw(0..gpu.round_mesh_vertex_count, 0..total_round as u32);
            }

            let total_sharp = self.sharp_instances.len().min(MAX_INSTANCES - total_round);
            if total_sharp > 0 {
                let instance_offset = (total_round * std::mem::size_of::<ClusterInstance>()) as u64;
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.sharp_mesh_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.instance_buffer.slice(instance_offset..));
                pass.draw(0..gpu.sharp_mesh_vertex_count, 0..total_sharp as u32);
            }
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.window = Some(window.clone());
        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    let (depth_texture, depth_view) =
                        create_depth_texture(&gpu.device, size.width, size.height);
                    gpu.depth_texture = depth_texture;
                    gpu.depth_view = depth_view;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == winit::event::ElementState::Pressed {
                        match code {
                            KeyCode::Space => self.paused = !self.paused,
                            KeyCode::KeyR => self.reset_sim(),
                            KeyCode::ArrowLeft => self.camera_angle -= 0.1,
                            KeyCode::ArrowRight => self.camera_angle += 0.1,
                            KeyCode::ArrowUp => self.camera_distance = (self.camera_distance - 0.4).max(2.0),
                            KeyCode::ArrowDown => self.camera_distance += 0.4,
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_sim(DT);
                self.rebuild_instances();
                self.update_gpu();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn camera_defaults(sim: &ClusterSimulation3D) -> (f32, f32) {
    let extent = sim.bounds_max - sim.bounds_min;
    let span = extent.x.max(extent.z);
    let distance = (span * 0.9 + 2.0).max(6.0);
    let height = (extent.y * 0.7 + 1.0).max(2.5);
    (distance, height)
}

fn build_sim() -> (
    ClusterSimulation3D,
    Vec<[f32; 4]>,
    Vec<MaterialTag>,
    Vec<Attachment>,
    DeckSpec,
) {
    let slope = -DECK_ANGLE_DEG.to_radians().tan();
    let deck_origin = Vec3::new(0.4, DECK_HEIGHT, 0.4);
    let deck = DeckSpec {
        origin: deck_origin,
        length: DECK_LENGTH,
        width: DECK_WIDTH,
        slope,
        hole_spacing: HOLE_SPACING,
        hole_radius: APERTURE_DIAMETER * 0.5,
    };

    let bounds_min = Vec3::new(0.0, 0.0, 0.0);
    let bounds_max = Vec3::new(
        deck_origin.x + DECK_LENGTH + 2.0,
        DECK_HEIGHT + 3.0,
        deck_origin.z + DECK_WIDTH + 1.0,
    );

    let mut sim = ClusterSimulation3D::new(bounds_min, bounds_max);
    sim.use_dem = true;
    sim.restitution = 0.15;
    sim.friction = 0.7;
    sim.floor_friction = 0.9;
    sim.normal_stiffness = 40_000.0;
    sim.tangential_stiffness = 18_000.0;
    sim.rolling_friction = 0.05;

    let rock_radius = 0.08;
    let rock_density = 2.7;
    let rock_mass = density_mass(rock_density, rock_radius);

    let gangue_radius = 0.02;
    let gangue_mass = density_mass(FINE_GANGUE_DENSITY, gangue_radius);

    let gold_radius = 0.012;
    let gold_mass = density_mass(FINE_GOLD_DENSITY, gold_radius);

    let rock_template = ClumpTemplate3D::generate(
        ClumpShape3D::Irregular {
            count: 7,
            seed: 10,
            style: IrregularStyle3D::Sharp,
        },
        rock_radius,
        rock_mass,
    );

    let gangue_template = ClumpTemplate3D::generate(
        ClumpShape3D::Irregular {
            count: 1,
            seed: 5,
            style: IrregularStyle3D::Round,
        },
        gangue_radius,
        gangue_mass,
    );

    let gold_template = ClumpTemplate3D::generate(ClumpShape3D::Flat4, gold_radius, gold_mass);

    let rock_idx = sim.add_template(rock_template);
    let gangue_idx = sim.add_template(gangue_template);
    let gold_idx = sim.add_template(gold_template);

    let mut template_colors = Vec::new();
    template_colors.push([0.35, 0.34, 0.32, 0.95]);
    template_colors.push([0.6, 0.58, 0.55, 0.95]);
    template_colors.push([0.95, 0.78, 0.2, 0.95]);

    let mut materials = Vec::new();
    let mut attachments = Vec::new();

    let rock_count = 40;
    for i in 0..rock_count {
        let i_u = i as u32;
        let x = deck_origin.x
            + 0.6
            + hash_range(0xA53A_91C3 ^ i_u.wrapping_mul(31), 0.0, DECK_LENGTH * 0.4);
        let z = deck_origin.z
            + 0.4
            + hash_range(0xC1B2_9ED1 ^ i_u.wrapping_mul(27), 0.0, DECK_WIDTH * 0.8);
        let y = deck.height_at(x)
            + 0.3
            + hash_range(0x9C5B_D781 ^ i_u.wrapping_mul(19), 0.0, 0.2);
        let rock_idx_spawn = sim.spawn(rock_idx, Vec3::new(x, y, z), Vec3::ZERO);
        materials.push(MaterialTag::Rock);

        let fine_count = 6 + (i % 4);
        for j in 0..fine_count {
            let is_gold = j % 3 == 0;
            let template_idx = if is_gold { gold_idx } else { gangue_idx };
            let fine_idx_spawn = sim.spawn(template_idx, Vec3::new(x, y + 0.05, z), Vec3::ZERO);
            materials.push(if is_gold { MaterialTag::Gold } else { MaterialTag::Gangue });
            let j_u = j as u32;
            let offset = Vec3::new(
                hash_range(0xB3A1_4F2D ^ i_u.wrapping_mul(13) ^ j_u.wrapping_mul(7), -0.06, 0.06),
                hash_range(0x56C9_1E77 ^ i_u.wrapping_mul(17) ^ j_u.wrapping_mul(11), 0.02, 0.07),
                hash_range(0x1D3F_9A55 ^ i_u.wrapping_mul(23) ^ j_u.wrapping_mul(5), -0.06, 0.06),
            );
            attachments.push(Attachment {
                fine_idx: fine_idx_spawn,
                rock_idx: rock_idx_spawn,
                local_offset: offset,
                break_accel: BREAK_ACCEL
                    + hash_range(0x6F2C_8D11 ^ i_u.wrapping_mul(29) ^ j_u.wrapping_mul(3), -2.0, 2.0),
                attached: true,
            });
        }
    }

    (sim, template_colors, materials, attachments, deck)
}

fn density_mass(density: f32, radius: f32) -> f32 {
    density * (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3)
}

fn hash_to_unit(seed: u32) -> f32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

fn hash_range(seed: u32, min: f32, max: f32) -> f32 {
    min + (max - min) * hash_to_unit(seed)
}

fn build_round_mesh() -> Vec<MeshVertex> {
    let mut vertices = Vec::new();
    let steps = 16;
    for i in 0..steps {
        let theta0 = (i as f32) * std::f32::consts::TAU / steps as f32;
        let theta1 = ((i + 1) as f32) * std::f32::consts::TAU / steps as f32;
        for j in 0..steps {
            let phi0 = (j as f32) * std::f32::consts::PI / steps as f32;
            let phi1 = ((j + 1) as f32) * std::f32::consts::PI / steps as f32;

            let p00 = spherical_point(theta0, phi0);
            let p10 = spherical_point(theta1, phi0);
            let p01 = spherical_point(theta0, phi1);
            let p11 = spherical_point(theta1, phi1);

            vertices.push(MeshVertex {
                position: p00.to_array(),
                normal: p00.to_array(),
            });
            vertices.push(MeshVertex {
                position: p10.to_array(),
                normal: p10.to_array(),
            });
            vertices.push(MeshVertex {
                position: p11.to_array(),
                normal: p11.to_array(),
            });

            vertices.push(MeshVertex {
                position: p00.to_array(),
                normal: p00.to_array(),
            });
            vertices.push(MeshVertex {
                position: p11.to_array(),
                normal: p11.to_array(),
            });
            vertices.push(MeshVertex {
                position: p01.to_array(),
                normal: p01.to_array(),
            });
        }
    }
    vertices
}

fn build_sharp_mesh() -> Vec<MeshVertex> {
    let points = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.9428, -0.3333),
        Vec3::new(-0.8165, -0.4714, -0.3333),
        Vec3::new(0.8165, -0.4714, -0.3333),
    ];
    let faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)];
    let mut vertices = Vec::new();
    for (a, b, c) in faces {
        let pa = points[a];
        let pb = points[b];
        let pc = points[c];
        let normal = (pb - pa).cross(pc - pa).normalize();
        vertices.push(MeshVertex {
            position: pa.to_array(),
            normal: normal.to_array(),
        });
        vertices.push(MeshVertex {
            position: pb.to_array(),
            normal: normal.to_array(),
        });
        vertices.push(MeshVertex {
            position: pc.to_array(),
            normal: normal.to_array(),
        });
    }
    vertices
}

fn spherical_point(theta: f32, phi: f32) -> Vec3 {
    let sin_phi = phi.sin();
    Vec3::new(theta.cos() * sin_phi, phi.cos(), theta.sin() * sin_phi)
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width,
        height,
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

const PLANE_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    base_scale: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    base_scale: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_pos: vec3<f32>,
    @location(3) instance_scale: f32,
    @location(4) instance_rot: vec4<f32>,
    @location(5) instance_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let scaled = input.position * input.instance_scale * uniforms.base_scale;
    let rotated = quat_rotate(input.instance_rot, scaled);
    let world_pos = rotated + input.instance_pos;
    out.world_pos = world_pos;
    out.clip_pos = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.normal = normalize(quat_rotate(input.instance_rot, input.normal));
    out.color = input.instance_color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let ndotl = max(dot(input.normal, light_dir), 0.1);
    return vec4<f32>(input.color.rgb * ndotl, input.color.a);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
