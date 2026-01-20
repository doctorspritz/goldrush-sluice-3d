//! Basic Washplant Visual - Shows 4-stage equipment geometry
//!
//! Run with: cargo run --example washplant_basic --release

#![allow(unused_imports)]
use bytemuck::{Pod, Zeroable};
use game::app::{run, App, FlyCamera, GpuContext};
use game::equipment_geometry::{
    GrateConfig, GrateGeometryBuilder, HopperConfig, HopperGeometryBuilder,
};
use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder, SluiceVertex};
use glam::Vec3;
use wgpu::util::DeviceExt;

const CELL_SIZE: f32 = 0.02;
const STAGE_SPACING: f32 = 2.0; // Space between stages in world units

/// Rendered stage with GPU buffers
struct RenderedStage {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct WashplantApp {
    camera: FlyCamera,
    pipeline: wgpu::RenderPipeline,
    stages: Vec<RenderedStage>,
}

impl WashplantApp {
    fn create_pipeline(ctx: &GpuContext) -> wgpu::RenderPipeline {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Equipment Shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Equipment Pipeline Layout"),
                bind_group_layouts: &[&ctx.view_bind_group_layout],
                push_constant_ranges: &[],
            });

        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Equipment Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[SluiceVertex::buffer_layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    }

    fn build_stages(device: &wgpu::Device) -> Vec<RenderedStage> {
        let mut stages = Vec::new();
        let mut x_offset = 0.0;

        // Stage 1: Hopper (60x80x60 grid)
        {
            let config = HopperConfig {
                grid_width: 60,
                grid_height: 80,
                grid_depth: 60,
                cell_size: CELL_SIZE,
                top_width: 50,
                top_depth: 50,
                bottom_width: 15,
                bottom_depth: 15,
                wall_thickness: 2,
                ..Default::default()
            };
            let mut builder = HopperGeometryBuilder::new(config.clone());
            builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

            let stage_width = config.grid_width as f32 * CELL_SIZE;
            let offset = Vec3::new(x_offset, 0.0, 0.0);
            stages.push(create_rendered_stage(
                device,
                builder.vertices(),
                builder.indices(),
                offset,
            ));
            x_offset += stage_width + STAGE_SPACING;
        }

        // Stage 2: Grizzly (80x30x60 grid, bars parallel to X)
        {
            let config = GrateConfig {
                grid_width: 80,
                grid_height: 30,
                grid_depth: 60,
                cell_size: CELL_SIZE,
                bar_spacing: 4,
                bar_thickness: 2,
                orientation: 0, // bars parallel to X
                ..Default::default()
            };
            let mut builder = GrateGeometryBuilder::new(config.clone());
            builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

            let stage_width = config.grid_width as f32 * CELL_SIZE;
            let offset = Vec3::new(x_offset, 0.0, 0.0);
            stages.push(create_rendered_stage(
                device,
                builder.vertices(),
                builder.indices(),
                offset,
            ));
            x_offset += stage_width + STAGE_SPACING;
        }

        // Stage 3: Shaker (100x30x60 grid, finer bars)
        {
            let config = GrateConfig {
                grid_width: 100,
                grid_height: 30,
                grid_depth: 60,
                cell_size: CELL_SIZE,
                bar_spacing: 2,
                bar_thickness: 1,
                orientation: 1, // bars parallel to Z
                color_top: [0.6, 0.55, 0.5, 1.0],
                color_side: [0.5, 0.45, 0.4, 1.0],
                color_bottom: [0.4, 0.35, 0.3, 1.0],
            };
            let mut builder = GrateGeometryBuilder::new(config.clone());
            builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

            let stage_width = config.grid_width as f32 * CELL_SIZE;
            let offset = Vec3::new(x_offset, 0.0, 0.0);
            stages.push(create_rendered_stage(
                device,
                builder.vertices(),
                builder.indices(),
                offset,
            ));
            x_offset += stage_width + STAGE_SPACING;
        }

        // Stage 4: Sluice (150x40x50 grid)
        {
            let config = SluiceConfig {
                grid_width: 150,
                grid_height: 40,
                grid_depth: 50,
                cell_size: CELL_SIZE,
                floor_height_left: 25,
                floor_height_right: 8,
                riffle_spacing: 15,
                riffle_height: 3,
                riffle_thickness: 2,
                riffle_start_x: 15,
                riffle_end_pad: 10,
                wall_margin: 5,
                exit_width_fraction: 0.7,
                exit_height: 12,
                ..Default::default()
            };
            let mut builder = SluiceGeometryBuilder::new(config.clone());
            builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

            let offset = Vec3::new(x_offset, 0.0, 0.0);
            stages.push(create_rendered_stage(
                device,
                builder.vertices(),
                builder.indices(),
                offset,
            ));
        }

        stages
    }
}

/// Create rendered stage with vertices offset by world position
fn create_rendered_stage(
    device: &wgpu::Device,
    vertices: &[SluiceVertex],
    indices: &[u32],
    world_offset: Vec3,
) -> RenderedStage {
    // Offset all vertices by world position
    let offset_vertices: Vec<SluiceVertex> = vertices
        .iter()
        .map(|v| SluiceVertex {
            position: [
                v.position[0] + world_offset.x,
                v.position[1] + world_offset.y,
                v.position[2] + world_offset.z,
            ],
            color: v.color,
        })
        .collect();

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Stage Vertex Buffer"),
        contents: bytemuck::cast_slice(&offset_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Stage Index Buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    RenderedStage {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
    }
}

impl App for WashplantApp {
    fn init(ctx: &GpuContext) -> Self {
        let mut camera = FlyCamera::new();
        camera.position = Vec3::new(6.0, 2.0, 6.0);
        camera.yaw = -0.7;
        camera.pitch = -0.2;

        let pipeline = Self::create_pipeline(ctx);
        let stages = Self::build_stages(&ctx.device);

        println!("Washplant Basic Visual");
        println!("======================");
        println!("4 stages: Hopper → Grizzly → Shaker → Sluice");
        println!();
        println!("Controls:");
        println!("  WASD      - Move camera");
        println!("  Space     - Move up");
        println!("  Shift     - Move down");
        println!("  Mouse     - Look around");
        println!("  Scroll    - Zoom");

        Self {
            camera,
            pipeline,
            stages,
        }
    }

    fn update(&mut self, _ctx: &GpuContext, _dt: f32) {
        // No simulation, just static geometry
    }

    fn render(
        &mut self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Equipment Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.12,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &ctx.view_bind_group, &[]);

        for stage in &self.stages {
            render_pass.set_vertex_buffer(0, stage.vertex_buffer.slice(..));
            render_pass.set_index_buffer(stage.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..stage.index_count, 0, 0..1);
        }
    }

    fn camera(&self) -> &FlyCamera {
        &self.camera
    }

    fn camera_mut(&mut self) -> &mut FlyCamera {
        &mut self.camera
    }

    fn title() -> &'static str {
        "Washplant Basic"
    }
}

const SHADER: &str = r#"
struct ViewUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = view.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting based on Y position
    let ambient = 0.3;
    let height_factor = clamp(in.world_pos.y / 2.0, 0.0, 1.0);
    let brightness = ambient + (1.0 - ambient) * height_factor;

    return vec4<f32>(in.color.rgb * brightness, in.color.a);
}
"#;

fn main() {
    run::<WashplantApp>();
}
